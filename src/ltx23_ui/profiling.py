from __future__ import annotations

import importlib
import logging
from collections import defaultdict
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from time import perf_counter
from typing import Any


PHASE_LABELS = {
    "model.load": "模型与 LoRA 加载",
    "conditioning.prompt": "Prompt 编码",
    "conditioning.audio_decode": "音频文件解码",
    "conditioning.audio_vae": "Audio VAE 编码",
    "conditioning.image_stage_1": "Stage 1 图片条件编码",
    "denoise.stage_1": "Stage 1 低分辨率扩散",
    "upscale.video": "空间上采样",
    "conditioning.image_stage_2": "Stage 2 图片条件编码",
    "denoise.stage_2": "Stage 2 高分辨率细化",
    "decode.video_setup": "Video VAE 解码器创建",
    "decode.video_vae": "Video VAE 分块解码",
    "encode.container": "视频编码与音频封装",
    "pipeline.other": "Pipeline 其他准备",
    "runtime.other": "运行时其他开销",
}

PIPELINE_COMPONENTS = {
    "conditioning.prompt",
    "conditioning.audio_decode",
    "conditioning.audio_vae",
    "conditioning.image_stage_1",
    "denoise.stage_1",
    "upscale.video",
    "conditioning.image_stage_2",
    "denoise.stage_2",
    "decode.video_setup",
}


class _TimedCallable:
    def __init__(
        self,
        target: Callable[..., Any],
        profiler: InferenceProfiler,
        phase_names: tuple[str, ...],
    ) -> None:
        self._target = target
        self._profiler = profiler
        self._phase_names = phase_names
        self._calls = 0

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        phase_name = self._phase_names[min(self._calls, len(self._phase_names) - 1)]
        self._calls += 1
        with self._profiler.phase(phase_name):
            return self._target(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._target, name)


class InferenceProfiler:
    """Phase profiler with CUDA-correct wall-clock timings."""

    def __init__(self, *, job_id: str, compile_mode: str, cold_start: bool) -> None:
        self.job_id = job_id
        self.compile_mode = compile_mode
        self.cold_start = cold_start
        self._phase_seconds: defaultdict[str, float] = defaultdict(float)
        self._phase_calls: defaultdict[str, int] = defaultdict(int)
        self._summary_seconds: defaultdict[str, float] = defaultdict(float)
        self._step_seconds: defaultdict[str, list[float]] = defaultdict(list)
        self._pending_cuda_steps: list[tuple[str, Any, Any, float]] = []
        self._result: dict[str, Any] | None = None
        self._torch = self._load_cuda()
        if self._torch is not None:
            try:
                self._torch.cuda.reset_peak_memory_stats()
            except (RuntimeError, AttributeError):
                self._torch = None
        self._started = perf_counter()

    @staticmethod
    def _load_cuda() -> Any | None:
        try:
            import torch

            if torch.cuda.is_available():
                return torch
        except (ImportError, RuntimeError):
            pass
        return None

    def _synchronize(self) -> None:
        if self._torch is None:
            return
        self._torch.cuda.synchronize()

    def _record_phase(self, name: str, seconds: float, *, summary: bool = False) -> None:
        target = self._summary_seconds if summary else self._phase_seconds
        target[name] += seconds
        if not summary:
            self._phase_calls[name] += 1

    @contextmanager
    def phase(self, name: str, *, summary: bool = False) -> Iterator[None]:
        self._synchronize()
        started = perf_counter()
        try:
            yield
        finally:
            self._synchronize()
            self._record_phase(name, perf_counter() - started, summary=summary)

    @contextmanager
    def denoising_step(self, stage: str) -> Iterator[None]:
        """Measure one sampler body without synchronizing after every CUDA step."""
        host_started = perf_counter()
        start_event = end_event = None
        if self._torch is not None:
            start_event = self._torch.cuda.Event(enable_timing=True)
            end_event = self._torch.cuda.Event(enable_timing=True)
            start_event.record()
        try:
            yield
        finally:
            host_seconds = perf_counter() - host_started
            if start_event is not None and end_event is not None:
                end_event.record()
                self._pending_cuda_steps.append(
                    (stage, start_event, end_event, host_seconds)
                )
            else:
                self._step_seconds[stage].append(host_seconds)

    @contextmanager
    def instrument_pipeline(self, pipeline: Any) -> Iterator[None]:
        """Temporarily wrap the public A2Vid pipeline components."""
        replacements: list[tuple[Any, str, Any]] = []
        component_specs = (
            ("prompt_encoder", ("conditioning.prompt",)),
            ("audio_conditioner", ("conditioning.audio_vae",)),
            (
                "image_conditioner",
                ("conditioning.image_stage_1", "conditioning.image_stage_2"),
            ),
            ("stage_1", ("denoise.stage_1",)),
            ("upsampler", ("upscale.video",)),
            ("stage_2", ("denoise.stage_2",)),
            ("video_decoder", ("decode.video_setup",)),
        )
        for attribute, phase_names in component_specs:
            target = getattr(pipeline, attribute, None)
            if target is None or not callable(target):
                continue
            replacements.append((pipeline, attribute, target))
            setattr(pipeline, attribute, _TimedCallable(target, self, phase_names))

        try:
            module = importlib.import_module(type(pipeline).__module__)
        except (ImportError, AttributeError):
            module = None
        if module is not None:
            target = getattr(module, "decode_audio_from_file", None)
            if callable(target):

                def timed_audio_decode(*args: Any, **kwargs: Any) -> Any:
                    with self.phase("conditioning.audio_decode"):
                        return target(*args, **kwargs)

                replacements.append((module, "decode_audio_from_file", target))
                setattr(module, "decode_audio_from_file", timed_audio_decode)

        try:
            yield
        finally:
            for owner, attribute, original in reversed(replacements):
                setattr(owner, attribute, original)

    def timed_iterator(self, iterable: Any, phase_name: str) -> Iterator[Any]:
        iterator = iter(iterable)
        while True:
            self._synchronize()
            started = perf_counter()
            try:
                item = next(iterator)
            except StopIteration:
                return
            self._synchronize()
            self._record_phase(phase_name, perf_counter() - started)
            yield item

    def _resolve_cuda_steps(self) -> None:
        if not self._pending_cuda_steps:
            return
        self._synchronize()
        for stage, start_event, end_event, host_seconds in self._pending_cuda_steps:
            cuda_seconds = start_event.elapsed_time(end_event) / 1000
            self._step_seconds[stage].append(max(host_seconds, cuda_seconds))
        self._pending_cuda_steps.clear()

    @staticmethod
    def _percentile(values: list[float], percentile: float) -> float:
        if not values:
            return 0.0
        ordered = sorted(values)
        index = round((len(ordered) - 1) * percentile)
        return ordered[index]

    def _step_summary(self) -> dict[str, dict[str, float | int]]:
        output: dict[str, dict[str, float | int]] = {}
        for stage, values in self._step_seconds.items():
            if not values:
                continue
            step_total = sum(values)
            stage_total = self._phase_seconds.get(f"denoise.{stage}", step_total)
            output[stage] = {
                "steps": len(values),
                "total_seconds": round(step_total, 3),
                "stage_total_seconds": round(stage_total, 3),
                "outside_steps_seconds": round(max(0.0, stage_total - step_total), 3),
                "average_seconds": round(step_total / len(values), 3),
                "p50_seconds": round(self._percentile(values, 0.50), 3),
                "p95_seconds": round(self._percentile(values, 0.95), 3),
                "max_seconds": round(max(values), 3),
                "first_step_seconds": round(values[0], 3),
            }
        return output

    def _cuda_summary(self) -> dict[str, Any] | None:
        if self._torch is None:
            return None
        try:
            gib = 1024**3
            return {
                "device": self._torch.cuda.get_device_name(),
                "peak_allocated_gib": round(
                    self._torch.cuda.max_memory_allocated() / gib, 3
                ),
                "peak_reserved_gib": round(
                    self._torch.cuda.max_memory_reserved() / gib, 3
                ),
            }
        except (RuntimeError, AttributeError):
            return None

    def finish(self, status: str) -> dict[str, Any]:
        if self._result is not None:
            return self._result
        self._resolve_cuda_steps()
        total_seconds = perf_counter() - self._started

        pipeline_total = self._summary_seconds.get("pipeline.total", 0.0)
        measured_pipeline = sum(
            self._phase_seconds.get(name, 0.0) for name in PIPELINE_COMPONENTS
        )
        pipeline_other = max(0.0, pipeline_total - measured_pipeline)
        if pipeline_other >= 0.001:
            self._record_phase("pipeline.other", pipeline_other)

        encode_total = self._summary_seconds.get("encode.total", 0.0)
        video_decode = self._phase_seconds.get("decode.video_vae", 0.0)
        container_encode = max(0.0, encode_total - video_decode)
        if container_encode >= 0.001:
            self._record_phase("encode.container", container_encode)

        measured_total = sum(self._phase_seconds.values())
        runtime_other = max(0.0, total_seconds - measured_total)
        if runtime_other >= 0.001:
            self._record_phase("runtime.other", runtime_other)

        phases = [
            {
                "name": name,
                "label": PHASE_LABELS.get(name, name),
                "seconds": round(seconds, 3),
                "percent": round(100 * seconds / total_seconds, 1)
                if total_seconds
                else 0.0,
                "calls": self._phase_calls[name],
            }
            for name, seconds in sorted(
                self._phase_seconds.items(), key=lambda item: item[1], reverse=True
            )
        ]
        summaries = {
            name: round(seconds, 3)
            for name, seconds in self._summary_seconds.items()
        }
        self._result = {
            "status": status,
            "job_id": self.job_id,
            "total_seconds": round(total_seconds, 3),
            "cold_start": self.cold_start,
            "compile_mode": self.compile_mode,
            "bottleneck": phases[0] if phases else None,
            "phases": phases,
            "summaries": summaries,
            "denoising_steps": self._step_summary(),
            "cuda": self._cuda_summary(),
        }
        return self._result

    def log_summary(self, output_logger: logging.Logger, result: dict[str, Any]) -> None:
        output_logger.info("-" * 88)
        output_logger.info(
            "PROFILE job=%s status=%s total=%.3fs run=%s compile=%s",
            self.job_id,
            result["status"],
            result["total_seconds"],
            "cold" if self.cold_start else "warm",
            self.compile_mode,
        )
        for phase in result["phases"]:
            output_logger.info(
                "PROFILE %6.1f%% %9.3fs  %s",
                phase["percent"],
                phase["seconds"],
                phase["label"],
            )
        for stage, stats in result["denoising_steps"].items():
            output_logger.info(
                "PROFILE %s steps=%d avg=%.3fs p95=%.3fs max=%.3fs "
                "first=%.3fs outside_steps=%.3fs",
                stage,
                stats["steps"],
                stats["average_seconds"],
                stats["p95_seconds"],
                stats["max_seconds"],
                stats["first_step_seconds"],
                stats["outside_steps_seconds"],
            )
        cuda = result["cuda"]
        if cuda:
            output_logger.info(
                "PROFILE CUDA device=%s peak_allocated=%.3fGiB peak_reserved=%.3fGiB",
                cuda["device"],
                cuda["peak_allocated_gib"],
                cuda["peak_reserved_gib"],
            )
        for recommendation in result.get("recommendations", []):
            output_logger.info("PROFILE 建议: %s", recommendation)
        output_logger.info("-" * 88)
