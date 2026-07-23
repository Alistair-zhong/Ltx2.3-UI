from __future__ import annotations

import gc
import logging
import queue
import shlex
import threading
import traceback
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from .models import GenerationRequest, validate_request
from .profiling import InferenceProfiler

logger = logging.getLogger(__name__)
terminal_logger = logging.getLogger("uvicorn.error")

JobState = Literal["queued", "loading", "generating", "encoding", "completed", "failed", "cancelled"]


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_cli_command(request: GenerationRequest) -> str:
    """Render the programmatic request as an equivalent copyable CLI command."""
    model = request.model
    gen = request.generation
    guide = gen.guidance
    args = [
        "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True",
        "uv",
        "run",
        "python",
        "-m",
        "ltx_pipelines.a2vid_two_stage",
        "--checkpoint-path",
        model.checkpoint_path,
        "--gemma-root",
        model.gemma_root,
        "--distilled-lora",
        model.distilled_lora.path,
        str(model.distilled_lora.strength),
        "--spatial-upsampler-path",
        model.spatial_upsampler_path,
    ]
    for lora in model.loras:
        args.extend(("--lora", lora.path, str(lora.strength)))
    args.extend(
        (
            "--audio-path",
            gen.audio_path,
            "--audio-start-time",
            str(gen.audio_start_time),
            "--audio-max-duration",
            str(gen.audio_max_duration or gen.num_frames / gen.frame_rate),
        )
    )
    for image in gen.images:
        args.extend(
            (
                "--image",
                image.path,
                str(image.frame_idx),
                str(image.strength),
                str(image.crf),
            )
        )
    args.extend(
        (
            "--prompt",
            gen.prompt,
            "--negative-prompt",
            gen.negative_prompt,
            "--height",
            str(gen.height),
            "--width",
            str(gen.width),
            "--num-frames",
            str(gen.num_frames),
            "--frame-rate",
            str(gen.frame_rate),
            "--num-inference-steps",
            str(gen.num_inference_steps),
            "--seed",
            str(gen.seed),
            "--video-cfg-guidance-scale",
            str(guide.cfg_scale),
            "--video-stg-guidance-scale",
            str(guide.stg_scale),
            "--video-rescale-scale",
            str(guide.rescale_scale),
            "--a2v-guidance-scale",
            str(guide.a2v_scale),
            "--video-skip-step",
            str(guide.skip_step),
            "--video-stg-blocks",
        )
    )
    args.extend(str(block) for block in guide.stg_blocks)
    args.extend(
        (
            "--max-batch-size",
            str(model.max_batch_size),
            "--offload",
            model.offload,
        )
    )
    if model.compile_mode != "none":
        args.append("--compile")
        if model.compile_mode != "default":
            args.append(f"mode={model.compile_mode}")
    args.extend(("--output-path", gen.output_path))
    if model.quantization != "none":
        args.extend(("--quantization", model.quantization))
    if gen.enhance_prompt:
        args.append("--enhance-prompt")
    return shlex.join(args)


@dataclass
class Job:
    id: str
    request: GenerationRequest
    state: JobState = "queued"
    progress: int = 0
    message: str = "等待执行"
    created_at: str = field(default_factory=now_iso)
    started_at: str | None = None
    finished_at: str | None = None
    error: str | None = None
    profile: dict[str, Any] | None = None

    def public(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "state": self.state,
            "progress": self.progress,
            "message": self.message,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "error": self.error,
            "output_path": self.request.generation.output_path,
            "prompt": self.request.generation.prompt,
            "seed": self.request.generation.seed,
            "profile": self.profile,
        }


class PipelineRuntime:
    def __init__(self) -> None:
        self._pipeline: Any | None = None
        self._active_key: tuple | None = None
        self._jobs: dict[str, Job] = {}
        self._queue: queue.Queue[str | None] = queue.Queue()
        self._lock = threading.RLock()
        self._worker: threading.Thread | None = None
        self._stop = threading.Event()

    @property
    def active_key(self) -> tuple | None:
        return self._active_key

    @property
    def model_loaded(self) -> bool:
        return self._pipeline is not None

    @property
    def queue_size(self) -> int:
        return self._queue.qsize()

    def start(self) -> None:
        if self._worker and self._worker.is_alive():
            return
        self._stop.clear()
        self._worker = threading.Thread(target=self._work_loop, name="ltx-a2v-worker", daemon=True)
        self._worker.start()

    def stop(self) -> None:
        self._stop.set()
        self._queue.put(None)
        if self._worker:
            self._worker.join(timeout=5)

    def submit(self, request: GenerationRequest) -> Job:
        result = validate_request(request, self._active_key)
        if not result.valid:
            messages = "; ".join(item.message for item in result.issues if item.level == "error")
            raise ValueError(messages)
        Path(request.generation.output_path).expanduser().parent.mkdir(parents=True, exist_ok=True)
        job = Job(id=uuid.uuid4().hex[:12], request=request)
        with self._lock:
            self._jobs[job.id] = job
        self._queue.put(job.id)
        return job

    def get_job(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def list_jobs(self) -> list[dict[str, Any]]:
        with self._lock:
            jobs = sorted(self._jobs.values(), key=lambda item: item.created_at, reverse=True)
            return [job.public() for job in jobs[:50]]

    def cancel(self, job_id: str) -> bool:
        with self._lock:
            job = self._jobs.get(job_id)
            if not job or job.state != "queued":
                return False
            job.state = "cancelled"
            job.message = "已取消"
            job.finished_at = now_iso()
            return True

    def unload(self) -> None:
        with self._lock:
            if any(job.state in {"loading", "generating", "encoding"} for job in self._jobs.values()):
                raise RuntimeError("生成任务执行期间不能卸载模型")
            self._pipeline = None
            self._active_key = None
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    def _set_state(self, job: Job, state: JobState, progress: int, message: str) -> None:
        with self._lock:
            job.state = state
            job.progress = progress
            job.message = message

    def _work_loop(self) -> None:
        while not self._stop.is_set():
            job_id = self._queue.get()
            if job_id is None:
                return
            job = self.get_job(job_id)
            if not job or job.state == "cancelled":
                continue
            job.started_at = now_iso()
            try:
                self._execute(job)
            except Exception as exc:  # GPU/runtime failures must be attached to the job.
                logger.exception("Generation job %s failed", job.id)
                self._set_state(job, "failed", job.progress, "生成失败")
                job.error = f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=8)}"
            finally:
                job.finished_at = now_iso()

    def _execute(self, job: Job) -> None:
        # The official CLI decorates main() with @torch.inference_mode().  Keep the
        # context active through video iterator consumption in _encode as decoding
        # is lazy; wrapping only the pipeline call would exit too early.
        import torch

        with torch.inference_mode():
            terminal_logger.info(
                "PyTorch inference_mode=%s, grad_enabled=%s",
                torch.is_inference_mode_enabled(),
                torch.is_grad_enabled(),
            )
            self._execute_inference(job)

    def _execute_inference(self, job: Job) -> None:
        request = job.request
        reload_required = self._active_key != request.model.cache_key() or self._pipeline is None
        profiler = (
            InferenceProfiler(
                job_id=job.id,
                compile_mode=request.model.compile_mode,
                cold_start=reload_required,
            )
            if request.generation.profile
            else None
        )
        terminal_logger.info("=" * 88)
        terminal_logger.info(
            "LTX A2V job %s starting (model=%s)",
            job.id,
            "reload" if reload_required else "reuse",
        )
        terminal_logger.info(
            "Requested runtime: quantization=%s, offload=%s, compile=%s, "
            "max_batch_size=%d, profile=%s",
            request.model.quantization,
            request.model.offload,
            request.model.compile_mode,
            request.model.max_batch_size,
            request.generation.profile,
        )
        terminal_logger.info("Equivalent CLI command:\n%s", build_cli_command(request))
        terminal_logger.info("=" * 88)
        status = "failed"
        try:
            if reload_required:
                self._set_state(job, "loading", 8, "正在加载模型与 LoRA")
                if profiler is None:
                    self._load_pipeline(request)
                else:
                    with profiler.phase("model.load"):
                        self._load_pipeline(request)
            else:
                self._set_state(job, "generating", 15, "复用已加载模型，开始生成")

            self._set_state(job, "generating", 12, "正在编码 Prompt、音频和图片条件")
            if profiler is None:
                with self._sampling_progress(job):
                    video, audio = self._call_pipeline(request)
            else:
                with (
                    self._sampling_progress(job, profiler),
                    profiler.instrument_pipeline(self._pipeline),
                    profiler.phase("pipeline.total", summary=True),
                ):
                    video, audio = self._call_pipeline(request)
            self._set_state(job, "encoding", 92, "Stage 2 完成 · 准备 VAE 分块解码")
            if profiler is None:
                self._encode(job, request, video, audio)
            else:
                with profiler.phase("encode.total", summary=True):
                    self._encode(job, request, video, audio, profiler)
            status = "completed"
        finally:
            if profiler is not None:
                try:
                    job.profile = profiler.finish(status)
                    job.profile["recommendations"] = self._profile_recommendations(
                        request, job.profile
                    )
                    profiler.log_summary(terminal_logger, job.profile)
                except Exception as exc:
                    terminal_logger.exception("Failed to finalize profiling for job %s", job.id)
                    job.profile = {
                        "status": "profiling_failed",
                        "job_id": job.id,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
        self._set_state(job, "completed", 100, "生成完成")

    @staticmethod
    def _profile_recommendations(
        request: GenerationRequest, profile: dict[str, Any]
    ) -> list[str]:
        recommendations: list[str] = []
        model = request.model
        generation = request.generation
        bottleneck = profile.get("bottleneck") or {}
        bottleneck_name = bottleneck.get("name")
        step_profiles = profile.get("denoising_steps") or {}

        if profile.get("cold_start"):
            recommendations.append("保持模型与生成形状不变再运行一次，以排除加载和首次编译成本")
        if model.compile_mode == "none":
            recommendations.append("为扩散 Transformer 启用 reduce-overhead，并比较第二次热运行")
        if model.offload == "disk":
            recommendations.append("磁盘 offload 会成为 I/O 瓶颈；内存允许时改用 CPU 或关闭 offload")
        if (
            model.offload == "cpu"
            and model.max_batch_size < 4
            and bottleneck_name in {"denoise.stage_1", "denoise.stage_2"}
        ):
            recommendations.append(
                "显存允许时把最大批次逐步提高到 2/4，减少 CPU offload 的逐层 PCIe 搬运"
            )
        if bottleneck_name == "model.load":
            recommendations.append("复用常驻 Pipeline；只修改 prompt、seed 等生成参数不会重载模型")
        elif bottleneck_name in {"denoise.stage_1", "denoise.stage_2"}:
            recommendations.append(
                "扩散是主瓶颈；可测试更少采样步数或 video skip step，改动后需重新评估画质"
            )
        elif bottleneck_name == "decode.video_vae":
            recommendations.append("Video VAE 解码是主瓶颈；优先减少帧数或输出分辨率")
        elif bottleneck_name == "encode.container":
            recommendations.append("编码/封装是主瓶颈；将输出写到本地高速磁盘并检查 FFmpeg 编码速度")
        elif bottleneck_name == "conditioning.prompt" and generation.enhance_prompt:
            recommendations.append("Prompt 编码是主瓶颈；不需要改写提示词时关闭 Gemma 增强")

        for stage, stats in step_profiles.items():
            p50 = stats.get("p50_seconds", 0.0)
            first = stats.get("first_step_seconds", 0.0)
            if stats.get("steps", 0) >= 3 and first > max(1.0, p50 * 1.8):
                recommendations.append(
                    f"{stage} 首步明显慢于 P50，可能发生 torch.compile 编译或形状重编译"
                )
            stage_total = stats.get("stage_total_seconds", 0.0)
            outside_steps = stats.get("outside_steps_seconds", 0.0)
            if stage_total and outside_steps / stage_total > 0.2:
                recommendations.append(
                    f"{stage} 有较多时间花在采样循环外，重点检查模型构建、权重 offload 与清理"
                )
        return recommendations

    @contextmanager
    def _sampling_progress(
        self, job: Job, profiler: InferenceProfiler | None = None
    ):
        """Track real denoising steps without modifying the installed LTX package."""
        from ltx_pipelines.utils import samplers

        original_tqdm = samplers.tqdm
        loop_number = 0

        def tracked_tqdm(iterable, *args, **kwargs):
            nonlocal loop_number
            loop_number += 1
            stage = loop_number
            try:
                total = len(iterable)
            except TypeError:
                total = None

            def progress_iterator():
                if stage == 1:
                    self._set_state(job, "generating", 20, "Stage 1/2 低分辨率扩散 · 0 步")
                elif stage == 2:
                    self._set_state(job, "generating", 78, "Stage 2/2 高分辨率细化 · 0 步")
                completed = 0
                for item in original_tqdm(iterable, *args, **kwargs):
                    if profiler is None:
                        yield item
                    else:
                        with profiler.denoising_step(f"stage_{stage}"):
                            yield item
                    completed += 1
                    if stage == 1:
                        fraction = completed / total if total else 0
                        progress = 20 + round(50 * fraction)
                        message = f"Stage 1/2 低分辨率扩散 · {completed}/{total or '?'} 步"
                    elif stage == 2:
                        fraction = completed / total if total else 0
                        progress = 78 + round(12 * fraction)
                        message = f"Stage 2/2 高分辨率细化 · {completed}/{total or '?'} 步"
                    else:
                        continue
                    self._set_state(job, "generating", progress, message)
                if stage == 1:
                    self._set_state(job, "generating", 72, "Stage 1 完成 · 空间放大与高分辨率条件编码")
                elif stage == 2:
                    self._set_state(job, "generating", 91, "Stage 2 完成 · 创建 VAE 解码器")

            return progress_iterator()

        samplers.tqdm = tracked_tqdm
        try:
            yield
        finally:
            samplers.tqdm = original_tqdm

    def _load_pipeline(self, request: GenerationRequest) -> None:
        try:
            from ltx_core.loader import LTXV_LORA_COMFY_RENAMING_MAP, LoraPathStrengthAndSDOps
            from ltx_pipelines.a2vid_two_stage import A2VidPipelineTwoStage
            from ltx_pipelines.utils.quantization_factory import QuantizationKind
            from ltx_pipelines.utils.types import OffloadMode
        except ImportError as exc:
            raise RuntimeError(
                "未找到 ltx_pipelines。请从 LTX-2 仓库运行本项目，或先安装 packages/ltx-pipelines。"
            ) from exc

        model = request.model

        def lora(item: Any) -> Any:
            return LoraPathStrengthAndSDOps(
                str(Path(item.path).expanduser().resolve()),
                item.strength,
                LTXV_LORA_COMFY_RENAMING_MAP,
            )

        quantization = None
        if model.quantization != "none":
            quantization = QuantizationKind(model.quantization).to_policy(
                checkpoint_path=str(Path(model.checkpoint_path).expanduser().resolve())
            )

        compilation_config = None
        if model.compile_mode != "none":
            try:
                from ltx_core.model.transformer.compiling import CompilationConfig
            except ImportError as exc:
                raise RuntimeError(
                    "当前 ltx-core 不支持 torch.compile 配置；请更新 LTX-2.3，"
                    "或把编译模式设为“关闭”"
                ) from exc
            mode = None if model.compile_mode == "default" else model.compile_mode
            compilation_config = CompilationConfig(mode=mode)

        # Drop the prior instance before loading a config that cannot reuse it.
        self._pipeline = None
        self._active_key = None
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

        pipeline_kwargs = {
            "checkpoint_path": model.checkpoint_path,
            "distilled_lora": [lora(model.distilled_lora)],
            "spatial_upsampler_path": model.spatial_upsampler_path,
            "gemma_root": model.gemma_root,
            "loras": [lora(item) for item in model.loras],
            "quantization": quantization,
            "offload_mode": OffloadMode(model.offload),
        }
        if compilation_config is not None:
            pipeline_kwargs["compilation_config"] = compilation_config
        try:
            self._pipeline = A2VidPipelineTwoStage(**pipeline_kwargs)
        except TypeError as exc:
            if compilation_config is not None and "compilation_config" in str(exc):
                raise RuntimeError(
                    "当前 ltx-pipelines 版本不接受 compilation_config；"
                    "请更新 LTX-2.3，或把编译模式设为“关闭”"
                ) from exc
            raise
        self._active_key = model.cache_key()

    def _call_pipeline(self, request: GenerationRequest) -> tuple[Any, Any]:
        from ltx_core.components.guiders import MultiModalGuiderParams
        from ltx_core.model.video_vae import TilingConfig
        from ltx_pipelines.utils.args import ImageConditioningInput

        gen = request.generation
        guide = gen.guidance
        images = [ImageConditioningInput(x.path, x.frame_idx, x.strength, x.crf) for x in gen.images]
        return self._pipeline(
            prompt=gen.prompt,
            negative_prompt=gen.negative_prompt,
            seed=gen.seed,
            height=gen.height,
            width=gen.width,
            num_frames=gen.num_frames,
            frame_rate=gen.frame_rate,
            num_inference_steps=gen.num_inference_steps,
            video_guider_params=MultiModalGuiderParams(
                cfg_scale=guide.cfg_scale,
                stg_scale=guide.stg_scale,
                rescale_scale=guide.rescale_scale,
                modality_scale=guide.a2v_scale,
                skip_step=guide.skip_step,
                stg_blocks=guide.stg_blocks,
            ),
            images=images,
            audio_path=gen.audio_path,
            audio_start_time=gen.audio_start_time,
            audio_max_duration=gen.audio_max_duration or gen.num_frames / gen.frame_rate,
            tiling_config=TilingConfig.default(),
            enhance_prompt=gen.enhance_prompt,
            max_batch_size=request.model.max_batch_size,
        )

    def _encode(
        self,
        job: Job,
        request: GenerationRequest,
        video: Any,
        audio: Any,
        profiler: InferenceProfiler | None = None,
    ) -> None:
        from ltx_core.model.video_vae import TilingConfig, get_video_chunks_number
        from ltx_pipelines.utils.media_io import encode_video

        gen = request.generation
        video_chunks_number = get_video_chunks_number(gen.num_frames, TilingConfig.default())

        def video_with_progress():
            decoded = 0
            source = (
                profiler.timed_iterator(video, "decode.video_vae")
                if profiler is not None
                else video
            )
            for chunk in source:
                decoded += 1
                fraction = decoded / video_chunks_number if video_chunks_number else 0
                progress = 92 + round(6 * min(fraction, 1.0))
                self._set_state(
                    job,
                    "encoding",
                    progress,
                    f"VAE 分块解码与编码 · {decoded}/{video_chunks_number or '?'} 块",
                )
                yield chunk
            self._set_state(job, "encoding", 99, "视频帧完成 · 正在封装音频与 MP4")
        encode_video(
            video=video_with_progress(),
            fps=gen.frame_rate,
            audio=audio,
            output_path=gen.output_path,
            video_chunks_number=video_chunks_number,
        )
