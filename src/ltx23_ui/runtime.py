from __future__ import annotations

import gc
import logging
import queue
import threading
import traceback
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from .models import GenerationRequest, validate_request

logger = logging.getLogger(__name__)

JobState = Literal["queued", "loading", "generating", "encoding", "completed", "failed", "cancelled"]


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


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
        request = job.request
        if self._active_key != request.model.cache_key() or self._pipeline is None:
            self._set_state(job, "loading", 8, "正在加载模型与 LoRA")
            self._load_pipeline(request)
        else:
            self._set_state(job, "generating", 15, "复用已加载模型，开始生成")

        self._set_state(job, "generating", 20, "Stage 1/2 扩散与上采样中")
        video, audio = self._call_pipeline(request)
        self._set_state(job, "encoding", 90, "正在编码 MP4")
        self._encode(request, video, audio)
        self._set_state(job, "completed", 100, "生成完成")

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

        self._pipeline = A2VidPipelineTwoStage(
            checkpoint_path=model.checkpoint_path,
            distilled_lora=[lora(model.distilled_lora)],
            spatial_upsampler_path=model.spatial_upsampler_path,
            gemma_root=model.gemma_root,
            loras=[lora(item) for item in model.loras],
            quantization=quantization,
            offload_mode=OffloadMode(model.offload),
        )
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

    def _encode(self, request: GenerationRequest, video: Any, audio: Any) -> None:
        from ltx_core.model.video_vae import TilingConfig, get_video_chunks_number
        from ltx_pipelines.utils.media_io import encode_video

        gen = request.generation
        encode_video(
            video=video,
            fps=gen.frame_rate,
            audio=audio,
            output_path=gen.output_path,
            video_chunks_number=get_video_chunks_number(gen.num_frames, TilingConfig.default()),
        )
