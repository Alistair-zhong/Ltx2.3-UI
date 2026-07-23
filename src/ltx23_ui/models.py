from __future__ import annotations

import math
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator


DEFAULT_NEGATIVE_PROMPT = (
    "blurry, out of focus, overexposed, underexposed, low contrast, washed out colors, "
    "excessive noise, flickering, motion blur, distorted proportions, deformed facial "
    "features, extra limbs, artifacts, camera shake, inconsistent lighting, wrong gaze "
    "direction, mismatched lip sync, off-sync audio, jittery movement, AI artifacts"
)


class LoraConfig(BaseModel):
    path: str
    strength: float = Field(default=1.0, ge=-4.0, le=4.0)

    @field_validator("path")
    @classmethod
    def path_must_not_be_empty(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("LoRA 路径不能为空")
        return value


class ModelConfig(BaseModel):
    checkpoint_path: str
    gemma_root: str
    distilled_lora: LoraConfig
    spatial_upsampler_path: str
    loras: list[LoraConfig] = Field(default_factory=list)
    quantization: Literal["none", "fp8-cast", "fp8-scaled-mm"] = "fp8-cast"
    offload: Literal["none", "cpu", "disk"] = "cpu"
    compile_mode: Literal["none", "default", "reduce-overhead", "max-autotune"] = (
        "reduce-overhead"
    )
    max_batch_size: int = Field(default=1, ge=1, le=4)

    @field_validator("checkpoint_path", "gemma_root", "spatial_upsampler_path")
    @classmethod
    def model_path_must_not_be_empty(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("模型路径不能为空")
        return value

    def cache_key(self) -> tuple:
        return (
            self.checkpoint_path,
            self.gemma_root,
            self.distilled_lora.path,
            self.distilled_lora.strength,
            self.spatial_upsampler_path,
            tuple((item.path, item.strength) for item in self.loras),
            self.quantization,
            self.offload,
            self.compile_mode,
        )


class ImageCondition(BaseModel):
    path: str
    frame_idx: int = Field(default=0, ge=0)
    strength: float = Field(default=1.0, ge=0.0, le=1.0)
    crf: int = Field(default=33, ge=0, le=51)

    @field_validator("path")
    @classmethod
    def path_must_not_be_empty(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("图片路径不能为空")
        return value


class GuidanceConfig(BaseModel):
    cfg_scale: float = Field(default=3.0, ge=0.0, le=20.0)
    stg_scale: float = Field(default=1.0, ge=0.0, le=10.0)
    rescale_scale: float = Field(default=0.7, ge=0.0, le=1.0)
    a2v_scale: float = Field(default=3.0, ge=0.0, le=20.0)
    skip_step: int = Field(default=0, ge=0, le=10)
    stg_blocks: list[int] = Field(default_factory=lambda: [28])


class GenerationConfig(BaseModel):
    prompt: str = Field(min_length=1)
    negative_prompt: str = DEFAULT_NEGATIVE_PROMPT
    audio_path: str
    audio_start_time: float = Field(default=0.0, ge=0.0)
    audio_max_duration: float | None = Field(default=None, gt=0.0)
    images: list[ImageCondition] = Field(default_factory=list)
    height: int = Field(default=1280, ge=64, le=4096)
    width: int = Field(default=768, ge=64, le=4096)
    num_frames: int = Field(default=121, ge=1)
    frame_rate: float = Field(default=25.0, gt=0.0, le=120.0)
    num_inference_steps: int = Field(default=30, ge=1, le=200)
    seed: int = Field(default=10, ge=0, le=2**63 - 1)
    output_path: str
    enhance_prompt: bool = False
    profile: bool = True
    guidance: GuidanceConfig = Field(default_factory=GuidanceConfig)

    @field_validator("audio_path", "output_path")
    @classmethod
    def generation_path_must_not_be_empty(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("路径不能为空")
        return value

    @model_validator(mode="after")
    def validate_ltx_constraints(self) -> GenerationConfig:
        if self.height % 64 or self.width % 64:
            raise ValueError("两阶段 A2V 的高度和宽度必须是 64 的倍数")
        if (self.num_frames - 1) % 8:
            raise ValueError("帧数必须满足 8k+1（例如 121、369、393）")
        duration = self.num_frames / self.frame_rate
        if self.audio_max_duration is not None and duration > self.audio_max_duration + 1 / self.frame_rate:
            raise ValueError("视频时长不能明显超过音频截取时长")
        for image in self.images:
            if image.frame_idx >= self.num_frames:
                raise ValueError(f"图片条件帧 {image.frame_idx} 超出了视频帧数")
        return self


class GenerationRequest(BaseModel):
    model: ModelConfig
    generation: GenerationConfig


class ValidationIssue(BaseModel):
    level: Literal["error", "warning", "info"]
    field: str
    message: str


class ValidationResult(BaseModel):
    valid: bool
    requires_reload: bool
    video_duration: float
    issues: list[ValidationIssue]


def frames_for_duration(duration: float, fps: float) -> int:
    """Largest valid 8k+1 frame count that does not exceed a duration."""
    if duration <= 0 or fps <= 0:
        raise ValueError("duration and fps must be positive")
    available = max(1, math.floor(duration * fps + 1e-6))
    return max(1, ((available - 1) // 8) * 8 + 1)


def validate_request(
    request: GenerationRequest, active_key: tuple | None, check_paths: bool = True
) -> ValidationResult:
    issues: list[ValidationIssue] = []
    model = request.model
    gen = request.generation

    if check_paths:
        required_files = {
            "model.checkpoint_path": model.checkpoint_path,
            "model.distilled_lora.path": model.distilled_lora.path,
            "model.spatial_upsampler_path": model.spatial_upsampler_path,
            "generation.audio_path": gen.audio_path,
        }
        required_files.update({f"model.loras.{i}.path": x.path for i, x in enumerate(model.loras)})
        required_files.update({f"generation.images.{i}.path": x.path for i, x in enumerate(gen.images)})
        for field, value in required_files.items():
            if not Path(value).expanduser().is_file():
                issues.append(ValidationIssue(level="error", field=field, message=f"文件不存在：{value}"))
        if not Path(model.gemma_root).expanduser().is_dir():
            issues.append(
                ValidationIssue(
                    level="error",
                    field="model.gemma_root",
                    message=f"Gemma 目录不存在：{model.gemma_root}",
                )
            )

    output = Path(gen.output_path).expanduser()
    if output.suffix.lower() != ".mp4":
        issues.append(ValidationIssue(level="error", field="generation.output_path", message="输出文件必须是 .mp4"))
    if not output.parent.exists():
        issues.append(ValidationIssue(level="info", field="generation.output_path", message="输出目录会自动创建"))

    megapixels = gen.height * gen.width / 1_000_000
    if megapixels > 1.5:
        issues.append(ValidationIssue(level="warning", field="generation.height", message="高分辨率会显著增加显存占用和生成时间"))
    if gen.num_frames > 401:
        issues.append(ValidationIssue(level="warning", field="generation.num_frames", message="长视频会显著增加生成时间和显存压力"))
    if model.quantization == "fp8-scaled-mm":
        issues.append(ValidationIssue(level="warning", field="model.quantization", message="fp8-scaled-mm 需要 TensorRT-LLM 与 Hopper GPU"))
    if model.offload == "disk":
        issues.append(ValidationIssue(level="warning", field="model.offload", message="磁盘 offload 最省显存，但生成会明显变慢"))
    if model.compile_mode != "none":
        issues.append(
            ValidationIssue(
                level="info",
                field="model.compile_mode",
                message=(
                    "torch.compile 首次生成含编译开销；"
                    "请用相同模型配置的第二次任务评估热运行速度"
                ),
            )
        )
    if model.offload == "cpu" and model.max_batch_size == 1:
        issues.append(
            ValidationIssue(
                level="info",
                field="model.max_batch_size",
                message=(
                    "CPU offload 下可在显存允许时测试最大批次 4，"
                    "以减少逐层 PCIe 搬运"
                ),
            )
        )

    return ValidationResult(
        valid=not any(issue.level == "error" for issue in issues),
        requires_reload=active_key != model.cache_key(),
        video_duration=round(gen.num_frames / gen.frame_rate, 3),
        issues=issues,
    )
