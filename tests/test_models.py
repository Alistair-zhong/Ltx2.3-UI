import shlex
import sys
import types
from enum import Enum
from pathlib import Path

import pytest
from pydantic import ValidationError

from ltx23_ui.models import (
    GenerationConfig,
    GenerationRequest,
    LoraConfig,
    ModelConfig,
    frames_for_duration,
    validate_request,
)
from ltx23_ui.profiling import InferenceProfiler
from ltx23_ui.runtime import Job, PipelineRuntime, build_cli_command


def test_frames_snap_down_to_8k_plus_1() -> None:
    assert frames_for_duration(16, 25) == 393
    assert frames_for_duration(5, 24) == 113
    assert frames_for_duration(0.01, 25) == 1


def test_generation_constraints() -> None:
    with pytest.raises(ValidationError, match="路径不能为空"):
        GenerationConfig(
            prompt="test",
            audio_path="  ",
            output_path="out.mp4",
            width=768,
            height=1280,
            num_frames=121,
            frame_rate=25,
        )

    with pytest.raises(ValidationError, match="8k\\+1"):
        GenerationConfig(
            prompt="test",
            audio_path="audio.wav",
            output_path="out.mp4",
            width=768,
            height=1280,
            num_frames=376,
            frame_rate=25,
        )

    with pytest.raises(ValidationError, match="64"):
        GenerationConfig(
            prompt="test",
            audio_path="audio.wav",
            output_path="out.mp4",
            width=770,
            height=1280,
            num_frames=393,
            frame_rate=25,
        )


def test_validation_and_reload_key(tmp_path: Path) -> None:
    checkpoint = tmp_path / "model.safetensors"
    gemma = tmp_path / "gemma"
    distilled = tmp_path / "distilled.safetensors"
    upsampler = tmp_path / "up.safetensors"
    audio = tmp_path / "audio.wav"
    image = tmp_path / "image.jpg"
    output = tmp_path / "new" / "out.mp4"
    for path in (checkpoint, distilled, upsampler, audio, image):
        path.touch()
    gemma.mkdir()

    model = ModelConfig(
        checkpoint_path=str(checkpoint),
        gemma_root=str(gemma),
        distilled_lora=LoraConfig(path=str(distilled), strength=0.5),
        spatial_upsampler_path=str(upsampler),
    )
    request = GenerationRequest(
        model=model,
        generation=GenerationConfig(
            prompt="test",
            audio_path=str(audio),
            output_path=str(output),
            width=768,
            height=1280,
            num_frames=393,
            frame_rate=25,
            audio_max_duration=16,
        ),
    )
    first = validate_request(request, active_key=None)
    assert first.valid
    assert first.requires_reload
    assert any(issue.level == "info" for issue in first.issues)

    reused = validate_request(request, active_key=model.cache_key())
    assert reused.valid
    assert not reused.requires_reload


def test_cli_command_contains_runtime_and_quoted_values() -> None:
    request = GenerationRequest(
        model=ModelConfig(
            checkpoint_path="/models/main model.safetensors",
            gemma_root="/models/gemma",
            distilled_lora=LoraConfig(path="/models/distilled.safetensors", strength=0.5),
            spatial_upsampler_path="/models/up.safetensors",
            loras=[LoraConfig(path="/models/test lora.safetensors", strength=0.8)],
            quantization="fp8-cast",
            offload="cpu",
        ),
        generation=GenerationConfig(
            prompt="a singer's close-up",
            audio_path="/inputs/song.wav",
            output_path="/outputs/result.mp4",
            width=768,
            height=1280,
            num_frames=121,
            frame_rate=25,
            audio_max_duration=5,
        ),
    )
    command = build_cli_command(request)
    tokens = shlex.split(command)
    assert "--quantization fp8-cast" in command
    assert "--offload cpu" in command
    assert "--compile mode=reduce-overhead" in command
    assert "'/models/main model.safetensors'" in command
    assert "--lora '/models/test lora.safetensors' 0.8" in command
    assert tokens[tokens.index("--prompt") + 1] == "a singer's close-up"


def test_compile_mode_is_part_of_model_cache_key() -> None:
    model = ModelConfig(
        checkpoint_path="/models/main.safetensors",
        gemma_root="/models/gemma",
        distilled_lora=LoraConfig(path="/models/distilled.safetensors"),
        spatial_upsampler_path="/models/up.safetensors",
    )
    eager = model.model_copy(update={"compile_mode": "none"})
    assert model.cache_key() != eager.cache_key()


def test_pipeline_receives_reduce_overhead_compilation_config(monkeypatch) -> None:
    captured: dict = {}

    class FakePipeline:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    class FakeCompilationConfig:
        def __init__(self, *, mode):
            self.mode = mode

    class FakeOffloadMode(Enum):
        NONE = "none"
        CPU = "cpu"
        DISK = "disk"

    class FakeLora:
        def __init__(self, *args):
            self.args = args

    loader = types.ModuleType("ltx_core.loader")
    loader.LTXV_LORA_COMFY_RENAMING_MAP = {}
    loader.LoraPathStrengthAndSDOps = FakeLora
    compiling = types.ModuleType("ltx_core.model.transformer.compiling")
    compiling.CompilationConfig = FakeCompilationConfig
    pipeline_module = types.ModuleType("ltx_pipelines.a2vid_two_stage")
    pipeline_module.A2VidPipelineTwoStage = FakePipeline
    quantization_module = types.ModuleType("ltx_pipelines.utils.quantization_factory")
    quantization_module.QuantizationKind = object
    types_module = types.ModuleType("ltx_pipelines.utils.types")
    types_module.OffloadMode = FakeOffloadMode
    for name, module in {
        "ltx_core.loader": loader,
        "ltx_core.model.transformer.compiling": compiling,
        "ltx_pipelines.a2vid_two_stage": pipeline_module,
        "ltx_pipelines.utils.quantization_factory": quantization_module,
        "ltx_pipelines.utils.types": types_module,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)

    request = GenerationRequest(
        model=ModelConfig(
            checkpoint_path="/models/main.safetensors",
            gemma_root="/models/gemma",
            distilled_lora=LoraConfig(path="/models/distilled.safetensors"),
            spatial_upsampler_path="/models/up.safetensors",
            quantization="none",
            compile_mode="reduce-overhead",
        ),
        generation=GenerationConfig(
            prompt="test",
            audio_path="/inputs/audio.wav",
            output_path="/outputs/result.mp4",
            width=768,
            height=1280,
            num_frames=121,
            frame_rate=25,
        ),
    )
    runtime = PipelineRuntime()
    runtime._load_pipeline(request)
    assert captured["compilation_config"].mode == "reduce-overhead"
    assert runtime.active_key == request.model.cache_key()


def test_sampling_progress_tracks_both_real_loops(monkeypatch) -> None:
    samplers = types.ModuleType("ltx_pipelines.utils.samplers")

    def original_tqdm(iterable, *args, **kwargs):
        return iterable

    samplers.tqdm = original_tqdm
    utils = types.ModuleType("ltx_pipelines.utils")
    utils.samplers = samplers
    package = types.ModuleType("ltx_pipelines")
    package.utils = utils
    monkeypatch.setitem(sys.modules, "ltx_pipelines", package)
    monkeypatch.setitem(sys.modules, "ltx_pipelines.utils", utils)
    monkeypatch.setitem(sys.modules, "ltx_pipelines.utils.samplers", samplers)

    runtime = PipelineRuntime()
    job = Job(id="progress-test", request=None)  # type: ignore[arg-type]
    with runtime._sampling_progress(job):
        assert list(samplers.tqdm(range(4))) == [0, 1, 2, 3]
        assert job.progress == 72
        assert "空间放大" in job.message
        assert list(samplers.tqdm(range(3))) == [0, 1, 2]
        assert job.progress == 91
        assert "Stage 2" in job.message
    assert samplers.tqdm is original_tqdm


def test_profiler_breaks_down_pipeline_components() -> None:
    class Component:
        def __call__(self, *args, **kwargs):
            return args, kwargs

    class FakePipeline:
        prompt_encoder = Component()
        audio_conditioner = Component()
        image_conditioner = Component()
        stage_1 = Component()
        upsampler = Component()
        stage_2 = Component()
        video_decoder = Component()

    pipeline = FakePipeline()
    profiler = InferenceProfiler(
        job_id="profile-test",
        compile_mode="reduce-overhead",
        cold_start=True,
    )
    with profiler.phase("model.load"):
        pass
    with profiler.instrument_pipeline(pipeline), profiler.phase(
        "pipeline.total", summary=True
    ):
        pipeline.prompt_encoder()
        pipeline.audio_conditioner()
        pipeline.image_conditioner()
        pipeline.stage_1()
        pipeline.upsampler()
        pipeline.image_conditioner()
        pipeline.stage_2()
        pipeline.video_decoder()
        with profiler.denoising_step("stage_1"):
            pass
    with profiler.phase("encode.total", summary=True):
        assert list(profiler.timed_iterator([1, 2], "decode.video_vae")) == [1, 2]

    result = profiler.finish("completed")
    phase_names = {phase["name"] for phase in result["phases"]}
    assert {
        "model.load",
        "conditioning.prompt",
        "conditioning.audio_vae",
        "conditioning.image_stage_1",
        "denoise.stage_1",
        "upscale.video",
        "conditioning.image_stage_2",
        "denoise.stage_2",
        "decode.video_setup",
        "decode.video_vae",
    } <= phase_names
    assert result["cold_start"] is True
    assert result["compile_mode"] == "reduce-overhead"
    assert result["denoising_steps"]["stage_1"]["steps"] == 1


def test_profile_recommendations_follow_measured_bottleneck() -> None:
    request = GenerationRequest(
        model=ModelConfig(
            checkpoint_path="/models/main.safetensors",
            gemma_root="/models/gemma",
            distilled_lora=LoraConfig(path="/models/distilled.safetensors"),
            spatial_upsampler_path="/models/up.safetensors",
            offload="cpu",
            max_batch_size=1,
        ),
        generation=GenerationConfig(
            prompt="test",
            audio_path="/inputs/audio.wav",
            output_path="/outputs/result.mp4",
            width=768,
            height=1280,
            num_frames=121,
            frame_rate=25,
        ),
    )
    recommendations = PipelineRuntime._profile_recommendations(
        request,
        {
            "cold_start": False,
            "bottleneck": {"name": "denoise.stage_1"},
        },
    )
    assert any("PCIe" in item for item in recommendations)
    assert any("采样步数" in item for item in recommendations)
