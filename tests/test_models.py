import shlex
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
from ltx23_ui.runtime import build_cli_command


def test_frames_snap_down_to_8k_plus_1() -> None:
    assert frames_for_duration(16, 25) == 393
    assert frames_for_duration(5, 24) == 113
    assert frames_for_duration(0.01, 25) == 1


def test_generation_constraints() -> None:
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
    assert "'/models/main model.safetensors'" in command
    assert "--lora '/models/test lora.safetensors' 0.8" in command
    assert tokens[tokens.index("--prompt") + 1] == "a singer's close-up"
