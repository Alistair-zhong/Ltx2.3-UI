from __future__ import annotations

import json
import subprocess
import wave
from pathlib import Path


def probe_duration(path: str) -> float:
    source = Path(path).expanduser()
    if not source.exists():
        raise FileNotFoundError(path)
    try:
        completed = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "json",
                str(source),
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=20,
        )
        return float(json.loads(completed.stdout)["format"]["duration"])
    except (FileNotFoundError, subprocess.SubprocessError, KeyError, ValueError, json.JSONDecodeError):
        if source.suffix.lower() != ".wav":
            raise RuntimeError("无法读取媒体时长；请安装 ffmpeg/ffprobe") from None
        with wave.open(str(source), "rb") as audio:
            return audio.getnframes() / audio.getframerate()

