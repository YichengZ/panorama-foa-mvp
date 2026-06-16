from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


def require_ffmpeg(ffmpeg_path: str = "ffmpeg") -> str:
    resolved = shutil.which(ffmpeg_path)
    if resolved is None:
        raise RuntimeError("ffmpeg is required to decode audio. Install ffmpeg and ensure it is on PATH.")
    return resolved


def decode_to_mono_wav(input_path: Path, output_path: Path, *, ffmpeg_path: str = "ffmpeg") -> Path:
    ffmpeg = require_ffmpeg(ffmpeg_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        ffmpeg,
        "-y",
        "-i",
        str(input_path),
        "-ac",
        "1",
        "-ar",
        "48000",
        "-c:a",
        "pcm_f32le",
        str(output_path),
    ]
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        message = exc.stderr.strip() or exc.stdout.strip() or str(exc)
        raise RuntimeError(f"ffmpeg failed to decode {input_path}: {message}") from exc
    return output_path

