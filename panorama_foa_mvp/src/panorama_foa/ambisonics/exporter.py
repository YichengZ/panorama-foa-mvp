from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf


DEFAULT_SAMPLE_RATE = 48_000
CHANNEL_LABELS = ["W", "Y", "Z", "X"]


def export_foa_wav(
    output_path: str | Path,
    foa_audio: np.ndarray,
    *,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
) -> Path:
    """Write a four-channel FOA WAV as 24-bit PCM."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    audio = np.asarray(foa_audio, dtype=np.float32)
    if audio.ndim != 2 or audio.shape[1] != 4:
        raise ValueError("FOA audio must have shape (samples, 4)")

    sf.write(path, audio, samplerate=int(sample_rate), subtype="PCM_24", format="WAV")
    return path


def build_metadata(
    *,
    duration_seconds: float,
    yaw_offset_deg: float = 0.0,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    file: str = "scene_foa.wav",
) -> dict[str, Any]:
    """Build the sidecar metadata that declares the FOA convention."""
    return {
        "schema_version": "1.0",
        "file": file,
        "sample_rate": int(sample_rate),
        "channels": 4,
        "duration_seconds": float(duration_seconds),
        "ambisonics": {
            "order": 1,
            "convention": "AmbiX",
            "channel_ordering": "ACN",
            "normalization": "SN3D",
            "channel_labels": CHANNEL_LABELS,
        },
        "coordinates": {
            "azimuth_zero": "front",
            "positive_azimuth": "left",
            "positive_elevation": "up",
            "yaw_offset_deg": float(yaw_offset_deg),
        },
        "listener_model": {
            "translation_changes_audio": False,
            "rotation_can_change_decode": True,
        },
    }


def write_metadata(
    output_path: str | Path,
    *,
    duration_seconds: float,
    yaw_offset_deg: float = 0.0,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    file: str = "scene_foa.wav",
) -> Path:
    """Write the JSON sidecar metadata for an exported FOA WAV."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = build_metadata(
        duration_seconds=duration_seconds,
        yaw_offset_deg=yaw_offset_deg,
        sample_rate=sample_rate,
        file=file,
    )
    path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return path
