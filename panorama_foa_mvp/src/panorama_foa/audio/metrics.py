from __future__ import annotations

import numpy as np


def audio_metrics(audio: np.ndarray, sample_rate: int) -> dict[str, float | int]:
    array = np.asarray(audio, dtype=np.float32)
    if array.ndim == 1:
        mono = array
        channels = 1
    else:
        channels = int(array.shape[1])
        mono = np.mean(array, axis=1)
    peak = float(np.max(np.abs(array))) if array.size else 0.0
    rms = float(np.sqrt(np.mean(np.asarray(array, dtype=np.float64) ** 2))) if array.size else 0.0
    frames = int(array.shape[0]) if array.ndim else 0
    edge_samples = min(round(0.1 * sample_rate), frames // 2)
    endpoint_jump = float(abs(mono[0] - mono[-1])) if frames else 0.0
    if edge_samples:
        edge_diff = mono[:edge_samples] - mono[-edge_samples:]
        edge_rms = float(np.sqrt(np.mean(np.asarray(edge_diff, dtype=np.float64) ** 2)))
    else:
        edge_rms = 0.0
    return {
        "sample_rate": int(sample_rate),
        "channels": channels,
        "frames": frames,
        "peak": peak,
        "peak_dbfs": dbfs(peak),
        "rms": rms,
        "rms_dbfs": dbfs(rms),
        "endpoint_jump": endpoint_jump,
        "edge_100ms_diff_dbfs": dbfs(edge_rms),
    }


def dbfs(value: float) -> float:
    return float(20.0 * np.log10(max(float(value), 1e-12)))

