from __future__ import annotations

import numpy as np
from scipy.signal import resample_poly


TARGET_SAMPLE_RATE = 48000
REFERENCE_PEAK = 10 ** (-6.0 / 20.0)
SILENCE_THRESHOLD = 1e-5


def process_mono_audio(
    audio: np.ndarray,
    *,
    sample_rate: int,
    duration_seconds: float,
    gain_db: float,
    source_id: str,
) -> np.ndarray:
    signal = np.asarray(audio, dtype=np.float32)
    if signal.ndim == 2:
        signal = np.mean(signal, axis=1)
    signal = signal.reshape(-1).astype(np.float32)

    if sample_rate != TARGET_SAMPLE_RATE:
        gcd = int(np.gcd(sample_rate, TARGET_SAMPLE_RATE))
        signal = resample_poly(signal, TARGET_SAMPLE_RATE // gcd, sample_rate // gcd).astype(np.float32)

    signal = signal - float(np.mean(signal)) if signal.size else signal
    target_samples = round(duration_seconds * TARGET_SAMPLE_RATE)
    signal = _fit_length(signal, target_samples)
    signal = signal - float(np.mean(signal)) if signal.size else signal

    peak = float(np.max(np.abs(signal))) if signal.size else 0.0
    if peak < SILENCE_THRESHOLD:
        raise ValueError(f"{source_id} is near silent")

    signal = signal * (REFERENCE_PEAK / peak)
    signal = signal * (10 ** (gain_db / 20.0))
    return signal.astype(np.float32)


def _fit_length(signal: np.ndarray, target_samples: int) -> np.ndarray:
    if signal.shape[0] == target_samples:
        return signal.astype(np.float32, copy=True)
    if signal.shape[0] > target_samples:
        return signal[:target_samples].astype(np.float32, copy=True)

    missing = target_samples - signal.shape[0]
    if missing <= round(0.25 * TARGET_SAMPLE_RATE):
        return np.pad(signal, (0, missing)).astype(np.float32)

    if signal.size == 0:
        return np.zeros(target_samples, dtype=np.float32)

    repeats = int(np.ceil(target_samples / signal.shape[0]))
    tiled = np.tile(signal, repeats)[:target_samples].astype(np.float32)
    fade = min(round(0.1 * TARGET_SAMPLE_RATE), signal.shape[0] // 2)
    if fade > 1:
        ramp = np.linspace(0.0, 1.0, fade, dtype=np.float32)
        for start in range(signal.shape[0], target_samples, signal.shape[0]):
            end = min(start + fade, target_samples)
            count = end - start
            if count > 0:
                tiled[start:end] *= ramp[:count]
    return tiled

