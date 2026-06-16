from __future__ import annotations

import numpy as np


def encode_regional_foa(
    mono: np.ndarray,
    *,
    azimuth_deg: float,
    elevation_deg: float,
    spread: float,
) -> np.ndarray:
    """Encode a mono regional source as AmbiX FOA in ACN/SN3D [W,Y,Z,X]."""
    signal = np.asarray(mono, dtype=np.float32).reshape(-1)
    azimuth = np.deg2rad(float(azimuth_deg))
    elevation = np.deg2rad(float(elevation_deg))
    directionality = 1.0 - float(np.clip(spread, 0.0, 1.0))

    w = signal
    y = signal * directionality * np.sin(azimuth) * np.cos(elevation)
    z = signal * directionality * np.sin(elevation)
    x = signal * directionality * np.cos(azimuth) * np.cos(elevation)
    return np.stack([w, y, z, x], axis=1).astype(np.float32, copy=False)


def encode_global_ambience(mono: np.ndarray) -> np.ndarray:
    """Encode global ambience as W-only AmbiX FOA."""
    signal = np.asarray(mono, dtype=np.float32).reshape(-1)
    output = np.zeros((signal.shape[0], 4), dtype=np.float32)
    output[:, 0] = signal
    return output
