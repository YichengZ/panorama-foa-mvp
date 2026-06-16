from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def mix_foa_layers(
    layers: Sequence[np.ndarray],
    *,
    target_dbfs: float = -1.0,
    return_scale: bool = False,
) -> np.ndarray | tuple[np.ndarray, float]:
    """Sum FOA layers and apply one final scalar limiter when needed."""
    arrays = [_as_foa_array(layer) for layer in layers]
    if not arrays:
        raise ValueError("at least one FOA layer is required")

    first_shape = arrays[0].shape
    if any(array.shape != first_shape for array in arrays):
        raise ValueError("all FOA layers must have the same shape")

    mix = np.zeros(first_shape, dtype=np.float32)
    for array in arrays:
        mix += array

    target_peak = float(10 ** (target_dbfs / 20.0))
    peak = float(np.max(np.abs(mix))) if mix.size else 0.0
    scale = target_peak / peak if peak > target_peak and peak > 0.0 else 1.0
    if scale != 1.0:
        mix *= scale

    if return_scale:
        return mix.astype(np.float32, copy=False), scale
    return mix.astype(np.float32, copy=False)


def _as_foa_array(layer: np.ndarray) -> np.ndarray:
    array = np.asarray(layer, dtype=np.float32)
    if array.ndim != 2 or array.shape[1] != 4:
        raise ValueError("FOA layers must have shape (samples, 4)")
    return array
