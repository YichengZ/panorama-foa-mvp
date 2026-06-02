from __future__ import annotations

from typing import Optional

import cv2
import numpy as np


class LaplacianPyramid:
    """Gaussian-pyramid mip sampler for perspective-to-panorama projection."""

    def __init__(self, image: np.ndarray, num_levels: Optional[int] = None) -> None:
        self.image = image.astype(np.float32)
        shortest_side = min(image.shape[0], image.shape[1])
        default_levels = int(np.log2(shortest_side)) if shortest_side > 1 else 0
        self.num_levels = default_levels if num_levels is None else int(num_levels)

        self.gauss = [self.image]
        for _ in range(self.num_levels):
            self.gauss.append(cv2.pyrDown(self.gauss[-1]))

    def sample_lapmip(
        self,
        x: np.ndarray,
        y: np.ndarray,
        dx: np.ndarray,
        dy: np.ndarray,
        scale: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Sample image values from an approximate mip level selected per point."""
        scale = np.maximum(np.maximum(np.abs(dx), np.abs(dy)) if scale is None else scale, 1e-8)
        level = np.clip(np.log2(scale) - .5, 0, self.num_levels)
        lo = np.floor(level).astype(int)
        hi = np.clip(lo + 1, 0, self.num_levels)
        t = level - lo

        if self.image.ndim == 3:
            c0 = np.zeros((*x.shape, self.image.shape[2]), dtype=np.float32)
        else:
            c0 = np.zeros(x.shape, dtype=np.float32)
        c1 = np.zeros_like(c0)

        for level_idx in range(self.num_levels + 1):
            mask = lo == level_idx
            if not mask.any():
                continue

            hi_idx = level_idx + 1 if level_idx < self.num_levels else level_idx
            c0[mask] = self._bilerp(
                self.gauss[level_idx],
                x[mask] / (2 ** level_idx),
                y[mask] / (2 ** level_idx),
            )
            c1[mask] = self._bilerp(
                self.gauss[hi_idx],
                x[mask] / (2 ** hi_idx),
                y[mask] / (2 ** hi_idx),
            )

        if self.image.ndim == 3:
            t = t[..., None]
        return c0 * (1 - t) + c1 * t

    def _bilerp(self, image: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        x0 = np.floor(x).astype(int)
        y0 = np.floor(y).astype(int)
        x1 = x0 + 1
        y1 = y0 + 1

        x0 = np.clip(x0, 0, image.shape[1] - 1)
        x1 = np.clip(x1, 0, image.shape[1] - 1)
        y0 = np.clip(y0, 0, image.shape[0] - 1)
        y1 = np.clip(y1, 0, image.shape[0] - 1)

        wx = x - x0
        wy = y - y0
        if image.ndim == 3:
            wx = wx[..., None]
            wy = wy[..., None]

        top = image[y0, x0] * (1 - wx) + image[y0, x1] * wx
        bottom = image[y1, x0] * (1 - wx) + image[y1, x1] * wx
        return top * (1 - wy) + bottom * wy
