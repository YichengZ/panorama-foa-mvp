from __future__ import annotations

import math


def normalized_panorama_to_angles(
    center_u: float,
    center_v: float,
    yaw_offset_deg: float = 0.0,
) -> tuple[float, float]:
    """Map normalized equirectangular panorama coordinates to azimuth/elevation.

    The image center is front. Positive azimuth points left, and azimuth is
    wrapped into the half-open interval [-180, 180).
    """
    u = float(center_u)
    v = float(center_v)
    yaw = float(yaw_offset_deg)
    if not all(math.isfinite(value) for value in (u, v, yaw)):
        raise ValueError("center_u, center_v, and yaw_offset_deg must be finite")

    azimuth_deg = -360.0 * (u - 0.5)
    azimuth_deg += yaw
    azimuth_deg = ((azimuth_deg + 180.0) % 360.0) - 180.0

    elevation_deg = 90.0 - 180.0 * v
    elevation_deg = max(-90.0, min(90.0, elevation_deg))
    return azimuth_deg, elevation_deg
