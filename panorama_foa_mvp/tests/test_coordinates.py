from __future__ import annotations

import pytest

from panorama_foa.coordinates import normalized_panorama_to_angles


@pytest.mark.parametrize(
    ("center_u", "center_v", "azimuth", "elevation"),
    [
        (0.50, 0.50, 0.0, 0.0),
        (0.25, 0.50, 90.0, 0.0),
        (0.75, 0.50, -90.0, 0.0),
        (0.00, 0.50, -180.0, 0.0),
        (0.50, 0.00, 0.0, 90.0),
        (0.50, 1.00, 0.0, -90.0),
    ],
)
def test_normalized_panorama_to_angles(center_u, center_v, azimuth, elevation):
    actual_azimuth, actual_elevation = normalized_panorama_to_angles(center_u, center_v)
    assert actual_azimuth == pytest.approx(azimuth)
    assert actual_elevation == pytest.approx(elevation)


def test_yaw_offset_wraps_to_half_open_range():
    azimuth, elevation = normalized_panorama_to_angles(0.25, 0.5, yaw_offset_deg=120.0)
    assert azimuth == pytest.approx(-150.0)
    assert elevation == pytest.approx(0.0)
    assert -180.0 <= azimuth < 180.0

