from __future__ import annotations

import numpy as np
import pytest

from panorama_foa.ambisonics.foa import encode_global_ambience, encode_regional_foa


def assert_channels(actual: np.ndarray, expected):
    assert actual.dtype == np.float32
    assert actual.shape == (8, 4)
    np.testing.assert_allclose(actual, np.tile(expected, (8, 1)), atol=1e-6)


@pytest.mark.parametrize(
    ("azimuth", "elevation", "expected"),
    [
        (0.0, 0.0, [1.0, 0.0, 0.0, 1.0]),
        (90.0, 0.0, [1.0, 1.0, 0.0, 0.0]),
        (-90.0, 0.0, [1.0, -1.0, 0.0, 0.0]),
        (180.0, 0.0, [1.0, 0.0, 0.0, -1.0]),
        (33.0, 90.0, [1.0, 0.0, 1.0, 0.0]),
    ],
)
def test_encode_regional_foa_acn_sn3d_w_y_z_x(azimuth, elevation, expected):
    mono = np.ones(8, dtype=np.float32)
    encoded = encode_regional_foa(
        mono,
        azimuth_deg=azimuth,
        elevation_deg=elevation,
        spread=0.0,
    )
    assert_channels(encoded, expected)


def test_spread_one_collapses_directional_channels_to_w_only():
    mono = np.ones(8, dtype=np.float32)
    encoded = encode_regional_foa(mono, azimuth_deg=45.0, elevation_deg=20.0, spread=1.0)
    assert_channels(encoded, [1.0, 0.0, 0.0, 0.0])


def test_encode_global_ambience_is_w_only():
    mono = np.linspace(-1.0, 1.0, 8, dtype=np.float32)
    encoded = encode_global_ambience(mono)
    assert encoded.shape == (8, 4)
    np.testing.assert_allclose(encoded[:, 0], mono)
    np.testing.assert_allclose(encoded[:, 1:], 0.0)

