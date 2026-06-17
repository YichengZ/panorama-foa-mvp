from __future__ import annotations

import numpy as np
import pytest

from panorama_foa.ambisonics.mixer import mix_foa_layers


def test_mix_foa_layers_uses_one_global_scalar_and_preserves_ratios():
    layer_a = np.ones((16, 4), dtype=np.float32)
    layer_b = np.tile([0.5, 0.25, 0.125, 0.0625], (16, 1)).astype(np.float32)
    original_a = layer_a.copy()
    original_b = layer_b.copy()

    mixed, scale = mix_foa_layers([layer_a, layer_b], target_rms_dbfs=None, target_dbfs=-1.0, return_scale=True)

    target_peak = 10 ** (-1.0 / 20.0)
    summed = original_a + original_b
    expected_scale = target_peak / np.max(np.abs(summed))
    assert scale == pytest.approx(expected_scale)
    np.testing.assert_allclose(mixed, summed * expected_scale, atol=1e-6)
    assert np.max(np.abs(mixed)) <= target_peak + 1e-6
    np.testing.assert_allclose(layer_a, original_a)
    np.testing.assert_allclose(layer_b, original_b)


def test_mix_foa_layers_rejects_mismatched_lengths():
    layer_a = np.ones((16, 4), dtype=np.float32)
    layer_b = np.ones((15, 4), dtype=np.float32)
    with pytest.raises(ValueError, match="same shape"):
        mix_foa_layers([layer_a, layer_b])


def test_mix_foa_layers_targets_environment_rms_before_peak_ceiling():
    layer = np.full((128, 4), 0.001, dtype=np.float32)

    mixed = mix_foa_layers([layer], target_rms_dbfs=-30.0, target_dbfs=-3.0)

    rms = float(np.sqrt(np.mean(mixed.astype(np.float64) ** 2)))
    assert 20 * np.log10(rms) == pytest.approx(-30.0, abs=0.05)
    assert np.max(np.abs(mixed)) <= 10 ** (-3.0 / 20.0)
