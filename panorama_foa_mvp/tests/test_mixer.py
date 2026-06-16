from __future__ import annotations

import numpy as np
import pytest

from panorama_foa.ambisonics.mixer import mix_foa_layers


def test_mix_foa_layers_uses_one_global_scalar_and_preserves_ratios():
    layer_a = np.ones((16, 4), dtype=np.float32)
    layer_b = np.tile([0.5, 0.25, 0.125, 0.0625], (16, 1)).astype(np.float32)
    original_a = layer_a.copy()
    original_b = layer_b.copy()

    mixed, scale = mix_foa_layers([layer_a, layer_b], return_scale=True)

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
