from __future__ import annotations

import numpy as np
import pytest

from panorama_foa.audio.processing import process_mono_audio


def test_process_mono_audio_folds_down_removes_dc_and_applies_gain():
    sample_rate = 24000
    t = np.arange(sample_rate // 2) / sample_rate
    left = 0.2 * np.sin(2 * np.pi * 440 * t) + 0.1
    right = 0.1 * np.sin(2 * np.pi * 440 * t) + 0.1
    stereo = np.stack([left, right], axis=1).astype(np.float32)

    processed = process_mono_audio(
        stereo,
        sample_rate=sample_rate,
        duration_seconds=1.0,
        gain_db=-6.0,
        source_id="tone",
    )

    assert processed.dtype == np.float32
    assert processed.shape == (48000,)
    assert abs(float(np.mean(processed))) < 1e-4
    expected_peak = 10 ** (-6.0 / 20.0) * 10 ** (-6.0 / 20.0)
    assert np.max(np.abs(processed)) == pytest.approx(expected_peak, rel=0.05)


def test_process_mono_audio_rejects_near_silence():
    with pytest.raises(ValueError, match="silent_source"):
        process_mono_audio(
            np.zeros(1000, dtype=np.float32),
            sample_rate=48000,
            duration_seconds=1.0,
            gain_db=-12.0,
            source_id="silent_source",
        )


def test_process_mono_audio_pads_short_small_gap():
    samples = np.sin(np.linspace(0, np.pi, 47000, dtype=np.float32))
    processed = process_mono_audio(
        samples,
        sample_rate=48000,
        duration_seconds=1.0,
        gain_db=-12.0,
        source_id="short_source",
    )
    assert processed.shape == (48000,)

