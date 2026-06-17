from __future__ import annotations

import numpy as np
import pytest

from panorama_foa.audio.processing import process_mono_audio, requested_generation_duration


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


def test_requested_generation_duration_adds_loop_post_roll_only_for_looped_sources():
    assert requested_generation_duration(15.0, loop=True) == pytest.approx(17.0)
    assert requested_generation_duration(15.0, loop=False) == pytest.approx(15.0)


def test_process_mono_audio_uses_post_roll_to_crossfade_loop_start():
    sample_rate = 48000
    t = np.arange(sample_rate, dtype=np.float32) / sample_rate
    body = (0.4 * np.sin(2 * np.pi * 3 * t)).astype(np.float32)
    tail = (0.4 * np.cos(2 * np.pi * 5 * t)).astype(np.float32)

    looped = process_mono_audio(
        np.concatenate([body, tail]),
        sample_rate=sample_rate,
        duration_seconds=1.0,
        gain_db=0.0,
        source_id="looped",
        loop=True,
        loop_crossfade_seconds=1.0,
    )
    unlooped = process_mono_audio(
        np.concatenate([body, tail]),
        sample_rate=sample_rate,
        duration_seconds=1.0,
        gain_db=0.0,
        source_id="unlooped",
        loop=False,
    )

    assert looped.shape == (sample_rate,)
    assert np.max(np.abs(looped)) > 0.0
    assert not np.allclose(looped[:1000], unlooped[:1000])
