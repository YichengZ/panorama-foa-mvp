from __future__ import annotations

import json

import numpy as np
import soundfile as sf

from panorama_foa.ambisonics.exporter import export_foa_wav, write_metadata


def test_exporter_writes_pcm24_four_channel_wav_and_metadata(tmp_path):
    duration = 0.25
    samples = round(duration * 48000)
    audio = np.zeros((samples, 4), dtype=np.float32)
    audio[:, 0] = 0.1
    output_path = tmp_path / "scene_foa.wav"
    metadata_path = tmp_path / "scene_foa.metadata.json"

    export_foa_wav(output_path, audio, sample_rate=48000)
    write_metadata(metadata_path, duration_seconds=duration, yaw_offset_deg=15.0)

    info = sf.info(output_path)
    assert info.samplerate == 48000
    assert info.channels == 4
    assert info.frames == samples
    assert info.subtype == "PCM_24"

    metadata = json.loads(metadata_path.read_text())
    assert metadata["sample_rate"] == 48000
    assert metadata["channels"] == 4
    assert metadata["duration_seconds"] == duration
    assert metadata["ambisonics"]["convention"] == "AmbiX"
    assert metadata["ambisonics"]["channel_ordering"] == "ACN"
    assert metadata["ambisonics"]["normalization"] == "SN3D"
    assert metadata["ambisonics"]["channel_labels"] == ["W", "Y", "Z", "X"]
    assert metadata["coordinates"]["positive_azimuth"] == "left"
    assert metadata["coordinates"]["yaw_offset_deg"] == 15.0
    assert metadata["listener_model"]["translation_changes_audio"] is False

