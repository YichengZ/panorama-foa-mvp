from __future__ import annotations

import numpy as np
import pytest
import soundfile as sf

from panorama_foa.audio.mmaudio import (
    MMAudioTextToAudioProvider,
    _as_mono_float32,
    _find_repo_root,
    _normalize_sonoworld_root,
)
from panorama_foa.config import ConfigurationError, MMAudioConfig, mmaudio_config_from_env


class FakeMMAudioAdapter:
    sample_rate = 32000

    def __init__(self) -> None:
        self.calls = []

    def generate(self, prompt: str, *, duration_seconds: float) -> np.ndarray:
        self.calls.append((prompt, duration_seconds))
        frames = int(round(self.sample_rate * duration_seconds))
        t = np.linspace(0.0, duration_seconds, frames, endpoint=False, dtype=np.float32)
        left = np.sin(2.0 * np.pi * 220.0 * t)
        right = np.sin(2.0 * np.pi * 330.0 * t)
        return np.stack([left, right], axis=0)


def test_mmaudio_provider_uses_adapter_and_writes_raw_wav(tmp_path):
    adapter = FakeMMAudioAdapter()
    provider = MMAudioTextToAudioProvider.from_config(
        MMAudioConfig(model_path=str(tmp_path / "weights"), device="cuda:0"),
        adapter=adapter,
    )

    output = provider.generate(
        prompt="soft cabin ventilation hum",
        duration_seconds=0.25,
        loop=True,
        output_path=tmp_path / "raw.wav",
    )

    assert output == tmp_path / "raw.wav"
    assert adapter.calls == [("soft cabin ventilation hum", 0.25)]
    assert provider.raw_extension == ".wav"
    info = sf.info(output)
    assert info.samplerate == 32000
    assert info.channels == 1
    assert info.frames == 8000


def test_as_mono_float32_folds_channel_first_and_channel_last_audio():
    channel_first = np.array([[0.2, 0.4, 0.8], [0.6, 1.4, -0.2]], dtype=np.float32)
    channel_last = np.array([[0.2, 0.6], [0.4, 1.4], [0.8, -0.2]], dtype=np.float32)
    expected = np.array([0.4, 0.9, 0.3], dtype=np.float32)

    assert np.allclose(_as_mono_float32(channel_first), expected)
    assert np.allclose(_as_mono_float32(channel_last), expected)


def test_as_mono_float32_clips_and_flattens_extra_candidate_dimension():
    audio = np.array([[[2.0, -2.0, 0.25]]], dtype=np.float32)

    assert np.allclose(
        _as_mono_float32(audio),
        np.array([1.0, -1.0, 0.25], dtype=np.float32),
    )


def test_mmaudio_config_from_env_parses_server_settings(tmp_path):
    config = mmaudio_config_from_env(
        environ={
            "MMAUDIO_MODEL_VARIANT": "small_44k_v2",
            "MMAUDIO_MODEL_PATH": "/models/mmaudio",
            "MMAUDIO_DEVICE": "cuda:3",
            "MMAUDIO_STEPS": "32",
            "MMAUDIO_GUIDANCE_SCALE": "8.25",
            "MMAUDIO_FULL_PRECISION": "true",
            "MMAUDIO_INFERENCE_MODE": "heun",
            "MMAUDIO_SONOWORLD_ROOT": str(tmp_path / "sonoworld"),
        }
    )

    assert config.model_variant == "small_44k_v2"
    assert config.model_path == "/models/mmaudio"
    assert config.device == "cuda:3"
    assert config.steps == 32
    assert config.guidance_scale == 8.25
    assert config.full_precision is True
    assert config.inference_mode == "heun"
    assert config.sonoworld_root == tmp_path / "sonoworld"


def test_mmaudio_config_rejects_invalid_bool():
    with pytest.raises(ConfigurationError):
        mmaudio_config_from_env(environ={"MMAUDIO_FULL_PRECISION": "maybe"})


def test_mmaudio_adapter_can_find_repo_root_for_sonoworld_imports():
    root = _find_repo_root()

    assert root is not None
    assert (root / "sonoworld" / "models" / "audio_diffusion" / "mmaudio.py").exists()


def test_mmaudio_sonoworld_root_accepts_repo_or_package_path():
    root = _find_repo_root()
    assert root is not None

    assert _normalize_sonoworld_root(root) == root
    assert _normalize_sonoworld_root(root / "sonoworld") == root
