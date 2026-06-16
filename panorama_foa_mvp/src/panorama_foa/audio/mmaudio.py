from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import soundfile as sf

from panorama_foa.config import (
    MMAudioConfig,
    PROJECT_ROOT,
    Settings,
    mmaudio_config_from_env,
)


class MMAudioProviderError(RuntimeError):
    """Raised when local MMAudio generation cannot run."""


class MMAudioAdapter(Protocol):
    sample_rate: int

    def generate(self, prompt: str, *, duration_seconds: float) -> np.ndarray:
        ...


class SonoWorldMMAudioAdapter:
    """Thin adapter around SonoWorld's existing MMAudio wrapper."""

    def __init__(self, config: MMAudioConfig) -> None:
        self.config = config
        sonoworld_root = _normalize_sonoworld_root(config.sonoworld_root) or _find_repo_root()
        if sonoworld_root is not None:
            root = str(sonoworld_root)
            if root not in sys.path:
                sys.path.insert(0, root)

        self.model_cls = self._load_model_cls()
        self.model: Any | None = None
        self._model_duration_seconds: float | None = None
        self.sample_rate = 44100

    def generate(self, prompt: str, *, duration_seconds: float) -> np.ndarray:
        model = self._get_model(duration_seconds)
        try:
            audio = model.generate_t2a(prompt)
        except Exception as exc:
            raise MMAudioProviderError(f"MMAudio generation failed: {exc}") from exc
        return _as_mono_float32(audio)

    def _load_model_cls(self) -> Any:
        try:
            from sonoworld.models.audio_diffusion.mmaudio import MMAudioDiffusion
        except ImportError as exc:
            raise MMAudioProviderError(
                "MMAudio is not installed. Install SonoWorld's MMAudio dependencies "
                "on the GPU server or set MMAUDIO_SONOWORLD_ROOT to that checkout."
            ) from exc
        return MMAudioDiffusion

    def _get_model(self, duration_seconds: float) -> Any:
        duration_seconds = float(duration_seconds)
        if (
            self.model is not None
            and self._model_duration_seconds is not None
            and abs(duration_seconds - self._model_duration_seconds) < 1e-6
        ):
            return self.model

        device = self.config.device or None
        try:
            model = self.model_cls(
                model_name="mmaudio",
                variant=self.config.model_variant,
                steps=self.config.steps,
                guidance_scale=self.config.guidance_scale,
                seconds_total=duration_seconds,
                full_precision=self.config.full_precision,
                inference_mode=self.config.inference_mode,
                model_path=self.config.model_path,
                device=device,
            )
        except Exception as exc:
            raise MMAudioProviderError(f"Failed to initialize MMAudio: {exc}") from exc

        if hasattr(model, "eval"):
            model = model.eval()
        self.model = model
        self._model_duration_seconds = duration_seconds
        self.sample_rate = int(model.get_config().sample_rate)
        return model


class MMAudioTextToAudioProvider:
    raw_extension = ".wav"

    def __init__(
        self,
        *,
        config: MMAudioConfig | None = None,
        settings: Settings | None = None,
        adapter: MMAudioAdapter | None = None,
    ) -> None:
        if settings is not None:
            config = settings.mmaudio_config()
        self.config = config or mmaudio_config_from_env()
        self.adapter = adapter or SonoWorldMMAudioAdapter(self.config)
        self.sample_rate = int(getattr(self.adapter, "sample_rate", 44100))

    @classmethod
    def from_config(
        cls,
        config: MMAudioConfig,
        *,
        adapter: MMAudioAdapter | None = None,
    ) -> "MMAudioTextToAudioProvider":
        return cls(config=config, adapter=adapter)

    def generate(
        self,
        *,
        prompt: str,
        duration_seconds: float,
        loop: bool,
        output_path: Path,
    ) -> Path:
        _ = loop
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        signal = self.adapter.generate(prompt, duration_seconds=duration_seconds)
        signal = _as_mono_float32(signal)
        self.sample_rate = int(getattr(self.adapter, "sample_rate", self.sample_rate))
        sf.write(output_path, signal, self.sample_rate, subtype="PCM_16")
        return output_path


def _as_mono_float32(audio: Any) -> np.ndarray:
    if hasattr(audio, "detach"):
        audio = audio.detach().cpu().numpy()
    array = np.asarray(audio, dtype=np.float32)
    while array.ndim > 2:
        array = array[0]
    if array.ndim == 2:
        if array.shape[0] <= 8 and array.shape[1] > array.shape[0]:
            array = array.mean(axis=0)
        else:
            array = array.mean(axis=1)
    elif array.ndim == 0:
        array = array.reshape(1)
    return np.clip(array.reshape(-1), -1.0, 1.0).astype(np.float32)


def _find_repo_root() -> Path | None:
    candidate = PROJECT_ROOT.parent
    marker = candidate / "sonoworld" / "models" / "audio_diffusion" / "mmaudio.py"
    if marker.exists():
        return candidate
    return None


def _normalize_sonoworld_root(path: Path | None) -> Path | None:
    if path is None:
        return None
    candidate = Path(path).expanduser()
    if (candidate / "sonoworld" / "models" / "audio_diffusion" / "mmaudio.py").exists():
        return candidate
    if (candidate / "models" / "audio_diffusion" / "mmaudio.py").exists():
        return candidate.parent
    return candidate
