from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class AudioDiffusionConfig:
    """Runtime metadata shared by audio diffusion backends."""

    sample_rate: int
    steps: int = 0
    guidance_scale: float = 1.0
    seconds_start: float | None = 0.0
    seconds_total: float | None = None
    model_name: str = ""
    variant: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AudioDiffusionConditioning:
    """Opaque conditioning payload owned by each backend."""

    metadata: dict[str, Any] = field(default_factory=dict)


class AudioDiffusionModel(ABC):
    """Small wrapper interface for optional local audio diffusion backends."""

    supports_t2a: bool = False
    supports_v2a: bool = False

    @abstractmethod
    def get_config(self) -> AudioDiffusionConfig:
        raise NotImplementedError

    def generate_t2a(
        self,
        prompt: str,
        *,
        negative_prompt: str | None = None,
        condition: AudioDiffusionConditioning | None = None,
        config: AudioDiffusionConfig | None = None,
        **kwargs: Any,
    ) -> Any:
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support text-to-audio generation."
        )

    def generate_v2a(
        self,
        video_path: str | Path,
        *,
        prompt: str | None = None,
        negative_prompt: str | None = None,
        condition: AudioDiffusionConditioning | None = None,
        config: AudioDiffusionConfig | None = None,
        **kwargs: Any,
    ) -> Any:
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support video-to-audio generation."
        )

    def generate_audio(
        self,
        *,
        text_prompt: str | None = None,
        video_path: str | Path | None = None,
        condition: AudioDiffusionConditioning | None = None,
        config: AudioDiffusionConfig | None = None,
        **kwargs: Any,
    ) -> Any:
        if video_path is not None:
            return self.generate_v2a(
                video_path,
                prompt=text_prompt,
                condition=condition,
                config=config,
                **kwargs,
            )

        if text_prompt is None:
            raise ValueError("text_prompt is required for text-to-audio generation.")

        return self.generate_t2a(
            text_prompt,
            condition=condition,
            config=config,
            **kwargs,
        )

    @property
    def sample_rate(self) -> int:
        return int(self.get_config().sample_rate)


def load_audio_diffusion_model(backend: str, **kwargs: Any) -> AudioDiffusionModel:
    key = backend.strip().lower().replace("-", "_")
    if key in {"mmaudio", "mm_audio"}:
        from panorama_foa.audio.backends.mmaudio_diffusion import MMAudioDiffusion

        return MMAudioDiffusion(**kwargs)

    raise ValueError(f"Unknown audio diffusion backend: {backend}")

