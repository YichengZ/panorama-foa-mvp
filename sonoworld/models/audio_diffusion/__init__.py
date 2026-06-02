from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class AudioDiffusionConfig:
    """Runtime metadata shared by audio diffusion backends."""

    sample_rate: int
    steps: int = 0
    guidance_scale: float = 1.0
    seconds_start: Optional[float] = 0.0
    seconds_total: Optional[float] = None
    model_name: str = ""
    variant: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AudioDiffusionConditioning:
    """Opaque conditioning payload owned by each backend."""

    metadata: Dict[str, Any] = field(default_factory=dict)


class AudioDiffusionModel(ABC):
    """Small release-facing wrapper for audio diffusion models.

    Subclasses expose text-to-audio and video-to-audio separately. A backend
    that does not support one of them should keep the default implementation,
    which raises a clear error when that feature is requested.
    """

    supports_t2a: bool = False
    supports_v2a: bool = False

    @abstractmethod
    def get_config(self) -> AudioDiffusionConfig:
        raise NotImplementedError

    def generate_t2a(
        self,
        prompt: str,
        *,
        negative_prompt: Optional[str] = None,
        condition: Optional[AudioDiffusionConditioning] = None,
        config: Optional[AudioDiffusionConfig] = None,
        **kwargs: Any,
    ) -> Any:
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support text-to-audio generation."
        )

    def generate_v2a(
        self,
        video_path: str | Path,
        *,
        prompt: Optional[str] = None,
        negative_prompt: Optional[str] = None,
        condition: Optional[AudioDiffusionConditioning] = None,
        config: Optional[AudioDiffusionConfig] = None,
        **kwargs: Any,
    ) -> Any:
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support video-to-audio generation."
        )

    def generate_audio(
        self,
        *,
        text_prompt: Optional[str] = None,
        video_path: Optional[str | Path] = None,
        condition: Optional[AudioDiffusionConditioning] = None,
        config: Optional[AudioDiffusionConfig] = None,
        **kwargs: Any,
    ) -> Any:
        """Compatibility dispatcher for older callers."""
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
    """Build an audio diffusion model by backend name.

    Imports are intentionally local so optional backend dependencies are only
    loaded when the backend is actually requested.
    """
    key = backend.strip().lower().replace("-", "_")
    if key in {"mmaudio", "mm_audio"}:
        from sonoworld.models.audio_diffusion.mmaudio import MMAudioDiffusion

        return MMAudioDiffusion(**kwargs)

    raise ValueError(f"Unknown audio diffusion backend: {backend}")
