from __future__ import annotations

from pathlib import Path
from typing import Protocol


class TextToAudioProvider(Protocol):
    def generate(
        self,
        *,
        prompt: str,
        duration_seconds: float,
        loop: bool,
        output_path: Path,
    ) -> Path:
        ...

