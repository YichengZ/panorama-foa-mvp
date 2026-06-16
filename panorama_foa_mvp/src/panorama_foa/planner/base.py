from __future__ import annotations

from pathlib import Path
from typing import Protocol

from panorama_foa.schemas import ScenePlan


class ScenePlanner(Protocol):
    def plan(
        self,
        *,
        panorama_path: Path,
        duration_seconds: float,
        scene_description: str | None,
        max_sources: int,
        allow_speech: bool,
        allow_music: bool,
    ) -> ScenePlan:
        ...
