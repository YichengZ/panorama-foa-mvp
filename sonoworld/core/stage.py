from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

from sonoworld.schemas.common import FileRef
from sonoworld.core.paths import ScenePaths


@dataclass
class StageContext:
    """Runtime context passed to every generation stage."""

    scene_root: Path
    config: Dict[str, Any]
    device: str = "cuda"
    force: bool = False

    @property
    def paths(self) -> ScenePaths:
        return ScenePaths(self.scene_root)

    def stage_config(self, stage_name: str) -> Dict[str, Any]:
        return self.config.get("stages", {}).get(stage_name, {})


@dataclass
class StageResult:
    """Result returned by one stage."""

    status: str

    inputs: Dict[str, FileRef] = field(default_factory=dict)
    outputs: Dict[str, FileRef] = field(default_factory=dict)

    message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class Stage(ABC):
    """Base class for all generation stages."""

    name: str
    backend: str

    def __init__(self, **kwargs: Any) -> None:
        ...

    @abstractmethod
    def run(self, ctx: StageContext) -> StageResult:
        raise NotImplementedError

    def should_skip(self, ctx: StageContext) -> bool:
        return False