from __future__ import annotations

from dataclasses import dataclass, field, asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Literal, Type, TypeVar, Union
import json


T = TypeVar("T")


JsonDict = Dict[str, Any]
PathLike = Union[str, Path]


def to_jsonable(value: Any) -> Any:
    """Convert dataclass, Path, list, and dict objects into JSON-serializable values."""
    if isinstance(value, Path):
        return str(value)

    if is_dataclass(value):
        return {
            key: to_jsonable(val)
            for key, val in asdict(value).items()
        }

    if isinstance(value, dict):
        return {
            str(key): to_jsonable(val)
            for key, val in value.items()
        }

    if isinstance(value, list):
        return [to_jsonable(item) for item in value]

    if isinstance(value, tuple):
        return [to_jsonable(item) for item in value]

    return value


def read_json(path: PathLike) -> JsonDict:
    """Read a JSON file."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: PathLike, data: Any, indent: int = 2) -> None:
    """Write data to a JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(to_jsonable(data), f, indent=indent, ensure_ascii=False)


@dataclass
class SerializableDataclass:
    """Base class for dataclasses that can be saved to and loaded from JSON."""

    def to_dict(self) -> JsonDict:
        return to_jsonable(self)

    def save_json(self, path: PathLike) -> None:
        write_json(path, self.to_dict())

    @classmethod
    def from_dict(cls: Type[T], data: JsonDict) -> T:
        return cls(**data)

    @classmethod
    def load_json(cls: Type[T], path: PathLike) -> T:
        return cls.from_dict(read_json(path))


@dataclass
class FileRef(SerializableDataclass):
    """A path reference stored relative to the scene root when possible."""

    path: str
    role: Optional[str] = None
    media_type: Optional[str] = None

    @classmethod
    def from_path(
        cls,
        path: PathLike,
        scene_root: Optional[PathLike] = None,
        role: Optional[str] = None,
        media_type: Optional[str] = None,
    ) -> "FileRef":
        path = Path(path)

        if scene_root is not None:
            scene_root = Path(scene_root)
            try:
                path = path.relative_to(scene_root)
            except ValueError:
                pass

        return cls(
            path=str(path),
            role=role,
            media_type=media_type,
        )

    def resolve(self, scene_root: PathLike) -> Path:
        path = Path(self.path)
        if path.is_absolute():
            return path
        return Path(scene_root) / path


@dataclass
class BBox(SerializableDataclass):
    """Bounding box in pixel coordinates."""

    y_min: int
    x_min: int
    y_max: int
    x_max: int

    @classmethod
    def from_xyxy(cls, x_min: int, y_min: int, x_max: int, y_max: int) -> "BBox":
        return cls(
            y_min=y_min,
            x_min=x_min,
            y_max=y_max,
            x_max=x_max,
        )

    @classmethod
    def from_yxyx(cls, y_min: int, x_min: int, y_max: int, x_max: int) -> "BBox":
        return cls(
            y_min=y_min,
            x_min=x_min,
            y_max=y_max,
            x_max=x_max,
        )

    def to_xyxy(self) -> List[int]:
        return [self.x_min, self.y_min, self.x_max, self.y_max]

    def to_yxyx(self) -> List[int]:
        return [self.y_min, self.x_min, self.y_max, self.x_max]


@dataclass
class StageIO(SerializableDataclass):
    """Input and output file references for a stage."""

    inputs: Dict[str, FileRef] = field(default_factory=dict)
    outputs: Dict[str, FileRef] = field(default_factory=dict)
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ErrorInfo(SerializableDataclass):
    """Serializable error payload for failed stages."""

    type: str
    message: str
    traceback: Optional[str] = None