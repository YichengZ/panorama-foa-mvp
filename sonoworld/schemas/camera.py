from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Literal

from .common import SerializableDataclass, FileRef


CameraConvention = Literal["opencv", "opengl", "marble", "pano"]


@dataclass
class CameraFrame(SerializableDataclass):
    """One camera frame in a trajectory."""

    camera_to_world: List[float]

    frame_index: Optional[int] = None
    timestamp_sec: Optional[float] = None

    fov_x_deg: Optional[float] = None
    width: Optional[int] = None
    height: Optional[int] = None

    metadata: Dict[str, Any] = field(default_factory=dict)

    def matrix_4x4(self) -> List[List[float]]:
        if len(self.camera_to_world) != 16:
            raise ValueError("camera_to_world must contain 16 values.")

        return [
            self.camera_to_world[0:4],
            self.camera_to_world[4:8],
            self.camera_to_world[8:12],
            self.camera_to_world[12:16],
        ]


@dataclass
class CameraTrajectory(SerializableDataclass):
    """Camera trajectory used by inference."""

    camera_path: List[CameraFrame] = field(default_factory=list)

    convention: CameraConvention = "marble"
    fps: float = 30.0

    source: Optional[FileRef] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_marble_json(cls, data: Dict[str, Any]) -> "CameraTrajectory":
        frames = []

        for idx, item in enumerate(data.get("camera_path", [])):
            frames.append(
                CameraFrame(
                    camera_to_world=item["camera_to_world"],
                    frame_index=idx,
                    fov_x_deg=item.get("fov"),
                    metadata={
                        key: value
                        for key, value in item.items()
                        if key not in {"camera_to_world", "fov"}
                    },
                )
            )

        return cls(
            camera_path=frames,
            convention="marble",
            metadata={
                key: value
                for key, value in data.items()
                if key != "camera_path"
            },
        )