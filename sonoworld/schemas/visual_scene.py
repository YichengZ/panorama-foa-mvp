from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Literal

from .common import SerializableDataclass, FileRef


VisualRepresentationType = Literal[
    "gaussian_splatting",
    "mesh",
    "point_cloud",
    "custom",
]


@dataclass
class PanoramaGeometry(SerializableDataclass):
    """Geometry maps aligned with the panorama image."""

    depth: FileRef
    points: FileRef

    mask: Optional[FileRef] = None
    normals: Optional[FileRef] = None
    rays: Optional[FileRef] = None

    width: Optional[int] = None
    height: Optional[int] = None
    max_depth: Optional[float] = None

    coordinate_system: str = "pano"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class VisualRepresentation(SerializableDataclass):
    """A renderable 3D visual representation owned by the visual scene module."""

    representation_type: VisualRepresentationType
    backend: str

    primary_file: Optional[FileRef] = None
    files: Dict[str, FileRef] = field(default_factory=dict)

    coordinate_system: str = "opencv"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RenderPreview(SerializableDataclass):
    """Optional preview renders for validation."""

    equirectangular: Optional[FileRef] = None
    pinhole: Optional[FileRef] = None
    minimap: Optional[FileRef] = None

    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class VisualSceneSummary(SerializableDataclass):
    """Unified output of the visual scene stage."""

    panorama: FileRef

    geometry: Optional[PanoramaGeometry] = None
    representation: Optional[VisualRepresentation] = None
    preview: Optional[RenderPreview] = None

    backend: Optional[str] = None
    status: str = "done"

    metadata: Dict[str, Any] = field(default_factory=dict)

    def has_renderable_representation(self) -> bool:
        return self.representation is not None and self.representation.primary_file is not None

    def has_panorama_geometry(self) -> bool:
        return self.geometry is not None and self.geometry.depth is not None and self.geometry.points is not None