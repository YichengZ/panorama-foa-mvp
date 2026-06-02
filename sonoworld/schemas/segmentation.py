from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .common import SerializableDataclass, FileRef, BBox


@dataclass
class SegmentationInstance(SerializableDataclass):
    """One segmented object or region in the panorama."""

    instance_id: str
    class_label: str

    mask: FileRef
    overlay: Optional[FileRef] = None

    score: Optional[float] = None
    area: Optional[int] = None
    bbox: Optional[BBox] = None

    source_prompt: Optional[str] = None
    source_tile_indices: List[int] = field(default_factory=list)

    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SegmentationSummary(SerializableDataclass):
    """Summary output of the segmentation stage."""

    panorama: Optional[FileRef] = None
    instances: List[SegmentationInstance] = field(default_factory=list)

    backend: Optional[str] = None
    model: Optional[str] = None

    metadata: Dict[str, Any] = field(default_factory=dict)

    def instances_by_label(self) -> Dict[str, List[SegmentationInstance]]:
        grouped: Dict[str, List[SegmentationInstance]] = {}
        for inst in self.instances:
            grouped.setdefault(inst.class_label, []).append(inst)
        return grouped

    def count_by_label(self) -> Dict[str, int]:
        return {
            label: len(items)
            for label, items in self.instances_by_label().items()
        }

    def mask_paths_by_label(self) -> Dict[str, List[str]]:
        grouped = self.instances_by_label()
        return {
            label: [inst.mask.path for inst in items]
            for label, items in grouped.items()
        }