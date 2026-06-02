from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Literal

from .common import SerializableDataclass, FileRef


SpatialSourceType = Literal["point", "area", "background"]


@dataclass
class SpatialPointCloud(SerializableDataclass):
    """Point cloud assets used by one spatial source."""

    full_points: Optional[FileRef] = None
    downsampled_points: Optional[FileRef] = None
    weights: Optional[FileRef] = None

    num_full_points: Optional[int] = None
    num_downsampled_points: Optional[int] = None

    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SpatialAudioSource(SerializableDataclass):
    """One spatialized audio source."""

    source_id: str

    class_label: str
    grounding_label: str
    source_type: SpatialSourceType

    audio: FileRef

    instance_id: Optional[str] = None
    prompt_id: Optional[int] = None

    centroid: Optional[List[float]] = None
    point_cloud: Optional[SpatialPointCloud] = None

    peak_db: float = -6.0

    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SpatialAudioOptionGroup(SerializableDataclass):
    """Alternative spatial sources for one object instance or semantic label."""

    group_id: str
    class_label: str
    instance_id: Optional[str] = None

    source_ids: List[str] = field(default_factory=list)
    selected_source_id: Optional[str] = None

    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SpatialAudioConfiguration(SerializableDataclass):
    """Final spatial audio configuration consumed by rendering."""

    sources: List[SpatialAudioSource] = field(default_factory=list)
    option_groups: List[SpatialAudioOptionGroup] = field(default_factory=list)
    selected_sources: List[str] = field(default_factory=list)

    coordinate_system: str = "pano"
    sample_rate: int = 48000

    backend: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def source_by_id(self) -> Dict[str, SpatialAudioSource]:
        return {
            source.source_id: source
            for source in self.sources
        }

    def selected_source_objects(self) -> List[SpatialAudioSource]:
        lookup = self.source_by_id()
        return [
            lookup[source_id]
            for source_id in self.selected_sources
            if source_id in lookup
        ]

    def to_legacy_choice_json(self) -> List[Dict[str, Any]]:
        """Export a format compatible with the current inference code."""
        items = []

        for source in self.selected_source_objects():
            item = {
                "class_label": source.class_label,
                "audio_path": self._legacy_audio_path(source.audio.path),
                "source_type": source.source_type,
                "peak_db": source.peak_db,
            }

            if source.centroid is not None:
                item["centroid"] = source.centroid

            if source.point_cloud is not None:
                if source.point_cloud.downsampled_points is not None:
                    path = self._legacy_spatial_path(source.point_cloud.downsampled_points.path)
                    item["pcd"] = path
                    item["pcd_points_down"] = path

                if source.point_cloud.full_points is not None:
                    item["pcd_points"] = self._legacy_spatial_path(
                        source.point_cloud.full_points.path
                    )

                if source.point_cloud.weights is not None:
                    item["pcd_weights"] = self._legacy_spatial_path(
                        source.point_cloud.weights.path
                    )

            items.append(item)

        return items

    def to_legacy_meta_grounding_json(self) -> List[Dict[str, Any]]:
        """Export all sources in a format close to the current meta-grounding.json."""
        items = []

        for source in self.sources:
            item = {
                "class_label": source.class_label,
                "audio_path": self._legacy_audio_path(source.audio.path),
                "source_type": source.source_type,
                "peak_db": source.peak_db,
            }

            if source.centroid is not None:
                item["centroid"] = source.centroid

            if source.point_cloud is not None:
                if source.point_cloud.downsampled_points is not None:
                    path = self._legacy_spatial_path(source.point_cloud.downsampled_points.path)
                    item["pcd"] = path
                    item["pcd_points_down"] = path

                if source.point_cloud.full_points is not None:
                    item["pcd_points"] = self._legacy_spatial_path(
                        source.point_cloud.full_points.path
                    )

                if source.point_cloud.weights is not None:
                    item["pcd_weights"] = self._legacy_spatial_path(
                        source.point_cloud.weights.path
                    )

            items.append(item)

        return items

    def to_legacy_options_json(self) -> Dict[str, List[List[int]]]:
        """Export option groups as index lists for compatibility with old code."""
        source_index = {
            source.source_id: idx
            for idx, source in enumerate(self.sources)
        }

        grouped: Dict[str, List[List[int]]] = {}

        for group in self.option_groups:
            indices = [
                source_index[source_id]
                for source_id in group.source_ids
                if source_id in source_index
            ]
            grouped.setdefault(group.class_label, []).append(indices)

        return grouped

    def _legacy_audio_path(self, path: str) -> str:
        prefix = "audio/"
        if path.startswith(prefix):
            return path[len(prefix):]
        return path

    def _legacy_spatial_path(self, path: str) -> str:
        prefix = "spatial/"
        if path.startswith(prefix):
            return path[len(prefix):]
        return path
