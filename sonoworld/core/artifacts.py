from __future__ import annotations

from pathlib import Path
from typing import Optional

from sonoworld.schemas.common import FileRef, write_json
from sonoworld.schemas.understanding import SceneUnderstanding
from sonoworld.schemas.segmentation import SegmentationSummary
from sonoworld.schemas.visual_scene import VisualSceneSummary
from sonoworld.schemas.audio import AudioGenerationSummary
from sonoworld.schemas.spatial import SpatialAudioConfiguration
from sonoworld.core.paths import ScenePaths


def ref(path: Path, scene_root: Path, role: Optional[str] = None, media_type: Optional[str] = None) -> FileRef:
    """Create a scene-root-relative file reference."""
    return FileRef.from_path(
        path=path,
        scene_root=scene_root,
        role=role,
        media_type=media_type,
    )


def save_understanding(paths: ScenePaths, data: SceneUnderstanding) -> Path:
    paths.understanding.mkdir(parents=True, exist_ok=True)
    data.save_json(paths.understanding_metadata)

    grounding_input = data.to_grounding_input()
    write_json(paths.grounding_input, grounding_input)

    legacy = data.to_legacy_understanding_list()
    write_json(paths.understanding / "understanding_legacy.json", legacy)

    return paths.understanding_metadata


def load_understanding(paths: ScenePaths) -> SceneUnderstanding:
    return SceneUnderstanding.load_json(paths.understanding_metadata)


def save_segmentation_summary(paths: ScenePaths, data: SegmentationSummary) -> Path:
    paths.segmentation.mkdir(parents=True, exist_ok=True)
    data.save_json(paths.segmentation_summary)
    return paths.segmentation_summary


def load_segmentation_summary(paths: ScenePaths) -> SegmentationSummary:
    return SegmentationSummary.load_json(paths.segmentation_summary)


def save_visual_scene_summary(paths: ScenePaths, data: VisualSceneSummary) -> Path:
    paths.visual_scene.mkdir(parents=True, exist_ok=True)
    data.save_json(paths.visual_scene_summary)
    return paths.visual_scene_summary


def load_visual_scene_summary(paths: ScenePaths) -> VisualSceneSummary:
    return VisualSceneSummary.load_json(paths.visual_scene_summary)


def save_audio_summary(paths: ScenePaths, data: AudioGenerationSummary) -> Path:
    paths.audio.mkdir(parents=True, exist_ok=True)
    data.save_json(paths.audio_summary)
    return paths.audio_summary


def load_audio_summary(paths: ScenePaths) -> AudioGenerationSummary:
    return AudioGenerationSummary.load_json(paths.audio_summary)


def save_spatial_configuration(paths: ScenePaths, data: SpatialAudioConfiguration) -> Path:
    paths.spatial.mkdir(parents=True, exist_ok=True)
    data.save_json(paths.spatial_summary)

    write_json(paths.legacy_choice, data.to_legacy_choice_json())
    write_json(paths.legacy_options, data.to_legacy_options_json())
    write_json(paths.legacy_meta_grounding, data.to_legacy_meta_grounding_json())

    return paths.spatial_summary


def load_spatial_configuration(paths: ScenePaths) -> SpatialAudioConfiguration:
    return SpatialAudioConfiguration.load_json(paths.spatial_summary)