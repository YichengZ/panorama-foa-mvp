from __future__ import annotations

import pytest

from panorama_foa.audio.mock import MockTextToAudioProvider
from panorama_foa.factory import (
    FactoryConfigurationError,
    build_audio_provider,
    build_pipeline,
    build_planner,
    normalize_choice,
)
from panorama_foa.pipeline import PanoramaToFOAPipeline
from panorama_foa.planner.manual import ManualPlanPlanner


def test_normalize_choice_trims_and_lowercases():
    assert normalize_choice("  MMAUDIO ") == "mmaudio"


def test_factory_builds_manual_mock_pipeline(fixtures_dir):
    pipeline = build_pipeline(
        planner_name="manual",
        plan_path=fixtures_dir / "manual_plan.json",
        audio_provider_name="mock",
        yaw_offset=15.0,
        force=True,
    )

    assert isinstance(pipeline, PanoramaToFOAPipeline)
    assert isinstance(pipeline.planner, ManualPlanPlanner)
    assert isinstance(pipeline.audio_provider, MockTextToAudioProvider)
    assert pipeline.yaw_offset_deg == 15.0
    assert pipeline.force is True


def test_factory_rejects_unknown_planner():
    with pytest.raises(FactoryConfigurationError, match="unknown planner"):
        build_planner(planner_name="manul", plan_path=None)


def test_factory_requires_plan_for_manual_planner():
    with pytest.raises(FactoryConfigurationError, match="--plan is required"):
        build_planner(planner_name="manual", plan_path=None)


def test_factory_rejects_unknown_audio_provider():
    with pytest.raises(FactoryConfigurationError, match="unknown audio provider"):
        build_audio_provider(audio_provider_name="elevenlab")


def test_factory_builds_mmaudio_provider_with_monkeypatch(monkeypatch):
    import panorama_foa.audio.mmaudio as mmaudio_module

    class FakeMMAudioProvider:
        raw_extension = ".wav"

        def __init__(self, *, settings):
            self.settings = settings

    monkeypatch.setattr(mmaudio_module, "MMAudioTextToAudioProvider", FakeMMAudioProvider)

    provider = build_audio_provider(audio_provider_name="mmaudio")

    assert isinstance(provider, FakeMMAudioProvider)
