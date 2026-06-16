from __future__ import annotations

import json

import soundfile as sf
from typer.testing import CliRunner

from panorama_foa.cli import app
from panorama_foa.pipeline import validate_and_filter_plan
from panorama_foa.schemas import GlobalAmbience, RegionalSource, ScenePlan


def test_validate_and_filter_plan_caps_filters_dedupes_and_overrides_duration():
    sources = [
        RegionalSource(
            id=f"source_{index}",
            label="duplicate" if index < 2 else f"source {index}",
            prompt=f"prompt {index}",
            center_u=0.1 * index,
            center_v=0.5,
            spread=0.2,
            gain_db=-12.0,
            confidence=1.0 if index != 6 else 0.2,
        )
        for index in range(7)
    ]
    plan = ScenePlan(
        scene_description="test",
        duration_seconds=9.0,
        global_ambience=GlobalAmbience(
            label="ambience",
            prompt="quiet ambience",
            gain_db=-24.0,
            confidence=1.0,
        ),
        regional_sources=sources,
    )

    filtered = validate_and_filter_plan(plan, duration_seconds=2.0, max_sources=5)
    assert filtered.duration_seconds == 2.0
    assert len(filtered.regional_sources) <= 5
    assert all(source.confidence >= 0.6 for source in filtered.regional_sources)
    labels = [source.label for source in filtered.regional_sources]
    assert len(labels) == len(set(labels))


def test_mock_cli_end_to_end_outputs_contract_files(tmp_path, panorama_fixture, fixtures_dir):
    output = tmp_path / "scene"
    result = CliRunner().invoke(
        app,
        [
            "generate",
            "--panorama",
            str(panorama_fixture),
            "--output",
            str(output),
            "--planner",
            "manual",
            "--plan",
            str(fixtures_dir / "manual_plan.json"),
            "--audio-provider",
            "mock",
            "--duration",
            "1.0",
        ],
    )
    assert result.exit_code == 0, result.output

    assert (output / "scene_plan.json").exists()
    assert (output / "scene_foa.wav").exists()
    assert (output / "scene_foa.metadata.json").exists()
    assert (output / "panorama_analysis.jpg").exists()
    assert (output / "raw_audio" / "global_ambience.wav").exists()
    assert (output / "stems" / "global_ambience.wav").exists()

    info = sf.info(output / "scene_foa.wav")
    assert info.samplerate == 48000
    assert info.channels == 4
    assert info.frames == 48000

    metadata = json.loads((output / "scene_foa.metadata.json").read_text())
    assert metadata["ambisonics"]["channel_labels"] == ["W", "Y", "Z", "X"]
    assert metadata["ambisonics"]["channel_ordering"] == "ACN"
    assert metadata["ambisonics"]["normalization"] == "SN3D"

    plan = json.loads((output / "scene_plan.json").read_text())
    assert plan["duration_seconds"] == 1.0

