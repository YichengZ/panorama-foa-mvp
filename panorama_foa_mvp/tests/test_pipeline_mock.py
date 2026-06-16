from __future__ import annotations

import json

import numpy as np
import pytest
import soundfile as sf
from typer.testing import CliRunner

from panorama_foa.cli import app
from panorama_foa.pipeline import validate_and_filter_plan
from panorama_foa.schemas import GlobalAmbience, MAX_REGIONAL_SOURCES, RegionalSource, ScenePlan


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
            confidence=1.0 if index != 4 else 0.2,
        )
        for index in range(MAX_REGIONAL_SOURCES)
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

    filtered = validate_and_filter_plan(plan, duration_seconds=2.0, max_sources=3)
    assert filtered.duration_seconds == 2.0
    assert len(filtered.regional_sources) == 3
    assert all(source.confidence >= 0.6 for source in filtered.regional_sources)
    labels = [source.label for source in filtered.regional_sources]
    assert len(labels) == len(set(labels))


def test_scene_plan_rejects_more_than_five_regional_sources():
    sources = [
        RegionalSource(
            id=f"source_{index}",
            label=f"source {index}",
            prompt=f"prompt {index}",
            center_u=0.1,
            center_v=0.5,
            spread=0.2,
            gain_db=-12.0,
            confidence=1.0,
        )
        for index in range(MAX_REGIONAL_SOURCES + 1)
    ]

    with pytest.raises(ValueError):
        ScenePlan(
            scene_description="too many",
            duration_seconds=1.0,
            global_ambience=GlobalAmbience(
                label="ambience",
                prompt="quiet ambience",
                gain_db=-24.0,
                confidence=1.0,
            ),
            regional_sources=sources,
        )


def test_validate_and_filter_plan_enforces_hard_source_cap_even_when_requested_higher():
    sources = [
        RegionalSource(
            id=f"source_{index}",
            label=f"source {index}",
            prompt=f"prompt {index}",
            center_u=0.1,
            center_v=0.5,
            spread=0.2,
            gain_db=-12.0,
            confidence=1.0,
        )
        for index in range(MAX_REGIONAL_SOURCES)
    ]
    plan = ScenePlan(
        scene_description="test",
        duration_seconds=1.0,
        global_ambience=GlobalAmbience(
            label="ambience",
            prompt="quiet ambience",
            gain_db=-24.0,
            confidence=1.0,
        ),
        regional_sources=sources,
    )

    filtered = validate_and_filter_plan(plan, duration_seconds=1.0, max_sources=50)

    assert len(filtered.regional_sources) == MAX_REGIONAL_SOURCES


def test_cli_rejects_unknown_planner_without_falling_back_to_openai(
    monkeypatch,
    tmp_path,
    panorama_fixture,
    fixtures_dir,
):
    monkeypatch.setenv("OPENAI_API_KEY", "should-not-be-used")
    result = CliRunner().invoke(
        app,
        [
            "generate",
            "--panorama",
            str(panorama_fixture),
            "--output",
            str(tmp_path / "scene"),
            "--planner",
            "manul",
            "--plan",
            str(fixtures_dir / "manual_plan.json"),
            "--audio-provider",
            "mock",
        ],
    )

    assert result.exit_code != 0
    assert "unknown planner" in result.output
    assert "OpenAI planner is not configured" not in result.output


def test_cli_rejects_unknown_audio_provider(
    tmp_path,
    panorama_fixture,
    fixtures_dir,
):
    result = CliRunner().invoke(
        app,
        [
            "generate",
            "--panorama",
            str(panorama_fixture),
            "--output",
            str(tmp_path / "scene"),
            "--planner",
            "manual",
            "--plan",
            str(fixtures_dir / "manual_plan.json"),
            "--audio-provider",
            "elevenlab",
        ],
    )

    assert result.exit_code != 0
    assert "unknown audio provider" in result.output


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


def test_mmaudio_cli_path_can_run_with_fake_local_provider(
    monkeypatch,
    tmp_path,
    panorama_fixture,
    fixtures_dir,
):
    from panorama_foa.audio.mock import MockTextToAudioProvider

    class FakeMMAudioProvider(MockTextToAudioProvider):
        raw_extension = ".wav"

    import panorama_foa.audio.mmaudio as mmaudio_module

    monkeypatch.setattr(
        mmaudio_module,
        "MMAudioTextToAudioProvider",
        lambda *, settings: FakeMMAudioProvider(),
    )

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
            "mmaudio",
            "--duration",
            "1.0",
        ],
    )

    assert result.exit_code == 0, result.output
    assert (output / "scene_foa.wav").exists()
    assert (output / "raw_audio" / "global_ambience.wav").exists()


def test_mock_cli_rejects_schema_invalid_duration_before_artifacts(
    tmp_path,
    panorama_fixture,
    fixtures_dir,
):
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
            "0.1",
        ],
    )

    assert result.exit_code != 0
    assert not (output / "scene_plan.json").exists()
    assert not (output / "scene_foa.wav").exists()


def test_mock_cli_rejects_duration_over_schema_max(
    tmp_path,
    panorama_fixture,
    fixtures_dir,
):
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
            "31",
        ],
    )

    assert result.exit_code != 0
    assert not (output / "scene_plan.json").exists()
    assert not (output / "scene_foa.wav").exists()


def test_pipeline_decodes_non_wav_raw_audio(monkeypatch, tmp_path, panorama_fixture, fixtures_dir):
    from panorama_foa.audio.mock import MockTextToAudioProvider
    from panorama_foa.pipeline import build_mock_manual_pipeline

    class Mp3NamedMockProvider(MockTextToAudioProvider):
        raw_extension = ".mp3"

        def generate(self, *, prompt, duration_seconds, loop, output_path):
            _ = (prompt, duration_seconds, loop)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(b"fake mp3 bytes")
            return output_path

    pipeline = build_mock_manual_pipeline(fixtures_dir / "manual_plan.json")
    pipeline.audio_provider = Mp3NamedMockProvider()
    decoded_calls = []

    def fake_decode(raw_path, output_path):
        decoded_calls.append((raw_path, output_path))
        sf.write(output_path, np.sin(np.linspace(0, 10, 48000)).astype("float32"), 48000)
        return output_path

    monkeypatch.setattr("panorama_foa.pipeline.decode_to_mono_wav", fake_decode)
    pipeline.run(
        panorama_path=panorama_fixture,
        output_dir=tmp_path / "scene",
        duration_seconds=1.0,
    )

    assert decoded_calls
    assert all(call[0].suffix == ".mp3" for call in decoded_calls)
