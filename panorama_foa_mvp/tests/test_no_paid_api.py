from __future__ import annotations

from typer.testing import CliRunner

from panorama_foa.cli import app


def test_mock_end_to_end_does_not_require_api_keys(
    monkeypatch,
    tmp_path,
    panorama_fixture,
    fixtures_dir,
):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)

    def fail_network(*args, **kwargs):
        raise AssertionError("network access is forbidden in mock tests")

    monkeypatch.setattr("httpx.Client.request", fail_network, raising=False)
    monkeypatch.setattr("httpx.request", fail_network, raising=False)

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
            "mock",
        ],
    )

    assert result.exit_code == 0, result.output
