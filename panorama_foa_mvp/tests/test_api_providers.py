from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from panorama_foa.audio.elevenlabs import ElevenLabsProviderError, ElevenLabsSoundEffectsProvider
from panorama_foa.config import Settings
from panorama_foa.planner.openai_vlm import OpenAIVLMScenePlanner
from panorama_foa.schemas import GlobalAmbience, ScenePlan


class FakeResponse:
    output_parsed = ScenePlan(
        scene_description="mock",
        duration_seconds=1.0,
        global_ambience=GlobalAmbience(
            label="quiet",
            prompt="quiet ambience",
            gain_db=-24.0,
            confidence=1.0,
        ),
        regional_sources=[],
    )


class FakeResponses:
    def __init__(self) -> None:
        self.kwargs = None

    def parse(self, **kwargs):
        self.kwargs = kwargs
        return FakeResponse()


class FakeOpenAIClient:
    def __init__(self) -> None:
        self.responses = FakeResponses()


def test_openai_planner_uses_responses_parse_with_scene_plan(panorama_fixture, tmp_path):
    prompt = tmp_path / "prompt.txt"
    prompt.write_text("Plan a soundscape.")
    client = FakeOpenAIClient()
    planner = OpenAIVLMScenePlanner(model="test-model", prompt_path=prompt, client=client)

    plan = planner.plan(
        panorama_path=panorama_fixture,
        duration_seconds=1.0,
        scene_description=None,
        max_sources=5,
        allow_speech=False,
        allow_music=False,
    )

    assert plan.global_ambience.id == "global_ambience"
    assert client.responses.kwargs["model"] == "test-model"
    assert client.responses.kwargs["text_format"] is ScenePlan
    content = client.responses.kwargs["input"][0]["content"]
    assert content[0]["type"] == "input_text"
    assert content[1]["type"] == "input_image"
    assert content[1]["image_url"].startswith("data:image/jpeg;base64,")


def test_elevenlabs_provider_writes_raw_audio_and_redacts_key(tmp_path):
    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, content=b"mp3-bytes")

    client = httpx.Client(transport=httpx.MockTransport(handler))
    provider = ElevenLabsSoundEffectsProvider(
        api_key="secret-key",
        model_id="model",
        client=client,
        sleep=lambda _: None,
    )
    output = provider.generate(
        prompt="rain",
        duration_seconds=1.0,
        loop=True,
        output_path=tmp_path / "raw.mp3",
    )

    assert output.read_bytes() == b"mp3-bytes"
    body = requests[0].read().decode()
    assert requests[0].headers["xi-api-key"] == "secret-key"
    assert '"model_id":"model"' in body
    assert "mono-compatible" in body
    assert provider.raw_extension == ".mp3"


def test_elevenlabs_provider_retries_429_then_succeeds(tmp_path):
    attempts = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        attempts["count"] += 1
        if attempts["count"] == 1:
            return httpx.Response(429, text="try later")
        return httpx.Response(200, content=b"ok")

    provider = ElevenLabsSoundEffectsProvider(
        api_key="secret-key",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        sleep=lambda _: None,
    )
    provider.generate(prompt="wind", duration_seconds=1.0, loop=True, output_path=tmp_path / "raw.mp3")
    assert attempts["count"] == 2


def test_elevenlabs_error_does_not_leak_api_key(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, text="bad secret-key request")

    provider = ElevenLabsSoundEffectsProvider(
        api_key="secret-key",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        sleep=lambda _: None,
    )
    with pytest.raises(ElevenLabsProviderError) as exc_info:
        provider.generate(prompt="wind", duration_seconds=1.0, loop=True, output_path=tmp_path / "raw.mp3")
    assert "secret-key" not in str(exc_info.value)
    assert "[redacted]" in str(exc_info.value)


def test_settings_from_env_keeps_api_keys_optional_for_mock_tests(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
    settings = Settings.from_env()
    assert settings.openai_api_key == ""
    assert settings.elevenlabs_api_key == ""
