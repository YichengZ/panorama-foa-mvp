from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx

from panorama_foa.config import (
    DEFAULT_ELEVENLABS_MAX_RETRIES,
    DEFAULT_ELEVENLABS_PROMPT_INFLUENCE,
    DEFAULT_ELEVENLABS_SOUND_MODEL,
    DEFAULT_ELEVENLABS_TIMEOUT_SECONDS,
    ElevenLabsConfig,
    Settings,
    elevenlabs_config_from_env,
    redact_secret,
)


ELEVENLABS_SOUND_GENERATION_URL = "https://api.elevenlabs.io/v1/sound-generation"
MIN_DURATION_SECONDS = 0.5
MAX_DURATION_SECONDS = 30.0
PROMPT_SUFFIX_PARTS = (
    "isolated environmental sound layer",
    "natural clean recording",
    "mono-compatible",
    "no added music",
    "no intelligible speech",
    "minimal unrelated background sounds",
)


class ElevenLabsProviderError(RuntimeError):
    """Raised when ElevenLabs sound generation fails."""


class ElevenLabsSoundEffectsProvider:
    """Generate raw sound layers with ElevenLabs Sound Effects API."""

    raw_extension = ".mp3"
    sample_rate = None

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model_id: str | None = None,
        prompt_influence: float = DEFAULT_ELEVENLABS_PROMPT_INFLUENCE,
        timeout_seconds: float = DEFAULT_ELEVENLABS_TIMEOUT_SECONDS,
        max_retries: int = DEFAULT_ELEVENLABS_MAX_RETRIES,
        client: httpx.Client | None = None,
        sleep: Callable[[float], None] = time.sleep,
        settings: Settings | None = None,
    ) -> None:
        if settings is not None:
            config = settings.elevenlabs_config()
            api_key = config.api_key
            model_id = model_id or config.model_id
            prompt_influence = config.prompt_influence
            timeout_seconds = config.timeout_seconds
            max_retries = config.max_retries
        if client is None and api_key is None:
            config = elevenlabs_config_from_env()
            api_key = config.api_key
            model_id = model_id or config.model_id
            prompt_influence = config.prompt_influence
            timeout_seconds = config.timeout_seconds
            max_retries = config.max_retries

        if not api_key:
            raise ElevenLabsProviderError(
                "ELEVENLABS_API_KEY is required for the ElevenLabs audio provider."
            )

        self.api_key = api_key
        self.model_id = model_id or DEFAULT_ELEVENLABS_SOUND_MODEL
        self.prompt_influence = float(prompt_influence)
        self.timeout_seconds = max(float(timeout_seconds), DEFAULT_ELEVENLABS_TIMEOUT_SECONDS)
        self.max_retries = max(0, int(max_retries))
        self.client = client or httpx.Client(timeout=self.timeout_seconds)
        self._owns_client = client is None
        self.sleep = sleep

    @classmethod
    def from_config(
        cls,
        config: ElevenLabsConfig,
        *,
        client: httpx.Client | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> "ElevenLabsSoundEffectsProvider":
        return cls(
            api_key=config.api_key,
            model_id=config.model_id,
            prompt_influence=config.prompt_influence,
            timeout_seconds=config.timeout_seconds,
            max_retries=config.max_retries,
            client=client,
            sleep=sleep,
        )

    def generate(
        self,
        *,
        prompt: str,
        duration_seconds: float,
        loop: bool,
        output_path: Path,
    ) -> Path:
        duration_seconds = _validate_duration(duration_seconds)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "text": enhance_sound_prompt(prompt),
            "loop": bool(loop),
            "duration_seconds": duration_seconds,
            "prompt_influence": self.prompt_influence,
            "model_id": self.model_id,
        }
        headers = {
            "xi-api-key": self.api_key,
            "Content-Type": "application/json",
        }

        response = self._request_with_retries(headers=headers, payload=payload)
        output_path.write_bytes(response.content)
        return output_path

    def close(self) -> None:
        if self._owns_client:
            self.client.close()

    def __enter__(self) -> "ElevenLabsSoundEffectsProvider":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def _request_with_retries(
        self,
        *,
        headers: dict[str, str],
        payload: dict[str, Any],
    ) -> httpx.Response:
        last_error: Exception | None = None
        max_attempts = self.max_retries + 1
        for attempt in range(1, max_attempts + 1):
            try:
                response = self.client.request(
                    "POST",
                    ELEVENLABS_SOUND_GENERATION_URL,
                    headers=headers,
                    json=payload,
                    timeout=self.timeout_seconds,
                )
            except httpx.HTTPError as exc:
                last_error = exc
                if attempt >= max_attempts:
                    break
                self.sleep(_backoff_seconds(attempt))
                continue

            if response.status_code < 400:
                return response

            if response.status_code == 429 or response.status_code >= 500:
                if attempt >= max_attempts:
                    raise _response_error(response, self.api_key)
                self.sleep(_backoff_seconds(attempt))
                continue

            raise _response_error(response, self.api_key)

        message = "ElevenLabs request failed after retries."
        if last_error is not None:
            message = f"{message} {redact_secret(str(last_error), self.api_key)}"
        raise ElevenLabsProviderError(message)


def enhance_sound_prompt(prompt: str) -> str:
    text = " ".join(prompt.strip().split())
    if not text:
        text = "quiet environmental ambience"

    lower_text = text.lower()
    additions = [part for part in PROMPT_SUFFIX_PARTS if part not in lower_text]
    if not additions:
        return text
    return f"{text}, {', '.join(additions)}"


def _validate_duration(duration_seconds: float) -> float:
    duration = float(duration_seconds)
    if not MIN_DURATION_SECONDS <= duration <= MAX_DURATION_SECONDS:
        raise ElevenLabsProviderError(
            "ElevenLabs sound duration_seconds must be between "
            f"{MIN_DURATION_SECONDS} and {MAX_DURATION_SECONDS}."
        )
    return duration


def _backoff_seconds(attempt: int) -> float:
    return min(2.0 ** (attempt - 1), 8.0)


def _response_error(response: httpx.Response, api_key: str) -> ElevenLabsProviderError:
    body = redact_secret(response.text, api_key)
    return ElevenLabsProviderError(
        f"ElevenLabs request failed with HTTP {response.status_code}: {body}"
    )
