from __future__ import annotations

from pathlib import Path
from typing import Any

from panorama_foa.config import (
    DEFAULT_OPENAI_VISION_MODEL,
    DEFAULT_PLANNER_PROMPT_PATH,
    OpenAIPlannerConfig,
    openai_config_from_env,
)
from panorama_foa.image_utils import analysis_image_data_url
from panorama_foa.schemas import ScenePlan


class OpenAIPlannerError(RuntimeError):
    """Raised when the OpenAI VLM planner cannot produce a valid plan."""


class OpenAIVLMPlanner:
    """Plan a static soundscape from a panorama using OpenAI Responses API."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str | None = None,
        prompt_path: Path | None = None,
        client: Any | None = None,
    ) -> None:
        if client is None and api_key is None:
            config = openai_config_from_env(prompt_path=prompt_path)
            api_key = config.api_key
            model = model or config.model
            prompt_path = config.prompt_path

        self.model = model or DEFAULT_OPENAI_VISION_MODEL
        self.prompt_path = prompt_path or DEFAULT_PLANNER_PROMPT_PATH
        self.client = client or self._create_client(api_key)

    @classmethod
    def from_config(
        cls,
        config: OpenAIPlannerConfig,
        *,
        client: Any | None = None,
    ) -> "OpenAIVLMPlanner":
        return cls(
            api_key=config.api_key,
            model=config.model,
            prompt_path=config.prompt_path,
            client=client,
        )

    def plan(
        self,
        *,
        panorama_path: Path,
        duration_seconds: float,
        scene_description: str | None,
        max_sources: int,
        allow_speech: bool,
        allow_music: bool,
    ) -> ScenePlan:
        prompt = self._build_prompt(
            duration_seconds=duration_seconds,
            scene_description=scene_description,
            max_sources=max_sources,
            allow_speech=allow_speech,
            allow_music=allow_music,
        )
        image_url = analysis_image_data_url(panorama_path)

        response = self.client.responses.parse(
            model=self.model,
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": prompt},
                        {"type": "input_image", "image_url": image_url},
                    ],
                }
            ],
            text_format=ScenePlan,
        )

        parsed = getattr(response, "output_parsed", None)
        if parsed is None:
            raise OpenAIPlannerError(
                "OpenAI Responses API returned no parsed ScenePlan."
            )
        if isinstance(parsed, ScenePlan):
            return parsed
        return ScenePlan.model_validate(parsed)

    def _build_prompt(
        self,
        *,
        duration_seconds: float,
        scene_description: str | None,
        max_sources: int,
        allow_speech: bool,
        allow_music: bool,
    ) -> str:
        base_prompt = self.prompt_path.read_text(encoding="utf-8").strip()
        if not base_prompt:
            raise OpenAIPlannerError(f"Planner prompt file is empty: {self.prompt_path}")

        clamped_max_sources = max(0, min(5, int(max_sources)))
        description = (scene_description or "").strip() or "None provided."
        return (
            f"{base_prompt}\n\n"
            "User request parameters:\n"
            f"- requested duration_seconds: {duration_seconds:.3f}\n"
            f"- optional scene description: {description}\n"
            f"- allow intelligible speech: {str(bool(allow_speech)).lower()}\n"
            f"- allow music: {str(bool(allow_music)).lower()}\n"
            f"- maximum regional sources: {clamped_max_sources}\n\n"
            "Respect the requested duration in the returned ScenePlan. "
            "If allow speech or allow music is false, avoid that content even if it "
            "might be weakly implied by the image."
        )

    def _create_client(self, api_key: str | None) -> Any:
        if not api_key:
            raise OpenAIPlannerError(
                "OPENAI_API_KEY is required for the OpenAI VLM planner."
            )
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise OpenAIPlannerError(
                "The OpenAI Python package is required for the OpenAI VLM planner."
            ) from exc

        return OpenAI(api_key=api_key)


OpenAIVLMScenePlanner = OpenAIVLMPlanner
