from __future__ import annotations

from pathlib import Path

from panorama_foa.audio.mock import MockTextToAudioProvider
from panorama_foa.audio.provider_base import TextToAudioProvider
from panorama_foa.config import Settings
from panorama_foa.pipeline import PanoramaToFOAPipeline
from panorama_foa.planner.base import ScenePlanner
from panorama_foa.planner.manual import ManualPlanPlanner

PLANNER_OPENAI = "openai"
PLANNER_MANUAL = "manual"
SUPPORTED_PLANNERS = frozenset({PLANNER_OPENAI, PLANNER_MANUAL})

AUDIO_PROVIDER_ELEVENLABS = "elevenlabs"
AUDIO_PROVIDER_MMAUDIO = "mmaudio"
AUDIO_PROVIDER_MOCK = "mock"
SUPPORTED_AUDIO_PROVIDERS = frozenset(
    {
        AUDIO_PROVIDER_ELEVENLABS,
        AUDIO_PROVIDER_MMAUDIO,
        AUDIO_PROVIDER_MOCK,
    }
)


class FactoryConfigurationError(RuntimeError):
    """Raised when a selected provider or planner cannot be configured."""


def build_pipeline(
    *,
    planner_name: str,
    plan_path: Path | None,
    audio_provider_name: str,
    yaw_offset: float,
    force: bool,
) -> PanoramaToFOAPipeline:
    planner = build_planner(planner_name=planner_name, plan_path=plan_path)
    provider = build_audio_provider(audio_provider_name=audio_provider_name)
    return PanoramaToFOAPipeline(
        planner=planner,
        audio_provider=provider,
        yaw_offset_deg=yaw_offset,
        force=force,
    )


def build_planner(*, planner_name: str, plan_path: Path | None) -> ScenePlanner:
    planner_name = normalize_choice(planner_name)
    if planner_name not in SUPPORTED_PLANNERS:
        raise FactoryConfigurationError(
            f"unknown planner: {planner_name}. "
            f"Expected one of: {', '.join(sorted(SUPPORTED_PLANNERS))}"
        )

    if planner_name == PLANNER_MANUAL:
        if plan_path is None:
            raise FactoryConfigurationError("--plan is required when --planner manual")
        return ManualPlanPlanner(plan_path)

    try:
        from panorama_foa.planner.openai_vlm import OpenAIVLMScenePlanner

        settings = Settings.from_env(require_openai=True)
        return OpenAIVLMScenePlanner(
            api_key=settings.openai_api_key,
            model=settings.openai_vision_model,
        )
    except Exception as exc:
        raise FactoryConfigurationError(f"OpenAI planner is not configured: {exc}") from exc


def build_audio_provider(*, audio_provider_name: str) -> TextToAudioProvider:
    audio_provider_name = normalize_choice(audio_provider_name)
    if audio_provider_name not in SUPPORTED_AUDIO_PROVIDERS:
        raise FactoryConfigurationError(
            f"unknown audio provider: {audio_provider_name}. "
            f"Expected one of: {', '.join(sorted(SUPPORTED_AUDIO_PROVIDERS))}"
        )

    if audio_provider_name == AUDIO_PROVIDER_MOCK:
        return MockTextToAudioProvider()

    if audio_provider_name == AUDIO_PROVIDER_MMAUDIO:
        try:
            from panorama_foa.audio.mmaudio import MMAudioTextToAudioProvider

            settings = Settings.from_env()
            return MMAudioTextToAudioProvider(settings=settings)
        except Exception as exc:
            raise FactoryConfigurationError(f"MMAudio provider is not configured: {exc}") from exc

    try:
        from panorama_foa.audio.elevenlabs import ElevenLabsSoundEffectsProvider

        settings = Settings.from_env()
        return ElevenLabsSoundEffectsProvider(settings=settings)
    except Exception as exc:
        raise FactoryConfigurationError(f"ElevenLabs provider is not configured: {exc}") from exc


def normalize_choice(value: str) -> str:
    return value.strip().lower()
