from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

from panorama_foa.audio.mock import MockTextToAudioProvider
from panorama_foa.pipeline import PanoramaToFOAPipeline, build_mock_manual_pipeline
from panorama_foa.planner.manual import ManualPlanPlanner
from panorama_foa.schemas import ScenePlan

app = typer.Typer(no_args_is_help=True)
console = Console()

PLANNER_OPENAI = "openai"
PLANNER_MANUAL = "manual"
SUPPORTED_PLANNERS = {PLANNER_OPENAI, PLANNER_MANUAL}
AUDIO_PROVIDER_ELEVENLABS = "elevenlabs"
AUDIO_PROVIDER_MOCK = "mock"
SUPPORTED_AUDIO_PROVIDERS = {AUDIO_PROVIDER_ELEVENLABS, AUDIO_PROVIDER_MOCK}


@app.command()
def generate(
    panorama: Path = typer.Option(..., exists=True, readable=True),
    output: Path = typer.Option(...),
    duration: float = typer.Option(16.0, min=0.5, max=30.0),
    planner: str = typer.Option("openai"),
    plan: Optional[Path] = typer.Option(None, exists=True, readable=True),
    audio_provider: str = typer.Option("elevenlabs"),
    yaw_offset: float = typer.Option(0.0),
    max_sources: int = typer.Option(5),
    allow_speech: bool = typer.Option(False),
    allow_music: bool = typer.Option(False),
    scene_description: Optional[str] = typer.Option(None),
    force: bool = typer.Option(False),
) -> None:
    pipeline = _build_pipeline(
        planner_name=planner,
        plan_path=plan,
        audio_provider_name=audio_provider,
        yaw_offset=yaw_offset,
        force=force,
    )
    result = pipeline.run(
        panorama_path=panorama,
        output_dir=output,
        duration_seconds=duration,
        scene_description=scene_description,
        max_sources=max_sources,
        allow_speech=allow_speech,
        allow_music=allow_music,
    )
    console.print(str(result.wav_path))


@app.command()
def render(
    plan: Path = typer.Option(..., exists=True, readable=True),
    output: Path = typer.Option(...),
    audio_provider: str = typer.Option("elevenlabs"),
    yaw_offset: float = typer.Option(0.0),
    force: bool = typer.Option(False),
) -> None:
    panorama_placeholder = output / "panorama_analysis.jpg"
    pipeline = _build_pipeline(
        planner_name="manual",
        plan_path=plan,
        audio_provider_name=audio_provider,
        yaw_offset=yaw_offset,
        force=force,
    )
    scene_plan = ScenePlan.model_validate_json(plan.read_text())
    output.mkdir(parents=True, exist_ok=True)
    if not panorama_placeholder.exists():
        from PIL import Image

        Image.new("RGB", (256, 128), (128, 128, 128)).save(panorama_placeholder)
    result = pipeline.run(
        panorama_path=panorama_placeholder,
        output_dir=output,
        duration_seconds=scene_plan.duration_seconds,
    )
    console.print(str(result.wav_path))


@app.command("plan")
def plan_command(
    panorama: Path = typer.Option(..., exists=True, readable=True),
    output: Path = typer.Option(...),
    duration: float = typer.Option(16.0, min=0.5, max=30.0),
    max_sources: int = typer.Option(5),
    allow_speech: bool = typer.Option(False),
    allow_music: bool = typer.Option(False),
    scene_description: Optional[str] = typer.Option(None),
) -> None:
    try:
        from panorama_foa.config import Settings
        from panorama_foa.planner.openai_vlm import OpenAIVLMScenePlanner

        settings = Settings.from_env(require_openai=True)
        planner = OpenAIVLMScenePlanner(
            api_key=settings.openai_api_key,
            model=settings.openai_vision_model,
        )
    except Exception as exc:
        raise typer.BadParameter(f"OpenAI planner is not configured: {exc}") from exc

    plan = planner.plan(
        panorama_path=panorama,
        duration_seconds=duration,
        scene_description=scene_description,
        max_sources=max_sources,
        allow_speech=allow_speech,
        allow_music=allow_music,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(plan.model_dump_json(indent=2) + "\n")
    console.print(str(output))


def _build_pipeline(
    *,
    planner_name: str,
    plan_path: Path | None,
    audio_provider_name: str,
    yaw_offset: float,
    force: bool,
) -> PanoramaToFOAPipeline:
    planner_name = planner_name.strip().lower()
    audio_provider_name = audio_provider_name.strip().lower()

    if planner_name not in SUPPORTED_PLANNERS:
        raise typer.BadParameter(
            f"unknown planner: {planner_name}. "
            f"Expected one of: {', '.join(sorted(SUPPORTED_PLANNERS))}"
        )
    if audio_provider_name not in SUPPORTED_AUDIO_PROVIDERS:
        raise typer.BadParameter(
            f"unknown audio provider: {audio_provider_name}. "
            f"Expected one of: {', '.join(sorted(SUPPORTED_AUDIO_PROVIDERS))}"
        )

    if planner_name == PLANNER_OPENAI:
        try:
            from panorama_foa.config import Settings
            from panorama_foa.planner.openai_vlm import OpenAIVLMScenePlanner

            settings = Settings.from_env(require_openai=True)
            planner = OpenAIVLMScenePlanner(
                api_key=settings.openai_api_key,
                model=settings.openai_vision_model,
            )
        except Exception as exc:
            raise typer.BadParameter(f"OpenAI planner is not configured: {exc}") from exc
    else:
        if plan_path is None:
            raise typer.BadParameter("--plan is required when --planner manual")
        planner = ManualPlanPlanner(plan_path)

    if audio_provider_name == AUDIO_PROVIDER_MOCK:
        provider = MockTextToAudioProvider()
    else:
        try:
            from panorama_foa.audio.elevenlabs import ElevenLabsSoundEffectsProvider
            from panorama_foa.config import Settings

            settings = Settings.from_env()
            provider = ElevenLabsSoundEffectsProvider(settings=settings)
        except Exception as exc:
            raise typer.BadParameter(f"ElevenLabs provider is not configured: {exc}") from exc

    return PanoramaToFOAPipeline(
        planner=planner,
        audio_provider=provider,
        yaw_offset_deg=yaw_offset,
        force=force,
    )


if __name__ == "__main__":
    app()
