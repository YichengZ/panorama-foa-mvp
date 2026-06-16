from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

from panorama_foa.factory import (
    AUDIO_PROVIDER_MMAUDIO,
    PLANNER_OPENAI,
    FactoryConfigurationError,
    build_pipeline,
    build_planner,
)
from panorama_foa.schemas import ScenePlan

app = typer.Typer(no_args_is_help=True)
console = Console()


@app.command()
def generate(
    panorama: Path = typer.Option(..., exists=True, readable=True),
    output: Path = typer.Option(...),
    duration: float = typer.Option(16.0, min=0.5, max=30.0),
    planner: str = typer.Option("openai"),
    plan: Optional[Path] = typer.Option(None, exists=True, readable=True),
    audio_provider: str = typer.Option(AUDIO_PROVIDER_MMAUDIO),
    yaw_offset: float = typer.Option(0.0),
    max_sources: int = typer.Option(5),
    allow_speech: bool = typer.Option(False),
    allow_music: bool = typer.Option(False),
    scene_description: Optional[str] = typer.Option(None),
    force: bool = typer.Option(False),
) -> None:
    pipeline = _build_cli_pipeline(
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
    audio_provider: str = typer.Option(AUDIO_PROVIDER_MMAUDIO),
    yaw_offset: float = typer.Option(0.0),
    force: bool = typer.Option(False),
) -> None:
    panorama_placeholder = output / "panorama_analysis.jpg"
    pipeline = _build_cli_pipeline(
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
        planner = build_planner(planner_name=PLANNER_OPENAI, plan_path=None)
    except FactoryConfigurationError as exc:
        raise typer.BadParameter(str(exc)) from exc

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


def _build_cli_pipeline(
    *,
    planner_name: str,
    plan_path: Path | None,
    audio_provider_name: str,
    yaw_offset: float,
    force: bool,
):
    try:
        return build_pipeline(
            planner_name=planner_name,
            plan_path=plan_path,
            audio_provider_name=audio_provider_name,
            yaw_offset=yaw_offset,
            force=force,
        )
    except FactoryConfigurationError as exc:
        raise typer.BadParameter(str(exc)) from exc


if __name__ == "__main__":
    app()
