from __future__ import annotations

from pathlib import Path

from panorama_foa.schemas import ScenePlan


class ManualPlanPlanner:
    def __init__(self, plan_path: Path) -> None:
        self.plan_path = Path(plan_path)

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
        _ = (panorama_path, scene_description, max_sources, allow_speech, allow_music)
        plan = ScenePlan.model_validate_json(self.plan_path.read_text())
        return plan.model_copy(update={"duration_seconds": duration_seconds})

