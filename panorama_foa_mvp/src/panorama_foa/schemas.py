from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class GlobalAmbience(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = "global_ambience"
    label: str
    prompt: str
    gain_db: float = Field(ge=-40.0, le=0.0)
    loop: bool = True
    confidence: float = Field(ge=0.0, le=1.0)


class RegionalSource(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    label: str
    prompt: str
    center_u: float = Field(ge=0.0, le=1.0)
    center_v: float = Field(ge=0.0, le=1.0)
    spread: float = Field(ge=0.0, le=1.0)
    gain_db: float = Field(ge=-40.0, le=0.0)
    loop: bool = True
    confidence: float = Field(ge=0.0, le=1.0)


class ScenePlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0"] = "1.0"
    scene_description: str
    duration_seconds: float = Field(ge=0.5, le=30.0)
    global_ambience: GlobalAmbience
    regional_sources: list[RegionalSource] = Field(default_factory=list)

    @field_validator("regional_sources")
    @classmethod
    def ids_are_unique(cls, sources: list[RegionalSource]) -> list[RegionalSource]:
        ids = [source.id for source in sources]
        if len(ids) != len(set(ids)):
            raise ValueError("regional source ids must be unique")
        return sources
