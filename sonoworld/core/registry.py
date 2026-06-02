# sonoworld/core/registry.py

from __future__ import annotations

from typing import Dict, Tuple, Type, Optional, List

from sonoworld.core.stage import Stage


StageKey = Tuple[str, str]


class StageRegistry:
    """Registry for stage backend implementations."""

    def __init__(self) -> None:
        self._stages: Dict[StageKey, Type[Stage]] = {}

    def register(
        self,
        stage_name: str,
        backend_name: str,
        cls: Type[Stage],
        *,
        overwrite: bool = False,
    ) -> None:
        key = (stage_name, backend_name)

        if key in self._stages and not overwrite:
            existing = self._stages[key]
            raise KeyError(
                f"Stage backend already registered: {stage_name}:{backend_name} "
                f"by {existing.__module__}.{existing.__name__}"
            )

        self._stages[key] = cls

    def get(self, stage_name: str, backend_name: str) -> Type[Stage]:
        key = (stage_name, backend_name)

        if key not in self._stages:
            available = self.available(stage_name)
            if available:
                available_msg = ", ".join(available)
                raise KeyError(
                    f"Unknown backend '{backend_name}' for stage '{stage_name}'. "
                    f"Available backends: {available_msg}"
                )

            raise KeyError(
                f"No backends registered for stage '{stage_name}'. "
                f"Make sure the stage implementation module is imported."
            )

        return self._stages[key]

    def create(self, stage_name: str, backend_name: str, **kwargs) -> Stage:
        cls = self.get(stage_name, backend_name)
        return cls(**kwargs)

    def available(self, stage_name: Optional[str] = None) -> List[str]:
        if stage_name is None:
            return sorted([
                f"{stage}:{backend}"
                for stage, backend in self._stages.keys()
            ])

        return sorted([
            backend
            for stage, backend in self._stages.keys()
            if stage == stage_name
        ])

    def has(self, stage_name: str, backend_name: str) -> bool:
        return (stage_name, backend_name) in self._stages


STAGE_REGISTRY = StageRegistry()


def register_stage(stage_name: str, backend_name: str, *, overwrite: bool = False):
    """Decorator used by stage implementations to register themselves."""

    def decorator(cls: Type[Stage]) -> Type[Stage]:
        STAGE_REGISTRY.register(
            stage_name=stage_name,
            backend_name=backend_name,
            cls=cls,
            overwrite=overwrite,
        )

        cls.name = stage_name
        cls.backend = backend_name
        return cls

    return decorator


def build_stage(stage_name: str, backend_name: str, **kwargs) -> Stage:
    """Create a registered stage backend."""
    return STAGE_REGISTRY.create(stage_name, backend_name, **kwargs)