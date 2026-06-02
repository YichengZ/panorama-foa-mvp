# sonoworld/core/factory.py

from __future__ import annotations

import importlib
from typing import Any, Type

from sonoworld.core.stage import Stage


def import_class(class_path: str) -> Type[Any]:
    """Import a class from a fully qualified class path."""
    if "." not in class_path:
        raise ValueError(
            f"Invalid class path: {class_path}. "
            "Expected format: package.module.ClassName"
        )

    module_name, class_name = class_path.rsplit(".", 1)

    module = importlib.import_module(module_name)

    if not hasattr(module, class_name):
        raise AttributeError(
            f"Module '{module_name}' does not define class '{class_name}'."
        )

    cls = getattr(module, class_name)

    if not isinstance(cls, type):
        raise TypeError(
            f"Imported object is not a class: {class_path}"
        )

    return cls


def build_stage_from_config(stage_name: str, stage_cfg: dict) -> Stage:
    """Build a stage instance from a config block."""
    class_path = stage_cfg.get("class_path")

    if class_path is None:
        raise ValueError(
            f"Missing 'class_path' for stage '{stage_name}'."
        )

    cls = import_class(class_path)

    if not issubclass(cls, Stage):
        raise TypeError(
            f"Configured class for stage '{stage_name}' must inherit from Stage: "
            f"{class_path}"
        )

    init_kwargs = stage_cfg.get("init", {})

    if init_kwargs is None:
        init_kwargs = {}

    if not isinstance(init_kwargs, dict):
        raise TypeError(
            f"'init' for stage '{stage_name}' must be a dictionary."
        )

    stage = cls(**init_kwargs)

    if not getattr(stage, "name", None):
        stage.name = stage_name

    if not getattr(stage, "backend", None):
        stage.backend = class_path

    return stage