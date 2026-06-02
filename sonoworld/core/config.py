# sonoworld/core/config.py

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional
import json

import yaml


ConfigDict = Dict[str, Any]


def load_config(path: str | Path) -> ConfigDict:
    """Load a YAML or JSON config file."""
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Config file does not exist: {path}")

    if path.suffix.lower() in {".yaml", ".yml"}:
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

    elif path.suffix.lower() == ".json":
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)

    else:
        raise ValueError(f"Unsupported config format: {path.suffix}")

    return data or {}


def get_runtime_config(config: ConfigDict) -> ConfigDict:
    """Return the runtime config block."""
    return config.get("runtime", {})


def get_stage_config(config: ConfigDict, stage_name: str) -> ConfigDict:
    """Return the config block for one stage."""
    return config.get("stages", {}).get(stage_name, {})


def is_stage_enabled(config: ConfigDict, stage_name: str, default: bool = True) -> bool:
    """Return whether a stage is enabled."""
    return bool(get_stage_config(config, stage_name).get("enabled", default))


def get_stage_class_path(config: ConfigDict, stage_name: str) -> Optional[str]:
    """Return the class path for one stage."""
    return get_stage_config(config, stage_name).get("class_path")


def require_stage_class_path(config: ConfigDict, stage_name: str) -> str:
    """Return the class path for a stage or raise a clear error."""
    class_path = get_stage_class_path(config, stage_name)

    if class_path is None:
        raise ValueError(f"Missing class_path for stage: {stage_name}")

    return class_path


def get_device(config: ConfigDict, default: str = "cuda") -> str:
    """Return the runtime device."""
    return str(get_runtime_config(config).get("device", default))


def get_low_vram(config: ConfigDict, default: bool = False) -> bool:
    """Return whether low VRAM mode is enabled."""
    return bool(get_runtime_config(config).get("low_vram", default))