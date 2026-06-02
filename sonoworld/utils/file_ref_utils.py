from __future__ import annotations

from pathlib import Path
from typing import Any, Optional


def path_from_ref(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        nested = value.get("path")
        if isinstance(nested, dict):
            return path_from_ref(nested)
        if nested is not None:
            return str(nested)
    return None
