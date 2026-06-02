from __future__ import annotations

from typing import Any


VALID_AUDIO_SOURCE_TYPES = {"area", "point", "background"}


def normalize_audio_source_type(value: Any, grounding_label: str) -> str:
    if isinstance(value, str) and value.strip().lower() in VALID_AUDIO_SOURCE_TYPES:
        source_type = value.strip().lower()
    else:
        source_type = "background" if grounding_label.lower() == "global" else "area"

    if grounding_label.lower() == "global":
        return "background"
    if source_type == "background":
        return "area"
    return source_type


def normalize_peak_db(value: Any, source_type: str) -> float:
    default = -28.0 if source_type == "background" else -18.0
    try:
        peak_db = float(value)
    except (TypeError, ValueError):
        peak_db = default

    if peak_db > -6.0:
        peak_db = -6.0
    if source_type == "background" and peak_db > -20.0:
        peak_db = -28.0
    return peak_db
