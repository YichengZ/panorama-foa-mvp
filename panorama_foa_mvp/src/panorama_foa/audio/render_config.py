from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AudioRenderConfig:
    """Output loudness and loop QA settings for FOA renders."""

    loop_post_roll_seconds: float = 2.0
    loopcheck_repeat_count: int = 2
    target_rms_dbfs: float = -30.0
    peak_ceiling_dbfs: float = -3.0
