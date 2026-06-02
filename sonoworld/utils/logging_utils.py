from __future__ import annotations

import logging
import os
from typing import Optional, Union


_DEFAULT_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
_DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
_LOG_LEVEL_ENV = "SONOWORLD_LOG_LEVEL"


def _resolve_level(level: Optional[Union[int, str]]) -> Union[int, str]:
    if level is not None:
        return level

    env_level = os.getenv(_LOG_LEVEL_ENV)
    if env_level:
        return env_level.upper()

    return logging.INFO


def getLogger(name: Optional[str] = None, level: Optional[Union[int, str]] = None) -> logging.Logger:
    """Return a configured stdlib logger.

    This mirrors ``logging.getLogger`` while giving Sonoworld modules a shared
    default formatter when the application has not configured logging yet.
    """
    resolved_level = _resolve_level(level)
    logging.basicConfig(
        level=resolved_level,
        format=_DEFAULT_FORMAT,
        datefmt=_DEFAULT_DATE_FORMAT,
    )

    logger = logging.getLogger(name)
    if level is not None:
        logger.setLevel(resolved_level)

    return logger
