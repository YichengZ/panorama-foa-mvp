from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image


@pytest.fixture
def fixtures_dir() -> Path:
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def panorama_fixture(fixtures_dir: Path) -> Path:
    path = fixtures_dir / "panorama_2x1.jpg"
    if not path.exists():
        image = Image.new("RGB", (256, 128))
        pixels = image.load()
        for y in range(128):
            for x in range(256):
                pixels[x, y] = (x % 256, y * 2 % 256, 128)
        image.save(path, quality=90)
    return path
