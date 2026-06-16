from __future__ import annotations

import pytest
from PIL import Image

from panorama_foa.image_utils import (
    PanoramaImageError,
    create_analysis_image,
    validate_panorama_image,
)


def test_validate_panorama_image_accepts_two_to_one(tmp_path):
    panorama = tmp_path / "panorama.jpg"
    Image.new("RGB", (400, 200), (10, 20, 30)).save(panorama)

    assert validate_panorama_image(panorama) == (400, 200)


def test_validate_panorama_image_rejects_non_panorama(tmp_path):
    perspective = tmp_path / "perspective.jpg"
    Image.new("RGB", (300, 200), (10, 20, 30)).save(perspective)

    with pytest.raises(PanoramaImageError, match="2:1 equirectangular panorama"):
        validate_panorama_image(perspective)


def test_create_analysis_image_preserves_two_to_one_aspect(tmp_path):
    panorama = tmp_path / "large_panorama.jpg"
    analysis = tmp_path / "analysis.jpg"
    Image.new("RGB", (800, 400), (10, 20, 30)).save(panorama)

    create_analysis_image(panorama, analysis, max_width=200)

    assert validate_panorama_image(analysis) == (200, 100)
