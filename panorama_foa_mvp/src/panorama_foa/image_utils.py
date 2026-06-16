from __future__ import annotations

import base64
import io
from pathlib import Path

from PIL import Image, UnidentifiedImageError


PANORAMA_TARGET_ASPECT = 2.0
PANORAMA_ASPECT_TOLERANCE = 0.05
ANALYSIS_MAX_WIDTH = 4096
ANALYSIS_JPEG_QUALITY = 90


class PanoramaImageError(ValueError):
    """Raised when the input image is not a valid 2:1 panorama."""


def validate_panorama_image(path: Path) -> tuple[int, int]:
    """Validate that path is a readable 2:1 equirectangular panorama."""

    try:
        with Image.open(path) as image:
            width, height = image.size
            image.verify()
    except FileNotFoundError:
        raise
    except (UnidentifiedImageError, OSError) as exc:
        raise PanoramaImageError(f"Input is not a readable image: {path}") from exc

    if width <= 0 or height <= 0:
        raise PanoramaImageError(f"Input image has invalid dimensions: {width}x{height}")

    aspect = width / height
    if abs(aspect - PANORAMA_TARGET_ASPECT) > PANORAMA_ASPECT_TOLERANCE:
        raise PanoramaImageError(
            "Input must be a complete 2:1 equirectangular panorama, not a normal "
            f"perspective image. Got {width}x{height} (aspect {aspect:.3f})."
        )

    return width, height


def create_analysis_image(
    panorama_path: Path,
    output_path: Path,
    *,
    max_width: int = ANALYSIS_MAX_WIDTH,
    quality: int = ANALYSIS_JPEG_QUALITY,
) -> Path:
    """Create a JPEG analysis copy for VLM planning.

    The source panorama is preserved. The analysis image is RGB JPEG with
    width capped at max_width and aspect ratio preserved.
    """

    validate_panorama_image(panorama_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with Image.open(panorama_path) as image:
        image = _rgb_image(image)
        if image.width > max_width:
            target_height = round(image.height * max_width / image.width)
            image = image.resize((max_width, target_height), Image.Resampling.LANCZOS)
        image.save(output_path, format="JPEG", quality=quality, optimize=True)

    validate_panorama_image(output_path)
    return output_path


def image_to_data_url(path: Path) -> str:
    """Encode an image file as a base64 data URL for Responses API input."""

    mime_type = "image/jpeg" if path.suffix.lower() in {".jpg", ".jpeg"} else "image/png"
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{data}"


def analysis_image_data_url(panorama_path: Path) -> str:
    """Create an in-memory max-width JPEG analysis image and return a data URL."""

    validate_panorama_image(panorama_path)
    with Image.open(panorama_path) as image:
        image = _rgb_image(image)
        if image.width > ANALYSIS_MAX_WIDTH:
            target_height = round(image.height * ANALYSIS_MAX_WIDTH / image.width)
            image = image.resize(
                (ANALYSIS_MAX_WIDTH, target_height),
                Image.Resampling.LANCZOS,
            )
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=ANALYSIS_JPEG_QUALITY, optimize=True)

    data = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{data}"


def _rgb_image(image: Image.Image) -> Image.Image:
    if image.mode == "RGB":
        return image.copy()
    return image.convert("RGB")
