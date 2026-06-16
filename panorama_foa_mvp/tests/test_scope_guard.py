from __future__ import annotations

from pathlib import Path


FORBIDDEN = [
    "torch",
    "torchaudio",
    "librosa",
    "sam3",
    "segment-anything",
    "MMAudio",
    "Marble",
    "HunyuanWorld",
    "3DGS",
    "HRTF",
    "HRIR",
    "Three.js",
    "WebXR",
]


def test_forbidden_dependencies_not_declared():
    pyproject = Path("pyproject.toml").read_text()
    for token in FORBIDDEN:
        assert token not in pyproject


def test_source_does_not_import_or_implement_forbidden_scope():
    root = Path("src")
    for path in root.rglob("*"):
        if path.is_file():
            text = path.read_text(errors="ignore")
            for token in FORBIDDEN:
                assert token not in text, f"{token} found in {path}"
