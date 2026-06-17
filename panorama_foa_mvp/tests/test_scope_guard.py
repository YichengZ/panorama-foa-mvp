from __future__ import annotations

from pathlib import Path


FORBIDDEN = [
    "torchaudio",
    "librosa",
    "stages.audio_generation",
    "sam3",
    "segment-anything",
    "Marble",
    "HunyuanWorld",
    "3DGS",
    "HRTF",
    "HRIR",
    "Three.js",
    "WebXR",
]

LOCAL_BACKEND_ALLOWED_TOKENS = {
    "torch": {
        Path("src") / "panorama_foa" / "audio" / "backends" / "mmaudio_diffusion.py",
    },
}


def test_forbidden_dependencies_not_declared():
    pyproject = Path("pyproject.toml").read_text()
    for token in FORBIDDEN:
        assert token not in pyproject


def test_source_does_not_import_or_implement_forbidden_scope():
    root = Path("src")
    for path in root.rglob("*"):
        if path.is_file() and path.suffix == ".py":
            text = path.read_text(errors="ignore")
            for token in FORBIDDEN:
                assert token not in text, f"{token} found in {path}"
            for token, allowed_paths in LOCAL_BACKEND_ALLOWED_TOKENS.items():
                if token in text:
                    assert path in allowed_paths, f"{token} found outside the local backend: {path}"


def test_legacy_sonoworld_pipeline_files_are_not_present():
    repo_root = Path(__file__).resolve().parents[2]
    forbidden_paths = [
        repo_root / "sonoworld",
        repo_root / "generate.py",
        repo_root / "configs" / "default.yaml",
        repo_root / "test-inputs" / "fall.jpg",
    ]
    for path in forbidden_paths:
        assert not path.exists(), f"legacy out-of-scope path still exists: {path}"
