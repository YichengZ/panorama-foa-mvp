from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Optional

from sonoworld.core.config import (
    load_config,
    get_device,
    get_stage_config,
    is_stage_enabled,
    get_stage_class_path,
)
from sonoworld.core.factory import build_stage_from_config
from sonoworld.core.stage import StageContext
from sonoworld.core.manifest import SceneManifest
from sonoworld.core.paths import ScenePaths
from sonoworld.schemas.common import FileRef


GENERATION_ORDER = [
    "outpainting",
    "understanding",
    "visual_scene",
    "segmentation",
    "audio_generation",
    "spatial_config",
]

def run_generation(
    scene_root: Path,
    config: dict,
    input_image: Optional[Path] = None,
    resume: bool = False,
    force: bool = False,
) -> None:
    scene_root = Path(scene_root)
    input_image = Path(input_image) if input_image is not None else None
    paths = ScenePaths(scene_root)
    paths.ensure_base_dirs()

    if paths.manifest.exists():
        manifest = SceneManifest.load(paths.manifest)
    else:
        manifest = SceneManifest.create(scene_root)
        manifest.save(paths.manifest)

    if input_image is not None and (not paths.input_image.exists() or not input_image.samefile(paths.input_image)):
        from PIL import Image
        Image.open(input_image).save(paths.input_image)

    if not paths.input_image.exists():
        raise FileNotFoundError(f"Input image not found at {paths.input_image}. Please provide an input image or place it at the expected location.")
    
    manifest.metadata["input_image"] = FileRef.from_path(
        paths.input_image,
        scene_root=scene_root,
        role="input_image",
    )
    manifest.save(paths.manifest)

    ctx = StageContext(
        scene_root=scene_root,
        config=config,
        device=get_device(config),
        force=force,
    )

    for stage_name in GENERATION_ORDER:
        stage_cfg = get_stage_config(config, stage_name)

        if not is_stage_enabled(config, stage_name, default=True):
            manifest.mark_skipped(
                stage_name,
                backend=None,
                message="Stage is disabled.",
            )
            manifest.save(paths.manifest)
            print(f"[{stage_name}] skipped, disabled.")
            continue

        class_path = get_stage_class_path(config, stage_name)
        if class_path is None:
            raise ValueError(f"Missing class_path for enabled stage: {stage_name}")

        record = manifest.stages.get(stage_name)

        if record is not None and record.status == "done" and resume and not force:
            print(f"[{stage_name}] skipped, already done.")
            continue

        if record is not None and record.status == "waiting" and resume:
            print(f"[{stage_name}] resuming from waiting state.")

        stage = build_stage_from_config(stage_name, stage_cfg)

        try:
            manifest.mark_running(stage_name, backend=class_path)
            manifest.save(paths.manifest)

            result = stage.run(ctx)

            if result.status == "waiting":
                manifest.mark_waiting(
                    stage_name,
                    backend=class_path,
                    message=result.message,
                    inputs=result.inputs,
                    outputs=result.outputs,
                    metadata=result.metadata,
                )
                manifest.save(paths.manifest)

                print(f"[{stage_name}] waiting.")
                print(result.message or "")
                return

            if result.status == "done":
                manifest.mark_done(
                    stage_name,
                    backend=class_path,
                    message=result.message,
                    inputs=result.inputs,
                    outputs=result.outputs,
                    metadata=result.metadata,
                )
                manifest.save(paths.manifest)

                print(f"[{stage_name}] done.")
                continue

            raise RuntimeError(
                result.message or f"Stage returned unsupported status: {result.status}"
            )

        except Exception as exc:
            manifest.mark_failed(stage_name, exc=exc, backend=class_path)
            manifest.save(paths.manifest)
            raise

    print("Generation finished.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene_root", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--input_image", type=str, required=False, default=None, help="Path to the input image. If not provided, it will look for input_image.png under scene_root.")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    config = load_config(Path(args.config))

    run_generation(
        scene_root=Path(args.scene_root),
        input_image=Path(args.input_image) if args.input_image else None,
        config=config,
        resume=args.resume,
        force=args.force,
    )


if __name__ == "__main__":
    main()
