# sonoworld/stages/visual_scene/marble.py

from __future__ import annotations

import shutil

from sonoworld.core.stage import Stage, StageContext, StageResult
from sonoworld.core.artifacts import ref, save_visual_scene_summary
from sonoworld.schemas.visual_scene import PanoramaGeometry, VisualSceneSummary, VisualRepresentation
from sonoworld.utils.gaussian_splat_utils import (
    panorama_depth_to_points,
    render_marble_panorama_depth,
    save_panorama_geometry_visualizations,
)


class MarbleVisualSceneStage(Stage):
    """Manual Marble backend for visual scene generation."""

    name = "visual_scene"
    backend = "marble"

    def run(self, ctx: StageContext) -> StageResult:
        paths = ctx.paths
        scene_root = ctx.scene_root

        panorama = paths.find_panorama()
        stage_cfg = ctx.stage_config(self.name)

        paths.marble_request.mkdir(parents=True, exist_ok=True)
        paths.marble_representation.mkdir(parents=True, exist_ok=True)

        request_panorama = paths.marble_request / "panorama.png"
        expected_gaussian = paths.marble_gaussian

        if not expected_gaussian.exists():
            shutil.copy2(panorama, request_panorama)

            readme = paths.marble_request / "README.md"
            readme.write_text(
                "\n".join([
                    "# Marble generation request",
                    "",
                    "Upload `panorama.png` to Marble.",
                    "Download the Marble Gaussian Splatting result.",
                    "Place the result at:",
                    "",
                    "`visual_scene/representation/marble/gaussian.ply`",
                    "",
                    "Then run:",
                    "",
                    "`python generate.py --scene_root <scene_root> --config <config> --resume`",
                    "",
                ]),
                encoding="utf-8",
            )

            return StageResult(
                status="waiting",
                inputs={
                    "panorama": ref(
                        panorama,
                        scene_root,
                        role="panorama",
                        media_type="image",
                    ),
                },
                outputs={
                    "request_panorama": ref(
                        request_panorama,
                        scene_root,
                        role="marble_upload",
                        media_type="image",
                    ),
                    "expected_gaussian": ref(
                        expected_gaussian,
                        scene_root,
                        role="marble_output",
                        media_type="model/ply",
                    ),
                },
                message=(
                    "Marble output is required. Upload "
                    "visual_scene/marble_request/panorama.png to Marble, "
                    "place gaussian.ply under visual_scene/representation/marble/, "
                    "then run generation.py with --resume."
                ),
            )

        summary = VisualSceneSummary(
            panorama=ref(
                panorama,
                scene_root,
                role="panorama",
                media_type="image",
            ),
            geometry=PanoramaGeometry(
                depth=ref(
                    paths.panorama_depth_npy,
                    scene_root,
                    role="panorama_depth",
                    media_type="application/x-npy",
                ),
                points=ref(
                    paths.panorama_points_npy,
                    scene_root,
                    role="panorama_points",
                    media_type="application/x-npy",
                ),
                coordinate_system="pano",
            ),
            representation=VisualRepresentation(
                representation_type="gaussian_splatting",
                backend="marble",
                primary_file=ref(
                    expected_gaussian,
                    scene_root,
                    role="gaussian",
                    media_type="model/ply",
                ),
                files={
                    "gaussian": ref(
                        expected_gaussian,
                        scene_root,
                        role="gaussian",
                        media_type="model/ply",
                    ),
                },
                coordinate_system="marble",
            ),
            backend="marble",
            status="done",
        )

        depth_path = paths.panorama_depth_npy
        points_path = paths.panorama_points_npy
        depth_vis_path = paths.visual_scene / "panorama_depth.jpg"
        points_vis_path = paths.visual_scene / "points.jpg"
        visuals_written = False
        if ctx.force or not depth_path.exists():
            from PIL import Image

            with Image.open(panorama) as image:
                panorama_width, panorama_height = image.size

            cube_map_width = int(
                stage_cfg.get(
                    "depth_cube_map_width",
                    min(512, max(128, panorama_width // 4)),
                )
            )
            render_marble_panorama_depth(
                expected_gaussian,
                depth_path,
                width=panorama_width,
                height=panorama_height,
                cube_map_width=cube_map_width,
                device=str(stage_cfg.get("depth_device", ctx.device)),
                near_plane=float(stage_cfg.get("depth_near_plane", 0.1)),
                far_plane=float(stage_cfg.get("depth_far_plane", 1000.0)),
                points_out_path=points_path,
                depth_vis_out_path=depth_vis_path,
                points_vis_out_path=points_vis_path,
            )
            visuals_written = True
        elif not points_path.exists():
            import numpy as np

            depth = np.load(depth_path).astype(np.float32)
            points = panorama_depth_to_points(depth)
            np.save(points_path, points.astype(np.float32))
        else:
            depth = None
            points = None

        if not visuals_written and (
            ctx.force or not depth_vis_path.exists() or not points_vis_path.exists()
        ):
            import numpy as np

            if depth is None:
                depth = np.load(depth_path).astype(np.float32)
            if points is None:
                points = np.load(points_path).astype(np.float32)
            save_panorama_geometry_visualizations(
                depth,
                points,
                depth_out_path=depth_vis_path,
                points_out_path=points_vis_path,
            )

        summary_path = save_visual_scene_summary(paths, summary)

        return StageResult(
            status="done",
            inputs={
                "panorama": ref(
                    panorama,
                    scene_root,
                    role="panorama",
                    media_type="image",
                ),
            },
            outputs={
                "summary": ref(
                    summary_path,
                    scene_root,
                    role="visual_scene_summary",
                    media_type="application/json",
                ),
                "gaussian": ref(
                    expected_gaussian,
                    scene_root,
                    role="gaussian",
                    media_type="model/ply",
                ),
                "depth": ref(
                    depth_path,
                    scene_root,
                    role="panorama_depth",
                    media_type="application/x-npy",
                ),
                "points": ref(
                    points_path,
                    scene_root,
                    role="panorama_points",
                    media_type="application/x-npy",
                ),
                "depth_visualization": ref(
                    depth_vis_path,
                    scene_root,
                    role="panorama_depth_visualization",
                    media_type="image/jpeg",
                ),
                "points_visualization": ref(
                    points_vis_path,
                    scene_root,
                    role="panorama_points_visualization",
                    media_type="image/jpeg",
                ),
            },
            message="Marble visual scene is ready.",
        )
