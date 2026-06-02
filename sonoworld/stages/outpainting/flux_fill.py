from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
from PIL import Image

from sonoworld.core.artifacts import ref
from sonoworld.core.stage import Stage, StageContext, StageResult
from sonoworld.schemas.common import write_json

from sonoworld.utils.logging_utils import getLogger

logger = getLogger(__name__)


DEFAULT_PROMPT = (
    "A high quality 360 panorama photo, HDR, RAW, 360 consistent, "
    "omnidirectional."
)


class FluxOutpaintingStage(Stage):
    """FLUX fill outpainting backend.

    The stage supports two inputs:
    - a partial panorama at inputs/panorama.png or scene_root/panorama.png
    - a perspective image at inputs/input.png, projected into a panorama first
    """

    name = "outpainting"
    backend = "flux_fill"

    def __init__(
        self,
        model_id: str = "black-forest-labs/FLUX.1-Fill-dev",
        lora_repo_id: str = "LeoXie/WorldGen",
        lora_filename: str = "models--WorldGen-Flux-Lora/worldgen_img2scene.safetensors",
        nunchaku_repo_template: str = "mit-han-lab/svdq-{precision}-flux.1-fill-dev",
        max_inpaint_width: int = 1920,
        panorama_resolution: int = 1024,
        bottom_mask_start: float = 0.77,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.0,
        blend_extend: int = 6,
        enable_cpu_offload: bool = True,
        enable_vae_tiling: bool = True,
    ) -> None:
        self.model_id = model_id
        self.lora_repo_id = lora_repo_id
        self.lora_filename = lora_filename
        self.nunchaku_repo_template = nunchaku_repo_template
        self.max_inpaint_width = max_inpaint_width
        self.panorama_resolution = panorama_resolution
        self.bottom_mask_start = bottom_mask_start
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        self.blend_extend = blend_extend
        self.enable_cpu_offload = enable_cpu_offload
        self.enable_vae_tiling = enable_vae_tiling
        self._pipe: Any = None
        self._last_projection_metadata: Dict[str, Any] = {}

    def run(self, ctx: StageContext) -> StageResult:
        paths = ctx.paths
        scene_root = ctx.scene_root
        stage_cfg = ctx.stage_config(self.name)
        paths.outpainting.mkdir(parents=True, exist_ok=True)

        prompt = str(stage_cfg.get("prompt", DEFAULT_PROMPT))
        seed = stage_cfg.get("seed")

        source, input_kind = self._find_source(paths)
        if input_kind == "panorama":
            init_image, mask = self._prepare_partial_panorama(source, ctx)
        else:
            init_image, mask = self._prepare_fov_image(source, ctx)

        pano_init_path = paths.outpainting / "pano_init.png"
        mask_path = paths.outpainting / "inpaint_mask.png"
        stage_json_path = paths.outpainting / "stage.json"

        init_image.save(pano_init_path)
        Image.fromarray((mask.astype(np.uint8) * 255), mode="L").save(mask_path)

        if not mask.any():
            init_image.save(paths.outpainted_panorama)
            summary = self._summary(
                prompt=prompt,
                input_kind=input_kind,
                source=source,
                output=paths.outpainted_panorama,
                mask_path=mask_path,
                pano_init_path=pano_init_path,
                skipped_inference=True,
            )
            write_json(stage_json_path, summary)
            return self._make_stage_result(
                ctx,
                source,
                pano_init_path,
                mask_path,
                stage_json_path,
                summary,
            )

        pipe = self._get_pipeline(ctx.device, self._get_low_vram(ctx.config))
        generator = self._make_generator(ctx.device, seed)

        kwargs: Dict[str, Any] = {
            "prompt": prompt,
            "height": init_image.height,
            "width": init_image.width,
            "image": init_image,
            "mask_image": Image.fromarray((mask.astype(np.uint8) * 255), mode="L"),
            "num_inference_steps": int(stage_cfg.get("num_inference_steps", self.num_inference_steps)),
            "guidance_scale": float(stage_cfg.get("guidance_scale", self.guidance_scale)),
            "blend_extend": int(stage_cfg.get("blend_extend", self.blend_extend)),
        }
        if generator is not None:
            kwargs["generator"] = generator

        with self._inference_context():
            filled = pipe(**kwargs).images[0].convert("RGB")

        output = self._restore_known_pixels(
            original=init_image,
            filled=filled,
            mask=mask,
            target_size=self._target_output_size(source, init_image, input_kind),
        )
        output.save(paths.outpainted_panorama)

        summary = self._summary(
            prompt=prompt,
            input_kind=input_kind,
            source=source,
            output=paths.outpainted_panorama,
            mask_path=mask_path,
            pano_init_path=pano_init_path,
            skipped_inference=False,
        )
        write_json(stage_json_path, summary)

        return self._make_stage_result(
            ctx,
            source,
            pano_init_path,
            mask_path,
            stage_json_path,
            summary,
        )

    def _find_source(self, paths: Any) -> Tuple[Path, str]:
        panorama_candidates = [
            paths.input_panorama,
            paths.root / "panorama.png",
            paths.root / "panorama.jpg",
        ]
        for path in panorama_candidates:
            if path.exists():
                return path, "panorama"

        return paths.find_input_image(), "fov_image"

    def _prepare_partial_panorama(
        self,
        source: Path,
        ctx: StageContext,
    ) -> Tuple[Image.Image, np.ndarray]:
        image = Image.open(source).convert("RGB")
        original_size = image.size
        image = self._resize_for_inpaint(image)

        stage_cfg = ctx.stage_config(self.name)
        bottom_mask_start = float(stage_cfg.get("bottom_mask_start", self.bottom_mask_start))
        bottom_mask_start = min(max(bottom_mask_start, 0.0), 1.0)

        mask = np.zeros((image.height, image.width), dtype=bool)
        mask[int(image.height * bottom_mask_start):, :] = True

        extra_mask = self._load_extra_mask(ctx, image.size)
        if extra_mask is not None:
            mask |= extra_mask

        logger.info(
            "Prepared partial panorama outpainting input: %s -> %s",
            original_size,
            image.size,
        )
        return image, mask

    def _prepare_fov_image(
        self,
        source: Path,
        ctx: StageContext,
    ) -> Tuple[Image.Image, np.ndarray]:
        from sonoworld.utils.panorama_utils import angle_to_radians, project_fov_to_panorama

        stage_cfg = ctx.stage_config(self.name)
        image = Image.open(source).convert("RGB")

        fov_x = stage_cfg.get("fov_x")
        elevation = stage_cfg.get("elevation")
        roll = stage_cfg.get("roll")
        if fov_x is None or elevation is None:
            calibrated_fov_x, calibrated_elevation, calibrated_roll = self._calibrate_fov_image(
                image,
                ctx.device,
            )
            fov_x = calibrated_fov_x if fov_x is None else angle_to_radians(float(fov_x))
            elevation = (
                calibrated_elevation
                if elevation is None
                else angle_to_radians(float(elevation))
            )
            roll = calibrated_roll if roll is None else angle_to_radians(float(roll))
        else:
            fov_x = angle_to_radians(float(fov_x))
            elevation = angle_to_radians(float(elevation))
            roll = 0.0 if roll is None else angle_to_radians(float(roll))

        self._last_projection_metadata = {
            "fov_x_deg": float(np.rad2deg(fov_x)),
            "elevation_deg": float(np.rad2deg(elevation)),
            "roll_deg": float(np.rad2deg(roll)),
        }

        pano, mask = project_fov_to_panorama(
            np.array(image),
            fov_x=fov_x,
            elevation=elevation,
            roll=roll,
            pano_res=int(stage_cfg.get("panorama_resolution", self.panorama_resolution)),
        )
        return Image.fromarray(pano.astype(np.uint8), mode="RGB"), mask

    def _get_pipeline(self, device: str, low_vram: bool) -> Any:
        if self._pipe is not None:
            return self._pipe

        import torch
        from huggingface_hub import hf_hub_download
        from sonoworld.third_party.flux_pano_fill_pipeline import FluxFillPipeline

        lora_path = hf_hub_download(
            repo_id=self.lora_repo_id,
            filename=self.lora_filename,
        )

        if low_vram:
            from nunchaku import NunchakuFluxTransformer2dModel
            from nunchaku.utils import get_precision
            from sonoworld.utils.lora_utils import load_and_fix_lora

            precision = get_precision()
            transformer = NunchakuFluxTransformer2dModel.from_pretrained(
                self.nunchaku_repo_template.format(precision=precision),
                offload=True,
            )
            pipe = FluxFillPipeline.from_pretrained(
                self.model_id,
                transformer=transformer,
                torch_dtype=torch.bfloat16,
                device=device,
            )
            state_dict, _ = load_and_fix_lora(lora_path)
            transformer.update_lora_params(state_dict)
        else:
            pipe = FluxFillPipeline.from_pretrained(
                self.model_id,
                torch_dtype=torch.bfloat16,
                device=device,
            )
            pipe.load_lora_weights(lora_path)

        if self.enable_cpu_offload and hasattr(pipe, "enable_model_cpu_offload"):
            pipe.enable_model_cpu_offload()
        if self.enable_vae_tiling and hasattr(pipe, "enable_vae_tiling"):
            pipe.enable_vae_tiling()

        self._pipe = pipe
        return pipe

    def _calibrate_fov_image(self, image: Image.Image, device: str) -> Tuple[float, float, float]:
        import torch
        from geocalib import GeoCalib
        from moge.model.v2 import MoGeModel

        image_tensor = (
            torch.from_numpy(np.array(image))
            .permute(2, 0, 1)
            .to(device)
            .float()
            / 255.0
        )

        calib_model = GeoCalib().eval().to(device)
        depth_model = MoGeModel.from_pretrained("Ruicheng/moge-2-vitl-normal").to(device).eval()

        geocalib_result = calib_model.calibrate(image_tensor)
        gravity = geocalib_result["gravity"][0]
        elevation = gravity.pitch.item()
        roll = gravity.roll.item() if hasattr(gravity, "roll") else 0.0

        with torch.inference_mode():
            predictions = depth_model.infer(image_tensor)

        intrinsics = predictions["intrinsics"]
        fx = intrinsics[0, 0].item()
        fov_x = 2 * np.arctan2(1, 2 * fx)

        calib_model.cpu()
        depth_model.cpu()
        return float(fov_x), float(elevation), float(roll)

    def _load_extra_mask(
        self,
        ctx: StageContext,
        size: Tuple[int, int],
    ) -> Optional[np.ndarray]:
        stage_cfg = ctx.stage_config(self.name)
        candidates = []
        if stage_cfg.get("mask_path"):
            candidates.append(Path(stage_cfg["mask_path"]))
        candidates.extend(
            [
                ctx.scene_root / "expanded_mask.png",
                ctx.scene_root / "mask.png",
                ctx.paths.inputs / "expanded_mask.png",
                ctx.paths.inputs / "mask.png",
            ]
        )

        for path in candidates:
            if not path.is_absolute():
                path = ctx.scene_root / path
            if path.exists():
                mask = Image.open(path).convert("L").resize(size, self._resampling("NEAREST"))
                return np.array(mask) > 127
        return None

    def _resize_for_inpaint(self, image: Image.Image) -> Image.Image:
        if self.max_inpaint_width <= 0 or image.width <= self.max_inpaint_width:
            return image

        scale = self.max_inpaint_width / image.width
        size = (self.max_inpaint_width, max(1, round(image.height * scale)))
        return image.resize(size, self._resampling("LANCZOS"))

    def _restore_known_pixels(
        self,
        original: Image.Image,
        filled: Image.Image,
        mask: np.ndarray,
        target_size: Tuple[int, int],
    ) -> Image.Image:
        filled = filled.resize(target_size, self._resampling("LANCZOS"))
        original = original.resize(target_size, self._resampling("LANCZOS"))
        mask_image = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")
        mask_image = mask_image.resize(target_size, self._resampling("NEAREST"))

        output = np.array(filled)
        original_array = np.array(original)
        mask_array = np.array(mask_image) > 127
        output[~mask_array] = original_array[~mask_array]
        return Image.fromarray(output, mode="RGB")

    def _target_output_size(
        self,
        source: Path,
        fallback: Image.Image,
        input_kind: str,
    ) -> Tuple[int, int]:
        if input_kind != "panorama":
            return fallback.size

        try:
            with Image.open(source) as image:
                return image.size
        except Exception:
            return fallback.size

    def _make_stage_result(
        self,
        ctx: StageContext,
        source: Path,
        pano_init_path: Path,
        mask_path: Path,
        stage_json_path: Path,
        summary: Dict[str, Any],
    ) -> StageResult:
        paths = ctx.paths
        scene_root = ctx.scene_root
        return StageResult(
            status="done",
            inputs={
                "source": ref(source, scene_root, role="outpainting_source", media_type="image"),
            },
            outputs={
                "panorama": ref(
                    paths.outpainted_panorama,
                    scene_root,
                    role="panorama",
                    media_type="image",
                ),
                "pano_init": ref(pano_init_path, scene_root, role="outpainting_init", media_type="image"),
                "inpaint_mask": ref(mask_path, scene_root, role="inpaint_mask", media_type="image"),
                "stage": ref(
                    stage_json_path,
                    scene_root,
                    role="outpainting_summary",
                    media_type="application/json",
                ),
            },
            message="FLUX outpainting complete.",
            metadata=summary,
        )

    def _summary(
        self,
        prompt: str,
        input_kind: str,
        source: Path,
        output: Path,
        mask_path: Path,
        pano_init_path: Path,
        skipped_inference: bool,
    ) -> Dict[str, Any]:
        return {
            "backend": self.backend,
            "model_id": self.model_id,
            "prompt": prompt,
            "input_kind": input_kind,
            "source": str(source),
            "output": str(output),
            "pano_init": str(pano_init_path),
            "inpaint_mask": str(mask_path),
            "num_inference_steps": self.num_inference_steps,
            "guidance_scale": self.guidance_scale,
            "blend_extend": self.blend_extend,
            "projection": self._last_projection_metadata,
            "skipped_inference": skipped_inference,
        }

    def _make_generator(self, device: str, seed: Optional[int]) -> Any:
        if seed is None:
            return None

        import torch

        return torch.Generator(device=device).manual_seed(int(seed))

    def _inference_context(self) -> Any:
        import torch

        return torch.no_grad()

    def _get_low_vram(self, config: Dict[str, Any]) -> bool:
        return bool(config.get("runtime", {}).get("low_vram", False))

    def _resampling(self, name: str) -> Any:
        resampling = getattr(Image, "Resampling", None)
        if resampling is not None:
            return getattr(resampling, name)
        return getattr(Image, name)
