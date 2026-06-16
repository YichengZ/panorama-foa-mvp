from __future__ import annotations

import hashlib
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
from PIL import Image

from sonoworld.core.artifacts import ref, save_segmentation_summary
from sonoworld.core.stage import Stage, StageContext, StageResult
from sonoworld.schemas.common import BBox, read_json, write_json
from sonoworld.schemas.segmentation import SegmentationInstance, SegmentationSummary
from sonoworld.utils.text_utils import slug_text



@dataclass
class TileData:
    image: np.ndarray
    yaw: float
    fov_x: float
    fov_y: float
    index: int


@dataclass
class MaskData:
    mask: np.ndarray
    bbox: Tuple[int, int, int, int]
    score: float
    area: int
    source_tile_indices: List[int] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mask(
        cls,
        mask: np.ndarray,
        score: float,
        source_tile_indices: Optional[Iterable[int]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional["MaskData"]:
        mask = np.asarray(mask).astype(bool)
        if not mask.any():
            return None
        bbox = bbox_from_mask(mask)
        return cls(
            mask=mask,
            bbox=bbox,
            score=float(score),
            area=int(mask.sum()),
            source_tile_indices=sorted(set(source_tile_indices or [])),
            metadata=dict(metadata or {}),
        )


@dataclass
class CandidateData:
    mask: np.ndarray
    votemap: np.ndarray
    visibility: np.ndarray
    pix_score: np.ndarray

    @classmethod
    def from_mask(cls, mask: np.ndarray) -> "CandidateData":
        mask = np.asarray(mask).astype(bool)
        return cls(
            mask=mask,
            visibility=np.zeros_like(mask, dtype=bool),
            pix_score=np.zeros_like(mask, dtype=np.float32),
            votemap=np.zeros_like(mask, dtype=bool),
        )


@dataclass
class VoteData:
    roi_mask: np.ndarray
    dom_mask: np.ndarray
    score: float
    source_tile_indices: List[int] = field(default_factory=list)

    @classmethod
    def from_mask(
        cls,
        roi_mask: np.ndarray,
        dom_mask: np.ndarray,
        score: float,
        source_tile_indices: Optional[Iterable[int]] = None,
    ) -> "VoteData":
        roi_mask = np.asarray(roi_mask)
        if roi_mask.size == 0 or float(roi_mask.max()) <= 0:
            binary_roi = np.zeros_like(roi_mask, dtype=bool)
        else:
            binary_roi = roi_mask > (float(roi_mask.max()) * 0.5)
        return cls(
            roi_mask=binary_roi,
            dom_mask=np.asarray(dom_mask).astype(bool),
            score=float(score),
            source_tile_indices=sorted(set(source_tile_indices or [])),
        )


def angle_to_radians(value: float) -> float:
    if abs(value) > 2 * np.pi:
        return float(np.deg2rad(value))
    return float(value)


def bbox_from_mask(mask: np.ndarray) -> Tuple[int, int, int, int]:
    y, x = np.nonzero(mask)
    return int(y.min()), int(x.min()), int(y.max() + 1), int(x.max() + 1)


def rough_contains(m1: np.ndarray, m2: np.ndarray, thr: float = 0.8) -> bool:
    area = int(np.asarray(m2).sum())
    if area <= 0:
        return False
    return float((np.asarray(m1).astype(bool) & np.asarray(m2).astype(bool)).sum()) / area > thr


def iou(m1: np.ndarray, m2: np.ndarray) -> float:
    m1 = np.asarray(m1).astype(bool)
    m2 = np.asarray(m2).astype(bool)
    union = int((m1 | m2).sum())
    if union <= 0:
        return 0.0
    return float((m1 & m2).sum()) / union


def merge_mask_data(base: MaskData, other: MaskData) -> None:
    base.mask = base.mask | other.mask
    base.score = max(base.score, other.score)
    base.area = int(base.mask.sum())
    base.bbox = bbox_from_mask(base.mask)
    base.source_tile_indices = sorted(
        set(base.source_tile_indices).union(other.source_tile_indices)
    )


def distinct_insts(
    masks: List[MaskData],
    mid_mask: Optional[np.ndarray],
) -> Tuple[List[MaskData], List[MaskData]]:
    masks = sorted(masks, key=lambda item: item.area, reverse=True)

    while True:
        distinct_masks: List[MaskData] = []
        for mask in masks:
            for inst in distinct_masks:
                if rough_contains(inst.mask, mask.mask) or rough_contains(mask.mask, inst.mask):
                    merge_mask_data(inst, mask)
                    break
            else:
                distinct_masks.append(mask)

        if len(distinct_masks) == len(masks):
            break

        masks = sorted(distinct_masks, key=lambda item: item.area, reverse=True)

    if mid_mask is None:
        return distinct_masks, []

    mid_mask = np.asarray(mid_mask).astype(bool)
    in_view_masks = [mask for mask in distinct_masks if (mask.mask & mid_mask).any()]
    return distinct_masks, in_view_masks


def binary_erode_3x3(mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask).astype(bool)
    if mask.size == 0:
        return mask

    padded = np.pad(mask, 1, mode="constant", constant_values=False)
    eroded = np.ones_like(mask, dtype=bool)
    height, width = mask.shape
    for dy in range(3):
        for dx in range(3):
            eroded &= padded[dy : dy + height, dx : dx + width]
    return eroded


def get_global_masks(
    pano_masks: List[MaskData],
    votes: List[VoteData],
    thr: float,
) -> Tuple[List[MaskData], List[int]]:
    global_masks: List[MaskData] = []
    mask_inds: List[int] = []

    for index, pano_mask in enumerate(pano_masks):
        candidate = CandidateData.from_mask(pano_mask.mask)
        matched_tiles: set[int] = set()

        for vote in votes:
            eroded_roi = binary_erode_3x3(vote.roi_mask)
            intersection = candidate.mask & eroded_roi

            if not intersection.any():
                continue

            matched_tiles.update(vote.source_tile_indices)
            candidate.visibility |= vote.dom_mask
            candidate.votemap |= eroded_roi
            candidate.pix_score[intersection] = np.maximum(
                candidate.pix_score[intersection],
                vote.score,
            )

        visible_area = int((candidate.visibility & (candidate.mask | candidate.votemap)).sum())
        if visible_area < 1:
            continue

        candidate_score = float(candidate.pix_score.sum()) / (visible_area + 1e-8)
        if candidate_score < thr:
            continue

        merged_mask = candidate.mask | candidate.votemap
        mask_data = MaskData.from_mask(
            merged_mask,
            score=max(candidate_score, pano_mask.score),
            source_tile_indices=matched_tiles,
            metadata={"merge": "sam3_pano_candidate_votes", "pano_candidate_index": index},
        )
        if mask_data is None:
            continue

        global_masks.append(mask_data)
        mask_inds.append(index)

    return global_masks, mask_inds


def refine_masks_by_pano(
    pano_masks: List[MaskData],
    pano_view_masks: List[VoteData],
    mask_inds: List[int],
    thr: float,
) -> List[MaskData]:
    refined_masks: List[MaskData] = []
    matched_pano_indices = set(mask_inds)

    for view_mask in pano_view_masks:
        ignore = False
        new_mask = view_mask.roi_mask.copy()
        merged_indices: List[int] = []

        for index, pano_mask in enumerate(pano_masks):
            if index in matched_pano_indices:
                if rough_contains(pano_mask.mask, view_mask.roi_mask):
                    ignore = True
                    break
            elif iou(pano_mask.mask, view_mask.roi_mask) > thr:
                new_mask |= pano_mask.mask
                merged_indices.append(index)

        if ignore:
            continue

        mask_data = MaskData.from_mask(
            new_mask,
            score=view_mask.score,
            source_tile_indices=view_mask.source_tile_indices,
            metadata={"merge": "sam3_tile_refinement", "merged_pano_indices": merged_indices},
        )
        if mask_data is not None:
            refined_masks.append(mask_data)

    return refined_masks


def spherical_uv_from_dirs(directions: np.ndarray) -> np.ndarray:
    directions = directions / np.linalg.norm(directions, axis=-1, keepdims=True).clip(1e-8)
    u = (1.0 - np.arctan2(directions[..., 1], directions[..., 0]) / (2 * np.pi)) % 1.0
    v = np.arccos(np.clip(directions[..., 2], -1.0, 1.0)) / np.pi
    return np.stack([u, v], axis=-1)


def pano_directions(height: int, width: int) -> np.ndarray:
    u = (np.arange(width, dtype=np.float32) + 0.5) / width
    v = (np.arange(height, dtype=np.float32) + 0.5) / height
    u, v = np.meshgrid(u, v)
    theta = (1.0 - u) * 2 * np.pi
    z = np.cos(v * np.pi)
    radius = np.sin(v * np.pi)
    return np.stack(
        [
            np.cos(theta) * radius,
            np.sin(theta) * radius,
            z,
        ],
        axis=-1,
    ).astype(np.float32)


def camera_basis(yaw: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    forward = np.array([np.cos(yaw), np.sin(yaw), 0.0], dtype=np.float32)
    right = np.array([-np.sin(yaw), np.cos(yaw), 0.0], dtype=np.float32)
    up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    return forward, right, up


def sample_image_wrap_x(image: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)

    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    x1 = x0 + 1
    y1 = y0 + 1

    wx = x - x0
    wy = y - y0

    x0 = np.mod(x0, width)
    x1 = np.mod(x1, width)
    y0 = np.clip(y0, 0, height - 1)
    y1 = np.clip(y1, 0, height - 1)

    top = image[y0, x0] * (1.0 - wx)[..., None] + image[y0, x1] * wx[..., None]
    bottom = image[y1, x0] * (1.0 - wx)[..., None] + image[y1, x1] * wx[..., None]
    return top * (1.0 - wy)[..., None] + bottom * wy[..., None]


def sample_scalar_clamped(image: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)

    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    x1 = x0 + 1
    y1 = y0 + 1

    wx = x - x0
    wy = y - y0

    x0 = np.clip(x0, 0, width - 1)
    x1 = np.clip(x1, 0, width - 1)
    y0 = np.clip(y0, 0, height - 1)
    y1 = np.clip(y1, 0, height - 1)

    top = image[y0, x0] * (1.0 - wx) + image[y0, x1] * wx
    bottom = image[y1, x0] * (1.0 - wx) + image[y1, x1] * wx
    return top * (1.0 - wy) + bottom * wy


def pano_to_fov(
    panorama: np.ndarray,
    yaw: float,
    fov_x: float,
    fov_y: float,
    resolution_h: int,
    resolution_w: int,
) -> np.ndarray:
    x = ((np.arange(resolution_w, dtype=np.float32) + 0.5) / resolution_w * 2.0 - 1.0)
    y = (1.0 - (np.arange(resolution_h, dtype=np.float32) + 0.5) / resolution_h * 2.0)
    x, y = np.meshgrid(x * np.tan(fov_x / 2.0), y * np.tan(fov_y / 2.0))

    forward, right, up = camera_basis(yaw)
    directions = forward + x[..., None] * right + y[..., None] * up
    spherical_uv = spherical_uv_from_dirs(directions)

    pano_h, pano_w = panorama.shape[:2]
    px = spherical_uv[..., 0] * pano_w - 0.5
    py = spherical_uv[..., 1] * pano_h - 0.5
    tile = sample_image_wrap_x(panorama.astype(np.float32), px, py)
    return np.clip(tile, 0, 255).astype(np.uint8)


def fov_to_pano(
    image: np.ndarray,
    yaw: float,
    fov_x: float,
    fov_y: float,
    pano_h: int,
    pano_w: int,
) -> Tuple[np.ndarray, np.ndarray]:
    directions = pano_directions(pano_h, pano_w)
    forward, right, up = camera_basis(yaw)

    cam_x = np.einsum("...i,i->...", directions, right)
    cam_y = np.einsum("...i,i->...", directions, up)
    cam_z = np.einsum("...i,i->...", directions, forward)

    with np.errstate(divide="ignore", invalid="ignore"):
        norm_x = cam_x / cam_z / np.tan(fov_x / 2.0)
        norm_y = cam_y / cam_z / np.tan(fov_y / 2.0)

    inbound = (
        (cam_z > 0)
        & (norm_x >= -1.0)
        & (norm_x <= 1.0)
        & (norm_y >= -1.0)
        & (norm_y <= 1.0)
        & np.isfinite(norm_x)
        & np.isfinite(norm_y)
    )

    fov_h, fov_w = image.shape[:2]
    px = (norm_x + 1.0) * 0.5 * fov_w - 0.5
    py = (1.0 - norm_y) * 0.5 * fov_h - 0.5

    image_f = np.asarray(image, dtype=np.float32)
    if image_f.ndim == 3:
        image_f = image_f[..., 0]
    sampled = sample_scalar_clamped(image_f, px, py)
    sampled = np.where(inbound, sampled, 0.0)
    return np.clip(sampled, 0, 255).astype(np.uint8), inbound


def make_tiles(
    panorama: np.ndarray,
    fov_x_deg: float = 60.0,
    fov_y_deg: float = 90.0,
    n_cam: int = 12,
    resolution_w: int = 640,
) -> Tuple[List[TileData], np.ndarray]:
    fov_x = angle_to_radians(float(fov_x_deg))
    fov_y = angle_to_radians(float(fov_y_deg))
    resolution_w = max(1, int(resolution_w))
    resolution_h = max(1, int(resolution_w / np.tan(fov_x / 2.0) * np.tan(fov_y / 2.0) + 0.5))

    yaws = np.linspace(0.0, 2.0 * np.pi, max(1, int(n_cam)), endpoint=False)
    tiles = [
        TileData(
            image=pano_to_fov(
                panorama,
                yaw=float(yaw),
                fov_x=fov_x,
                fov_y=fov_y,
                resolution_h=resolution_h,
                resolution_w=resolution_w,
            ),
            yaw=float(yaw),
            fov_x=fov_x,
            fov_y=fov_y,
            index=index,
        )
        for index, yaw in enumerate(yaws)
    ]

    stride_theta = 2.0 * np.pi / max(1, int(n_cam))
    mid_mask = np.zeros((resolution_h, resolution_w), dtype=bool)
    mid_length = int(resolution_w / 2.0 / np.tan(fov_x / 2.0) * np.tan(stride_theta / 2.0) + 0.5)
    x0 = max(0, resolution_w // 2 - mid_length)
    x1 = min(resolution_w, resolution_w // 2 + mid_length)
    mid_mask[:, x0:x1] = True

    return tiles, mid_mask


class SAM3SegmentationStage(Stage):
    """SAM3-backed open-vocabulary panorama segmentation stage."""

    name = "segmentation"
    backend = "sam3"

    def __init__(
        self,
        sam3_root: str,
        checkpoint_path: Optional[str] = None,
        model_id: str = "facebook/sam3",
        load_from_hf: bool = True,
        processor_resolution: int = 1008,
        model_confidence_threshold: float = 0.0,
        default_score_threshold: float = 0.7,
        fov_x: float = 60.0,
        fov_y: float = 90.0,
        n_cam: int = 12,
        tile_resolution_w: int = 640,
        max_pano_height: int = 1024,
        global_vote_threshold: float = 0.5,
        refine_iou_threshold: float = 0.8,
        min_mask_area: int = 16,
        max_instances_per_label: int = 50,
        save_tiles: bool = True,
        dump_all_pano_candidates: bool = False,
    ) -> None:
        self.sam3_root = sam3_root
        self.checkpoint_path = checkpoint_path
        self.model_id = model_id
        self.load_from_hf = bool(load_from_hf)
        self.processor_resolution = int(processor_resolution)
        self.model_confidence_threshold = float(model_confidence_threshold)
        self.default_score_threshold = float(default_score_threshold)
        self.fov_x = float(fov_x)
        self.fov_y = float(fov_y)
        self.n_cam = int(n_cam)
        self.tile_resolution_w = int(tile_resolution_w)
        self.max_pano_height = int(max_pano_height)
        self.global_vote_threshold = float(global_vote_threshold)
        self.refine_iou_threshold = float(refine_iou_threshold)
        self.min_mask_area = int(min_mask_area)
        self.max_instances_per_label = int(max_instances_per_label)
        self.save_tiles = bool(save_tiles)
        self.dump_all_pano_candidates = bool(dump_all_pano_candidates)
        self._model: Any = None
        self._processor: Any = None

    def run(self, ctx: StageContext) -> StageResult:
        paths = ctx.paths
        scene_root = ctx.scene_root
        stage_cfg = ctx.stage_config(self.name)
        paths.segmentation.mkdir(parents=True, exist_ok=True)

        panorama_path = paths.find_panorama()
        grounding = self._load_grounding(ctx)
        write_json(paths.grounding_input, grounding)
        prompts = list(grounding.keys())

        if not prompts:
            summary = SegmentationSummary(
                panorama=ref(panorama_path, scene_root, role="panorama", media_type="image"),
                instances=[],
                backend=self.backend,
                model=self.model_id,
                metadata={"message": "No grounding prompts to segment."},
            )
            summary_path = save_segmentation_summary(paths, summary)
            return self._make_stage_result(ctx, panorama_path, summary_path, summary)

        original_pil = Image.open(panorama_path).convert("RGB")
        panorama, work_pil = self._prepare_panorama(original_pil, stage_cfg)
        work_h, work_w = panorama.shape[:2]

        tiles, mid_mask = make_tiles(
            panorama,
            fov_x_deg=float(stage_cfg.get("fov_x", self.fov_x)),
            fov_y_deg=float(stage_cfg.get("fov_y", self.fov_y)),
            n_cam=int(stage_cfg.get("n_cam", self.n_cam)),
            resolution_w=int(stage_cfg.get("tile_resolution_w", self.tile_resolution_w)),
        )

        if bool(stage_cfg.get("save_tiles", self.save_tiles)):
            self._save_tiles(paths.segmentation / "tiles", tiles)

        predictor = self._get_predictor(ctx.device, stage_cfg)
        pano_predictions = predictor.predict_prompts(work_pil, grounding)
        if bool(stage_cfg.get("dump_all_pano_candidates", self.dump_all_pano_candidates)):
            self._dump_pano_candidates(paths.segmentation / "pano_all_candidates", pano_predictions)

        pano_vote_masks: Dict[str, List[VoteData]] = {prompt: [] for prompt in prompts}
        pano_view_masks: Dict[str, List[VoteData]] = {prompt: [] for prompt in prompts}

        for tile in tiles:
            tile_predictions = predictor.predict_prompts(Image.fromarray(tile.image), grounding)
            for prompt, masks in tile_predictions.items():
                masks = sorted(masks, key=lambda item: item.area, reverse=True)
                distinct_masks, in_view_masks = distinct_insts(masks, mid_mask)

                for mask_data in distinct_masks:
                    projected, dom_mask = fov_to_pano(
                        (mask_data.mask.astype(np.uint8) * 255),
                        yaw=tile.yaw,
                        fov_x=tile.fov_x,
                        fov_y=tile.fov_y,
                        pano_h=work_h,
                        pano_w=work_w,
                    )
                    pano_vote_masks[prompt].append(
                        VoteData.from_mask(
                            projected,
                            dom_mask,
                            score=mask_data.score,
                            source_tile_indices=[tile.index],
                        )
                    )

                for mask_data in in_view_masks:
                    projected, dom_mask = fov_to_pano(
                        (mask_data.mask.astype(np.uint8) * 255),
                        yaw=tile.yaw,
                        fov_x=tile.fov_x,
                        fov_y=tile.fov_y,
                        pano_h=work_h,
                        pano_w=work_w,
                    )
                    pano_view_masks[prompt].append(
                        VoteData.from_mask(
                            projected,
                            dom_mask,
                            score=mask_data.score,
                            source_tile_indices=[tile.index],
                        )
                    )

        final_masks = self._merge_prompt_masks(
            prompts=prompts,
            pano_predictions=pano_predictions,
            pano_vote_masks=pano_vote_masks,
            pano_view_masks=pano_view_masks,
            stage_cfg=stage_cfg,
        )

        instances = self._write_instances(
            final_masks=final_masks,
            panorama=original_pil,
            work_size=(work_w, work_h),
            scene_root=scene_root,
            output_root=paths.segmentation / "global_masks",
        )

        summary = SegmentationSummary(
            panorama=ref(panorama_path, scene_root, role="panorama", media_type="image"),
            instances=instances,
            backend=self.backend,
            model=self.model_id,
            metadata={
                "grounding": grounding,
                "count_by_label": self._count_by_label(instances),
                "sam3_root": str(stage_cfg.get("sam3_root", self.sam3_root)),
                "checkpoint_path": stage_cfg.get("checkpoint_path", self.checkpoint_path),
                "processor_resolution": int(stage_cfg.get("processor_resolution", self.processor_resolution)),
                "model_confidence_threshold": float(
                    stage_cfg.get("model_confidence_threshold", self.model_confidence_threshold)
                ),
                "panorama_size": [original_pil.width, original_pil.height],
                "inference_panorama_size": [work_w, work_h],
                "n_tiles": len(tiles),
            },
        )
        summary_path = save_segmentation_summary(paths, summary)
        return self._make_stage_result(ctx, panorama_path, summary_path, summary)

    def _load_grounding(self, ctx: StageContext) -> Dict[str, float]:
        paths = ctx.paths
        stage_cfg = ctx.stage_config(self.name)

        grounding: Dict[str, float] = {}
        default_threshold = float(stage_cfg.get("default_score_threshold", self.default_score_threshold))

        grounding_path = self._resolve_optional_path(stage_cfg.get("grounding_input"), ctx.scene_root)
        if grounding_path is None:
            grounding_path = paths.grounding_input

        if grounding_path.exists():
            grounding.update(
                self._normalize_grounding_payload(
                    read_json(grounding_path),
                    default_threshold=default_threshold,
                )
            )
        elif paths.understanding_metadata.exists():
            grounding.update(
                self._grounding_from_understanding_json(
                    read_json(paths.understanding_metadata),
                    default_threshold=default_threshold,
                )
            )
            write_json(grounding_path, grounding)

        if stage_cfg.get("prompts"):
            grounding.update(
                self._normalize_grounding_payload(
                    stage_cfg["prompts"],
                    default_threshold=default_threshold,
                )
            )

        return {
            label: threshold
            for label, threshold in grounding.items()
            if label.strip() and label.strip().lower() != "global"
        }

    def _normalize_grounding_payload(self, payload: Any, default_threshold: float) -> Dict[str, float]:
        grounding: Dict[str, float] = {}

        if isinstance(payload, dict):
            items = payload.items()
        elif isinstance(payload, (list, tuple, set)):
            items = [(item, default_threshold) for item in payload]
        else:
            raise TypeError("Segmentation prompts must be a dict or a list.")

        for label, threshold in items:
            clean_label = " ".join(str(label).strip().split())
            if not clean_label or clean_label.lower() == "global":
                continue
            try:
                score_threshold = float(threshold)
            except (TypeError, ValueError):
                score_threshold = default_threshold
            grounding[clean_label] = score_threshold

        return grounding

    def _grounding_from_understanding_json(
        self,
        payload: Dict[str, Any],
        default_threshold: float,
    ) -> Dict[str, float]:
        grounding: Dict[str, float] = {}
        for item in payload.get("objects", []):
            if not isinstance(item, dict):
                continue
            if not bool(item.get("should_segment", True)):
                continue
            label = " ".join(str(item.get("grounding_label", "")).strip().split())
            if label and label.lower() != "global":
                grounding[label] = default_threshold
        return grounding

    def _prepare_panorama(
        self,
        panorama: Image.Image,
        stage_cfg: Dict[str, Any],
    ) -> Tuple[np.ndarray, Image.Image]:
        max_height = int(stage_cfg.get("max_pano_height", self.max_pano_height))
        if max_height > 0 and panorama.height > max_height:
            scale = max_height / panorama.height
            size = (max(1, int(round(panorama.width * scale))), max_height)
            work = panorama.resize(size, self._resampling("LANCZOS"))
        else:
            work = panorama
        return np.asarray(work.convert("RGB")), work.convert("RGB")

    def _merge_prompt_masks(
        self,
        prompts: List[str],
        pano_predictions: Dict[str, List[MaskData]],
        pano_vote_masks: Dict[str, List[VoteData]],
        pano_view_masks: Dict[str, List[VoteData]],
        stage_cfg: Dict[str, Any],
    ) -> Dict[str, List[MaskData]]:
        final_masks: Dict[str, List[MaskData]] = {}
        global_vote_threshold = float(stage_cfg.get("global_vote_threshold", self.global_vote_threshold))
        refine_iou_threshold = float(stage_cfg.get("refine_iou_threshold", self.refine_iou_threshold))
        max_instances = int(stage_cfg.get("max_instances_per_label", self.max_instances_per_label))

        for prompt in prompts:
            pano_masks = list(pano_predictions.get(prompt, []))
            voted_global, mask_inds = get_global_masks(
                pano_masks,
                pano_vote_masks.get(prompt, []),
                thr=global_vote_threshold,
            )
            global_view_masks = refine_masks_by_pano(
                pano_masks,
                pano_view_masks.get(prompt, []),
                mask_inds,
                thr=refine_iou_threshold,
            )

            merged = sorted(
                pano_masks + voted_global + global_view_masks,
                key=lambda item: item.score,
                reverse=True,
            )
            merged = distinct_insts(merged, None)[0]
            final_masks[prompt] = sorted(merged, key=lambda item: item.score, reverse=True)[:max_instances]

        return final_masks

    def _write_instances(
        self,
        final_masks: Dict[str, List[MaskData]],
        panorama: Image.Image,
        work_size: Tuple[int, int],
        scene_root: Path,
        output_root: Path,
    ) -> List[SegmentationInstance]:
        reset_dir(output_root)
        instances: List[SegmentationInstance] = []
        original_size = panorama.size
        used_slugs: set[str] = set()

        for label, masks in final_masks.items():
            slug = slug_text(label, default="object")
            if slug in used_slugs:
                slug = f"{slug}_{hashlib.sha1(label.encode('utf-8')).hexdigest()[:8]}"
            used_slugs.add(slug)
            label_dir = output_root / slug
            label_dir.mkdir(parents=True, exist_ok=True)

            for index, mask_data in enumerate(masks):
                mask = mask_data.mask
                if work_size != original_size:
                    mask = resize_mask(mask, original_size)

                instance = MaskData.from_mask(
                    mask,
                    score=mask_data.score,
                    source_tile_indices=mask_data.source_tile_indices,
                    metadata=mask_data.metadata,
                )
                if instance is None:
                    continue

                mask_path = label_dir / f"{index}_mask.png"
                overlay_path = label_dir / f"{index}_overlay.png"
                Image.fromarray((instance.mask.astype(np.uint8) * 255), mode="L").save(mask_path)
                self._make_overlay(panorama, instance.mask, label, index).save(overlay_path)

                bbox = BBox.from_yxyx(*instance.bbox)
                instances.append(
                    SegmentationInstance(
                        instance_id=f"{slug}_{index}",
                        class_label=label,
                        mask=ref(mask_path, scene_root, role="segmentation_mask", media_type="image"),
                        overlay=ref(overlay_path, scene_root, role="segmentation_overlay", media_type="image"),
                        score=float(instance.score),
                        area=int(instance.area),
                        bbox=bbox,
                        source_prompt=label,
                        source_tile_indices=instance.source_tile_indices,
                        metadata=instance.metadata,
                    )
                )

        return instances

    def _make_overlay(
        self,
        panorama: Image.Image,
        mask: np.ndarray,
        label: str,
        index: int,
        alpha: float = 0.45,
    ) -> Image.Image:
        image = np.asarray(panorama.convert("RGB"), dtype=np.float32)
        color = np.asarray(color_for_label(label, index), dtype=np.float32)
        mask = np.asarray(mask).astype(bool)

        overlay = image.copy()
        overlay[mask] = overlay[mask] * (1.0 - alpha) + color * alpha
        return Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8), mode="RGB")

    def _save_tiles(self, tile_dir: Path, tiles: List[TileData]) -> None:
        reset_dir(tile_dir)
        for tile in tiles:
            Image.fromarray(tile.image, mode="RGB").save(tile_dir / f"tile_{tile.index:02d}.png")

    def _dump_pano_candidates(
        self,
        output_dir: Path,
        predictions: Dict[str, List[MaskData]],
    ) -> None:
        reset_dir(output_dir)
        for label, masks in predictions.items():
            label_dir = output_dir / slug_text(label, default="object")
            label_dir.mkdir(parents=True, exist_ok=True)
            for index, mask_data in enumerate(masks):
                Image.fromarray((mask_data.mask.astype(np.uint8) * 255), mode="L").save(
                    label_dir / f"{index:03d}_mask.png"
                )

    def _get_predictor(self, device: str, stage_cfg: Dict[str, Any]) -> "SAM3ImagePredictor":
        if self._processor is not None:
            return SAM3ImagePredictor(
                processor=self._processor,
                min_mask_area=int(stage_cfg.get("min_mask_area", self.min_mask_area)),
            )

        sam3_root = Path(str(stage_cfg.get("sam3_root", self.sam3_root)))
        if sam3_root.exists() and str(sam3_root) not in sys.path:
            sys.path.insert(0, str(sam3_root))

        try:
            from sam3.model_builder import build_sam3_image_model
            from sam3.model.sam3_image_processor import Sam3Processor
        except ImportError as exc:
            raise ImportError(
                "SAM3SegmentationStage requires the SAM3 package. "
                f"Install it or set `sam3_root` to a valid checkout. Tried: {sam3_root}"
            ) from exc

        checkpoint_path = stage_cfg.get("checkpoint_path", self.checkpoint_path)
        model = build_sam3_image_model(
            device=device,
            checkpoint_path=checkpoint_path,
            load_from_HF=bool(stage_cfg.get("load_from_hf", self.load_from_hf)),
        )
        processor = Sam3Processor(
            model,
            resolution=int(stage_cfg.get("processor_resolution", self.processor_resolution)),
            device=device,
            confidence_threshold=float(
                stage_cfg.get("model_confidence_threshold", self.model_confidence_threshold)
            ),
        )

        self._model = model
        self._processor = processor
        return SAM3ImagePredictor(
            processor=processor,
            min_mask_area=int(stage_cfg.get("min_mask_area", self.min_mask_area)),
        )

    def _make_stage_result(
        self,
        ctx: StageContext,
        panorama_path: Path,
        summary_path: Path,
        summary: SegmentationSummary,
    ) -> StageResult:
        return StageResult(
            status="done",
            inputs={
                "panorama": ref(panorama_path, ctx.scene_root, role="panorama", media_type="image"),
                "grounding_input": ref(
                    ctx.paths.grounding_input,
                    ctx.scene_root,
                    role="grounding_input",
                    media_type="application/json",
                ),
            },
            outputs={
                "summary": ref(
                    summary_path,
                    ctx.scene_root,
                    role="segmentation_summary",
                    media_type="application/json",
                ),
            },
            message="SAM3 segmentation complete.",
            metadata={
                "backend": self.backend,
                "model": self.model_id,
                "num_instances": len(summary.instances),
                "count_by_label": summary.count_by_label(),
            },
        )

    def _resolve_optional_path(self, path_value: Any, scene_root: Path) -> Optional[Path]:
        if not path_value:
            return None
        path = Path(str(path_value))
        if path.is_absolute():
            return path
        return scene_root / path

    def _resampling(self, name: str) -> Any:
        resampling = getattr(Image, "Resampling", None)
        if resampling is not None:
            return getattr(resampling, name)
        return getattr(Image, name)

    def _count_by_label(self, instances: List[SegmentationInstance]) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for instance in instances:
            counts[instance.class_label] = counts.get(instance.class_label, 0) + 1
        return counts


class SAM3ImagePredictor:
    def __init__(self, processor: Any, min_mask_area: int = 16, **kwargs: Any) -> None:
        self.processor = processor
        self.min_mask_area = int(min_mask_area)

    def predict_prompts(
        self,
        image: Image.Image,
        thresholds: Dict[str, float],
    ) -> Dict[str, List[MaskData]]:
        state = self.processor.set_image(image)
        predictions: Dict[str, List[MaskData]] = {}

        for prompt, threshold in thresholds.items():
            self.processor.reset_all_prompts(state)
            output = self.processor.set_text_prompt(state=state, prompt=prompt)
            predictions[prompt] = self._extract_masks(output, prompt, threshold)

        return predictions

    def _extract_masks(
        self,
        output: Dict[str, Any],
        prompt: str,
        threshold: float,
    ) -> List[MaskData]:
        masks = to_numpy(output.get("masks"))
        scores = to_numpy(output.get("scores"))
        boxes = to_numpy(output.get("boxes"))

        if masks is None or scores is None:
            return []

        if masks.ndim == 4 and masks.shape[1] == 1:
            masks = masks[:, 0]
        elif masks.ndim == 4 and masks.shape[-1] == 1:
            masks = masks[..., 0]

        if masks.ndim == 2:
            masks = masks[None, ...]

        scores = np.asarray(scores).reshape(-1)
        results: List[MaskData] = []

        for index, mask in enumerate(masks):
            score = float(scores[index]) if index < len(scores) else 0.0
            if score < threshold:
                continue

            mask_data = MaskData.from_mask(
                mask.astype(bool),
                score=score,
                metadata={
                    "prompt": prompt,
                    "sam3_index": index,
                    "sam3_box_xyxy": boxes[index].astype(float).tolist()
                    if boxes is not None and index < len(boxes)
                    else None,
                },
            )
            if mask_data is None or mask_data.area < self.min_mask_area:
                continue

            results.append(mask_data)

        return sorted(results, key=lambda item: item.score, reverse=True)


def to_numpy(value: Any) -> Optional[np.ndarray]:
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        return value.numpy()
    return np.asarray(value)


def resize_mask(mask: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    image = Image.fromarray((np.asarray(mask).astype(np.uint8) * 255), mode="L")
    resampling = getattr(Image, "Resampling", Image).NEAREST
    return np.asarray(image.resize(size, resampling)) > 127


def color_for_label(label: str, index: int) -> Tuple[int, int, int]:
    digest = hashlib.sha1(f"{label}:{index}".encode("utf-8")).digest()
    return int(digest[0]), int(digest[1]), int(digest[2])


def reset_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
