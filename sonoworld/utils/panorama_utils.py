from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from sonoworld.utils.laplacian_pyramid_utils import LaplacianPyramid


def angle_to_radians(value: float) -> float:
    """Treat values outside +/-2pi as degrees, otherwise radians."""
    if abs(value) > 2 * np.pi:
        return float(np.deg2rad(value))
    return float(value)


def rotation_from_gravity(elevation: float, roll: float = 0.0) -> np.ndarray:
    """Return camera-space correction that cancels GeoCalib pitch and roll."""
    elevation = -elevation
    roll = -roll

    rx = np.array(
        [
            [1, 0, 0],
            [0, np.cos(elevation), -np.sin(elevation)],
            [0, np.sin(elevation), np.cos(elevation)],
        ],
        dtype=np.float64,
    )
    rz = np.array(
        [
            [np.cos(roll), -np.sin(roll), 0],
            [np.sin(roll), np.cos(roll), 0],
            [0, 0, 1],
        ],
        dtype=np.float64,
    )
    return rx @ rz


def project_fov_to_panorama(
    image: np.ndarray,
    fov_x: float,
    elevation: float = 0.0,
    roll: float = 0.0,
    pano_res: int = 1024,
    num_pyramid_levels: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Project a perspective image into an equirectangular panorama.

    Args:
        image: Input image as HxW or HxWxC array.
        fov_x: Horizontal field of view in radians.
        elevation: GeoCalib pitch/elevation in radians.
        roll: GeoCalib roll in radians.
        pano_res: Output panorama height. Width is ``2 * pano_res``.
        num_pyramid_levels: Optional cap for the mip sampler.

    Returns:
        ``(panorama, invalid_mask)`` where ``invalid_mask`` is true for pixels
        outside the perspective camera frustum.
    """
    img_h, img_w = image.shape[:2]
    fov_y = 2 * np.arctan(np.tan(fov_x / 2) * img_h / img_w)

    fx = img_w / (2 * np.tan(fov_x / 2))
    fy = img_h / (2 * np.tan(fov_y / 2))
    cx = img_w / 2
    cy = img_h / 2
    intrinsics = np.array(
        [
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1],
        ],
        dtype=np.float64,
    )

    pano_w, pano_h = pano_res * 2, pano_res
    yaw = np.linspace(-np.pi, np.pi, pano_w, endpoint=False)
    pitch = np.linspace(-np.pi / 2, np.pi / 2, pano_h, endpoint=False)
    yaw, pitch = np.meshgrid(yaw, pitch)

    dirs = np.stack(
        [
            np.cos(pitch) * np.sin(yaw),
            np.sin(pitch),
            np.cos(pitch) * np.cos(yaw),
        ],
        axis=-1,
    )
    dirs_cam = dirs @ rotation_from_gravity(elevation, roll).T

    dirs_img = np.einsum("...j,ij->...i", dirs_cam, intrinsics)
    cam_z = dirs_img[..., 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        cam_x = dirs_img[..., 0] / cam_z
        cam_y = dirs_img[..., 1] / cam_z

    invalid = (
        (cam_x < 0)
        | (cam_x >= img_w)
        | (cam_y < 0)
        | (cam_y >= img_h)
        | (cam_z <= 0)
        | ~np.isfinite(cam_x)
        | ~np.isfinite(cam_y)
    )
    valid = ~invalid

    pano_shape = (pano_h, pano_w, image.shape[2]) if image.ndim == 3 else (pano_h, pano_w)
    panorama = np.zeros(pano_shape, dtype=image.dtype)
    if not valid.any():
        return panorama, invalid

    with np.errstate(divide="ignore", invalid="ignore"):
        du = 1 / fx / cam_z
        dv = 1 / fy / cam_z
    du_dj, du_di = np.gradient(cam_x)
    dv_dj, dv_di = np.gradient(cam_y)
    rho = np.maximum(
        np.sqrt(du_di**2 + dv_di**2),
        np.sqrt(du_dj**2 + dv_dj**2),
    )
    rho = np.where(np.isfinite(rho), rho, 1e-8)

    sampler = LaplacianPyramid(image, num_levels=num_pyramid_levels)
    colors_f32 = sampler.sample_lapmip(
        cam_x[valid],
        cam_y[valid],
        du[valid],
        dv[valid],
        rho[valid],
    )

    if image.dtype == np.uint8:
        colors = np.clip(colors_f32, 0, 255).astype(np.uint8)
    else:
        colors = colors_f32.clip(min(0, image.min()), image.max()).astype(image.dtype)

    panorama[valid] = colors
    return panorama, invalid
