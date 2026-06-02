import torch
import numpy as np
from PIL import Image
from tqdm import trange

import cv2
import utils3d

from typing import Union

from moge.model.v1 import MoGeModel as MoGeModelV1
from moge.model.v2 import MoGeModel as MoGeModelV2
from moge.utils.panorama import spherical_uv_to_directions, get_panorama_cameras, split_panorama_image, merge_panorama_depth

from sonoworld.utils.logging_utils import getLogger

logger = getLogger(__name__)

class MogeDepth:
    def __init__(self, device: torch.device = 'cuda', version: str = 'v2'):
        self.device = device
        self.model: Union[MoGeModelV1, MoGeModelV2] = self.build_depth_model(device, version)

    @staticmethod
    def build_depth_model(device: torch.device = 'cuda', version: str = 'v2'):
        logger.info(f'⚠️ Loading MoGe depth model... version: {version}')

        if version == 'v1':
            model = MoGeModelV1.from_pretrained('Ruicheng/moge-vitl')
        elif version == 'v2':
            model = MoGeModelV2.from_pretrained('Ruicheng/moge-2-vitl-normal')
        else:
            raise ValueError(f"Unsupported MoGe version: {version}. Use 'v1' or 'v2'.")
        
        model = model.to(device).eval()

        return model

    def pred_pano_depth(self, image: Image.Image, batch_size: int = 8, max_distance: float = 300.0, return_tensors: bool = False):
        device = self.device
        
        image = np.array(image)
        height, width = image.shape[:2]
    
        splitted_extrinsics, splitted_intriniscs = get_panorama_cameras()
        splitted_resolution = 512
        splitted_images = split_panorama_image(image, splitted_extrinsics, splitted_intriniscs, splitted_resolution)

        splitted_distance_maps, splitted_masks = [], []
        for i in trange(0, len(splitted_images), batch_size, desc='Inferring splitted views', disable=len(splitted_images) <= batch_size, leave=False):
            image_tensor = torch.tensor(np.stack(splitted_images[i:i + batch_size]) / 255, dtype=torch.float32, device=device).permute(0, 3, 1, 2)
            fov_x, fov_y = np.rad2deg(utils3d.numpy.intrinsics_to_fov(np.array(splitted_intriniscs[i:i + batch_size])))
            fov_x = torch.tensor(fov_x, dtype=torch.float32, device=device)
            output = self.model.infer(image_tensor, fov_x=fov_x, apply_mask=False)
            distance_map, mask = output['points'].norm(dim=-1).cpu().numpy(), output['mask'].cpu().numpy()
            splitted_distance_maps.extend(list(distance_map))
            splitted_masks.extend(list(mask))

        merging_width, merging_height = min(1920, width), min(960, height)
        panorama_depth, panorama_mask = merge_panorama_depth(merging_width, merging_height, splitted_distance_maps, splitted_masks, splitted_extrinsics, splitted_intriniscs)

        panorama_depth = panorama_depth.astype(np.float32)
        panorama_depth = cv2.resize(panorama_depth, (width, height), cv2.INTER_LINEAR)
        panorama_mask = cv2.resize(panorama_mask.astype(np.uint8), (width, height), cv2.INTER_NEAREST) > 0
        panorama_depth[panorama_mask == 0] = max_distance
        panorama_points = panorama_depth[:, :, None] * spherical_uv_to_directions(utils3d.numpy.image_uv(width=width, height=height))

        if return_tensors:
            rgb = torch.from_numpy(image).float().to(device)
            depth = torch.from_numpy(panorama_depth).float().to(device)
            points = torch.from_numpy(panorama_points).float().to(device)
        else:
            rgb = image
            depth = panorama_depth
            points = panorama_points

        return {
            "rgb": rgb,
            "depth": depth,
            "points": points
        }