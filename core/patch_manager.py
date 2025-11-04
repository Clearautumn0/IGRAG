"""PatchManager: split images into non-overlapping or sliding patches for local retrieval.

Provides utilities to split a PIL image into patches and (optionally) merge them back for debugging.
"""
from typing import List, Tuple, Dict, Optional, Union
import math
import numpy as np
from PIL import Image
import yaml
import os
import logging


logger = logging.getLogger(__name__)


class PatchManager:
    """Manage image patching for local (patch-level) retrieval.

    Splitting strategy:
    - Given patch_size and stride, compute top-left coordinates for patches.
    - If image dimensions are smaller than patch_size or do not divide evenly, pad the image
      on the right/bottom with black pixels so that the last patch aligns with the edge.

    Returned data:
    - patches: list of PIL.Image patches (RGB)
    - coords: list of dicts with keys: x, y, w, h (coordinates on the ORIGINAL image), and pad info
    """

    def __init__(self, config: Optional[Union[dict, str]] = None):
        """Load patching configuration.

        Args:
            config: dict or path to YAML config. If omitted, will try `configs/config.yaml`.
        """
        cfg = {}
        if config is None:
            cfg_path = os.path.join("configs", "config.yaml")
            if os.path.exists(cfg_path):
                with open(cfg_path, "r") as f:
                    cfg = yaml.safe_load(f)
        elif isinstance(config, str):
            if os.path.exists(config):
                with open(config, "r") as f:
                    cfg = yaml.safe_load(f)
        else:
            cfg = config

        patch_cfg = cfg.get("patch_config", {}) if cfg else {}
        self.enabled = bool(patch_cfg.get("enabled", False))
        self.patch_size = int(patch_cfg.get("patch_size", 224))
        self.stride = int(patch_cfg.get("stride", self.patch_size))
        self.top_k_patches = int(patch_cfg.get("top_k_patches", 3))

    def split_image(self, image: Union[str, Image.Image]) -> Tuple[List[Image.Image], List[Dict]]:
        """Split a PIL image (or image path) into patches.

        Args:
            image: PIL.Image.Image or path to image file.

        Returns:
            patches: list of PIL.Image patches (RGB)
            coords: list of dicts with keys: x, y, w, h, padded (tuple pad_right, pad_bottom)
        """
        if isinstance(image, str):
            img = Image.open(image)
        elif isinstance(image, Image.Image):
            img = image
        else:
            raise ValueError("image must be a file path or PIL.Image.Image")

        if img.mode != "RGB":
            img = img.convert("RGB")

        arr = np.array(img)
        h, w, c = arr.shape
        ps = self.patch_size
        st = self.stride

        # if image smaller than patch, pad to patch size
        pad_right = 0
        pad_bottom = 0
        if w < ps:
            pad_right = ps - w
        if h < ps:
            pad_bottom = ps - h

        # compute start coordinates
        starts_x = list(range(0, max(1, w - ps + 1), st))
        starts_y = list(range(0, max(1, h - ps + 1), st))

        # ensure last patch covers the right/bottom edge
        if (w - ps) % st != 0:
            last_x = max(0, w - ps)
            if len(starts_x) == 0 or starts_x[-1] != last_x:
                starts_x.append(last_x)
        if (h - ps) % st != 0:
            last_y = max(0, h - ps)
            if len(starts_y) == 0 or starts_y[-1] != last_y:
                starts_y.append(last_y)

        # if image smaller than patch, starts lists will be [0]
        # pad array
        pad_w = 0
        pad_h = 0
        if starts_x and starts_x[-1] + ps > w:
            pad_w = starts_x[-1] + ps - w
        if starts_y and starts_y[-1] + ps > h:
            pad_h = starts_y[-1] + ps - h

        total_pad_right = pad_right + pad_w
        total_pad_bottom = pad_bottom + pad_h

        if total_pad_right > 0 or total_pad_bottom > 0:
            arr = np.pad(arr, ((0, total_pad_bottom), (0, total_pad_right), (0, 0)), mode="constant", constant_values=0)

        patches = []
        coords = []
        for y in starts_y:
            for x in starts_x:
                patch_arr = arr[y : y + ps, x : x + ps, :]
                patch_img = Image.fromarray(patch_arr)
                patches.append(patch_img)
                # coordinates relative to original image (clipped)
                coord = {
                    "x": int(x),
                    "y": int(y),
                    "w": int(min(ps, w - x)),
                    "h": int(min(ps, h - y)),
                    "pad_right": int(total_pad_right),
                    "pad_bottom": int(total_pad_bottom),
                }
                coords.append(coord)

        return patches, coords

    def merge_patches(self, patches: List[Image.Image], coords: List[Dict], original_size: Tuple[int, int]) -> Image.Image:
        """Merge patches back into an image for visualization.

        Args:
            patches: list of PIL.Image patches
            coords: list of dicts with x,y
            original_size: (width, height) of original image

        Returns:
            PIL.Image reconstructed (may include padding areas)
        """
        ow, oh = original_size
        # determine canvas size from max x+patch_w and y+patch_h
        max_w = ow
        max_h = oh
        # if padded coords indicate padding, extend canvas
        pad_right = 0
        pad_bottom = 0
        if coords:
            pad_right = max(c.get("pad_right", 0) for c in coords)
            pad_bottom = max(c.get("pad_bottom", 0) for c in coords)
            max_w = ow + pad_right
            max_h = oh + pad_bottom

        canvas = Image.new("RGB", (max_w, max_h), (0, 0, 0))
        for patch, c in zip(patches, coords):
            x = c["x"]
            y = c["y"]
            canvas.paste(patch, (x, y))

        return canvas
