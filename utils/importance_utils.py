"""Semantic importance masks used by RGB Gaussian optimization."""

import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image


MASK_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def find_mask(mask_dir, image_name):
    if not mask_dir:
        return None
    stem = Path(image_name).stem
    candidates = [os.path.join(mask_dir, image_name)]
    candidates.extend(os.path.join(mask_dir, stem + suffix) for suffix in MASK_EXTENSIONS)
    return next((path for path in candidates if os.path.isfile(path)), None)


def load_importance_mask(mask_dir, image_name, size, device, cache):
    key = (image_name, tuple(size))
    if key in cache:
        return cache[key]
    path = find_mask(mask_dir, image_name)
    if path is None:
        cache[key] = None
        return None
    mask = Image.open(path).convert("L")
    mask = torch.from_numpy(np.asarray(mask, dtype=np.float32) / 255.0)[None, None]
    mask = torch.nn.functional.interpolate(
        mask, size=size, mode="bilinear", align_corners=False
    )[0].to(device)
    cache[key] = mask
    return mask


def importance_to_tiers(mask):
    """Convert legacy binary or new 0/127/255 maps into integer tiers 0/1/2."""
    if mask is None:
        return None
    return torch.where(mask >= 0.75, 2, torch.where(mask >= 0.25, 1, 0)).long()


def weighted_l1(rendered, target, mask, foreground_weight, background_weight):
    pixel_error = torch.abs(rendered - target).mean(dim=0, keepdim=True)
    weights = background_weight + mask * (foreground_weight - background_weight)
    return (pixel_error * weights).sum() / weights.sum().clamp_min(1e-8)


def weighted_tier_l1(rendered, target, tiers, tier_weights=(0.35, 1.0, 4.0)):
    """RGB loss with explicit background, normal-object and important-object tiers."""
    pixel_error = torch.abs(rendered - target).mean(dim=0, keepdim=True)
    values = torch.as_tensor(tier_weights, dtype=rendered.dtype, device=rendered.device)
    weights = values[tiers.long().clamp(0, 2)]
    return (pixel_error * weights).sum() / weights.sum().clamp_min(1e-8)
