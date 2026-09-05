"""Joint RGB/semantic supervision inspired by LaGa and SAGA."""

from functools import lru_cache

import numpy as np
import torch
from PIL import Image
from torch import nn


GRANULARITIES = ("coarse", "middle", "fine")


class ScaleGate(nn.Module):
    """SAGA-style learned channel gate conditioned on a normalized scale."""

    def __init__(self, dimensions):
        super().__init__()
        self.linear = nn.Linear(1, int(dimensions))
        nn.init.zeros_(self.linear.weight)
        nn.init.constant_(self.linear.bias, 3.0)

    def forward(self, level):
        value = torch.as_tensor(
            [[float(level) / 2.0]],
            dtype=self.linear.weight.dtype,
            device=self.linear.weight.device,
        )
        return torch.sigmoid(self.linear(value))[0]


def granularity_for_step(step):
    """Cycle through coarse, middle and fine supervision without extra renders."""
    return int(step) % len(GRANULARITIES)


@lru_cache(maxsize=8)
def load_joint_map(path):
    """Load a semantic map while remaining compatible with pre-hierarchy artifacts."""
    with np.load(path) as data:
        features = data["features"].astype(np.float32)
        valid = data["valid"].astype(bool)
        confidence = (
            data["confidence"].astype(np.float32)
            if "confidence" in data.files
            else np.ones(valid.shape, dtype=np.float32)
        )
        hierarchy_features = (
            data["hierarchy_features"].astype(np.float32)
            if "hierarchy_features" in data.files else None
        )
        hierarchy_valid = (
            data["hierarchy_valid"].astype(bool)
            if "hierarchy_valid" in data.files else None
        )
        hierarchy_confidence = (
            data["hierarchy_confidence"].astype(np.float32)
            if "hierarchy_confidence" in data.files else None
        )
        importance = (
            data["importance"].astype(np.uint8)
            if "importance" in data.files
            else valid.astype(np.uint8)
        )
        detail_weight = (
            data["detail_weight"].astype(np.float32)
            if "detail_weight" in data.files
            else np.ones(valid.shape, dtype=np.float32)
        )
        boundary = (
            data["boundary"].astype(bool)
            if "boundary" in data.files
            else np.zeros(valid.shape, dtype=bool)
        )
        prototype_ids = (
            data["prototype_ids"].astype(np.int64)
            if "prototype_ids" in data.files
            else np.full(valid.shape, -1, dtype=np.int64)
        )
        hierarchy_prototype_ids = (
            data["hierarchy_prototype_ids"].astype(np.int64)
            if "hierarchy_prototype_ids" in data.files else None
        )
    return {
        "features": torch.from_numpy(features),
        "valid": torch.from_numpy(valid),
        "confidence": torch.from_numpy(confidence),
        "hierarchy_features": (
            torch.from_numpy(hierarchy_features) if hierarchy_features is not None else None
        ),
        "hierarchy_valid": (
            torch.from_numpy(hierarchy_valid) if hierarchy_valid is not None else None
        ),
        "hierarchy_confidence": (
            torch.from_numpy(hierarchy_confidence)
            if hierarchy_confidence is not None else None
        ),
        "importance": torch.from_numpy(importance),
        "detail_weight": torch.from_numpy(detail_weight),
        "boundary": torch.from_numpy(boundary),
        "prototype_ids": torch.from_numpy(prototype_ids),
        "hierarchy_prototype_ids": (
            torch.from_numpy(hierarchy_prototype_ids)
            if hierarchy_prototype_ids is not None else None
        ),
    }


@lru_cache(maxsize=256)
def load_importance_tiers(path):
    """Load the compact three-tier PNG without decompressing semantic tensors."""
    values = np.asarray(Image.open(path).convert("L"), dtype=np.uint8)
    tiers = np.where(values >= 191, 2, np.where(values >= 63, 1, 0)).astype(
        np.uint8
    )
    return torch.from_numpy(tiers)


def select_granularity(supervision, level):
    hierarchy = supervision["hierarchy_features"]
    hierarchy_valid = supervision["hierarchy_valid"]
    if hierarchy is not None and hierarchy_valid is not None and hierarchy_valid[level].any():
        hierarchy_confidence = supervision["hierarchy_confidence"]
        confidence = (
            hierarchy_confidence[level]
            if hierarchy_confidence is not None else supervision["confidence"]
        )
        prototypes = supervision["hierarchy_prototype_ids"]
        prototype_ids = (
            prototypes[level]
            if prototypes is not None else supervision["prototype_ids"]
        )
        return hierarchy[level], hierarchy_valid[level], confidence, prototype_ids
    return (
        supervision["features"], supervision["valid"],
        supervision["confidence"], supervision["prototype_ids"],
    )


def tier_weights(tiers, weights):
    """Map integer tiers 0/1/2 to background/normal/important weights."""
    values = torch.as_tensor(weights, dtype=torch.float32, device=tiers.device)
    return values[tiers.long().clamp(0, 2)]


@torch.no_grad()
def project_tiers_to_gaussians(xyz, camera, tiers, visible_indices):
    """Project visible Gaussian centers and sample their current view's tier."""
    visible_indices = visible_indices.reshape(-1)
    if visible_indices.dtype == torch.bool:
        visible_indices = torch.nonzero(
            visible_indices, as_tuple=False
        ).reshape(-1)
    if visible_indices.numel() == 0:
        return visible_indices, torch.empty(0, device=xyz.device)
    points = xyz[visible_indices]
    homogeneous = torch.cat((points, torch.ones_like(points[:, :1])), dim=1)
    clip = homogeneous @ camera.full_proj_transform
    ndc = clip[:, :3] / clip[:, 3:].clamp_min(1e-7)
    height, width = tiers.shape[-2:]
    x = ((ndc[:, 0] + 1.0) * 0.5 * width).long()
    y = ((1.0 - ndc[:, 1]) * 0.5 * height).long()
    inside = (
        (clip[:, 3] > 0) & (x >= 0) & (x < width) & (y >= 0) & (y < height)
    )
    selected = visible_indices[inside]
    observations = tiers[y[inside], x[inside]].to(
        device=xyz.device, dtype=torch.float32
    ) / 2.0
    return selected, observations


def local_semantic_consistency(gaussians, samples=512, edge_sigma=0.20):
    """Geometry-aware smoothing that avoids bleeding across semantic boundaries."""
    features = gaussians.get_semantic_features
    count = min(int(samples), features.shape[0])
    if count < 2:
        return features.new_zeros(())
    probability = 0.25 + gaussians.importance_score.detach()
    indices = torch.multinomial(probability, count, replacement=False)
    xyz = gaussians.get_xyz[indices]
    distance = torch.cdist(xyz, xyz)
    diagonal = torch.eye(count, dtype=torch.bool, device=distance.device)
    search_distance = distance.masked_fill(diagonal, float("inf"))
    neighbor_column = search_distance.argmin(dim=1)
    neighbors = indices[neighbor_column]
    error = torch.nn.functional.smooth_l1_loss(
        features[indices], features[neighbors], reduction="none"
    ).mean(dim=-1)
    neighbor_distance = distance.gather(1, neighbor_column[:, None]).squeeze(1)
    spatial_scale = neighbor_distance.median()
    edge_weight = torch.exp(
        -neighbor_distance / spatial_scale.clamp_min(1e-7)
    )
    # Detaching this affinity prevents the regularizer from winning by making
    # unrelated object features artificially similar.
    semantic_distance = torch.abs(
        features[indices].detach() - features[neighbors].detach()
    ).mean(dim=-1)
    edge_weight *= torch.exp(
        -semantic_distance / max(float(edge_sigma), 1e-6)
    )
    tier_agreement = 1.0 - torch.abs(
        gaussians.importance_score[indices]
        - gaussians.importance_score[neighbors]
    ).detach()
    edge_weight *= tier_agreement.clamp_min(0.1)
    return (error * edge_weight).sum() / edge_weight.sum().clamp_min(1e-7)
