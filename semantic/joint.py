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


def semantic_chunk_indices(dimensions, chunks_per_step, iteration, validation=False):
    """Keep validation coverage fixed while sampling channels during training."""
    chunks = (int(dimensions) + 2) // 3
    if chunks < 1 or int(chunks_per_step) < 1:
        raise ValueError("semantic dimensions and chunks_per_step must be positive")
    if validation:
        return list(range(chunks))
    first = (int(iteration) * int(chunks_per_step)) % chunks
    return [(first + offset) % chunks for offset in range(min(int(chunks_per_step), chunks))]


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
        region_ids = (
            data["region_ids"].astype(np.int64)
            if "region_ids" in data.files
            else np.full(valid.shape, -1, dtype=np.int64)
        )
        hierarchy_region_ids = (
            data["hierarchy_region_ids"].astype(np.int64)
            if "hierarchy_region_ids" in data.files else None
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
        "region_ids": torch.from_numpy(region_ids),
        "hierarchy_region_ids": (
            torch.from_numpy(hierarchy_region_ids)
            if hierarchy_region_ids is not None else None
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


def select_region_ids(supervision, level):
    hierarchy = supervision["hierarchy_region_ids"]
    if hierarchy is not None and (hierarchy[level] >= 0).any():
        return hierarchy[level]
    return supervision["region_ids"]


def region_boundaries(region_ids, valid):
    """Boundaries for the selected granularity, including its valid-side rim."""
    boundary = torch.zeros_like(valid, dtype=torch.bool)
    horizontal = region_ids[:, 1:] != region_ids[:, :-1]
    vertical = region_ids[1:, :] != region_ids[:-1, :]
    boundary[:, 1:] |= horizontal
    boundary[:, :-1] |= horizontal
    boundary[1:, :] |= vertical
    boundary[:-1, :] |= vertical
    return boundary & valid


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
    # Match cuda_rasterizer/auxiliary.h::ndc2Pix, including pixel centers.
    # COLMAP/rasterizer image Y already points down; flipping it swaps object tiers.
    pixel_x = ((ndc[:, 0] + 1.0) * width - 1.0) * 0.5
    pixel_y = ((ndc[:, 1] + 1.0) * height - 1.0) * 0.5
    x = torch.floor(pixel_x + 0.5).long()
    y = torch.floor(pixel_y + 0.5).long()
    inside = (
        (clip[:, 3] > 0) & torch.isfinite(ndc).all(dim=1)
        & (x >= 0) & (x < width) & (y >= 0) & (y < height)
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
    edge_weight = edge_weight * torch.exp(
        -semantic_distance / max(float(edge_sigma), 1e-6)
    )
    tier_agreement = 1.0 - torch.abs(
        gaussians.importance_score[indices]
        - gaussians.importance_score[neighbors]
    ).detach()
    edge_weight = edge_weight * tier_agreement.clamp_min(0.1)
    return (error * edge_weight).sum() / edge_weight.sum().clamp_min(1e-7)


def boundary_alignment_loss(
    prediction, target, boundary, valid, pixel_weights=None, trusted_background=None,
):
    """Match semantic feature discontinuities to SAM object boundaries.

    The loss supervises both sides of an edge and also discourages false edges
    in mask interiors.  It uses the already rendered feature chunk, so the
    stronger boundary objective does not add another rasterization pass.
    """
    if prediction.ndim != 3 or target.shape != prediction.shape:
        raise ValueError("prediction and target must both be [C,H,W]")
    if min(prediction.shape[-2:]) < 2:
        return prediction.new_zeros(())
    pred_x = torch.abs(prediction[:, :, 1:] - prediction[:, :, :-1]).mean(dim=0)
    pred_y = torch.abs(prediction[:, 1:, :] - prediction[:, :-1, :]).mean(dim=0)
    target_x = torch.abs(target[:, :, 1:] - target[:, :, :-1]).mean(dim=0)
    target_y = torch.abs(target[:, 1:, :] - target[:, :-1, :]).mean(dim=0)
    # Unannotated pixels are not automatically background. A caller may include
    # explicitly verified background only when its target features are available.
    known = valid if trusted_background is None else valid | trusted_background.bool()
    valid_x = known[:, 1:] & known[:, :-1]
    valid_y = known[1:, :] & known[:-1, :]
    boundary_x = boundary[:, 1:] | boundary[:, :-1]
    boundary_y = boundary[1:, :] | boundary[:-1, :]

    def axis_loss(pred_edge, target_edge, axis_valid, axis_boundary, axis_weights):
        if not axis_valid.any():
            return pred_edge.new_zeros(())
        weight = torch.where(
            axis_boundary, torch.full_like(pred_edge, 3.0), torch.ones_like(pred_edge)
        )
        if axis_weights is not None:
            weight = weight * axis_weights
        error = torch.nn.functional.smooth_l1_loss(
            pred_edge, target_edge.detach(), reduction="none", beta=0.05
        )
        return (error[axis_valid] * weight[axis_valid]).sum() / weight[
            axis_valid
        ].sum().clamp_min(1e-7)

    weight_x = None if pixel_weights is None else 0.5 * (
        pixel_weights[:, 1:] + pixel_weights[:, :-1]
    )
    weight_y = None if pixel_weights is None else 0.5 * (
        pixel_weights[1:, :] + pixel_weights[:-1, :]
    )
    return 0.5 * (
        axis_loss(pred_x, target_x, valid_x, boundary_x, weight_x)
        + axis_loss(pred_y, target_y, valid_y, boundary_y, weight_y)
    )


def region_contrastive_loss(
    prediction, region_ids, valid, samples=320, pixel_weights=None,
    positive_margin=0.75, negative_margin=0.50,
):
    """SAGA-style balanced affinity loss over SAM region identities."""
    active = valid & (region_ids >= 0)
    flat_indices = torch.nonzero(active.reshape(-1), as_tuple=False).reshape(-1)
    count = min(int(samples), int(flat_indices.numel()))
    if count < 3:
        return prediction.new_zeros(())
    if pixel_weights is None:
        chosen = flat_indices[torch.randperm(flat_indices.numel(), device=prediction.device)[:count]]
        sample_weight = prediction.new_ones(count)
    else:
        probabilities = pixel_weights.reshape(-1)[flat_indices].detach().clamp_min(1e-5)
        local = torch.multinomial(probabilities, count, replacement=False)
        chosen = flat_indices[local]
        sample_weight = probabilities[local] / probabilities[local].mean().clamp_min(1e-7)
    features = prediction.permute(1, 2, 0).reshape(-1, prediction.shape[0])[chosen]
    features = torch.nn.functional.normalize(features, dim=-1, p=2)
    labels = region_ids.reshape(-1)[chosen]
    similarity = features @ features.T
    upper = torch.triu(
        torch.ones_like(similarity, dtype=torch.bool), diagonal=1
    )
    positive = (labels[:, None] == labels[None, :]) & upper
    negative = (labels[:, None] != labels[None, :]) & upper
    pair_weight = torch.sqrt(sample_weight[:, None] * sample_weight[None, :])
    positive_loss = prediction.new_zeros(())
    negative_loss = prediction.new_zeros(())
    if positive.any():
        values = torch.relu(float(positive_margin) - similarity)
        positive_loss = (values[positive] * pair_weight[positive]).sum() / pair_weight[
            positive
        ].sum().clamp_min(1e-7)
    if negative.any():
        values = torch.relu(similarity - float(negative_margin))
        negative_loss = (values[negative] * pair_weight[negative]).sum() / pair_weight[
            negative
        ].sum().clamp_min(1e-7)
    return positive_loss + negative_loss
