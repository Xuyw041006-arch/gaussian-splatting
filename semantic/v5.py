"""V5 feature objectives: language distillation and independent SAM affinity.

Inspired by the separation of language and affinity in SAGA/LaGa, not an
implementation or reproduction of either paper. Coarse/middle/fine affinities
use nested feature prefixes; contradictory non-nested mask pairs are ignored.
"""

import torch
from torch.nn import functional as F


def normalize_rendered_features(premultiplied, alpha, epsilon=1e-4):
    """Undo alpha compositing without silently detaching the denominator.

    The renderer must either detach geometry in BOTH passes, or differentiate
    BOTH passes. N / stop_gradient(alpha) is a biased joint-geometry gradient.
    """
    return premultiplied / alpha.clamp_min(float(epsilon))


def region_balance_weights(region_ids, valid, power=0.5, cap=8.0):
    """Bounded inverse-area weighting so large masks do not drown thin objects."""
    result = torch.ones_like(region_ids, dtype=torch.float32)
    active = valid & (region_ids >= 0)
    if not active.any() or float(power) == 0:
        return result
    _, inverse, counts = torch.unique(
        region_ids[active], sorted=True, return_inverse=True, return_counts=True,
    )
    area = counts.float()
    # Median area is robust to one near-full-frame background mask. Capping is
    # important: a one-pixel SAM error must not dominate a whole object.
    factors = (area.median().clamp_min(1) / area).pow(float(power))
    factors = factors.clamp(min=1.0 / float(cap), max=float(cap))
    result[active] = factors[inverse]
    return result


def decoded_clip_cosine_loss(
    prediction, target, active, weights, feature_min, feature_range,
    pca_components, pca_mean, max_pixels=2048, deterministic=False,
):
    """Cosine distillation in reconstructed CLIP space, never positive PCA bins.

    PCA truncation is still an approximation. This restores the PCA mean and
    channel ranges before measuring cosine; it is not full CLIP supervision.
    """
    indices = torch.nonzero(active.reshape(-1), as_tuple=False).reshape(-1)
    if not indices.numel():
        return prediction.sum() * 0.0
    if indices.numel() > int(max_pixels):
        if deterministic:
            local = torch.linspace(
                0, indices.numel() - 1, int(max_pixels), device=indices.device,
            ).long()
        else:
            local = torch.randperm(indices.numel(), device=indices.device)[:int(max_pixels)]
        indices = indices[local]
    dimensions = prediction.shape[0]
    predicted = prediction.reshape(dimensions, -1)[:, indices].T
    teacher = target.reshape(dimensions, -1)[:, indices].T
    predicted = (predicted * feature_range + feature_min) @ pca_components + pca_mean
    teacher = (teacher * feature_range + feature_min) @ pca_components + pca_mean
    error = 1.0 - F.cosine_similarity(predicted, teacher.detach(), dim=-1, eps=1e-8)
    pixel_weights = weights.reshape(-1)[indices]
    return (error * pixel_weights).sum() / pixel_weights.sum().clamp_min(1e-8)


def affinity_prefix_dimensions(dimensions):
    """Three nested prefixes; 16 dimensions gives 8 / 12 / 16."""
    dimensions = int(dimensions)
    if dimensions < 4 or dimensions % 4:
        raise ValueError("affinity dimensions must be a positive multiple of four (>=4)")
    return (dimensions // 2, 3 * dimensions // 4, dimensions)


def normalize_affinity_groups(features):
    """Normalize separate residual groups before splatting (LaGa-inspired)."""
    prefixes = affinity_prefix_dimensions(features.shape[-1])
    previous = 0
    groups = []
    for end in prefixes:
        groups.append(F.normalize(features[..., previous:end], dim=-1, eps=1e-8))
        previous = end
    return torch.cat(groups, dim=-1)


def hierarchical_pair_targets(labels):
    """Build non-conflicting affinity targets from [coarse,middle,fine,N] IDs.

    Same in a fine mask implies same at every coarser scale; different in a
    coarse mask implies different at every finer scale. Unknown (-1) is never
    a negative. When imperfect SAM masks violate containment, contradictory
    positive/negative pairs are ignored instead of fighting each other.
    """
    if labels.ndim != 2 or labels.shape[0] != 3:
        raise ValueError("labels must have shape [3,N] in coarse/middle/fine order")
    known = (labels[:, :, None] >= 0) & (labels[:, None, :] >= 0)
    same = known & (labels[:, :, None] == labels[:, None, :])
    different = known & ~same
    positive = torch.stack([same[level:].any(dim=0) for level in range(3)])
    negative = torch.stack([different[:level + 1].any(dim=0) for level in range(3)])
    conflict = positive & negative
    upper = torch.triu(torch.ones_like(positive[0]), diagonal=1)
    return positive & ~conflict & upper, negative & ~conflict & upper, conflict & upper


def hierarchical_affinity_loss(
    prediction, hierarchy_ids, valid, weights=None, samples=320,
    positive_margin=0.8, negative_margin=0.2, norm_weight=0.05,
):
    """Object-balanced, scale-consistent pair loss on an independent field.

    Language features are deliberately NOT accepted: distinct SAM instances
    may have identical text meaning. Pushing their CLIP vectors apart hurts
    open-vocabulary retrieval. This loss has no global cross-view clustering.
    """
    prefixes = affinity_prefix_dimensions(prediction.shape[0])
    if hierarchy_ids.shape != (3, *valid.shape):
        raise ValueError("hierarchy_ids must be [3,H,W] matching valid")
    active = valid & (hierarchy_ids >= 0).any(dim=0)
    if weights is not None:
        active &= weights > 0
    indices = torch.nonzero(active.reshape(-1), as_tuple=False).reshape(-1)
    count = min(int(samples), indices.numel())
    zero = prediction.sum() * 0.0
    if count < 3:
        return zero, {"pairs": 0, "conflicting_pairs": 0}
    sampling = torch.zeros_like(valid, dtype=prediction.dtype)
    for ids in hierarchy_ids:
        sampling += region_balance_weights(ids, active) * (ids >= 0)
    if weights is not None:
        sampling *= weights.detach().clamp_min(0)
    probability = sampling.reshape(-1)[indices].clamp_min(1e-8)
    chosen = indices[torch.multinomial(probability, count, replacement=False)]
    labels = hierarchy_ids.reshape(3, -1)[:, chosen]
    positive, negative, conflicts = hierarchical_pair_targets(labels)
    sampled = prediction.reshape(prediction.shape[0], -1)[:, chosen].T
    losses = []
    previous = 0
    norm_losses = []
    for level, end in enumerate(prefixes):
        feature = F.normalize(sampled[:, :end], dim=-1, eps=1e-8)
        similarity = feature @ feature.T
        positive_loss = (
            F.relu(float(positive_margin) - similarity[positive[level]]).mean()
            if positive[level].any() else zero
        )
        negative_loss = (
            F.relu(similarity[negative[level]] - float(negative_margin)).mean()
            if negative[level].any() else zero
        )
        losses.append(positive_loss + negative_loss)
        norm_losses.append((sampled[:, previous:end].norm(dim=-1) - 1.0).square().mean())
        previous = end
    return torch.stack(losses).mean() + float(norm_weight) * torch.stack(norm_losses).mean(), {
        "pairs": int((positive | negative).sum().item()),
        "conflicting_pairs": int(conflicts.sum().item()),
    }
