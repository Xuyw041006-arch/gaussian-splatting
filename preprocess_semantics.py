"""Extract SAM regions, CLIP semantics, PCA maps, and importance masks."""

import hashlib
import json
import os
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
from PIL import Image

from semantic.inventory import (
    normalize_label,
    parse_inventory_config,
    rank_scene_inventory,
)


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
IMPORTANCE_NEGATIVE_PROMPTS = ("object", "things", "stuff", "texture")
COMPETITIVE_IMPORTANCE_DEFAULTS = {
    "min_cosine": 0.25, "min_relevancy": 0.60, "min_margin": 0.04,
    "temperature": 10.0, "max_important_area": 0.80,
}


def validate_importance_prompts(tiers):
    """Deduplicate labels and reject contradictory user/LLM tier assignments."""
    normalized = {tier: sorted({normalize_label(label) for label in tiers.get(tier, [])
                                if normalize_label(label)})
                  for tier in ("background", "normal", "important")}
    owners = {}
    for tier, labels in normalized.items():
        for label in labels:
            if label in owners:
                raise ValueError(f"Importance label {label!r} occurs in both {owners[label]} and {tier}")
            owners[label] = tier
    return normalized


def importance_policy_fingerprint(configuration):
    serialized = json.dumps(configuration, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def competitive_importance(features, tier_text, negative_text, area_ratios, **overrides):
    """Conservative tier decisions from raw normalized CLIP, not SAM quality.

    Relevancy is a contrastive score, NOT a calibrated correctness probability.
    No feature array is modified; unknowns and competing matches stay normal.
    """
    config = {**COMPETITIVE_IMPORTANCE_DEFAULTS, **overrides}
    features = np.asarray(features)
    count = len(features)
    scores = np.full((count, 3), -1.0, dtype=np.float32)
    available = np.zeros(3, dtype=bool)
    for tier_id, tier in enumerate(("background", "normal", "important")):
        text = tier_text.get(tier)
        if text is not None and len(text):
            values = features @ np.asarray(text).T
            scores[:, tier_id] = np.max(np.where(np.isfinite(values), values, -1.0), axis=1)
            available[tier_id] = True
    if negative_text is None or not len(negative_text):
        raise ValueError("competitive importance requires explicit generic negative descriptors")
    negative_values = features @ np.asarray(negative_text).T
    negative_scores = np.max(np.where(np.isfinite(negative_values), negative_values, 1.0), axis=1)
    relevancy = 1.0 / (1.0 + np.exp(np.clip(
        -config["temperature"] * (scores - negative_scores[:, None]), -80, 80)))
    margins = np.stack([scores[:, i] - np.max(scores[:, [j for j in range(3) if j != i]], axis=1)
                        for i in range(3)], axis=1)
    finite = np.isfinite(features).all(axis=1) & (np.linalg.norm(features, axis=1) > 1e-8)
    supported = (available[None, :] & finite[:, None]
                 & (scores >= config["min_cosine"])
                 & (relevancy >= config["min_relevancy"])
                 & (margins >= config["min_margin"]))
    tiers = np.ones(count, dtype=np.uint8)
    reasons = np.full(count, "normal_unconfirmed", dtype="U32")
    reasons[supported[:, 1]] = "normal_supported"
    near_tie = ((scores.max(axis=1) >= config["min_cosine"])
                & (margins.max(axis=1) < config["min_margin"]))
    reasons[near_tie] = "normal_ambiguous"
    large = np.asarray(area_ratios) >= config["max_important_area"]
    reasons[supported[:, 2] & large] = "normal_large_region"
    for tier_id, reason in ((0, "background_supported"), (2, "important_supported")):
        selected = supported[:, tier_id] & (~large if tier_id == 2 else True)
        tiers[selected], reasons[selected] = tier_id, reason
    reasons[~finite] = "normal_invalid_features"
    return {"tiers": tiers, "scores": scores, "margins": margins,
            "relevancy": relevancy.astype(np.float32), "reasons": reasons}


def image_files(directory):
    return sorted(
        path for path in Path(directory).iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def masked_crop(rgb, mask, bbox=None):
    """Keep the whole region through CLIP's Resize + CenterCrop transform.

    LangSplat/LaGa pad masked crops before encoding. A raw elongated rectangle
    would lose its ends in the standard OpenCLIP center crop. Derive inclusive
    bounds from the mask so a one-pixel-wide region never makes an empty crop.
    """
    ys, xs = np.nonzero(mask)
    if not len(xs):
        raise ValueError("Cannot encode an empty SAM mask")
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    crop = rgb[y0:y1, x0:x1].copy()
    crop_mask = mask[y0:y1, x0:x1]
    crop[~crop_mask] = 255
    side = max(crop.shape[:2])
    padded = np.full((side, side, 3), 255, dtype=np.uint8)
    y, x = (side - crop.shape[0]) // 2, (side - crop.shape[1]) // 2
    padded[y:y + crop.shape[0], x:x + crop.shape[1]] = crop
    return Image.fromarray(padded)


def encode_regions(model, preprocess, rgb, regions, device, batch_size):
    import torch
    crops = [masked_crop(rgb, region["segmentation"], region["bbox"]) for region in regions]
    outputs = []
    with torch.no_grad():
        for start in range(0, len(crops), batch_size):
            batch = torch.stack([preprocess(image) for image in crops[start:start + batch_size]])
            batch = batch.to(
                device, dtype=torch.float16 if device.type == "cuda" else torch.float32
            )
            features = model.encode_image(batch)
            features = torch.nn.functional.normalize(features.float(), dim=-1, p=2)
            outputs.append(features.cpu().numpy())
    return np.concatenate(outputs, axis=0).astype(np.float32)


def resize_mask(mask, size):
    """Keep thin-mask pixel coverage when downsampling instead of dropping it."""
    mask = np.asarray(mask, dtype=bool)
    if size[0] < mask.shape[1] or size[1] < mask.shape[0]:
        # Positive box occupancy conserves subpixel structures. The resulting
        # rasterized boundary can expand by one feature pixel; record this rule.
        return np.asarray(Image.fromarray(mask.astype(np.float32)).resize(size, Image.Resampling.BOX)) > 0
    return np.asarray(Image.fromarray(mask.astype(np.uint8)).resize(size, Image.Resampling.NEAREST)) > 0


def select_balanced_regions(regions, max_masks, image_area, fine_ratio=0.05, coarse_ratio=0.25):
    """Reserve mask capacity across scales instead of discarding all small masks."""
    if max_masks < 1:
        raise ValueError("max_masks must be positive")
    ranked = sorted(regions, key=lambda region: (
        float(region.get("predicted_iou", 1)) * float(region.get("stability_score", 1)),
        -float(region["area"]),
    ), reverse=True)
    groups = [[], [], []]
    for index, region in enumerate(ranked):
        ratio = float(region["area"]) / max(image_area, 1)
        groups[2 if ratio < fine_ratio else 1 if ratio < coarse_ratio else 0].append(index)
    selected = set()
    quota = max_masks // 3
    for group in groups:
        selected.update(group[:quota])
    for index in range(len(ranked)):
        if len(selected) >= max_masks:
            break
        selected.add(index)
    return sorted((ranked[index] for index in selected), key=lambda region: region["area"], reverse=True)


def build_region_map(regions, size, indices=None):
    region_map = np.full((size[1], size[0]), -1, dtype=np.int16)
    # Broad regions are assigned first; smaller objects overwrite them.
    if indices is None:
        indices = range(len(regions))
    for index in sorted(indices, key=lambda value: regions[value]["area"], reverse=True):
        region = regions[index]
        region_map[resize_mask(region["segmentation"], size)] = index
    return region_map


def mask_containment_parents(masks, containment=0.90, minimum_growth=1.15):
    """Find each region's smallest genuinely containing larger region."""
    masks = np.asarray(masks, dtype=bool)
    flat = masks.reshape(len(masks), -1)
    areas = flat.sum(axis=1)
    parents = np.full(len(masks), -1, dtype=np.int32)
    ascending = np.argsort(areas, kind="stable")
    for child in ascending:
        if areas[child] == 0:
            continue
        pixels = np.flatnonzero(flat[child])
        for parent in ascending:
            if areas[parent] < areas[child] * minimum_growth:
                continue
            if flat[parent, pixels].sum() / areas[child] >= containment:
                parents[child] = parent
                break
    return parents


def build_hierarchy_region_maps(regions, size, image_area, fine_ratio=0.05, coarse_ratio=0.25,
                                method="containment", return_parents=False):
    """Build nested whole/part/subpart targets from mask containment.

    This is an image-mask hierarchy inspired by LangSplat/LaGa, not SAGA's 3D
    physical-scale gate. Area bins remain available only for a legacy ablation.
    """
    if method == "containment":
        masks = np.stack([resize_mask(region["segmentation"], size) for region in regions])
        parents = mask_containment_parents(masks)
        fine = build_region_map(regions, size)
        ancestors = np.zeros((3, len(regions)), dtype=np.int32)
        for index in range(len(regions)):
            chain = [index]
            while parents[chain[-1]] >= 0:
                chain.append(int(parents[chain[-1]]))
            ancestors[:, index] = [chain[-1], chain[len(chain) // 2], index]
        maps = np.full((3, *fine.shape), -1, dtype=np.int16)
        valid = fine >= 0
        ys, xs = np.nonzero(valid)
        for level in range(3):
            assigned = ancestors[level, fine[valid]]
            # A near-containment relation may exclude a few edge pixels. Never
            # supervise them with a parent mask that does not actually cover it.
            maps[level, valid] = np.where(masks[assigned, ys, xs], assigned, fine[valid])
        return (maps, parents) if return_parents else maps
    if method != "area":
        raise ValueError("Unknown hierarchy method")
    levels = [[], [], []]
    for index, region in enumerate(regions):
        ratio = float(region["area"]) / max(float(image_area), 1.0)
        level = 2 if ratio < fine_ratio else (1 if ratio < coarse_ratio else 0)
        levels[level].append(index)
    maps = np.stack(
        [build_region_map(regions, size, indices) for indices in levels], axis=0
    )
    return (maps, np.full(len(regions), -1, dtype=np.int32)) if return_parents else maps


def supported_prototype_assignments(features, centers, labels, fit_mask, view_ids=None,
                                    min_similarity=0.0, min_margin=0.0, min_views=1):
    """Reject ambiguous/single-view appearance pooling without selecting a label."""
    similarity = np.sum(features * centers[labels], axis=1)
    supported = similarity >= min_similarity
    if min_margin > 0:
        similarities = features @ centers.T
        similarities[np.arange(len(features)), labels] = -np.inf
        supported &= similarity - similarities.max(axis=1) >= min_margin
    if min_views > 1:
        if view_ids is None or len(view_ids) != len(features):
            raise ValueError("Conservative prototype pooling requires per-region view identities")
        view_ids = np.asarray(view_ids)
        view_counts = np.array([len(np.unique(view_ids[(labels == cluster) & fit_mask]))
                                for cluster in range(len(centers))])
        supported &= view_counts[labels] >= min_views
    return supported


def aggregate_cross_view_features(
    features, confidences, max_prototypes=64, weight=0.65,
    return_centers=False, fit_mask=None, view_ids=None,
    min_similarity=0.0, min_margin=0.0, min_views=1, max_blend=1.0,
):
    """Fit scene prototypes on training regions, then transform every view."""
    features = np.asarray(features, dtype=np.float32)
    confidences = np.asarray(confidences, dtype=np.float32)
    fit_mask = np.ones(len(features), dtype=bool) if fit_mask is None else np.asarray(fit_mask, dtype=bool)
    if fit_mask.shape != (len(features),) or not fit_mask.any():
        raise ValueError("Prototype fitting needs a nonempty training-region mask")
    fit_features = features[fit_mask]
    if len(fit_features) < 2 or max_prototypes < 1 or weight <= 0:
        result = (
            features.copy(), np.full(len(features), -1, dtype=np.int32),
            np.zeros(len(features), dtype=np.float32),
        )
        if return_centers:
            centers = fit_features[:1].copy()
            return (*result, centers)
        return result
    from sklearn.cluster import MiniBatchKMeans
    clusters = min(int(max_prototypes), len(fit_features), max(2, int(round(np.sqrt(len(fit_features))))))
    model = MiniBatchKMeans(
        n_clusters=clusters, random_state=42,
        batch_size=min(2048, len(fit_features)), n_init=3,
    )
    model.fit(fit_features)
    labels = model.predict(features)
    centers = model.cluster_centers_.astype(np.float32)
    centers /= np.maximum(np.linalg.norm(centers, axis=1, keepdims=True), 1e-8)
    similarity = np.sum(features * centers[labels], axis=1)
    compactness = np.zeros(clusters, dtype=np.float32)
    for cluster in range(clusters):
        selected = similarity[(labels == cluster) & fit_mask]
        compactness[cluster] = max(
            float(selected.mean()) if len(selected) else 0.0, 0.0
        )
    blend = float(weight) * np.clip(
        similarity * compactness[labels] * confidences, 0.0, 1.0
    )
    blend = np.minimum(blend, float(max_blend))
    supported = supported_prototype_assignments(
        features, centers, labels, fit_mask, view_ids, min_similarity, min_margin, min_views,
    )
    blend[~supported] = 0
    aggregated = (1.0 - blend[:, None]) * features + blend[:, None] * centers[labels]
    aggregated /= np.maximum(np.linalg.norm(aggregated, axis=1, keepdims=True), 1e-8)
    safe_labels = labels.astype(np.int32)
    safe_labels[~supported] = -1
    result = (
        aggregated.astype(np.float32), safe_labels,
        blend.astype(np.float32),
    )
    return (*result, centers) if return_centers else result


def fit_training_pca(features, dimensions, fit_mask):
    """Fit PCA and encoding bounds without incorporating held-out descriptors."""
    from sklearn.decomposition import PCA

    features = np.asarray(features, dtype=np.float32)
    fit_mask = np.asarray(fit_mask, dtype=bool)
    if fit_mask.shape != (len(features),) or not fit_mask.any():
        raise ValueError("PCA fitting needs a nonempty training-region mask")
    pca = PCA(n_components=int(dimensions), random_state=42)
    pca.fit(features[fit_mask])
    projected = pca.transform(features)
    fit_projected = projected[fit_mask]
    return pca, projected, fit_projected.min(axis=0), fit_projected.max(axis=0)


def _expand_binary(mask, iterations):
    """Small dependency-free 8-neighbour dilation for feature-resolution maps."""
    expanded = np.asarray(mask, dtype=bool)
    for _ in range(max(0, int(iterations))):
        padded = np.pad(expanded, 1, mode="constant")
        expanded = np.logical_or.reduce([
            padded[y:y + expanded.shape[0], x:x + expanded.shape[1]]
            for y in range(3) for x in range(3)
        ])
    return expanded


def build_detail_supervision(
    region_map, importance, boundary_width=2, boundary_boost=1.75,
    thin_boost=1.25, thin_compactness=0.35, thin_aspect_ratio=3.0,
    promote_importance=True,
):
    """Create boundary/thin-object weights and promote their densification tier.

    Thinness combines region compactness and bounding-box aspect ratio.  The
    implementation intentionally uses NumPy only so preprocessing behaves the
    same in Colab and minimal local test environments.
    """
    region_map = np.asarray(region_map)
    enhanced = np.asarray(importance, dtype=np.uint8).copy()
    height, width = region_map.shape
    boundary = np.zeros((height, width), dtype=bool)
    boundary[:-1] |= region_map[:-1] != region_map[1:]
    boundary[1:] |= region_map[:-1] != region_map[1:]
    boundary[:, :-1] |= region_map[:, :-1] != region_map[:, 1:]
    boundary[:, 1:] |= region_map[:, :-1] != region_map[:, 1:]
    boundary &= region_map >= 0
    boundary = _expand_binary(boundary, max(0, int(boundary_width) - 1))
    boundary &= region_map >= 0

    thinness = np.zeros((height, width), dtype=np.float32)
    for region_id in np.unique(region_map[region_map >= 0]):
        mask = region_map == region_id
        ys, xs = np.nonzero(mask)
        area = float(len(xs))
        if area < 2:
            score = 1.0
        else:
            box_height = float(ys.max() - ys.min() + 1)
            box_width = float(xs.max() - xs.min() + 1)
            aspect = max(box_height, box_width) / max(min(box_height, box_width), 1.0)
            perimeter = float(
                np.count_nonzero(mask[:-1] != mask[1:])
                + np.count_nonzero(mask[:, :-1] != mask[:, 1:])
                + 2 * np.count_nonzero(mask[0]) + 2 * np.count_nonzero(mask[:, 0])
            )
            compactness = 4.0 * np.pi * area / max(perimeter * perimeter, 1.0)
            compactness_score = np.clip(
                (float(thin_compactness) - compactness)
                / max(float(thin_compactness), 1e-6), 0.0, 1.0
            )
            aspect_score = np.clip(
                (aspect - float(thin_aspect_ratio))
                / max(2.0 * float(thin_aspect_ratio), 1e-6), 0.0, 1.0
            )
            score = float(max(compactness_score, aspect_score))
        if score > 0:
            thinness[mask] = score
            if promote_importance:
                enhanced[mask] = np.maximum(enhanced[mask], 1)

    # Preserve both sides of meaningful object boundaries.  Important-object
    # boundaries remain tier 2; other SAM boundaries become at least tier 1.
    important_nearby = _expand_binary(importance >= 2, boundary_width) & boundary
    if promote_importance:
        enhanced[boundary] = np.maximum(enhanced[boundary], 1)
        enhanced[important_nearby] = 2
    tier_scale = 0.65 + 0.175 * enhanced.astype(np.float32)
    detail_weight = (
        1.0
        + float(boundary_boost) * boundary.astype(np.float32) * tier_scale
        + float(thin_boost) * thinness * tier_scale
    )
    return (
        detail_weight.astype(np.float32), boundary,
        thinness.astype(np.float32), enhanced,
    )


def select_prompt_regions(features, text_features, threshold, topk):
    """Select supported matches; top-k caps matches and never bypasses confidence."""
    if text_features is None or len(text_features) == 0 or len(features) == 0:
        return set()
    similarities = features @ text_features.T
    chosen = set()
    for prompt_index in range(text_features.shape[0]):
        scores = similarities[:, prompt_index]
        supported = np.flatnonzero(np.isfinite(scores) & (scores >= threshold))
        if topk > 0 and len(supported) > topk:
            supported = supported[np.argsort(scores[supported])[-int(topk):]]
        chosen.update(supported.tolist())
    return chosen


def region_confidences(regions, power=0.5, floor=0.05):
    """Combine SAM's mask-quality signals into stable supervision weights."""
    values = []
    for region in regions:
        predicted_iou = float(np.clip(region.get("predicted_iou", 1.0), 0.0, 1.0))
        stability = float(np.clip(region.get("stability_score", 1.0), 0.0, 1.0))
        values.append(max(floor, (predicted_iou * stability) ** power))
    return np.asarray(values, dtype=np.float32)


def parse_prompts(value):
    return [item.strip() for item in value.split(",") if item.strip()]


def load_inventory_candidates(repo, inline="", path=""):
    candidate_path = Path(path) if path else repo / "assets" / "scene_vocabulary.txt"
    labels = parse_prompts(inline)
    if candidate_path.is_file():
        labels.extend(
            line.strip() for line in candidate_path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        )
    return sorted({normalize_label(label) for label in labels if normalize_label(label)})


def main():
    import torch
    from tqdm import tqdm

    parser = ArgumentParser(description="Prepare semantic supervision for 3DGS")
    parser.add_argument("--scene", required=True, help="COLMAP scene root containing images/")
    parser.add_argument("--images_subdir", default="images")
    parser.add_argument(
        "--fit_exclude_list", default="",
        help="Newline-delimited held-out image names: transform them but exclude from prototype/PCA fitting",
    )
    parser.add_argument("--sam_checkpoint", required=True)
    parser.add_argument("--sam_model", choices=["vit_b", "vit_l", "vit_h"], default="vit_h")
    parser.add_argument("--clip_model", default="ViT-H-14")
    parser.add_argument("--clip_pretrained", default="laion2b_s32b_b79k")
    parser.add_argument("--feature_dim", type=int, default=32)
    parser.add_argument("--feature_width", type=int, default=320)
    parser.add_argument("--min_mask_area", type=int, default=100)
    parser.add_argument("--max_masks", type=int, default=128)
    parser.add_argument("--points_per_side", type=int, default=24)
    parser.add_argument("--sam_crop_n_layers", type=int, default=1,
                        help="SAM crop pyramid depth; 1 improves small-region proposals at additional preprocessing cost")
    parser.add_argument("--mask_selection", choices=["balanced", "largest"], default="balanced")
    parser.add_argument("--hierarchy_method", choices=["containment", "area"], default="containment")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument(
        "--sam_confidence_power", type=float, default=0.5,
        help="Exponent applied to predicted-IoU × stability mask weights",
    )
    parser.add_argument("--sam_confidence_floor", type=float, default=0.05)
    parser.add_argument(
        "--important", default="",
        help="Comma-separated user/LLM-selected object names, for example apple,cup",
    )
    parser.add_argument(
        "--important_json", default="",
        help="Optional JSON mapping image filename/stem to an LLM-produced list of important objects",
    )
    parser.add_argument(
        "--importance_config", default="",
        help="User/LLM scene inventory JSON with background/normal/important tiers",
    )
    parser.add_argument("--normal", default="", help="Normal-priority object prompts")
    parser.add_argument("--background", default="", help="Explicit background object/material prompts; unknown pixels are not confirmed background")
    parser.add_argument(
        "--normal_json", default="",
        help="Optional JSON mapping image filename/stem to normal-priority objects",
    )
    parser.add_argument("--importance_threshold", type=float, default=0.24)
    parser.add_argument("--importance_policy", choices=["legacy", "competitive_v1"], default="legacy",
                        help="Competitive tiers require semantic support; ambiguous regions remain normal")
    parser.add_argument(
        "--importance_topk", type=int, default=0,
        help="Maximum above-threshold regions per prompt; 0 keeps all supported matches",
    )
    parser.add_argument("--fine_area_ratio", type=float, default=0.05)
    parser.add_argument("--coarse_area_ratio", type=float, default=0.25)
    parser.add_argument("--background_area_ratio", type=float, default=0.80)
    parser.add_argument("--cross_view_prototypes", type=int, default=96)
    parser.add_argument("--cross_view_weight", type=float, default=0.72)
    parser.add_argument("--prototype_mode", choices=["conservative", "off", "legacy"], default="conservative",
                        help="Conservative appearance pooling is not geometric object correspondence")
    parser.add_argument("--boundary_width", type=int, default=3)
    parser.add_argument("--boundary_boost", type=float, default=2.25)
    parser.add_argument("--thin_boost", type=float, default=1.50)
    parser.add_argument("--thin_compactness", type=float, default=0.40)
    parser.add_argument("--thin_aspect_ratio", type=float, default=2.5)
    parser.add_argument(
        "--inventory_candidates", default="",
        help="Extra comma-separated object words for SAM-region CLIP discovery",
    )
    parser.add_argument(
        "--inventory_candidates_file", default="",
        help="Optional newline-delimited object vocabulary; defaults to assets/scene_vocabulary.txt",
    )
    parser.add_argument("--inventory_threshold", type=float, default=0.22)
    parser.add_argument("--inventory_topk_per_region", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if args.feature_dim < 3:
        parser.error("--feature_dim must be at least 3")
    if args.feature_width < 32 or args.max_masks < 1 or args.batch_size < 1:
        parser.error("feature width, max masks, and batch size must be positive")
    if args.sam_crop_n_layers < 0 or args.max_masks > np.iinfo(np.int16).max:
        parser.error("SAM crop layers must be nonnegative and max_masks must fit int16 region IDs")
    if args.importance_topk < 0:
        parser.error("--importance_topk must be >= 0")
    if not 0 < args.fine_area_ratio < args.coarse_area_ratio < args.background_area_ratio <= 1:
        parser.error("area ratios must satisfy 0 < fine < coarse < background <= 1")
    if args.cross_view_prototypes < 1 or not 0 <= args.cross_view_weight <= 1:
        parser.error("cross-view prototypes must be positive and weight must be in [0, 1]")
    if args.sam_confidence_power <= 0 or not 0 <= args.sam_confidence_floor <= 1:
        parser.error("SAM confidence power must be positive and floor must be in [0, 1]")
    if (
        args.boundary_width < 1 or args.boundary_boost < 0 or args.thin_boost < 0
        or args.thin_compactness <= 0 or args.thin_aspect_ratio <= 1
    ):
        parser.error("Boundary/thin-object parameters are invalid")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        parser.error("CUDA was requested but is unavailable")
    scene = Path(args.scene).resolve()
    images_dir = scene / args.images_subdir
    checkpoint = Path(args.sam_checkpoint).resolve()
    if not images_dir.is_dir():
        parser.error(f"Missing image directory: {images_dir}")
    if not checkpoint.is_file():
        parser.error(f"Missing SAM checkpoint: {checkpoint}")
    paths = image_files(images_dir)
    if not paths:
        parser.error(f"No supported images in {images_dir}")
    excluded_names = set()
    if args.fit_exclude_list:
        exclude_path = Path(args.fit_exclude_list).resolve()
        if not exclude_path.is_file():
            parser.error(f"Missing fitting exclusion list: {exclude_path}")
        excluded_names = {
            Path(line.strip()).name for line in exclude_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
        known_names = {value for path in paths for value in (path.name, path.stem)}
        if excluded_names - known_names:
            parser.error("Unknown held-out images in fitting exclusion list: " + ", ".join(sorted(excluded_names - known_names)))
    fit_image_names = [
        path.name for path in paths
        if path.name not in excluded_names and path.stem not in excluded_names
    ]
    if not fit_image_names:
        parser.error("Fitting exclusions leave no training images")
    fit_image_set = set(fit_image_names)
    heldout_image_names = [path.name for path in paths if path.name not in fit_image_set]

    try:
        import open_clip
        from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
        from sklearn.decomposition import PCA
    except ImportError as error:
        parser.error(
            f"Missing semantic dependency ({error}). Install requirements-semantic.txt."
        )

    device = torch.device(args.device)
    precision = "fp16" if device.type == "cuda" else "fp32"
    clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
        args.clip_model, pretrained=args.clip_pretrained, precision=precision
    )
    clip_model = clip_model.eval().to(device)
    tokenizer = open_clip.get_tokenizer(args.clip_model)
    sam = sam_model_registry[args.sam_model](checkpoint=str(checkpoint)).to(device)
    generator = SamAutomaticMaskGenerator(
        sam, points_per_side=args.points_per_side,
        pred_iou_thresh=0.7, stability_score_thresh=0.85,
        crop_n_layers=args.sam_crop_n_layers, crop_n_points_downscale_factor=1,
        min_mask_region_area=args.min_mask_area,
    )

    raw_dir = scene / "semantic_raw"
    maps_dir = scene / "semantic_maps"
    importance_dir = scene / "importance_masks"
    detail_dir = scene / "detail_weights"
    boundary_dir = scene / "boundary_masks"
    raw_dir.mkdir(exist_ok=True)
    maps_dir.mkdir(exist_ok=True)
    importance_dir.mkdir(exist_ok=True)
    detail_dir.mkdir(exist_ok=True)
    boundary_dir.mkdir(exist_ok=True)

    prompts = parse_prompts(args.important)
    normal_prompts = parse_prompts(args.normal)
    background_prompts = parse_prompts(args.background)
    selected_inventory_tiers = {}
    if args.importance_config:
        with open(args.importance_config, encoding="utf-8") as handle:
            tier_config = parse_inventory_config(json.load(handle))
        prompts = sorted(set(prompts + tier_config["important"]))
        normal_prompts = sorted(set(normal_prompts + tier_config["normal"]))
        background_prompts = sorted(set(background_prompts + tier_config["background"]))
        selected_inventory_tiers = {
            label: tier for tier, labels in tier_config.items() for label in labels
        }
    prompt_map = {}
    normal_prompt_map = {}
    if args.important_json:
        with open(args.important_json, encoding="utf-8") as handle:
            prompt_map = json.load(handle)
        if not isinstance(prompt_map, dict):
            parser.error("--important_json must contain a JSON object")
    if args.normal_json:
        with open(args.normal_json, encoding="utf-8") as handle:
            normal_prompt_map = json.load(handle)
        if not isinstance(normal_prompt_map, dict):
            parser.error("--normal_json must contain a JSON object")
    prompt_feature_cache = {}

    def prompts_for(mapping, path, defaults, field=None):
        value = mapping.get(path.name, mapping.get(path.stem, defaults))
        if isinstance(value, dict):
            value = value.get(field, defaults)
        if isinstance(value, str):
            value = parse_prompts(value)
        if not isinstance(value, list):
            raise ValueError(f"Prompts for {path.name} must be a list or comma string")
        return [str(item).strip() for item in value if str(item).strip()]

    def features_for_prompts(current_prompts):
        key = tuple(current_prompts)
        if not key:
            return None
        if key not in prompt_feature_cache:
            with torch.no_grad():
                tokens = tokenizer(list(key)).to(device)
                prompt_feature_cache[key] = torch.nn.functional.normalize(
                    clip_model.encode_text(tokens).float(), dim=-1, p=2
                ).cpu().numpy()
        return prompt_feature_cache[key]

    all_features = []
    records = []
    for path in tqdm(paths, desc="SAM + CLIP"):
        normal_mapping = normal_prompt_map if normal_prompt_map else prompt_map
        current_tier_prompts = {
            "important": prompts_for(prompt_map, path, prompts, "important"),
            "normal": prompts_for(normal_mapping, path, normal_prompts, "normal"),
            "background": background_prompts,
        }
        if args.importance_policy == "competitive_v1":
            current_tier_prompts = validate_importance_prompts(current_tier_prompts)
        rgb = np.asarray(Image.open(path).convert("RGB"))
        regions = generator.generate(rgb)
        regions = [region for region in regions if region["area"] >= args.min_mask_area]
        proposal_count = len(regions)
        if args.mask_selection == "balanced":
            regions = select_balanced_regions(
                regions, args.max_masks, rgb.shape[0] * rgb.shape[1],
                args.fine_area_ratio, args.coarse_area_ratio,
            )
        else:
            regions = sorted(regions, key=lambda region: region["area"], reverse=True)[:args.max_masks]
        if not regions:
            raise RuntimeError(f"SAM found no regions in {path}")

        features = encode_regions(
            clip_model, clip_preprocess, rgb, regions, device, args.batch_size
        )
        confidences = region_confidences(
            regions, args.sam_confidence_power, args.sam_confidence_floor
        )
        scale = args.feature_width / rgb.shape[1]
        feature_size = (args.feature_width, max(1, round(rgb.shape[0] * scale)))
        region_map = build_region_map(regions, feature_size)
        hierarchy_region_maps, parent_ids = build_hierarchy_region_maps(
            regions, feature_size, rgb.shape[0] * rgb.shape[1],
            args.fine_area_ratio, args.coarse_area_ratio,
            method=args.hierarchy_method, return_parents=True,
        )
        region_masks = np.stack([resize_mask(region["segmentation"], feature_size) for region in regions])
        raw_path = raw_dir / f"{path.stem}.npz"
        np.savez_compressed(
            raw_path,
            region_map=region_map,
            hierarchy_region_maps=hierarchy_region_maps,
            features=features.astype(np.float16),
            confidences=confidences.astype(np.float16),
            packed_region_masks=np.packbits(region_masks.reshape(len(regions), -1), axis=1),
            mask_shape=np.asarray(region_map.shape, dtype=np.int32),
            parent_ids=parent_ids,
            image_name=np.asarray(path.name),
            original_image_shape=np.asarray(rgb.shape[:2], dtype=np.int32),
            proposal_count=np.asarray(proposal_count),
            area_ratios=np.asarray(
                [region["area"] / (rgb.shape[0] * rgb.shape[1]) for region in regions],
                dtype=np.float32,
            ),
        )
        all_features.append(features)
        records.append({
            "path": path,
            "raw_path": raw_path,
            "features": features,
            "confidences": confidences,
            "area_ratios": np.asarray(
                [region["area"] / (rgb.shape[0] * rgb.shape[1]) for region in regions],
                dtype=np.float32,
            ),
            "important_prompts": current_tier_prompts["important"],
            "normal_prompts": current_tier_prompts["normal"],
            "background_prompts": current_tier_prompts["background"],
        })

    stacked_raw = np.concatenate(all_features, axis=0)
    fit_mask = np.concatenate([
        np.full(len(record["features"]), record["path"].name in fit_image_set, dtype=bool)
        for record in records
    ])
    repo = Path(__file__).resolve().parent
    candidate_labels = load_inventory_candidates(
        repo, args.inventory_candidates, args.inventory_candidates_file
    )
    candidate_labels = sorted(set(
        candidate_labels + prompts + normal_prompts + background_prompts
    ))
    inventory = []
    if candidate_labels:
        candidate_text = features_for_prompts(candidate_labels)
        view_ids = []
        region_areas = []
        for record in records:
            view_ids.extend([record["path"].name] * len(record["features"]))
            region_areas.extend(record["area_ratios"].tolist())
        inventory = rank_scene_inventory(
            stacked_raw[fit_mask], candidate_text, candidate_labels,
            np.asarray(view_ids)[fit_mask].tolist(),
            np.asarray(region_areas)[fit_mask], args.inventory_threshold, args.inventory_topk_per_region,
        )
        for item in inventory:
            item["selected_tier"] = selected_inventory_tiers.get(item["label"])
        with open(scene / "scene_inventory.json", "w", encoding="utf-8") as handle:
            json.dump({
                "version": 1,
                "method": "SAM regions + CLIP verification",
                "candidate_source": (
                    str(Path(args.inventory_candidates_file).resolve())
                    if args.inventory_candidates_file else "assets/scene_vocabulary.txt"
                ),
                "objects": inventory,
            }, handle, ensure_ascii=False, indent=2)
    stacked_confidence = np.concatenate(
        [record["confidences"] for record in records], axis=0
    )
    view_ids = np.concatenate([np.repeat(record["path"].name, len(record["features"])) for record in records])
    prototype_guards = ({"view_ids": view_ids, "min_similarity": 0.90,
                         "min_margin": 0.05, "min_views": 2, "max_blend": 0.20}
                        if args.prototype_mode == "conservative" else {})
    stacked, prototype_ids, prototype_weights, prototype_centers = aggregate_cross_view_features(
        stacked_raw, stacked_confidence,
        args.cross_view_prototypes, 0.0 if args.prototype_mode == "off" else args.cross_view_weight,
        return_centers=True, fit_mask=fit_mask, **prototype_guards,
    )
    dimensions = min(args.feature_dim, int(fit_mask.sum()), stacked.shape[1])
    if dimensions < 3:
        raise RuntimeError("Too few SAM regions to fit a semantic feature space")
    # The language target remains the original mask descriptor. Appearance
    # prototypes are an optional auxiliary/ablation, never the default teacher.
    pca, projected, feature_min, feature_max = fit_training_pca(stacked_raw, dimensions, fit_mask)
    aggregated_projected = pca.transform(stacked)
    feature_range = np.maximum(feature_max - feature_min, 1e-6)
    encoded_prototypes = (
        pca.transform(prototype_centers) - feature_min
    ) / feature_range
    encoded_prototypes = np.clip(encoded_prototypes, 0.0, 1.0)

    offset = 0
    confidence_values = []
    tier_counts = np.zeros(3, dtype=np.int64)
    importance_configuration = {
        "policy": args.importance_policy,
        "parameters": (COMPETITIVE_IMPORTANCE_DEFAULTS if args.importance_policy == "competitive_v1"
                       else {"threshold": args.importance_threshold, "topk": args.importance_topk,
                             "background_area_ratio": args.background_area_ratio}),
        "generic_negatives": list(IMPORTANCE_NEGATIVE_PROMPTS) if args.importance_policy == "competitive_v1" else [],
        "prompts_by_image": {record["path"].name: {
            tier: record[f"{tier}_prompts"] for tier in ("background", "normal", "important")
        } for record in records},
        "boundary_promotes_semantic_tier": args.importance_policy == "legacy",
        "semantic_relevancy_is_calibrated_probability": False,
        "confidence_field_meaning": "SAM_mask_quality_not_semantic_confidence",
    }
    importance_fingerprint = importance_policy_fingerprint(importance_configuration)
    importance_reason_counts = {}
    detail_statistics = {
        "valid_pixels": 0, "boundary_pixels": 0,
        "thinness_sum": 0.0, "detail_weight_sum": 0.0,
    }
    for record in tqdm(records, desc="Dense semantic maps"):
        path = record["path"]
        confidences = record["confidences"]
        with np.load(record["raw_path"]) as raw:
            raw_payload = {key: raw[key] for key in raw.files}
        region_map = raw_payload["region_map"]
        hierarchy_region_maps = raw_payload["hierarchy_region_maps"]
        count = len(record["features"])
        current_prototype_ids = prototype_ids[offset:offset + count]
        encoded = np.clip(
            (projected[offset:offset + count] - feature_min) / feature_range, 0.0, 1.0,
        )
        aggregated_features = stacked[offset:offset + count]
        raw_payload.update(
            aggregated_features=aggregated_features.astype(np.float16),
            projected_features=projected[offset:offset + count].astype(np.float32),
            aggregated_projected_features=aggregated_projected[offset:offset + count].astype(np.float32),
            prototype_ids=current_prototype_ids.astype(np.int32),
            prototype_blend=prototype_weights[offset:offset + count].astype(np.float32),
        )
        np.savez_compressed(record["raw_path"], **raw_payload)
        offset += count
        valid = region_map >= 0
        dense = np.zeros((*region_map.shape, dimensions), dtype=np.float32)
        dense[valid] = encoded[region_map[valid]]
        hierarchy_dense = np.zeros(
            (3, *region_map.shape, dimensions), dtype=np.float32
        )
        hierarchy_valid = hierarchy_region_maps >= 0
        hierarchy_confidence = np.zeros(hierarchy_region_maps.shape, dtype=np.float32)
        for level in range(3):
            level_valid = hierarchy_valid[level]
            hierarchy_dense[level, level_valid] = encoded[
                hierarchy_region_maps[level, level_valid]
            ]
            hierarchy_confidence[level, level_valid] = confidences[
                hierarchy_region_maps[level, level_valid]
            ]
        confidence = np.zeros(region_map.shape, dtype=np.float32)
        confidence[valid] = confidences[region_map[valid]]
        confidence_values.extend(confidences.tolist())

        importance_diagnostics = {}
        if args.importance_policy == "competitive_v1":
            decision = competitive_importance(
                record["features"],
                {tier: features_for_prompts(record[f"{tier}_prompts"])
                 for tier in ("background", "normal", "important")},
                features_for_prompts(IMPORTANCE_NEGATIVE_PROMPTS), record["area_ratios"],
            )
            # Uncovered pixels are unknown, not positively identified background.
            importance = np.ones(region_map.shape, dtype=np.uint8)
            importance[valid] = decision["tiers"][region_map[valid]]
            importance_known = np.zeros(region_map.shape, dtype=np.uint8)
            region_known = np.isin(decision["reasons"], [
                "background_supported", "normal_supported", "important_supported",
            ])
            importance_known[valid] = region_known[region_map[valid]]
            importance_diagnostics = {
                "importance_region_tiers": decision["tiers"],
                "importance_region_scores": decision["scores"],
                "importance_region_margins": decision["margins"],
                "importance_region_relevancy": decision["relevancy"],
                "importance_region_reasons": decision["reasons"],
                "importance_known": importance_known,
                "mask_quality": confidence.astype(np.float16),
            }
            reason_names, reason_counts = np.unique(decision["reasons"], return_counts=True)
            for name, value in zip(reason_names, reason_counts):
                importance_reason_counts[str(name)] = importance_reason_counts.get(str(name), 0) + int(value)
        else:
            object_like = set(np.flatnonzero(
                record["area_ratios"] < args.background_area_ratio
            ).tolist())
            normal_text = features_for_prompts(record["normal_prompts"])
            normal_regions = (
                select_prompt_regions(
                    record["features"], normal_text,
                    args.importance_threshold, args.importance_topk,
                ) if normal_text is not None else object_like
            )
            normal_regions &= object_like
            background_regions = select_prompt_regions(
                record["features"],
                features_for_prompts(record["background_prompts"]),
                args.importance_threshold, args.importance_topk,
            )
            normal_regions -= background_regions
            # Keep the legacy independent prompt-union behavior unchanged.
            important_regions = select_prompt_regions(
                record["features"], features_for_prompts(record["important_prompts"]),
                args.importance_threshold, args.importance_topk,
            )
            importance = np.zeros(region_map.shape, dtype=np.uint8)
            importance[np.isin(region_map, list(normal_regions))] = 1
            importance[np.isin(region_map, list(important_regions))] = 2
            importance[np.isin(region_map, list(background_regions))] = 0
        detail_weight, boundary, thinness, importance = build_detail_supervision(
            region_map, importance,
            boundary_width=args.boundary_width,
            boundary_boost=args.boundary_boost,
            thin_boost=args.thin_boost,
            thin_compactness=args.thin_compactness,
            thin_aspect_ratio=args.thin_aspect_ratio,
            promote_importance=args.importance_policy == "legacy",
        )
        detail_statistics["valid_pixels"] += int(valid.sum())
        detail_statistics["boundary_pixels"] += int(boundary.sum())
        detail_statistics["thinness_sum"] += float(thinness[valid].sum())
        detail_statistics["detail_weight_sum"] += float(
            detail_weight[valid].sum()
        )
        prototype_map = np.full(region_map.shape, -1, dtype=np.int16)
        prototype_map[valid] = current_prototype_ids[region_map[valid]]
        hierarchy_prototype_ids = np.full(
            hierarchy_region_maps.shape, -1, dtype=np.int16
        )
        for level in range(3):
            level_valid = hierarchy_valid[level]
            hierarchy_prototype_ids[level, level_valid] = current_prototype_ids[
                hierarchy_region_maps[level, level_valid]
            ]
        tier_counts += np.bincount(importance.reshape(-1), minlength=3)[:3]
        Image.fromarray(np.take([0, 127, 255], importance).astype(np.uint8)).save(
            importance_dir / f"{path.stem}.png"
        )
        detail_preview = np.clip(
            255.0 * (detail_weight - 1.0)
            / max(float(detail_weight.max() - 1.0), 1e-6), 0, 255
        ).astype(np.uint8)
        Image.fromarray(detail_preview).save(detail_dir / f"{path.stem}.png")
        Image.fromarray(boundary.astype(np.uint8) * 255).save(
            boundary_dir / f"{path.stem}.png"
        )
        np.savez_compressed(
            maps_dir / f"{path.stem}.npz",
            features=dense.transpose(2, 0, 1).astype(np.float16),
            valid=valid.astype(np.uint8),
            region_ids=region_map.astype(np.int16),
            confidence=confidence.astype(np.float16),
            hierarchy_features=hierarchy_dense.transpose(0, 3, 1, 2).astype(np.float16),
            hierarchy_valid=hierarchy_valid.astype(np.uint8),
            hierarchy_region_ids=hierarchy_region_maps.astype(np.int16),
            hierarchy_confidence=hierarchy_confidence.astype(np.float16),
            importance=importance,
            detail_weight=detail_weight.astype(np.float16),
            boundary=boundary.astype(np.uint8),
            thinness=thinness.astype(np.float16),
            prototype_ids=prototype_map,
            hierarchy_prototype_ids=hierarchy_prototype_ids,
            **importance_diagnostics,
        )

    np.savez(
        scene / "semantic_meta.npz",
        pca_components=pca.components_.astype(np.float32),
        pca_mean=pca.mean_.astype(np.float32),
        feature_min=feature_min.astype(np.float32),
        feature_max=feature_max.astype(np.float32),
        clip_model=np.array(args.clip_model),
        clip_pretrained=np.array(args.clip_pretrained),
        prototype_features=encoded_prototypes.astype(np.float32),
        fit_image_names=np.asarray(fit_image_names),
        heldout_image_names=np.asarray(heldout_image_names),
        fit_region_count=np.asarray(int(fit_mask.sum())),
        teacher_preprocessing_version=np.asarray(2),
        hierarchy_method=np.asarray(args.hierarchy_method),
        prototype_mode=np.asarray(args.prototype_mode),
        language_target=np.asarray("raw_CLIP_projected_training_PCA"),
        prototype_clip_features=prototype_centers.astype(np.float32),
        importance_policy=np.asarray(args.importance_policy),
        importance_policy_fingerprint=np.asarray(importance_fingerprint),
        importance_policy_json=np.asarray(json.dumps(importance_configuration, sort_keys=True)),
    )
    summary = {
        "images": len(paths), "regions": int(stacked.shape[0]),
        "fit_protocol": {
            "fit_exclude_list": str(Path(args.fit_exclude_list).resolve()) if args.fit_exclude_list else None,
            "fit_image_names": fit_image_names,
            "heldout_image_names": heldout_image_names,
            "fit_regions": int(fit_mask.sum()),
            "heldout_regions": int((~fit_mask).sum()),
            "prototype_pca_bounds_training_only": bool(heldout_image_names),
        },
        "feature_dim": dimensions, "important_prompts": prompts,
        "normal_prompts": normal_prompts,
        "background_prompts": background_prompts,
        "importance_policy": args.importance_policy,
        "importance_policy_fingerprint": importance_fingerprint,
        "importance_policy_configuration": importance_configuration,
        "importance_region_reason_counts": importance_reason_counts,
        "clip_model": args.clip_model, "clip_pretrained": args.clip_pretrained,
        "sam_model": args.sam_model,
        "teacher_preprocessing": {
            "version": 2, "crop": "full_mask_bbox_square_padding_white",
            "mask_selection": args.mask_selection, "sam_crop_n_layers": args.sam_crop_n_layers,
            "mask_resize": "positive_box_occupancy_when_downsampling",
            "hierarchy_method": args.hierarchy_method,
            "hierarchy_note": "2D mask containment, not SAGA 3D physical scale",
            "importance_descriptors": "raw_unpooled_CLIP",
            "language_target": "raw_CLIP_projected_training_PCA",
            "raw_region_masks_saved": True,
        },
        "mean_sam_confidence": float(np.mean(confidence_values)),
        "cross_view_prototypes": int(prototype_ids.max() + 1),
        "mean_prototype_blend": float(prototype_weights.mean()),
        "prototype_mode": args.prototype_mode,
        "prototype_supported_region_ratio": float(np.mean(prototype_ids >= 0)),
        "prototype_guards": {key: value for key, value in prototype_guards.items() if key != "view_ids"},
        "boundary": {
            "width": args.boundary_width, "boost": args.boundary_boost,
        },
        "thin_objects": {
            "boost": args.thin_boost,
            "compactness_threshold": args.thin_compactness,
            "aspect_ratio_threshold": args.thin_aspect_ratio,
        },
        "detail_statistics": {
            "boundary_pixel_ratio": (
                detail_statistics["boundary_pixels"]
                / max(detail_statistics["valid_pixels"], 1)
            ),
            "mean_thinness": (
                detail_statistics["thinness_sum"]
                / max(detail_statistics["valid_pixels"], 1)
            ),
            "mean_detail_weight": (
                detail_statistics["detail_weight_sum"]
                / max(detail_statistics["valid_pixels"], 1)
            ),
        },
        "granularity_area_ratios": {
            "fine": args.fine_area_ratio, "coarse": args.coarse_area_ratio,
        },
        "importance_tier_pixel_ratios": (
            tier_counts / max(int(tier_counts.sum()), 1)
        ).tolist(),
        "important_json": str(Path(args.important_json).resolve()) if args.important_json else None,
        "images_subdir": args.images_subdir,
        "scene_inventory": str(scene / "scene_inventory.json"),
        "inventory_objects": len(inventory),
        "semantic_maps": str(maps_dir), "importance_masks": str(importance_dir),
        "detail_weights": str(detail_dir), "boundary_masks": str(boundary_dir),
    }
    with open(scene / "semantic_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
