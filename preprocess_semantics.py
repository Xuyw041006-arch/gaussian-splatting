"""Extract SAM regions, CLIP semantics, PCA maps, and importance masks."""

import json
import os
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def image_files(directory):
    return sorted(
        path for path in Path(directory).iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def masked_crop(rgb, mask, bbox):
    x, y, width, height = [int(value) for value in bbox]
    crop = rgb[y:y + height, x:x + width].copy()
    crop_mask = mask[y:y + height, x:x + width]
    crop[~crop_mask] = 255
    return Image.fromarray(crop)


def encode_regions(model, preprocess, rgb, regions, device, batch_size):
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
    return np.asarray(
        Image.fromarray(mask.astype(np.uint8) * 255).resize(size, Image.Resampling.NEAREST)
    ) > 0


def build_region_map(regions, size, indices=None):
    region_map = np.full((size[1], size[0]), -1, dtype=np.int16)
    # Broad regions are assigned first; smaller objects overwrite them.
    if indices is None:
        indices = range(len(regions))
    for index in indices:
        region = regions[index]
        region_map[resize_mask(region["segmentation"], size)] = index
    return region_map


def build_hierarchy_region_maps(regions, size, image_area, fine_ratio=0.05, coarse_ratio=0.25):
    """Group SAM masks into SAGA-style coarse, middle and fine levels."""
    levels = [[], [], []]
    for index, region in enumerate(regions):
        ratio = float(region["area"]) / max(float(image_area), 1.0)
        level = 2 if ratio < fine_ratio else (1 if ratio < coarse_ratio else 0)
        levels[level].append(index)
    return np.stack(
        [build_region_map(regions, size, indices) for indices in levels], axis=0
    )


def aggregate_cross_view_features(
    features, confidences, max_prototypes=64, weight=0.65,
    return_centers=False,
):
    """LaGa-inspired scene prototypes suppress view-specific CLIP noise."""
    features = np.asarray(features, dtype=np.float32)
    confidences = np.asarray(confidences, dtype=np.float32)
    if len(features) < 2 or max_prototypes < 1 or weight <= 0:
        result = (
            features.copy(), np.zeros(len(features), dtype=np.int32),
            np.zeros(len(features), dtype=np.float32),
        )
        if return_centers:
            centers = features[:1].copy() if len(features) else np.zeros(
                (0, features.shape[-1]), dtype=np.float32
            )
            return (*result, centers)
        return result
    from sklearn.cluster import MiniBatchKMeans
    clusters = min(int(max_prototypes), max(2, int(round(np.sqrt(len(features))))))
    model = MiniBatchKMeans(
        n_clusters=clusters, random_state=42,
        batch_size=min(2048, len(features)), n_init=3,
    )
    labels = model.fit_predict(features)
    centers = model.cluster_centers_.astype(np.float32)
    centers /= np.maximum(np.linalg.norm(centers, axis=1, keepdims=True), 1e-8)
    similarity = np.sum(features * centers[labels], axis=1)
    compactness = np.zeros(clusters, dtype=np.float32)
    for cluster in range(clusters):
        selected = similarity[labels == cluster]
        compactness[cluster] = max(
            float(selected.mean()) if len(selected) else 0.0, 0.0
        )
    blend = float(weight) * np.clip(
        similarity * compactness[labels] * confidences, 0.0, 1.0
    )
    aggregated = (1.0 - blend[:, None]) * features + blend[:, None] * centers[labels]
    aggregated /= np.maximum(np.linalg.norm(aggregated, axis=1, keepdims=True), 1e-8)
    result = (
        aggregated.astype(np.float32), labels.astype(np.int32),
        blend.astype(np.float32),
    )
    return (*result, centers) if return_centers else result


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
            enhanced[mask] = np.maximum(enhanced[mask], 1)

    # Preserve both sides of meaningful object boundaries.  Important-object
    # boundaries remain tier 2; other SAM boundaries become at least tier 1.
    important_nearby = _expand_binary(importance >= 2, boundary_width) & boundary
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
    if text_features is None or len(features) == 0:
        return set()
    similarities = features @ text_features.T
    chosen = set(np.flatnonzero(similarities.max(axis=1) >= threshold))
    if topk > 0:
        for prompt_index in range(text_features.shape[0]):
            count = min(int(topk), len(features))
            chosen.update(np.argsort(similarities[:, prompt_index])[-count:].tolist())
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


def main():
    parser = ArgumentParser(description="Prepare semantic supervision for 3DGS")
    parser.add_argument("--scene", required=True, help="COLMAP scene root containing images/")
    parser.add_argument("--images_subdir", default="images")
    parser.add_argument("--sam_checkpoint", required=True)
    parser.add_argument("--sam_model", choices=["vit_b", "vit_l", "vit_h"], default="vit_h")
    parser.add_argument("--clip_model", default="ViT-H-14")
    parser.add_argument("--clip_pretrained", default="laion2b_s32b_b79k")
    parser.add_argument("--feature_dim", type=int, default=32)
    parser.add_argument("--feature_width", type=int, default=320)
    parser.add_argument("--min_mask_area", type=int, default=100)
    parser.add_argument("--max_masks", type=int, default=128)
    parser.add_argument("--points_per_side", type=int, default=24)
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
    parser.add_argument("--normal", default="", help="Normal-priority object prompts")
    parser.add_argument(
        "--normal_json", default="",
        help="Optional JSON mapping image filename/stem to normal-priority objects",
    )
    parser.add_argument("--importance_threshold", type=float, default=0.24)
    parser.add_argument("--importance_topk", type=int, default=1)
    parser.add_argument("--fine_area_ratio", type=float, default=0.05)
    parser.add_argument("--coarse_area_ratio", type=float, default=0.25)
    parser.add_argument("--background_area_ratio", type=float, default=0.80)
    parser.add_argument("--cross_view_prototypes", type=int, default=96)
    parser.add_argument("--cross_view_weight", type=float, default=0.72)
    parser.add_argument("--boundary_width", type=int, default=3)
    parser.add_argument("--boundary_boost", type=float, default=2.25)
    parser.add_argument("--thin_boost", type=float, default=1.50)
    parser.add_argument("--thin_compactness", type=float, default=0.40)
    parser.add_argument("--thin_aspect_ratio", type=float, default=2.5)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if args.feature_dim < 3:
        parser.error("--feature_dim must be at least 3")
    if args.feature_width < 32 or args.max_masks < 1 or args.batch_size < 1:
        parser.error("feature width, max masks, and batch size must be positive")
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
        rgb = np.asarray(Image.open(path).convert("RGB"))
        regions = generator.generate(rgb)
        regions = [region for region in regions if region["area"] >= args.min_mask_area]
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
        hierarchy_region_maps = build_hierarchy_region_maps(
            regions, feature_size, rgb.shape[0] * rgb.shape[1],
            args.fine_area_ratio, args.coarse_area_ratio,
        )
        raw_path = raw_dir / f"{path.stem}.npz"
        np.savez_compressed(
            raw_path,
            region_map=region_map,
            hierarchy_region_maps=hierarchy_region_maps,
            features=features.astype(np.float16),
            confidences=confidences.astype(np.float16),
        )
        all_features.append(features)
        normal_mapping = normal_prompt_map if normal_prompt_map else prompt_map
        records.append({
            "path": path,
            "raw_path": raw_path,
            "features": features,
            "confidences": confidences,
            "area_ratios": np.asarray(
                [region["area"] / (rgb.shape[0] * rgb.shape[1]) for region in regions],
                dtype=np.float32,
            ),
            "important_prompts": prompts_for(prompt_map, path, prompts, "important"),
            "normal_prompts": prompts_for(
                normal_mapping, path, normal_prompts, "normal"
            ),
        })

    stacked_raw = np.concatenate(all_features, axis=0)
    stacked_confidence = np.concatenate(
        [record["confidences"] for record in records], axis=0
    )
    stacked, prototype_ids, prototype_weights, prototype_centers = aggregate_cross_view_features(
        stacked_raw, stacked_confidence,
        args.cross_view_prototypes, args.cross_view_weight, return_centers=True,
    )
    dimensions = min(args.feature_dim, stacked.shape[0], stacked.shape[1])
    if dimensions < 3:
        raise RuntimeError("Too few SAM regions to fit a semantic feature space")
    pca = PCA(n_components=dimensions, random_state=42)
    projected = pca.fit_transform(stacked)
    feature_min = projected.min(axis=0)
    feature_max = projected.max(axis=0)
    feature_range = np.maximum(feature_max - feature_min, 1e-6)
    encoded_prototypes = (
        pca.transform(prototype_centers) - feature_min
    ) / feature_range
    encoded_prototypes = np.clip(encoded_prototypes, 0.0, 1.0)

    offset = 0
    confidence_values = []
    tier_counts = np.zeros(3, dtype=np.int64)
    detail_statistics = {
        "valid_pixels": 0, "boundary_pixels": 0,
        "thinness_sum": 0.0, "detail_weight_sum": 0.0,
    }
    for record in tqdm(records, desc="Dense semantic maps"):
        path = record["path"]
        confidences = record["confidences"]
        with np.load(record["raw_path"]) as raw:
            region_map = raw["region_map"]
            hierarchy_region_maps = raw["hierarchy_region_maps"]
        count = len(record["features"])
        current_prototype_ids = prototype_ids[offset:offset + count]
        encoded = (projected[offset:offset + count] - feature_min) / feature_range
        aggregated_features = stacked[offset:offset + count]
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

        object_like = set(np.flatnonzero(
            record["area_ratios"] < args.background_area_ratio
        ).tolist())
        normal_text = features_for_prompts(record["normal_prompts"])
        normal_regions = (
            select_prompt_regions(
                aggregated_features, normal_text,
                args.importance_threshold, args.importance_topk,
            ) if normal_text is not None else object_like
        )
        normal_regions &= object_like
        important_regions = select_prompt_regions(
            aggregated_features,
            features_for_prompts(record["important_prompts"]),
            args.importance_threshold,
            args.importance_topk,
        )
        importance = np.zeros(region_map.shape, dtype=np.uint8)
        importance[np.isin(region_map, list(normal_regions))] = 1
        importance[np.isin(region_map, list(important_regions))] = 2
        detail_weight, boundary, thinness, importance = build_detail_supervision(
            region_map, importance,
            boundary_width=args.boundary_width,
            boundary_boost=args.boundary_boost,
            thin_boost=args.thin_boost,
            thin_compactness=args.thin_compactness,
            thin_aspect_ratio=args.thin_aspect_ratio,
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
            confidence=confidence.astype(np.float16),
            hierarchy_features=hierarchy_dense.transpose(0, 3, 1, 2).astype(np.float16),
            hierarchy_valid=hierarchy_valid.astype(np.uint8),
            hierarchy_confidence=hierarchy_confidence.astype(np.float16),
            importance=importance,
            detail_weight=detail_weight.astype(np.float16),
            boundary=boundary.astype(np.uint8),
            thinness=thinness.astype(np.float16),
            prototype_ids=prototype_map,
            hierarchy_prototype_ids=hierarchy_prototype_ids,
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
    )
    summary = {
        "images": len(paths), "regions": int(stacked.shape[0]),
        "feature_dim": dimensions, "important_prompts": prompts,
        "normal_prompts": normal_prompts,
        "clip_model": args.clip_model, "clip_pretrained": args.clip_pretrained,
        "sam_model": args.sam_model,
        "mean_sam_confidence": float(np.mean(confidence_values)),
        "cross_view_prototypes": int(prototype_ids.max() + 1),
        "mean_prototype_blend": float(prototype_weights.mean()),
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
        "semantic_maps": str(maps_dir), "importance_masks": str(importance_dir),
        "detail_weights": str(detail_dir), "boundary_masks": str(boundary_dir),
    }
    with open(scene / "semantic_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
