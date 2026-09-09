#!/usr/bin/env python3
"""Evaluate held-out 2D SAM/CLIP teachers without fitting any scene parameters."""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image, ImageDraw

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from preprocess_semantics import (
    build_region_map, encode_regions, image_files, resize_mask, select_balanced_regions,
)
from semantic.artifact import (
    DEFAULT_NEGATIVE_PROMPTS, cosine_scores, pairwise_relevancy,
    project_clip_feature, text_retrieval_scores,
)


def match_image(paths, split, camera_map=None):
    name = camera_map.get(split) if camera_map is not None else None
    stems = {Path(name).stem} if name else {f"test_{split}", split}
    matches = [path for path in paths if path.stem in stems]
    if len(matches) != 1:
        raise ValueError(f"Mask split {split!r} must map to exactly one image, got {[p.name for p in matches]}; provide --camera_map")
    return matches[0]


def unpack_masks(raw):
    height, width = map(int, raw["mask_shape"])
    packed = np.asarray(raw["packed_region_masks"])
    return np.unpackbits(packed, axis=1, count=height * width).reshape(len(packed), height, width).astype(bool)


def nearest_mask(mask, shape):
    return np.asarray(Image.fromarray(np.asarray(mask, dtype=np.uint8)).resize(
        (shape[1], shape[0]), Image.Resampling.NEAREST)) > 0


def iou(prediction, target):
    union = np.logical_or(prediction, target).sum()
    return float(np.logical_and(prediction, target).sum() / max(int(union), 1))


def boundary(mask, ratio=0.008):
    eroded = np.asarray(mask, dtype=bool).copy()
    for _ in range(max(1, round(ratio * np.hypot(*eroded.shape)))):
        padded = np.pad(eroded, 1, constant_values=True)
        eroded = np.logical_and.reduce([padded[y:y + eroded.shape[0], x:x + eroded.shape[1]]
                                       for y in range(3) for x in range(3)])
    return np.asarray(mask, dtype=bool) & ~eroded


def prediction_masks(scores, masks, region_map, threshold):
    selected = np.isfinite(scores) & (scores >= threshold)
    union = np.any(masks[selected], axis=0) if selected.any() else np.zeros(region_map.shape, dtype=bool)
    dense = np.zeros(region_map.shape, dtype=bool)
    valid = region_map >= 0
    dense[valid] = selected[region_map[valid]]
    return union, dense, int(selected.sum())


def score_variants(features, positive_text, negative_text, meta=None, aggregated=None):
    """Compare fixed representations/scorers; no thresholds or PCA are fitted."""
    features = np.asarray(features, dtype=np.float32)
    raw = np.column_stack([cosine_scores(features, query) for query in positive_text])
    negatives = np.column_stack([cosine_scores(features, query) for query in negative_text])
    tables = {"raw_clip_cosine": raw, "raw_clip_relevancy": pairwise_relevancy(raw, negatives)}
    if aggregated is not None:
        aggregated = np.asarray(aggregated, dtype=np.float32)
        if aggregated.shape != features.shape:
            raise ValueError("Cached aggregated descriptors do not match raw descriptors")
        tables["cached_aggregated_clip_cosine"] = np.column_stack([
            cosine_scores(aggregated, query) for query in positive_text])
    if meta is not None:
        mean, components = meta["pca_mean"], meta["pca_components"]
        projected = project_clip_feature(features, mean, components)
        bounded = np.clip(projected, meta["feature_min"], meta["feature_max"])
        for name, values in (("pca", projected), ("pca_bounded", bounded)):
            tables[name + "_clip_cosine"] = text_retrieval_scores(
                values, positive_text, mean, components, mode="clip_cosine")
        tables["pca_bounded_legacy_cosine"] = text_retrieval_scores(
            bounded, positive_text, mean, components, mode="legacy_pca_cosine")
        tables["pca_bounded_clip_relevancy"] = text_retrieval_scores(
            bounded, positive_text, mean, components, mode="clip_relevancy", negative_text=negative_text)
    return tables


def evaluate_label(masks, region_map, scores, target, threshold):
    union, dense, count = prediction_masks(scores, masks, region_map, threshold)
    union, dense = nearest_mask(union, target.shape), nearest_mask(dense, target.shape)
    metrics = {"selected_regions": count, "union_iou": iou(union, target),
               "region_map_iou": iou(dense, target),
               "region_map_boundary_iou": iou(boundary(dense), boundary(target)),
               "score_max": float(np.max(scores)), "score_min": float(np.min(scores))}
    return metrics, dense


def oracle_single_mask(masks, target):
    best_iou, best_index, best_mask = -1.0, -1, np.zeros(target.shape, dtype=bool)
    for index, mask in enumerate(masks):
        prediction = nearest_mask(mask, target.shape)
        value = iou(prediction, target)
        if value > best_iou:
            best_iou, best_index, best_mask = value, index, prediction
    return max(best_iou, 0.0), best_index, best_mask


def save_panel(path, rgb, target, prediction, oracle):
    base = Image.fromarray(rgb).resize((target.shape[1], target.shape[0]), Image.Resampling.BILINEAR)
    images = [base]
    for mask in (target, prediction, oracle):
        array = np.asarray(base).copy()
        array[mask] = (0.5 * array[mask] + 0.5 * np.array([255, 40, 40])).astype(np.uint8)
        images.append(Image.fromarray(array))
    width, height = images[0].size
    canvas = Image.new("RGB", (width * 4, height + 26), "white")
    draw = ImageDraw.Draw(canvas)
    for index, (title, image) in enumerate(zip(("RGB", "Ground truth", "Raw CLIP region map", "Best retained SAM mask (oracle)"), images)):
        canvas.paste(image, (index * width, 26))
        draw.text((index * width + 4, 4), title, fill="black")
    canvas.save(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--sam_checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--test_mask", help="Defaults to scene/test_mask")
    parser.add_argument("--images_subdir", default="images")
    parser.add_argument("--semantic_meta", help="Fixed training PCA metadata; never refitted")
    parser.add_argument("--no_pca", action="store_true", help="Run raw 2D teacher diagnostics before training-only PCA is available")
    parser.add_argument("--raw_dir", help="Optional v2 semantic_raw cache; absent images are freshly inferred")
    parser.add_argument("--camera_map", help="JSON map from annotation split to exact image identity")
    parser.add_argument("--sam_model", choices=["vit_b", "vit_l", "vit_h"], default="vit_h")
    parser.add_argument("--clip_model", default="ViT-H-14")
    parser.add_argument("--clip_pretrained", default="laion2b_s32b_b79k")
    parser.add_argument("--feature_width", type=int, default=512)
    parser.add_argument("--max_masks", type=int, default=192)
    parser.add_argument("--min_mask_area", type=int, default=100)
    parser.add_argument("--points_per_side", type=int, default=32)
    parser.add_argument("--sam_crop_n_layers", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--cosine_threshold", type=float, default=0.25)
    parser.add_argument("--relevancy_threshold", type=float, default=0.50)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if min(args.feature_width, args.max_masks, args.batch_size, args.points_per_side) < 1 or args.sam_crop_n_layers < 0:
        parser.error("Invalid mask/feature/inference settings")
    if not -1 <= args.cosine_threshold <= 1 or not 0 <= args.relevancy_threshold <= 1:
        parser.error("Cosine threshold must be in [-1,1] and relevancy threshold in [0,1]")
    scene = Path(args.scene).resolve()
    mask_root = Path(args.test_mask).resolve() if args.test_mask else scene / "test_mask"
    splits = sorted(path for path in mask_root.iterdir() if path.is_dir() and list(path.glob("*.png")))
    if not splits:
        parser.error("No held-out annotated mask splits")
    paths = image_files(scene / args.images_subdir)
    camera_map = json.loads(Path(args.camera_map).read_text()) if args.camera_map else None
    matched = {split.name: match_image(paths, split.name, camera_map) for split in splits}
    if len({path.name for path in matched.values()}) != len(matched):
        parser.error("Distinct mask splits cannot share one test image")
    meta_path = Path(args.semantic_meta).resolve() if args.semantic_meta else scene / "semantic_meta.npz"
    if args.semantic_meta and args.no_pca:
        parser.error("Choose either --semantic_meta or --no_pca")
    if args.semantic_meta and not meta_path.is_file():
        parser.error(f"Fixed PCA metadata does not exist: {meta_path}")
    meta = None
    if meta_path.is_file() and not args.no_pca:
        with np.load(meta_path) as data:
            meta = {key: data[key].copy() for key in data.files}
        if str(meta["clip_model"].item()) != args.clip_model or str(meta["clip_pretrained"].item()) != args.clip_pretrained:
            parser.error("CLI CLIP model must match the fixed PCA metadata")
        fitted = {Path(str(name)).stem for name in meta.get("fit_image_names", [])}
        if fitted.intersection(path.stem for path in matched.values()):
            parser.error("Annotated test images appear in this PCA's fit_image_names; use training-only metadata")
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "views").mkdir()

    import torch
    import open_clip
    from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
    device = torch.device(args.device)
    model, _, transform = open_clip.create_model_and_transforms(
        args.clip_model, pretrained=args.clip_pretrained, precision="fp16" if device.type == "cuda" else "fp32")
    model = model.eval().to(device)
    tokenizer = open_clip.get_tokenizer(args.clip_model)
    labels = sorted({path.stem for split in splits for path in split.glob("*.png")})
    with torch.no_grad():
        text = torch.nn.functional.normalize(model.encode_text(tokenizer(labels + list(DEFAULT_NEGATIVE_PROMPTS)).to(device)).float(), dim=-1).cpu().numpy()
    positive, negative = text[:len(labels)], text[len(labels):]
    generator = None
    rows, elapsed = [], []
    for split in splits:
        started = time.monotonic()
        image_path = matched[split.name]
        rgb = np.asarray(Image.open(image_path).convert("RGB"))
        cache_path = Path(args.raw_dir) / f"{image_path.stem}.npz" if args.raw_dir else None
        raw, proposal_masks = None, None
        if cache_path is not None and cache_path.is_file():
            with np.load(cache_path) as data:
                raw = {key: data[key].copy() for key in data.files}
            if "packed_region_masks" not in raw:
                raise ValueError("Legacy raw cache lacks original masks; omit --raw_dir to infer a new teacher")
            masks, features, region_map = unpack_masks(raw), raw["features"].astype(np.float32), raw["region_map"]
            provenance = "cached_teacher"
        else:
            if generator is None:
                sam = sam_model_registry[args.sam_model](checkpoint=args.sam_checkpoint).to(device)
                generator = SamAutomaticMaskGenerator(sam, points_per_side=args.points_per_side,
                    pred_iou_thresh=0.7, stability_score_thresh=0.85, crop_n_layers=args.sam_crop_n_layers,
                    crop_n_points_downscale_factor=1, min_mask_region_area=args.min_mask_area)
            proposed = [region for region in generator.generate(rgb) if region["area"] >= args.min_mask_area]
            if not proposed:
                raise RuntimeError(f"SAM found no masks in {image_path}")
            proposal_masks = [region["segmentation"] for region in proposed]
            regions = select_balanced_regions(proposed, args.max_masks, rgb.shape[0] * rgb.shape[1])
            features = encode_regions(model, transform, rgb, regions, device, args.batch_size)
            size = (args.feature_width, max(1, round(rgb.shape[0] * args.feature_width / rgb.shape[1])))
            masks = np.stack([resize_mask(region["segmentation"], size) for region in regions])
            region_map = build_region_map(regions, size)
            provenance = "fresh_v2_teacher_not_the_deleted_legacy_teacher"
        if len(features) != len(masks):
            raise ValueError("Teacher mask and descriptor counts disagree")
        variants = score_variants(features, positive, negative, meta,
                                  raw.get("aggregated_features") if raw is not None else None)
        for annotation in sorted(split.glob("*.png")):
            label = annotation.stem
            index = labels.index(label)
            target = np.asarray(Image.open(annotation).convert("L")) > 127
            oracle_iou, oracle_index, oracle = oracle_single_mask(masks, target)
            row = {"split": split.name, "image": image_path.name, "label": label,
                   "teacher_provenance": provenance, "retained_regions": len(masks),
                   "oracle_retained_single_mask_iou": oracle_iou, "oracle_region_index": oracle_index,
                   "oracle_proposed_single_mask_iou": oracle_single_mask(proposal_masks, target)[0] if proposal_masks is not None else None,
                   "scores": {}}
            for variant, table in variants.items():
                threshold = args.relevancy_threshold if "relevancy" in variant else args.cosine_threshold
                row["scores"][variant], prediction = evaluate_label(masks, region_map, table[:, index], target, threshold)
                row["scores"][variant]["threshold"] = threshold
                if variant == "raw_clip_cosine":
                    filename = hashlib.sha256(f"{split.name}:{label}".encode()).hexdigest()[:16] + ".png"
                    save_panel(output / "views" / filename, rgb, target, prediction, oracle)
                    row["comparison"] = f"views/{filename}"
            rows.append(row)
        elapsed.append({"image": image_path.name, "seconds": time.monotonic() - started})
        print(json.dumps(elapsed[-1]), flush=True)
        (output / "progress.json").write_text(json.dumps({"rows": rows, "timings": elapsed}, indent=2))
    variants = sorted({name for row in rows for name in row["scores"]})
    proposed_oracles = [row["oracle_proposed_single_mask_iou"] for row in rows
                        if row["oracle_proposed_single_mask_iou"] is not None]
    summary = {"protocol": {
        "test_inference_only_no_fitting": True, "thresholds_fixed_before_evaluation": True,
        "cosine_threshold": args.cosine_threshold, "relevancy_threshold": args.relevancy_threshold,
        "evaluation_grid": "native annotation resolution; stored feature masks upsampled by nearest neighbour",
        "oracle_note": "GT chooses a single best mask for diagnosis only; this is not a deployable model or a bound for arbitrary mask unions",
        "pca_fit_provenance": "recorded_training_names_exclude_test" if meta is not None and "fit_image_names" in meta else "unverified_or_unavailable",
        "fixed_pca_metadata": str(meta_path) if meta is not None else None,
        "aggregation_note": "Actual cached aggregated descriptors only; no clustering/refitting on held-out images",
        "sam_crop_n_layers": args.sam_crop_n_layers,
    }, "mean_oracle_retained_single_mask_iou": float(np.mean([row["oracle_retained_single_mask_iou"] for row in rows])),
        "mean_oracle_proposed_single_mask_iou": float(np.mean(proposed_oracles)) if proposed_oracles else None,
        "methods": {name: {**{metric: float(np.mean([row["scores"][name][metric] for row in rows if name in row["scores"]]))
                           for metric in ("union_iou", "region_map_iou", "region_map_boundary_iou")},
                           "evaluated_label_views": sum(name in row["scores"] for row in rows)} for name in variants},
        "aggregated_comparison_available": "cached_aggregated_clip_cosine" in variants,
        "per_label": {label: {name: float(np.mean([row["scores"][name]["region_map_iou"] for row in rows
                                                   if row["label"] == label and name in row["scores"]])) if any(
                                                       row["label"] == label and name in row["scores"] for row in rows) else None
                              for name in variants} for label in labels},
        "rows": rows, "timings": elapsed}
    (output / "teacher_diagnostic.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({"methods": summary["methods"], "report": str(output / "teacher_diagnostic.json")}, indent=2))


if __name__ == "__main__":
    main()
