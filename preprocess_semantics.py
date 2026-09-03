"""Extract SAM regions, CLIP semantics, PCA maps, and importance masks."""

import json
import os
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.decomposition import PCA
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


def build_region_map(regions, size):
    region_map = np.full((size[1], size[0]), -1, dtype=np.int16)
    # Broad regions are assigned first; smaller objects overwrite them.
    for index, region in enumerate(regions):
        region_map[resize_mask(region["segmentation"], size)] = index
    return region_map


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
    parser.add_argument("--sam_checkpoint", required=True)
    parser.add_argument("--sam_model", choices=["vit_b", "vit_l", "vit_h"], default="vit_b")
    parser.add_argument("--clip_model", default="ViT-B-16")
    parser.add_argument("--clip_pretrained", default="laion2b_s34b_b88k")
    parser.add_argument("--feature_dim", type=int, default=12)
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
    parser.add_argument("--importance_threshold", type=float, default=0.24)
    parser.add_argument("--importance_topk", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if args.feature_dim < 3:
        parser.error("--feature_dim must be at least 3")
    if args.feature_width < 32 or args.max_masks < 1 or args.batch_size < 1:
        parser.error("feature width, max masks, and batch size must be positive")
    if args.importance_topk < 0:
        parser.error("--importance_topk must be >= 0")
    if args.sam_confidence_power <= 0 or not 0 <= args.sam_confidence_floor <= 1:
        parser.error("SAM confidence power must be positive and floor must be in [0, 1]")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        parser.error("CUDA was requested but is unavailable")
    scene = Path(args.scene).resolve()
    images_dir = scene / "images"
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
    raw_dir.mkdir(exist_ok=True)
    maps_dir.mkdir(exist_ok=True)
    importance_dir.mkdir(exist_ok=True)

    prompts = parse_prompts(args.important)
    prompt_map = {}
    if args.important_json:
        with open(args.important_json, encoding="utf-8") as handle:
            prompt_map = json.load(handle)
        if not isinstance(prompt_map, dict):
            parser.error("--important_json must contain a JSON object")
    prompt_feature_cache = {}

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
        raw_path = raw_dir / f"{path.stem}.npz"
        np.savez_compressed(
            raw_path,
            region_map=region_map,
            features=features.astype(np.float16),
            confidences=confidences.astype(np.float16),
        )
        all_features.append(features)
        records.append((path, raw_path, rgb.shape[:2], features, confidences))

        importance = np.zeros(region_map.shape, dtype=np.uint8)
        current_prompts = prompt_map.get(path.name, prompt_map.get(path.stem, prompts))
        if isinstance(current_prompts, str):
            current_prompts = parse_prompts(current_prompts)
        if not isinstance(current_prompts, list):
            raise ValueError(f"Important objects for {path.name} must be a list or comma string")
        text_features = features_for_prompts(current_prompts)
        if text_features is not None:
            similarities = features @ text_features.T
            chosen = set(np.flatnonzero(similarities.max(axis=1) >= args.importance_threshold))
            if args.importance_topk > 0:
                for prompt_index in range(len(current_prompts)):
                    topk = min(args.importance_topk, len(features))
                    chosen.update(np.argsort(similarities[:, prompt_index])[-topk:].tolist())
            importance[np.isin(region_map, list(chosen))] = 255
        Image.fromarray(importance).save(importance_dir / f"{path.stem}.png")

    stacked = np.concatenate(all_features, axis=0)
    dimensions = min(args.feature_dim, stacked.shape[0], stacked.shape[1])
    if dimensions < 3:
        raise RuntimeError("Too few SAM regions to fit a semantic feature space")
    pca = PCA(n_components=dimensions, random_state=42)
    projected = pca.fit_transform(stacked)
    feature_min = projected.min(axis=0)
    feature_max = projected.max(axis=0)
    feature_range = np.maximum(feature_max - feature_min, 1e-6)

    offset = 0
    confidence_values = []
    for path, raw_path, _, features, confidences in tqdm(records, desc="Dense semantic maps"):
        with np.load(raw_path) as raw:
            region_map = raw["region_map"]
        count = len(features)
        encoded = (projected[offset:offset + count] - feature_min) / feature_range
        offset += count
        valid = region_map >= 0
        dense = np.zeros((*region_map.shape, dimensions), dtype=np.float32)
        dense[valid] = encoded[region_map[valid]]
        confidence = np.zeros(region_map.shape, dtype=np.float32)
        confidence[valid] = confidences[region_map[valid]]
        confidence_values.extend(confidences.tolist())
        np.savez_compressed(
            maps_dir / f"{path.stem}.npz",
            features=dense.transpose(2, 0, 1).astype(np.float16),
            valid=valid.astype(np.uint8),
            confidence=confidence.astype(np.float16),
        )

    np.savez(
        scene / "semantic_meta.npz",
        pca_components=pca.components_.astype(np.float32),
        pca_mean=pca.mean_.astype(np.float32),
        feature_min=feature_min.astype(np.float32),
        feature_max=feature_max.astype(np.float32),
        clip_model=np.array(args.clip_model),
        clip_pretrained=np.array(args.clip_pretrained),
    )
    summary = {
        "images": len(paths), "regions": int(stacked.shape[0]),
        "feature_dim": dimensions, "important_prompts": prompts,
        "clip_model": args.clip_model, "clip_pretrained": args.clip_pretrained,
        "sam_model": args.sam_model,
        "mean_sam_confidence": float(np.mean(confidence_values)),
        "important_json": str(Path(args.important_json).resolve()) if args.important_json else None,
        "semantic_maps": str(maps_dir), "importance_masks": str(importance_dir),
    }
    with open(scene / "semantic_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
