#!/usr/bin/env python3
"""Apply user/LLM importance tiers without rerunning expensive SAM inference."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from preprocess_semantics import (
    build_detail_supervision,
    select_prompt_regions,
)
from semantic.inventory import parse_inventory_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--clip_model", default="ViT-H-14")
    parser.add_argument("--clip_pretrained", default="laion2b_s32b_b79k")
    parser.add_argument("--threshold", type=float, default=0.24)
    parser.add_argument("--topk", type=int, default=1)
    parser.add_argument("--background_area_ratio", type=float, default=0.80)
    parser.add_argument("--boundary_width", type=int, default=3)
    parser.add_argument("--boundary_boost", type=float, default=2.25)
    parser.add_argument("--thin_boost", type=float, default=1.50)
    parser.add_argument("--thin_compactness", type=float, default=0.40)
    parser.add_argument("--thin_aspect_ratio", type=float, default=2.5)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    scene = Path(args.scene).resolve()
    config_path = Path(args.config).resolve()
    with open(config_path, encoding="utf-8") as handle:
        tiers = parse_inventory_config(json.load(handle))
    raw_paths = sorted((scene / "semantic_raw").glob("*.npz"))
    if not raw_paths:
        parser.error("semantic_raw is missing; run preprocess_semantics.py once first")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        parser.error("CUDA was requested but is unavailable")

    try:
        import open_clip
    except ImportError as error:
        parser.error(f"Missing open_clip dependency: {error}")
    device = torch.device(args.device)
    precision = "fp16" if device.type == "cuda" else "fp32"
    model, _, _ = open_clip.create_model_and_transforms(
        args.clip_model, pretrained=args.clip_pretrained, precision=precision
    )
    model = model.eval().to(device)
    tokenizer = open_clip.get_tokenizer(args.clip_model)

    def encode(labels):
        if not labels:
            return None
        with torch.inference_mode():
            features = model.encode_text(tokenizer(labels).to(device)).float()
            return torch.nn.functional.normalize(features, dim=-1).cpu().numpy()

    text = {tier: encode(labels) for tier, labels in tiers.items()}
    importance_dir = scene / "importance_masks"
    detail_dir = scene / "detail_weights"
    boundary_dir = scene / "boundary_masks"
    for directory in (importance_dir, detail_dir, boundary_dir):
        directory.mkdir(exist_ok=True)

    counts = np.zeros(3, dtype=np.int64)
    for raw_path in tqdm(raw_paths, desc="Apply importance tiers"):
        map_path = scene / "semantic_maps" / raw_path.name
        if not map_path.is_file():
            raise FileNotFoundError(map_path)
        with np.load(raw_path) as raw:
            region_map = raw["region_map"]
            features = raw["features"].astype(np.float32)
            area_ratios = (
                raw["area_ratios"].astype(np.float32)
                if "area_ratios" in raw.files else np.asarray([
                    np.count_nonzero(region_map == index) / region_map.size
                    for index in range(len(features))
                ], dtype=np.float32)
            )
        object_like = set(np.flatnonzero(
            area_ratios < args.background_area_ratio
        ).tolist())
        background = select_prompt_regions(
            features, text["background"], args.threshold, args.topk
        )
        normal = (
            select_prompt_regions(features, text["normal"], args.threshold, args.topk)
            if text["normal"] is not None else object_like.copy()
        ) & object_like
        important = select_prompt_regions(
            features, text["important"], args.threshold, args.topk
        ) & object_like
        normal -= background
        importance = np.zeros(region_map.shape, dtype=np.uint8)
        importance[np.isin(region_map, list(normal))] = 1
        importance[np.isin(region_map, list(important))] = 2
        importance[np.isin(region_map, list(background))] = 0
        detail_weight, boundary, thinness, importance = build_detail_supervision(
            region_map, importance, args.boundary_width, args.boundary_boost,
            args.thin_boost, args.thin_compactness, args.thin_aspect_ratio,
        )
        with np.load(map_path) as loaded:
            payload = {key: loaded[key].copy() for key in loaded.files}
        payload.update({
            "importance": importance,
            "detail_weight": detail_weight.astype(np.float16),
            "boundary": boundary.astype(np.uint8),
            "thinness": thinness.astype(np.float16),
        })
        np.savez_compressed(map_path, **payload)
        Image.fromarray(np.take([0, 127, 255], importance).astype(np.uint8)).save(
            importance_dir / f"{raw_path.stem}.png"
        )
        Image.fromarray(boundary.astype(np.uint8) * 255).save(
            boundary_dir / f"{raw_path.stem}.png"
        )
        preview = np.clip(
            255.0 * (detail_weight - 1.0)
            / max(float(detail_weight.max() - 1.0), 1e-6), 0, 255
        ).astype(np.uint8)
        Image.fromarray(preview).save(detail_dir / f"{raw_path.stem}.png")
        counts += np.bincount(importance.reshape(-1), minlength=3)[:3]

    output = {
        "config": str(config_path),
        "objects": tiers,
        "pixel_ratios": (counts / max(int(counts.sum()), 1)).tolist(),
        "maps_updated": len(raw_paths),
    }
    with open(scene / "importance_application.json", "w", encoding="utf-8") as handle:
        json.dump(output, handle, ensure_ascii=False, indent=2)
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
