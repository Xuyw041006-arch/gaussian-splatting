#!/usr/bin/env python3
"""Propose scene vocabulary with RAM++ before SAM + CLIP verification."""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm


EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff"}


def uniform_subset(paths, count):
    if count < 1 or len(paths) <= count:
        return paths
    return [paths[index] for index in np.linspace(0, len(paths) - 1, count, dtype=int)]


def split_tags(value):
    return [item.strip() for item in str(value).split("|") if item.strip()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--images_subdir", default="images")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--max_images", type=int, default=48)
    parser.add_argument("--min_views", type=int, default=2)
    parser.add_argument("--image_size", type=int, default=384)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    scene = Path(args.scene).resolve()
    paths = sorted(
        path for path in (scene / args.images_subdir).iterdir()
        if path.is_file() and path.suffix.lower() in EXTENSIONS
    )
    paths = uniform_subset(paths, args.max_images)
    if not paths:
        parser.error("no input images found")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        parser.error("CUDA was requested but is unavailable")
    try:
        from ram import get_transform, inference_ram
        from ram.models import ram_plus
    except ImportError as error:
        parser.error(
            "RAM++ is optional. Install the official package with "
            "pip install git+https://github.com/xinyu1205/recognize-anything.git "
            f"({error})"
        )
    device = torch.device(args.device)
    transform = get_transform(image_size=args.image_size)
    model = ram_plus(
        pretrained=str(Path(args.checkpoint).resolve()),
        image_size=args.image_size, vit="swin_l",
    ).eval().to(device)
    english = Counter()
    chinese = Counter()
    per_image = {}
    for path in tqdm(paths, desc="RAM++ scene vocabulary"):
        image = transform(Image.open(path).convert("RGB")).unsqueeze(0).to(device)
        with torch.inference_mode():
            result = inference_ram(image, model)
        en_tags = split_tags(result[0])
        zh_tags = split_tags(result[1]) if len(result) > 1 else []
        english.update(set(en_tags))
        chinese.update(set(zh_tags))
        per_image[path.name] = {"english": en_tags, "chinese": zh_tags}
    selected = sorted(
        (tag for tag, count in english.items() if count >= args.min_views),
        key=lambda tag: (-english[tag], tag),
    )
    (scene / "scene_tag_candidates.txt").write_text(
        "\n".join(selected) + "\n", encoding="utf-8"
    )
    payload = {
        "version": 1, "method": "RAM++ proposal",
        "images_sampled": len(paths), "min_views": args.min_views,
        "candidates": [
            {"label": tag, "view_count": english[tag]} for tag in selected
        ],
        "per_image": per_image,
    }
    (scene / "scene_tag_candidates.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in payload.items() if key != "per_image"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

