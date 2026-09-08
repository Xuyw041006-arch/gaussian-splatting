#!/usr/bin/env python3
"""Generate 16-bit inverse-depth priors with Depth Anything V2."""

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm


EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--images_subdir", default="images")
    parser.add_argument("--output_subdir", default="depths")
    parser.add_argument(
        "--model", default="depth-anything/Depth-Anything-V2-Small-hf",
        help="Small is Apache-2.0; larger official checkpoints may be non-commercial",
    )
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    scene = Path(args.scene).resolve()
    paths = sorted(
        path for path in (scene / args.images_subdir).iterdir()
        if path.is_file() and path.suffix.lower() in EXTENSIONS
    )
    if not paths:
        parser.error("no input images found")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        parser.error("CUDA was requested but is unavailable")
    try:
        from transformers import pipeline
    except ImportError as error:
        parser.error(f"install transformers before generating depth priors: {error}")
    device = 0 if args.device.startswith("cuda") else -1
    estimator = pipeline("depth-estimation", model=args.model, device=device)
    output = scene / args.output_subdir
    output.mkdir(parents=True, exist_ok=True)
    for path in tqdm(paths, desc="Depth Anything V2"):
        prediction = estimator(Image.open(path).convert("RGB"))
        depth = np.asarray(prediction["predicted_depth"], dtype=np.float32)
        depth -= float(depth.min())
        depth /= max(float(depth.max()), 1e-6)
        # The original 3DGS depth loader expects a 16-bit inverse-depth PNG.
        inverse = 1.0 - depth
        Image.fromarray(np.round(inverse * 65535.0).astype(np.uint16)).save(
            output / f"{path.stem}.png"
        )
    print(
        f"Wrote {len(paths)} priors. Next run: python utils/make_depth_scale.py "
        f"--base_dir {scene} --depths_dir {output}"
    )


if __name__ == "__main__":
    main()
