#!/usr/bin/env python3
"""Build a small 3DGS-ready subset from one NeRF Synthetic scene."""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from plyfile import PlyData, PlyElement


def evenly_spaced(items, count):
    if count >= len(items):
        return items
    indices = np.linspace(0, len(items) - 1, count, dtype=int)
    return [items[index] for index in indices]


def image_source(source, frame):
    relative = frame["file_path"].removeprefix("./")
    path = source / relative
    return path if path.suffix else path.with_suffix(".png")


def prepare_split(source, destination, split, count, size, copy_for_semantics=False):
    with open(source / f"transforms_{split}.json", encoding="utf-8") as handle:
        transforms = json.load(handle)
    frames = evenly_spaced(transforms["frames"], count)
    output_frames = []
    (destination / split).mkdir(parents=True, exist_ok=True)
    for index, frame in enumerate(frames):
        src = image_source(source, frame)
        if not src.is_file():
            raise FileNotFoundError(src)
        stem = f"{split}_{index:02d}"
        with Image.open(src) as opened:
            image = opened.convert("RGBA")
            image.thumbnail((size, size), Image.Resampling.LANCZOS)
            image.save(destination / split / f"{stem}.png")
            if copy_for_semantics:
                white = Image.new("RGBA", image.size, (255, 255, 255, 255))
                Image.alpha_composite(white, image).convert("RGB").save(
                    destination / "images" / f"{stem}.png"
                )
        copied = dict(frame)
        copied["file_path"] = f"./{split}/{stem}"
        output_frames.append(copied)
    with open(destination / f"transforms_{split}.json", "w", encoding="utf-8") as handle:
        json.dump({"camera_angle_x": transforms["camera_angle_x"], "frames": output_frames}, handle, indent=2)
    return len(output_frames)


def write_initial_cloud(path, count, seed):
    rng = np.random.default_rng(seed)
    xyz = rng.uniform(-1.3, 1.3, size=(count, 3)).astype(np.float32)
    normals = np.zeros_like(xyz)
    colors = rng.integers(90, 190, size=(count, 3), dtype=np.uint8)
    dtype = [
        ("x", "f4"), ("y", "f4"), ("z", "f4"),
        ("nx", "f4"), ("ny", "f4"), ("nz", "f4"),
        ("red", "u1"), ("green", "u1"), ("blue", "u1"),
    ]
    vertices = np.empty(count, dtype=dtype)
    for index, name in enumerate(("x", "y", "z")):
        vertices[name] = xyz[:, index]
    for index, name in enumerate(("nx", "ny", "nz")):
        vertices[name] = normals[:, index]
    for index, name in enumerate(("red", "green", "blue")):
        vertices[name] = colors[:, index]
    PlyData([PlyElement.describe(vertices, "vertex")], text=False).write(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="Full NeRF Synthetic scene")
    parser.add_argument("--output", required=True)
    parser.add_argument("--train_views", type=int, default=8)
    parser.add_argument("--test_views", type=int, default=2)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--points", type=int, default=8000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.train_views < 3 or args.test_views < 1 or args.size < 64 or args.points < 100:
        parser.error("Use at least 3 train views, 1 test view, 64 px, and 100 points")

    source = Path(args.source).resolve()
    output = Path(args.output).resolve()
    if output.exists() and any(output.iterdir()):
        parser.error(f"Output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    (output / "images").mkdir(exist_ok=True)
    train_count = prepare_split(source, output, "train", args.train_views, args.size, True)
    test_count = prepare_split(source, output, "test", args.test_views, args.size)
    write_initial_cloud(output / "points3d.ply", args.points, args.seed)
    summary = {
        "source": str(source), "scene": str(output), "train_views": train_count,
        "test_views": test_count, "max_image_size": args.size, "initial_points": args.points,
    }
    with open(output / "sparse_scene.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
