#!/usr/bin/env python3
"""Create a tiny deterministic Blender-format scene for CUDA smoke tests.

The scene deliberately contains two large, easy-to-segment coloured spheres.
It is not a reconstruction benchmark; it exists so Colab can exercise the real
3D Gaussian rasterizer and the complete semantic pipeline without user data.
"""

import argparse
import json
import math
import shutil
from pathlib import Path

import numpy as np
from PIL import Image
from plyfile import PlyData, PlyElement


SPHERES = (
    (np.array([-0.58, 0.0, 0.0], dtype=np.float32), 0.55,
     np.array([225, 45, 38], dtype=np.float32)),
    (np.array([0.58, 0.0, 0.0], dtype=np.float32), 0.55,
     np.array([36, 88, 230], dtype=np.float32)),
)


def normalize(values):
    return values / np.maximum(np.linalg.norm(values, axis=-1, keepdims=True), 1e-8)


def camera_pose(position, target=np.zeros(3, dtype=np.float32)):
    """Return an OpenGL camera-to-world transform (Y up, Z backward)."""
    position = np.asarray(position, dtype=np.float32)
    backward = normalize((position - target)[None])[0]
    world_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    right = normalize(np.cross(world_up, backward)[None])[0]
    up = np.cross(backward, right)
    transform = np.eye(4, dtype=np.float32)
    transform[:3, :3] = np.stack((right, up, backward), axis=1)
    transform[:3, 3] = position
    return transform


def render(transform, size, fov_x):
    focal = 0.5 * size / math.tan(0.5 * fov_x)
    pixel_x = (np.arange(size, dtype=np.float32) + 0.5 - size / 2) / focal
    pixel_y = (size / 2 - np.arange(size, dtype=np.float32) - 0.5) / focal
    xx, yy = np.meshgrid(pixel_x, pixel_y)
    local = normalize(np.stack((xx, yy, -np.ones_like(xx)), axis=-1))
    directions = local @ transform[:3, :3].T
    origin = transform[:3, 3]

    depth = np.full((size, size), np.inf, dtype=np.float32)
    rgb = np.empty((size, size, 3), dtype=np.float32)
    vertical = np.linspace(0.0, 1.0, size, dtype=np.float32)[:, None, None]
    rgb[:] = np.array([15.0, 18.0, 25.0], dtype=np.float32)
    rgb += vertical * np.array([8.0, 8.0, 12.0], dtype=np.float32)

    light = normalize(np.array([[0.4, 0.8, -0.45]], dtype=np.float32))[0]
    for center, radius, color in SPHERES:
        offset = origin - center
        b = 2.0 * np.sum(directions * offset, axis=-1)
        c = float(np.dot(offset, offset) - radius * radius)
        discriminant = b * b - 4.0 * c
        valid = discriminant >= 0.0
        candidate = (-b - np.sqrt(np.maximum(discriminant, 0.0))) * 0.5
        valid &= candidate > 0.0
        closer = valid & (candidate < depth)
        if not np.any(closer):
            continue
        hit = origin + directions * candidate[..., None]
        normal = normalize(hit - center)
        lambert = np.clip(np.sum(normal * light, axis=-1), 0.0, 1.0)
        shade = 0.42 + 0.58 * lambert
        rgb[closer] = color * shade[closer, None]
        depth[closer] = candidate[closer]

    return Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8), mode="RGB")


def sample_point_cloud(count, seed):
    rng = np.random.default_rng(seed)
    sphere_ids = rng.integers(0, len(SPHERES), size=count)
    directions = normalize(rng.normal(size=(count, 3)).astype(np.float32))
    points = np.empty((count, 3), dtype=np.float32)
    colors = np.empty((count, 3), dtype=np.uint8)
    for sphere_id, (center, radius, color) in enumerate(SPHERES):
        chosen = sphere_ids == sphere_id
        points[chosen] = center + directions[chosen] * radius
        colors[chosen] = color.astype(np.uint8)
    normals = directions.astype(np.float32)
    return points, normals, colors


def write_ply(path, points, normals, colors):
    dtype = [
        ("x", "f4"), ("y", "f4"), ("z", "f4"),
        ("nx", "f4"), ("ny", "f4"), ("nz", "f4"),
        ("red", "u1"), ("green", "u1"), ("blue", "u1"),
    ]
    vertices = np.empty(len(points), dtype=dtype)
    for index, name in enumerate(("x", "y", "z")):
        vertices[name] = points[:, index]
    for index, name in enumerate(("nx", "ny", "nz")):
        vertices[name] = normals[:, index]
    for index, name in enumerate(("red", "green", "blue")):
        vertices[name] = colors[:, index]
    PlyData([PlyElement.describe(vertices, "vertex")], text=False).write(path)


def make_frames(root, split, count, size, fov_x, phase):
    split_dir = root / split
    split_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    for index in range(count):
        angle = phase + 2.0 * math.pi * index / count
        position = np.array([
            3.0 * math.sin(angle),
            0.35 + 0.18 * math.sin(2.0 * angle),
            3.0 * math.cos(angle),
        ], dtype=np.float32)
        transform = camera_pose(position)
        stem = ("r" if split == "train" else "t") + f"_{index:02d}"
        image = render(transform, size, fov_x)
        image.save(split_dir / f"{stem}.png")
        image.save(root / "images" / f"{stem}.png")
        frames.append({
            "file_path": f"./{split}/{stem}",
            "transform_matrix": transform.tolist(),
        })
    return frames


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--train_views", type=int, default=8)
    parser.add_argument("--test_views", type=int, default=2)
    parser.add_argument("--size", type=int, default=160)
    parser.add_argument("--points", type=int, default=6000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.train_views < 3 or args.test_views < 1 or args.size < 64 or args.points < 100:
        parser.error("Use at least 3 train views, 1 test view, 64 px, and 100 points")

    root = Path(args.output).resolve()
    if root.exists() and any(root.iterdir()):
        parser.error(f"Output directory is not empty: {root}")
    root.mkdir(parents=True, exist_ok=True)
    (root / "images").mkdir(exist_ok=True)
    fov_x = 0.78
    train = make_frames(root, "train", args.train_views, args.size, fov_x, 0.0)
    test = make_frames(root, "test", args.test_views, args.size, fov_x, math.pi / args.train_views)
    for split, frames in (("train", train), ("test", test)):
        with open(root / f"transforms_{split}.json", "w", encoding="utf-8") as handle:
            json.dump({"camera_angle_x": fov_x, "frames": frames}, handle, indent=2)

    points, normals, colors = sample_point_cloud(args.points, args.seed)
    write_ply(root / "points3d.ply", points, normals, colors)
    summary = {
        "scene": str(root), "train_views": len(train), "test_views": len(test),
        "image_size": args.size, "initial_points": args.points,
    }
    with open(root / "smoke_scene.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
