"""Translate, rotate, or scale all/selected Gaussians into a new model."""

import math
import shutil
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement

from semantic_query import latest_iteration


def quaternion_multiply(left, right):
    lw, lx, ly, lz = np.moveaxis(left, -1, 0)
    rw, rx, ry, rz = np.moveaxis(right, -1, 0)
    return np.stack([
        lw * rw - lx * rx - ly * ry - lz * rz,
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
    ], axis=-1)


def copy_if_present(source, destination):
    if source.is_file():
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def main():
    parser = ArgumentParser(description="Non-destructive Gaussian point-cloud transform")
    parser.add_argument("--model", required=True)
    parser.add_argument("--output_model", required=True)
    parser.add_argument("--selection", default="", help="Optional semantic_query .npz; empty transforms all")
    parser.add_argument("--iteration", type=int, default=-1)
    parser.add_argument("--translate", nargs=3, type=float, default=[0, 0, 0], metavar=("X", "Y", "Z"))
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--rotate_z", type=float, default=0.0, help="Degrees around selected centroid")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    source = Path(args.model).resolve()
    destination = Path(args.output_model).resolve()
    if source == destination:
        parser.error("Output must differ from source")
    if destination.exists() and any(destination.iterdir()) and not args.overwrite:
        parser.error("Output model is not empty; pass --overwrite to replace known outputs")
    if args.scale <= 0:
        parser.error("--scale must be positive")
    iteration = latest_iteration(source) if args.iteration < 0 else args.iteration
    source_ply = source / "point_cloud" / f"iteration_{iteration}" / "point_cloud.ply"
    ply = PlyData.read(source_ply)
    vertices = ply["vertex"].data.copy()
    chosen = np.arange(len(vertices), dtype=np.int64)
    if args.selection:
        with np.load(args.selection) as selection:
            chosen = selection["indices"].astype(np.int64)
    if not len(chosen):
        parser.error("Selection is empty")
    if chosen.min() < 0 or chosen.max() >= len(vertices):
        parser.error("Selection contains point indices outside the model")

    xyz = np.column_stack([vertices[axis] for axis in ("x", "y", "z")])
    center = xyz[chosen].mean(axis=0)
    angle = math.radians(args.rotate_z)
    rotation = np.array([
        [math.cos(angle), -math.sin(angle), 0],
        [math.sin(angle), math.cos(angle), 0],
        [0, 0, 1],
    ], dtype=np.float32)
    xyz[chosen] = (xyz[chosen] - center) @ rotation.T * args.scale + center + np.asarray(args.translate)
    for axis, name in enumerate(("x", "y", "z")):
        vertices[name][chosen] = xyz[chosen, axis]
    for axis in range(3):
        vertices[f"scale_{axis}"][chosen] += math.log(args.scale)

    old_quaternions = np.column_stack([vertices[f"rot_{axis}"][chosen] for axis in range(4)])
    z_quaternion = np.broadcast_to(
        np.array([math.cos(angle / 2), 0, 0, math.sin(angle / 2)], dtype=np.float32),
        old_quaternions.shape,
    )
    new_quaternions = quaternion_multiply(z_quaternion, old_quaternions)
    new_quaternions /= np.maximum(np.linalg.norm(new_quaternions, axis=1, keepdims=True), 1e-8)
    for axis in range(4):
        vertices[f"rot_{axis}"][chosen] = new_quaternions[:, axis]

    output_ply = destination / "point_cloud" / f"iteration_{iteration}" / "point_cloud.ply"
    output_ply.parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(vertices, "vertex")], text=ply.text).write(output_ply)
    for name in ("cfg_args", "cameras.json", "input.ply", "exposure.json"):
        copy_if_present(source / name, destination / name)
    copy_if_present(
        source / "semantic" / f"iteration_{iteration}" / "semantic_features.pt",
        destination / "semantic" / f"iteration_{iteration}" / "semantic_features.pt",
    )
    print(
        f"Created {destination}; transformed {len(chosen)} Gaussians "
        f"(scale={args.scale}, rotate_z={args.rotate_z}, translate={args.translate})."
    )


if __name__ == "__main__":
    main()
