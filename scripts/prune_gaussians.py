#!/usr/bin/env python3
"""Remove scale, opacity, and spatial outliers from a trained 3DGS PLY."""

import argparse
import json
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement


REQUIRED_FIELDS = {
    "x", "y", "z", "opacity", "scale_0", "scale_1", "scale_2",
}


def select_vertices(
    vertices,
    max_scale=None,
    min_opacity=None,
    max_radius=None,
    center=(0.0, 0.0, 0.0),
):
    """Return a boolean keep mask for standard 3DGS vertex properties."""
    names = set(vertices.dtype.names or ())
    missing = sorted(REQUIRED_FIELDS - names)
    if missing:
        raise ValueError(f"PLY vertex is missing fields: {', '.join(missing)}")

    keep = np.ones(len(vertices), dtype=bool)
    if max_scale is not None:
        log_scales = np.column_stack([
            vertices["scale_0"], vertices["scale_1"], vertices["scale_2"],
        ])
        keep &= np.exp(log_scales).max(axis=1) <= max_scale
    if min_opacity is not None:
        logits = np.clip(vertices["opacity"].astype(np.float64), -60.0, 60.0)
        opacity = 1.0 / (1.0 + np.exp(-logits))
        keep &= opacity >= min_opacity
    if max_radius is not None:
        xyz = np.column_stack([vertices["x"], vertices["y"], vertices["z"]])
        keep &= np.linalg.norm(xyz - np.asarray(center), axis=1) <= max_radius
    return keep


def prune_file(input_path, output_path, **filters):
    input_path = Path(input_path).resolve()
    output_path = Path(output_path).resolve()
    if input_path == output_path:
        raise ValueError("Input and output must be different; keep the source model recoverable")

    source = PlyData.read(input_path)
    if len(source.elements) != 1 or source.elements[0].name != "vertex":
        raise ValueError("Expected a Gaussian PLY containing one vertex element")
    vertices = source["vertex"].data
    keep = select_vertices(vertices, **filters)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    PlyData(
        [PlyElement.describe(vertices[keep], "vertex")],
        text=source.text,
        byte_order=source.byte_order,
        comments=source.comments,
        obj_info=source.obj_info,
    ).write(output_path)
    return {
        "input": str(input_path),
        "output": str(output_path),
        "input_splats": int(len(vertices)),
        "output_splats": int(keep.sum()),
        "removed_splats": int((~keep).sum()),
    }


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Trained point_cloud.ply")
    parser.add_argument("--output", required=True, help="New, pruned PLY path")
    parser.add_argument("--max_scale", type=float)
    parser.add_argument("--min_opacity", type=float)
    parser.add_argument("--max_radius", type=float)
    parser.add_argument("--center", type=float, nargs=3, default=(0.0, 0.0, 0.0))
    return parser


def main():
    args = make_parser().parse_args()
    if args.max_scale is None and args.min_opacity is None and args.max_radius is None:
        raise SystemExit("Specify at least one pruning threshold")
    if args.max_scale is not None and args.max_scale <= 0:
        raise SystemExit("--max_scale must be positive")
    if args.min_opacity is not None and not 0 <= args.min_opacity <= 1:
        raise SystemExit("--min_opacity must be between 0 and 1")
    if args.max_radius is not None and args.max_radius <= 0:
        raise SystemExit("--max_radius must be positive")
    result = prune_file(
        args.input,
        args.output,
        max_scale=args.max_scale,
        min_opacity=args.min_opacity,
        max_radius=args.max_radius,
        center=args.center,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
