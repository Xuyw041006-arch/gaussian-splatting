"""Create a non-destructive edited 3DGS model from a semantic selection."""

import os
import shutil
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch
from plyfile import PlyData, PlyElement


def copy_if_present(source, destination):
    if source.is_file():
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def main():
    parser = ArgumentParser(description="Remove or isolate text-selected Gaussians")
    parser.add_argument("--model", required=True)
    parser.add_argument("--selection", required=True)
    parser.add_argument("--output_model", required=True)
    parser.add_argument("--action", choices=["remove", "keep"], default="remove")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    source = Path(args.model).resolve()
    destination = Path(args.output_model).resolve()
    if source == destination:
        parser.error("--output_model must differ from --model")
    if destination.exists() and any(destination.iterdir()) and not args.overwrite:
        parser.error("Output model is not empty; pass --overwrite to replace known outputs")

    with np.load(args.selection) as selection:
        chosen = selection["indices"].astype(np.int64)
        iteration = int(selection["scene_iteration"].item())
    source_ply = source / "point_cloud" / f"iteration_{iteration}" / "point_cloud.ply"
    source_semantic = source / "semantic" / f"iteration_{iteration}" / "semantic_features.pt"
    if not source_ply.is_file() or not source_semantic.is_file():
        parser.error(f"Missing model data for iteration {iteration}")

    ply = PlyData.read(source_ply)
    vertices = ply["vertex"].data
    if len(chosen) and (chosen.min() < 0 or chosen.max() >= len(vertices)):
        parser.error("Selection contains point indices outside the model")
    selected_mask = np.zeros(len(vertices), dtype=bool)
    selected_mask[chosen] = True
    keep_mask = ~selected_mask if args.action == "remove" else selected_mask

    output_ply = destination / "point_cloud" / f"iteration_{iteration}" / "point_cloud.ply"
    output_ply.parent.mkdir(parents=True, exist_ok=True)
    PlyData(
        [PlyElement.describe(vertices[keep_mask], "vertex")], text=ply.text
    ).write(output_ply)

    artifact = torch.load(source_semantic, map_location="cpu")
    artifact["features"] = artifact["features"][torch.from_numpy(keep_mask)]
    output_semantic = destination / "semantic" / f"iteration_{iteration}" / "semantic_features.pt"
    output_semantic.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, output_semantic)

    for name in ("cfg_args", "cameras.json", "input.ply", "exposure.json"):
        copy_if_present(source / name, destination / name)
    copy_if_present(Path(args.selection).resolve(), destination / "edit_selection.npz")
    print(
        f"Created {destination}: kept {int(keep_mask.sum())}/{len(vertices)} Gaussians "
        f"after action={args.action}. The source model was not modified."
    )


if __name__ == "__main__":
    main()
