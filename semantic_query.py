"""Search all trained Gaussians with an open-vocabulary text prompt."""

import json
import os
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch
from plyfile import PlyData, PlyElement

from semantic.artifact import (
    apply_scale_gate, cosine_scores, decode_features, project_clip_feature, select_indices,
)


def latest_iteration(model_path):
    point_cloud = Path(model_path) / "point_cloud"
    values = [
        int(path.name.split("_")[-1]) for path in point_cloud.glob("iteration_*")
        if path.name.split("_")[-1].isdigit()
        and (
            Path(model_path) / "semantic" / path.name / "semantic_features.pt"
        ).is_file()
    ]
    if not values:
        raise FileNotFoundError(f"No RGB + semantic iteration pair under {model_path}")
    return max(values)


def save_filtered_ply(source_path, destination_path, indices):
    ply = PlyData.read(source_path)
    vertex = ply["vertex"].data[indices]
    Path(destination_path).parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(vertex, "vertex")], text=ply.text).write(destination_path)


def main():
    parser = ArgumentParser(description="Open-vocabulary query over semantic 3D Gaussians")
    parser.add_argument("--model", required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument("--iteration", type=int, default=-1)
    parser.add_argument("--threshold", type=float, default=0.25)
    parser.add_argument("--granularity", type=int, choices=[0, 1, 2], default=1)
    parser.add_argument("--top_k", type=int, default=0, help="0 keeps every match")
    parser.add_argument("--output", default="selection.npz")
    parser.add_argument("--json", default="")
    parser.add_argument("--export_selected", default="")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    model_path = Path(args.model).resolve()
    iteration = latest_iteration(model_path) if args.iteration < 0 else args.iteration
    artifact_path = model_path / "semantic" / f"iteration_{iteration}" / "semantic_features.pt"
    ply_path = model_path / "point_cloud" / f"iteration_{iteration}" / "point_cloud.ply"
    if not artifact_path.is_file() or not ply_path.is_file():
        parser.error(f"Missing semantic artifact or point cloud for iteration {iteration}")

    artifact = torch.load(artifact_path, map_location="cpu")
    try:
        import open_clip
    except ImportError as error:
        parser.error(f"Missing open-clip-torch: {error}")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is unavailable; pass --device cpu for queries")
    precision = "fp16" if device.type == "cuda" else "fp32"
    clip_model, _, _ = open_clip.create_model_and_transforms(
        artifact["clip_model"], pretrained=artifact["clip_pretrained"], precision=precision
    )
    clip_model = clip_model.eval().to(device)
    tokenizer = open_clip.get_tokenizer(artifact["clip_model"])
    with torch.no_grad():
        query_512 = torch.nn.functional.normalize(
            clip_model.encode_text(tokenizer([args.text]).to(device)).float(), dim=-1, p=2
        )[0].cpu().numpy()

    encoded = apply_scale_gate(
        artifact["features"].float().numpy(), artifact, args.granularity
    )
    decoded = decode_features(
        encoded, artifact["feature_min"].numpy(), artifact["feature_max"].numpy()
    )
    query = project_clip_feature(
        query_512, artifact["pca_mean"].numpy(), artifact["pca_components"].numpy()
    )
    scores = cosine_scores(decoded, query)
    indices = select_indices(scores, args.threshold, args.top_k)

    ply = PlyData.read(ply_path)
    vertices = ply["vertex"].data
    if len(vertices) != len(scores):
        raise RuntimeError(
            f"Point/semantic count mismatch: {len(vertices)} vs {len(scores)}"
        )
    xyz = np.column_stack([vertices[axis] for axis in ("x", "y", "z")])
    selected_xyz = xyz[indices]
    result = {
        "query": args.text,
        "threshold": args.threshold,
        "matched_gaussians": int(len(indices)),
        "total_gaussians": int(len(scores)),
        "score_max": float(scores.max()) if len(scores) else None,
        "score_mean_selected": float(scores[indices].mean()) if len(indices) else None,
        "centroid": selected_xyz.mean(axis=0).tolist() if len(indices) else None,
        "bbox_min": selected_xyz.min(axis=0).tolist() if len(indices) else None,
        "bbox_max": selected_xyz.max(axis=0).tolist() if len(indices) else None,
    }
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path, indices=indices.astype(np.int64), scores=scores[indices].astype(np.float32),
        query=np.array(args.text), threshold=np.array(args.threshold),
        scene_iteration=np.array(iteration),
    )
    if args.export_selected:
        save_filtered_ply(ply_path, args.export_selected, indices)
    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(result, handle, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"Selection saved to {output_path}")


if __name__ == "__main__":
    main()
