"""Inspect the Gaussian/object nearest a click in a known training view."""

import json
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch
from plyfile import PlyData

from semantic.artifact import cosine_scores, decode_features, project_clip_feature
from semantic.inspection import pick_point
from semantic_query import latest_iteration


def sigmoid(value):
    return 1.0 / (1.0 + np.exp(-value))


def main():
    parser = ArgumentParser(description="Click-based information lookup for semantic Gaussians")
    parser.add_argument("--model", required=True)
    parser.add_argument("--view", required=True, help="Image name or numeric camera id")
    parser.add_argument("--x", required=True, type=float)
    parser.add_argument("--y", required=True, type=float)
    parser.add_argument("--labels", default="", help="Comma-separated candidate object names")
    parser.add_argument("--iteration", type=int, default=-1)
    parser.add_argument("--radius", type=float, default=8.0)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    model = Path(args.model).resolve()
    iteration = latest_iteration(model) if args.iteration < 0 else args.iteration
    with open(model / "cameras.json", encoding="utf-8") as handle:
        cameras = json.load(handle)
    camera = next(
        (
            item for item in cameras
            if str(item["id"]) == args.view or item["img_name"] == args.view
            or Path(item["img_name"]).stem == Path(args.view).stem
        ),
        None,
    )
    if camera is None:
        parser.error(f"Unknown camera/view: {args.view}")

    ply = PlyData.read(model / "point_cloud" / f"iteration_{iteration}" / "point_cloud.ply")
    vertices = ply["vertex"].data
    xyz = np.column_stack([vertices[axis] for axis in ("x", "y", "z")])
    index = pick_point(xyz, camera, args.x, args.y, args.radius)
    if index is None:
        parser.error("No Gaussian center found near that pixel; increase --radius")

    result = {
        "view": camera["img_name"], "pixel": [args.x, args.y],
        "gaussian_index": index, "xyz": xyz[index].tolist(),
        "opacity": float(sigmoid(vertices["opacity"][index])),
        "scale": [float(np.exp(vertices[f"scale_{axis}"][index])) for axis in range(3)],
    }

    labels = [label.strip() for label in args.labels.split(",") if label.strip()]
    if labels:
        artifact_path = model / "semantic" / f"iteration_{iteration}" / "semantic_features.pt"
        artifact = torch.load(artifact_path, map_location="cpu")
        try:
            import open_clip
        except ImportError as error:
            parser.error(f"Missing open-clip-torch: {error}")
        device = torch.device(args.device)
        clip_model, _, _ = open_clip.create_model_and_transforms(
            artifact["clip_model"], pretrained=artifact["clip_pretrained"],
            precision="fp16" if device.type == "cuda" else "fp32",
        )
        clip_model = clip_model.eval().to(device)
        tokenizer = open_clip.get_tokenizer(artifact["clip_model"])
        with torch.no_grad():
            text = torch.nn.functional.normalize(
                clip_model.encode_text(tokenizer(labels).to(device)).float(), dim=-1, p=2
            ).cpu().numpy()
        queries = np.stack([
            project_clip_feature(
                value, artifact["pca_mean"].numpy(), artifact["pca_components"].numpy()
            ) for value in text
        ])
        decoded = decode_features(
            artifact["features"][index].float().numpy(),
            artifact["feature_min"].numpy(), artifact["feature_max"].numpy(),
        )
        label_scores = [float(cosine_scores(decoded[None], query)[0]) for query in queries]
        order = np.argsort(label_scores)[::-1]
        result["semantic_candidates"] = [
            {"label": labels[i], "score": label_scores[i]} for i in order
        ]
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
