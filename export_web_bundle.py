"""Export a trained 3DGS model and semantic object index for Gaussian Atlas."""

import hashlib
import json
import shutil
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch

from semantic.artifact import apply_scale_gate, cosine_scores, decode_features, project_clip_feature


def read_label_specs(labels, labels_json):
    if labels_json:
        with open(labels_json, encoding="utf-8") as handle:
            raw = json.load(handle)
    else:
        raw = [value.strip() for value in labels.split(",") if value.strip()]
    if isinstance(raw, dict):
        raw = raw.get("objects", raw.get("labels", []))
    if not isinstance(raw, list) or not raw:
        raise ValueError("At least one semantic label is required")

    specs = []
    for index, value in enumerate(raw):
        if isinstance(value, str):
            value = {"label": value}
        if not isinstance(value, dict):
            raise ValueError(f"Label entry {index} must be a string or object")
        label = str(value.get("label") or value.get("label_zh") or "").strip()
        prompt = str(value.get("prompt") or label).strip()
        if not label or not prompt:
            raise ValueError(f"Label entry {index} has no label/prompt")
        specs.append({
            "id": str(value.get("id") or f"object-{index:03d}"),
            "label": label,
            "label_zh": str(value.get("label_zh") or "").strip() or None,
            "prompt": prompt,
            "description": str(value.get("description") or "").strip() or None,
            "aliases": [str(item).strip() for item in value.get("aliases", []) if str(item).strip()],
        })
    return specs


def assign_disjoint_indices(score_matrix, threshold, top_k=0):
    """Assign each Gaussian to its best matching label, then threshold/cap it."""
    scores = np.asarray(score_matrix, dtype=np.float32)
    if scores.ndim != 2 or scores.shape[1] == 0:
        raise ValueError("score_matrix must have shape [gaussians, labels]")
    best_labels = np.argmax(scores, axis=1)
    groups = []
    for label_index in range(scores.shape[1]):
        indices = np.flatnonzero(
            (best_labels == label_index) & (scores[:, label_index] >= threshold)
        )
        if top_k > 0 and len(indices) > top_k:
            order = np.argsort(scores[indices, label_index])[-top_k:]
            indices = indices[order]
        groups.append(indices[np.argsort(scores[indices, label_index])[::-1]])
    return groups


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = ArgumentParser(description="Export PLY + semantic_objects.json for Gaussian Atlas")
    parser.add_argument("--model", required=True)
    labels_group = parser.add_mutually_exclusive_group(required=True)
    labels_group.add_argument("--labels", help="Comma-separated CLIP prompts")
    labels_group.add_argument(
        "--labels_json",
        help="JSON list with label, optional label_zh, prompt, aliases and description",
    )
    parser.add_argument("--iteration", type=int, default=-1)
    parser.add_argument("--threshold", type=float, default=0.25)
    parser.add_argument("--granularity", type=int, choices=[0, 1, 2], default=1)
    parser.add_argument("--top_k", type=int, default=30000, help="Maximum splats per label; 0 keeps all")
    parser.add_argument("--output_dir", default="")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if not -1.0 <= args.threshold <= 1.0 or args.top_k < 0:
        parser.error("threshold must be in [-1, 1] and top_k must be non-negative")
    try:
        specs = read_label_specs(args.labels or "", args.labels_json or "")
    except (OSError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))

    from plyfile import PlyData
    from semantic_query import latest_iteration

    model_path = Path(args.model).resolve()
    iteration = latest_iteration(model_path) if args.iteration < 0 else args.iteration
    artifact_path = model_path / "semantic" / f"iteration_{iteration}" / "semantic_features.pt"
    ply_path = model_path / "point_cloud" / f"iteration_{iteration}" / "point_cloud.ply"
    if not artifact_path.is_file() or not ply_path.is_file():
        parser.error(f"Missing RGB/semantic artifact for iteration {iteration}")

    artifact = torch.load(artifact_path, map_location="cpu")
    try:
        import open_clip
    except ImportError as error:
        parser.error(f"Missing open-clip-torch: {error}")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable; pass --device cpu")
    clip_model, _, _ = open_clip.create_model_and_transforms(
        artifact["clip_model"],
        pretrained=artifact["clip_pretrained"],
        precision="fp16" if device.type == "cuda" else "fp32",
    )
    clip_model = clip_model.eval().to(device)
    tokenizer = open_clip.get_tokenizer(artifact["clip_model"])
    with torch.no_grad():
        text_features = torch.nn.functional.normalize(
            clip_model.encode_text(tokenizer([item["prompt"] for item in specs]).to(device)).float(),
            dim=-1,
            p=2,
        ).cpu().numpy()

    encoded = apply_scale_gate(
        artifact["features"].float().numpy(), artifact, args.granularity
    )
    decoded = decode_features(
        encoded,
        artifact["feature_min"].numpy(),
        artifact["feature_max"].numpy(),
    )
    queries = np.stack([
        project_clip_feature(
            feature, artifact["pca_mean"].numpy(), artifact["pca_components"].numpy()
        )
        for feature in text_features
    ])
    score_matrix = np.column_stack([cosine_scores(decoded, query) for query in queries])
    groups = assign_disjoint_indices(score_matrix, args.threshold, args.top_k)

    vertex_count = len(PlyData.read(ply_path)["vertex"].data)
    if vertex_count != len(decoded):
        raise RuntimeError(f"Point/semantic count mismatch: {vertex_count} vs {len(decoded)}")
    output_dir = Path(args.output_dir).resolve() if args.output_dir else (
        model_path / "web_export" / f"iteration_{iteration}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    output_ply = output_dir / "point_cloud.ply"
    shutil.copy2(ply_path, output_ply)

    objects = []
    for label_index, (spec, indices) in enumerate(zip(specs, groups)):
        selected_scores = score_matrix[indices, label_index]
        objects.append({
            "id": spec["id"],
            "label": spec["label"],
            "label_zh": spec["label_zh"],
            "description": spec["description"] or f"CLIP prompt: {spec['prompt']}",
            "aliases": spec["aliases"],
            "indices": indices.astype(int).tolist(),
            "score_max": float(selected_scores.max()) if len(indices) else None,
            "score_mean": float(selected_scores.mean()) if len(indices) else None,
        })
    payload = {
        "version": 1,
        "model": output_ply.name,
        "model_sha256": file_sha256(output_ply),
        "scene_iteration": iteration,
        "total_gaussians": vertex_count,
        "threshold": args.threshold,
        "granularity": args.granularity,
        "objects": objects,
    }
    output_json = output_dir / "semantic_objects.json"
    with open(output_json, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, separators=(",", ":"))

    summary = {
        "output_dir": str(output_dir),
        "model": str(output_ply),
        "semantics": str(output_json),
        "total_gaussians": vertex_count,
        "object_matches": {item["label_zh"] or item["label"]: len(item["indices"]) for item in objects},
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
