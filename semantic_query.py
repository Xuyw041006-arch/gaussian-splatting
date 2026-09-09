"""Search all trained Gaussians with an open-vocabulary text prompt."""

import json
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch
from plyfile import PlyData, PlyElement

from semantic.artifact import (
    apply_scale_gate, decode_features, select_indices,
    DEFAULT_NEGATIVE_PROMPTS, SCORE_MODES, text_retrieval_scores, affinity_point_scores,
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
    prompt = parser.add_mutually_exclusive_group(required=True)
    prompt.add_argument("--text")
    prompt.add_argument("--affinity_point_index", type=int,
                        help="v5 hierarchy query from a picked Gaussian index, without CLIP")
    parser.add_argument("--iteration", type=int, default=-1)
    parser.add_argument("--threshold", type=float, default=None,
                        help="Defaults: cosine/bank .25, clip_relevancy .5, point affinity .7; different score scales")
    parser.add_argument("--score_mode", choices=SCORE_MODES, default="legacy_pca_cosine")
    parser.add_argument("--negative_prompts", nargs="+", default=list(DEFAULT_NEGATIVE_PROMPTS))
    parser.add_argument("--relevancy_temperature", type=float, default=10.0)
    parser.add_argument("--granularity", type=int, choices=[0, 1, 2], default=1)
    parser.add_argument("--top_k", type=int, default=0, help="0 keeps every match")
    parser.add_argument("--output", default="selection.npz")
    parser.add_argument("--json", default="")
    parser.add_argument("--export_selected", default="")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--descriptor_bank", help="Optional v5 training-view region alignment .npz for --text")
    parser.add_argument("--bank_text_threshold", type=float, default=0.5)
    parser.add_argument("--bank_affinity_threshold", type=float, default=0.7)
    parser.add_argument("--bank_cross_view_threshold", type=float, default=0.8)
    parser.add_argument("--bank_min_views", type=int, default=2)
    parser.add_argument("--bank_max_candidates", type=int, default=64)
    args = parser.parse_args()
    if args.descriptor_bank and args.affinity_point_index is not None:
        parser.error("Descriptor banks require --text, not an affinity point prompt")
    if args.threshold is None:
        args.threshold = 0.25 if args.descriptor_bank else 0.7 if args.affinity_point_index is not None else 0.5 if args.score_mode == "clip_relevancy" else 0.25
    if not np.isfinite(args.relevancy_temperature) or args.relevancy_temperature <= 0:
        parser.error("relevancy_temperature must be finite and positive")
    if not np.isfinite(args.threshold) or not -1 <= args.threshold <= 1 or args.top_k < 0:
        parser.error("threshold must be finite within [-1,1], and top_k nonnegative")
    if args.affinity_point_index is None and args.score_mode == "clip_relevancy" and args.threshold < 0:
        parser.error("clip_relevancy threshold must be within [0,1]")
    if args.descriptor_bank and args.threshold < 0:
        parser.error("Descriptor-bank threshold must be within [0,1]")

    model_path = Path(args.model).resolve()
    iteration = latest_iteration(model_path) if args.iteration < 0 else args.iteration
    artifact_path = model_path / "semantic" / f"iteration_{iteration}" / "semantic_features.pt"
    ply_path = model_path / "point_cloud" / f"iteration_{iteration}" / "point_cloud.ply"
    if not artifact_path.is_file() or not ply_path.is_file():
        parser.error(f"Missing semantic artifact or point cloud for iteration {iteration}")

    artifact = torch.load(artifact_path, map_location="cpu")
    bank = None
    bank_result = None
    if args.descriptor_bank:
        from semantic.descriptor_bank import file_fingerprint, load_bank, model_signature, score_descriptor_bank
        if artifact.get("semantic_protocol") != "v5" or "affinity_features" not in artifact:
            parser.error("Descriptor bank queries require trained independent v5 affinity")
        signature = model_signature(artifact["affinity_features"].float().numpy(), iteration,
                                    artifact["clip_model"], artifact["clip_pretrained"],
                                    artifact["affinity_prefix_dimensions"], file_fingerprint(ply_path))
        bank = load_bank(args.descriptor_bank, signature)
        args.score_mode = "descriptor_bank"
    if args.affinity_point_index is not None:
        if "affinity_features" not in artifact or "affinity_prefix_dimensions" not in artifact:
            parser.error("This artifact has no independent affinity hierarchy; use --text for legacy models")
        try:
            scores = affinity_point_scores(artifact["affinity_features"].float().numpy(), args.affinity_point_index,
                                           artifact["affinity_prefix_dimensions"], args.granularity)
        except ValueError as error:
            parser.error(str(error))
        args.score_mode = "affinity_cosine"
        query_name = f"point:{args.affinity_point_index}"
    else:
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
        phrases = [args.text] + (args.negative_prompts if args.score_mode in ("clip_relevancy", "descriptor_bank") else [])
        with torch.no_grad():
            text_features = torch.nn.functional.normalize(
                clip_model.encode_text(tokenizer(phrases).to(device)).float(), dim=-1, p=2
            ).cpu().numpy()
        if bank is not None:
            bank_result = score_descriptor_bank(
                bank, artifact["affinity_features"].float().numpy(), text_features[:1], text_features[1:],
                level=args.granularity, text_threshold=args.bank_text_threshold,
                affinity_threshold=args.bank_affinity_threshold, cross_view_threshold=args.bank_cross_view_threshold,
                min_views=args.bank_min_views, max_candidates=args.bank_max_candidates,
                temperature=args.relevancy_temperature,
            )
            scores = bank_result["scores"][:, 0]
        else:
            encoded = apply_scale_gate(artifact["features"].float().numpy(), artifact, args.granularity)
            decoded = decode_features(encoded, artifact["feature_min"].numpy(), artifact["feature_max"].numpy())
            scores = text_retrieval_scores(
                decoded, text_features[:1], artifact["pca_mean"].numpy(), artifact["pca_components"].numpy(),
                mode=args.score_mode, negative_text=text_features[1:], temperature=args.relevancy_temperature,
            )[:, 0]
        query_name = args.text
    indices = select_indices(scores, args.threshold, args.top_k)
    if bank_result is not None:
        indices = indices[(scores[indices] > 0) & (bank_result["support_views"][indices, 0] >= args.bank_min_views)]

    ply = PlyData.read(ply_path)
    vertices = ply["vertex"].data
    if len(vertices) != len(scores):
        raise RuntimeError(
            f"Point/semantic count mismatch: {len(vertices)} vs {len(scores)}"
        )
    xyz = np.column_stack([vertices[axis] for axis in ("x", "y", "z")])
    selected_xyz = xyz[indices]
    result = {
        "query": query_name,
        "affinity_point_index": args.affinity_point_index,
        "threshold": args.threshold,
        "score_mode": args.score_mode,
        "granularity": args.granularity,
        "negative_prompts": args.negative_prompts if args.score_mode in ("clip_relevancy", "descriptor_bank") else [],
        "relevancy_temperature": args.relevancy_temperature if args.score_mode in ("clip_relevancy", "descriptor_bank") else None,
        "matched_gaussians": int(len(indices)),
        "total_gaussians": int(len(scores)),
        "score_max": float(scores.max()) if len(scores) else None,
        "score_mean_selected": float(scores[indices].mean()) if len(indices) else None,
        "centroid": selected_xyz.mean(axis=0).tolist() if len(indices) else None,
        "bbox_min": selected_xyz.min(axis=0).tolist() if len(indices) else None,
        "bbox_max": selected_xyz.max(axis=0).tolist() if len(indices) else None,
    }
    if bank_result is not None:
        result["descriptor_bank"] = {
            "path": str(Path(args.descriptor_bank).resolve()), "sha256": file_fingerprint(args.descriptor_bank),
            "protocol": bank_result["protocol"], "queries": bank_result["queries"],
            "source_views": bank["metadata"]["source_views"],
            "selected_min_support_views": int(bank_result["support_views"][indices, 0].min()) if len(indices) else None,
            "not_official_laga_adaptive_object_clustering": True,
        }
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path, indices=indices.astype(np.int64), scores=scores[indices].astype(np.float32),
        query=np.array(query_name), threshold=np.array(args.threshold),
        scene_iteration=np.array(iteration),
        score_mode=np.array(args.score_mode), granularity=np.array(args.granularity),
        **({"support_views": bank_result["support_views"][indices, 0],
            "descriptor_bank_sha256": np.array(result["descriptor_bank"]["sha256"])} if bank_result is not None else {}),
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
