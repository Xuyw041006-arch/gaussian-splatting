#!/usr/bin/env python3
"""Conservatively remove suspicious splats while preserving semantic alignment.

Default is a read-only dry run. These geometry/opacity heuristics detect some
floaters and broad translucent splats; they cannot identify every kind of blur.
"""

import argparse
import ast
import json
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile

import numpy as np
from plyfile import PlyData, PlyElement

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.prune_gaussians import select_vertices


GLOBAL_FIELDS = {
    "pca_components", "pca_mean", "feature_min", "feature_max", "scale_gate",
    "tier_rgb_weights", "tier_semantic_weights", "tier_sh_degrees",
}
POINT_FIELDS = {
    "features", "importance_score", "importance", "tier_ids", "instance_ids",
    "object_ids", "semantic_confidence", "confidence",
}


def read_config(path):
    node = ast.parse(Path(path).read_text(), mode="eval").body
    if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name) or node.func.id != "Namespace" or node.args:
        raise ValueError("Expected a literal Namespace(...) in cfg_args")
    if any(keyword.arg is None for keyword in node.keywords):
        raise ValueError("Expanded expressions in cfg_args are not supported")
    return {keyword.arg: ast.literal_eval(keyword.value) for keyword in node.keywords}


def validated_cameras(path):
    raw = json.loads(Path(path).read_text())
    if not isinstance(raw, list) or not raw:
        raise ValueError("A nonempty cameras.json is required for camera support/rendering")
    for camera in raw:
        rotation = np.asarray(camera["rotation"], dtype=float)
        position = np.asarray(camera["position"], dtype=float)
        if rotation.shape != (3, 3) or position.shape != (3,) or not np.isfinite(rotation).all() or not np.isfinite(position).all():
            raise ValueError("Invalid camera pose")
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-3):
            raise ValueError("Camera rotation is not orthonormal")
        if any(not math.isfinite(float(camera[key])) or float(camera[key]) <= 0 for key in ("fx", "fy", "width", "height")):
            raise ValueError("Invalid camera intrinsics")
    return raw


def camera_support(xyz, cameras):
    """Count image-frustum inclusion, not occlusion-aware visibility."""
    counts = np.zeros(len(xyz), dtype=np.int32)
    for camera in cameras:
        local = (xyz - np.asarray(camera["position"])) @ np.asarray(camera["rotation"])
        z = local[:, 2]
        counts += (
            (z > 1e-6)
            & (np.abs(local[:, 0]) * camera["fx"] < z * camera["width"] / 2)
            & (np.abs(local[:, 1]) * camera["fy"] < z * camera["height"] / 2)
        )
    return counts


def make_plan(vertices, importance=None, max_remove_fraction=0.10,
              large_radius_fraction=0.03, low_opacity=0.10,
              extreme_radius_multiplier=4.0, thin_axis_ratio=8.0,
              cameras=None, min_camera_support=2):
    if not 0 <= max_remove_fraction <= 0.10:
        raise ValueError("The hard deletion limit is 10%; fraction must be in [0, 0.10]")
    if not all(math.isfinite(value) for value in (
        large_radius_fraction, extreme_radius_multiplier, thin_axis_ratio, low_opacity
    )):
        raise ValueError("Cleaning thresholds must be finite")
    if large_radius_fraction <= 0 or extreme_radius_multiplier <= 1 or thin_axis_ratio <= 1 or not 0 < low_opacity < 1:
        raise ValueError("Invalid scale, opacity, radius or anisotropy threshold")
    if min_camera_support < 1:
        raise ValueError("min_camera_support must be positive")
    select_vertices(vertices)  # Reuse standard 3DGS field validation.
    if len(vertices) == 0:
        raise ValueError("Cannot clean an empty model")
    xyz = np.column_stack([vertices[name] for name in ("x", "y", "z")]).astype(float)
    logs = np.column_stack([vertices[f"scale_{i}"] for i in range(3)]).astype(float)
    if not np.isfinite(xyz).all() or not np.isfinite(logs).all() or not np.isfinite(vertices["opacity"]).all():
        raise ValueError("Nonfinite Gaussian data requires a separate corruption audit")
    scales = np.exp(np.clip(logs, -60, 60))
    axes = np.sort(scales, axis=1)
    center = np.median(xyz, axis=0)
    distance = np.linalg.norm(xyz - center, axis=1)
    radius = max(float(np.quantile(distance, 0.90)),
                 float(np.median(axes[:, 1])) * 10, 1e-12)
    alpha = 1 / (1 + np.exp(-np.clip(vertices["opacity"].astype(float), -60, 60)))
    translucent = ~select_vertices(vertices, min_opacity=low_opacity)
    distant = ~select_vertices(vertices, max_radius=radius * extreme_radius_multiplier, center=center)
    # The middle axis rejects blobs without classifying every long thin splat
    # (e.g. chopsticks) as too large just because its longest axis is large.
    large = axes[:, 1] > radius * large_radius_fraction
    # This conservative legacy policy protects ALL strong anisotropy, including
    # valid planar surface splats. max/min is not a test for a thin rod alone.
    anisotropic = axes[:, 2] / np.maximum(axes[:, 0], 1e-30) >= thin_axis_ratio
    elongated = axes[:, 2] / np.maximum(axes[:, 1], 1e-30) >= thin_axis_ratio
    planar = axes[:, 1] / np.maximum(axes[:, 0], 1e-30) >= thin_axis_ratio
    protected = anisotropic.copy()
    important = np.zeros(len(vertices), dtype=bool)
    if importance is not None:
        importance = np.asarray(importance).reshape(-1)
        if len(importance) != len(vertices) or not np.isfinite(importance).all():
            raise ValueError("Importance array must match the PLY vertex count and be finite")
        important = importance >= 0.75
        protected |= important
    reasons = {
        "large_translucent_blob": large & translucent & ~protected,
        "extreme_translucent_floater": distant & translucent & ~protected,
    }
    candidate = np.logical_or.reduce(list(reasons.values()))
    support = None
    if cameras is not None:
        support = camera_support(xyz, cameras)
        candidate &= support < min_camera_support
    severity = (1 - alpha) * (axes[:, 1] / (radius * large_radius_fraction)
                             + distance / radius)
    candidates = np.flatnonzero(candidate)
    limit = int(math.floor(len(vertices) * max_remove_fraction))
    selected = candidates[np.argsort(-severity[candidates], kind="stable")[:limit]]
    keep = np.ones(len(vertices), dtype=bool)
    keep[selected] = False
    shape_groups = {
        "elongated_only": elongated & ~planar,
        "planar_only": planar & ~elongated,
        "ribbon_elongated_and_planar": elongated & planar,
        "distributed_anisotropy": anisotropic & ~elongated & ~planar,
        "not_strongly_anisotropic": ~anisotropic,
    }
    large_translucent = large & translucent
    diagnostics = {
        "shape_protection_policy": "all_strong_anisotropy_including_surface_splats",
        "shape_definitions": {
            "elongated": "max_axis / middle_axis >= thin_axis_ratio",
            "planar": "middle_axis / min_axis >= thin_axis_ratio",
            "distributed_anisotropy": "max/min passes threshold but neither adjacent-axis ratio does",
        },
        "shape_counts_disjoint": {name: int(mask.sum()) for name, mask in shape_groups.items()},
        "importance_protected_count": int(important.sum()),
        "anisotropy_protected_count": int(anisotropic.sum()),
        "importance_and_anisotropy_overlap": int((important & anisotropic).sum()),
        "low_opacity_count": int(translucent.sum()),
        "large_low_opacity_before_protection": int(large_translucent.sum()),
        "large_low_opacity_by_shape_disjoint": {
            name: int((mask & large_translucent).sum()) for name, mask in shape_groups.items()
        },
        "large_low_opacity_important": int((large_translucent & important).sum()),
        "distance_quantiles": {f"q{int(q * 100)}": float(np.quantile(distance, q))
                               for q in (0.5, 0.75, 0.90, 0.95)},
        "middle_axis_over_scene_radius_quantiles": {
            f"q{q * 100:g}": float(np.quantile(axes[:, 1] / radius, q))
            for q in (0.5, 0.9, 0.95, 0.99, 0.999)
        },
        "scale_sensitivity_diagnostic_only": [],
        "note": "These counts do not change the keep mask. Large planar splats can be valid surfaces; shape alone cannot prove blur.",
    }
    for fraction in sorted({large_radius_fraction, 0.01, 0.005}, reverse=True):
        probe = (axes[:, 1] > radius * fraction) & translucent
        diagnostics["scale_sensitivity_diagnostic_only"].append({
            "middle_axis_radius_fraction": fraction,
            "middle_axis_threshold": radius * fraction,
            "large_low_opacity_count": int(probe.sum()),
            "unprotected_before_camera_filter": int((probe & ~protected).sum()),
            "planar_nonimportant_protected": int((probe & planar & ~elongated & ~important).sum()),
        })
    plan = {
        "input_gaussians": len(vertices), "output_gaussians": int(keep.sum()),
        "removed_gaussians": len(selected), "candidate_gaussians": len(candidates),
        "removed_fraction": len(selected) / len(vertices),
        "hard_max_remove_fraction": 0.10, "requested_max_remove_fraction": max_remove_fraction,
        "cap_applied": len(candidates) > limit,
        "robust_center": center.tolist(), "robust_radius_q90": radius,
        "thresholds": {"large_middle_axis": radius * large_radius_fraction,
                       "low_opacity": low_opacity, "extreme_radius": radius * extreme_radius_multiplier,
                       "thin_axis_ratio_protection": thin_axis_ratio, "important_score_protection": 0.75},
        "protected_gaussians": int(protected.sum()),
        "diagnostics": diagnostics,
        "removed_by_reason": {name: int((mask & ~keep).sum()) for name, mask in reasons.items()},
        "camera_support": {"enabled": cameras is not None, "minimum": min_camera_support,
                           "definition": "center inside image frustum; not occlusion-aware visibility"},
        "limitation": "Geometry/opacity heuristic; blur, surface holes and semantic quality require rendered inspection.",
        "training_resumable": False,
    }
    reason_codes = np.zeros(len(vertices), dtype=np.uint8)
    for bit, mask in enumerate(reasons.values()):
        reason_codes[mask & ~keep] |= 1 << bit
    return keep, reason_codes, plan


def filter_semantics(artifact, keep):
    """Apply the identical point order/mask; never slice global PCA/gate state."""
    count = len(keep)
    output = dict(artifact)
    if "features" not in artifact:
        raise ValueError("Semantic artifact has no features")
    fields = POINT_FIELDS | set(artifact.get("per_gaussian_fields", []))
    filtered_fields = []
    for key, value in artifact.items():
        if key in GLOBAL_FIELDS:
            continue
        shape = getattr(value, "shape", ())
        if key in fields and (not shape or shape[0] != count):
            raise ValueError(f"Semantic field {key!r} does not match {count} Gaussians")
        if shape and shape[0] == count:
            if isinstance(value, np.ndarray):
                output[key] = value[keep].copy()
            else:
                import torch
                if not torch.is_tensor(value):
                    raise ValueError(f"Unsupported per-Gaussian data type: {key}")
                output[key] = value[torch.as_tensor(keep, dtype=torch.bool, device=value.device)].clone()
            filtered_fields.append(key)
    output["num_gaussians"] = int(keep.sum())
    for key in ("gaussian_count", "total_gaussians"):
        if key in output:
            output[key] = int(keep.sum())
    output["per_gaussian_fields"] = filtered_fields
    return output, filtered_fields


def clean_model(model, output_model, iteration=-1, apply=False, camera_support_filter=False, **thresholds):
    source, destination = Path(model).resolve(), Path(output_model).resolve()
    if source == destination or source in destination.parents or destination in source.parents:
        raise ValueError("Source and output must be separate, non-nested model directories")
    if destination.exists():
        raise FileExistsError(f"Output already exists; choose a new directory: {destination}")
    config = read_config(source / "cfg_args")
    if iteration < 0:
        iterations = [int(path.name.removeprefix("iteration_")) for path in (source / "point_cloud").glob("iteration_*")
                      if path.name.removeprefix("iteration_").isdigit() and (path / "point_cloud.ply").is_file()]
        if not iterations:
            raise ValueError("No trained point_cloud.ply found")
        iteration = max(iterations)
    relative = Path("point_cloud") / f"iteration_{iteration}" / "point_cloud.ply"
    ply = PlyData.read(source / relative)
    if len(ply.elements) != 1 or ply.elements[0].name != "vertex":
        raise ValueError("Expected a Gaussian PLY with one vertex element")
    semantic_relative = Path("semantic") / f"iteration_{iteration}" / "semantic_features.pt"
    artifact = importance = None
    if (source / semantic_relative).is_file():
        import torch
        artifact = torch.load(source / semantic_relative, map_location="cpu", weights_only=False)
        if "scene_iteration" in artifact and int(artifact["scene_iteration"]) != iteration:
            raise ValueError("Semantic artifact scene_iteration does not match the PLY iteration")
        importance = artifact.get("importance_score")
        if importance is not None and torch.is_tensor(importance):
            importance = importance.float().numpy()
    cameras = validated_cameras(source / "cameras.json") if camera_support_filter else None
    keep, reasons, plan = make_plan(ply["vertex"].data, importance, cameras=cameras, **thresholds)
    filtered = None
    if artifact is not None:
        filtered, fields = filter_semantics(artifact, keep)
        plan["semantic_fields_filtered"] = fields
    plan.update(source_model=str(source), output_model=str(destination), iteration=iteration,
                dry_run=not apply, semantic_present=artifact is not None,
                source_preserved=True, render_comparison_status="not_run")
    if not apply:
        return plan
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent))
    try:
        (staging / relative).parent.mkdir(parents=True)
        PlyData([PlyElement.describe(ply["vertex"].data[keep], "vertex")], text=ply.text,
                byte_order=ply.byte_order, comments=ply.comments, obj_info=ply.obj_info).write(staging / relative)
        if filtered is not None:
            (staging / semantic_relative).parent.mkdir(parents=True)
            torch.save(filtered, staging / semantic_relative)
        config["model_path"] = str(destination)
        (staging / "cfg_args").write_text("Namespace(" + ", ".join(f"{key}={value!r}" for key, value in config.items()) + ")")
        for name in ("cameras.json", "exposure.json", "input.ply"):
            if (source / name).is_file():
                shutil.copy2(source / name, staging / name)
        np.savez_compressed(staging / "cleaning_indices.npz", keep_indices=np.flatnonzero(keep),
                            removed_indices=np.flatnonzero(~keep), removed_reason_codes=reasons[~keep],
                            source_count=np.array(len(keep)), scene_iteration=np.array(iteration))
        plan["index_mapping"] = "cleaning_indices.npz: output row i maps to source row keep_indices[i]"
        plan["reason_code_bits"] = {"1": "large_translucent_blob", "2": "extreme_translucent_floater"}
        (staging / "cleaning_report.json").write_text(json.dumps(plan, indent=2))
        os.rename(staging, destination)
    except BaseException:
        shutil.rmtree(staging)
        raise
    return plan


def render_comparison(source, destination, iteration, view_count=4, max_width=960):
    """Render identical stored camera poses without writing into the source."""
    import torch
    from PIL import Image, ImageDraw
    from argparse import Namespace
    from scene import GaussianModel
    from scene.cameras import MiniCam
    from gaussian_renderer import render
    from utils.graphics_utils import getProjectionMatrix

    if not torch.cuda.is_available():
        raise RuntimeError("Rendered comparison requires the trained CUDA renderer")
    cameras = validated_cameras(Path(source) / "cameras.json")
    chosen = np.linspace(0, len(cameras) - 1, min(view_count, len(cameras)), dtype=int)
    config = read_config(Path(source) / "cfg_args")
    models = []
    for path in (source, destination):
        model = GaussianModel(int(config.get("sh_degree", 3)))
        model.load_ply(str(Path(path) / "point_cloud" / f"iteration_{iteration}" / "point_cloud.ply"),
                       bool(config.get("train_test_exp", False)))
        models.append(model)
    pipeline = Namespace(convert_SHs_python=False, compute_cov3D_python=False, debug=False, antialiasing=False)
    bg = torch.full((3,), float(bool(config.get("white_background", False))), device="cuda")
    folder = Path(destination) / "cleaning_renders"
    folder.mkdir(exist_ok=False)
    rows = []
    with torch.no_grad():
        for index in chosen:
            entry = cameras[index]
            factor = min(1.0, max_width / entry["width"])
            width, height = max(1, round(entry["width"] * factor)), max(1, round(entry["height"] * factor))
            c2w = np.eye(4)
            c2w[:3, :3], c2w[:3, 3] = entry["rotation"], entry["position"]
            view = torch.tensor(np.linalg.inv(c2w), dtype=torch.float32, device="cuda").T
            fovx, fovy = 2 * math.atan(entry["width"] / (2 * entry["fx"])), 2 * math.atan(entry["height"] / (2 * entry["fy"]))
            projection = getProjectionMatrix(0.01, 1000.0, fovx, fovy).T.cuda()
            camera = MiniCam(width, height, fovy, fovx, 0.01, 1000.0, view, view @ projection)
            camera.image_name = entry.get("img_name", str(index))
            rendered = [render(camera, model, pipeline, bg,
                               use_trained_exp=bool(config.get("train_test_exp", False)))["render"].clamp(0, 1).cpu() for model in models]
            images = [Image.fromarray((value.permute(1, 2, 0).numpy() * 255).astype(np.uint8)) for value in rendered]
            diff = (rendered[1] - rendered[0]).abs()
            images.append(Image.fromarray((diff.permute(1, 2, 0).numpy() * 255).astype(np.uint8)))
            montage = Image.new("RGB", (width * 3, height + 24), "white")
            draw = ImageDraw.Draw(montage)
            for column, (label, image) in enumerate(zip(("Before", "After", "Absolute difference (1x)"), images)):
                draw.text((column * width + 4, 4), label, fill="black")
                montage.paste(image, (column * width, 24))
            filename = f"camera_{index:05d}.png"
            montage.save(folder / filename)
            rows.append({"camera_index": int(index), "image_name": camera.image_name,
                         "rgb_mean_absolute_change": float(diff.mean()), "comparison": filename})
    (folder / "comparison.json").write_text(json.dumps({"views": rows,
        "note": "Image change, not a quality metric. Inspect thin objects, boundaries and newly exposed holes."}, indent=2))
    report_path = Path(destination) / "cleaning_report.json"
    report = json.loads(report_path.read_text())
    report["render_comparison_status"] = "completed"
    report["render_comparison"] = str(folder / "comparison.json")
    report_path.write_text(json.dumps(report, indent=2))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output_model", required=True)
    parser.add_argument("--iteration", type=int, default=-1)
    parser.add_argument("--apply", action="store_true", help="Write a new cleaned model; default only prints the plan")
    parser.add_argument("--plan_json", help="Optional new plan file, never overwritten")
    parser.add_argument("--max_remove_fraction", type=float, default=0.10)
    parser.add_argument("--large_radius_fraction", type=float, default=0.03)
    parser.add_argument("--low_opacity", type=float, default=0.10)
    parser.add_argument("--extreme_radius_multiplier", type=float, default=4.0)
    parser.add_argument("--thin_axis_ratio", type=float, default=8.0,
                        help="Conservative max/min anisotropy protection, including planar surfaces; diagnostics separate rods and sheets")
    parser.add_argument("--camera_support_filter", action="store_true")
    parser.add_argument("--min_camera_support", type=int, default=2)
    parser.add_argument("--render_compare", action="store_true")
    parser.add_argument("--render_views", type=int, default=4)
    args = parser.parse_args()
    if args.render_compare and not args.apply:
        parser.error("--render_compare requires --apply")
    if args.render_views < 1:
        parser.error("--render_views must be positive")
    plan = clean_model(args.model, args.output_model, args.iteration, args.apply,
                       args.camera_support_filter, **{key: getattr(args, key) for key in (
                           "max_remove_fraction", "large_radius_fraction", "low_opacity",
                           "extreme_radius_multiplier", "thin_axis_ratio", "min_camera_support")})
    if args.plan_json:
        with open(args.plan_json, "x", encoding="utf-8") as handle:
            json.dump(plan, handle, indent=2)
    print(json.dumps(plan, indent=2))
    if args.render_compare:
        render_comparison(args.model, args.output_model, plan["iteration"], args.render_views)


if __name__ == "__main__":
    main()
