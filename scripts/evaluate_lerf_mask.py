"""Evaluate open-vocabulary Gaussian masks on the LERF-Mask protocol."""

import hashlib
import json
import math
import re
from argparse import ArgumentParser
from pathlib import Path


def mask_iou(prediction, target):
    import numpy as np

    prediction = np.asarray(prediction, dtype=bool)
    target = np.asarray(target, dtype=bool)
    union = np.logical_or(prediction, target).sum()
    return float(np.logical_and(prediction, target).sum() / max(int(union), 1))


def mask_boundary(mask, dilation_ratio=0.008, pad_edges=False):
    import cv2
    import numpy as np

    mask = np.asarray(mask, dtype=np.uint8)
    radius = max(1, int(round(dilation_ratio * np.hypot(*mask.shape))))
    kernel = np.ones((3, 3), dtype=np.uint8)
    if pad_edges:
        padded = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
        eroded = cv2.erode(padded, kernel, iterations=radius)[1:-1, 1:-1]
    else:
        eroded = cv2.erode(mask, kernel, iterations=radius)
    return mask.astype(bool) & ~eroded.astype(bool)


def boundary_iou(prediction, target, dilation_ratio=0.008, pad_edges=False):
    return mask_iou(
        mask_boundary(prediction, dilation_ratio, pad_edges),
        mask_boundary(target, dilation_ratio, pad_edges),
    )


def evaluate_prediction_mask(prediction, rendered_target, native_target, protocol, boundary_ratio):
    """Legacy metrics remain unchanged; GG-native uses original annotation pixels."""
    import numpy as np
    from PIL import Image

    if protocol == "gg_native":
        target = np.asarray(native_target) > 128
        resized = Image.fromarray(np.asarray(prediction, dtype=np.uint8) * 255).resize(
            (target.shape[1], target.shape[0]), Image.Resampling.NEAREST
        )
        prediction = np.asarray(resized) > 128
    elif protocol == "legacy":
        target = rendered_target
    else:
        raise ValueError(f"Unknown mask metric protocol: {protocol}")
    return {"iou": mask_iou(prediction, target),
            "boundary_iou": boundary_iou(prediction, target, boundary_ratio, protocol == "gg_native"),
            "metric_width": int(target.shape[1]), "metric_height": int(target.shape[0])}


def aggregate_mask_rows(rows, labels, protocol):
    import numpy as np

    per_label = {label: float(np.mean([row["iou"] for row in rows if row["label"] == label])) for label in labels}
    per_boundary = {label: float(np.mean([row["boundary_iou"] for row in rows if row["label"] == label])) for label in labels}
    return {
        "per_label_iou": per_label, "per_label_boundary_iou": per_boundary,
        "mean_iou": float(np.mean(list(per_label.values()))) if protocol == "gg_native" else float(np.mean([row["iou"] for row in rows])),
        "mean_boundary_iou": float(np.mean(list(per_boundary.values()))) if protocol == "gg_native" else float(np.mean([row["boundary_iou"] for row in rows])),
    }


def render_semantic_feature_map(camera, gaussians, pipeline, background, encoded, alpha, render_function):
    """Composite encoded feature channels first, then score pixels in CLIP space."""
    import numpy as np
    import torch

    height, width = alpha.shape
    dimensions = encoded.shape[1]
    feature_map = np.empty((height, width, dimensions), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, dimensions, 3):
            size = min(3, dimensions - start)
            colors = torch.zeros((len(encoded), 3), device=background.device)
            colors[:, :size] = torch.from_numpy(encoded[:, start:start + size]).to(background.device)
            rendered = render_function(camera, gaussians, pipeline, background, override_color=colors)["render"]
            normalized = rendered[:size] / alpha.clamp_min(1e-4)
            feature_map[:, :, start:start + size] = normalized.permute(1, 2, 0).cpu().numpy()
    return feature_map


def camera_for_split(cameras, split_name, camera_map=None):
    """Match explicit image identities; a mask folder number is never an index."""
    if camera_map is not None and split_name in camera_map:
        image_name = camera_map[split_name]
        if not isinstance(image_name, str) or not image_name:
            raise ValueError(f"Invalid explicit camera mapping for split {split_name!r}")
        matches = [camera for camera in cameras if camera.image_name == image_name]
        if not matches:
            matches = [camera for camera in cameras
                       if Path(camera.image_name).stem == Path(image_name).stem]
    else:
        candidates = {f"test_{split_name}", split_name}
        matches = [camera for camera in cameras if Path(camera.image_name).stem in candidates]
    if len(matches) != 1:
        found = [camera.image_name for camera in matches]
        raise ValueError(
            f"Mask split {split_name!r} needs one unambiguous test camera; found {found}. "
            "Use --camera_map with a JSON object mapping split names to image names. "
            "Numeric-index camera fallback is intentionally forbidden."
        )
    return matches[0]


def mask_splits(test_mask_root):
    """Reject empty annotations before loading CLIP or running a GPU evaluation."""
    root = Path(test_mask_root)
    if not root.is_dir():
        raise ValueError(f"Missing LERF-Mask directory: {root}")
    splits = sorted(path for path in root.iterdir() if path.is_dir())
    if not splits:
        raise ValueError(f"No annotated mask splits in {root}")
    for split in splits:
        if not list(split.glob("*.png")):
            raise ValueError(f"Mask split {split.name!r} has no PNG annotations")
    return splits


def weight_selection(model_path, iteration, artifact_path):
    """Record the selected export without mistaking a best-model alias for 15k."""
    model_path = Path(model_path)
    summary_path = model_path / "validation_summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.is_file() else {}
    is_alias = summary.get("selected_iteration_alias") == iteration
    result = {
        "requested_iteration": iteration,
        "selection": "recorded_validation_best_alias" if is_alias else "requested_iteration_export_origin_unverified",
        "source_training_iteration": summary.get("best_iteration") if is_alias else None,
        "validation_summary_path": str(summary_path) if summary_path.is_file() else None,
        "recorded_best_iteration": summary.get("best_iteration"),
        "recorded_best_psnr": summary.get("best_psnr"),
        "trained_iterations": summary.get("trained_iterations"),
        "selected_iteration_alias": summary.get("selected_iteration_alias"),
    }
    for name, path in (("point_cloud", model_path / "point_cloud" / f"iteration_{iteration}" / "point_cloud.ply"),
                       ("semantic_artifact", Path(artifact_path))):
        result[name] = {"path": str(path.resolve()), "exists": path.is_file(),
                        "bytes": path.stat().st_size if path.is_file() else None}
    return result


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_name(value):
    slug = re.sub(r"[^A-Za-z0-9_.-]", "_", str(value)).strip(".")[:100] or "item"
    # The hash prevents distinct labels containing spaces/unicode sharing a file.
    return slug + "_" + hashlib.sha256(str(value).encode()).hexdigest()[:8]


def rgb_image(tensor):
    import numpy as np
    from PIL import Image

    array = tensor.detach().cpu().permute(1, 2, 0).numpy()
    return Image.fromarray(np.round(np.clip(array, 0, 1) * 255).astype(np.uint8), "RGB")


def comparison_panel(images, titles):
    from PIL import Image, ImageDraw

    width, height = images[0].size
    if any(image.size != (width, height) for image in images):
        raise ValueError("Comparison panel images must share dimensions")
    canvas = Image.new("RGB", (width * len(images), height + 28), (24, 24, 24))
    draw = ImageDraw.Draw(canvas)
    for index, (image, title) in enumerate(zip(images, titles)):
        canvas.paste(image, (index * width, 28))
        draw.text((index * width + 6, 7), title, fill=(255, 255, 255))
    return canvas


def save_rgb_visuals(output_dir, split_name, camera_name, ground_truth, rendered):
    relative = Path("views") / safe_name(f"{split_name}_{camera_name}")
    directory = Path(output_dir) / relative
    directory.mkdir(parents=True, exist_ok=True)
    ground_truth.save(directory / "ground_truth.png")
    rendered.save(directory / "rendered.png")
    comparison_panel([ground_truth, rendered], ["Ground truth RGB", "Rendered RGB"]).save(directory / "rgb_comparison.png")
    return {"ground_truth": str(relative / "ground_truth.png"),
            "rendered": str(relative / "rendered.png"),
            "rgb_comparison": str(relative / "rgb_comparison.png")}


def save_mask_visuals(output_dir, view_relative, label, ground_truth, prediction, target):
    import numpy as np
    from PIL import Image

    directory = Path(output_dir) / Path(view_relative).parent / "masks"
    directory.mkdir(parents=True, exist_ok=True)
    stem = safe_name(label)
    overlays = []
    result = {}
    for name, mask, color in (("target", target, (40, 220, 80)),
                              ("prediction", prediction, (250, 100, 40))):
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != (ground_truth.height, ground_truth.width):
            raise ValueError("Mask and RGB dimensions do not match")
        mask_path = directory / f"{stem}_{name}.png"
        Image.fromarray(mask.astype(np.uint8) * 255).save(mask_path)
        overlay = np.asarray(ground_truth, dtype=np.float32).copy()
        overlay[mask] = overlay[mask] * 0.55 + np.asarray(color) * 0.45
        overlays.append(Image.fromarray(np.round(overlay).astype(np.uint8), "RGB"))
        result[name + "_mask"] = str(mask_path.relative_to(output_dir))
    comparison_path = directory / f"{stem}_comparison.png"
    comparison_panel(overlays, ["Target mask on GT (green)", "Predicted mask on GT (orange)"]).save(comparison_path)
    result["mask_comparison"] = str(comparison_path.relative_to(output_dir))
    return result


def protocol_metadata(views, all_camera_names, threshold, granularity, boundary_ratio):
    """Dataset fingerprint excludes model-specific outputs and includes GT/masks."""
    canonical = json.dumps(views, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return {
        "metric_scope": "annotated_test_mask_views",
        "threshold": threshold, "granularity": granularity, "boundary_ratio": boundary_ratio,
        "evaluator_version": file_sha256(__file__),
        "dataset_fingerprint": hashlib.sha256(canonical.encode()).hexdigest(),
        "test_camera_names": sorted(all_camera_names),
        "annotated_camera_names": [view["camera"] for view in views],
        "annotated_labels": sorted({label for view in views for label in view["labels"]}),
        "views": views,
        "mask_resize": "nearest_to_evaluated_rgb_dimensions",
        "mask_foreground": "uint8 > 0", "iou_aggregation": "mean_over_view_label_rows",
        "empty_union_iou": 0.0,
    }


def masked_psnr(rendered, target, mask):
    import torch

    mask = mask.to(device=rendered.device, dtype=torch.bool)
    if not mask.any():
        return None
    error = (rendered - target).square().mean(dim=0)
    mse = error[mask].mean().clamp_min(1e-12)
    return float((-10.0 * torch.log10(mse)).item())


def main():
    parser = ArgumentParser(description="Evaluate a Gaussian model on LERF-Mask")
    parser.add_argument("--model", required=True)
    parser.add_argument("--test_mask", required=True)
    parser.add_argument("--iteration", type=int, required=True)
    parser.add_argument("--threshold", type=float, default=None,
                        help="Default .25 cosine or .5 clip_relevancy; never select using test masks")
    parser.add_argument("--score_mode", choices=("legacy_pca_cosine", "clip_cosine", "clip_relevancy"), default="legacy_pca_cosine")
    parser.add_argument("--mask_protocol", choices=("legacy", "gg_native"), default="legacy")
    parser.add_argument("--negative_prompts", nargs="+", default=["object", "things", "stuff", "texture"])
    parser.add_argument("--relevancy_temperature", type=float, default=10.0)
    parser.add_argument("--alpha_min", type=float, default=None,
                        help="Default zero in legacy scoring, 1e-4 for CLIP-space pixel scoring")
    parser.add_argument("--granularity", type=int, choices=[0, 1, 2], default=1)
    parser.add_argument("--boundary_ratio", type=float, default=None,
                        help="Default .008 legacy; .02 GG-native protocol")
    parser.add_argument("--important_labels", default="")
    parser.add_argument("--normal_labels", default="")
    parser.add_argument("--output", default="")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--camera_map", help="JSON object mapping mask split to exact test image name")
    parser.add_argument("--all_test_rgb", action="store_true",
                        help="Also evaluate all test cameras under separate all_test_rgb fields")
    args = parser.parse_args()
    if args.threshold is None:
        args.threshold = 0.5 if args.score_mode == "clip_relevancy" else 0.25
    if args.boundary_ratio is None:
        args.boundary_ratio = 0.02 if args.mask_protocol == "gg_native" else 0.008
    if args.alpha_min is None:
        args.alpha_min = 0.0 if args.score_mode == "legacy_pca_cosine" else 1e-4

    model_path = Path(args.model).resolve()
    test_mask_root = Path(args.test_mask).resolve()
    artifact_path = (
        model_path / "semantic" / f"iteration_{args.iteration}" / "semantic_features.pt"
    )
    if not artifact_path.is_file() or not test_mask_root.is_dir():
        parser.error("Missing semantic artifact or LERF-Mask directory")
    if (args.iteration < 1 or not 0 <= args.threshold <= 1 or args.boundary_ratio <= 0
            or not all(math.isfinite(x) for x in (args.threshold, args.boundary_ratio, args.alpha_min, args.relevancy_temperature))
            or not 0 <= args.alpha_min <= 1 or args.relevancy_temperature <= 0):
        parser.error("iteration must be positive, threshold within [0,1], and boundary_ratio positive")
    suffix = "" if (args.score_mode, args.mask_protocol) == ("legacy_pca_cosine", "legacy") else f"_{args.score_mode}_{args.mask_protocol}"
    output_dir = Path(args.output).resolve() if args.output else model_path / ("lerf_mask_eval" + suffix)
    if (output_dir / "metrics.json").is_file():
        previous = json.loads((output_dir / "metrics.json").read_text())
        old_protocol = previous.get("protocol", {})
        if (old_protocol.get("score_mode", "legacy_pca_cosine") != args.score_mode
                or old_protocol.get("mask_metric_protocol", "legacy") != args.mask_protocol):
            parser.error("Use a separate --output directory for each score/mask protocol; do not overwrite legacy results")
    try:
        splits = mask_splits(test_mask_root)
        camera_map = json.loads(Path(args.camera_map).read_text()) if args.camera_map else None
        if camera_map is not None and not isinstance(camera_map, dict):
            raise ValueError("--camera_map must contain a JSON object")
    except (OSError, ValueError) as error:
        parser.error(str(error))

    import numpy as np
    import torch
    from PIL import Image

    from gaussian_renderer import render
    from interactive_renderer import extract_dataset_and_pipeline, read_model_config
    from scene import GaussianModel, Scene
    from semantic.artifact import apply_scale_gate, cosine_scores, decode_features, project_clip_feature, text_retrieval_scores
    from utils.image_utils import psnr
    from utils.loss_utils import ssim

    config = read_model_config(model_path)
    config.eval = True
    dataset, pipeline = extract_dataset_and_pipeline(config)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False)
    cameras = scene.getTestCameras()
    if not cameras:
        raise RuntimeError("No test cameras; create sparse/0/test.txt before training")
    split_cameras = {split.name: camera_for_split(cameras, split.name, camera_map) for split in splits}
    mapped_names = [camera.image_name for camera in split_cameras.values()]
    if len(mapped_names) != len(set(mapped_names)):
        raise ValueError("Distinct annotated splits map to the same test camera; check --camera_map")

    artifact = torch.load(artifact_path, map_location="cpu")
    if artifact.get("scene_iteration", args.iteration) != args.iteration:
        raise RuntimeError("Semantic artifact scene_iteration differs from the requested RGB export")
    encoded = artifact["features"].float().numpy()
    if len(encoded) != gaussians.get_xyz.shape[0]:
        raise RuntimeError("RGB and semantic Gaussian counts do not match")
    encoded = apply_scale_gate(encoded, artifact, args.granularity)
    decoded = decode_features(encoded, artifact["feature_min"].numpy(), artifact["feature_max"].numpy()) if args.score_mode == "legacy_pca_cosine" else None

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

    labels = sorted({path.stem for split in splits for path in split.glob("*.png")})
    phrases = labels + (args.negative_prompts if args.score_mode == "clip_relevancy" else [])
    with torch.no_grad():
        text = torch.nn.functional.normalize(
            clip_model.encode_text(tokenizer(phrases).to(device)).float(), dim=-1
        ).cpu().numpy()
    queries = np.stack([
        project_clip_feature(
            feature, artifact["pca_mean"].numpy(), artifact["pca_components"].numpy()
        ) for feature in text[:len(labels)]
    ])
    scores = {
        label: cosine_scores(decoded, query) for label, query in zip(labels, queries)
    } if args.score_mode == "legacy_pca_cosine" else {}

    output_dir.mkdir(parents=True, exist_ok=True)
    important_labels = {
        value.strip() for value in args.important_labels.split(",") if value.strip()
    }
    normal_labels = {
        value.strip() for value in args.normal_labels.split(",") if value.strip()
    }
    if important_labels & normal_labels:
        parser.error("Important and normal label groups must be disjoint")
    background = torch.zeros(3, dtype=torch.float32, device="cuda")
    rows = []
    reconstruction_rows = []
    protocol_views = []
    for split in splits:
        camera = split_cameras[split.name]
        target_masks = {}
        native_masks = {}
        mask_sources = {}
        for target_path in sorted(split.glob("*.png")):
            with Image.open(target_path) as source_image:
                native_masks[target_path.stem] = np.asarray(source_image.convert("L")).copy()
                mask_sources[target_path.stem] = {
                    "source_size": list(source_image.size), "sha256": file_sha256(target_path),
                }
                image = source_image.convert("L").resize(
                    (camera.image_width, camera.image_height), Image.Resampling.NEAREST
                )
            target_masks[target_path.stem] = np.asarray(image) > 0
        with torch.no_grad():
            rgb_render = render(
                camera, gaussians, pipeline, background
            )["render"].clamp(0, 1)
            ground_truth = camera.original_image[:3].cuda().clamp(0, 1)
            reconstruction_row = {
                "split": split.name,
                "camera": camera.image_name,
                "width": int(camera.image_width), "height": int(camera.image_height),
                "metric_scope": "annotated_test_mask_views",
                "psnr": float(psnr(rgb_render, ground_truth).mean().item()),
                "ssim": float(ssim(rgb_render, ground_truth).item()),
            }
            gt_image, rendered_image = rgb_image(ground_truth), rgb_image(rgb_render)
            reconstruction_row["visualizations"] = save_rgb_visuals(
                output_dir, split.name, camera.image_name, gt_image, rendered_image
            )
            transforms = {}
            for attribute in ("world_view_transform", "full_proj_transform"):
                tensor = getattr(camera, attribute, None)
                if tensor is not None:
                    transforms[attribute] = hashlib.sha256(tensor.detach().cpu().numpy().tobytes()).hexdigest()
            protocol_views.append({
                "split": split.name, "camera": camera.image_name,
                "width": int(camera.image_width), "height": int(camera.image_height),
                "labels": sorted(target_masks), "mask_sources": mask_sources,
                "ground_truth_sha256": hashlib.sha256(ground_truth.detach().cpu().numpy().tobytes()).hexdigest(),
                "camera_transforms_sha256": transforms,
            })
            for tier_name, tier_labels in (
                ("important", important_labels), ("normal", normal_labels)
            ):
                selected = [
                    target_masks[label] for label in tier_labels
                    if label in target_masks
                ]
                if selected:
                    union = torch.from_numpy(np.logical_or.reduce(selected)).cuda()
                    reconstruction_row[f"{tier_name}_psnr"] = masked_psnr(
                        rgb_render, ground_truth, union
                    )
            reconstruction_rows.append(reconstruction_row)
            alpha = render(
                camera, gaussians, pipeline, background,
                override_color=torch.ones(
                    (len(encoded), 3), dtype=torch.float32, device="cuda"
                ),
            )["render"][0].clamp(0, 1)
        pixel_scores = None
        if args.score_mode != "legacy_pca_cosine":
            feature_map = render_semantic_feature_map(camera, gaussians, pipeline, background, encoded, alpha, render)
            pixel_decoded = decode_features(feature_map.reshape(-1, encoded.shape[1]), artifact["feature_min"].numpy(), artifact["feature_max"].numpy())
            pixel_scores = text_retrieval_scores(
                pixel_decoded, text[:len(labels)], artifact["pca_mean"].numpy(), artifact["pca_components"].numpy(),
                mode=args.score_mode, negative_text=text[len(labels):], temperature=args.relevancy_temperature,
            ).reshape(camera.image_height, camera.image_width, len(labels))
            del feature_map, pixel_decoded
        alpha_valid = (alpha >= args.alpha_min).cpu().numpy()
        for target_path in sorted(split.glob("*.png")):
            label = target_path.stem
            if args.score_mode == "legacy_pca_cosine":
                values = torch.from_numpy(scores[label]).float().cuda()
                colors = values[:, None].repeat(1, 3).clamp(0, 1)
                with torch.no_grad():
                    score_render = render(camera, gaussians, pipeline, background, override_color=colors)["render"][0]
                    score_render = score_render / alpha.clamp_min(1e-4)
                score_array = score_render.cpu().numpy()
            else:
                score_array = pixel_scores[:, :, labels.index(label)]
            prediction = (score_array >= args.threshold) & alpha_valid
            target = target_masks[label]
            Image.fromarray(prediction.astype(np.uint8) * 255).save(
                output_dir / f"{split.name}_{label}.png"
            )
            visualizations = save_mask_visuals(
                output_dir, reconstruction_row["visualizations"]["ground_truth"],
                label, gt_image, prediction, target,
            )
            rows.append({
                "split": split.name,
                "camera": camera.image_name,
                "label": label,
                "width": int(camera.image_width), "height": int(camera.image_height),
                "visualizations": visualizations,
                **evaluate_prediction_mask(prediction, target, native_masks[label], args.mask_protocol, args.boundary_ratio),
                "score_min": float(score_array[alpha_valid].min()) if alpha_valid.any() else None,
                "score_max": float(score_array[alpha_valid].max()) if alpha_valid.any() else None,
                "prediction_pixels": int(prediction.sum()), "target_pixels_render_size": int(target.sum()),
            })

    protocol = protocol_metadata(protocol_views, [camera.image_name for camera in cameras],
                                 args.threshold, args.granularity, args.boundary_ratio)
    protocol.update({
        "score_mode": args.score_mode, "mask_metric_protocol": args.mask_protocol,
        "negative_prompts": args.negative_prompts if args.score_mode == "clip_relevancy" else [],
        "relevancy_temperature": args.relevancy_temperature if args.score_mode == "clip_relevancy" else None,
        "alpha_min": args.alpha_min,
        "score_compositing": "clipped_per_gaussian_cosine_then_render" if args.score_mode == "legacy_pca_cosine" else "render_encoded_features_then_clip_space_score",
        "score_space": "centered_pca" if args.score_mode == "legacy_pca_cosine" else "pca_reconstructed_clip",
        "mask_resize": "prediction_nearest_to_native_gt" if args.mask_protocol == "gg_native" else "nearest_to_evaluated_rgb_dimensions",
        "mask_foreground": "uint8 > 128" if args.mask_protocol == "gg_native" else "uint8 > 0",
        "boundary_pad_edges": args.mask_protocol == "gg_native",
        "iou_aggregation": "macro_mean_over_labels" if args.mask_protocol == "gg_native" else "mean_over_view_label_rows",
        "layer_selection": "fixed_explicit_granularity_no_test_selection",
    })
    result = {
        "model": str(model_path),
        "iteration": args.iteration,
        "threshold": args.threshold,
        "granularity": args.granularity,
        "boundary_ratio": args.boundary_ratio,
        "metric_scope": "annotated_test_mask_views",
        "evaluator_version": protocol["evaluator_version"],
        "dataset_fingerprint": protocol["dataset_fingerprint"],
        "protocol": protocol,
        "weight_selection": weight_selection(model_path, args.iteration, artifact_path),
        "semantic_artifact": {
            "version": artifact.get("version"),
            "semantic_protocol": artifact.get("semantic_protocol", "legacy_unrecorded"),
            "language_granularity": "saved_scale_gate" if "scale_gate" in artifact else "single_language_field_no_scale_gate",
            "has_independent_affinity": "affinity_features" in artifact,
        },
        "important_labels": sorted(important_labels), "normal_labels": sorted(normal_labels),
        "gaussians": int(len(encoded)),
        "test_psnr": float(np.mean([row["psnr"] for row in reconstruction_rows])),
        "test_ssim": float(np.mean([row["ssim"] for row in reconstruction_rows])),
        **aggregate_mask_rows(rows, labels, args.mask_protocol),
        "reconstruction_rows": reconstruction_rows,
        "rows": rows,
    }
    if args.all_test_rgb:
        by_camera = {row["camera"]: row for row in reconstruction_rows}
        all_rows = []
        for camera in cameras:
            if camera.image_name in by_camera:
                row = {key: by_camera[camera.image_name][key] for key in ("camera", "width", "height", "psnr", "ssim")}
            else:
                with torch.no_grad():
                    rgb = render(camera, gaussians, pipeline, background)["render"].clamp(0, 1)
                    gt = camera.original_image[:3].cuda().clamp(0, 1)
                    row = {"camera": camera.image_name, "width": int(camera.image_width),
                           "height": int(camera.image_height), "psnr": float(psnr(rgb, gt).mean().item()),
                           "ssim": float(ssim(rgb, gt).item())}
            all_rows.append(row)
        result["all_test_rgb"] = {
            "metric_scope": "all_test_cameras", "camera_count": len(all_rows),
            "psnr": float(np.mean([row["psnr"] for row in all_rows])),
            "ssim": float(np.mean([row["ssim"] for row in all_rows])), "rows": all_rows,
        }
    for tier_name in ("important", "normal"):
        values = [
            row[f"{tier_name}_psnr"] for row in reconstruction_rows
            if row.get(f"{tier_name}_psnr") is not None
        ]
        if values:
            result[f"test_{tier_name}_psnr"] = float(np.mean(values))
    if "importance_score" in artifact:
        importance = artifact["importance_score"].float().numpy()
        result["tier_gaussians"] = {
            "background": int((importance < 0.25).sum()),
            "normal": int(((importance >= 0.25) & (importance < 0.75)).sum()),
            "important": int((importance >= 0.75).sum()),
        }
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"Saved LERF-Mask evaluation to {metrics_path}")


if __name__ == "__main__":
    main()
