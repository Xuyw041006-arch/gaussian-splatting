"""Evaluate open-vocabulary Gaussian masks on the LERF-Mask protocol."""

import json
from argparse import ArgumentParser
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from gaussian_renderer import render
from interactive_renderer import extract_dataset_and_pipeline, read_model_config
from scene import GaussianModel, Scene
from semantic.artifact import (
    apply_scale_gate, cosine_scores, decode_features, project_clip_feature,
)
from utils.image_utils import psnr
from utils.loss_utils import ssim


def mask_iou(prediction, target):
    prediction = np.asarray(prediction, dtype=bool)
    target = np.asarray(target, dtype=bool)
    union = np.logical_or(prediction, target).sum()
    return float(np.logical_and(prediction, target).sum() / max(int(union), 1))


def mask_boundary(mask, dilation_ratio=0.008):
    mask = np.asarray(mask, dtype=np.uint8)
    radius = max(1, int(round(dilation_ratio * np.hypot(*mask.shape))))
    kernel = np.ones((3, 3), dtype=np.uint8)
    eroded = cv2.erode(mask, kernel, iterations=radius)
    return mask.astype(bool) & ~eroded.astype(bool)


def boundary_iou(prediction, target, dilation_ratio=0.008):
    return mask_iou(
        mask_boundary(prediction, dilation_ratio),
        mask_boundary(target, dilation_ratio),
    )


def camera_for_split(cameras, split_name):
    by_stem = {Path(camera.image_name).stem: camera for camera in cameras}
    for candidate in (f"test_{split_name}", split_name):
        if candidate in by_stem:
            return by_stem[candidate]
    index = int(split_name)
    ordered = sorted(cameras, key=lambda camera: camera.image_name)
    if index >= len(ordered):
        raise KeyError(f"No test camera for mask split {split_name}")
    return ordered[index]


def masked_psnr(rendered, target, mask):
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
    parser.add_argument("--threshold", type=float, default=0.25)
    parser.add_argument("--granularity", type=int, choices=[0, 1, 2], default=1)
    parser.add_argument("--boundary_ratio", type=float, default=0.008)
    parser.add_argument("--important_labels", default="")
    parser.add_argument("--normal_labels", default="")
    parser.add_argument("--output", default="")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    model_path = Path(args.model).resolve()
    test_mask_root = Path(args.test_mask).resolve()
    artifact_path = (
        model_path / "semantic" / f"iteration_{args.iteration}" / "semantic_features.pt"
    )
    if not artifact_path.is_file() or not test_mask_root.is_dir():
        parser.error("Missing semantic artifact or LERF-Mask directory")

    config = read_model_config(model_path)
    config.eval = True
    dataset, pipeline = extract_dataset_and_pipeline(config)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False)
    cameras = scene.getTestCameras()
    if not cameras:
        raise RuntimeError("No test cameras; create sparse/0/test.txt before training")

    artifact = torch.load(artifact_path, map_location="cpu")
    encoded = artifact["features"].float().numpy()
    if len(encoded) != gaussians.get_xyz.shape[0]:
        raise RuntimeError("RGB and semantic Gaussian counts do not match")
    encoded = apply_scale_gate(encoded, artifact, args.granularity)
    decoded = decode_features(
        encoded, artifact["feature_min"].numpy(), artifact["feature_max"].numpy()
    )

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

    labels = sorted({path.stem for path in test_mask_root.glob("*/*.png")})
    with torch.no_grad():
        text = torch.nn.functional.normalize(
            clip_model.encode_text(tokenizer(labels).to(device)).float(), dim=-1
        ).cpu().numpy()
    queries = np.stack([
        project_clip_feature(
            feature, artifact["pca_mean"].numpy(), artifact["pca_components"].numpy()
        ) for feature in text
    ])
    scores = {
        label: cosine_scores(decoded, query) for label, query in zip(labels, queries)
    }

    output_dir = Path(args.output).resolve() if args.output else model_path / "lerf_mask_eval"
    output_dir.mkdir(parents=True, exist_ok=True)
    important_labels = {
        value.strip() for value in args.important_labels.split(",") if value.strip()
    }
    normal_labels = {
        value.strip() for value in args.normal_labels.split(",") if value.strip()
    }
    background = torch.zeros(3, dtype=torch.float32, device="cuda")
    rows = []
    reconstruction_rows = []
    for split in sorted(path for path in test_mask_root.iterdir() if path.is_dir()):
        camera = camera_for_split(cameras, split.name)
        target_masks = {}
        for target_path in sorted(split.glob("*.png")):
            image = Image.open(target_path).convert("L").resize(
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
                "psnr": float(psnr(rgb_render, ground_truth).mean().item()),
                "ssim": float(ssim(rgb_render, ground_truth).item()),
            }
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
        for target_path in sorted(split.glob("*.png")):
            label = target_path.stem
            values = torch.from_numpy(scores[label]).float().cuda()
            colors = values[:, None].repeat(1, 3).clamp(0, 1)
            with torch.no_grad():
                score_render = render(
                    camera, gaussians, pipeline, background, override_color=colors
                )["render"][0]
                score_render = score_render / alpha.clamp_min(1e-4)
            prediction = (score_render >= args.threshold).cpu().numpy()
            target = target_masks[label]
            Image.fromarray(prediction.astype(np.uint8) * 255).save(
                output_dir / f"{split.name}_{label}.png"
            )
            rows.append({
                "split": split.name,
                "camera": camera.image_name,
                "label": label,
                "iou": mask_iou(prediction, target),
                "boundary_iou": boundary_iou(
                    prediction, target, args.boundary_ratio
                ),
            })

    result = {
        "model": str(model_path),
        "iteration": args.iteration,
        "threshold": args.threshold,
        "granularity": args.granularity,
        "gaussians": int(len(encoded)),
        "test_psnr": float(np.mean([row["psnr"] for row in reconstruction_rows])),
        "test_ssim": float(np.mean([row["ssim"] for row in reconstruction_rows])),
        "mean_iou": float(np.mean([row["iou"] for row in rows])),
        "mean_boundary_iou": float(np.mean([row["boundary_iou"] for row in rows])),
        "per_label_iou": {
            label: float(np.mean([row["iou"] for row in rows if row["label"] == label]))
            for label in labels
        },
        "reconstruction_rows": reconstruction_rows,
        "rows": rows,
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
