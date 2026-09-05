"""Reproducible sequential-vs-joint benchmark on LERF-Mask ramen."""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


IMPORTANT = "egg,pork belly,wavy noodles in bowl"
NORMAL = "yellow bowl,chopsticks,glass of water"


def run(command, cwd):
    print("\n$", " ".join(map(str, command)), flush=True)
    started = time.monotonic()
    subprocess.run([str(value) for value in command], cwd=cwd, check=True)
    return time.monotonic() - started


def select_validation_views(paths, count=12):
    """Select deterministic, uniformly-spaced held-out training views."""
    paths = sorted(paths, key=lambda path: path.name)
    count = min(max(int(count), 1), max(len(paths) - 2, 0))
    if count < 1:
        return []
    indices = [
        min(len(paths) - 1, int((index + 0.5) * len(paths) / count))
        for index in range(count)
    ]
    return [paths[index] for index in sorted(set(indices))]


def load_json(path, default):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def save_json(path, payload):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def latest_checkpoint(model, maximum):
    candidates = []
    for path in Path(model).glob("chkpnt*.pth"):
        try:
            step = int(path.stem.replace("chkpnt", ""))
        except ValueError:
            continue
        if step < int(maximum):
            candidates.append((step, path))
    return max(candidates, default=(0, None))[1]


def semantic_training_complete(model, iteration, target):
    marker = (
        Path(model) / "semantic" / f"iteration_{iteration}"
        / "training_complete.json"
    )
    if not marker.is_file():
        return False
    try:
        return int(json.loads(marker.read_text())["semantic_iterations"]) >= int(target)
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False


def semantic_time_budget_complete(model, iteration, target):
    marker = (
        Path(model) / "semantic" / f"iteration_{iteration}"
        / "training_complete.json"
    )
    payload = load_json(marker, {})
    return bool(payload.get("stopped_by_time")) or int(
        payload.get("semantic_iterations", 0)
    ) >= int(target)


def detail_preprocessing_complete(scene):
    meta_path = Path(scene) / "semantic_meta.npz"
    maps = sorted((Path(scene) / "semantic_maps").glob("*.npz"))
    if not meta_path.is_file() or not maps:
        return False
    try:
        import numpy as np
        with np.load(meta_path) as meta:
            if "prototype_features" not in meta.files:
                return False
        with np.load(maps[0]) as semantic_map:
            required = {
                "detail_weight", "boundary", "thinness", "prototype_ids",
                "hierarchy_prototype_ids",
            }
            return required.issubset(semantic_map.files)
    except (OSError, ValueError):
        return False


def main():
    parser = argparse.ArgumentParser(description="Benchmark adaptive joint 3DGS on ramen")
    parser.add_argument("--scene", required=True)
    parser.add_argument("--sam_checkpoint", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--iterations", type=int, default=15000)
    parser.add_argument("--semantic_iterations", type=int, default=5000)
    parser.add_argument("--semantic_start", type=int, default=1000)
    parser.add_argument("--feature_dim", type=int, default=32)
    parser.add_argument("--feature_width", type=int, default=512)
    parser.add_argument("--validation_views", type=int, default=12)
    parser.add_argument("--validation_interval", type=int, default=1000)
    parser.add_argument("--early_stop_patience", type=int, default=4)
    parser.add_argument("--equal_time_semantic_cap", type=int, default=15000)
    parser.add_argument(
        "--no_equal_time", dest="equal_time", action="store_false",
        help="Use fixed semantic iterations instead of matching joint wall time",
    )
    parser.set_defaults(equal_time=True)
    parser.add_argument("--skip_preprocess", action="store_true")
    parser.add_argument("--skip_training", action="store_true")
    parser.add_argument("--skip_baseline", action="store_true")
    parser.add_argument("--skip_joint", action="store_true")
    parser.add_argument(
        "--resume", action="store_true",
        help="Reuse preprocessing, completed stages, and the latest training checkpoints",
    )
    args = parser.parse_args()

    if min(
        args.iterations, args.semantic_iterations, args.feature_dim,
        args.validation_views, args.validation_interval,
        args.equal_time_semantic_cap,
    ) < 1:
        parser.error("iteration and feature values must be positive")
    if args.early_stop_patience < 0:
        parser.error("early-stop patience must be non-negative")
    if not 0 <= args.semantic_start < args.iterations:
        parser.error("semantic start must be in [0, iterations)")
    repo = Path(__file__).resolve().parents[1]
    scene = Path(args.scene).resolve()
    output_root = Path(args.output_root).resolve()
    baseline = output_root / "sequential"
    joint = output_root / "joint"
    output_root.mkdir(parents=True, exist_ok=True)

    test_images = sorted((scene / "images").glob("test_*.*"))
    if len(test_images) < 3:
        parser.error("ramen/images must contain its official test_*.jpg views")
    test_file = scene / "sparse" / "0" / "test.txt"
    test_file.write_text("".join(f"{path.name}\n" for path in test_images), encoding="utf-8")
    print("Explicit test views:", [path.name for path in test_images])

    training_image_dir = scene / "images_train"
    training_images = sorted(
        path for path in training_image_dir.glob("*.*") if path.is_file()
    )
    if len(training_images) < 3:
        training_images = sorted(
            path for path in (scene / "images").glob("*.*")
            if path.is_file() and path.name not in {item.name for item in test_images}
        )
    validation_images = select_validation_views(
        training_images, args.validation_views
    )
    if not validation_images:
        parser.error("Unable to create a validation split from training images")
    val_file = scene / "sparse" / "0" / "val.txt"
    val_file.write_text(
        "".join(f"{path.name}\n" for path in validation_images), encoding="utf-8"
    )
    print("Validation views:", [path.name for path in validation_images])

    preprocessed = detail_preprocessing_complete(scene)
    if not args.skip_preprocess and not (args.resume and preprocessed):
        run([
            sys.executable, repo / "preprocess_semantics.py",
            "--scene", scene, "--images_subdir", "images_train",
            "--sam_checkpoint", Path(args.sam_checkpoint).resolve(), "--sam_model", "vit_h",
            "--clip_model", "ViT-H-14", "--clip_pretrained", "laion2b_s32b_b79k",
            "--feature_dim", args.feature_dim, "--feature_width", args.feature_width,
            "--max_masks", 192, "--points_per_side", 32, "--batch_size", 16,
            "--important", IMPORTANT, "--normal", NORMAL,
            "--cross_view_prototypes", 96, "--cross_view_weight", 0.72,
            "--boundary_width", 3, "--boundary_boost", 2.25,
            "--thin_boost", 1.50, "--thin_compactness", 0.40,
            "--thin_aspect_ratio", 2.5,
        ], repo)

    # Keep splitting through 75% of a 15k run, while leaving a final refinement
    # window. Detail tiers use lower thresholds and gentler opacity pruning.
    densify_until = min(12000, max(1000, int(args.iterations * 3 / 4)))
    save_iterations = sorted({
        args.iterations, min(7000, args.iterations),
        min(10000, args.iterations), min(15000, args.iterations)
    })
    checkpoint_iterations = [
        step for step in save_iterations if step < args.iterations
    ]
    validation_start = min(
        max(args.validation_interval, densify_until // 2),
        max(0, args.iterations - args.validation_interval),
    )
    common_train = [
        "-s", scene, "--eval", "--iterations", args.iterations,
        "--save_iterations", *save_iterations,
        "--test_iterations", args.iterations,
        "--disable_viewer", "--densify_from_iter", 500,
        "--densify_until_iter", densify_until, "--densify_grad_threshold", 0.00010,
        "--importance_mask_dir", scene / "importance_masks",
        "--validation_file", val_file,
        "--validation_interval", args.validation_interval,
        "--validation_start", validation_start,
        "--early_stop_patience", args.early_stop_patience,
        "--early_stop_min_delta", 0.02,
        "--select_best_validation",
    ]
    if checkpoint_iterations:
        common_train.extend(("--checkpoint_iterations", *checkpoint_iterations))
    timing_path = output_root / "training_times.json"
    timings = load_json(timing_path, {})

    def run_stage(name, command):
        elapsed = run(command, repo)
        timings[name] = float(elapsed)
        save_json(timing_path, timings)
        return elapsed

    if not args.skip_training:
        # Joint is intentionally run first: its measured optimizer wall time is
        # the budget granted to RGB + post-hoc semantic baseline training.
        if not args.skip_joint:
            joint_ply = (
                joint / "point_cloud" / f"iteration_{args.iterations}"
                / "point_cloud.ply"
            )
            joint_semantic = (
                joint / "semantic" / f"iteration_{args.iterations}"
                / "semantic_features.pt"
            )
            if not (args.resume and joint_ply.is_file() and joint_semantic.is_file()):
                command = [
                    sys.executable, repo / "train.py", "-m", joint,
                    *common_train,
                    "--joint_semantics", "--semantic_dir", scene / "semantic_maps",
                    "--sh_degree", 5, "--semantic_start", args.semantic_start,
                    "--semantic_weight", 0.22, "--semantic_lr", 0.01,
                    "--scale_gate_lr", 0.001,
                    "--rgb_tier_weights", 0.30, 1.20, 5.0,
                    "--semantic_tier_weights", 0.12, 1.25, 5.0,
                    "--tier_densify_multipliers", 1.25, 0.72, 0.35,
                    "--tier_opacity_multipliers", 1.25, 0.70, 0.25,
                    "--tier_sh_degrees", 1, 3, 5,
                    "--semantic_spatial_weight", 0.012,
                    "--semantic_spatial_every", 8,
                    "--semantic_spatial_samples", 768,
                    "--semantic_edge_sigma", 0.12,
                    "--semantic_cross_view_weight", 0.06,
                    "--semantic_chunks_per_step", 3,
                ]
                checkpoint = latest_checkpoint(joint, args.iterations)
                if args.resume and checkpoint is not None:
                    command.extend(("--start_checkpoint", checkpoint))
                run_stage("joint_train_seconds", command)
            else:
                print("Reusing completed joint model", flush=True)

        if not args.skip_baseline:
            baseline_ply = (
                baseline / "point_cloud" / f"iteration_{args.iterations}"
                / "point_cloud.ply"
            )
            if not (args.resume and baseline_ply.is_file()):
                command = [
                    sys.executable, repo / "train.py", "-m", baseline,
                    *common_train, "--foreground_weight", 3.0,
                    "--background_weight", 0.75,
                ]
                checkpoint = latest_checkpoint(baseline, args.iterations)
                if args.resume and checkpoint is not None:
                    command.extend(("--start_checkpoint", checkpoint))
                run_stage("sequential_rgb_seconds", command)
            else:
                print("Reusing completed sequential RGB model", flush=True)

            joint_seconds = timings.get("joint_train_seconds")
            rgb_seconds = timings.get("sequential_rgb_seconds", 0.0)
            equal_time_available = (
                args.equal_time and joint_seconds is not None
            )
            semantic_target = (
                args.equal_time_semantic_cap
                if equal_time_available else args.semantic_iterations
            )
            semantic_complete = (
                semantic_time_budget_complete(
                    baseline, args.iterations, semantic_target
                ) if equal_time_available else semantic_training_complete(
                    baseline, args.iterations, semantic_target
                )
            )
            if not (args.resume and semantic_complete):
                command = [
                    sys.executable, repo / "train_semantics.py", "-m", baseline,
                    "--iteration", args.iterations,
                    "--semantic_iterations", semantic_target,
                    "--semantic_lr", 0.005, "--spatial_weight", 0.02,
                    "--spatial_k", 8, "--spatial_samples", 4096,
                ]
                if equal_time_available:
                    remaining = max(1.0, float(joint_seconds) - float(rgb_seconds))
                    command.extend(("--max_seconds", f"{remaining:.3f}"))
                    timings["sequential_semantic_budget_seconds"] = remaining
                if args.resume:
                    command.append("--resume")
                run_stage("sequential_semantic_seconds", command)
            else:
                print("Reusing completed sequential semantic model", flush=True)

    results = {}
    for name, model in (("sequential", baseline), ("joint", joint)):
        output = output_root / f"eval_{name}"
        run([
            sys.executable, "-m", "scripts.evaluate_lerf_mask",
            "--model", model, "--test_mask", scene / "test_mask",
            "--iteration", args.iterations, "--threshold", 0.25,
            "--granularity", 1, "--output", output,
            "--important_labels", IMPORTANT, "--normal_labels", NORMAL,
        ], repo)
        results[name] = json.loads((output / "metrics.json").read_text())

    timings["sequential_total_seconds"] = float(
        timings.get("sequential_rgb_seconds", 0.0)
        + timings.get("sequential_semantic_seconds", 0.0)
    )
    if "joint_train_seconds" in timings:
        timings["wall_clock_delta_seconds"] = float(
            timings["sequential_total_seconds"]
            - timings["joint_train_seconds"]
        )
    save_json(timing_path, timings)

    summary = {
        "dataset": "LERF-Mask ramen",
        "important": IMPORTANT.split(","),
        "normal": NORMAL.split(","),
        "background": "all remaining pixels/regions",
        "iterations": args.iterations,
        "semantic_iterations_baseline": args.semantic_iterations,
        "protocol": {
            "equal_wall_clock": bool(args.equal_time),
            "requested_iteration_cap": args.iterations,
            "validation_views": len(validation_images),
            "validation_interval": args.validation_interval,
            "early_stop_patience": args.early_stop_patience,
            "timings_seconds": timings,
        },
        "sequential": {
            key: results["sequential"][key]
            for key in (
                "gaussians", "test_psnr", "test_ssim",
                "test_important_psnr", "test_normal_psnr",
                "mean_iou", "mean_boundary_iou",
            )
        },
        "joint": {
            key: results["joint"][key]
            for key in (
                "gaussians", "test_psnr", "test_ssim",
                "test_important_psnr", "test_normal_psnr",
                "mean_iou", "mean_boundary_iou", "tier_gaussians",
            )
            if key in results["joint"]
        },
    }
    for name in ("sequential", "joint"):
        validation = load_json(
            output_root / name / "validation_summary.json", {}
        )
        summary[name]["validation"] = validation
        per_label = results[name]["per_label_iou"]
        summary[name]["important_mean_iou"] = float(sum(
            per_label[label] for label in IMPORTANT.split(",")
        ) / len(IMPORTANT.split(",")))
        summary[name]["normal_mean_iou"] = float(sum(
            per_label[label] for label in NORMAL.split(",")
        ) / len(NORMAL.split(",")))
    summary["delta"] = {
        "gaussians": summary["joint"]["gaussians"] - summary["sequential"]["gaussians"],
        "test_psnr": summary["joint"]["test_psnr"] - summary["sequential"]["test_psnr"],
        "test_ssim": summary["joint"]["test_ssim"] - summary["sequential"]["test_ssim"],
        "test_important_psnr": (
            summary["joint"]["test_important_psnr"]
            - summary["sequential"]["test_important_psnr"]
        ),
        "test_normal_psnr": (
            summary["joint"]["test_normal_psnr"]
            - summary["sequential"]["test_normal_psnr"]
        ),
        "mean_iou": summary["joint"]["mean_iou"] - summary["sequential"]["mean_iou"],
        "mean_boundary_iou": (
            summary["joint"]["mean_boundary_iou"]
            - summary["sequential"]["mean_boundary_iou"]
        ),
        "important_mean_iou": (
            summary["joint"]["important_mean_iou"]
            - summary["sequential"]["important_mean_iou"]
        ),
        "normal_mean_iou": (
            summary["joint"]["normal_mean_iou"]
            - summary["sequential"]["normal_mean_iou"]
        ),
    }
    summary_path = output_root / "comparison.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Saved comparison to {summary_path}")


if __name__ == "__main__":
    main()
