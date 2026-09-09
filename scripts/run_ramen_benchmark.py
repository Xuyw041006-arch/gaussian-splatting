"""Reproducible sequential-vs-joint benchmark on LERF-Mask ramen."""

import argparse
import json
import math
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


def estimate_completed_training_seconds(model, iteration):
    """Estimate an artifact interval, never a certified training duration.

    Copies, restarts and downtime can change this interval. Keep it separate
    from measured durations when recovering models whose trainer did not exit.
    """
    model = Path(model)
    starts = [model / "cfg_args", model / "cameras.json"]
    ends = [
        model / "point_cloud" / f"iteration_{iteration}" / "point_cloud.ply",
        model / "semantic" / f"iteration_{iteration}" / "semantic_features.pt",
    ]
    starts = [path for path in starts if path.is_file()]
    ends = [path for path in ends if path.is_file()]
    if not starts or not ends:
        return None
    elapsed = max(path.stat().st_mtime for path in ends) - min(
        path.stat().st_mtime for path in starts
    )
    return float(elapsed) if elapsed > 0 else None


def valid_seconds(value):
    return (
        isinstance(value, (int, float)) and not isinstance(value, bool)
        and math.isfinite(value) and value >= 0
    )


def record_stage_timing(timings, name, elapsed, resumed=False, completed=True):
    """Preserve known runtime, but never turn a resumed segment into a total."""
    previous = timings.get(name)
    history_known = not resumed or (
        valid_seconds(previous)
        and timings.get(name + "_timing_complete") is True
        and not timings.get(name + "_interrupted_unknown")
    )
    observed = timings.get(name + "_observed_seconds", 0.0) if resumed else 0.0
    timings[name + "_observed_seconds"] = float(observed or 0.0) + elapsed
    timings[name] = (float(previous or 0.0) if resumed else 0.0) + elapsed if history_known else None
    timings[name + "_timing_complete"] = history_known
    timings[name + "_completed"] = completed
    timings[name + "_source"] = "measured_subprocess" if history_known else "resumed_missing_history"
    timings[name + "_run_in_progress"] = False


def timing_protocol(timings, requested):
    """Distinguish a requested budget from a verifiable equal-time result."""
    names = ("joint_train_seconds", "sequential_rgb_seconds", "sequential_semantic_seconds")
    missing = [name for name in names if not (
        valid_seconds(timings.get(name))
        and timings.get(name + "_timing_complete") is True
        and timings.get(name + "_completed") is True
    )]
    known = not missing
    total = sum(timings[name] for name in names[1:]) if not any(
        name in missing for name in names[1:]
    ) else None
    delta = total - timings[names[0]] if known else None
    timings["sequential_total_seconds"] = total
    timings["wall_clock_delta_seconds"] = delta
    # Process setup, checkpoint serialization and a final optimizer step can
    # slightly overshoot. Report the tolerance explicitly, never hide the delta.
    tolerance = max(5.0, 0.02 * timings[names[0]]) if known else None
    certified = bool(requested and known and abs(delta) <= tolerance)
    if not requested:
        reason = "fixed_iteration_protocol"
    elif missing:
        reason = "unverified_or_missing_duration: " + ", ".join(missing)
    elif not certified:
        reason = "measured_durations_do_not_match"
    else:
        reason = "measured_durations_match_within_tolerance"
    return {
        "equal_wall_clock_requested": bool(requested),
        "equal_wall_clock": certified,
        "timing_status": reason,
        "equal_time_tolerance_seconds": tolerance,
    }


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


def detail_preprocessing_complete(scene, heldout_names=None):
    meta_path = Path(scene) / "semantic_meta.npz"
    maps = sorted((Path(scene) / "semantic_maps").glob("*.npz"))
    if not meta_path.is_file() or not maps:
        return False
    try:
        import numpy as np
        with np.load(meta_path) as meta:
            if "prototype_features" not in meta.files:
                return False
            if heldout_names is not None:
                if "heldout_image_names" not in meta.files or "fit_image_names" not in meta.files:
                    return False
                expected = set(heldout_names)
                if set(meta["heldout_image_names"].tolist()) != expected:
                    return False
                if expected.intersection(meta["fit_image_names"].tolist()):
                    return False
        with np.load(maps[0]) as semantic_map:
            required = {
                "detail_weight", "boundary", "thinness", "prototype_ids",
                "hierarchy_prototype_ids", "region_ids",
                "hierarchy_region_ids",
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
    parser.add_argument("--semantic_ramp_iterations", type=int, default=2500)
    parser.add_argument("--feature_dim", type=int, default=32)
    parser.add_argument("--feature_width", type=int, default=512)
    parser.add_argument(
        "--joint_sh_degree", type=int, default=3, choices=range(4),
        help="New joint training SH degree supported by the stock CUDA rasterizer (0-3)",
    )
    parser.add_argument(
        "--tier_sh_degrees", type=int, nargs=3, default=[1, 2, 3],
        metavar=("BACKGROUND", "NORMAL", "IMPORTANT"),
        help="New joint training SH degrees for each importance tier",
    )
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
    if args.skip_baseline and args.skip_joint:
        parser.error("At least one of baseline or joint must be selected")
    if any(not 0 <= degree <= args.joint_sh_degree for degree in args.tier_sh_degrees):
        parser.error("tier SH degrees must be between 0 and --joint_sh_degree (at most 3)")

    if min(
        args.iterations, args.semantic_iterations, args.feature_dim,
        args.validation_views, args.validation_interval,
        args.equal_time_semantic_cap,
    ) < 1:
        parser.error("iteration and feature values must be positive")
    if args.early_stop_patience < 0:
        parser.error("early-stop patience must be non-negative")
    if not 0 <= args.semantic_start < args.iterations or args.semantic_ramp_iterations < 0:
        parser.error("semantic start/ramp must fit a non-negative curriculum")
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

    preprocessed = detail_preprocessing_complete(scene, [p.name for p in validation_images])
    if not args.skip_preprocess and not (args.resume and preprocessed):
        run([
            sys.executable, repo / "preprocess_semantics.py",
            "--scene", scene, "--images_subdir", "images_train",
            "--fit_exclude_list", val_file,
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

    def run_stage(name, command, resumed=False):
        if resumed and timings.get(name + "_run_in_progress"):
            timings[name + "_interrupted_unknown"] = True
        timings[name + "_run_in_progress"] = True
        save_json(timing_path, timings)
        started = time.monotonic()
        try:
            elapsed = run(command, repo)
        except (subprocess.CalledProcessError, KeyboardInterrupt):
            record_stage_timing(timings, name, time.monotonic() - started, resumed, False)
            save_json(timing_path, timings)
            raise
        record_stage_timing(timings, name, elapsed, resumed)
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
                    "--sh_degree", args.joint_sh_degree, "--semantic_start", args.semantic_start,
                    "--semantic_ramp_iterations", args.semantic_ramp_iterations,
                    "--semantic_weight", 0.22, "--semantic_lr", 0.01,
                    "--scale_gate_lr", 0.001,
                    "--rgb_tier_weights", 0.30, 1.20, 5.0,
                    "--semantic_tier_weights", 0.12, 1.25, 5.0,
                    "--tier_densify_multipliers", 1.25, 0.72, 0.35,
                    "--tier_opacity_multipliers", 1.25, 0.70, 0.25,
                    "--tier_sh_degrees", *args.tier_sh_degrees,
                    "--semantic_spatial_weight", 0.012,
                    "--semantic_spatial_every", 8,
                    "--semantic_spatial_samples", 768,
                    "--semantic_edge_sigma", 0.12,
                    "--semantic_cross_view_weight", 0.08,
                    "--semantic_boundary_weight", 0.08,
                    "--semantic_contrastive_weight", 0.05,
                    "--semantic_contrastive_samples", 320,
                    "--semantic_contrastive_every", 4,
                    "--semantic_chunks_per_step", 3,
                ]
                checkpoint = latest_checkpoint(joint, args.iterations)
                if args.resume and checkpoint is not None:
                    command.extend(("--start_checkpoint", checkpoint))
                run_stage("joint_train_seconds", command, bool(args.resume and checkpoint))
            else:
                print("Reusing completed joint model", flush=True)
                if "joint_train_seconds" not in timings:
                    recovered = estimate_completed_training_seconds(
                        joint, args.iterations
                    )
                    if recovered is not None:
                        timings["joint_train_seconds"] = recovered
                        timings["joint_train_seconds_recovered"] = True
                        timings["joint_train_seconds_timing_complete"] = False
                        timings["joint_train_seconds_source"] = "artifact_interval_estimate"
                        save_json(timing_path, timings)
                        print(
                            f"Estimated joint artifact interval: {recovered:.3f}s (unverified)",
                            flush=True,
                        )

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
                run_stage("sequential_rgb_seconds", command, bool(args.resume and checkpoint))
            else:
                print("Reusing completed sequential RGB model", flush=True)

            joint_seconds = timings.get("joint_train_seconds")
            rgb_seconds = timings.get("sequential_rgb_seconds")
            equal_time_available = (
                args.equal_time and valid_seconds(joint_seconds)
                and valid_seconds(rgb_seconds) and joint_seconds > rgb_seconds
            )
            if args.equal_time and not equal_time_available:
                print(
                    "Equal-time budget unavailable (missing history or RGB already "
                    "exceeds budget); using fixed semantic iterations. The comparison "
                    "will not certify equal wall time.", flush=True,
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
                    remaining = float(joint_seconds) - float(rgb_seconds)
                    command.extend(("--max_seconds", f"{remaining:.3f}"))
                    timings["sequential_semantic_budget_seconds"] = remaining
                if args.resume:
                    command.append("--resume")
                semantic_checkpoint = (
                    baseline / "semantic" / f"iteration_{args.iterations}"
                    / "semantic_checkpoint.pt"
                )
                run_stage(
                    "sequential_semantic_seconds", command,
                    args.resume and semantic_checkpoint.is_file(),
                )
            else:
                print("Reusing completed sequential semantic model", flush=True)

    results = {}
    for name, model in (("sequential", baseline), ("joint", joint)):
        if (name == "sequential" and args.skip_baseline) or (name == "joint" and args.skip_joint):
            continue
        output = output_root / f"eval_{name}"
        run([
            sys.executable, "-m", "scripts.evaluate_lerf_mask",
            "--model", model, "--test_mask", scene / "test_mask",
            "--iteration", args.iterations, "--threshold", 0.25,
            "--granularity", 1, "--output", output,
            "--important_labels", IMPORTANT, "--normal_labels", NORMAL,
        ], repo)
        results[name] = json.loads((output / "metrics.json").read_text())

    protocol = timing_protocol(timings, args.equal_time)
    save_json(timing_path, timings)

    summary = {
        "dataset": "LERF-Mask ramen",
        "important": IMPORTANT.split(","),
        "normal": NORMAL.split(","),
        "background": "all remaining pixels/regions",
        "iterations": args.iterations,
        "semantic_iterations_baseline": args.semantic_iterations,
        "protocol": {
            **protocol,
            "requested_iteration_cap": args.iterations,
            "validation_views": len(validation_images),
            "validation_interval": args.validation_interval,
            "early_stop_patience": args.early_stop_patience,
            "timings_seconds": timings,
        },
    }
    for name in results:
        summary[name] = {
            key: results[name][key]
            for key in (
                "gaussians", "test_psnr", "test_ssim",
                "test_important_psnr", "test_normal_psnr",
                "mean_iou", "mean_boundary_iou", "tier_gaussians",
            ) if key in results[name]
        }
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
    if len(results) == 2:
        summary["delta"] = {
            key: summary["joint"][key] - summary["sequential"][key]
            for key in (
                "gaussians", "test_psnr", "test_ssim", "test_important_psnr",
                "test_normal_psnr", "mean_iou", "mean_boundary_iou",
                "important_mean_iou", "normal_mean_iou",
            ) if key in summary["joint"] and key in summary["sequential"]
        }
        summary_path = output_root / "comparison.json"
    else:
        summary["partial_evaluation"] = True
        summary_path = output_root / f"comparison_{next(iter(results))}.json"
    save_json(summary_path, summary)
    print(json.dumps(summary, indent=2))
    print(f"Saved comparison to {summary_path}")


if __name__ == "__main__":
    main()
