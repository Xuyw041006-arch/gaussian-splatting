"""Reproducible sequential-vs-joint benchmark on LERF-Mask ramen."""

import argparse
import json
import subprocess
import sys
from pathlib import Path


IMPORTANT = "egg,pork belly,wavy noodles in bowl"
NORMAL = "yellow bowl,chopsticks,glass of water"


def run(command, cwd):
    print("\n$", " ".join(map(str, command)), flush=True)
    subprocess.run([str(value) for value in command], cwd=cwd, check=True)


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
    parser.add_argument("--skip_preprocess", action="store_true")
    parser.add_argument("--skip_training", action="store_true")
    parser.add_argument("--skip_baseline", action="store_true")
    parser.add_argument("--skip_joint", action="store_true")
    parser.add_argument(
        "--resume", action="store_true",
        help="Reuse preprocessing, completed stages, and the latest training checkpoints",
    )
    args = parser.parse_args()

    if min(args.iterations, args.semantic_iterations, args.feature_dim) < 1:
        parser.error("iteration and feature values must be positive")
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

    preprocessed = (scene / "semantic_meta.npz").is_file()
    if not args.skip_preprocess and not (args.resume and preprocessed):
        run([
            sys.executable, repo / "preprocess_semantics.py",
            "--scene", scene, "--images_subdir", "images_train",
            "--sam_checkpoint", Path(args.sam_checkpoint).resolve(), "--sam_model", "vit_h",
            "--clip_model", "ViT-H-14", "--clip_pretrained", "laion2b_s32b_b79k",
            "--feature_dim", args.feature_dim, "--feature_width", args.feature_width,
            "--max_masks", 192, "--points_per_side", 32, "--batch_size", 16,
            "--important", IMPORTANT, "--normal", NORMAL,
            "--cross_view_prototypes", 64, "--cross_view_weight", 0.65,
        ], repo)

    # Finish densification before the final third of training so newly split
    # Gaussians still receive enough RGB and semantic refinement.  This keeps
    # the original 15k cutoff for 30k runs and uses 10k for the 15k default.
    densify_until = min(15000, max(1000, int(args.iterations * 2 / 3)))
    save_iterations = sorted({
        args.iterations, min(7000, args.iterations),
        min(10000, args.iterations), min(15000, args.iterations)
    })
    checkpoint_iterations = [
        step for step in save_iterations if step < args.iterations
    ]
    common_train = [
        "-s", scene, "--eval", "--iterations", args.iterations,
        "--save_iterations", *save_iterations,
        "--test_iterations", args.iterations,
        "--disable_viewer", "--densify_from_iter", 500,
        "--densify_until_iter", densify_until, "--densify_grad_threshold", 0.00015,
        "--importance_mask_dir", scene / "importance_masks",
    ]
    if checkpoint_iterations:
        common_train.extend(("--checkpoint_iterations", *checkpoint_iterations))
    if not args.skip_training:
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
                run(command, repo)
            else:
                print("Reusing completed sequential RGB model", flush=True)
            if not (
                args.resume and semantic_training_complete(
                    baseline, args.iterations, args.semantic_iterations
                )
            ):
                command = [
                    sys.executable, repo / "train_semantics.py", "-m", baseline,
                    "--iteration", args.iterations,
                    "--semantic_iterations", args.semantic_iterations,
                    "--semantic_lr", 0.005, "--spatial_weight", 0.02,
                    "--spatial_k", 8, "--spatial_samples", 4096,
                ]
                if args.resume:
                    command.append("--resume")
                run(command, repo)
            else:
                print("Reusing completed sequential semantic model", flush=True)
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
                    "--semantic_weight", 0.20, "--semantic_lr", 0.01,
                    "--scale_gate_lr", 0.001,
                    "--rgb_tier_weights", 0.35, 1.0, 4.0,
                    "--semantic_tier_weights", 0.15, 1.0, 4.0,
                    "--tier_densify_multipliers", 1.80, 1.0, 0.55,
                    "--tier_opacity_multipliers", 2.0, 1.0, 0.5,
                    "--tier_sh_degrees", 1, 3, 5,
                    "--semantic_spatial_weight", 0.02,
                    "--semantic_spatial_every", 8,
                    "--semantic_spatial_samples", 512,
                    "--semantic_chunks_per_step", 3,
                ]
                checkpoint = latest_checkpoint(joint, args.iterations)
                if args.resume and checkpoint is not None:
                    command.extend(("--start_checkpoint", checkpoint))
                run(command, repo)
            else:
                print("Reusing completed joint model", flush=True)

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

    summary = {
        "dataset": "LERF-Mask ramen",
        "important": IMPORTANT.split(","),
        "normal": NORMAL.split(","),
        "background": "all remaining pixels/regions",
        "iterations": args.iterations,
        "semantic_iterations_baseline": args.semantic_iterations,
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
