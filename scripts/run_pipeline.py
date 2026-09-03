"""Run or print the complete custom-data 3DGS + semantics pipeline."""

import argparse
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Step:
    name: str
    command: list
    expected_output: Path


def build_steps(args):
    repo = Path(__file__).resolve().parents[1]
    python = args.python
    scene = Path(args.scene).resolve()
    model = Path(args.model).resolve()
    scene_iteration = args.scene_iterations
    max_views = args.max_train_views
    if args.sparse and max_views < 0:
        max_views = 8

    colmap = [python, str(repo / "convert.py"), "-s", str(scene), "--matcher", args.matcher]
    if args.no_colmap_gpu:
        colmap.append("--no_gpu")

    semantics = [
        python, str(repo / "preprocess_semantics.py"), "--scene", str(scene),
        "--images_subdir", args.semantic_images,
        "--sam_checkpoint", str(Path(args.sam_checkpoint).resolve()),
        "--sam_model", args.sam_model, "--feature_dim", str(args.feature_dim),
        "--feature_width", str(args.feature_width),
        "--clip_model", args.clip_model, "--clip_pretrained", args.clip_pretrained,
        "--max_masks", str(args.max_masks),
        "--points_per_side", str(args.points_per_side),
        "--batch_size", str(args.clip_batch_size),
        "--cross_view_prototypes", str(args.cross_view_prototypes),
        "--cross_view_weight", str(args.cross_view_weight),
    ]
    if args.important:
        semantics.extend(["--important", args.important])
    if args.important_json:
        semantics.extend(["--important_json", str(Path(args.important_json).resolve())])
    if args.normal:
        semantics.extend(["--normal", args.normal])
    if args.normal_json:
        semantics.extend(["--normal_json", str(Path(args.normal_json).resolve())])

    rgb = [
        python, str(repo / "train.py"), "-s", str(scene), "-m", str(model),
        "--iterations", str(scene_iteration), "--save_iterations", str(scene_iteration),
        "--test_iterations", str(scene_iteration), "--disable_viewer",
        "--max_train_views", str(max_views), "--view_stride", str(args.view_stride),
        "--densify_from_iter", str(args.densify_from_iter),
        "--densify_until_iter", str(
            min(args.densify_until_iter, max(1, int(scene_iteration * 0.75)))
        ),
        "--densify_grad_threshold", str(args.densify_grad_threshold),
    ]
    if args.training_mode == "joint" or args.important or args.important_json:
        rgb.extend([
            "--importance_mask_dir", str(scene / "importance_masks"),
            "--foreground_weight", str(args.foreground_weight),
            "--background_weight", str(args.background_weight),
        ])
    if args.training_mode == "joint":
        rgb.extend([
            "--joint_semantics", "--semantic_dir", str(scene / "semantic_maps"),
            "--sh_degree", str(args.joint_sh_degree),
            "--semantic_start", str(args.semantic_start),
            "--semantic_weight", str(args.joint_semantic_weight),
            "--semantic_lr", str(args.semantic_lr),
            "--scale_gate_lr", str(args.scale_gate_lr),
            "--semantic_spatial_weight", str(args.spatial_weight),
            "--semantic_spatial_every", str(args.spatial_every),
            "--semantic_spatial_samples", str(args.joint_spatial_samples),
            "--importance_ema", str(args.importance_ema),
            "--rgb_tier_weights", *map(str, args.rgb_tier_weights),
            "--semantic_tier_weights", *map(str, args.semantic_tier_weights),
            "--tier_densify_multipliers", *map(str, args.tier_densify_multipliers),
            "--tier_opacity_multipliers", *map(str, args.tier_opacity_multipliers),
            "--tier_sh_degrees", *map(str, args.tier_sh_degrees),
        ])
    if args.depths:
        rgb.extend(["-d", args.depths])
    if args.sparse:
        rgb.extend([
            "--random_background", "--opacity_reset_interval", "1000",
        ])

    semantic_train = [
        python, str(repo / "train_semantics.py"), "-m", str(model),
        "--iteration", str(scene_iteration),
        "--semantic_iterations", str(args.semantic_iterations),
        "--semantic_lr", str(args.semantic_lr),
        "--spatial_weight", str(args.spatial_weight),
        "--spatial_k", str(args.spatial_k),
        "--spatial_samples", str(args.spatial_samples),
    ]
    common = [
        Step("colmap", colmap, scene / "sparse" / "0" / "images.bin"),
        Step("semantics", semantics, scene / "semantic_meta.npz"),
    ]
    if args.training_mode == "joint":
        return common + [Step(
            "joint", rgb,
            model / "semantic" / f"iteration_{scene_iteration}" / "semantic_features.pt",
        )]
    return common + [
        Step(
            "rgb", rgb,
            model / "point_cloud" / f"iteration_{scene_iteration}" / "point_cloud.ply",
        ),
        Step(
            "semantic", semantic_train,
            model / "semantic" / f"iteration_{scene_iteration}" / "semantic_features.pt",
        ),
    ]


def make_parser():
    parser = argparse.ArgumentParser(description="3DGS semantic project pipeline")
    parser.add_argument("--scene", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--sam_checkpoint", required=True)
    parser.add_argument("--semantic_images", default="images")
    parser.add_argument("--sam_model", choices=["vit_b", "vit_l", "vit_h"], default="vit_h")
    parser.add_argument("--important", default="")
    parser.add_argument("--important_json", default="")
    parser.add_argument("--normal", default="")
    parser.add_argument("--normal_json", default="")
    parser.add_argument(
        "--training_mode", choices=["joint", "sequential"], default="joint"
    )
    parser.add_argument("--foreground_weight", type=float, default=3.0)
    parser.add_argument("--background_weight", type=float, default=0.75)
    parser.add_argument("--clip_model", default="ViT-H-14")
    parser.add_argument("--clip_pretrained", default="laion2b_s32b_b79k")
    parser.add_argument("--feature_dim", type=int, default=32)
    parser.add_argument("--feature_width", type=int, default=512)
    parser.add_argument("--max_masks", type=int, default=192)
    parser.add_argument("--points_per_side", type=int, default=32)
    parser.add_argument("--clip_batch_size", type=int, default=16)
    parser.add_argument("--scene_iterations", type=int, default=40000)
    parser.add_argument("--semantic_iterations", type=int, default=8000)
    parser.add_argument("--semantic_lr", type=float, default=0.005)
    parser.add_argument("--spatial_weight", type=float, default=0.02)
    parser.add_argument("--spatial_k", type=int, default=8)
    parser.add_argument("--spatial_samples", type=int, default=4096)
    parser.add_argument("--cross_view_prototypes", type=int, default=64)
    parser.add_argument("--cross_view_weight", type=float, default=0.65)
    parser.add_argument("--joint_sh_degree", type=int, default=5)
    parser.add_argument("--semantic_start", type=int, default=1000)
    parser.add_argument("--joint_semantic_weight", type=float, default=0.15)
    parser.add_argument("--scale_gate_lr", type=float, default=0.001)
    parser.add_argument("--spatial_every", type=int, default=8)
    parser.add_argument("--joint_spatial_samples", type=int, default=512)
    parser.add_argument("--importance_ema", type=float, default=0.90)
    parser.add_argument("--rgb_tier_weights", nargs=3, type=float, default=(0.35, 1.0, 4.0))
    parser.add_argument("--semantic_tier_weights", nargs=3, type=float, default=(0.15, 1.0, 4.0))
    parser.add_argument("--tier_densify_multipliers", nargs=3, type=float, default=(1.80, 1.0, 0.55))
    parser.add_argument("--tier_opacity_multipliers", nargs=3, type=float, default=(2.0, 1.0, 0.5))
    parser.add_argument("--tier_sh_degrees", nargs=3, type=int, default=(1, 3, 5))
    parser.add_argument("--densify_from_iter", type=int, default=500)
    parser.add_argument("--densify_until_iter", type=int, default=22000)
    parser.add_argument("--densify_grad_threshold", type=float, default=0.0001)
    parser.add_argument("--sparse", action="store_true")
    parser.add_argument("--max_train_views", type=int, default=-1)
    parser.add_argument("--view_stride", type=int, default=1)
    parser.add_argument("--depths", default="", help="Depth directory name inside scene")
    parser.add_argument("--matcher", choices=["exhaustive", "sequential"], default="exhaustive")
    parser.add_argument("--no_colmap_gpu", action="store_true")
    parser.add_argument(
        "--stages", default="all",
        help="Comma-separated subset: colmap,semantics,joint,rgb,semantic; default all",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--python", default=sys.executable)
    return parser


def main():
    parser = make_parser()
    args = parser.parse_args()
    selected = {name.strip() for name in args.stages.split(",") if name.strip()}
    allowed = {"colmap", "semantics", "joint", "rgb", "semantic"}
    if selected == {"all"}:
        selected = {step.name for step in build_steps(args)}
    if not selected or not selected <= allowed:
        parser.error(f"--stages must use only: {','.join(sorted(allowed))}")
    if args.view_stride < 1 or args.max_train_views == 0 or args.max_train_views == 1:
        parser.error("--view_stride must be >=1; --max_train_views must be -1 or >=2")
    if min(
        args.feature_dim, args.feature_width, args.max_masks, args.points_per_side,
        args.clip_batch_size, args.scene_iterations, args.semantic_iterations,
        args.spatial_k, args.spatial_samples, args.densify_from_iter,
        args.densify_until_iter, args.cross_view_prototypes,
        args.spatial_every, args.joint_spatial_samples,
    ) < 1:
        parser.error("Training counts, dimensions, and sampling values must be positive")
    if min(args.foreground_weight, args.background_weight, args.semantic_lr) <= 0:
        parser.error("RGB weights and semantic learning rate must be positive")
    if args.spatial_weight < 0 or args.densify_grad_threshold <= 0:
        parser.error("Spatial weight must be non-negative and densify threshold positive")
    if not 0 <= args.cross_view_weight <= 1 or not 0 <= args.importance_ema < 1:
        parser.error("Cross-view weight must be in [0, 1] and importance EMA in [0, 1)")
    if max(args.tier_sh_degrees) > args.joint_sh_degree or min(args.tier_sh_degrees) < 0:
        parser.error("Tier SH degrees must be within [0, --joint_sh_degree]")
    adaptive_weights = (
        list(args.rgb_tier_weights) + list(args.semantic_tier_weights)
        + list(args.tier_densify_multipliers)
        + list(args.tier_opacity_multipliers)
        + [args.joint_semantic_weight, args.scale_gate_lr]
    )
    if min(adaptive_weights) <= 0:
        parser.error("Joint tier weights, multipliers, and learning rates must be positive")

    repo = Path(__file__).resolve().parents[1]
    for step in build_steps(args):
        if step.name not in selected:
            continue
        print(f"\n[{step.name}] {shlex.join(step.command)}", flush=True)
        if args.resume and step.expected_output.exists():
            print(f"skip: output exists at {step.expected_output}")
            continue
        if not args.dry_run:
            subprocess.run(step.command, cwd=repo, check=True)
    print("\nPipeline complete." if not args.dry_run else "\nDry run complete; nothing was changed.")


if __name__ == "__main__":
    main()
