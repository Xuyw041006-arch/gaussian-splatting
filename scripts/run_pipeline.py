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
        "--sam_checkpoint", str(Path(args.sam_checkpoint).resolve()),
        "--sam_model", args.sam_model, "--feature_dim", str(args.feature_dim),
        "--feature_width", str(args.feature_width),
    ]
    if args.important:
        semantics.extend(["--important", args.important])
    if args.important_json:
        semantics.extend(["--important_json", str(Path(args.important_json).resolve())])

    rgb = [
        python, str(repo / "train.py"), "-s", str(scene), "-m", str(model),
        "--iterations", str(scene_iteration), "--save_iterations", str(scene_iteration),
        "--test_iterations", str(scene_iteration), "--disable_viewer",
        "--max_train_views", str(max_views), "--view_stride", str(args.view_stride),
    ]
    if args.important or args.important_json:
        rgb.extend([
            "--importance_mask_dir", str(scene / "importance_masks"),
            "--foreground_weight", str(args.foreground_weight),
            "--background_weight", str(args.background_weight),
        ])
    if args.depths:
        rgb.extend(["-d", args.depths])
    if args.sparse:
        rgb.extend([
            "--random_background", "--densify_from_iter", "100",
            "--densify_grad_threshold", "0.0001",
            "--densify_until_iter", str(min(15000, int(scene_iteration * 0.75))),
            "--opacity_reset_interval", "1000",
        ])

    semantic_train = [
        python, str(repo / "train_semantics.py"), "-m", str(model),
        "--iteration", str(scene_iteration),
        "--semantic_iterations", str(args.semantic_iterations),
        "--semantic_lr", str(args.semantic_lr),
    ]
    return [
        Step("colmap", colmap, scene / "sparse" / "0" / "images.bin"),
        Step("semantics", semantics, scene / "semantic_meta.npz"),
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
    parser.add_argument("--sam_model", choices=["vit_b", "vit_l", "vit_h"], default="vit_b")
    parser.add_argument("--important", default="")
    parser.add_argument("--important_json", default="")
    parser.add_argument("--foreground_weight", type=float, default=4.0)
    parser.add_argument("--background_weight", type=float, default=0.25)
    parser.add_argument("--feature_dim", type=int, default=12)
    parser.add_argument("--feature_width", type=int, default=320)
    parser.add_argument("--scene_iterations", type=int, default=30000)
    parser.add_argument("--semantic_iterations", type=int, default=5000)
    parser.add_argument("--semantic_lr", type=float, default=0.01)
    parser.add_argument("--sparse", action="store_true")
    parser.add_argument("--max_train_views", type=int, default=-1)
    parser.add_argument("--view_stride", type=int, default=1)
    parser.add_argument("--depths", default="", help="Depth directory name inside scene")
    parser.add_argument("--matcher", choices=["exhaustive", "sequential"], default="exhaustive")
    parser.add_argument("--no_colmap_gpu", action="store_true")
    parser.add_argument(
        "--stages", default="colmap,semantics,rgb,semantic",
        help="Comma-separated subset: colmap,semantics,rgb,semantic",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--python", default=sys.executable)
    return parser


def main():
    parser = make_parser()
    args = parser.parse_args()
    selected = {name.strip() for name in args.stages.split(",") if name.strip()}
    allowed = {"colmap", "semantics", "rgb", "semantic"}
    if not selected or not selected <= allowed:
        parser.error(f"--stages must use only: {','.join(sorted(allowed))}")
    if args.view_stride < 1 or args.max_train_views == 0 or args.max_train_views == 1:
        parser.error("--view_stride must be >=1; --max_train_views must be -1 or >=2")

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
