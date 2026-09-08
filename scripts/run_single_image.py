#!/usr/bin/env python3
"""Run LucidDreamer as an explicit single-image *generative* completion mode.

Unlike multi-view reconstruction, unseen geometry is hallucinated.  The
manifest records that distinction so downstream UIs cannot present the result
as metrically verified reconstruction.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path


def build_command(args, prompt_path):
    repo = Path(args.luciddreamer_repo).resolve()
    return [
        args.python, str(repo / "run.py"),
        "--image", str(Path(args.image).resolve()),
        "--text", str(prompt_path),
        "--campath_gen", args.camera_path,
        "--diff_steps", str(args.diffusion_steps),
        "--seed", str(args.seed),
        "--save_dir", str(Path(args.output).resolve()),
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--luciddreamer_repo", required=True)
    parser.add_argument("--prompt", default="a coherent photorealistic 3D scene")
    parser.add_argument(
        "--camera_path", choices=["lookdown", "lookaround", "rotate360"],
        default="lookaround",
    )
    parser.add_argument("--diffusion_steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()
    image = Path(args.image).resolve()
    repo = Path(args.luciddreamer_repo).resolve()
    output = Path(args.output).resolve()
    if not image.is_file():
        parser.error(f"input image does not exist: {image}")
    if not (repo / "run.py").is_file():
        parser.error(
            "LucidDreamer run.py was not found. Clone the official repository "
            "from https://github.com/luciddreamer-cvlab/LucidDreamer"
        )
    if args.diffusion_steps < 1:
        parser.error("--diffusion_steps must be positive")
    output.mkdir(parents=True, exist_ok=True)
    prompt_path = output / "single_image_prompt.txt"
    prompt_path.write_text(args.prompt.strip() + "\n", encoding="utf-8")
    command = build_command(args, prompt_path)
    manifest = {
        "version": 1,
        "mode": "single_image_generative_completion",
        "metric_reconstruction": False,
        "source_image": str(image),
        "backend": "LucidDreamer",
        "backend_repository": "https://github.com/luciddreamer-cvlab/LucidDreamer",
        "camera_path": args.camera_path,
        "diffusion_steps": args.diffusion_steps,
        "seed": args.seed,
        "uncertainty_notice": (
            "Geometry and appearance outside the source view are generated priors "
            "and must not be treated as measured ground truth."
        ),
        "status": "planned" if args.dry_run else "running",
    }
    manifest_path = output / "single_completion.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(" ".join(command), flush=True)
    if not args.dry_run:
        subprocess.run(command, cwd=repo, check=True)
        ply_files = sorted(output.rglob("*.ply"), key=lambda path: path.stat().st_mtime)
        manifest["status"] = "complete"
        manifest["output_ply"] = str(ply_files[-1]) if ply_files else None
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

