#!/usr/bin/env python3
"""Launch the pipeline from a Gaussian Atlas App configuration JSON."""

import argparse
import json
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--sam_checkpoint", default="")
    parser.add_argument("--ram_checkpoint", default="")
    parser.add_argument("--single_image", default="")
    parser.add_argument("--single_backend_repo", default="./third_party/LucidDreamer")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    preset = str(payload.get("preset", "balanced"))
    capture = str(payload.get("capture_mode", "auto"))
    semantic_mode = str(payload.get("semantic_mode", "joint"))
    if preset not in {"quick", "balanced", "quality"}:
        parser.error(f"invalid preset in config: {preset}")
    if capture not in {"auto", "dense", "sparse", "single"}:
        parser.error(f"invalid capture_mode in config: {capture}")
    if semantic_mode not in {"joint", "sequential", "off"}:
        parser.error(f"invalid semantic_mode in config: {semantic_mode}")
    if semantic_mode != "off" and not args.sam_checkpoint:
        parser.error("--sam_checkpoint is required by this semantic configuration")
    if capture == "single" and not args.single_image:
        parser.error("--single_image is required by this single-image configuration")
    repo = Path(__file__).resolve().parents[1]
    command = [
        args.python, str(repo / "scripts" / "run_pipeline.py"),
        "--scene", str(Path(args.scene).resolve()),
        "--model", str(Path(args.model).resolve()),
        "--preset", preset, "--capture_mode", capture,
        "--training_mode", semantic_mode,
        "--importance_config", str(config_path),
    ]
    if args.sam_checkpoint:
        command.extend(["--sam_checkpoint", str(Path(args.sam_checkpoint).resolve())])
    if args.ram_checkpoint:
        command.extend(["--ram_checkpoint", str(Path(args.ram_checkpoint).resolve())])
    if args.single_image:
        command.extend([
            "--single_image", str(Path(args.single_image).resolve()),
            "--single_backend_repo", str(Path(args.single_backend_repo).resolve()),
        ])
    if args.resume:
        command.append("--resume")
    if args.dry_run:
        command.append("--dry_run")
    print(" ".join(command), flush=True)
    subprocess.run(command, cwd=repo, check=True)


if __name__ == "__main__":
    main()

