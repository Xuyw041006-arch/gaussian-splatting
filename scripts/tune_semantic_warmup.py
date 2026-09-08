#!/usr/bin/env python3
"""Pilot several RGB warm-up ratios and optionally launch the selected run."""

import argparse
import json
import subprocess
import sys
from pathlib import Path


def read_score(model):
    path = model / "validation_summary.json"
    if not path.is_file():
        raise RuntimeError(f"missing validation summary: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    candidates = [item for item in payload.get("history", []) if "semantic_l1" in item]
    if not candidates:
        raise RuntimeError(f"validation summary has no semantic measurements: {path}")
    item = max(candidates, key=lambda row: float(row.get("psnr", 0.0)))
    # PSNR remains the primary objective; semantic and boundary reconstruction
    # errors break ties without pretending to be ground-truth mIoU.
    score = (
        float(item["psnr"])
        - 5.0 * float(item.get("semantic_l1", 0.0))
        - 2.0 * float(item.get("boundary_l1", 0.0))
    )
    return score, item


def pipeline_command(args, model, iterations, warmup):
    repo = Path(__file__).resolve().parents[1]
    ramp = max(1, round(iterations * args.ramp_ratio))
    return [
        args.python, str(repo / "scripts" / "run_pipeline.py"),
        "--scene", str(Path(args.scene).resolve()),
        "--model", str(model),
        "--sam_checkpoint", str(Path(args.sam_checkpoint).resolve()),
        "--preset", args.preset,
        "--training_mode", "joint",
        "--stages", "joint",
        "--scene_iterations", str(iterations),
        "--semantic_start", str(warmup),
        "--semantic_ramp_iterations", str(ramp),
        "--validation_file", str(Path(args.validation_file).resolve()),
        "--validation_start", str(max(warmup + ramp, args.validation_interval)),
        "--validation_interval", str(args.validation_interval),
        "--resume",
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--sam_checkpoint", required=True)
    parser.add_argument("--validation_file", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--preset", choices=["quick", "balanced", "quality"], default="balanced")
    parser.add_argument("--pilot_iterations", type=int, default=3500)
    parser.add_argument("--final_iterations", type=int, default=15000)
    parser.add_argument("--warmup_ratios", nargs="+", type=float, default=(0.06, 0.10, 0.16))
    parser.add_argument("--ramp_ratio", type=float, default=0.16)
    parser.add_argument("--validation_interval", type=int, default=500)
    parser.add_argument("--run_final", action="store_true")
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args()
    if args.pilot_iterations < 100 or args.final_iterations < args.pilot_iterations:
        parser.error("final iterations must be at least the pilot iterations")
    if any(not 0 < ratio < 0.5 for ratio in args.warmup_ratios):
        parser.error("warmup ratios must lie between 0 and 0.5")
    output = Path(args.output_root).resolve()
    output.mkdir(parents=True, exist_ok=True)
    results = []
    for ratio in args.warmup_ratios:
        warmup = round(args.pilot_iterations * ratio)
        model = output / f"pilot_warmup_{warmup}"
        subprocess.run(
            pipeline_command(args, model, args.pilot_iterations, warmup),
            cwd=Path(__file__).resolve().parents[1], check=True,
        )
        score, validation = read_score(model)
        results.append({
            "ratio": ratio, "warmup": warmup, "score": score,
            "validation": validation, "model": str(model),
        })
    selected = max(results, key=lambda item: item["score"])
    final_warmup = round(args.final_iterations * selected["ratio"])
    summary = {
        "objective": "validation_psnr - 5*semantic_l1 - 2*boundary_l1",
        "pilots": results,
        "selected_ratio": selected["ratio"],
        "selected_final_warmup": final_warmup,
    }
    (output / "warmup_selection.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)
    if args.run_final:
        subprocess.run(
            pipeline_command(
                args, output / "selected_final", args.final_iterations, final_warmup
            ), cwd=Path(__file__).resolve().parents[1], check=True,
        )


if __name__ == "__main__":
    main()
