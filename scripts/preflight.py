"""Read-only environment and dataset validation for the project."""

import argparse
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path


def package_status(module):
    return importlib.util.find_spec(module) is not None


def main():
    parser = argparse.ArgumentParser(description="Check 3DGS prerequisites without training")
    parser.add_argument("--scene", default="")
    parser.add_argument("--model", default="")
    parser.add_argument("--sam_checkpoint", default="")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    checks = []
    checks.append(("python>=3.8", sys.version_info >= (3, 8), sys.version.split()[0]))
    checks.append(("colmap", shutil.which("colmap") is not None, shutil.which("colmap") or "not found"))
    checks.append(("nvidia-smi", shutil.which("nvidia-smi") is not None, shutil.which("nvidia-smi") or "not found"))
    for module in (
        "torch", "torchvision", "numpy", "PIL", "cv2", "plyfile", "tqdm",
        "diff_gaussian_rasterization", "simple_knn", "open_clip", "segment_anything",
        "sklearn",
    ):
        checks.append((f"python:{module}", package_status(module), "importable" if package_status(module) else "missing"))

    if args.scene:
        scene = Path(args.scene).resolve()
        checks.extend([
            ("scene", scene.is_dir(), str(scene)),
            ("scene/images-or-input", (scene / "images").is_dir() or (scene / "input").is_dir(), str(scene)),
            ("scene/COLMAP", (scene / "sparse" / "0").is_dir(), str(scene / "sparse" / "0")),
        ])
    if args.sam_checkpoint:
        checkpoint = Path(args.sam_checkpoint).resolve()
        checks.append(("SAM checkpoint", checkpoint.is_file(), str(checkpoint)))
    if args.model:
        model = Path(args.model).resolve()
        checks.append(("model", model.is_dir(), str(model)))

    failures = 0
    for name, okay, detail in checks:
        print(f"[{'OK' if okay else 'MISSING'}] {name}: {detail}")
        failures += int(not okay)
    print(json.dumps({"checks": len(checks), "missing": failures}))
    if args.strict and failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
