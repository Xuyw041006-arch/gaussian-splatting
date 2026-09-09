"""Copy a small pose-preserving Ramen subset for CUDA integration checks.

Not a sparse-view reconstruction benchmark: COLMAP geometry/poses come from the
full source dataset. Source files are never changed and output must be new.
"""

import argparse
import importlib.util
import json
import shutil
from pathlib import Path

from scripts.run_ramen_benchmark import select_validation_views


def colmap_loader():
    path = Path(__file__).resolve().parents[1] / "scene/colmap_loader.py"
    spec = importlib.util.spec_from_file_location("_smoke_colmap_io", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prepare(source, output, views=8):
    source, output = Path(source).resolve(), Path(output).resolve()
    if source == output or source in output.parents or output in source.parents:
        raise ValueError("Smoke subset must be separate from its source")
    if output.exists():
        raise ValueError("Smoke subset output already exists; choose a new directory")
    if views < 4:
        raise ValueError("Use at least four non-test candidate views")
    candidates = sorted(path for path in (source / "images_train").iterdir() if path.is_file())
    selected = select_validation_views(candidates, views)
    tests = sorted((source / "images").glob("test_*.*"))
    if len(selected) != views or len(tests) < 3:
        raise ValueError("Insufficient candidate views or official test images")
    names = {path.name for path in selected + tests}
    loader = colmap_loader()
    sparse = source / "sparse/0"
    if (sparse / "images.bin").is_file():
        poses = loader.read_extrinsics_binary(sparse / "images.bin")
        cameras = loader.read_intrinsics_binary(sparse / "cameras.bin")
    else:
        poses = loader.read_extrinsics_text(sparse / "images.txt")
        cameras = loader.read_intrinsics_text(sparse / "cameras.txt")
    chosen = [pose for pose in poses.values() if pose.name in names]
    if {pose.name for pose in chosen} != names or len(chosen) != len(names):
        raise ValueError("Subset names must have unique exact COLMAP camera poses")
    intrinsics = []
    for camera_id in sorted({pose.camera_id for pose in chosen}):
        camera = cameras[camera_id]
        if camera.model == "PINHOLE":
            params = camera.params
        elif camera.model == "SIMPLE_PINHOLE":
            f, cx, cy = camera.params
            params = (f, f, cx, cy)
        else:
            raise ValueError("Smoke subset requires already-undistorted pinhole images")
        intrinsics.append(f"{camera.id} PINHOLE {camera.width} {camera.height} " + " ".join(format(float(v), ".17g") for v in params))
    for name in ("images", "images_train", "sparse/0"):
        (output / name).mkdir(parents=True, exist_ok=True)
    for path in selected + tests:
        if path.name != str(path.relative_to(path.parent)) or path.is_symlink():
            raise ValueError("Unexpected image path")
        shutil.copy2(path, output / "images" / path.name)
    for path in selected:
        shutil.copy2(path, output / "images_train" / path.name)
    rows = []
    for pose in sorted(chosen, key=lambda pose: pose.name):
        rows.append(f"{pose.id} " + " ".join(format(float(v), ".17g") for v in (*pose.qvec, *pose.tvec)) + f" {pose.camera_id} {pose.name}\n\n")
    (output / "sparse/0/images.txt").write_text("".join(rows))
    (output / "sparse/0/cameras.txt").write_text("\n".join(intrinsics) + "\n")
    for name in ("points3D.ply", "points3D.bin", "points3D.txt"):
        if (sparse / name).is_file():
            shutil.copy2(sparse / name, output / "sparse/0" / name)
    shutil.copytree(source / "test_mask", output / "test_mask")
    summary = {"source": str(source), "non_test_candidate_views": sorted(path.name for path in selected),
               "test_views": sorted(path.name for path in tests),
               "scope": "CUDA integration only, not full benchmark or independent sparse-view reconstruction",
               "full_source_geometry_and_poses_used": True, "old_semantic_teachers_copied": False}
    (output / "smoke_subset_provenance.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--views", type=int, default=8)
    args = parser.parse_args()
    prepare(args.source, args.output, args.views)


if __name__ == "__main__":
    main()
