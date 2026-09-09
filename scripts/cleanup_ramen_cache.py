"""Plan or remove a fixed list of redundant Ramen files after verified backup.

Never enumerates arbitrary user data for deletion. Final joint artifacts, best
validation checkpoint, baseline checkpoints, datasets and metrics are retained.
The backup must be outside Drive; Colab-local backup is temporary, not archival.
"""

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


CACHES = (
    "ramen_detail_v2_15k/assets/sam_vit_h_4b8939.pth",
    "ramen_detail_v2_15k/assets/ramen.zip",
)
OLD_JOINT = (
    "ramen_curriculum_v3_15k/outputs_full/joint/chkpnt7000.pth",
    "ramen_curriculum_v3_15k/outputs_full/joint/point_cloud/iteration_7000/point_cloud.ply",
    "ramen_curriculum_v3_15k/outputs_full/joint/point_cloud/iteration_10000/point_cloud.ply",
    "ramen_curriculum_v3_15k/outputs_full/joint/semantic/iteration_7000/semantic_features.pt",
    "ramen_curriculum_v3_15k/outputs_full/joint/semantic/iteration_10000/semantic_features.pt",
)


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--backup", required=True)
    parser.add_argument("--superseded_joint", action="store_true")
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    backup = Path(args.backup).resolve()
    if root.name != "semantic_adaptive_3dgs" or not root.is_dir():
        parser.error("Expected the existing semantic_adaptive_3dgs experiment directory")
    if backup.is_relative_to(root) or root.is_relative_to(backup):
        parser.error("Backup must be a separate directory outside the experiment tree")
    output = root / "ramen_curriculum_v3_15k" / "outputs_full"
    joint = output / "joint"
    required = [joint / "point_cloud/iteration_15000/point_cloud.ply",
                joint / "semantic/iteration_15000/semantic_features.pt",
                joint / "best_val_chkpnt.pth",
                output / "sequential/chkpnt7000.pth"]
    for path in required:
        if not path.is_file() or path.stat().st_size < 1024:
            parser.error(f"Required retained artifact unavailable: {path}")
    if args.superseded_joint:
        import torch
        state, step = torch.load(required[2], map_location="cpu", weights_only=False)
        if int(step) <= 7000 or len(state) < 13 or state[12] is None:
            parser.error("Best checkpoint does not supersede the old joint checkpoint")
        if state[1].ndim != 2 or not torch.isfinite(state[1]).all():
            parser.error("Best checkpoint geometry is invalid")
        print(f"Validated retained best checkpoint: iteration={step}, gaussians={len(state[1])}", flush=True)
        del state
    names = CACHES + (OLD_JOINT if args.superseded_joint else ())
    paths = [root / name for name in names if (root / name).is_file()]
    for path in paths:
        if path.is_symlink() or not path.resolve().is_relative_to(root):
            parser.error(f"Refusing symlink or path escape: {path}")
    print(json.dumps({"apply": args.apply, "bytes": sum(p.stat().st_size for p in paths),
                      "paths": [str(p.relative_to(root)) for p in paths]}, indent=2), flush=True)
    if not args.apply:
        return
    manifest_path = output / "cleanup_manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {"items": []}
    backup.mkdir(parents=True, exist_ok=True)
    for path in paths:
        relative = path.relative_to(root)
        target = backup / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        original_hash = digest(path)
        if not target.is_file() or digest(target) != original_hash:
            shutil.copy2(path, target)
        if digest(target) != original_hash:
            raise RuntimeError(f"Backup verification failed: {path}")
        row = {"path": str(path), "backup": str(target), "bytes": path.stat().st_size,
               "sha256": original_hash, "action": "verified_backup_then_remove",
               "recoverable": "Colab temporary backup; expires on runtime reset",
               "reason": "redownloadable upstream asset" if str(relative) in CACHES else "superseded joint snapshot; best and final retained",
               "utc": datetime.now(timezone.utc).isoformat(), "status": "backed_up"}
        manifest["items"].append(row)
        manifest_path.write_text(json.dumps(manifest, indent=2))
        path.unlink()
        row["status"] = "removed"
        manifest_path.write_text(json.dumps(manifest, indent=2))
        print(f"REMOVED {relative} ({row['bytes']} bytes); SHA256 backup verified", flush=True)
    print("Retained: best/final joint, all baseline checkpoints, dataset, semantic maps, metrics.", flush=True)


if __name__ == "__main__":
    main()
