"""One-shot bounded V5 persistence; invoke periodically from the training host.

Restoration mapping is in snapshot_manifest.json: scene/* -> original scene,
output/* -> original output_root, and checkpoints/MODEL/latest.pth ->
output_root/MODEL/<original_name>. The third semantic_latest.pt slot maps to its
recorded source_relative, together with its immutable dependency RGB PLY.
No source files or existing remote directories
are deleted. Only manifest-owned slots may be replaced. DriveFS version/Trash
accounting can still consume extra quota despite two visible checkpoint slots.
"""

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
from datetime import datetime, timezone


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.run_ramen_recovery import complete_torch_archive, semantic_space_fingerprint


OWNER = ".v5_snapshot_owner.json"
MANIFEST = "snapshot_manifest.json"
RESERVE = 64 * 1024 * 1024
SMALL_LIMIT = 16 * 1024 * 1024
CHECKPOINT = re.compile(r"chkpnt([0-9]+)\.pth\Z")
SCENE_REFERENCE = (
    "semantic_meta.npz", "semantic_summary.json", "scene_inventory.json",
    "sparse/0/train.txt", "sparse/0/val.txt", "sparse/0/test.txt",
)


def utc():
    return datetime.now(timezone.utc).isoformat()


def stamp(path):
    value = Path(path).stat()
    return {"bytes": value.st_size, "mtime_ns": value.st_mtime_ns}


def sha256(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"Not a regular snapshot file: {path}")
    before = stamp(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    if stamp(path) != before:
        raise RuntimeError(f"File changed while hashing: {path}")
    return digest.hexdigest()


def read_json(path):
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def safe_path(root, relative):
    relative = Path(relative)
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise RuntimeError(f"Unsafe managed relative path: {relative}")
    result = root / relative
    for path in (result, *result.parents):
        if path == root.parent:
            break
        if path.is_symlink():
            raise RuntimeError(f"Snapshot paths cannot traverse symlinks: {path}")
    return result


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".snapshot-json.tmp")
    if path.is_symlink() or temporary.is_symlink():
        raise RuntimeError("Snapshot manifest paths cannot be symlinks")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def require_space(destination, new_bytes):
    free = shutil.disk_usage(destination).free
    required = int(new_bytes) + RESERVE
    if free < required:
        raise RuntimeError(f"Insufficient reported free space: have {free} bytes, need {required}; previous backup retained, nothing removed")


def owner_for(scene, output):
    return {"version": 1, "tool": "snapshot_v5_progress", "scene": str(scene), "output_root": str(output)}


def open_destination(scene, output, destination):
    for source in (scene, output):
        if source == destination or source in destination.parents or destination in source.parents:
            raise ValueError("Snapshot destination and sources must be separate non-overlapping directories")
    if destination.is_symlink():
        raise RuntimeError("Snapshot destination cannot be a symlink")
    expected = owner_for(scene, output)
    for name in (OWNER, MANIFEST, ".snapshot.lock"):
        if (destination / name).is_symlink():
            raise RuntimeError(f"Snapshot control file cannot be a symlink: {name}")
    if destination.exists():
        if not destination.is_dir():
            raise RuntimeError("Snapshot destination is not a directory")
        if not (destination / OWNER).is_file():
            if any(destination.iterdir()):
                raise RuntimeError("Refusing an existing nonempty destination without this tool's V5 ownership marker")
        elif read_json(destination / OWNER) != expected:
            raise RuntimeError("Snapshot destination belongs to a different run")
    destination.mkdir(parents=True, exist_ok=True)
    if not (destination / OWNER).exists():
        atomic_json(destination / OWNER, expected)
    manifest_path = destination / MANIFEST
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        if manifest.get("owner") != expected:
            raise RuntimeError("Snapshot manifest belongs to a different run")
    else:
        # A missing manifest cannot authorize adoption of existing checkpoint slots.
        unexpected = [p.name for p in destination.iterdir() if p.name not in (OWNER, ".snapshot.lock")]
        if unexpected:
            raise RuntimeError("Missing snapshot manifest for an existing managed directory; refusing to adopt files")
        manifest = {"version": 1, "owner": expected, "files": {}, "pending": None,
                    "teacher": {"state": "waiting_for_complete_teacher"}}
        atomic_json(manifest_path, manifest)
    return manifest


def save_manifest(destination, manifest):
    manifest["updated_at"] = utc()
    atomic_json(destination / MANIFEST, manifest)


def current_matches(path, record):
    return bool(record and path.is_file() and not path.is_symlink()
                and path.stat().st_size == record["bytes"] and sha256(path) == record["sha256"])


def recover_pending(destination, manifest):
    """Resolve the manifest/atomic-rename boundary without guessing file identity."""
    pending = manifest.get("pending")
    if not pending:
        return
    target = safe_path(destination, pending["target"])
    temporary = safe_path(destination, pending["temporary"])
    old, new = pending.get("old"), pending["new"]
    if current_matches(target, new):
        manifest["files"][pending["target"]] = new
    elif (current_matches(target, old) if old else not target.exists()):
        if current_matches(temporary, new):
            os.replace(temporary, target)
            manifest["files"][pending["target"]] = new
        else:
            # Only this exact journal-owned incomplete copy may be discarded.
            if temporary.exists():
                if temporary.is_symlink() or not temporary.is_file():
                    raise RuntimeError("Invalid journal-owned temporary file")
                temporary.unlink()
            manifest["last_recovery"] = "Discarded incomplete managed temporary; previous slot retained"
    else:
        raise RuntimeError("Pending snapshot matches neither recorded old nor new SHA; preserve files for manual recovery")
    manifest["pending"] = None
    save_manifest(destination, manifest)


def copy_managed(source, relative, destination, manifest, immutable=False, checkpoint=False, extra=None, expected=None):
    """Journal -> verified temp copy -> atomic replace -> committed manifest."""
    source = Path(source)
    before = stamp(source)
    source_hash = sha256(source)
    record = {**before, "sha256": source_hash, "source": str(source), **(extra or {})}
    if expected and any(record[key] != expected[key] for key in ("bytes", "sha256")):
        raise RuntimeError(f"Source changed after snapshot planning: {source}")
    target = safe_path(destination, relative)
    old = manifest["files"].get(relative)
    if old and old["sha256"] == source_hash:
        if not current_matches(target, old):
            raise RuntimeError(f"Managed backup missing, truncated, or SHA mismatch: {target}")
        return
    if immutable and old:
        raise RuntimeError(f"Frozen teacher changed: {source}; old backup retained")
    if target.exists() and not current_matches(target, old):
        raise RuntimeError(f"Refusing to overwrite an unowned/modified snapshot target: {target}")
    require_space(destination, before["bytes"])
    target.parent.mkdir(parents=True, exist_ok=True)
    temp_relative = relative + ".snapshot-copy.tmp"
    temporary = safe_path(destination, temp_relative)
    if temporary.exists():
        raise RuntimeError(f"Unjournaled temporary exists; refusing to overwrite it: {temporary}")
    manifest["pending"] = {"target": relative, "temporary": temp_relative, "old": old, "new": record}
    save_manifest(destination, manifest)
    digest = hashlib.sha256()
    with source.open("rb") as incoming, temporary.open("xb") as outgoing:
        for block in iter(lambda: incoming.read(8 * 1024 * 1024), b""):
            outgoing.write(block)
            digest.update(block)
        outgoing.flush()
        os.fsync(outgoing.fileno())
    if stamp(source) != before or digest.hexdigest() != source_hash or sha256(temporary) != source_hash:
        raise RuntimeError(f"Source changed or snapshot read-back SHA failed: {source}; previous slot retained")
    if checkpoint and not complete_torch_archive(temporary):
        raise RuntimeError("Snapshot checkpoint ZIP/CRC check failed; previous slot retained")
    os.replace(temporary, target)
    manifest["files"][relative] = record
    manifest["pending"] = None
    save_manifest(destination, manifest)


def teacher_plan(scene, output):
    frozen_path = output / "v5_teacher_files.json"
    reference = output / "teacher_reference/semantic_space_reference.json"
    if not frozen_path.is_file() or not reference.is_file():
        return None
    try:
        frozen = read_json(frozen_path)
        protocol = read_json(output / "experiment_protocol.json")
        reference_info = read_json(reference)
    except (OSError, ValueError):
        return None  # Incomplete initial preparation is not a frozen teacher.
    if protocol.get("semantic_protocol") != "v5":
        raise RuntimeError("Snapshot source is not a recorded V5 run")
    names = frozen.get("image_names", [])
    entries = frozen.get("files", {})
    expected = {p for name in names for p in (
        f"semantic_maps/{Path(name).stem}.npz", f"importance_masks/{Path(name).stem}.png")}
    if not names or set(entries) != expected:
        raise RuntimeError("Incomplete or unexpected frozen teacher file inventory")
    from scripts.run_ramen_benchmark import detail_preprocessing_complete
    if not detail_preprocessing_complete(scene, teacher_version=2, image_names=names):
        return None
    if semantic_space_fingerprint(scene)["fingerprint"] != reference_info.get("fingerprint"):
        raise RuntimeError("Teacher metadata/splits differ from the frozen semantic-space reference")
    plan = {}
    raw_regions = [f"semantic_raw/{Path(name).stem}.npz" for name in names]
    for relative in (*SCENE_REFERENCE, *sorted(entries), *raw_regions):
        source = safe_path(scene, relative)
        if not source.is_file():
            return None
        info = {"bytes": source.stat().st_size, "sha256": sha256(source)}
        if relative in entries and any(info[key] != entries[relative][key] for key in ("bytes", "sha256")):
            raise RuntimeError(f"Source teacher does not match frozen training SHA: {relative}")
        plan[f"scene/{relative}"] = info
    return plan


def small_metadata(scene, output):
    """Bounded config/diagnostics, never recursive model tensors or raw images."""
    selected = {}
    for relative in ("sparse/0/train.txt", "sparse/0/val.txt", "sparse/0/test.txt"):
        path = scene / relative
        if path.is_file():
            selected[f"progress_scene/{relative}"] = path
    for path in output.rglob("*"):
        if not path.is_file() or path.is_symlink() or path.stat().st_size > SMALL_LIMIT:
            continue
        relative = path.relative_to(output)
        if path.suffix.lower() in (".json", ".jsonl", ".txt", ".md") or path.name in ("cfg_args", "cameras.json"):
            if path.suffix.lower() in (".json", ".jsonl"):
                try:
                    if path.suffix.lower() == ".jsonl":
                        with path.open(encoding="utf-8") as handle:
                            for line in handle:
                                if line.strip():
                                    json.loads(line)
                    else:
                        read_json(path)
                except (OSError, ValueError):
                    continue  # A writer may currently be producing this diagnostic.
            selected[f"output/{relative.as_posix()}"] = path
    return selected


def snapshot(scene, output, destination):
    scene, output = Path(scene).resolve(), Path(output).resolve()
    destination_input = Path(destination).absolute()
    if destination_input.is_symlink():
        raise RuntimeError("Snapshot destination cannot be a symlink")
    destination = destination_input.resolve()
    manifest = open_destination(scene, output, destination)
    with (destination / ".snapshot.lock").open("a") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return {"state": "busy", "destination": str(destination)}
        # Another process may have committed state between open_destination and lock.
        manifest = read_json(destination / MANIFEST)
        recover_pending(destination, manifest)
        plan = teacher_plan(scene, output)
        if plan is None:
            if manifest["teacher"]["state"] == "complete":
                raise RuntimeError("Previously complete source teacher is now missing/incomplete; old backup retained")
        else:
            fingerprint = hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()
            previous = manifest["teacher"].get("fingerprint")
            if previous and previous != fingerprint:
                raise RuntimeError("Frozen teacher fingerprint changed; refusing to mix training spaces")
            additional = sum(value["bytes"] for relative, value in plan.items() if relative not in manifest["files"])
            if additional:
                require_space(destination, additional)
            for relative in plan:
                copy_managed(scene / relative.removeprefix("scene/"), relative, destination, manifest,
                             immutable=True, expected=plan[relative])
            # A crash before this commit leaves teacher.state != complete.
            manifest["teacher"] = {"state": "complete", "fingerprint": fingerprint,
                                   "files": len(plan), "note": "Training teachers, raw CLIP/packed masks and splits; RGB dataset images are not included"}
            save_manifest(destination, manifest)
        for relative, source in small_metadata(scene, output).items():
            copy_managed(source, relative, destination, manifest)
        if manifest["teacher"]["state"] == "complete":
            for model in ("joint", "sequential"):
                candidates = sorted((int(match.group(1)), path) for path in (output / model).glob("chkpnt*.pth")
                                    if (match := CHECKPOINT.fullmatch(path.name)) and path.is_file() and not path.is_symlink())
                slot = f"checkpoints/{model}/latest.pth"
                old_step = manifest["files"].get(slot, {}).get("iteration", -1)
                for step, path in reversed(candidates):
                    if step < old_step:
                        break
                    if not complete_torch_archive(path):
                        continue
                    copy_managed(path, slot, destination, manifest, checkpoint=True,
                                 extra={"model": model, "original_name": path.name, "iteration": step,
                                        "source_relative": path.relative_to(output).as_posix()})
                    break
            semantic_candidates = sorted(
                (int(match.group(1)), path)
                for path in (output / "sequential/semantic").glob("iteration_*/semantic_checkpoint.pt")
                if (match := re.fullmatch(r"iteration_([0-9]+)", path.parent.name))
                and path.is_file() and not path.is_symlink()
            )
            semantic_slot = "checkpoints/sequential/semantic_latest.pt"
            old_rgb_iteration = manifest["files"].get(semantic_slot, {}).get("rgb_export_iteration", -1)
            for rgb_iteration, path in reversed(semantic_candidates):
                if rgb_iteration < old_rgb_iteration:
                    break
                if not complete_torch_archive(path):
                    continue
                geometry_relative = f"sequential/point_cloud/iteration_{rgb_iteration}/point_cloud.ply"
                geometry = safe_path(output, geometry_relative)
                if not geometry.is_file() or geometry.stat().st_size == 0:
                    raise RuntimeError("Semantic checkpoint requires its exact RGB export PLY; semantic backup was not advanced")
                dependency_slot = "dependencies/sequential/rgb_export.ply"
                existing_dependency = manifest["files"].get(dependency_slot)
                if existing_dependency and existing_dependency.get("rgb_export_iteration") != rgb_iteration:
                    raise RuntimeError("Frozen semantic RGB dependency belongs to a different iteration; use a new dedicated destination")
                semantic_unchanged = manifest["files"].get(semantic_slot, {}).get("sha256") == sha256(path)
                require_space(destination, (0 if existing_dependency else geometry.stat().st_size)
                              + (0 if semantic_unchanged else path.stat().st_size))
                copy_managed(geometry, dependency_slot, destination, manifest, immutable=True, extra={
                    "model": "sequential", "source_relative": geometry_relative,
                    "rgb_export_iteration": rgb_iteration, "stage": "immutable_semantic_geometry_dependency",
                })
                copy_managed(path, semantic_slot, destination, manifest, checkpoint=True, extra={
                    "model": "sequential", "stage": "posthoc_semantics", "original_name": path.name,
                    "source_relative": path.relative_to(output).as_posix(), "rgb_export_iteration": rgb_iteration,
                    "required_rgb_export": geometry_relative,
                    "rgb_export_in_periodic_snapshot": True,
                    "rgb_dependency_slot": dependency_slot,
                    "rgb_dependency_sha256": manifest["files"][dependency_slot]["sha256"],
                    "semantic_iteration": None,
                    "note": "Optimizer step is inside the checkpoint. Restore the frozen dependency PLY, not the potentially different latest RGB checkpoint geometry.",
                })
                break
        manifest["last_success"] = utc()
        manifest["storage_note"] = "Two RGB/joint slots plus one posthoc-semantic slot; Google Drive version/Trash quota and actual remote durability are not independently certified"
        save_manifest(destination, manifest)
        return {"state": "snapshot_complete" if manifest["teacher"]["state"] == "complete" else "waiting_for_complete_teacher",
                "destination": str(destination), "teacher": manifest["teacher"],
                "scope": "Training teachers, checkpoints, one frozen sequential semantic RGB dependency PLY, and small metadata; not a full final-artifact or dataset archive",
                "checkpoints": {key: value for key, value in manifest["files"].items() if key.startswith("checkpoints/")}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--destination", required=True)
    args = parser.parse_args()
    try:
        result = snapshot(args.scene, args.output_root, args.destination)
    except (OSError, RuntimeError, ValueError, KeyError) as error:
        print(json.dumps({"state": "failed", "error": str(error), "backup_safety": "committed or journal-recorded verified slot retained; inspect pending journal if present"}, ensure_ascii=False), file=sys.stderr)
        raise SystemExit(1)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
