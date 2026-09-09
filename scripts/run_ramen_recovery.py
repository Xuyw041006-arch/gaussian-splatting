"""Resume Ramen v3, then run isolated corrected v4 with checkpoint mirrors.

Only this script's v4 checkpoint mirrors are rotated. Existing v3 files and
datasets are never deleted. Logs/status remain on local disk if Drive is full.
DriveFS removal may move old mirrors into Trash, where they remain recoverable
and can still count toward account quota. Rotation does not bound total Drive
storage. This script never empties Trash or permanently deletes Drive objects.
"""

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


V3 = "ramen_curriculum_v3_15k"
V4 = "ramen_corrected_v4_15k"
SCENE_DIRS = ("images", "images_train", "sparse", "test_mask")
IMPORTANT = "egg,pork belly,wavy noodles in bowl"
NORMAL = "yellow bowl,chopsticks,glass of water"
CHECKPOINT = re.compile(r"chkpnt([0-9]+)\.pth\Z")
SEMANTIC_REFERENCE_FILES = (
    "semantic_meta.npz", "semantic_summary.json", "scene_inventory.json",
    "sparse/0/train.txt", "sparse/0/val.txt", "sparse/0/test.txt",
)
DRIVE_QUOTA_NOTE = (
    "DriveFS unlink may move old checkpoint mirrors into Trash. They can remain recoverable "
    "and count toward account quota; rotation does not guarantee reclaimed space or bounded "
    "total storage. Repeated backups can exceed the planning estimate. Trash is never purged."
)


class StorageExhaustedError(RuntimeError):
    """A stopped stage must not automatically consume more storage elsewhere."""


def log_reports_storage_exhaustion(path):
    with Path(path).open("rb") as handle:
        handle.seek(max(0, Path(path).stat().st_size - 128 * 1024))
        tail = handle.read().decode("utf-8", errors="replace").lower()
    return any(message in tail for message in (
        "no space left on device", "disk quota exceeded", "errno 28", "errno 122",
        "storagequotaexceeded", "storage quota exceeded",
    ))


def utc():
    return datetime.now(timezone.utc).isoformat()


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {} if default is None else default


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".recovery-write.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def file_stamp(path):
    info = Path(path).stat()
    return {"bytes": info.st_size, "mtime_ns": info.st_mtime_ns}


def complete_torch_archive(path):
    """Validate a modern torch.save ZIP without unpickling its Python payload."""
    try:
        before = file_stamp(path)
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
            has_pickle = any(name.endswith("/data.pkl") for name in names)
            has_version = any(name.endswith("/version") for name in names)
            complete = has_pickle and has_version and archive.testzip() is None
        return complete and file_stamp(path) == before
    except (OSError, ValueError, zipfile.BadZipFile, EOFError):
        return False


def atomic_copy(source, target, verify_checkpoint=False):
    """Copy a stable completed file; expose it only after a successful close."""
    source, target = Path(source), Path(target)
    if source.is_symlink() or not source.is_file():
        raise ValueError(f"Expected a regular source file: {source}")
    before = file_stamp(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".recovery-copy.tmp")
    digest = hashlib.sha256()
    try:
        with source.open("rb") as incoming, temporary.open("wb") as outgoing:
            for block in iter(lambda: incoming.read(8 * 1024 * 1024), b""):
                outgoing.write(block)
                digest.update(block)
        if file_stamp(source) != before or temporary.stat().st_size != before["bytes"]:
            raise RuntimeError(f"Source changed or copy was truncated: {source}")
        if verify_checkpoint and not complete_torch_archive(temporary):
            raise RuntimeError(f"Copied torch checkpoint failed ZIP/CRC validation: {source}")
        shutil.copystat(source, temporary)
        temporary.replace(target)
    finally:
        if temporary.is_file():
            temporary.unlink()  # Only the exact temporary file owned by this copy.
    return {**before, "sha256": digest.hexdigest()}


def require_space(path, minimum_bytes):
    path = Path(path)
    free = shutil.disk_usage(path).free
    if free < minimum_bytes:
        raise RuntimeError(
            f"Insufficient reported free space at {path}: {free / 2**30:.2f} GiB; "
            f"need {minimum_bytes / 2**30:.2f} GiB. No files were removed."
        )
    return free


def copy_scene(source, destination):
    """Copy only geometry/images/GT; deliberately omit old semantic supervision."""
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if source == destination or source in destination.parents or destination in source.parents:
        raise ValueError("Scene source and destination must be separate directories")
    for name in SCENE_DIRS:
        if not (source / name).is_dir():
            raise FileNotFoundError(f"Required scene folder is missing: {source / name}")
    destination.mkdir(parents=True, exist_ok=True)
    for name in SCENE_DIRS:
        for path in sorted((source / name).rglob("*")):
            if path.is_symlink():
                raise ValueError(f"Unexpected scene symlink: {path}")
            if path.is_file():
                target = destination / path.relative_to(source)
                if not target.is_file() or file_stamp(target) != file_stamp(path):
                    atomic_copy(path, target)
    atomic_json(destination / "recovery_scene_provenance.json", {
        "source": str(source), "copied_directories": list(SCENE_DIRS),
        "old_semantic_maps_copied": False, "utc": utc(),
    })


def latest_checkpoint(model):
    paths = []
    for path in Path(model).glob("chkpnt*.pth"):
        match = CHECKPOINT.fullmatch(path.name)
        if match and path.is_file() and path.stat().st_size > 0:
            paths.append((int(match.group(1)), path))
    for _, path in sorted(paths, reverse=True):
        if complete_torch_archive(path):
            return path
        print(f"Ignoring incomplete/corrupt torch checkpoint: {path}", flush=True)
    return None


def backup_model(source, destination, final=False, iteration=15000):
    """Rotate a manifest-owned visible mirror; Drive Trash may retain its quota."""
    source, destination = Path(source), Path(destination)
    if not source.is_dir():
        return
    destination.mkdir(parents=True, exist_ok=True)
    manifest_path = destination / "recovery_backup_manifest.json"
    manifest = read_json(manifest_path, {"files": {}})
    records = manifest.setdefault("files", {})
    latest = latest_checkpoint(source)
    managed_latest = manifest.get("latest_checkpoint")
    if (latest is not None and managed_latest and CHECKPOINT.fullmatch(managed_latest)
            and managed_latest in records and (destination / managed_latest).is_file()
            and int(CHECKPOINT.fullmatch(latest.name).group(1)) < int(CHECKPOINT.fullmatch(managed_latest).group(1))):
        # A corrupt/missing newer local file must not roll a good Drive mirror back.
        latest = None
    selected = [path for path in (latest, source / "best_val_chkpnt.pth") if path and path.is_file()]
    selected.extend(path for path in source.glob("*.json") if path.name != manifest_path.name)
    selected.extend(path for path in (source / "cfg_args",) if path.is_file())
    semantic = source / "semantic" / f"iteration_{iteration}"
    selected.extend(path for path in (semantic / "semantic_checkpoint.pt", semantic / "training_complete.json") if path.is_file())
    if final:
        final_artifacts = (
            source / "point_cloud" / f"iteration_{iteration}" / "point_cloud.ply",
            semantic / "semantic_features.pt",
        )
        for path in final_artifacts:
            if not path.is_file() or path.stat().st_size == 0:
                raise RuntimeError(f"Required final artifact is missing or empty: {path}")
        selected.extend(final_artifacts)
    for path in selected:
        serialized = path.suffix in (".pth", ".pt")
        if serialized and not complete_torch_archive(path):
            if final and path.name == "semantic_features.pt":
                raise RuntimeError(f"Final semantic artifact is corrupt or incomplete: {path}")
            print(f"Keeping previous backup; source torch archive is incomplete: {path}", flush=True)
            continue
        relative = str(path.relative_to(source))
        target = destination / relative
        previous = records.get(relative, {})
        if target.is_file() and target.stat().st_size == path.stat().st_size and all(
            previous.get(key) == value for key, value in file_stamp(path).items()
        ):
            continue
        require_space(destination, path.stat().st_size + 64 * 1024 * 1024)
        records[relative] = atomic_copy(path, target, verify_checkpoint=serialized)
        atomic_json(manifest_path, manifest)
    if latest is not None:
        old = manifest.get("latest_checkpoint")
        manifest["latest_checkpoint"] = latest.name
        atomic_json(manifest_path, manifest)
        if old and old != latest.name and CHECKPOINT.fullmatch(old) and old in records:
            old_path = destination / old
            if old_path.is_file() and not old_path.is_symlink():
                old_path.unlink()
                manifest.setdefault("rotation_history", []).append({
                    "name": old, "utc": utc(), "action": "unlink_old_managed_mirror",
                    "quota_reclaimed": "unverified", "recoverability": "DriveFS may retain it in Trash",
                })
                print(f"Rotated managed mirror {old}; Drive Trash may retain it and its quota.", flush=True)
            records.pop(old, None)
            atomic_json(manifest_path, manifest)


def backup_outputs(source, destination, final=False, iteration=15000):
    source, destination = Path(source), Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    if final:
        candidates = []
        for model in ("joint", "sequential"):
            candidates.extend((
                source / model / "point_cloud" / f"iteration_{iteration}" / "point_cloud.ply",
                source / model / "semantic" / f"iteration_{iteration}" / "semantic_features.pt",
            ))
        for name in ("eval_joint", "eval_sequential", "report"):
            candidates.extend(path for path in (source / name).rglob("*") if path.is_file())
        additional, largest_overwrite = 0, 0
        for path in candidates:
            if not path.is_file():
                continue
            target = destination / path.relative_to(source)
            if not target.exists():
                additional += path.stat().st_size
            else:
                additional += max(0, path.stat().st_size - target.stat().st_size)
                largest_overwrite = max(largest_overwrite, path.stat().st_size)
        require_space(destination, additional + largest_overwrite + 64 * 1024 * 1024)
    for model in ("joint", "sequential"):
        backup_model(source / model, destination / model, final, iteration)
    for path in source.glob("*.json"):
        atomic_copy(path, destination / path.name)
    if final:
        for name in ("eval_joint", "eval_sequential", "report"):
            for path in (source / name).rglob("*"):
                if path.is_file() and not path.name.endswith(".tmp"):
                    atomic_copy(path, destination / path.relative_to(source))


def restore_outputs(source, destination):
    """Restore this wrapper's prior snapshots after a Colab runtime reset."""
    source, destination = Path(source), Path(destination)
    if not source.is_dir():
        return
    for path in source.rglob("*"):
        if path.is_file() and not path.name.endswith(".tmp"):
            target = destination / path.relative_to(source)
            if not target.exists():
                atomic_copy(path, target, verify_checkpoint=path.suffix in (".pth", ".pt"))


def semantic_space_fingerprint(scene):
    """Hash NPZ contents, not timestamp-dependent ZIP bytes, plus exact splits."""
    import numpy as np

    scene = Path(scene)
    digest = hashlib.sha256()
    arrays = {}
    with np.load(scene / "semantic_meta.npz", allow_pickle=False) as metadata:
        required = {"pca_components", "pca_mean", "feature_min", "feature_max", "prototype_features",
                    "clip_model", "clip_pretrained", "fit_image_names", "heldout_image_names"}
        if not required.issubset(metadata.files):
            raise RuntimeError("Semantic metadata lacks the train-only fit protocol")
        for name in sorted(metadata.files):
            value = np.ascontiguousarray(metadata[name])
            descriptor = json.dumps({"name": name, "dtype": value.dtype.str, "shape": value.shape}, sort_keys=True).encode()
            value_hash = hashlib.sha256(descriptor + value.tobytes()).hexdigest()
            arrays[name] = value_hash
            digest.update(name.encode() + b"\0" + value_hash.encode() + b"\0")
    splits = {}
    for name in ("train", "val", "test"):
        path = scene / "sparse/0" / (name + ".txt")
        splits[name] = sorted(line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    digest.update(json.dumps(splits, sort_keys=True).encode())
    return {"fingerprint": digest.hexdigest(), "arrays": arrays, "splits": splits}


def has_saved_weights(output):
    output = Path(output)
    return any(path.is_file() for path in output.rglob("*.pth")) or any(
        path.is_file() for path in output.rglob("semantic_features.pt")
    )


def establish_semantic_reference(scene, reference, output):
    """Persist the immutable fit before training; reject incompatible resumed maps."""
    scene, reference, output = Path(scene), Path(reference), Path(output)
    current = semantic_space_fingerprint(scene)
    manifest_path = reference / "semantic_space_reference.json"
    if manifest_path.exists():
        saved = read_json(manifest_path)
        if not saved.get("fingerprint") or saved["fingerprint"] != current["fingerprint"]:
            raise RuntimeError(
                "Semantic embedding space changed after preprocessing: PCA/prototypes/encoding bounds "
                "or held-out splits differ from the saved weights. Resume is blocked; keep these "
                "weights and rebuild maps with their original fit, or start a separate new run."
            )
        # Detect a lost/corrupt reference file as well as a changed regenerated fit.
        if semantic_space_fingerprint(reference)["fingerprint"] != saved["fingerprint"]:
            raise RuntimeError("Persisted semantic-space reference is corrupted; resume is blocked")
    else:
        if has_saved_weights(output) or has_saved_weights(reference.parent / "outputs_full"):
            raise RuntimeError("Saved weights have no immutable semantic-space reference; cannot safely resume")
        reference.mkdir(parents=True, exist_ok=True)
        for name in SEMANTIC_REFERENCE_FILES:
            atomic_copy(scene / name, reference / name)
        atomic_json(manifest_path, {**current, "established_at": utc(), "immutable": True})
    for model in ("joint", "sequential"):
        model_reference = output / model / "semantic_space_reference.json"
        previous = read_json(model_reference)
        if previous and previous.get("fingerprint") != current["fingerprint"]:
            raise RuntimeError(f"Model snapshot has a different semantic space: {model_reference}")
        atomic_json(model_reference, {"fingerprint": current["fingerprint"], "reference": str(manifest_path)})
    return current


def metrics_reusable(metrics_path, model, masks, evaluator, iteration=15000):
    metrics_path, model, masks = Path(metrics_path), Path(model), Path(masks)
    value = read_json(metrics_path)
    protocol = value.get("protocol", {})
    if (value.get("model") != str(model.resolve())
            or value.get("iteration") != iteration or value.get("threshold") != 0.25
            or value.get("granularity") != 1 or value.get("boundary_ratio") != 0.008
            or protocol.get("metric_scope") != "annotated_test_mask_views"
            or value.get("evaluator_version") != hashlib.sha256(Path(evaluator).read_bytes()).hexdigest()):
        return False
    splits = {path.name for path in masks.iterdir() if path.is_dir() and list(path.glob("*.png"))}
    if {row.get("split") for row in protocol.get("views", [])} != splits:
        return False
    weights = [model / "point_cloud" / f"iteration_{iteration}" / "point_cloud.ply",
               model / "semantic" / f"iteration_{iteration}" / "semantic_features.pt"]
    if not all(path.is_file() and path.stat().st_mtime <= metrics_path.stat().st_mtime for path in weights):
        return False
    if not all(path.stat().st_mtime <= metrics_path.stat().st_mtime for path in masks.glob("*/*.png")):
        return False
    for view in protocol.get("views", []):
        image = masks.parent / "images" / view.get("camera", "")
        if image.is_file() and image.stat().st_mtime > metrics_path.stat().st_mtime:
            return False
    return True


def make_comparison(output):
    """Join already evaluated v3 artifacts without loading CLIP again."""
    from scripts.run_ramen_benchmark import timing_protocol

    output = Path(output)
    results = {name: read_json(output / f"eval_{name}" / "metrics.json") for name in ("joint", "sequential")}
    fingerprints = [row.get("dataset_fingerprint") for row in results.values()]
    if not all(fingerprints) or len(set(fingerprints)) != 1:
        raise RuntimeError("v3 evaluation dataset fingerprints differ; comparison is not valid")
    for key in ("evaluator_version", "threshold", "granularity", "boundary_ratio", "metric_scope"):
        if results["joint"].get(key) != results["sequential"].get(key):
            raise RuntimeError(f"v3 evaluation protocols differ at {key}")
    keys = ("gaussians", "test_psnr", "test_ssim", "test_important_psnr", "test_normal_psnr",
            "mean_iou", "mean_boundary_iou", "tier_gaussians")
    timings = read_json(output / "training_times.json")
    summary = {"dataset": "LERF-Mask ramen", "iterations": 15000,
               "important": IMPORTANT.split(","), "normal": NORMAL.split(","),
               "background": "all remaining pixels/regions", "semantic_iterations_baseline": 5000,
               "protocol": {**timing_protocol(timings, False), "timings_seconds": timings,
                            "requested_iteration_cap": 15000,
                            "recovery_note": "v3 baseline resumed; fixed-iteration comparison, not equal-time"}}
    for name, result in results.items():
        summary[name] = {key: result[key] for key in keys if key in result}
        summary[name]["validation"] = read_json(output / name / "validation_summary.json")
        for tier, labels in (("important", IMPORTANT), ("normal", NORMAL)):
            values = [result["per_label_iou"][label] for label in labels.split(",")]
            summary[name][tier + "_mean_iou"] = sum(values) / len(values)
    summary["delta"] = {
        key: value - summary["sequential"][key] for key, value in summary["joint"].items()
        if isinstance(value, (int, float)) and isinstance(summary["sequential"].get(key), (int, float))
    }
    atomic_json(output / "comparison.json", summary)


class Recovery:
    def __init__(self, args):
        self.args = args
        self.repo = Path(__file__).resolve().parents[1]
        self.work = Path(args.work_root).resolve()
        self.persist = Path(args.persist_root).resolve()
        self.v3 = Path(args.v3_root).resolve() if args.v3_root else self.persist / V3
        self.scene = Path(args.source_scene).resolve() if args.source_scene else self.persist / "ramen_detail_v2_15k/data/ramen"
        self.v4 = self.work / V4
        self.v4_drive = self.persist / V4
        self.status = read_json(self.work / "recovery_status.json", {"stages": {}})
        self.status.setdefault("stages", {})
        self.stop_backup = threading.Event()
        self.backup_thread = None

    def save_status(self):
        self.status["updated_at"] = utc()
        atomic_json(self.work / "recovery_status.json", self.status)
        try:
            atomic_json(self.persist / "recovery_status.json", self.status)
        except OSError as error:
            print(f"Drive status backup failed; local status retained: {error}", flush=True)

    def stage(self, name, action):
        print(f"RECOVERY STAGE {name}: started", flush=True)
        self.status["current_stage"] = name
        self.status["state"] = "running"
        self.status["stages"][name] = {"state": "started", "started_at": utc()}
        self.save_status()
        try:
            action()
        except BaseException as error:
            self.status["stages"][name].update(state="failed", error=repr(error), ended_at=utc())
            self.status["state"] = "failed"
            self.save_status()
            raise
        self.status["stages"][name].update(state="completed", ended_at=utc())
        self.save_status()
        print(f"RECOVERY STAGE {name}: completed", flush=True)

    def command(self, name, command):
        log = self.work / "logs" / f"{name}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        print("COMMAND", " ".join(map(str, command)), "LOG", log, flush=True)
        with log.open("a", encoding="utf-8") as handle:
            process = subprocess.Popen(list(map(str, command)), cwd=self.repo, stdout=handle, stderr=subprocess.STDOUT)
            try:
                returncode = process.wait()
            except BaseException:
                process.terminate()
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                raise
        if returncode:
            if log_reports_storage_exhaustion(log):
                raise StorageExhaustedError(
                    f"{name} stopped because storage is full. Existing results/checkpoints are retained; "
                    f"recovery will exit without advancing to another training stage. Inspect {log}."
                )
            raise RuntimeError(f"{name} exited {returncode}; inspect {log}")

    def benchmark(self, scene, output, *extra):
        return [sys.executable, "-u", "-m", "scripts.run_ramen_benchmark", "--scene", scene,
                "--output_root", output, "--sam_checkpoint", self.args.sam_checkpoint,
                "--iterations", 15000, "--semantic_iterations", 5000, "--semantic_start", 1500,
                "--semantic_ramp_iterations", 2500, "--validation_views", 12,
                "--validation_interval", 1000, "--early_stop_patience", 4, "--resume", *extra]

    def report(self, output, legacy=None):
        command = [sys.executable, "-m", "scripts.build_ramen_report", "--output_root", output,
                   "--report_dir", Path(output) / "report"]
        if legacy and Path(legacy).is_dir():
            command.extend(("--legacy_root", legacy))
        self.command("report_" + Path(output).parent.name, command)

    def prepare_v4_semantics(self):
        from scripts.run_ramen_benchmark import detail_preprocessing_complete, select_validation_views

        scene = self.v4 / "data/ramen"
        images = sorted(path for path in (scene / "images_train").glob("*.*") if path.is_file())
        validation = select_validation_views(images, 12)
        tests = sorted((scene / "images").glob("test_*.*"))
        if not validation or len(tests) < 3:
            raise RuntimeError("Cannot establish the expected Ramen train/validation/test split")
        val_file = scene / "sparse/0/val.txt"
        val_file.write_text("".join(path.name + "\n" for path in validation), encoding="utf-8")
        heldout_names = {path.name for path in validation}
        (scene / "sparse/0/train.txt").write_text(
            "".join(path.name + "\n" for path in images if path.name not in heldout_names), encoding="utf-8",
        )
        (scene / "sparse/0/test.txt").write_text("".join(path.name + "\n" for path in tests), encoding="utf-8")
        maps_complete = detail_preprocessing_complete(scene, [path.name for path in validation]) and all(
            (scene / "semantic_maps" / (path.stem + ".npz")).is_file() for path in images
        ) and (scene / "semantic_summary.json").is_file()
        if not maps_complete:
            self.command("v4_semantic_preprocessing", [
                sys.executable, "-u", "-m", "preprocess_semantics", "--scene", scene,
                "--images_subdir", "images_train", "--fit_exclude_list", val_file,
                "--sam_checkpoint", self.args.sam_checkpoint, "--sam_model", "vit_h",
                "--clip_model", "ViT-H-14", "--clip_pretrained", "laion2b_s32b_b79k",
                "--feature_dim", 32, "--feature_width", 512, "--max_masks", 192,
                "--points_per_side", 32, "--batch_size", 16, "--important", IMPORTANT,
                "--normal", NORMAL, "--cross_view_prototypes", 96, "--cross_view_weight", 0.72,
                "--boundary_width", 3, "--boundary_boost", 2.25, "--thin_boost", 1.50,
                "--thin_compactness", 0.40, "--thin_aspect_ratio", 2.5,
            ])
        reference = establish_semantic_reference(
            scene, self.v4_drive / "data_provenance", self.v4 / "outputs_full",
        )
        self.status["semantic_space_fingerprint"] = reference["fingerprint"]
        self.save_status()

    def run_v3_stages(self, output):
        if self.args.skip_v3_recovery:
            print("V3 recovery explicitly skipped; only isolated corrected V4 training will run.", flush=True)
            self.status["v3_recovery"] = "explicitly_skipped"
            self.save_status()
            return

        def evaluate_joint():
            if metrics_reusable(output / "eval_joint/metrics.json", output / "joint", self.scene / "test_mask", self.repo / "scripts/evaluate_lerf_mask.py"):
                print("Reusing current-protocol v3 joint evaluation", flush=True)
                return
            self.command("v3_joint_eval", self.benchmark(self.scene, output, "--skip_training", "--skip_preprocess", "--skip_baseline", "--no_equal_time"))

        self.stage("v3_joint_evaluation", evaluate_joint)
        self.stage("v3_baseline_resume", lambda: self.command("v3_baseline_resume", self.benchmark(
            self.scene, output, "--skip_joint", "--skip_preprocess", "--no_equal_time")))
        self.stage("v3_comparison_report", lambda: (make_comparison(output), self.report(output, self.v3 / "legacy_metrics_backup")))

    def backup_loop(self):
        while not self.stop_backup.is_set():
            try:
                backup_outputs(self.v4 / "outputs_full", self.v4_drive / "outputs_full")
                available = has_saved_weights(self.v4_drive / "outputs_full")
                self.record_backup_status({"state": "completed" if available else "waiting_for_checkpoint",
                                           "backed_up": available, "utc": utc()})
            except Exception as error:
                self.record_backup_status({"state": "failed", "backed_up": False, "utc": utc(), "error": repr(error),
                                           "training_continues_locally": True,
                                           "runtime_reset_risk": "Progress newer than the last successful Drive backup can be lost"})
                print(f"CHECKPOINT BACKUP FAILED — backed_up=false; training continues locally. "
                      f"A runtime reset can lose newer progress. Check {self.work / 'backup_status.json'}. {error}", flush=True)
            self.stop_backup.wait(self.args.backup_interval)

    def record_backup_status(self, payload):
        payload = {**payload, "quota_note": DRIVE_QUOTA_NOTE}
        atomic_json(self.work / "backup_status.json", payload)
        try:
            atomic_json(self.v4_drive / "backup_status.json", payload)
        except OSError as error:
            payload["drive_status_record_error"] = repr(error)
            atomic_json(self.work / "backup_status.json", payload)
        self.status["backup"] = payload

    def run(self):
        if not self.persist.is_dir() or not self.scene.is_dir():
            raise FileNotFoundError("Drive must be mounted and the source Ramen scene must exist")
        if self.work == self.persist or self.work in self.persist.parents or self.persist in self.work.parents:
            raise ValueError("work_root and persist_root must be separate directory trees")
        self.work.mkdir(parents=True, exist_ok=True)
        work_free = require_space(self.work, int(self.args.min_work_free_gb * 2**30))
        drive_free = require_space(self.persist, int(self.args.min_drive_free_gb * 2**30))
        self.status["storage_estimate"] = {
            "reported_work_free_gib": work_free / 2**30,
            "reported_drive_free_gib": drive_free / 2**30,
            "estimated_peak_additional_drive_gib": 8.0 if self.args.skip_v3_recovery else 13.0,
            "estimated_peak_additional_local_gib": 20.0,
            "strict_guarantee": False,
            "note": "Planning estimate for roughly one-million-Gaussian models, not a quota guarantee. "
                    "Drive FUSE may not report account quota accurately. A failed backup retains local "
                    "weights and previous persistent checkpoints; incomplete final persistence is failed.",
            "drive_trash_quota_risk": DRIVE_QUOTA_NOTE,
        }
        print("STORAGE PLANNING (non-guaranteed): " + json.dumps(self.status["storage_estimate"]), flush=True)
        self.status["backup_status_file"] = str(self.work / "backup_status.json")
        self.status["backup"] = {"state": "not_started", "backed_up": False, "quota_note": DRIVE_QUOTA_NOTE}
        if not Path(self.args.sam_checkpoint).is_file():
            raise FileNotFoundError(self.args.sam_checkpoint)
        self.status.update(state="running", work_root=str(self.work), persist_root=str(self.persist), started_at=utc())
        self.save_status()
        v3_output = self.v3 / "outputs_full"

        try:
            self.run_v3_stages(v3_output)
            self.stage("v4_scene_preparation", lambda: copy_scene(self.scene, self.v4 / "data/ramen"))
            self.stage("v4_checkpoint_restore", lambda: restore_outputs(self.v4_drive / "outputs_full", self.v4 / "outputs_full"))
            self.stage("v4_semantic_space_verification", self.prepare_v4_semantics)
            self.backup_thread = threading.Thread(target=self.backup_loop, name="ramen-checkpoint-backup", daemon=True)
            self.backup_thread.start()
            self.stage("v4_training_and_evaluation", lambda: self.command("v4_training", self.benchmark(
                self.v4 / "data/ramen", self.v4 / "outputs_full", "--skip_preprocess")))
            self.stop_backup.set()
            self.backup_thread.join()
            self.stage("v4_report", lambda: self.report(self.v4 / "outputs_full", v3_output))
            self.stage("v4_final_persistence", lambda: backup_outputs(self.v4 / "outputs_full", self.v4_drive / "outputs_full", final=True))
            # The immutable fit was persisted before the first training step.
            establish_semantic_reference(self.v4 / "data/ramen", self.v4_drive / "data_provenance", self.v4 / "outputs_full")
            self.status["state"] = "completed"
            self.status["completed_at"] = utc()
        finally:
            self.stop_backup.set()
            if self.backup_thread:
                self.backup_thread.join()
            for output, legacy in ((v3_output, self.v3 / "legacy_metrics_backup"), (self.v4 / "outputs_full", v3_output)):
                if output.is_dir():
                    try:
                        self.report(output, legacy)
                    except Exception as error:
                        print(f"Final partial report failed: {error}", flush=True)
            try:
                backup_outputs(self.v4 / "outputs_full", self.v4_drive / "outputs_full", final=self.status.get("state") == "completed")
                self.record_backup_status({"state": "completed", "backed_up": True, "utc": utc(),
                                           "final_artifacts_persisted": self.status.get("state") == "completed"})
            except Exception as error:
                self.status["final_backup_error"] = repr(error)
                self.record_backup_status({"state": "failed", "backed_up": False, "utc": utc(), "error": repr(error),
                                           "final_artifacts_persisted": False, "local_weights_retained": True})
                if self.status.get("state") == "completed":
                    self.status["state"] = "failed"
            self.save_status()
            print(json.dumps(self.status, ensure_ascii=False, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--persist_root", required=True)
    parser.add_argument("--work_root", required=True)
    parser.add_argument("--sam_checkpoint", required=True)
    parser.add_argument("--source_scene", default="")
    parser.add_argument("--v3_root", default="")
    parser.add_argument(
        "--skip_v3_recovery", action="store_true",
        help="Run only corrected V4; do not resume or re-evaluate V3 training",
    )
    parser.add_argument("--backup_interval", type=float, default=120.0)
    parser.add_argument("--min_work_free_gb", type=float, default=10.0)
    parser.add_argument("--min_drive_free_gb", type=float, default=2.0)
    args = parser.parse_args()
    if args.backup_interval < 5 or min(args.min_work_free_gb, args.min_drive_free_gb) < 0:
        parser.error("backup interval must be >=5 seconds and minimum space non-negative")
    # A second notebook click must not launch a competing paid GPU process.
    import fcntl

    recovery = Recovery(args)
    recovery.work.mkdir(parents=True, exist_ok=True)
    with (recovery.work / ".ramen_recovery.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error("A recovery process is already running for this work_root")
        try:
            recovery.run()
            if recovery.status.get("state") != "completed":
                raise RuntimeError("Recovery did not complete; inspect recovery_status.json")
        except BaseException as error:
            recovery.status.update(state="failed", error=repr(error), failed_at=utc())
            recovery.save_status()
            raise


if __name__ == "__main__":
    main()
