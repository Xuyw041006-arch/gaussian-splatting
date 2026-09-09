import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock
import zipfile

import numpy as np

from scripts import snapshot_v5_progress as snapshot


def checkpoint(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("archive/data.pkl", str(value))
        archive.writestr("archive/version", "3")
        archive.writestr("archive/data/0", bytes([value]) * 128)


class V5SnapshotTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        root = Path(self.directory.name)
        self.scene, self.output, self.destination = root / "scene", root / "run", root / "backup"
        self.scene.mkdir()
        self.output.mkdir()
        (self.output / "experiment_protocol.json").write_text(json.dumps({"semantic_protocol": "v5"}))

    def complete_teacher(self):
        names = ["a.jpg", "b.jpg", "val.jpg"]
        (self.scene / "semantic_maps").mkdir()
        (self.scene / "semantic_raw").mkdir()
        (self.scene / "importance_masks").mkdir()
        (self.scene / "sparse/0").mkdir(parents=True)
        for split, members in (("train", names[:2]), ("val", names[2:]), ("test", ["test_0.jpg"])):
            (self.scene / f"sparse/0/{split}.txt").write_text("\n".join(members))
        for name in ("semantic_summary.json", "scene_inventory.json"):
            (self.scene / name).write_text("{}")
        np.savez(self.scene / "semantic_meta.npz", teacher_preprocessing_version=2,
                 hierarchy_method="containment", prototype_mode="off", importance_policy="competitive_v1",
                 pca_components=np.zeros((3, 8)), pca_mean=np.zeros(8), feature_min=np.zeros(3),
                 feature_max=np.ones(3), prototype_features=np.zeros((1, 3)), clip_model="test",
                 clip_pretrained="test", fit_image_names=names[:2], heldout_image_names=names[2:])
        payload = {name: np.zeros((2, 3)) for name in (
            "detail_weight", "boundary", "thinness", "prototype_ids", "region_ids")}
        payload.update(hierarchy_prototype_ids=np.zeros((3, 2, 3)), hierarchy_region_ids=np.zeros((3, 2, 3)),
                       importance=np.ones((2, 3)), importance_known=np.ones((2, 3)))
        entries = {}
        for name in names:
            stem = Path(name).stem
            np.savez(self.scene / f"semantic_maps/{stem}.npz", **payload)
            np.savez(self.scene / f"semantic_raw/{stem}.npz", features=np.zeros((1, 8)),
                     confidences=np.ones(1), packed_region_masks=np.array([[252]], dtype=np.uint8),
                     mask_shape=np.array([2, 3]), hierarchy_region_maps=np.zeros((3, 2, 3)),
                     image_name=np.array(name))
            (self.scene / f"importance_masks/{stem}.png").write_bytes(b"fixture-png")
            for relative in (f"semantic_maps/{stem}.npz", f"importance_masks/{stem}.png"):
                path = self.scene / relative
                entries[relative] = {"bytes": path.stat().st_size, "sha256": snapshot.sha256(path)}
        (self.output / "v5_teacher_files.json").write_text(json.dumps({"version": 1, "image_names": names, "files": entries}))
        reference = self.output / "teacher_reference/semantic_space_reference.json"
        reference.parent.mkdir()
        reference.write_text(json.dumps(snapshot.semantic_space_fingerprint(self.scene)))

    def run_snapshot(self):
        return snapshot.snapshot(self.scene, self.output, self.destination)

    def test_unowned_directory_and_different_run_are_rejected_without_deletion(self):
        self.destination.mkdir()
        unrelated = self.destination / "user.txt"
        unrelated.write_text("keep")
        with self.assertRaisesRegex(RuntimeError, "nonempty"):
            self.run_snapshot()
        self.assertEqual(unrelated.read_text(), "keep")
        safe_destination = self.destination.parent / "new-backup"
        snapshot.snapshot(self.scene, self.output, safe_destination)
        other_run = self.output.parent / "different-run"
        other_run.mkdir()
        with self.assertRaisesRegex(RuntimeError, "different run"):
            snapshot.snapshot(self.scene, other_run, safe_destination)

    def test_partial_teacher_waits_but_preserves_small_diagnostics(self):
        (self.output / "diagnostic.json").write_text('{"state":"running"}')
        (self.output / "semantic_training_diagnostics.jsonl").write_text('{"iteration":50}\n{"iteration":100}\n')
        checkpoint(self.output / "joint/chkpnt1000.pth", 1)
        result = self.run_snapshot()
        self.assertEqual(result["state"], "waiting_for_complete_teacher")
        self.assertFalse((self.destination / "checkpoints/joint/latest.pth").exists())
        self.assertTrue((self.destination / "output/diagnostic.json").is_file())
        self.assertEqual((self.destination / "output/semantic_training_diagnostics.jsonl").read_text(),
                         (self.output / "semantic_training_diagnostics.jsonl").read_text())

    def test_raw_teacher_descriptors_and_packed_masks_are_frozen(self):
        self.complete_teacher()
        result = self.run_snapshot()
        self.assertEqual(result["state"], "snapshot_complete")
        for name in ("a", "b", "val"):
            source = self.scene / f"semantic_raw/{name}.npz"
            target = self.destination / f"scene/semantic_raw/{name}.npz"
            self.assertEqual(snapshot.sha256(source), snapshot.sha256(target))
        source = self.scene / "semantic_raw/a.npz"
        target = self.destination / "scene/semantic_raw/a.npz"
        before = snapshot.sha256(target)
        source.write_bytes(b"different raw descriptors")
        with self.assertRaisesRegex(RuntimeError, "fingerprint changed"):
            self.run_snapshot()
        self.assertEqual(snapshot.sha256(target), before)

    def test_same_size_destination_tampering_cannot_be_skipped(self):
        self.complete_teacher()
        self.run_snapshot()
        target = self.destination / "scene/semantic_raw/a.npz"
        target.write_bytes(b"x" * target.stat().st_size)
        with self.assertRaisesRegex(RuntimeError, "SHA mismatch"):
            self.run_snapshot()

    def test_teacher_is_immutable_and_slots_retain_only_latest_with_mapping(self):
        self.complete_teacher()
        first = self.output / "joint/chkpnt1000.pth"
        second = self.output / "joint/chkpnt2000.pth"
        checkpoint(first, 1)
        self.run_snapshot()
        teacher = self.destination / "scene/semantic_meta.npz"
        teacher_stamp = teacher.stat().st_mtime_ns
        checkpoint(second, 2)
        result = self.run_snapshot()
        slot = self.destination / "checkpoints/joint/latest.pth"
        self.assertEqual(snapshot.sha256(slot), snapshot.sha256(second))
        self.assertEqual(result["checkpoints"]["checkpoints/joint/latest.pth"]["original_name"], "chkpnt2000.pth")
        self.assertEqual(teacher.stat().st_mtime_ns, teacher_stamp)
        self.assertTrue(first.is_file() and second.is_file())
        self.assertEqual([p.name for p in slot.parent.iterdir()], ["latest.pth"])
        old_hash = snapshot.sha256(teacher)
        (self.scene / "semantic_summary.json").write_text('{"changed":true}')
        with self.assertRaisesRegex(RuntimeError, "fingerprint changed"):
            self.run_snapshot()
        self.assertEqual(snapshot.sha256(teacher), old_hash)

    def test_space_failure_preserves_previous_checkpoint_and_manifest(self):
        self.complete_teacher()
        checkpoint(self.output / "joint/chkpnt1000.pth", 1)
        self.run_snapshot()
        slot = self.destination / "checkpoints/joint/latest.pth"
        before = snapshot.sha256(slot)
        checkpoint(self.output / "joint/chkpnt2000.pth", 2)
        with mock.patch.object(snapshot.shutil, "disk_usage", return_value=SimpleNamespace(free=0)):
            with self.assertRaisesRegex(RuntimeError, "previous backup retained"):
                self.run_snapshot()
        self.assertEqual(snapshot.sha256(slot), before)
        manifest = snapshot.read_json(self.destination / snapshot.MANIFEST)
        self.assertEqual(manifest["files"]["checkpoints/joint/latest.pth"]["iteration"], 1000)

    def test_corrupt_new_checkpoint_does_not_replace_good_slot(self):
        self.complete_teacher()
        checkpoint(self.output / "joint/chkpnt1000.pth", 1)
        self.run_snapshot()
        (self.output / "joint/chkpnt2000.pth").write_bytes(b"partial zip")
        result = self.run_snapshot()
        self.assertEqual(result["checkpoints"]["checkpoints/joint/latest.pth"]["iteration"], 1000)

    def test_best_validation_slots_are_independent_and_keep_restore_names(self):
        self.complete_teacher()
        for model in ("joint", "sequential"):
            checkpoint(self.output / model / "chkpnt2000.pth", 2)
            checkpoint(self.output / model / "best_val_chkpnt.pth", 1)
            (self.output / model / "validation_summary.json").write_text('{"best_iteration":1000}')
        result = self.run_snapshot()
        for model in ("joint", "sequential"):
            latest = self.destination / f"checkpoints/{model}/latest.pth"
            best = self.destination / f"checkpoints/{model}/best_val_chkpnt.pth"
            self.assertEqual(snapshot.sha256(latest), snapshot.sha256(self.output / model / "chkpnt2000.pth"))
            self.assertEqual(snapshot.sha256(best), snapshot.sha256(self.output / model / "best_val_chkpnt.pth"))
            record = result["checkpoints"][f"checkpoints/{model}/best_val_chkpnt.pth"]
            self.assertEqual(record["stage"], "best_validation")
            self.assertEqual(record["source_relative"], f"{model}/best_val_chkpnt.pth")
            self.assertEqual(record["validation_summary_relative"], f"{model}/validation_summary.json")
            self.assertEqual(record["validation_summary"]["best_iteration"], 1000)
            self.assertEqual(record["validation_summary_sha256"], snapshot.sha256(self.output / model / "validation_summary.json"))

    def test_new_best_quota_failure_preserves_old_best_and_latest(self):
        self.complete_teacher()
        latest_source = self.output / "joint/chkpnt2000.pth"
        best_source = self.output / "joint/best_val_chkpnt.pth"
        checkpoint(latest_source, 2)
        checkpoint(best_source, 1)
        summary = self.output / "joint/validation_summary.json"
        summary.write_text('{"best_iteration":1000}')
        self.run_snapshot()
        best_slot = self.destination / "checkpoints/joint/best_val_chkpnt.pth"
        latest_slot = self.destination / "checkpoints/joint/latest.pth"
        original_best, original_latest = snapshot.sha256(best_slot), snapshot.sha256(latest_slot)
        checkpoint(best_source, 3)
        summary.write_text('{"best_iteration":3000}')
        with mock.patch.object(snapshot.shutil, "disk_usage", return_value=SimpleNamespace(free=0)):
            with self.assertRaisesRegex(RuntimeError, "previous backup retained"):
                self.run_snapshot()
        self.assertEqual(snapshot.sha256(best_slot), original_best)
        self.assertEqual(snapshot.sha256(latest_slot), original_latest)
        self.assertTrue(best_source.is_file() and latest_source.is_file())
        manifest = snapshot.read_json(self.destination / snapshot.MANIFEST)
        self.assertEqual(manifest["files"]["checkpoints/joint/best_val_chkpnt.pth"]["validation_summary"]["best_iteration"], 1000)
        best_source.write_bytes(b"incomplete-new-best")
        self.run_snapshot()
        self.assertEqual(snapshot.sha256(best_slot), original_best)

    def test_sequential_semantic_slot_preserves_rgb_and_records_geometry_dependency(self):
        self.complete_teacher()
        rgb = self.output / "sequential/chkpnt15000.pth"
        semantic = self.output / "sequential/semantic/iteration_15000/semantic_checkpoint.pt"
        checkpoint(rgb, 1)
        checkpoint(semantic, 2)
        geometry = self.output / "sequential/point_cloud/iteration_15000/point_cloud.ply"
        geometry.parent.mkdir(parents=True)
        geometry.write_bytes(b"fixture-geometry")
        result = self.run_snapshot()
        rgb_slot = self.destination / "checkpoints/sequential/latest.pth"
        semantic_slot = self.destination / "checkpoints/sequential/semantic_latest.pt"
        self.assertEqual(snapshot.sha256(rgb_slot), snapshot.sha256(rgb))
        self.assertEqual(snapshot.sha256(semantic_slot), snapshot.sha256(semantic))
        record = result["checkpoints"]["checkpoints/sequential/semantic_latest.pt"]
        self.assertEqual(record["source_relative"], "sequential/semantic/iteration_15000/semantic_checkpoint.pt")
        self.assertEqual(record["rgb_export_iteration"], 15000)
        self.assertTrue(record["rgb_export_in_periodic_snapshot"])
        self.assertEqual(record["rgb_dependency_sha256"], snapshot.sha256(geometry))
        self.assertEqual(snapshot.sha256(self.destination / record["rgb_dependency_slot"]), snapshot.sha256(geometry))
        before = snapshot.sha256(semantic_slot)
        checkpoint(semantic, 3)
        with mock.patch.object(snapshot.shutil, "disk_usage", return_value=SimpleNamespace(free=0)):
            with self.assertRaisesRegex(RuntimeError, "previous backup retained"):
                self.run_snapshot()
        self.assertEqual(snapshot.sha256(semantic_slot), before)
        self.assertEqual(snapshot.sha256(rgb_slot), snapshot.sha256(rgb))
        geometry.write_bytes(b"different-geometry")
        with self.assertRaisesRegex(RuntimeError, "Frozen teacher changed"):
            self.run_snapshot()
        self.assertEqual(snapshot.sha256(semantic_slot), before)

    def test_source_change_during_copy_retains_previous_slot(self):
        self.complete_teacher()
        checkpoint(self.output / "joint/chkpnt1000.pth", 1)
        self.run_snapshot()
        newer = self.output / "joint/chkpnt2000.pth"
        checkpoint(newer, 2)
        original = snapshot.sha256
        changed = False

        def changing_source(path):
            nonlocal changed
            digest = original(path)
            if Path(path).resolve() == newer.resolve() and not changed:
                changed = True
                checkpoint(newer, 3)
            return digest

        with mock.patch.object(snapshot, "sha256", side_effect=changing_source):
            with self.assertRaisesRegex(RuntimeError, "Source changed"):
                self.run_snapshot()
        self.assertEqual(original(self.destination / "checkpoints/joint/latest.pth"),
                         original(self.output / "joint/chkpnt1000.pth"))

    def test_pending_after_atomic_replace_recovers_new_manifest_mapping(self):
        self.complete_teacher()
        checkpoint(self.output / "joint/chkpnt1000.pth", 1)
        self.run_snapshot()
        checkpoint(self.output / "joint/chkpnt2000.pth", 2)
        original = snapshot.save_manifest

        def crash_after_replace(destination, manifest):
            record = manifest["files"].get("checkpoints/joint/latest.pth", {})
            if manifest["pending"] is None and record.get("iteration") == 2000:
                raise RuntimeError("simulated crash after atomic replacement")
            original(destination, manifest)

        with mock.patch.object(snapshot, "save_manifest", side_effect=crash_after_replace):
            with self.assertRaisesRegex(RuntimeError, "simulated crash"):
                self.run_snapshot()
        pending = snapshot.read_json(self.destination / snapshot.MANIFEST)["pending"]
        self.assertEqual(pending["new"]["iteration"], 2000)
        result = self.run_snapshot()
        self.assertEqual(result["checkpoints"]["checkpoints/joint/latest.pth"]["iteration"], 2000)
        self.assertIsNone(snapshot.read_json(self.destination / snapshot.MANIFEST)["pending"])

    def test_pending_before_atomic_replace_recovers_verified_temporary(self):
        self.complete_teacher()
        checkpoint(self.output / "joint/chkpnt1000.pth", 1)
        self.run_snapshot()
        checkpoint(self.output / "joint/chkpnt2000.pth", 2)
        original = snapshot.os.replace

        def crash_at_slot(source, target):
            if Path(target).name == "latest.pth":
                raise RuntimeError("simulated pre-rename crash")
            original(source, target)

        with mock.patch.object(snapshot.os, "replace", side_effect=crash_at_slot):
            with self.assertRaisesRegex(RuntimeError, "pre-rename"):
                self.run_snapshot()
        result = self.run_snapshot()
        self.assertEqual(result["checkpoints"]["checkpoints/joint/latest.pth"]["iteration"], 2000)

    def test_readback_corruption_keeps_old_slot_and_discards_only_owned_temp(self):
        self.complete_teacher()
        checkpoint(self.output / "joint/chkpnt1000.pth", 1)
        self.run_snapshot()
        original_sha = snapshot.sha256
        checkpoint(self.output / "joint/chkpnt2000.pth", 2)

        def corrupt_temp(path):
            if str(path).endswith("latest.pth.snapshot-copy.tmp"):
                Path(path).write_bytes(b"broken-copy")
            return original_sha(path)

        with mock.patch.object(snapshot, "sha256", side_effect=corrupt_temp):
            with self.assertRaisesRegex(RuntimeError, "read-back"):
                self.run_snapshot()
        slot = self.destination / "checkpoints/joint/latest.pth"
        self.assertEqual(original_sha(slot), original_sha(self.output / "joint/chkpnt1000.pth"))
        manifest = snapshot.read_json(self.destination / snapshot.MANIFEST)
        snapshot.recover_pending(self.destination, manifest)
        self.assertEqual(manifest["files"]["checkpoints/joint/latest.pth"]["iteration"], 1000)
        self.assertFalse(slot.with_name(slot.name + ".snapshot-copy.tmp").exists())


if __name__ == "__main__":
    unittest.main()
