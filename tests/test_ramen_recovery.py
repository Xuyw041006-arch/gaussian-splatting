import hashlib
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from scripts.run_ramen_recovery import (
    atomic_copy, backup_model, complete_torch_archive, copy_scene,
    establish_semantic_reference, latest_checkpoint, metrics_reusable,
    semantic_space_fingerprint, log_reports_storage_exhaustion, Recovery, StorageExhaustedError,
)


def checkpoint_file(path, payload=b"data"):
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("checkpoint/data.pkl", payload)
        archive.writestr("checkpoint/version", "3\n")
        archive.writestr("checkpoint/data/0", payload)


class RamenRecoveryTests(unittest.TestCase):
    def test_failed_backup_status_exposes_local_only_and_trash_quota_risk(self):
        with tempfile.TemporaryDirectory() as directory:
            recovery = Recovery.__new__(Recovery)
            recovery.work = Path(directory) / "work"
            recovery.v4_drive = Path(directory) / "drive"
            recovery.status = {}
            recovery.record_backup_status({"state": "failed", "backed_up": False, "training_continues_locally": True})
            value = json.loads((recovery.work / "backup_status.json").read_text())
            self.assertFalse(value["backed_up"])
            self.assertIn("Trash", value["quota_note"])
            self.assertTrue(value["training_continues_locally"])

    def recovery_route(self, skip):
        recovery = Recovery.__new__(Recovery)
        recovery.args = SimpleNamespace(skip_v3_recovery=skip)
        recovery.status = {}
        recovery.save_status = Mock()
        recovery.stage = Mock()
        return recovery

    def test_skip_v3_routes_to_no_v3_training_or_evaluation(self):
        recovery = self.recovery_route(True)
        recovery.run_v3_stages(Path("unused"))
        recovery.stage.assert_not_called()
        self.assertEqual(recovery.status["v3_recovery"], "explicitly_skipped")

    def test_default_route_includes_all_three_v3_stages(self):
        recovery = self.recovery_route(False)
        recovery.run_v3_stages(Path("unused"))
        self.assertEqual([call.args[0] for call in recovery.stage.call_args_list], [
            "v3_joint_evaluation", "v3_baseline_resume", "v3_comparison_report",
        ])

    def test_v3_storage_failure_does_not_advance_to_comparison(self):
        recovery = self.recovery_route(False)
        recovery.stage.side_effect = [None, StorageExhaustedError("quota full")]
        with self.assertRaises(StorageExhaustedError):
            recovery.run_v3_stages(Path("unused"))
        self.assertEqual([call.args[0] for call in recovery.stage.call_args_list], [
            "v3_joint_evaluation", "v3_baseline_resume",
        ])

    def test_storage_error_is_detected_from_failed_process_log(self):
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "train.log"
            log.write_text("checkpoint save failed: OSError: [Errno 28] No space left on device")
            self.assertTrue(log_reports_storage_exhaustion(log))
            log.write_text("Training completed without errors")
            self.assertFalse(log_reports_storage_exhaustion(log))

    def test_scene_copy_omits_old_semantics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "old"
            for name in ("images", "images_train", "sparse", "test_mask", "semantic_maps", "importance_masks"):
                (source / name).mkdir(parents=True)
                (source / name / "file").write_bytes(name.encode())
            copy_scene(source, root / "new")
            self.assertTrue((root / "new/images/file").is_file())
            self.assertTrue((root / "new/sparse/file").is_file())
            self.assertFalse((root / "new/semantic_maps").exists())
            self.assertFalse((root / "new/importance_masks").exists())

    def test_latest_ignores_temporary_and_malformed_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("chkpnt7000.pth", "chkpnt10000.pth", "chkpnt15000.pth.tmp", "chkpntbad.pth"):
                checkpoint_file(root / name)
            (root / "chkpnt14000.pth").write_bytes(b"truncated torch.save file")
            self.assertEqual(latest_checkpoint(root).name, "chkpnt10000.pth")

    def test_backup_rotates_only_script_owned_prior_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            source, destination = Path(directory) / "source", Path(directory) / "mirror"
            source.mkdir()
            destination.mkdir()
            checkpoint_file(source / "chkpnt7000.pth", b"old")
            (destination / "chkpnt1234.pth").write_bytes(b"unmanaged")
            backup_model(source, destination)
            checkpoint_file(source / "chkpnt10000.pth", b"newest")
            backup_model(source, destination)
            self.assertEqual((destination / "chkpnt10000.pth").read_bytes(), (source / "chkpnt10000.pth").read_bytes())
            self.assertFalse((destination / "chkpnt7000.pth").exists())
            self.assertEqual((destination / "chkpnt1234.pth").read_bytes(), b"unmanaged")
            self.assertTrue((source / "chkpnt7000.pth").exists())

    def test_failed_new_copy_preserves_previous_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            source, destination = Path(directory) / "source", Path(directory) / "mirror"
            source.mkdir()
            checkpoint_file(source / "chkpnt7000.pth", b"safe")
            backup_model(source, destination)
            checkpoint_file(source / "chkpnt10000.pth", b"new")
            with patch("scripts.run_ramen_recovery.atomic_copy", side_effect=OSError("quota exceeded")):
                with self.assertRaises(OSError):
                    backup_model(source, destination)
            self.assertEqual((destination / "chkpnt7000.pth").read_bytes(), (source / "chkpnt7000.pth").read_bytes())
            self.assertFalse((destination / "chkpnt10000.pth").exists())

    def test_atomic_copy_reports_exact_bytes_and_checksum(self):
        with tempfile.TemporaryDirectory() as directory:
            source, target = Path(directory) / "source", Path(directory) / "target"
            source.write_bytes(b"completed checkpoint")
            record = atomic_copy(source, target)
            self.assertEqual(target.read_bytes(), source.read_bytes())
            self.assertEqual(record["sha256"], hashlib.sha256(source.read_bytes()).hexdigest())
            self.assertEqual(list(Path(directory).glob("*.tmp")), [])

    def test_old_unfingerprinted_metrics_are_not_reused(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metrics = root / "metrics.json"
            metrics.write_text(json.dumps({"test_psnr": 30.2, "iteration": 15000}))
            self.assertFalse(metrics_reusable(metrics, root, root, __file__))

    def test_corrupted_new_snapshot_and_best_do_not_replace_good_backups(self):
        with tempfile.TemporaryDirectory() as directory:
            source, destination = Path(directory) / "source", Path(directory) / "mirror"
            source.mkdir()
            checkpoint_file(source / "chkpnt7000.pth")
            checkpoint_file(source / "chkpnt10000.pth")
            checkpoint_file(source / "best_val_chkpnt.pth", b"best")
            backup_model(source, destination)
            old_best = (destination / "best_val_chkpnt.pth").read_bytes()
            (source / "chkpnt10000.pth").write_bytes(b"broken")
            (source / "best_val_chkpnt.pth").write_bytes(b"partial")
            backup_model(source, destination)
            self.assertTrue(complete_torch_archive(destination / "chkpnt10000.pth"))
            self.assertEqual((destination / "best_val_chkpnt.pth").read_bytes(), old_best)
            self.assertFalse((destination / "chkpnt7000.pth").exists())

    def test_atomic_checkpoint_copy_rejects_incomplete_archive(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "source").write_bytes(b"not complete")
            (root / "target").write_bytes(b"old")
            with self.assertRaises(RuntimeError):
                atomic_copy(root / "source", root / "target", verify_checkpoint=True)
            self.assertEqual((root / "target").read_bytes(), b"old")


class SemanticSpaceRecoveryTests(unittest.TestCase):
    def scene(self, root, change=0):
        import numpy as np

        root.mkdir(parents=True, exist_ok=True)
        (root / "sparse/0").mkdir(parents=True, exist_ok=True)
        for split, names in (("train", "train.jpg\n"), ("val", "val.jpg\n"), ("test", "test_00.jpg\n")):
            (root / "sparse/0" / (split + ".txt")).write_text(names)
        np.savez(root / "semantic_meta.npz", pca_components=np.eye(3) + change,
                 pca_mean=np.zeros(3), feature_min=-np.ones(3), feature_max=np.ones(3),
                 prototype_features=np.ones((2, 3)), clip_model=np.array("ViT-H-14"),
                 clip_pretrained=np.array("weights"), fit_image_names=np.array(["train.jpg"]),
                 heldout_image_names=np.array(["val.jpg"]))
        for name in ("semantic_summary.json", "scene_inventory.json"):
            (root / name).write_text("{}")

    def test_regenerated_matching_npz_can_resume_and_reference_is_persisted(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.scene(root / "scene")
            first = establish_semantic_reference(root / "scene", root / "drive/data_provenance", root / "work/outputs")
            self.scene(root / "scene")
            second = establish_semantic_reference(root / "scene", root / "drive/data_provenance", root / "work/outputs")
            self.assertEqual(first["fingerprint"], second["fingerprint"])
            self.assertTrue((root / "drive/data_provenance/semantic_meta.npz").is_file())
            self.assertTrue((root / "drive/data_provenance/sparse/0/train.txt").is_file())

    def test_changed_fit_blocks_resume_without_overwriting_original_reference(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.scene(root / "scene")
            reference = root / "drive/data_provenance"
            establish_semantic_reference(root / "scene", reference, root / "outputs")
            before = semantic_space_fingerprint(reference)
            self.scene(root / "scene", change=0.01)
            with self.assertRaisesRegex(RuntimeError, "embedding space changed"):
                establish_semantic_reference(root / "scene", reference, root / "outputs")
            self.assertEqual(semantic_space_fingerprint(reference), before)

    def test_weights_without_saved_reference_cannot_be_adopted_silently(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.scene(root / "scene")
            (root / "outputs/joint").mkdir(parents=True)
            checkpoint_file(root / "outputs/joint/chkpnt7000.pth")
            with self.assertRaisesRegex(RuntimeError, "no immutable semantic-space reference"):
                establish_semantic_reference(root / "scene", root / "drive/data_provenance", root / "outputs")


if __name__ == "__main__":
    unittest.main()
