import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from scripts.evaluate_lerf_mask import (
    camera_for_split, mask_splits, protocol_metadata, save_mask_visuals,
    save_rgb_visuals, weight_selection,
)


class LerfEvalProtocolTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_numeric_split_does_not_fall_back_to_an_unrelated_camera(self):
        cameras = [SimpleNamespace(image_name="frame_0007.jpg")]
        with self.assertRaisesRegex(ValueError, "Numeric-index"):
            camera_for_split(cameras, "0")
        self.assertIs(camera_for_split(cameras, "0", {"0": "frame_0007.jpg"}), cameras[0])

    def test_explicit_identity_and_ambiguity(self):
        camera = SimpleNamespace(image_name="test_0.png")
        self.assertIs(camera_for_split([camera], "0"), camera)
        with self.assertRaisesRegex(ValueError, "unambiguous"):
            camera_for_split([camera, SimpleNamespace(image_name="0.png")], "0")
        with self.assertRaisesRegex(ValueError, "unambiguous"):
            camera_for_split([camera, SimpleNamespace(image_name="test_0.jpg")], "0")

    def test_empty_mask_root_and_empty_split_fail(self):
        with self.assertRaisesRegex(ValueError, "No annotated"):
            mask_splits(self.root)
        split = self.root / "0"
        split.mkdir()
        with self.assertRaisesRegex(ValueError, "no PNG"):
            mask_splits(self.root)
        (split / "apple.png").write_bytes(b"fixture")
        self.assertEqual(mask_splits(self.root), [split])

    def test_best_alias_source_is_not_reported_as_final_training_iteration(self):
        artifact = self.root / "semantic_features.pt"
        artifact.write_bytes(b"fixture")
        (self.root / "validation_summary.json").write_text(json.dumps({
            "best_iteration": 14000, "best_psnr": 24.0, "trained_iterations": 15000,
            "selected_iteration_alias": 15000,
        }))
        selected = weight_selection(self.root, 15000, artifact)
        self.assertEqual(selected["source_training_iteration"], 14000)
        self.assertEqual(selected["selection"], "recorded_validation_best_alias")
        other = weight_selection(self.root, 7000, artifact)
        self.assertIsNone(other["source_training_iteration"])
        self.assertEqual(other["selection"], "requested_iteration_export_origin_unverified")

    def test_protocol_separates_annotated_from_all_test_and_fingerprints_input(self):
        views = [{"split": "0", "camera": "test_0", "labels": ["apple"],
                  "width": 10, "height": 12, "ground_truth_sha256": "a"}]
        protocol = protocol_metadata(views, ["test_0", "test_1"], 0.25, 1, 0.008)
        self.assertEqual(protocol["annotated_camera_names"], ["test_0"])
        self.assertEqual(protocol["test_camera_names"], ["test_0", "test_1"])
        self.assertEqual(protocol["boundary_ratio"], 0.008)
        self.assertEqual(protocol["metric_scope"], "annotated_test_mask_views")
        altered = [dict(views[0], width=20)]
        changed = protocol_metadata(altered, ["test_0", "test_1"], 0.25, 1, 0.008)
        self.assertNotEqual(protocol["dataset_fingerprint"], changed["dataset_fingerprint"])

    def test_visualizations_keep_target_and_prediction_aligned_to_same_gt(self):
        try:
            import numpy as np
            from PIL import Image
        except ImportError:
            self.skipTest("Visual fixture needs optional NumPy and Pillow")
        gt = Image.new("RGB", (20, 10), (30, 40, 50))
        rgb = Image.new("RGB", (20, 10), (31, 40, 50))
        paths = save_rgb_visuals(self.root, "0", "test_0", gt, rgb)
        target = np.zeros((10, 20), dtype=bool)
        target[2:5, 3:8] = True
        prediction = np.roll(target, 1, axis=1)
        masks = save_mask_visuals(self.root, paths["ground_truth"], "apple", gt, prediction, target)
        for path in [*paths.values(), *masks.values()]:
            self.assertTrue((self.root / path).is_file())
        with Image.open(self.root / paths["rgb_comparison"]) as panel:
            self.assertEqual(panel.size, (40, 38))
        with Image.open(self.root / masks["target_mask"]) as saved:
            self.assertTrue(np.array_equal(np.asarray(saved) > 0, target))
        with Image.open(self.root / masks["prediction_mask"]) as saved:
            self.assertTrue(np.array_equal(np.asarray(saved) > 0, prediction))


if __name__ == "__main__":
    unittest.main()
