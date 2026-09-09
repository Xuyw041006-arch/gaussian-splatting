import hashlib
import importlib.util
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from scripts.score_saved_lerf_masks import evaluate, inspect_masks, write_result


HAS_IMAGES = all(importlib.util.find_spec(name) for name in ("numpy", "PIL"))
HAS_CV2 = importlib.util.find_spec("cv2") is not None


@unittest.skipUnless(HAS_IMAGES, "NumPy/Pillow unavailable")
class SavedMaskScoreTests(unittest.TestCase):
    labels = ["chopsticks", "egg", "glass of water", "pork belly", "wavy noodles in bowl", "yellow bowl"]

    def fixtures(self, root, prediction_size=(20, 15)):
        import numpy as np
        from PIL import Image
        predictions, gt = root / "predictions", root / "gt"
        predictions.mkdir()
        mask = np.zeros((30, 40), dtype=np.uint8)
        mask[4:26, 6:32] = 255
        image = Image.fromarray(mask)
        for split in ("0", "1", "2"):
            (gt / split).mkdir(parents=True)
            for label in self.labels:
                image.save(gt / split / f"{label}.png")
                image.resize(prediction_size, Image.Resampling.NEAREST).save(predictions / f"{split}_{label}.png")
        return predictions, gt

    def test_three_views_six_labels_and_exact_published_annotation_digest(self):
        with tempfile.TemporaryDirectory() as directory:
            predictions, gt = self.fixtures(Path(directory))
            result = inspect_masks(predictions, gt)
            self.assertEqual(len(result["rows"]), 18)
            self.assertEqual(result["splits"], ["0", "1", "2"])
            digest = hashlib.sha256()
            for path in sorted(gt.glob("*/*.png")):
                digest.update(str(path.relative_to(gt)).encode())
                digest.update(path.read_bytes())
            self.assertEqual(result["annotation_sha256"], digest.hexdigest())
            self.assertEqual(result["rows"][0]["prediction_width"], 20)
            self.assertEqual(result["rows"][0]["native_width"], 40)

    def test_missing_prediction_empty_directories_and_wrong_counts_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            predictions, gt = self.fixtures(root)
            (predictions / "0_egg.png").unlink()
            with self.assertRaisesRegex(ValueError, "Missing"):
                inspect_masks(predictions, gt)
            with self.assertRaisesRegex(ValueError, "Expected"):
                inspect_masks(predictions, gt, expected_views=4)
            with self.assertRaisesRegex(ValueError, "Expected"):
                inspect_masks(predictions, gt, expected_labels=7)
            empty = root / "empty"
            empty.mkdir()
            with self.assertRaises(ValueError):
                inspect_masks(predictions, empty)
            with self.assertRaises(ValueError):
                inspect_masks(empty, gt)

    def test_mixed_dimensions_bad_aspect_and_nonbinary_maps_rejected(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as directory:
            predictions, gt = self.fixtures(Path(directory))
            path = predictions / "0_egg.png"
            Image.new("L", (19, 15)).save(path)
            with self.assertRaisesRegex(ValueError, "dimensions"):
                inspect_masks(predictions, gt)
            Image.new("L", (20, 15), 120).save(path)
            with self.assertRaisesRegex(ValueError, "nonbinary"):
                inspect_masks(predictions, gt)
            Image.new("RGB", (20, 15)).save(path)
            with self.assertRaisesRegex(ValueError, "overlays"):
                inspect_masks(predictions, gt)
        with tempfile.TemporaryDirectory() as directory:
            predictions, gt = self.fixtures(Path(directory), (15, 20))
            with self.assertRaisesRegex(ValueError, "aspect"):
                inspect_masks(predictions, gt)

    def test_changed_gt_changes_both_native_fingerprints(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as directory:
            predictions, gt = self.fixtures(Path(directory))
            before = inspect_masks(predictions, gt)
            Image.new("L", (40, 30), 255).save(gt / "0/egg.png")
            after = inspect_masks(predictions, gt)
            self.assertNotEqual(before["annotation_sha256"], after["annotation_sha256"])
            self.assertNotEqual(before["native_annotation_fingerprint"], after["native_annotation_fingerprint"])
            self.assertEqual(before["prediction_masks_sha256"], after["prediction_masks_sha256"])

    def test_output_never_overwrites(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "new.json"
            write_result(output, {"saved": True})
            with self.assertRaises(FileExistsError):
                write_result(output, {"saved": False})
            self.assertTrue(json.loads(output.read_text())["saved"])

    @unittest.skipUnless(HAS_CV2, "OpenCV unavailable for shared boundary evaluator")
    def test_actual_metrics_match_published_mask_scorer_without_changing_inputs(self):
        from scripts.evaluate_published_masks import evaluate as evaluate_published
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            predictions, gt = self.fixtures(root, (40, 30))
            before = {path: path.read_bytes() for path in predictions.glob("*.png")}
            archive = root / "published.zip"
            with zipfile.ZipFile(archive, "w") as output:
                for path in sorted(gt.glob("*/*.png")):
                    output.writestr(f"lerf_mask/ramen/{path.parent.name}/{path.name}",
                                    (predictions / f"{path.parent.name}_{path.name}").read_bytes())
            local = evaluate(predictions, gt)
            official = evaluate_published(archive, gt, expected_sha256=None)
            self.assertEqual(local["annotation_sha256"], official["annotation_sha256"])
            self.assertEqual(local["mean_iou"], official["mean_iou"])
            self.assertEqual(local["mean_boundary_iou"], official["mean_boundary_iou"])
            self.assertEqual(local["mean_iou"], 1.0)
            self.assertFalse(local["protocol"]["threshold_tuning"])
            self.assertNotIn("test_psnr", local)
            self.assertEqual(before, {path: path.read_bytes() for path in predictions.glob("*.png")})


if __name__ == "__main__":
    unittest.main()
