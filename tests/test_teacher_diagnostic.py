import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.diagnose_semantic_teacher import (
    evaluate_label, match_image, oracle_single_mask, prediction_masks,
    score_variants, unpack_masks,
)


class TeacherDiagnosticTests(unittest.TestCase):
    def test_missing_test_identity_is_not_guessed_from_folder_number(self):
        paths = [Path("frame_000.jpg"), Path("test_03.jpg")]
        with self.assertRaises(ValueError):
            match_image(paths, "0")
        self.assertEqual(match_image(paths, "3", {"3": "test_03.jpg"}), paths[1])

    def test_packed_masks_roundtrip_handles_non_multiple_of_eight(self):
        masks = np.array([[[True, False, True], [False, True, False]]])
        raw = {"packed_region_masks": np.packbits(masks.reshape(1, -1), axis=1), "mask_shape": [2, 3]}
        np.testing.assert_array_equal(unpack_masks(raw), masks)

    def test_union_and_overwrite_map_expose_parent_coverage_loss(self):
        parent = np.ones((4, 4), dtype=bool)
        child = np.zeros((4, 4), dtype=bool)
        child[1:3, 1:3] = True
        region_map = np.where(child, 1, 0)
        masks = np.stack([parent, child])
        union, dense, selected = prediction_masks(np.array([0.9, 0.1]), masks, region_map, 0.25)
        self.assertTrue(union.all())
        self.assertFalse(dense[1:3, 1:3].any())
        self.assertEqual(selected, 1)
        metrics, _ = evaluate_label(masks, region_map, np.array([0.9, 0.1]), parent, 0.25)
        self.assertEqual(metrics["union_iou"], 1.0)
        self.assertEqual(metrics["region_map_iou"], 0.75)

    def test_below_threshold_does_not_force_any_region(self):
        mask = np.ones((4, 4), dtype=bool)
        union, dense, count = prediction_masks(np.array([0.1]), mask[None], np.zeros((4, 4), dtype=int), 0.25)
        self.assertFalse(union.any() or dense.any())
        self.assertEqual(count, 0)

    def test_oracle_selects_mask_but_does_not_change_prediction(self):
        masks = np.zeros((2, 4, 4), dtype=bool)
        masks[0, :2] = True
        masks[1, 2:] = True
        target = masks[1]
        best_iou, index, best = oracle_single_mask(masks, target)
        self.assertEqual(best_iou, 1.0)
        self.assertEqual(index, 1)
        np.testing.assert_array_equal(best, target)
        union, _, _ = prediction_masks(np.array([0.9, 0.1]), masks, np.where(target, 1, 0), 0.25)
        self.assertFalse(np.logical_and(union, target).any())

    def test_fixed_full_rank_pca_reconstruction_preserves_clip_scores(self):
        features = np.array([[1., 0.], [0., 1.]], dtype=np.float32)
        positives = features.copy()
        negatives = np.array([[-1., 0.]], dtype=np.float32)
        metadata = {"pca_mean": np.array([0.2, 0.3], dtype=np.float32),
                    "pca_components": np.eye(2, dtype=np.float32),
                    "feature_min": np.array([-2., -2.]), "feature_max": np.array([2., 2.])}
        before = {key: value.copy() for key, value in metadata.items()}
        variants = score_variants(features, positives, negatives, metadata)
        np.testing.assert_allclose(variants["raw_clip_cosine"], variants["pca_clip_cosine"], atol=1e-6)
        self.assertFalse(np.allclose(variants["raw_clip_cosine"], variants["pca_bounded_legacy_cosine"]))
        self.assertNotIn("cached_aggregated_clip_cosine", variants)
        for key in metadata:
            np.testing.assert_array_equal(metadata[key], before[key])


if __name__ == "__main__":
    unittest.main()
