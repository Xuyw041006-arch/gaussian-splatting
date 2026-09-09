import unittest

import numpy as np

from semantic.artifact import (
    clip_space_cosines, cosine_scores, pairwise_relevancy, project_clip_feature,
    text_retrieval_scores, affinity_point_scores,
)
from scripts.evaluate_lerf_mask import aggregate_mask_rows, evaluate_prediction_mask, mask_boundary


class SemanticScoringTests(unittest.TestCase):
    def test_clip_cosine_matches_explicit_inverse_pca_without_mutating_queries(self):
        rng = np.random.default_rng(42)
        # Non-orthonormal components check the Gram-matrix formula, too.
        components = rng.normal(size=(3, 7)).astype(np.float32)
        mean = rng.normal(size=7).astype(np.float32)
        latent = rng.normal(size=(8, 3)).astype(np.float32)
        text = rng.normal(size=(2, 7)).astype(np.float32)
        original = text.copy()
        reconstructed = latent @ components + mean
        expected = np.stack([cosine_scores(reconstructed, query) for query in text], axis=1)
        actual = clip_space_cosines(latent, text, mean, components, chunk_size=3)
        np.testing.assert_allclose(actual, expected, atol=2e-6)
        np.testing.assert_array_equal(text, original)

    def test_centered_pca_cosine_is_not_clip_cosine_and_legacy_stays_unchanged(self):
        values, text = np.array([[-0.6, 0.6]]), np.array([[1.0, 0.0]])
        mean, components = np.array([0.8, 0.2]), np.eye(2)
        legacy = text_retrieval_scores(values, text, mean, components)
        expected = cosine_scores(values, project_clip_feature(text[0], mean, components))
        np.testing.assert_allclose(legacy[:, 0], expected)
        corrected = text_retrieval_scores(values, text, mean, components, mode="clip_cosine")
        self.assertLess(legacy[0, 0], -0.99)
        self.assertGreater(corrected[0, 0], 0.24)

    def test_relevancy_uses_hardest_negative_not_average_or_multiclass_softmax(self):
        positive = np.array([[0.3, 0.8]], dtype=np.float32)
        negatives = np.array([[0.2, 0.4]], dtype=np.float32)
        actual = pairwise_relevancy(positive, negatives, 10)
        expected = 1 / (1 + np.exp(-10 * (positive - 0.4)))
        np.testing.assert_allclose(actual, expected, atol=1e-6)
        unchanged = pairwise_relevancy(positive, np.concatenate([negatives, negatives], axis=1))
        np.testing.assert_array_equal(actual, unchanged)
        with self.assertRaises(ValueError):
            pairwise_relevancy(positive, np.empty((1, 0)))

    def test_zero_clip_feature_and_large_margin_are_finite(self):
        result = clip_space_cosines(np.zeros((1, 2)), np.eye(2), np.zeros(2), np.eye(2))
        np.testing.assert_array_equal(result, np.zeros((1, 2)))
        self.assertTrue(np.isfinite(pairwise_relevancy([[1, -1]], [[0]], 10000)).all())

    def test_affinity_hierarchy_uses_exported_prefix_without_language_transform(self):
        features = np.array([[1, 0, 1, 1], [1, 0, -1, -1], [0, 1, 1, -1]], dtype=np.float32)
        coarse = affinity_point_scores(features, 0, (2, 3, 4), 0)
        fine = affinity_point_scores(features, 0, (2, 3, 4), 2)
        self.assertAlmostEqual(coarse[1], 1.0)
        self.assertLess(fine[1], 0)
        self.assertAlmostEqual(fine[0], 1.0, places=6)
        with self.assertRaises(ValueError):
            affinity_point_scores(features, 5, (2, 3, 4), 0)

    def test_macro_vs_view_label_mean_are_recorded_different_protocols(self):
        rows = [{"label": "a", "iou": 1, "boundary_iou": 1},
                {"label": "a", "iou": 1, "boundary_iou": 1},
                {"label": "b", "iou": 0, "boundary_iou": 0}]
        self.assertAlmostEqual(aggregate_mask_rows(rows, ["a", "b"], "legacy")["mean_iou"], 2 / 3)
        self.assertEqual(aggregate_mask_rows(rows, ["a", "b"], "gg_native")["mean_iou"], 0.5)

    def test_native_gt_and_padded_boundary_are_explicit_opt_in(self):
        try:
            import cv2  # noqa: F401
            from PIL import Image  # noqa: F401
        except ImportError:
            self.skipTest("Boundary protocol fixture needs OpenCV/Pillow")
        full = np.ones((5, 7), dtype=bool)
        self.assertFalse(mask_boundary(full, 0.02).any())
        self.assertTrue(mask_boundary(full, 0.02, pad_edges=True).any())
        prediction = np.ones((2, 2), dtype=bool)
        native = np.ones((4, 4), dtype=np.uint8) * 255
        native[0, 0] = 0
        old = evaluate_prediction_mask(prediction, prediction, native, "legacy", .008)
        new = evaluate_prediction_mask(prediction, prediction, native, "gg_native", .02)
        self.assertEqual(old["iou"], 1)
        self.assertEqual(new["iou"], 15 / 16)
        self.assertEqual((new["metric_width"], new["metric_height"]), (4, 4))


if __name__ == "__main__":
    unittest.main()
