import unittest
import importlib.util

import numpy as np

try:
    import torch

    from preprocess_semantics import (
        aggregate_cross_view_features,
        build_detail_supervision,
        build_hierarchy_region_maps,
        fit_training_pca,
        select_prompt_regions,
    )
    from semantic.joint import (
        GRANULARITIES, boundary_alignment_loss, granularity_for_step,
        local_semantic_consistency, project_tiers_to_gaussians,
        region_boundaries, region_contrastive_loss, semantic_chunk_indices,
    )
    DEPENDENCIES_AVAILABLE = True
except ModuleNotFoundError:
    DEPENDENCIES_AVAILABLE = False


@unittest.skipUnless(DEPENDENCIES_AVAILABLE, "semantic dependencies are optional locally")
class JointSemanticTests(unittest.TestCase):
    def test_hierarchy_separates_area_scales(self):
        regions = [
            {"area": 64, "segmentation": np.ones((8, 8), dtype=bool)},
            {"area": 12, "segmentation": np.pad(np.ones((3, 4), dtype=bool), ((0, 5), (0, 4)))},
            {"area": 1, "segmentation": np.pad(np.ones((1, 1), dtype=bool), ((0, 7), (0, 7)))},
        ]
        maps = build_hierarchy_region_maps(
            regions, (8, 8), image_area=64, fine_ratio=0.05, coarse_ratio=0.25
        )
        self.assertEqual(maps.shape, (3, 8, 8))
        self.assertTrue(np.any(maps[0] == 0))
        self.assertTrue(np.any(maps[1] == 1))
        self.assertTrue(np.any(maps[2] == 2))

    @unittest.skipUnless(importlib.util.find_spec("sklearn"), "scikit-learn is optional locally")
    def test_cross_view_prototypes_pull_related_descriptors_together(self):
        features = np.array([
            [1.0, 0.0], [0.8, 0.2], [-1.0, 0.0], [-0.8, 0.2],
        ], dtype=np.float32)
        features /= np.linalg.norm(features, axis=1, keepdims=True)
        output, labels, blend = aggregate_cross_view_features(
            features, np.ones(4, dtype=np.float32), max_prototypes=2, weight=0.8
        )
        self.assertEqual(len(np.unique(labels)), 2)
        self.assertTrue(np.all(blend > 0))
        original_similarity = np.dot(features[0], features[1])
        output_similarity = np.dot(output[0], output[1])
        self.assertGreater(output_similarity, original_similarity)

    @unittest.skipUnless(importlib.util.find_spec("sklearn"), "scikit-learn is optional locally")
    def test_cross_view_prototypes_can_return_global_centers(self):
        features = np.array([
            [1.0, 0.0], [0.9, 0.1], [-1.0, 0.0], [-0.9, 0.1],
        ], dtype=np.float32)
        features /= np.linalg.norm(features, axis=1, keepdims=True)
        _, labels, _, centers = aggregate_cross_view_features(
            features, np.ones(4, dtype=np.float32),
            max_prototypes=2, weight=0.8, return_centers=True,
        )
        self.assertEqual(centers.shape, (2, 2))
        self.assertEqual(len(np.unique(labels)), 2)

    @unittest.skipUnless(importlib.util.find_spec("sklearn"), "scikit-learn is optional locally")
    def test_heldout_features_do_not_change_fitted_prototypes_or_training_targets(self):
        training = np.array([
            [1.0, 0.0], [0.9, 0.1], [-1.0, 0.0], [-0.9, 0.1],
        ], dtype=np.float32)
        training /= np.linalg.norm(training, axis=1, keepdims=True)
        original, _, _, centers = aggregate_cross_view_features(
            training, np.ones(4, dtype=np.float32), max_prototypes=2,
            return_centers=True,
        )
        all_features = np.vstack([training, [[0.0, 1.0], [0.0, -1.0]]]).astype(np.float32)
        transformed, _, _, fitted_centers = aggregate_cross_view_features(
            all_features, np.ones(6, dtype=np.float32), max_prototypes=2,
            return_centers=True, fit_mask=np.array([True] * 4 + [False] * 2),
        )
        np.testing.assert_allclose(fitted_centers, centers)
        np.testing.assert_allclose(transformed[:4], original)
        self.assertEqual(transformed.shape, all_features.shape)

    @unittest.skipUnless(importlib.util.find_spec("sklearn"), "scikit-learn is optional locally")
    def test_heldout_outlier_cannot_change_pca_or_encoding_bounds(self):
        training = np.array([[1., 0., 2.], [0., 2., 1.], [3., 1., 0.], [1., 3., 4.]], dtype=np.float32)
        pca, projected, lower, upper = fit_training_pca(training, 3, np.ones(4, dtype=bool))
        all_features = np.vstack([training, [[1000., -1000., 1000.]]]).astype(np.float32)
        fit, transformed, fit_lower, fit_upper = fit_training_pca(
            all_features, 3, np.array([True] * 4 + [False]),
        )
        np.testing.assert_allclose(fit.mean_, pca.mean_)
        np.testing.assert_allclose(fit.components_, pca.components_)
        np.testing.assert_allclose(transformed[:4], projected)
        np.testing.assert_allclose(fit_lower, lower)
        np.testing.assert_allclose(fit_upper, upper)
        self.assertTrue(np.any((transformed[4] < fit_lower) | (transformed[4] > fit_upper)))

    def test_thin_regions_and_boundaries_receive_more_weight(self):
        region_map = np.full((24, 36), -1, dtype=np.int16)
        region_map[2:14, 2:14] = 0
        region_map[18:20, 3:33] = 1
        importance = np.zeros_like(region_map, dtype=np.uint8)
        detail, boundary, thinness, enhanced = build_detail_supervision(
            region_map, importance, boundary_width=2,
        )
        self.assertGreater(thinness[region_map == 1].mean(), thinness[region_map == 0].mean())
        self.assertTrue(np.all(enhanced[region_map == 1] >= 1))
        self.assertGreater(detail[boundary].mean(), detail[~boundary].mean())

    def test_granularity_cycle(self):
        self.assertEqual(
            [GRANULARITIES[granularity_for_step(step)] for step in range(6)],
            ["coarse", "middle", "fine", "coarse", "middle", "fine"],
        )

    def test_importance_projection_accepts_renderer_boolean_mask(self):
        class Camera:
            full_proj_transform = torch.eye(4)

        xyz = torch.tensor([
            [0.0, 0.0, 1.0], [0.5, 0.0, 1.0], [-0.5, 0.0, 1.0],
        ])
        tiers = torch.ones((4, 4), dtype=torch.long)
        selected, observations = project_tiers_to_gaussians(
            xyz, Camera(), tiers, torch.tensor([True, False, True])
        )
        self.assertEqual(selected.tolist(), [0, 2])
        self.assertEqual(observations.tolist(), [0.5, 0.5])

    def test_importance_projection_matches_cuda_pixel_centers_without_y_flip(self):
        class Camera:
            full_proj_transform = torch.eye(4)

        # Invert CUDA ndc2Pix for exact pixel centers. The labels intentionally
        # differ above and below the center so a vertical flip cannot pass.
        height, width = 6, 4
        pixels = [(1, 0), (1, 5), (3, 2)]
        xyz = torch.tensor([
            [(2 * x + 1) / width - 1, (2 * y + 1) / height - 1, 1.0]
            for x, y in pixels
        ])
        tiers = torch.zeros((height, width), dtype=torch.long)
        tiers[0, 1] = 2
        tiers[2, 3] = 1
        selected, observations = project_tiers_to_gaussians(
            xyz, Camera(), tiers, torch.arange(len(pixels))[:, None]
        )
        self.assertEqual(selected.tolist(), [0, 1, 2])
        self.assertEqual(observations.tolist(), [1.0, 0.0, 0.5])

    def test_importance_projection_rejects_points_just_outside_image(self):
        class Camera:
            full_proj_transform = torch.eye(4)

        xyz = torch.tensor([
            [-1.001, 0.0, 1.0], [0.0, -1.001, 1.0],
            [1.001, 0.0, 1.0], [0.0, 1.001, 1.0], [0.0, 0.0, 1.0],
        ])
        selected, _ = project_tiers_to_gaussians(
            xyz, Camera(), torch.ones((6, 4), dtype=torch.long), torch.arange(5)
        )
        self.assertEqual(selected.tolist(), [4])

    def test_validation_covers_every_channel_independent_of_iteration(self):
        for iteration in (1000, 7000, 14000, 15000):
            chunks = semantic_chunk_indices(32, 3, iteration, validation=True)
            channels = [
                channel for chunk in chunks
                for channel in range(3 * chunk, min(3 * chunk + 3, 32))
            ]
            self.assertEqual(channels, list(range(32)))
            self.assertEqual(len(semantic_chunk_indices(32, 3, iteration)), 3)

    def test_prompt_topk_never_forces_an_absent_object(self):
        features = np.array([[0.1, 0.9], [0.2, 0.8]], dtype=np.float32)
        text = np.array([[1.0, 0.0]], dtype=np.float32)
        self.assertEqual(select_prompt_regions(features, text, 0.24, topk=1), set())

    def test_prompt_threshold_keeps_all_supported_instances_by_default(self):
        features = np.array([[0.9, 0.1], [0.8, 0.2], [0.1, 0.9]], dtype=np.float32)
        text = np.array([[1.0, 0.0]], dtype=np.float32)
        self.assertEqual(select_prompt_regions(features, text, 0.24, topk=0), {0, 1})
        self.assertEqual(select_prompt_regions(features, text, 0.24, topk=1), {0})

    def test_local_semantic_consistency_supports_backward(self):
        class Gaussians:
            def __init__(self):
                self._features = torch.nn.Parameter(torch.randn(8, 4))
                self._xyz = torch.nn.Parameter(torch.randn(8, 3))
                self.importance_score = torch.linspace(0.0, 1.0, 8)

            @property
            def get_semantic_features(self):
                return self._features

            @property
            def get_xyz(self):
                return self._xyz

        gaussians = Gaussians()
        loss = local_semantic_consistency(
            gaussians, samples=8, edge_sigma=0.12
        )
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertIsNotNone(gaussians._features.grad)
        self.assertIsNotNone(gaussians._xyz.grad)

    def test_boundary_alignment_prefers_matching_edges(self):
        target = torch.zeros((3, 5, 6))
        target[:, :, 3:] = 1.0
        boundary = torch.zeros((5, 6), dtype=torch.bool)
        boundary[:, 2:4] = True
        valid = torch.ones((5, 6), dtype=torch.bool)
        matching = boundary_alignment_loss(target.clone(), target, boundary, valid)
        flat = boundary_alignment_loss(torch.zeros_like(target), target, boundary, valid)
        self.assertLess(float(matching), float(flat))

    def test_boundaries_follow_selected_hierarchy_instead_of_fine_parts(self):
        fine_ids = torch.zeros((5, 6), dtype=torch.long)
        fine_ids[:, 3:] = 1
        coarse_ids = torch.zeros_like(fine_ids)
        valid = torch.ones_like(fine_ids, dtype=torch.bool)
        self.assertTrue(region_boundaries(fine_ids, valid).any())
        self.assertFalse(region_boundaries(coarse_ids, valid).any())

    def test_boundary_outer_rim_needs_explicit_trusted_background(self):
        target = torch.zeros((3, 5, 6))
        target[:, :, :3] = 1.0
        prediction = torch.ones_like(target)
        valid = torch.zeros((5, 6), dtype=torch.bool)
        valid[:, :3] = True
        boundary = torch.zeros_like(valid)
        boundary[:, 2:4] = True
        unannotated = boundary_alignment_loss(prediction, target, boundary, valid)
        trusted = boundary_alignment_loss(
            prediction, target, boundary, valid, trusted_background=~valid,
        )
        self.assertEqual(float(unannotated), 0.0)
        self.assertGreater(float(trusted), 0.0)

    def test_region_contrastive_loss_supports_backward(self):
        prediction = torch.randn(3, 6, 6, requires_grad=True)
        ids = torch.zeros((6, 6), dtype=torch.long)
        ids[:, 3:] = 1
        loss = region_contrastive_loss(
            prediction, ids, torch.ones_like(ids, dtype=torch.bool), samples=24
        )
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertIsNotNone(prediction.grad)


if __name__ == "__main__":
    unittest.main()
