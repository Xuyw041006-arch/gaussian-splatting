import unittest

import numpy as np

try:
    from preprocess_semantics import (
        aggregate_cross_view_features,
        build_detail_supervision,
        build_hierarchy_region_maps,
    )
    from semantic.joint import (
        GRANULARITIES, granularity_for_step, project_tiers_to_gaussians,
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


if __name__ == "__main__":
    unittest.main()
