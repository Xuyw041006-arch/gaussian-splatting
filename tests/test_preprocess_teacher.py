import unittest

import numpy as np

from preprocess_semantics import (
    aggregate_cross_view_features, build_hierarchy_region_maps,
    masked_crop, mask_containment_parents, resize_mask, select_balanced_regions,
    supported_prototype_assignments,
)


def region(mask, quality=0.95):
    return {"segmentation": mask, "area": int(mask.sum()),
            "predicted_iou": quality, "stability_score": quality}


class PreprocessTeacherTests(unittest.TestCase):
    def test_square_crop_preserves_both_ends_of_thin_object(self):
        rgb = np.full((8, 40, 3), 120, dtype=np.uint8)
        rgb[3:5, 0] = [255, 0, 0]
        rgb[3:5, -1] = [0, 0, 255]
        mask = np.zeros((8, 40), dtype=bool)
        mask[3:5, :] = True
        crop = np.asarray(masked_crop(rgb, mask, [0, 3, 39, 1]))
        self.assertEqual(crop.shape, (40, 40, 3))
        self.assertTrue(np.any(np.all(crop == [255, 0, 0], axis=-1)))
        self.assertTrue(np.any(np.all(crop == [0, 0, 255], axis=-1)))
        single_pixel = np.zeros((8, 40), dtype=bool)
        single_pixel[3, 0] = True
        self.assertEqual(masked_crop(rgb, single_pixel, [0, 3, 0, 0]).size, (1, 1))

    def test_downsampling_preserves_thin_mask_coverage(self):
        mask = np.zeros((64, 64), dtype=bool)
        mask[1:62, 1] = True
        reduced = resize_mask(mask, (8, 8))
        self.assertEqual(int(reduced[:, 0].sum()), 8)
        self.assertEqual(int(reduced[:, 1:].sum()), 0)

    def test_mask_cap_reserves_small_high_quality_masks(self):
        masks = []
        for side in (20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10):
            mask = np.zeros((20, 20), dtype=bool)
            mask[:side, :side] = True
            masks.append(region(mask))
        thin = np.zeros((20, 20), dtype=bool)
        thin[2:18, 4] = True
        masks.append(region(thin, quality=0.98))
        selected = select_balanced_regions(masks, 6, 400)
        self.assertEqual(len(selected), 6)
        self.assertTrue(any(np.array_equal(item["segmentation"], thin) for item in selected))
        self.assertGreaterEqual(selected[0]["area"], selected[-1]["area"])

    def test_containment_hierarchy_retains_small_root_and_thin_child_at_all_scales(self):
        # Every region is <5% of the image, yet they form a real whole/part/leaf.
        whole = np.zeros((100, 100), dtype=bool)
        whole[5:25, 5:25] = True
        part = np.zeros_like(whole)
        part[10:20, 10:20] = True
        thin = np.zeros_like(whole)
        thin[11:19, 14] = True
        maps, parents = build_hierarchy_region_maps(
            [region(whole), region(part), region(thin)], (100, 100), 10000,
            return_parents=True,
        )
        self.assertEqual(parents.tolist(), [-1, 0, 1])
        self.assertEqual(maps[:, 14, 14].tolist(), [0, 1, 2])
        self.assertTrue(np.all(maps[:, whole] >= 0))

    def test_unrelated_overlap_is_not_a_parent_and_hierarchy_is_order_independent(self):
        first = np.zeros((20, 20), dtype=bool)
        first[:10, :10] = True
        second = np.zeros_like(first)
        second[5:20, 5:20] = True
        parents = mask_containment_parents(np.stack([first, second]))
        self.assertEqual(parents.tolist(), [-1, -1])
        maps = build_hierarchy_region_maps([region(first), region(second)], (20, 20), 400)
        self.assertEqual(maps[:, 7, 7].tolist(), [0, 0, 0])

    def test_heldout_view_does_not_satisfy_multiview_support(self):
        features = np.array([[1., 0.], [1., 0.], [1., 0.], [0., 1.]], dtype=np.float32)
        centers = np.eye(2, dtype=np.float32)
        labels = np.array([0, 0, 0, 1])
        fit = np.array([True, True, False, True])
        views = ["train_a", "train_a", "test_0", "train_b"]
        supported = supported_prototype_assignments(
            features, centers, labels, fit, views, min_similarity=0.9,
            min_margin=0.05, min_views=2,
        )
        self.assertFalse(supported.any())
        views[1] = "train_c"
        supported = supported_prototype_assignments(
            features, centers, labels, fit, views, min_similarity=0.9,
            min_margin=0.05, min_views=2,
        )
        self.assertEqual(supported.tolist(), [True, True, True, False])

    def test_ambiguous_prototypes_are_not_blended(self):
        features = np.array([[1., 0.]], dtype=np.float32)
        centers = np.array([[1., 0.], [0.999, 0.01]], dtype=np.float32)
        supported = supported_prototype_assignments(
            features, centers, np.array([0]), np.array([True]), min_margin=0.05,
        )
        self.assertFalse(supported[0])

    def test_disabled_prototypes_cannot_supervise_arbitrary_prototype_zero(self):
        features = np.eye(3, dtype=np.float32)
        transformed, labels, blend = aggregate_cross_view_features(features, np.ones(3), weight=0)
        np.testing.assert_array_equal(transformed, features)
        self.assertTrue(np.all(labels == -1))
        self.assertTrue(np.all(blend == 0))


if __name__ == "__main__":
    unittest.main()
