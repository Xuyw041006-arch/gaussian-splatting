import copy
import contextlib
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

from semantic.descriptor_bank import (
    choose_regions, cross_view_support, load_bank, model_signature, pool_view_regions,
    records_to_bank, save_bank, score_descriptor_bank, select_training_views, validate_bank,
)


def fixture_bank():
    affinity = np.array([[1, 0, 1, 1], [1, 0, 1, 1], [0, 1, -1, -1]], dtype=np.float32)
    signature = model_signature(affinity, 300, "fixture", "test", (2, 3, 4), "ply-sha")
    metadata = {"schema_version": 1, "construction": "training_view_region_alignment",
                "model_signature": signature, "source_views": ["a.jpg", "b.jpg", "c.jpg"],
                "fit_image_names": ["a.jpg", "b.jpg", "c.jpg"],
                "heldout_image_names": ["val.jpg"], "test_image_names": ["test_0.jpg"]}
    records = []
    for view, feature in enumerate(affinity):
        records.append({"clip_features": np.array([1, 0], dtype=np.float32),
                        "affinity_features": feature, "view_id": view, "region_id": 0, "level": 1,
                        "confidence": 1., "coverage": 1., "coherence": 1.,
                        "mask_pixels": 100, "sampled_pixels": 10})
    return records_to_bank(records, metadata), affinity


class DescriptorBankTests(unittest.TestCase):
    def test_view_selection_is_bounded_deterministic_and_train_only(self):
        train = [f"frame_{i:02d}.jpg" for i in range(10)]
        first = select_training_views(train, train, ["val.jpg"], ["test.jpg"], train, 3)
        self.assertEqual(len(first), 3)
        self.assertEqual(first, select_training_views(train, train, ["val.jpg"], ["test.jpg"], train[::-1], 3))
        for val, tests, cameras in (([train[0]], [], train), ([], [train[0]], train), ([], [], train[:-1])):
            with self.assertRaises(ValueError):
                select_training_views(train, train, val, tests, cameras)
        with self.assertRaises(ValueError):
            select_training_views(["a.jpg", "a.png"], ["a.jpg", "a.png"], [], [], ["a.jpg", "a.png"])

    def test_original_overlapping_masks_and_multiple_levels_are_retained(self):
        masks = np.array([[[1, 1], [1, 1]], [[1, 0], [1, 0]]], dtype=bool)
        hierarchy = np.array([[[0, 0], [0, 0]], [[1, 0], [1, 0]], [[1, 0], [1, 0]]])
        raw = {"mask_shape": [2, 2], "features": np.eye(2, dtype=np.float32), "confidences": np.ones(2),
               "packed_region_masks": np.packbits(masks.reshape(2, -1), axis=1), "hierarchy_region_maps": hierarchy}
        affinity = np.array([[[1, 0, 1, 1], [0, 1, 1, 1]], [[1, 0, 1, 1], [0, 1, 1, 1]]], dtype=np.float32)
        rows = pool_view_regions(raw, affinity, np.ones((2, 2)), 0, (2, 3, 4), 2, 4)
        broad = [row for row in rows if row["region_id"] == 0 and row["level"] == 0][0]
        np.testing.assert_allclose(broad["affinity_features"][:2], [2 ** -.5, 2 ** -.5], atol=1e-6)
        np.testing.assert_array_equal(broad["clip_features"], [1, 0])
        self.assertEqual(broad["mask_pixels"], 4)
        self.assertEqual({row["level"] for row in rows if row["region_id"] == 1}, {1, 2})
        self.assertLessEqual(len(rows), 2 * 3)
        self.assertLessEqual(max(row["sampled_pixels"] for row in rows), 4)
        self.assertEqual(pool_view_regions(raw, affinity, np.zeros((2, 2)), 0, (2, 3, 4)), [])

    def test_region_selection_caps_without_using_text_or_ground_truth(self):
        hierarchy = np.array([[[0, 1]], [[2, 3]], [[4, 5]]])
        self.assertEqual(choose_regions(hierarchy, np.ones(6), 3), [0, 2, 4])
        with self.assertRaises(ValueError):
            choose_regions(hierarchy, np.ones(3), 3)

    def test_crossview_support_uses_affinity_not_identical_clip(self):
        bank, affinity = fixture_bank()
        np.testing.assert_array_equal(cross_view_support(bank, 1), [2, 2, 1])
        result = score_descriptor_bank(bank, affinity, [[1, 0]], [[0, 1]])
        self.assertGreater(result["scores"][0, 0], .99)
        self.assertEqual(result["scores"][2, 0], 0)
        np.testing.assert_array_equal(result["support_views"][:, 0], [2, 2, 0])

    def test_same_view_duplicates_or_other_levels_cannot_supply_support(self):
        bank, _ = fixture_bank()
        bank["view_id"][1] = 0
        np.testing.assert_array_equal(cross_view_support(bank, 1), [1, 1, 1])
        bank, _ = fixture_bank()
        bank["level"][1] = 2
        np.testing.assert_array_equal(cross_view_support(bank, 1), [1, 0, 1])

    def test_chunks_queries_and_empty_matches_are_equivalent(self):
        bank, affinity = fixture_bank()
        one = score_descriptor_bank(bank, affinity, [[1, 0], [-1, 0]], [[0, 1]], chunk_size=1)
        all_rows = score_descriptor_bank(bank, affinity, [[1, 0], [-1, 0]], [[0, 1]], chunk_size=100)
        np.testing.assert_array_equal(one["scores"], all_rows["scores"])
        self.assertFalse(one["scores"][:, 1].any())
        self.assertEqual(one["queries"][1]["selected_regions"], [])
        zero = score_descriptor_bank(bank, np.zeros_like(affinity), [[1, 0]], [[0, 1]], affinity_threshold=0)
        self.assertFalse(zero["scores"].any())

    def test_query_requires_two_distinct_views_and_bounded_candidates(self):
        bank, affinity = fixture_bank()
        with self.assertRaises(ValueError):
            score_descriptor_bank(bank, affinity, [[1, 0]], [[0, 1]], min_views=1)
        result = score_descriptor_bank(bank, affinity, [[1, 0]], [[0, 1]], max_candidates=2)
        self.assertLessEqual(len(result["queries"][0]["selected_regions"]), 2)
        self.assertEqual(len(result["queries"][0]["selected_view_ids"]), 2)

    def test_bank_is_bound_to_model_and_never_overwrites(self):
        bank, _ = fixture_bank()
        signature = bank["metadata"]["model_signature"]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bank.npz"
            save_bank(path, bank)
            loaded = load_bank(path, signature)
            np.testing.assert_array_equal(loaded["clip_features"], bank["clip_features"])
            with self.assertRaises(FileExistsError):
                save_bank(path, bank)
            with self.assertRaises(ValueError):
                load_bank(path, {**signature, "point_cloud_sha256": "different-geometry"})
        changed = copy.deepcopy(bank)
        changed["metadata"]["source_views"][0] = "test_0.jpg"
        with self.assertRaises(ValueError):
            validate_bank(changed)

    def test_zero_or_invalid_affinity_fails_closed(self):
        with self.assertRaises(ValueError):
            model_signature(np.zeros((3, 16)), 100, "clip", "weights")
        with self.assertRaises(ValueError):
            model_signature(np.ones((3, 16)) * np.nan, 100, "clip", "weights")

    def test_bank_rejects_duplicate_source_view_names_and_corrupt_shapes(self):
        bank, _ = fixture_bank()
        bank["metadata"]["source_views"][1] = "a.jpg"
        with self.assertRaises(ValueError):
            validate_bank(bank)
        bank, _ = fixture_bank()
        bank["affinity_features"] = bank["affinity_features"][:, :2]
        with self.assertRaises(ValueError):
            validate_bank(bank)
        bank, _ = fixture_bank()
        bank["clip_features"][0] = 0
        with self.assertRaises(ValueError):
            validate_bank(bank)

    def test_signed_render_normalizes_alpha_and_restores_camera_without_torch(self):
        # Lightweight tensor shim tests host-side integration, NOT CUDA quality.
        from scripts.build_semantic_descriptor_bank import render_affinity_image

        class Tensor:
            def __init__(self, array):
                self.array = np.asarray(array, dtype=np.float32)
            def __getitem__(self, item):
                return Tensor(self.array[item])
            def __setitem__(self, item, value):
                self.array[item] = value.array if isinstance(value, Tensor) else value
            def __truediv__(self, other):
                return Tensor(self.array / other.array)
            def cuda(self):
                return self
            def cpu(self):
                return self
            def numpy(self):
                return self.array
            def clamp_min(self, minimum):
                return Tensor(np.maximum(self.array, minimum))
            def permute(self, *axes):
                return Tensor(self.array.transpose(axes))

        torch = SimpleNamespace(float32=np.float32, no_grad=contextlib.nullcontext,
                                zeros=lambda shape, **kwargs: Tensor(np.zeros(shape)),
                                ones=lambda shape, **kwargs: Tensor(np.ones(shape)), from_numpy=Tensor)
        calls = []
        def render(camera, gaussians, pipeline, background, override_color, **kwargs):
            calls.append(kwargs)
            self.assertEqual((camera.image_height, camera.image_width), (2, 3))
            field = np.broadcast_to(.5 * override_color.array[0, :, None, None], (3, 2, 3))
            return {"render": Tensor(field)}
        camera = SimpleNamespace(image_height=99, image_width=88)
        affinity = np.array([[-.4, 1.3, .7, -1.0]], dtype=np.float32)
        with mock.patch.dict("sys.modules", {"torch": torch, "gaussian_renderer": SimpleNamespace(render=render)}):
            image, alpha = render_affinity_image(camera, None, None, affinity, 2, 3)
        np.testing.assert_allclose(image, np.broadcast_to(affinity[0], (2, 3, 4)))
        np.testing.assert_array_equal(alpha, np.full((2, 3), .5))
        self.assertTrue(all(item == {"detach_geometry": True, "clamp_output": False} for item in calls))
        self.assertEqual((camera.image_height, camera.image_width), (99, 88))
        with mock.patch.dict("sys.modules", {"torch": torch, "gaussian_renderer": SimpleNamespace(render=mock.Mock(side_effect=RuntimeError("fixture failure")))}):
            with self.assertRaises(RuntimeError):
                render_affinity_image(camera, None, None, affinity, 2, 3)
        self.assertEqual((camera.image_height, camera.image_width), (99, 88))


if __name__ == "__main__":
    unittest.main()
