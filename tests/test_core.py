import unittest
import json
import tempfile
from pathlib import Path

import numpy as np
import torch

from export_web_bundle import assign_disjoint_indices, read_label_specs
from semantic.artifact import (
    apply_scale_gate, cosine_scores, decode_features, project_clip_feature, select_indices,
)
from semantic.inspection import pick_point, project_points
from utils.view_selection import select_uniform
from semantic.curriculum import cosine_ramp, curriculum_phase
from semantic.inventory import parse_inventory_config, rank_scene_inventory
from semantic.presets import estimate_minutes, preset


class ArtifactTests(unittest.TestCase):
    def test_encode_decode_geometry(self):
        encoded = np.array([[0.0, 0.5, 1.0]], dtype=np.float32)
        decoded = decode_features(encoded, [-2, 0, 3], [2, 4, 5])
        np.testing.assert_allclose(decoded, [[-2, 2, 5]])

    def test_project_score_and_select_all_matches(self):
        projected = project_clip_feature([2, 1, 0], [1, 1, 0], [[1, 0, 0], [0, 1, 0]])
        np.testing.assert_allclose(projected, [1, 0])
        scores = cosine_scores([[1, 0], [0, 1], [0.8, 0.2]], projected)
        indices = select_indices(scores, threshold=0.7)
        self.assertEqual(indices.tolist(), [0, 2])

    def test_saved_scale_gate_changes_granularity(self):
        artifact = {"scale_gate": {
            "linear.weight": torch.tensor([[2.0], [-2.0]]),
            "linear.bias": torch.tensor([0.0, 0.0]),
        }}
        encoded = np.ones((1, 2), dtype=np.float32)
        coarse = apply_scale_gate(encoded, artifact, 0)
        fine = apply_scale_gate(encoded, artifact, 2)
        self.assertGreater(fine[0, 0], coarse[0, 0])
        self.assertLess(fine[0, 1], coarse[0, 1])


class SparseViewTests(unittest.TestCase):
    def test_uniform_selection_keeps_endpoints(self):
        selected = select_uniform(list(range(10)), max_views=4)
        self.assertEqual(selected[0], 0)
        self.assertEqual(selected[-1], 9)
        self.assertEqual(len(selected), 4)

    def test_stride(self):
        self.assertEqual(select_uniform(list(range(8)), stride=3), [0, 3, 6])


class CurriculumAndInventoryTests(unittest.TestCase):
    def test_semantic_curriculum_is_rgb_then_smooth_then_joint(self):
        self.assertEqual(cosine_ramp(999, 1000, 2000), 0.0)
        self.assertAlmostEqual(cosine_ramp(2000, 1000, 2000), 0.5)
        self.assertEqual(cosine_ramp(3000, 1000, 2000), 1.0)
        self.assertEqual(curriculum_phase(500, 1000, 2000), "rgb_warmup")
        self.assertEqual(curriculum_phase(1500, 1000, 2000), "semantic_ramp")

    def test_user_or_llm_config_maps_three_tiers(self):
        tiers = parse_inventory_config({"objects": [
            {"label": "apple", "tier": "important", "aliases": ["fruit"]},
            {"label": "cup", "tier": "normal"},
            {"label": "wall", "tier": "background"},
        ]})
        self.assertEqual(tiers["important"], ["apple", "fruit"])
        self.assertEqual(tiers["background"], ["wall"])

    def test_clip_inventory_requires_cross_view_evidence(self):
        regions = np.array([[1, 0], [0.9, 0.1], [0, 1]], dtype=np.float32)
        regions /= np.linalg.norm(regions, axis=1, keepdims=True)
        candidates = np.eye(2, dtype=np.float32)
        result = rank_scene_inventory(
            regions, candidates, ["apple", "wall"], ["a", "b", "a"],
            [0.1, 0.2, 0.8], threshold=0.5, topk_per_region=1,
        )
        self.assertEqual(result[0]["label"], "apple")
        self.assertEqual(result[0]["view_count"], 2)

    def test_presets_trade_runtime_for_capacity(self):
        self.assertLess(preset("quick")["scene_iterations"], preset("quality")["scene_iterations"])
        quick = estimate_minutes("quick", 30, semantics=False)
        quality = estimate_minutes("quality", 30, semantics=True)
        self.assertLess(quick[1], quality[0])


class InspectionTests(unittest.TestCase):
    def setUp(self):
        self.camera = {
            "position": [0, 0, 0], "rotation": np.eye(3).tolist(),
            "fx": 100, "fy": 100, "width": 200, "height": 100,
        }

    def test_projection(self):
        pixels, depth = project_points([[0, 0, 2], [1, 0, 2]], self.camera)
        np.testing.assert_allclose(pixels, [[100, 50], [150, 50]])
        np.testing.assert_allclose(depth, [2, 2])

    def test_pick_nearest_visible_point(self):
        index = pick_point([[0, 0, 2], [0, 0, 5], [2, 0, -1]], self.camera, 100, 50)
        self.assertEqual(index, 0)


class WebBundleTests(unittest.TestCase):
    def test_label_specs_support_bilingual_metadata(self):
        specs = read_label_specs("apple,cup", "")
        self.assertEqual([item["label"] for item in specs], ["apple", "cup"])
        self.assertEqual([item["importance"] for item in specs], ["normal", "normal"])

    def test_label_specs_support_importance_tiers(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "labels.json"
            path.write_text(json.dumps({"objects": [
                {"label": "ramen", "label_zh": "拉面", "tier": "important"},
                {"label": "table", "importance": "background"},
            ]}), encoding="utf-8")
            specs = read_label_specs("", str(path))
        self.assertEqual([item["importance"] for item in specs], ["important", "background"])

    def test_label_specs_reject_unknown_importance_tier(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "labels.json"
            path.write_text(json.dumps([{"label": "ramen", "tier": "urgent"}]), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "invalid importance"):
                read_label_specs("", str(path))

    def test_assignment_is_disjoint_and_respects_threshold(self):
        groups = assign_disjoint_indices(
            [[0.9, 0.2], [0.3, 0.8], [0.1, 0.2], [0.7, 0.6]], threshold=0.5
        )
        self.assertEqual(groups[0].tolist(), [0, 3])
        self.assertEqual(groups[1].tolist(), [1])
        self.assertTrue(set(groups[0]).isdisjoint(set(groups[1])))


if __name__ == "__main__":
    unittest.main()
