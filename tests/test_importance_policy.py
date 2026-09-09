import argparse
import ast
from functools import lru_cache
from pathlib import Path
import tempfile
import unittest

import numpy as np

from preprocess_semantics import (
    build_detail_supervision, competitive_importance,
    importance_policy_fingerprint, validate_importance_prompts,
)


class ImportancePolicyTests(unittest.TestCase):
    def test_background_cli_is_accepted_without_changing_legacy_default(self):
        source = Path(__file__).resolve().parents[1] / "preprocess_semantics.py"
        main = next(node for node in ast.parse(source.read_text()).body
                    if isinstance(node, ast.FunctionDef) and node.name == "main")
        statements = []
        for node in main.body:
            if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "args" for target in node.targets):
                break
            if not isinstance(node, (ast.Import, ast.ImportFrom)):
                statements.append(node)
        namespace = {"ArgumentParser": argparse.ArgumentParser}
        exec(compile(ast.Module(body=statements, type_ignores=[]), str(source), "exec"), namespace)
        parser = namespace["parser"]
        required = ["--scene", "unused", "--sam_checkpoint", "unused", "--background", "table,wall"]
        self.assertEqual(parser.parse_args(required).importance_policy, "legacy")
        args = parser.parse_args(required + ["--importance_policy", "competitive_v1"])
        self.assertEqual(args.background, "table,wall")
        self.assertEqual(args.importance_policy, "competitive_v1")

    def decide(self, scores, areas=None, absent=()):
        # Orthonormal teacher rows allow explicit, reproducible cosine scores.
        features = np.eye(len(scores), dtype=np.float32)
        text = {tier: np.asarray(scores, dtype=np.float32)[:, i][None]
                for i, tier in enumerate(("background", "normal", "important"))
                if tier not in absent}
        return competitive_importance(features, text, np.zeros((1, len(scores))),
                                      np.full(len(scores), .1) if areas is None else areas)

    def test_competing_high_matches_do_not_force_important(self):
        result = self.decide([[.1, .42, .40], [.1, .35, .50], [.5, .3, .2]])
        np.testing.assert_array_equal(result["tiers"], [1, 2, 0])
        self.assertEqual(result["reasons"][0], "normal_ambiguous")

    def test_missing_background_and_low_support_are_normal(self):
        result = self.decide([[.9, .12, .13], [.8, .2, .5]], absent=("background", "important"))
        np.testing.assert_array_equal(result["tiers"], [1, 1])

    def test_large_important_match_is_not_auto_promoted(self):
        result = self.decide([[.1, .2, .5], [.1, .2, .5]], areas=[.9, .05])
        np.testing.assert_array_equal(result["tiers"], [1, 2])
        self.assertEqual(result["reasons"][0], "normal_large_region")

    def test_generic_negative_can_reject_an_apparently_good_match(self):
        result = competitive_importance(np.eye(1), {"important": np.array([[.5]])},
                                        np.array([[.49]]), [.1])
        self.assertEqual(result["tiers"][0], 1)
        self.assertLess(result["relevancy"][0, 2], .6)

    def test_input_features_are_bitwise_unchanged(self):
        features = np.eye(3, dtype=np.float32)
        before = features.tobytes()
        competitive_importance(features, {"important": np.ones((1, 3)) * .5},
                               np.zeros((1, 3)), [.1] * 3)
        self.assertEqual(features.tobytes(), before)

    def test_invalid_features_are_unknown_not_important(self):
        result = competitive_importance(np.array([[np.nan, 0], [0, 0]]),
                                        {"important": np.ones((1, 2))}, np.zeros((1, 2)), [.1, .1])
        np.testing.assert_array_equal(result["tiers"], [1, 1])

    def test_conflicting_aliases_rejected_and_duplicates_deduplicated(self):
        with self.assertRaises(ValueError):
            validate_importance_prompts({"important": ["Egg"], "normal": [" egg "]})
        result = validate_importance_prompts({"important": ["Egg", "egg"]})
        self.assertEqual(result["important"], ["egg"])

    def test_detail_boost_keeps_semantic_tiers_unchanged(self):
        regions = np.zeros((8, 8), dtype=np.int16)
        regions[:, 4:] = 1
        tiers = np.zeros((8, 8), dtype=np.uint8)
        tiers[:, :4] = 2
        weights, boundaries, _, output = build_detail_supervision(
            regions, tiers, promote_importance=False)
        np.testing.assert_array_equal(output, tiers)
        self.assertGreater(float(weights[boundaries].mean()), 1)
        legacy = build_detail_supervision(regions, tiers)[-1]
        self.assertTrue(np.any(legacy[:, 4:] == 2))

    def test_policy_fingerprint_is_stable_and_tracks_changes(self):
        self.assertEqual(importance_policy_fingerprint({"a": 1, "b": 2}),
                         importance_policy_fingerprint({"b": 2, "a": 1}))
        self.assertNotEqual(importance_policy_fingerprint({"policy": "legacy"}),
                            importance_policy_fingerprint({"policy": "competitive_v1"}))

    def test_unknown_importance_observations_are_nan_not_normal_evidence(self):
        # Isolate the NumPy-only loader so this safety check runs without Torch/CUDA.
        source = Path(__file__).resolve().parents[1] / "semantic/joint_trainer.py"
        node = next(node for node in ast.parse(source.read_text()).body
                    if isinstance(node, ast.FunctionDef) and node.name == "load_v5_importance_observations")
        namespace = {"np": np, "lru_cache": lru_cache}
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), namespace)
        load = namespace["load_v5_importance_observations"]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "map.npz"
            np.savez(path, importance=np.array([[2, 1, 0]], dtype=np.uint8),
                     importance_known=np.array([[1, 0, 1]], dtype=np.uint8))
            result = load(str(path))
            np.testing.assert_array_equal(result[:, [0, 2]], [[2, 0]])
            self.assertTrue(np.isnan(result[0, 1]))
            load.cache_clear()
            np.savez(path, importance=np.ones((1, 3)))
            with self.assertRaisesRegex(ValueError, "importance_known"):
                load(str(path))
            np.savez(path, importance=np.ones((1, 3)), importance_known=np.ones((2, 3)))
            with self.assertRaisesRegex(ValueError, "shape"):
                load(str(path))


if __name__ == "__main__":
    unittest.main()
