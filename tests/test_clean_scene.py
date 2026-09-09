import importlib.util
import json
from pathlib import Path
import pickle
import tempfile
import types
import unittest
from unittest import mock

import numpy as np
from plyfile import PlyData, PlyElement

from scripts.clean_scene import (
    camera_support, clean_model, filter_semantics, make_plan, read_config,
)


def vertices(count=100):
    fields = [(name, "f4") for name in (
        "x", "y", "z", "opacity", "scale_0", "scale_1", "scale_2",
        "f_dc_0", "f_dc_1", "f_dc_2", "rot_0", "rot_1", "rot_2", "rot_3",
    )]
    data = np.zeros(count, dtype=fields)
    xyz = np.random.default_rng(4).uniform(-1, 1, (count, 3))
    for axis, name in enumerate(("x", "y", "z")):
        data[name] = xyz[:, axis]
    for axis in range(3):
        data[f"scale_{axis}"] = np.log(0.002)
    data["opacity"] = 5
    data["rot_0"] = 1
    data["f_dc_0"] = np.arange(count)
    for index in (0, 2):
        for axis in range(3):
            data[f"scale_{axis}"][index] = np.log(0.2)
    data["scale_0"][1] = np.log(1.0)  # A long thin splat must survive.
    data["opacity"][:4] = -5
    data["x"][3] = 100
    return data


def semantic_artifact(count):
    return {
        "features": np.arange(count * 3).reshape(count, 3),
        "importance_score": np.where(np.arange(count) == 2, 0.9, 0.2),
        "instance_ids": np.arange(count),
        # Deliberately the same first dimension as point count: this is global.
        "pca_components": np.ones((count, 6)),
        "pca_mean": np.arange(6), "scale_gate": {"test": np.ones((count, 1))},
    }


class CleanSceneTests(unittest.TestCase):
    def test_blobs_floaters_removed_but_thin_and_important_points_kept(self):
        data = vertices()
        importance = semantic_artifact(len(data))["importance_score"]
        keep, reasons, plan = make_plan(data, importance)
        self.assertEqual(np.flatnonzero(~keep).tolist(), [0, 3])
        self.assertTrue(keep[1])
        self.assertTrue(keep[2])
        self.assertEqual(reasons[0], 1)
        self.assertEqual(reasons[3], 2)
        self.assertEqual(plan["removed_gaussians"], 2)

    def test_hard_cap_never_removes_more_than_ten_percent(self):
        data = vertices()
        for axis in range(3):
            data[f"scale_{axis}"][:30] = np.log(0.2)
        data["opacity"][:30] = -5
        keep, _, plan = make_plan(data)
        self.assertLessEqual((~keep).sum(), 10)
        self.assertTrue(plan["cap_applied"])
        with self.assertRaises(ValueError):
            make_plan(data, max_remove_fraction=0.11)

    def test_semantic_arrays_share_identical_order_without_slicing_global_state(self):
        artifact = semantic_artifact(20)
        keep = np.ones(20, dtype=bool)
        keep[[1, 7]] = False
        result, fields = filter_semantics(artifact, keep)
        for key in ("features", "importance_score", "instance_ids"):
            np.testing.assert_array_equal(result[key], artifact[key][keep])
            self.assertIn(key, fields)
        self.assertIs(result["pca_components"], artifact["pca_components"])
        self.assertIs(result["scale_gate"], artifact["scale_gate"])
        self.assertEqual(result["num_gaussians"], 18)
        artifact["importance_score"] = np.zeros(19)
        with self.assertRaises(ValueError):
            filter_semantics(artifact, keep)

    def test_camera_support_is_geometry_only_and_keeps_well_supported_candidates(self):
        camera = {"position": [0, 0, 0], "rotation": np.eye(3).tolist(),
                  "width": 100, "height": 100, "fx": 50, "fy": 50}
        counts = camera_support(np.array([[0, 0, 1], [0, 0, -1], [3, 0, 1]]), [camera, camera])
        self.assertEqual(counts.tolist(), [2, 0, 0])

    def _model(self, root):
        model = root / "original"
        path = model / "point_cloud" / "iteration_15000" / "point_cloud.ply"
        path.parent.mkdir(parents=True)
        PlyData([PlyElement.describe(vertices(), "vertex")], text=False).write(path)
        (model / "cfg_args").write_text(f"Namespace(model_path={str(model)!r}, source_path='/data/ramen', sh_degree=0)")
        (model / "cameras.json").write_text("[]")
        (model / "exposure.json").write_text('{"frame": [[1,0,0,0],[0,1,0,0],[0,0,1,0]]}')
        return model, path

    def test_dry_run_has_no_writes_and_export_keeps_model_files_and_mapping(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model, source_ply = self._model(root)
            original = source_ply.read_bytes()
            output = root / "cleaned"
            plan = clean_model(model, output)
            self.assertTrue(plan["dry_run"])
            self.assertFalse(output.exists())
            plan = clean_model(model, output, apply=True)
            self.assertEqual(source_ply.read_bytes(), original)
            with np.load(output / "cleaning_indices.npz") as mapping:
                keep_indices = mapping["keep_indices"]
            cleaned = PlyData.read(output / "point_cloud/iteration_15000/point_cloud.ply")["vertex"].data
            np.testing.assert_array_equal(cleaned, vertices()[keep_indices])
            self.assertEqual(read_config(output / "cfg_args")["model_path"], str(output.resolve()))
            self.assertEqual((output / "exposure.json").read_bytes(), (model / "exposure.json").read_bytes())
            self.assertFalse(plan["training_resumable"])
            with self.assertRaises(FileExistsError):
                clean_model(model, output, apply=True)
            with self.assertRaises(ValueError):
                clean_model(model, model, apply=True)
            with self.assertRaises(ValueError):
                clean_model(model, model / "inside", apply=True)
            with self.assertRaises(ValueError):
                clean_model(model, root / "other", camera_support_filter=True)

    def test_export_semantics_uses_same_indices_as_ply(self):
        # CPU serialization double exercises export coordination without making
        # laptop tests depend on a CUDA/PyTorch install. Tensor indexing below is
        # also tested with real PyTorch when available in the Colab environment.
        fake_torch = types.SimpleNamespace(
            load=lambda path, **kwargs: pickle.loads(Path(path).read_bytes()),
            save=lambda data, path: Path(path).write_bytes(pickle.dumps(data)),
            is_tensor=lambda _value: False,
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model, _ = self._model(root)
            semantic_path = model / "semantic/iteration_15000/semantic_features.pt"
            semantic_path.parent.mkdir(parents=True)
            artifact = semantic_artifact(100)
            semantic_path.write_bytes(pickle.dumps(artifact))
            with mock.patch.dict("sys.modules", {"torch": fake_torch}):
                clean_model(model, root / "cleaned", apply=True)
            with np.load(root / "cleaned/cleaning_indices.npz") as mapping:
                indices = mapping["keep_indices"]
            cleaned = pickle.loads((root / "cleaned/semantic/iteration_15000/semantic_features.pt").read_bytes())
            for key in ("features", "importance_score", "instance_ids"):
                np.testing.assert_array_equal(cleaned[key], artifact[key][indices])
            self.assertEqual(len(indices), 98)

    @unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch is available on Colab")
    def test_real_torch_feature_and_importance_alignment(self):
        import torch
        artifact = {key: torch.from_numpy(value) if isinstance(value, np.ndarray) else value
                    for key, value in semantic_artifact(20).items()}
        keep = np.ones(20, dtype=bool)
        keep[[0, 3]] = False
        result, _ = filter_semantics(artifact, keep)
        self.assertTrue(torch.equal(result["features"], artifact["features"][keep]))
        self.assertTrue(torch.equal(result["importance_score"], artifact["importance_score"][keep]))
        self.assertEqual(result["pca_components"].shape[0], 20)


if __name__ == "__main__":
    unittest.main()
