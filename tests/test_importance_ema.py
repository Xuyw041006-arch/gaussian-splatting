import importlib.util
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

try:
    import torch
    from semantic.joint import symmetric_importance_ema
except ModuleNotFoundError:
    torch = None


@unittest.skipIf(torch is None, "torch is optional locally; run these tests in Colab")
class ImportanceEMATests(unittest.TestCase):
    def test_rise_and_fall_have_equal_inertia(self):
        previous = torch.tensor([.5, .5])
        result = symmetric_importance_ema(previous, torch.tensor([1., 0.]))
        torch.testing.assert_close(result, torch.tensor([.55, .45]))
        torch.testing.assert_close(previous, torch.tensor([.5, .5]))
        self.assertLess(result[0].item(), .75)

    def test_old_false_positive_decays_without_lock_in(self):
        score = torch.tensor([1.])
        for _ in range(8):
            score = symmetric_importance_ema(score, torch.tensor([.5]))
        self.assertLess(score.item(), .75)

    def test_invalid_observation_preserves_score(self):
        result = symmetric_importance_ema(torch.tensor([.5]), torch.tensor([float("nan")]))
        torch.testing.assert_close(result, torch.tensor([.5]))

    def test_invalid_configuration_is_rejected(self):
        with self.assertRaises(ValueError):
            symmetric_importance_ema(torch.tensor([.5]), torch.tensor([1.]), momentum=1.1)
        with self.assertRaises(ValueError):
            symmetric_importance_ema(torch.tensor([.5]), torch.tensor([[1.]]))

    def test_v5_does_not_update_unknown_invisible_or_offscreen_gaussians(self):
        source = Path(__file__).resolve().parents[1] / "semantic/joint_trainer.py"
        spec = importlib.util.spec_from_file_location("importance_supervisor_test", source)
        module = importlib.util.module_from_spec(spec)
        with mock.patch.dict(sys.modules, {"gaussian_renderer": SimpleNamespace(render=None)}):
            spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "map.npz"
            np.savez(path, importance=np.array([[2, 1, 0]], dtype=np.uint8),
                     importance_known=np.array([[1, 0, 1]], dtype=np.uint8))
            supervisor = module.JointSemanticSupervisor.__new__(module.JointSemanticSupervisor)
            supervisor.v5 = True
            supervisor.args = SimpleNamespace(importance_ema=.9)
            supervisor.map_path = lambda _: path
            supervisor.importance_path = mock.Mock(side_effect=AssertionError("v5 must use semantic evidence, not PNG"))
            supervisor.gaussians = SimpleNamespace(
                get_xyz=torch.tensor([[-2/3, 0, 1], [0, 0, 1], [2/3, 0, 1], [3, 0, 1], [2/3, 0, 1]]),
                importance_score=torch.full((5,), .8),
                update_importance_score=mock.Mock(side_effect=AssertionError("v5 must use symmetric EMA")),
            )
            camera = SimpleNamespace(full_proj_transform=torch.eye(4))
            with mock.patch.object(torch.Tensor, "cuda", lambda value, **_: value):
                supervisor.observe_importance(camera, torch.tensor([True, True, False, True, True]))
                torch.testing.assert_close(supervisor.gaussians.importance_score,
                                           torch.tensor([.82, .8, .8, .8, .72]))
                before = supervisor.gaussians.importance_score.clone()
                supervisor.observe_importance(camera, torch.zeros(5, dtype=torch.bool))
                torch.testing.assert_close(supervisor.gaussians.importance_score, before)

    def test_v5_rejects_appearance_pooled_teacher_before_cuda_initialization(self):
        source = Path(__file__).resolve().parents[1] / "semantic/joint_trainer.py"
        spec = importlib.util.spec_from_file_location("importance_guard_test", source)
        module = importlib.util.module_from_spec(spec)
        with mock.patch.dict(sys.modules, {"gaussian_renderer": SimpleNamespace(render=None)}):
            spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "semantic_maps").mkdir()
            np.savez(root / "semantic_meta.npz", pca_components=np.zeros((3, 8)),
                     teacher_preprocessing_version=2, hierarchy_method="containment",
                     importance_policy="competitive_v1", prototype_mode="conservative")
            args = SimpleNamespace(semantic_dir="", importance_mask_dir="", semantic_protocol="v5")
            with self.assertRaisesRegex(ValueError, "V5 requires"):
                module.JointSemanticSupervisor(SimpleNamespace(source_path=str(root)), object(), object(), args)


if __name__ == "__main__":
    unittest.main()
