"""CPU objective tests, plus a mocked-render v5 training integration test.

These do not establish reconstruction/segmentation quality. The real CUDA
rasterizer and Ramen metrics must additionally be tested on the training GPU.
"""

import ast
import importlib.util
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

try:
    import torch
    from semantic.v5 import (
        affinity_prefix_dimensions, decoded_clip_cosine_loss,
        hierarchical_affinity_loss, hierarchical_pair_targets,
        normalize_affinity_groups, normalize_rendered_features,
        region_balance_weights,
    )
    TORCH_AVAILABLE = True
except ModuleNotFoundError:
    TORCH_AVAILABLE = False

REPO = Path(__file__).resolve().parents[1]


class V5SourceCompatibilityTests(unittest.TestCase):
    def test_renderer_preserves_rgb_defaults(self):
        tree = ast.parse((REPO / "gaussian_renderer/__init__.py").read_text())
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "render")
        names = [arg.arg for arg in function.args.args][-len(function.args.defaults):]
        defaults = dict(zip(names, function.args.defaults))
        self.assertTrue(ast.literal_eval(defaults["clamp_output"]))
        self.assertFalse(ast.literal_eval(defaults["detach_geometry"]))


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is optional locally")
class V5ObjectiveTests(unittest.TestCase):
    def test_alpha_normalization_has_zero_geometry_gradient_for_constant_feature(self):
        opacity = torch.tensor(0.4, requires_grad=True)
        features = torch.tensor([0.2, 0.5, 0.8], requires_grad=True)
        output = normalize_rendered_features(opacity * features, opacity)
        output.square().sum().backward()
        self.assertLess(abs(opacity.grad.item()), 1e-6)
        torch.testing.assert_close(features.grad, 2 * features.detach())

    def test_detached_alpha_reproduces_legacy_spurious_gradient(self):
        opacity = torch.tensor(0.4, requires_grad=True)
        value = opacity * 0.6 / opacity.detach()
        value.backward()
        self.assertGreater(opacity.grad.item(), 1.0)

    def test_low_alpha_division_remains_finite(self):
        output = normalize_rendered_features(torch.zeros(3), torch.tensor(0.0))
        self.assertTrue(torch.isfinite(output).all())

    def test_region_balancing_upweights_small_masks_but_is_bounded(self):
        ids = torch.zeros(10, 10, dtype=torch.long)
        ids[0, :2] = 1
        weights = region_balance_weights(ids, torch.ones_like(ids, dtype=torch.bool))
        self.assertGreater(weights[ids == 1].mean(), weights[ids == 0].mean())
        self.assertGreaterEqual(weights.min().item(), 1 / 8)
        self.assertLessEqual(weights.max().item(), 8)

    def test_region_balancing_ignores_unknown_pixels(self):
        ids = torch.full((3, 3), -1, dtype=torch.long)
        weights = region_balance_weights(ids, torch.ones_like(ids, dtype=torch.bool))
        torch.testing.assert_close(weights, torch.ones_like(weights))

    def test_clip_cosine_restores_pca_mean_before_similarity(self):
        # In centered latent space these directions are opposite. In original
        # CLIP space a large shared mean makes them nearly aligned.
        pred = torch.tensor([[[1.0]], [[0.5]]], requires_grad=True)
        target = torch.tensor([[[0.0]], [[0.5]]])
        lower, scale = torch.tensor([-1., -1.]), torch.tensor([2., 2.])
        loss = decoded_clip_cosine_loss(
            pred, target, torch.ones(1, 1, dtype=torch.bool), torch.ones(1, 1),
            lower, scale, torch.eye(2), torch.tensor([0., 10.]),
        )
        self.assertLess(loss.item(), 0.03)
        loss.backward()
        self.assertTrue(torch.isfinite(pred.grad).all())

    def test_clip_cosine_validation_sampling_is_repeatable(self):
        torch.manual_seed(4)
        pred, target = torch.rand(4, 8, 8), torch.rand(4, 8, 8)
        args = (pred, target, torch.ones(8, 8, dtype=torch.bool), torch.ones(8, 8),
                torch.zeros(4), torch.ones(4), torch.eye(4), torch.zeros(4))
        one = decoded_clip_cosine_loss(*args, max_pixels=8, deterministic=True)
        torch.rand(100)
        two = decoded_clip_cosine_loss(*args, max_pixels=8, deterministic=True)
        torch.testing.assert_close(one, two)

    def test_nested_targets_allow_coarse_same_and_fine_different(self):
        labels = torch.tensor([[0, 0, 1], [0, 1, 2], [0, 1, 2]])
        positive, negative, conflict = hierarchical_pair_targets(labels)
        self.assertTrue(positive[0, 0, 1])
        self.assertTrue(negative[2, 0, 1])
        self.assertFalse(conflict.any())
        self.assertTrue(negative[:, 0, 2].all())

    def test_nonnested_contradictions_are_not_optimized_both_ways(self):
        labels = torch.tensor([[0, 1], [-1, -1], [0, 0]])
        positive, negative, conflict = hierarchical_pair_targets(labels)
        self.assertTrue(conflict[:, 0, 1].all())
        self.assertFalse(positive.any())
        self.assertFalse(negative.any())

    def test_unknown_mask_ids_are_not_negatives(self):
        labels = torch.full((3, 4), -1, dtype=torch.long)
        positive, negative, conflict = hierarchical_pair_targets(labels)
        self.assertFalse(positive.any() or negative.any() or conflict.any())

    def test_affinity_prefix_and_group_norms(self):
        self.assertEqual(affinity_prefix_dimensions(16), (8, 12, 16))
        raw = torch.randn(8, 16, requires_grad=True)
        feature = normalize_affinity_groups(raw)
        for start, end in ((0, 8), (8, 12), (12, 16)):
            torch.testing.assert_close(feature[:, start:end].norm(dim=-1), torch.ones(8))
        feature.sum().backward()
        self.assertTrue(torch.isfinite(raw.grad).all())

    def test_affinity_backward_only_updates_affinity_not_language(self):
        parameters = torch.nn.Parameter(torch.randn(16, 48))
        field = normalize_affinity_groups(parameters[:, 32:]).T.reshape(16, 4, 4)
        ids = torch.tensor([[0, 0, 1, 1]]).expand(4, -1)
        loss, stats = hierarchical_affinity_loss(field, ids[None].expand(3, -1, -1), torch.ones(4, 4, dtype=torch.bool), samples=16)
        loss.backward()
        self.assertGreater(stats["pairs"], 0)
        self.assertEqual(parameters.grad[:, :32].abs().sum().item(), 0)
        self.assertGreater(parameters.grad[:, 32:].abs().sum().item(), 0)


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is optional locally")
class V5RendererTests(unittest.TestCase):
    def test_feature_pass_detaches_geometry_but_retains_signed_feature_gradients(self):
        inputs = []

        class Rasterizer:
            def __init__(self, raster_settings):
                self.settings = raster_settings

            def __call__(self, **kwargs):
                inputs.append(kwargs)
                image = kwargs["colors_precomp"].mean(dim=0)[:, None, None]
                return image, torch.ones(2), torch.zeros(1, 1, 1)

        spec = importlib.util.spec_from_file_location("_renderer_v5_test", REPO / "gaussian_renderer/__init__.py")
        module = importlib.util.module_from_spec(spec)
        stubs = {
            "diff_gaussian_rasterization": SimpleNamespace(
                GaussianRasterizationSettings=lambda **kwargs: SimpleNamespace(**kwargs),
                GaussianRasterizer=Rasterizer,
            ),
            "scene.gaussian_model": SimpleNamespace(GaussianModel=object),
        }
        with mock.patch.dict(sys.modules, stubs):
            spec.loader.exec_module(module)
        parameter = lambda *shape: torch.randn(*shape, requires_grad=True)
        gaussians = SimpleNamespace(
            get_xyz=parameter(2, 3), get_opacity=parameter(2, 1),
            get_scaling=parameter(2, 3), get_rotation=parameter(2, 4), active_sh_degree=3,
        )
        camera = SimpleNamespace(
            image_height=1, image_width=1, FoVx=1., FoVy=1.,
            world_view_transform=torch.eye(4), full_proj_transform=torch.eye(4),
            camera_center=torch.zeros(3),
        )
        pipeline = SimpleNamespace(debug=False, antialiasing=False, compute_cov3D_python=False)
        colors = torch.tensor([[-.5, .5, 1.5], [-.5, .5, 1.5]], requires_grad=True)
        original_zeros = torch.zeros_like

        def zeros_like(value, **kwargs):
            kwargs.pop("device", None)
            return original_zeros(value, **kwargs)

        with mock.patch.object(module.torch, "zeros_like", zeros_like):
            output = module.render(camera, gaussians, pipeline, torch.zeros(3), override_color=colors, detach_geometry=True, clamp_output=False)
        self.assertLess(output["render"].min().item(), 0)
        self.assertGreater(output["render"].max().item(), 1)
        for name in ("means3D", "means2D", "opacities", "scales", "rotations"):
            self.assertFalse(inputs[0][name].requires_grad)
        output["render"].sum().backward()
        self.assertTrue((colors.grad > 0).all())
        self.assertIsNone(gaussians.get_xyz.grad)


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is optional locally")
class V5TrainingIntegrationTests(unittest.TestCase):
    def setUp(self):
        spec = importlib.util.spec_from_file_location("_v5_supervisor_test", REPO / "semantic/joint_trainer.py")
        self.module = importlib.util.module_from_spec(spec)
        with mock.patch.dict(sys.modules, {"gaussian_renderer": SimpleNamespace(render=None)}):
            spec.loader.exec_module(self.module)
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.map_file = Path(self.directory.name) / "frame.npz"
        self.map_file.touch()
        self.supervisor = self.module.JointSemanticSupervisor.__new__(self.module.JointSemanticSupervisor)
        supervisor = self.supervisor
        supervisor.v5, supervisor.dimensions, supervisor.affinity_dimensions = True, 5, 4
        supervisor.args = SimpleNamespace(
            semantic_start=0, semantic_ramp_iterations=0, semantic_tier_weights=(1., 1., 1.),
            semantic_region_balance_power=.5, semantic_region_balance_cap=8.,
            semantic_clip_cosine_weight=.1, semantic_clip_cosine_every=8,
            affinity_weight=.05, affinity_every=4, affinity_samples=12,
            semantic_chunks_per_step=1, semantic_geometry_grad=False,
            semantic_min_alpha=.05, semantic_weight=.22, semantic_boundary_weight=.08,
        )
        supervisor.gaussians = SimpleNamespace(_semantic_features=torch.nn.Parameter(torch.randn(3, 9)))
        supervisor.pipeline, supervisor.background = object(), torch.zeros(3)
        supervisor.feature_min, supervisor.feature_range = torch.zeros(5), torch.ones(5)
        supervisor.pca_components, supervisor.pca_mean = torch.eye(5), torch.zeros(5)
        supervisor.map_path = lambda _: self.map_file
        supervisor.scale_gate = mock.Mock(side_effect=AssertionError("v5 must never gate CLIP features"))
        ids = torch.tensor([[0, 0, 1, 1]]).expand(3, -1)
        self.supervision = {
            "features": torch.rand(5, 3, 4), "valid": torch.ones(3, 4, dtype=torch.bool),
            "confidence": torch.ones(3, 4), "region_ids": ids,
            "importance": torch.ones(3, 4, dtype=torch.long), "detail_weight": torch.ones(3, 4),
            "hierarchy_region_ids": ids[None].expand(3, -1, -1),
        }
        self.camera = SimpleNamespace(image_height=30, image_width=40)
        self.calls = []

        def render(camera, gaussians, pipeline, background, override_color, **options):
            self.calls.append(options)
            return {"render": override_color.mean(dim=0)[:, None, None].expand(3, camera.image_height, camera.image_width)}

        self.module.render = render

    def compute(self, iteration, validation=False):
        with mock.patch.object(torch.Tensor, "cuda", lambda value, **_: value), mock.patch.object(self.module, "load_joint_map", return_value=self.supervision):
            return self.supervisor.compute(self.camera, iteration, validation)

    def test_v5_full_step_trains_both_fields_without_gating_language(self):
        result = self.compute(8)
        result["loss"].backward()
        gradient = self.supervisor.gaussians._semantic_features.grad
        self.assertGreater(gradient[:, :5].abs().sum().item(), 0)
        self.assertGreater(gradient[:, 5:].abs().sum().item(), 0)
        self.assertEqual(result["evaluated_dimensions"], 5)
        self.assertEqual((self.camera.image_height, self.camera.image_width), (30, 40))
        self.assertTrue(all(call["detach_geometry"] and not call["clamp_output"] for call in self.calls))

    def test_validation_has_fixed_language_coverage_and_no_affinity_sampling(self):
        first, second = self.compute(9, validation=True), self.compute(17, validation=True)
        self.assertEqual(first["evaluated_dimensions"], 5)
        self.assertEqual(first["affinity_stats"]["pairs"], 0)
        torch.testing.assert_close(first["data_loss"], second["data_loss"])
        torch.testing.assert_close(first["clip_cosine_loss"], second["clip_cosine_loss"])

    def test_geometry_gradient_ablation_enables_both_alpha_and_feature_passes(self):
        self.supervisor.args.semantic_geometry_grad = True
        self.compute(8)
        self.assertTrue(all(not call["detach_geometry"] for call in self.calls))

    def test_camera_size_is_restored_after_render_failure(self):
        self.module.render = mock.Mock(side_effect=RuntimeError("simulated renderer error"))
        with self.assertRaisesRegex(RuntimeError, "simulated"):
            self.compute(8)
        self.assertEqual((self.camera.image_height, self.camera.image_width), (30, 40))


if __name__ == "__main__":
    unittest.main()
