import sys
import unittest

from scripts.run_pipeline import build_steps, make_parser


class PipelineTests(unittest.TestCase):
    def test_joint_pipeline_calls_real_training(self):
        args = make_parser().parse_args([
            "--scene", "/tmp/example-scene",
            "--model", "/tmp/example-model",
            "--sam_checkpoint", "/tmp/sam.pth",
            "--important", "apple,cup",
            "--sparse", "--max_train_views", "8",
            "--scene_iterations", "100", "--semantic_iterations", "20",
            "--python", sys.executable,
        ])
        steps = build_steps(args)
        self.assertEqual([step.name for step in steps], ["colmap", "semantics", "joint"])
        rgb = steps[2].command
        self.assertTrue(any(value.endswith("train.py") for value in rgb))
        self.assertIn("--importance_mask_dir", rgb)
        self.assertIn("--joint_semantics", rgb)
        self.assertIn("--rgb_tier_weights", rgb)
        self.assertEqual(rgb[rgb.index("--sh_degree") + 1], "5")
        self.assertIn("--random_background", rgb)
        self.assertEqual(rgb[rgb.index("--densify_until_iter") + 1], "75")
        self.assertIn("ViT-H-14", steps[1].command)
        self.assertIn("--cross_view_prototypes", steps[1].command)

    def test_sequential_mode_remains_available_as_baseline(self):
        args = make_parser().parse_args([
            "--scene", "/tmp/example-scene", "--model", "/tmp/example-model",
            "--sam_checkpoint", "/tmp/sam.pth", "--training_mode", "sequential",
        ])
        steps = build_steps(args)
        self.assertEqual(
            [step.name for step in steps], ["colmap", "semantics", "rgb", "semantic"]
        )
        self.assertNotIn("--joint_semantics", steps[2].command)
        self.assertTrue(
            any(value.endswith("train_semantics.py") for value in steps[3].command)
        )


if __name__ == "__main__":
    unittest.main()
