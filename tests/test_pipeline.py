import sys
import unittest

from scripts.run_pipeline import build_steps, make_parser


class PipelineTests(unittest.TestCase):
    def test_full_pipeline_calls_real_training(self):
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
        self.assertEqual([step.name for step in steps], ["colmap", "semantics", "rgb", "semantic"])
        rgb = steps[2].command
        semantic = steps[3].command
        self.assertTrue(any(value.endswith("train.py") for value in rgb))
        self.assertIn("--importance_mask_dir", rgb)
        self.assertIn("--random_background", rgb)
        self.assertTrue(any(value.endswith("train_semantics.py") for value in semantic))


if __name__ == "__main__":
    unittest.main()
