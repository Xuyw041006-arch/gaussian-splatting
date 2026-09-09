import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts import run_ramen_benchmark as benchmark
from tests import test_ramen_resume


class RamenV5BenchmarkTests(unittest.TestCase):
    def test_v5_uses_independent_affinity_uniform_rgb_and_identical_retrieval(self):
        helper = test_ramen_resume.RamenResumeTests()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scene = helper._scene(root)
            commands = []
            argv = ["benchmark", "--scene", str(scene), "--sam_checkpoint", "unused",
                    "--output_root", str(root / "out"), "--semantic_protocol", "v5", "--skip_preprocess"]
            with mock.patch.object(sys, "argv", argv), mock.patch.object(
                benchmark, "run", helper._fake_run(commands)
            ), mock.patch.object(benchmark, "detail_preprocessing_complete", return_value=True), mock.patch(
                "scripts.run_ramen_recovery.establish_semantic_reference"
            ), contextlib.redirect_stdout(io.StringIO()):
                benchmark.main()
            joint, rgb = [command for command in commands if any(x.endswith("/train.py") for x in command)]
            self.assertEqual(joint[joint.index("--semantic_start") + 1], "2500")
            self.assertEqual(joint[joint.index("--affinity_dimensions") + 1], "16")
            self.assertNotIn("--semantic_geometry_grad", joint)
            self.assertEqual(rgb[rgb.index("--importance_mask_dir", rgb.index("--importance_mask_dir") + 1) + 1], "")
            evaluations = [command for command in commands if "scripts.evaluate_lerf_mask" in command]
            for command in evaluations:
                self.assertEqual(command[command.index("--score_mode") + 1], "clip_relevancy")
                self.assertEqual(command[command.index("--mask_protocol") + 1], "gg_native")
            result = json.loads((root / "out" / "comparison.json").read_text())
            self.assertEqual(result["protocol"]["baseline_definition"], "uniform_RGB_then_posthoc_semantics")
            self.assertFalse(result["protocol"]["equal_wall_clock"])

    def test_v5_cannot_reuse_unrecorded_legacy_model(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "out" / "joint"
            model.mkdir(parents=True)
            (model / "chkpnt7000.pth").touch()
            argv = ["benchmark", "--scene", str(root / "scene"), "--sam_checkpoint", "unused",
                    "--output_root", str(root / "out"), "--semantic_protocol", "v5", "--resume"]
            with mock.patch.object(sys, "argv", argv), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    benchmark.main()


if __name__ == "__main__":
    unittest.main()
