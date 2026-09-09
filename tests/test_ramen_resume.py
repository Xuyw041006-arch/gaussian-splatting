import json
import ast
import contextlib
import io
import math
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts import run_ramen_benchmark as benchmark

from scripts.run_ramen_benchmark import (
    detail_preprocessing_complete,
    estimate_completed_training_seconds,
    latest_checkpoint,
    select_validation_views,
    semantic_training_complete,
    semantic_time_budget_complete,
    record_stage_timing,
    timing_protocol,
)


class RamenResumeTests(unittest.TestCase):
    def _scene(self, root):
        scene = root / "scene"
        (scene / "images").mkdir(parents=True)
        (scene / "images_train").mkdir()
        (scene / "sparse" / "0").mkdir(parents=True)
        for index in range(4):
            (scene / "images" / f"test_{index}.jpg").touch()
        for index in range(6):
            (scene / "images_train" / f"frame_{index}.jpg").touch()
        return scene

    def _fake_run(self, commands):
        def run(command, _cwd):
            command = list(map(str, command))
            commands.append(command)
            if "scripts.evaluate_lerf_mask" in command:
                output = Path(command[command.index("--output") + 1])
                output.mkdir(parents=True, exist_ok=True)
                metrics = {key: 0.5 for key in (
                    "test_psnr", "test_ssim", "test_important_psnr", "test_normal_psnr",
                    "mean_iou", "mean_boundary_iou",
                )}
                metrics.update(gaussians=100, per_label_iou={
                    label: 0.5 for label in (benchmark.IMPORTANT + "," + benchmark.NORMAL).split(",")
                })
                (output / "metrics.json").write_text(json.dumps(metrics))
            return 10.0
        return run

    def test_partial_evaluation_skips_other_model_and_preserves_full_comparison(self):
        for skip, selected in (("--skip_baseline", "joint"), ("--skip_joint", "sequential")):
            with self.subTest(selected=selected), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                scene = self._scene(root)
                output = root / "output"
                output.mkdir()
                previous = '{"previous_complete_comparison": true}'
                (output / "comparison.json").write_text(previous)
                commands = []
                args = ["benchmark", "--scene", str(scene), "--sam_checkpoint", "unused",
                        "--output_root", str(output), "--skip_preprocess", "--skip_training", skip]
                with mock.patch.object(sys, "argv", args), mock.patch.object(
                    benchmark, "run", self._fake_run(commands)
                ), contextlib.redirect_stdout(io.StringIO()):
                    benchmark.main()
                self.assertEqual(len(commands), 1)
                self.assertIn(str((output / selected).resolve()), commands[0])
                summary = json.loads((output / f"comparison_{selected}.json").read_text())
                self.assertTrue(summary["partial_evaluation"])
                self.assertIn(selected, summary)
                self.assertNotIn("delta", summary)
                self.assertEqual((output / "comparison.json").read_text(), previous)

    def test_resumed_rgb_missing_history_uses_fixed_semantics_and_reports_unknown(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scene = self._scene(root)
            output = root / "output"
            baseline = output / "sequential"
            baseline.mkdir(parents=True)
            checkpoint = baseline / "chkpnt7000.pth"
            checkpoint.touch()
            (output / "training_times.json").write_text(json.dumps({
                "joint_train_seconds": 8920.0, "joint_train_seconds_recovered": True,
            }))
            commands = []
            args = ["benchmark", "--scene", str(scene), "--sam_checkpoint", "unused",
                    "--output_root", str(output), "--skip_preprocess", "--skip_joint", "--resume"]
            with mock.patch.object(sys, "argv", args), mock.patch.object(
                benchmark, "run", self._fake_run(commands)
            ), contextlib.redirect_stdout(io.StringIO()):
                benchmark.main()
            self.assertIn(str(checkpoint.resolve()), commands[0])
            semantic = next(command for command in commands if any(
                part.endswith("train_semantics.py") for part in command
            ))
            self.assertNotIn("--max_seconds", semantic)
            self.assertEqual(semantic[semantic.index("--semantic_iterations") + 1], "5000")
            timings = json.loads((output / "training_times.json").read_text())
            self.assertIsNone(timings["sequential_rgb_seconds"])
            self.assertEqual(timings["sequential_rgb_seconds_observed_seconds"], 10.0)
            self.assertEqual(timings["joint_train_seconds"], 8920.0)
            self.assertIsNone(timings["sequential_total_seconds"])
            summary = json.loads((output / "comparison_sequential.json").read_text())
            self.assertFalse(summary["protocol"]["equal_wall_clock"])
            self.assertIn("unverified_or_missing", summary["protocol"]["timing_status"])

    def test_completed_joint_without_duration_is_evaluated_without_retraining(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scene = self._scene(root)
            output = root / "output"
            for relative in ("point_cloud/iteration_15000/point_cloud.ply",
                             "semantic/iteration_15000/semantic_features.pt"):
                artifact = output / "joint" / relative
                artifact.parent.mkdir(parents=True, exist_ok=True)
                artifact.touch()
            commands = []
            args = ["benchmark", "--scene", str(scene), "--sam_checkpoint", "unused",
                    "--output_root", str(output), "--skip_preprocess", "--skip_baseline", "--resume"]
            with mock.patch.object(sys, "argv", args), mock.patch.object(
                benchmark, "run", self._fake_run(commands)
            ), contextlib.redirect_stdout(io.StringIO()):
                benchmark.main()
            self.assertEqual(len(commands), 1)
            self.assertIn("scripts.evaluate_lerf_mask", commands[0])
            summary = json.loads((output / "comparison_joint.json").read_text())
            self.assertFalse(summary["protocol"]["equal_wall_clock"])

    def test_new_joint_uses_rasterizer_supported_sh_degrees(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scene = self._scene(root)
            commands = []
            args = ["benchmark", "--scene", str(scene), "--sam_checkpoint", "unused",
                    "--output_root", str(root / "output"), "--skip_preprocess", "--skip_baseline"]
            with mock.patch.object(sys, "argv", args), mock.patch.object(
                benchmark, "run", self._fake_run(commands)
            ), contextlib.redirect_stdout(io.StringIO()):
                benchmark.main()
            training = commands[0]
            self.assertEqual(training[training.index("--sh_degree") + 1], "3")
            tier_index = training.index("--tier_sh_degrees") + 1
            self.assertEqual(training[tier_index:tier_index + 3], ["1", "2", "3"])

    def test_unsupported_joint_sh_degree_is_rejected_before_training(self):
        args = ["benchmark", "--scene", "unused", "--sam_checkpoint", "unused",
                "--output_root", "unused", "--joint_sh_degree", "5"]
        with mock.patch.object(sys, "argv", args), contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as error:
                benchmark.main()
            self.assertEqual(error.exception.code, 2)

    def test_duration_certification_requires_complete_measured_history(self):
        timings = {}
        record_stage_timing(timings, "joint_train_seconds", 100.0)
        record_stage_timing(timings, "sequential_rgb_seconds", 60.0)
        record_stage_timing(timings, "sequential_semantic_seconds", 40.0)
        self.assertTrue(timing_protocol(timings, True)["equal_wall_clock"])
        record_stage_timing(timings, "sequential_rgb_seconds", 12.0, resumed=True)
        self.assertEqual(timings["sequential_rgb_seconds"], 72.0)
        self.assertFalse(timing_protocol(timings, True)["equal_wall_clock"])
        timings["sequential_rgb_seconds_run_in_progress"] = True
        timings["sequential_rgb_seconds_interrupted_unknown"] = True
        record_stage_timing(timings, "sequential_rgb_seconds", 5.0, resumed=True)
        self.assertIsNone(timings["sequential_rgb_seconds"])

    def test_semantic_budget_keeps_checkpointed_time_across_resume(self):
        # Load only pure timing helpers: this test must also run without CUDA
        # rasterizer extensions on the laptop used to supervise Colab.
        source = Path(benchmark.__file__).resolve().parents[1] / "train_semantics.py"
        tree = ast.parse(source.read_text())
        names = {"semantic_elapsed_state", "semantic_budget_exhausted"}
        helpers = ast.Module(body=[node for node in tree.body if isinstance(
            node, ast.FunctionDef
        ) and node.name in names], type_ignores=[])
        namespace = {"math": math}
        exec(compile(helpers, str(source), "exec"), namespace)
        elapsed, complete = namespace["semantic_elapsed_state"]({
            "observed_elapsed_seconds": 80.0, "elapsed_seconds_complete": True,
        })
        self.assertTrue(complete)
        exhausted = namespace["semantic_budget_exhausted"]
        self.assertFalse(exhausted(100.0, elapsed, 19.0))
        self.assertTrue(exhausted(100.0, elapsed, 20.0))
        self.assertTrue(exhausted(80.0, elapsed, 0.0))
        self.assertFalse(exhausted(0.0, elapsed, 999.0))
        self.assertEqual(namespace["semantic_elapsed_state"]({"step": 1000}), (0.0, False))

    def test_preprocessing_cache_requires_region_hierarchy(self):
        try:
            import numpy as np
        except ImportError:
            self.skipTest("numpy is optional for this unit test")

        with tempfile.TemporaryDirectory() as directory:
            scene = Path(directory)
            maps = scene / "semantic_maps"
            maps.mkdir()
            np.savez(scene / "semantic_meta.npz", prototype_features=np.zeros((1, 2)))
            legacy = {
                "detail_weight": np.ones((2, 2)),
                "boundary": np.zeros((2, 2)),
                "thinness": np.zeros((2, 2)),
                "prototype_ids": np.zeros((2, 2)),
                "hierarchy_prototype_ids": np.zeros((3, 2, 2)),
            }
            np.savez(maps / "frame.npz", **legacy)
            self.assertFalse(detail_preprocessing_complete(scene))

            np.savez(
                maps / "frame.npz",
                **legacy,
                region_ids=np.zeros((2, 2)),
                hierarchy_region_ids=np.zeros((3, 2, 2)),
            )
            self.assertTrue(detail_preprocessing_complete(scene))

    def test_completed_training_time_can_be_recovered_from_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            start = root / "cfg_args"
            end = (
                root / "point_cloud" / "iteration_15000" / "point_cloud.ply"
            )
            end.parent.mkdir(parents=True)
            start.touch()
            end.touch()
            os.utime(start, (1000, 1000))
            os.utime(end, (1123.5, 1123.5))
            self.assertEqual(
                estimate_completed_training_seconds(root, 15000), 123.5
            )

    def test_latest_checkpoint_ignores_final_and_malformed_names(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("chkpnt7000.pth", "chkpnt15000.pth", "chkpnt30000.pth", "chkpntbad.pth"):
                (root / name).touch()
            self.assertEqual(
                latest_checkpoint(root, 30000).name, "chkpnt15000.pth"
            )

    def test_semantic_completion_marker_honors_target(self):
        with tempfile.TemporaryDirectory() as directory:
            marker = (
                Path(directory) / "semantic" / "iteration_30000"
                / "training_complete.json"
            )
            marker.parent.mkdir(parents=True)
            marker.write_text(json.dumps({"semantic_iterations": 4000}))
            self.assertFalse(semantic_training_complete(directory, 30000, 5000))
            marker.write_text(json.dumps({"semantic_iterations": 5000}))
            self.assertTrue(semantic_training_complete(directory, 30000, 5000))

    def test_time_limited_semantics_count_as_complete(self):
        with tempfile.TemporaryDirectory() as directory:
            marker = (
                Path(directory) / "semantic" / "iteration_15000"
                / "training_complete.json"
            )
            marker.parent.mkdir(parents=True)
            marker.write_text(json.dumps({
                "semantic_iterations": 6200, "stopped_by_time": True,
            }))
            self.assertTrue(
                semantic_time_budget_complete(directory, 15000, 15000)
            )

    def test_validation_views_are_uniform_and_leave_training_views(self):
        paths = [Path(f"frame_{index:03d}.jpg") for index in range(20)]
        selected = select_validation_views(paths, 5)
        self.assertEqual(len(selected), 5)
        self.assertEqual(selected[0], paths[2])
        self.assertEqual(selected[-1], paths[18])
        self.assertGreaterEqual(len(paths) - len(selected), 2)


if __name__ == "__main__":
    unittest.main()
