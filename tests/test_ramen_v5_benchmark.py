import contextlib
import importlib.util
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts import run_ramen_benchmark as benchmark
# Avoid accidentally importing a third-party site-packages/tests package.
_helper_spec = importlib.util.spec_from_file_location(
    "local_ramen_resume_helpers", Path(__file__).with_name("test_ramen_resume.py")
)
test_ramen_resume = importlib.util.module_from_spec(_helper_spec)
_helper_spec.loader.exec_module(test_ramen_resume)


class RamenV5BenchmarkTests(unittest.TestCase):
    def test_optional_checkpoint_cadence_unions_legacy_milestones(self):
        helper = test_ramen_resume.RamenResumeTests()
        for interval, expected in ((0, [7000, 10000]), (1000, list(range(1000, 15001, 1000)))):
            with self.subTest(interval=interval), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                scene = helper._scene(root)
                commands = []
                argv = ["benchmark", "--scene", str(scene), "--sam_checkpoint", "unused",
                        "--output_root", str(root / "out"), "--semantic_protocol", "v5",
                        "--skip_preprocess", "--checkpoint_interval", str(interval)]
                with mock.patch.object(sys, "argv", argv), mock.patch.object(
                    benchmark, "run", helper._fake_run(commands)
                ), mock.patch.object(benchmark, "detail_preprocessing_complete", return_value=True), mock.patch(
                    "scripts.run_ramen_recovery.establish_semantic_reference"
                ), mock.patch.object(benchmark, "establish_v5_teacher_files"), contextlib.redirect_stdout(io.StringIO()):
                    benchmark.main()
                for command in commands:
                    if "--checkpoint_iterations" not in command:
                        continue
                    values = []
                    for value in command[command.index("--checkpoint_iterations") + 1:]:
                        if value.startswith("--"):
                            break
                        values.append(int(value))
                    self.assertEqual(values, expected)
                saved = json.loads((root / "out/experiment_protocol.json").read_text())
                self.assertEqual(saved["checkpoint_interval"], interval)

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
            ), mock.patch.object(benchmark, "establish_v5_teacher_files"
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
            train = set((scene / "sparse/0/train.txt").read_text().splitlines())
            validation = set((scene / "sparse/0/val.txt").read_text().splitlines())
            test = set((scene / "sparse/0/test.txt").read_text().splitlines())
            expected = {path.name for path in (scene / "images_train").iterdir()}
            self.assertEqual(train, expected - validation - test)
            self.assertFalse(train & validation or train & test)
            self.assertEqual(len(train), 2)
            protocol = json.loads((root / "out" / "experiment_protocol.json").read_text())
            self.assertEqual(protocol["importance_policy"], "competitive_v1")
            self.assertEqual(protocol["background_prompts"], "table,wall")

    def test_competitive_preprocessing_arguments_only_apply_to_v5(self):
        helper = test_ramen_resume.RamenResumeTests()
        for protocol in ("v5", "legacy"):
            with self.subTest(protocol=protocol), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                scene = helper._scene(root)
                commands = []
                argv = ["benchmark", "--scene", str(scene), "--sam_checkpoint", "unused",
                        "--output_root", str(root / "out"), "--semantic_protocol", protocol, "--prepare_only"]
                with mock.patch.object(sys, "argv", argv), mock.patch.object(
                    benchmark, "run", helper._fake_run(commands)
                ), mock.patch.object(benchmark, "detail_preprocessing_complete", return_value=True), mock.patch(
                    "scripts.run_ramen_recovery.establish_semantic_reference"
                ), mock.patch.object(benchmark, "establish_v5_teacher_files"), contextlib.redirect_stdout(io.StringIO()):
                    benchmark.main()
                preprocess = commands[0]
                if protocol == "v5":
                    self.assertEqual(preprocess[preprocess.index("--importance_policy") + 1], "competitive_v1")
                    self.assertEqual(preprocess[preprocess.index("--background") + 1], benchmark.BACKGROUND)
                else:
                    self.assertNotIn("--importance_policy", preprocess)
                    self.assertNotIn("--background", preprocess)

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

    @unittest.skipUnless(importlib.util.find_spec("numpy"), "numpy needed for teacher cache tests")
    def test_v5_cache_checks_every_map_and_exact_teacher_configuration(self):
        import numpy as np

        with tempfile.TemporaryDirectory() as directory:
            scene = Path(directory)
            (scene / "semantic_maps").mkdir()
            names = ["a.jpg", "b.jpg", "val.jpg"]
            metadata = {
                "teacher_preprocessing_version": np.array(2), "hierarchy_method": np.array("containment"),
                "prototype_mode": np.array("off"), "prototype_features": np.zeros((1, 3)),
                "importance_policy": np.array("competitive_v1"),
                "pca_components": np.zeros((3, 8)), "fit_image_names": np.array(names[:2]),
                "heldout_image_names": np.array(names[2:]),
            }
            payload = {"features": np.zeros((3, 2, 4)), "detail_weight": np.ones((2, 4)),
                       "boundary": np.zeros((2, 4)), "thinness": np.zeros((2, 4)),
                       "prototype_ids": np.full((2, 4), -1), "region_ids": np.zeros((2, 4)),
                       "importance": np.ones((2, 4)), "importance_known": np.ones((2, 4)),
                       "hierarchy_prototype_ids": np.full((3, 2, 4), -1),
                       "hierarchy_region_ids": np.zeros((3, 2, 4))}
            np.savez(scene / "semantic_meta.npz", **metadata)
            for name in names:
                np.savez(scene / "semantic_maps" / f"{Path(name).stem}.npz", **payload)
            kwargs = dict(heldout_names=["val.jpg"], teacher_version=2, feature_dim=3,
                          feature_width=4, image_names=names)
            self.assertTrue(benchmark.detail_preprocessing_complete(scene, **kwargs))
            np.savez(scene / "semantic_maps/a.npz", **{k: v for k, v in payload.items() if k != "importance_known"})
            self.assertFalse(benchmark.detail_preprocessing_complete(scene, **kwargs))
            np.savez(scene / "semantic_maps/a.npz", **{**payload, "importance": np.ones((1, 4)), "importance_known": np.ones((1, 4))})
            self.assertFalse(benchmark.detail_preprocessing_complete(scene, **kwargs))
            np.savez(scene / "semantic_maps/a.npz", **payload)
            for key, wrong in (("prototype_mode", "conservative"), ("hierarchy_method", "area"),
                               ("importance_policy", "legacy"),
                               ("teacher_preprocessing_version", 1)):
                np.savez(scene / "semantic_meta.npz", **{**metadata, key: np.array(wrong)})
                self.assertFalse(benchmark.detail_preprocessing_complete(scene, **kwargs))
            np.savez(scene / "semantic_meta.npz", **metadata)
            self.assertFalse(benchmark.detail_preprocessing_complete(scene, **{**kwargs, "feature_dim": 4}))
            np.savez(scene / "semantic_maps/val.npz", **{**payload, "features": np.zeros((3, 2, 5))})
            self.assertFalse(benchmark.detail_preprocessing_complete(scene, **kwargs))
            (scene / "semantic_maps/val.npz").unlink()
            self.assertFalse(benchmark.detail_preprocessing_complete(scene, **kwargs))

    def test_map_manifest_detects_modified_missing_and_unrecorded_teacher_files(self):
        with tempfile.TemporaryDirectory() as directory:
            scene, output = Path(directory) / "scene", Path(directory) / "out"
            (scene / "semantic_maps").mkdir(parents=True)
            (scene / "importance_masks").mkdir()
            output.mkdir()
            for name in ("a", "val"):
                (scene / "semantic_maps" / f"{name}.npz").write_bytes(name.encode())
                (scene / "importance_masks" / f"{name}.png").write_bytes(name.encode())
            names = ["a.jpg", "val.jpg"]
            with self.assertRaises(RuntimeError):
                benchmark.establish_v5_teacher_files(scene, output, names, existing_weights=True)
            first = benchmark.establish_v5_teacher_files(scene, output, names)
            again = benchmark.establish_v5_teacher_files(scene, output, names, existing_weights=True)
            self.assertEqual(first, again)
            (scene / "importance_masks/val.png").write_bytes(b"changed weight")
            with self.assertRaises(RuntimeError):
                benchmark.establish_v5_teacher_files(scene, output, names, existing_weights=True)
            (scene / "importance_masks/val.png").write_bytes(b"val")
            (scene / "semantic_maps/val.npz").write_bytes(b"changed")
            with self.assertRaises(RuntimeError):
                benchmark.establish_v5_teacher_files(scene, output, names, existing_weights=True)
            (scene / "semantic_maps/val.npz").unlink()
            with self.assertRaises(RuntimeError):
                benchmark.establish_v5_teacher_files(scene, output, names, existing_weights=True)


if __name__ == "__main__":
    unittest.main()
