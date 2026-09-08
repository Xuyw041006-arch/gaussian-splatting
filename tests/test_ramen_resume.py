import json
import os
import tempfile
import unittest
from pathlib import Path

from scripts.run_ramen_benchmark import (
    detail_preprocessing_complete,
    estimate_completed_training_seconds,
    latest_checkpoint,
    select_validation_views,
    semantic_training_complete,
    semantic_time_budget_complete,
)


class RamenResumeTests(unittest.TestCase):
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
