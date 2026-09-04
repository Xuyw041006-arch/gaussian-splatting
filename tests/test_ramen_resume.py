import json
import tempfile
import unittest
from pathlib import Path

from scripts.run_ramen_benchmark import (
    latest_checkpoint,
    semantic_training_complete,
)


class RamenResumeTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
