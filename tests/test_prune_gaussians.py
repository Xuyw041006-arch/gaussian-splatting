import tempfile
import unittest
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement

from scripts.prune_gaussians import prune_file, select_vertices


def sample_vertices():
    dtype = [
        ("x", "f4"), ("y", "f4"), ("z", "f4"),
        ("opacity", "f4"),
        ("scale_0", "f4"), ("scale_1", "f4"), ("scale_2", "f4"),
    ]
    values = np.zeros(4, dtype=dtype)
    values["x"] = [0.0, 0.2, 2.0, 0.1]
    values["opacity"] = [4.0, -4.0, 4.0, 4.0]
    values["scale_0"] = np.log([0.01, 0.01, 0.01, 0.20])
    values["scale_1"] = values["scale_0"]
    values["scale_2"] = values["scale_0"]
    return values


class GaussianPruningTests(unittest.TestCase):
    def test_combines_scale_opacity_and_radius_filters(self):
        keep = select_vertices(
            sample_vertices(), max_scale=0.05, min_opacity=0.5, max_radius=1.0
        )
        self.assertEqual(keep.tolist(), [True, False, False, False])

    def test_prune_file_never_overwrites_source(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.ply"
            PlyData([PlyElement.describe(sample_vertices(), "vertex")], text=False).write(source)
            with self.assertRaises(ValueError):
                prune_file(source, source, max_scale=0.05)

    def test_prune_file_writes_filtered_copy(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.ply"
            output = Path(directory) / "clean" / "point_cloud.ply"
            PlyData([PlyElement.describe(sample_vertices(), "vertex")], text=False).write(source)
            summary = prune_file(
                source, output, max_scale=0.05, min_opacity=0.5, max_radius=1.0
            )
            self.assertEqual(summary["input_splats"], 4)
            self.assertEqual(summary["output_splats"], 1)
            self.assertEqual(len(PlyData.read(output)["vertex"].data), 1)


if __name__ == "__main__":
    unittest.main()
