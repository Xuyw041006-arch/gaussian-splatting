import importlib.util
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from scripts.prepare_ramen_smoke_subset import colmap_loader, prepare


@unittest.skipUnless(importlib.util.find_spec("numpy"), "numpy unavailable")
class SmokeSubsetTests(unittest.TestCase):
    def test_preserves_exact_colmap_poses_and_source_images(self):
        import numpy as np
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, output = root / "source", root / "smoke"
            for name in ("images", "images_train", "sparse/0", "test_mask/0"):
                (source / name).mkdir(parents=True)
            names = [f"frame_{index}.jpg" for index in range(8)] + [f"test_{index}.jpg" for index in range(4)]
            poses = {}
            for index, name in enumerate(names):
                (source / "images" / name).write_bytes(name.encode())
                if name.startswith("frame"):
                    (source / "images_train" / name).write_bytes(name.encode())
                poses[index] = SimpleNamespace(id=index, name=name, camera_id=1,
                                              qvec=np.array([1., 0, 0, 0]), tvec=np.array([.1234567891234567, index, 2.]))
            (source / "sparse/0/images.bin").touch()
            (source / "sparse/0/points3D.ply").write_bytes(b"retained geometry")
            camera = SimpleNamespace(id=1, model="SIMPLE_PINHOLE", width=100, height=80, params=(60., 50., 40.))
            fake = SimpleNamespace(read_extrinsics_binary=lambda _: poses, read_intrinsics_binary=lambda _: {1: camera})
            with mock.patch("scripts.prepare_ramen_smoke_subset.colmap_loader", return_value=fake):
                result = prepare(source, output, 4)
            actual = colmap_loader().read_extrinsics_text(output / "sparse/0/images.txt")
            self.assertEqual(len(actual), 8)
            for index, pose in actual.items():
                np.testing.assert_array_equal(pose.qvec, poses[index].qvec)
                np.testing.assert_array_equal(pose.tvec, poses[index].tvec)
            intrinsics = colmap_loader().read_intrinsics_text(output / "sparse/0/cameras.txt")
            np.testing.assert_array_equal(intrinsics[1].params, [60., 60., 50., 40.])
            self.assertEqual(len(list((source / "images_train").iterdir())), 8)
            self.assertTrue(result["full_source_geometry_and_poses_used"])
            self.assertFalse((output / "semantic_maps").exists())
            with self.assertRaises(ValueError):
                prepare(source, output, 4)


if __name__ == "__main__":
    unittest.main()
