import unittest

import numpy as np

from export_web_bundle import assign_disjoint_indices, read_label_specs
from semantic.artifact import cosine_scores, decode_features, project_clip_feature, select_indices
from semantic.inspection import pick_point, project_points
from utils.view_selection import select_uniform


class ArtifactTests(unittest.TestCase):
    def test_encode_decode_geometry(self):
        encoded = np.array([[0.0, 0.5, 1.0]], dtype=np.float32)
        decoded = decode_features(encoded, [-2, 0, 3], [2, 4, 5])
        np.testing.assert_allclose(decoded, [[-2, 2, 5]])

    def test_project_score_and_select_all_matches(self):
        projected = project_clip_feature([2, 1, 0], [1, 1, 0], [[1, 0, 0], [0, 1, 0]])
        np.testing.assert_allclose(projected, [1, 0])
        scores = cosine_scores([[1, 0], [0, 1], [0.8, 0.2]], projected)
        indices = select_indices(scores, threshold=0.7)
        self.assertEqual(indices.tolist(), [0, 2])


class SparseViewTests(unittest.TestCase):
    def test_uniform_selection_keeps_endpoints(self):
        selected = select_uniform(list(range(10)), max_views=4)
        self.assertEqual(selected[0], 0)
        self.assertEqual(selected[-1], 9)
        self.assertEqual(len(selected), 4)

    def test_stride(self):
        self.assertEqual(select_uniform(list(range(8)), stride=3), [0, 3, 6])


class InspectionTests(unittest.TestCase):
    def setUp(self):
        self.camera = {
            "position": [0, 0, 0], "rotation": np.eye(3).tolist(),
            "fx": 100, "fy": 100, "width": 200, "height": 100,
        }

    def test_projection(self):
        pixels, depth = project_points([[0, 0, 2], [1, 0, 2]], self.camera)
        np.testing.assert_allclose(pixels, [[100, 50], [150, 50]])
        np.testing.assert_allclose(depth, [2, 2])

    def test_pick_nearest_visible_point(self):
        index = pick_point([[0, 0, 2], [0, 0, 5], [2, 0, -1]], self.camera, 100, 50)
        self.assertEqual(index, 0)


class WebBundleTests(unittest.TestCase):
    def test_label_specs_support_bilingual_metadata(self):
        specs = read_label_specs("apple,cup", "")
        self.assertEqual([item["label"] for item in specs], ["apple", "cup"])

    def test_assignment_is_disjoint_and_respects_threshold(self):
        groups = assign_disjoint_indices(
            [[0.9, 0.2], [0.3, 0.8], [0.1, 0.2], [0.7, 0.6]], threshold=0.5
        )
        self.assertEqual(groups[0].tolist(), [0, 3])
        self.assertEqual(groups[1].tolist(), [1])
        self.assertTrue(set(groups[0]).isdisjoint(set(groups[1])))


if __name__ == "__main__":
    unittest.main()
