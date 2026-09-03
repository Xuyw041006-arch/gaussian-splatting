import unittest

import numpy as np

from semantic.regularization import build_neighbor_graph


class SemanticQualityTests(unittest.TestCase):
    def test_neighbor_graph_prefers_close_similar_colors(self):
        try:
            import sklearn  # noqa: F401
        except ImportError:
            self.skipTest("scikit-learn is an optional semantic dependency")
        xyz = np.array([
            [0.0, 0.0, 0.0],
            [0.1, 0.0, 0.0],
            [0.2, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ], dtype=np.float32)
        colors = np.array([
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
        ], dtype=np.float32)
        neighbors, weights = build_neighbor_graph(xyz, colors, k=2, color_sigma=0.1)
        self.assertEqual(neighbors.shape, (4, 2))
        self.assertEqual(weights.shape, (4, 2))
        red_neighbor = np.flatnonzero(neighbors[1] == 0)[0]
        blue_neighbor = np.flatnonzero(neighbors[1] == 2)[0]
        self.assertGreater(weights[1, red_neighbor], weights[1, blue_neighbor])


if __name__ == "__main__":
    unittest.main()
