import unittest

import numpy as np

from scripts.evaluate_lerf_mask import render_semantic_feature_map


class SignedAffinityEvaluationTests(unittest.TestCase):
    def test_signed_affinity_is_not_clipped_and_is_alpha_normalized(self):
        try:
            import torch
        except ImportError:
            self.skipTest("Requires torch; exercised in Colab")
        features = np.array([[-.8, .4, -.2, .6]], dtype=np.float32)
        calls = []

        def renderer(camera, gaussian, pipeline, background, override_color, **kwargs):
            calls.append(kwargs)
            return {"render": override_color.T.reshape(3, 1, 1) * .5}

        result = render_semantic_feature_map(
            None, None, None, torch.zeros(3), features, torch.full((1, 1), .5),
            renderer, signed=True,
        )
        np.testing.assert_allclose(result[0, 0], features[0])
        self.assertEqual(calls, [{"clamp_output": False}, {"clamp_output": False}])

    def test_language_render_preserves_legacy_call_signature(self):
        try:
            import torch
        except ImportError:
            self.skipTest("Requires torch; exercised in Colab")

        def renderer(camera, gaussian, pipeline, background, override_color):
            return {"render": override_color.T.reshape(3, 1, 1)}

        values = np.array([[.2, .4, .6]], dtype=np.float32)
        result = render_semantic_feature_map(None, None, None, torch.zeros(3), values,
                                             torch.ones((1, 1)), renderer)
        np.testing.assert_allclose(result[0, 0], values[0])


if __name__ == "__main__":
    unittest.main()
