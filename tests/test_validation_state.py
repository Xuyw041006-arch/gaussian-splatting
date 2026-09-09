import unittest

from utils.validation_state import validation_decision


class ValidationStateTests(unittest.TestCase):
    def test_small_improvement_still_saves_best(self):
        selected, reset, reference = validation_decision(23.91, 23.90, 23.90, 0.02)
        self.assertTrue(selected)
        self.assertFalse(reset)
        self.assertEqual(reference, 23.90)

    def test_accumulated_improvements_reset_patience(self):
        selected, reset, reference = validation_decision(23.925, 23.91, 23.90, 0.02)
        self.assertTrue(selected)
        self.assertTrue(reset)
        self.assertEqual(reference, 23.925)

    def test_nonfinite_score_never_selects_checkpoint(self):
        for score in (float("nan"), float("inf"), -float("inf")):
            self.assertEqual(validation_decision(score, 23.9, 23.9, .02), (False, False, 23.9))
