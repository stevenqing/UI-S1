import sys
import unittest
from pathlib import Path

import numpy as np


RUN_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(RUN_DIR))
import evaluate_arm_a as evaluation


class FinalEvaluationTest(unittest.TestCase):
    def test_dependence_identical_slots(self):
        errors = np.asarray([[0, 0], [1, 1], [0, 0], [1, 1]], dtype=float)
        result = evaluation.dependence_endpoint(errors, np.ones(4))
        self.assertEqual(result["valid_pair_count"], 1)
        self.assertAlmostEqual(result["neff_zero"], 1.0)
        self.assertAlmostEqual(result["neff_mean"], 1.0)

    def test_dependence_constant_slots(self):
        errors = np.zeros((4, 2), dtype=float)
        result = evaluation.dependence_endpoint(errors, np.ones(4))
        self.assertEqual(result["constant_slot_count"], 2)
        self.assertIsNone(result["neff_mean"])

    def test_dependence_label_boundaries(self):
        self.assertEqual(evaluation.dependence_label([0.11, 0.20]), "MATERIAL_DEPENDENCE_MISMATCH")
        self.assertEqual(evaluation.dependence_label([0.01, 0.09]), "APPROXIMATELY_MATCHED")
        self.assertEqual(evaluation.dependence_label([0.05, 0.15]), "DEPENDENCE_MATCH_INDETERMINATE")


if __name__ == "__main__":
    unittest.main()