import sys
import unittest
from pathlib import Path


RUN_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(RUN_DIR))
import evaluate_common as evaluation


class CommonEvaluationTest(unittest.TestCase):
    def test_weighted_ratio(self):
        rows = [{"application": "a", "inverse_probability_weight": 2.0, "value": 1}, {"application": "b", "inverse_probability_weight": 1.0, "value": 0}]
        self.assertEqual(evaluation.weighted_ratio(rows, "value"), 2 / 3)

    def test_identity_is_exact(self):
        rows = [
            {"application": "a", "inverse_probability_weight": 2.0, "size_half": "common_small", "oracle_b3_correct": 1, "existing_b3_correct": 0},
            {"application": "b", "inverse_probability_weight": 1.0, "size_half": "common_large", "oracle_b3_correct": 0, "existing_b3_correct": 1},
        ]
        self.assertAlmostEqual(evaluation.identity_delta(rows), 0.0, places=15)


if __name__ == "__main__":
    unittest.main()