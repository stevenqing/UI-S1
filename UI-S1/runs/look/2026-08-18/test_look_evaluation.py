import sys
import unittest
from pathlib import Path


RUN_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(RUN_DIR))
import evaluate_look as evaluation


class LookEvaluationTest(unittest.TestCase):
    def test_weighted_auc(self):
        records = [
            {"application": "a", "inverse_probability_weight": 1, "label": 0, "score": 0.1},
            {"application": "a", "inverse_probability_weight": 1, "label": 1, "score": 0.9},
        ]
        self.assertEqual(evaluation.weighted_auc(records), 1.0)

    def test_point_inside(self):
        self.assertTrue(evaluation.point_inside([1, 1], [0, 0, 2, 2]))
        self.assertFalse(evaluation.point_inside([3, 1], [0, 0, 2, 2]))


if __name__ == "__main__":
    unittest.main()