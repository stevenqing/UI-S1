import sys
import unittest
from pathlib import Path


RUN_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(RUN_DIR))

from sequential_oof_diagnostic import ranking_metrics


class SequentialOOFDiagnosticTest(unittest.TestCase):
    def test_ranking_metrics_compute_topk_and_mrr(self):
        rows = [
            {"sample_key": "a", "candidate_order": [1, 0, 2]},
            {"sample_key": "b", "candidate_order": [0, 1, 2]},
            {"sample_key": "c", "candidate_order": [2, 1, 0]},
        ]
        labels = {
            "a": [True, False, False],
            "b": [False, False, True],
            "c": [False, False, False],
        }
        result = ranking_metrics(rows, labels)
        self.assertEqual(result["top1"], 0.0)
        self.assertAlmostEqual(result["hit_at_k"]["2"], 1 / 3)
        self.assertAlmostEqual(result["oracle"], 2 / 3)
        self.assertAlmostEqual(result["mrr"], (0.5 + 1 / 3) / 3)

    def test_ranking_metrics_reject_non_permutation(self):
        with self.assertRaisesRegex(ValueError, "order"):
            ranking_metrics(
                [{"sample_key": "a", "candidate_order": [0, 0, 1]}],
                {"a": [True, False, False]},
            )


if __name__ == "__main__":
    unittest.main()