import sys
import unittest
from pathlib import Path

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parent))

from sequential_atlas import budget_curve, candidate_scores, ranked_success


class SequentialAtlasTest(unittest.TestCase):
    def test_scores_restore_display_permutation(self):
        prediction = {
            "sample_key": "row",
            "display_to_candidate": [2, 0, 1],
            "label_probabilities": [0.6, 0.3, 0.1],
        }
        self.assertTrue(np.allclose(candidate_scores(prediction, 3), [0.3, 0.1, 0.6]))

    def test_ranked_success_uses_descending_stable_score(self):
        values = ranked_success(
            [False, True, False], [0.2, 0.5, 0.3]
        )
        self.assertEqual(values.tolist(), [True, False, False])

    def test_budget_curve_reports_recovery_and_first_success(self):
        curve = budget_curve([
            np.asarray([True, False, False]),
            np.asarray([False, True, False]),
            np.asarray([False, False, False]),
        ], 3)
        self.assertAlmostEqual(curve["hit_at_k"]["1"], 1 / 3)
        self.assertAlmostEqual(curve["hit_at_k"]["2"], 2 / 3)
        self.assertEqual(curve["minimum_budget_for_90_percent_oracle_recovery"], 2)
        self.assertEqual(curve["mean_first_success_rank_given_success"], 1.5)


if __name__ == "__main__":
    unittest.main()