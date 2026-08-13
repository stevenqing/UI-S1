import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent))

from adjudicate_gran import fixed_strata, spearman


class AdjudicateGranTest(unittest.TestCase):
    def test_fixed_strata_use_p_then_row_id(self):
        rows = [
            {"row_id": "b", "p_hat": 0.0, "q_max_hat": 1.0, "margin": 0},
            {"row_id": "a", "p_hat": 0.0, "q_max_hat": 1.0, "margin": 0},
            {"row_id": "c", "p_hat": 1.0, "q_max_hat": 0.0, "margin": 1},
            {"row_id": "d", "p_hat": 1.0, "q_max_hat": 0.0, "margin": 1},
        ]
        strata = fixed_strata(rows, [2, 2])
        self.assertEqual(
            [row["row_id"] for row in strata[0]["rows"]], ["a", "b"]
        )

    def test_spearman_positive(self):
        self.assertAlmostEqual(spearman([0, 1, 2], [-1, 0, 1]), 1.0)


if __name__ == "__main__":
    unittest.main()