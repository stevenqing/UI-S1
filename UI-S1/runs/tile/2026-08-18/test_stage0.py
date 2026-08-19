import sys
import unittest
from pathlib import Path


RUN_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(RUN_DIR))
import tile_common as common
import stage0


class TileStage0Test(unittest.TestCase):
    def test_half_open_center(self):
        self.assertTrue(common.contains_center([0, 0, 10, 10], [8, 8, 10, 10]))
        self.assertFalse(common.contains_center([0, 0, 5, 5], [9, 9, 11, 11]))

    def test_curve_repeated_boundaries_and_fallback(self):
        pairs = [{"row_id": f"r{i}", "eccentricity": 0.1, "correct": i % 2 == 0} for i in range(10)]
        curve = common.fit_curve(pairs, {f"r{i}": i for i in range(10)})
        probability, scale, index = common.curve_probability(curve, 0, 0.1)
        self.assertEqual(index, 9)
        self.assertIn(scale, {"small", "large"})
        self.assertTrue(0 <= probability <= 1)

    def test_ledger_identity(self):
        for correct in (False, True):
            row = common.ledger_record(0.7, correct)
            self.assertAlmostEqual(row["expected_repair"] - row["expected_damage"], row["expected_net"])

    def test_select_n_tie_smaller(self):
        self.assertEqual(common.select_n({4: 0.1, 5: 0.1, 6: 0.0}), 4)

    def test_contextual_summary_uses_prefixed_fields(self):
        rows = [{"C_uni_expected_repair": 0.2, "C_uni_expected_damage": 0.7, "hard_below_0_5": True}]
        value = stage0.summarize_rows(rows, "C_uni_")
        self.assertAlmostEqual(value["expected_repair"], 0.2)
        self.assertAlmostEqual(value["expected_damage"], 0.7)
        self.assertAlmostEqual(value["expected_net"], -0.5)


if __name__ == "__main__":
    unittest.main()
