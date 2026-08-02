import unittest

from f1_paired_bootstrap import paired_bootstrap


class ClosingStatisticsTest(unittest.TestCase):
    def test_paired_bootstrap_direction_and_plus_one_p(self):
        rows = [
            {"id": f"row-{index}", "application": f"app-{index % 10}"}
            for index in range(100)
        ]
        left = {row["id"]: True for row in rows}
        right = {row["id"]: False for row in rows}
        result = paired_bootstrap(rows, left, right, resamples=1000, seed=7)
        self.assertAlmostEqual(result["point_delta"], 1.0)
        self.assertGreater(result["ci_99"][0], 0)
        self.assertEqual(result["p_one_sided_delta_le_zero"], 1 / 1001)


if __name__ == "__main__":
    unittest.main()