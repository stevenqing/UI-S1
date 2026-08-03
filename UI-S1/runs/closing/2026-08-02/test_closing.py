import unittest

from f1_paired_bootstrap import paired_bootstrap
from k8_x2_eval import chain_candidates


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

    def test_k8_fixed_adaptive_share_global_prefix(self):
        globals_ = [
            {"point": [index, index], "region": [0, 0, 100, 100]}
            for index in range(8)
        ]
        chain = {
            "global_K8": globals_,
            "confirmation": {"point": [80, 80], "region": [0, 0, 100, 100]},
            "refinement": {"point": [90, 90], "region": [50, 50, 100, 100]},
        }
        fixed = chain_candidates(chain, "model", 0, False)
        adaptive = chain_candidates(chain, "model", 0, True)
        self.assertEqual(len(fixed), len(adaptive), 9)
        self.assertEqual(fixed[:8], adaptive[:8])
        self.assertEqual(fixed[8]["point"], [80.0, 80.0])
        self.assertEqual(adaptive[8]["point"], [90.0, 90.0])


if __name__ == "__main__":
    unittest.main()