import unittest

from arm3_headroom import coupling


class ArmThreeTest(unittest.TestCase):
    def test_independent_errors_have_zero_kappa(self):
        value = coupling(0.3, 0.2, 0.0)
        self.assertAlmostEqual(value["both_wrong"], 0.06)
        self.assertAlmostEqual(value["achieved_kappa"], 0.0)

    def test_infeasible_kappa_is_projected(self):
        value = coupling(0.1, 0.1, -0.9)
        self.assertTrue(value["projected_to_feasible"])
        self.assertGreaterEqual(value["both_wrong"], 0.0)


if __name__ == "__main__":
    unittest.main()