import unittest

from g1_lineage_gate import adjudicate_gate, cohen_kappa


GATE = {
    "pass_requires": {
        "minimum_pass_at_3": 0.78,
        "at_least_one_pairwise_kappa_below": 0.45,
    },
    "cancel_g2_if_pass_at_3_below": 0.75,
    "lineage_concentrated_if_all_pairwise_kappa_at_least": 0.55,
    "default_g2_threshold": 0.731,
    "concentrated_g2_effective_threshold": 0.704,
    "stretch_threshold": 0.731,
}


class ScaleUpGateTest(unittest.TestCase):
    def test_kappa_identical_and_opposed(self):
        self.assertAlmostEqual(cohen_kappa([0, 0, 1, 1], [0, 0, 1, 1]), 1.0)
        self.assertAlmostEqual(cohen_kappa([0, 0, 1, 1], [1, 1, 0, 0]), -1.0)

    def test_gate_standard_pass(self):
        result = adjudicate_gate(0.80, [0.40, 0.58, 0.62], GATE)
        self.assertTrue(result["G1_pass"])
        self.assertFalse(result["G2_cancelled"])
        self.assertEqual(result["G2_effective_threshold"], 0.731)

    def test_gate_concentrated_relaxes_threshold(self):
        result = adjudicate_gate(0.79, [0.60, 0.61, 0.62], GATE)
        self.assertFalse(result["G1_pass"])
        self.assertTrue(result["lineage_concentrated"])
        self.assertEqual(result["G2_effective_threshold"], 0.704)

    def test_gate_common_failure_cancels(self):
        result = adjudicate_gate(0.74, [0.40, 0.50, 0.60], GATE)
        self.assertTrue(result["G2_cancelled"])
        self.assertIsNone(result["G2_effective_threshold"])

    def test_gate_marginal_runs_without_pass(self):
        result = adjudicate_gate(0.76, [0.40, 0.50, 0.60], GATE)
        self.assertFalse(result["G1_pass"])
        self.assertFalse(result["G2_cancelled"])
        self.assertEqual(result["G2_action"], "RUN_G2_MARGINAL_GATE_STANDARD_THRESHOLD")


if __name__ == "__main__":
    unittest.main()