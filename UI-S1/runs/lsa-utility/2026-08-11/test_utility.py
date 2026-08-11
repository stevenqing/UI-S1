import unittest
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from behavior_policy import cyclic_validation_fold
from utility_common import feature_indices_for_mode, no_action_indices, utility_targets


class Candidate:
    def __init__(self, success):
        self.success = success


class Row:
    def __init__(self, successes):
        self.candidates = tuple(Candidate(success) for success in successes)


class UtilityContractTest(unittest.TestCase):
    def test_raw_utility_is_relative_to_fallback(self):
        utility, target = utility_targets(Row([False, True, False]), 0, "U_RAW")
        np.testing.assert_array_equal(utility, [0, 1, 0])
        np.testing.assert_array_equal(target, [0, 1, 0])

    def test_breaking_correct_fallback_is_negative(self):
        utility, _ = utility_targets(Row([True, False, True]), 0, "U_RAW")
        np.testing.assert_array_equal(utility, [0, -1, 0])

    def test_grpo_uses_sample_standard_deviation(self):
        utility, target = utility_targets(Row([False, True, False]), 0, "U_GRPO")
        expected = (utility - utility.mean()) / (utility.std(ddof=1) + 1e-4)
        np.testing.assert_allclose(target, expected)

    def test_hybrid_is_half_raw_half_grpo(self):
        row = Row([False, True, False])
        utility, grpo = utility_targets(row, 0, "U_GRPO")
        _, hybrid = utility_targets(row, 0, "U_HYBRID")
        np.testing.assert_allclose(hybrid, 0.5 * utility + 0.5 * grpo)

    def test_nested_validation_is_inside_fit_folds(self):
        fit_folds = [0, 2, 4]
        selected = cyclic_validation_fold(1, fit_folds)
        self.assertIn(selected, fit_folds)
        self.assertNotEqual(selected, 1)

    def test_no_mvp_is_strict_subset_of_no_action(self):
        no_action = set(no_action_indices())
        no_mvp = set(feature_indices_for_mode("no_mvp"))
        self.assertTrue(no_mvp < no_action)


if __name__ == "__main__":
    unittest.main()