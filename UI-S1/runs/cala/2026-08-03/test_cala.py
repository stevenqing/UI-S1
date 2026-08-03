import unittest

import numpy as np

from cala_common import SHARED_ACTIONS, UNIFORM_SEQUENCE, V_ONLY_SEQUENCE, cohen_kappa, mean_failure_kappa
from cala_static import random_sequence
from cala_adaptive import SCOUT, deterministic_order, development_statistics, feature


class CalaContractTest(unittest.TestCase):
    def test_action_universes_and_sequences(self):
        self.assertEqual(len(SHARED_ACTIONS), 36)
        self.assertEqual(len(set(SHARED_ACTIONS)), 36)
        self.assertEqual(len(V_ONLY_SEQUENCE), 16)
        self.assertEqual(UNIFORM_SEQUENCE[:3], tuple((model, 0) for model in ("GTA1-7B", "Qwen3-VL-8B-Instruct", "UI-TARS-7B-SFT")))

    def test_random_sequence_is_fold_deterministic(self):
        self.assertEqual(random_sequence(2), random_sequence(2))
        self.assertNotEqual(random_sequence(2), random_sequence(3))
        self.assertEqual(set(random_sequence(2)), set(SHARED_ACTIONS))

    def test_failure_kappa(self):
        left = np.asarray([False, False, True, True])
        self.assertAlmostEqual(cohen_kappa(left, left), 1.0)
        correct = {SHARED_ACTIONS[0]: ~left, SHARED_ACTIONS[1]: ~left}
        self.assertAlmostEqual(mean_failure_kappa(correct, SHARED_ACTIONS[:2]), 1.0)

    def test_adaptive_trajectory_is_deterministic(self):
        self.assertEqual(deterministic_order("row", 2, 1), deterministic_order("row", 2, 1))
        self.assertNotEqual(deterministic_order("row", 2, 1), deterministic_order("row", 2, 2))
        self.assertFalse(set(deterministic_order("row", 2, 1)) & set(SCOUT))


if __name__ == "__main__":
    unittest.main()