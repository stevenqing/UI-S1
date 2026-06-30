import unittest
from types import SimpleNamespace

import numpy as np

from gui360_long_horizon.data.difficulty import (
    DifficultyProxyInvalid,
    calibrated_difficulty,
    fit_buckets,
    structural_difficulty,
    validity_gate,
)


class DifficultyTests(unittest.TestCase):
    def test_structural_difficulty_counts_controls_and_tree(self):
        step = SimpleNamespace(
            control_infos={"uia_controls_info": [{"a": 1}, {"b": 2}]},
            ui_tree={"children": [{"children": [{}]}, {}]},
            image_area=100,
        )
        feats = structural_difficulty(step)
        self.assertEqual(feats["n_controls"], 2)
        self.assertEqual(feats["tree_size"], 4)
        self.assertEqual(feats["tree_depth"], 3)
        self.assertAlmostEqual(feats["density"], 0.02)

    def test_calibrated_difficulty_uses_held_out_model_probability(self):
        step = SimpleNamespace(gt_rect=(0, 0, 1, 1), gt_xy=(0.5, 0.5))
        strong = SimpleNamespace(prob_correct=lambda _: 0.8)
        self.assertAlmostEqual(calibrated_difficulty(step, strong), 0.2)
        bad = SimpleNamespace(is_model_under_test=True, prob_correct=lambda _: 0.8)
        with self.assertRaises(ValueError):
            calibrated_difficulty(step, bad)

    def test_buckets_are_monotone_on_synthetic_signal(self):
        scores = np.linspace(0, 1, 100)
        bucketizer = fit_buckets(scores, k=10)
        buckets = bucketizer.transform(scores)
        correct = scores < 0.5
        means = [correct[buckets == idx].mean() for idx in range(10)]
        self.assertTrue(all(left >= right for left, right in zip(means, means[1:])))

    def test_validity_gate_passes_real_signal_and_raises_on_shuffle(self):
        scores = np.linspace(0, 1, 200)
        correct = scores < 0.5
        result = validity_gate(scores, correct)
        self.assertTrue(result.passed)
        rng = np.random.default_rng(7)
        shuffled = rng.permutation(correct)
        with self.assertRaises(DifficultyProxyInvalid):
            validity_gate(scores, shuffled)


if __name__ == "__main__":
    unittest.main()
