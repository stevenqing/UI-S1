import unittest
import json
import tempfile
from pathlib import Path

import numpy as np

from set_ranker_data import assign_weights, original_visual_values, visual_features
from set_ranker_train import load_test_labels_after_pretest


class SetRankerDataTest(unittest.TestCase):
    def test_outer_labels_require_fsynced_pretest_record(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            labels = root / "private_labels_fold-2.jsonl"
            labels.write_text(json.dumps({"sample_key": "key", "candidate_success": [False] * 12}) + "\n")
            pretest = root / "outer-2.pretest.json"
            with self.assertRaises(PermissionError):
                load_test_labels_after_pretest(2, pretest, label_dir=root)
            pretest.write_text(json.dumps({
                "status": "PASS_SELECTION_FROZEN_BEFORE_OUTER_LABEL_ACCESS",
                "outer_fold": 2,
                "opened_development_label_folds": [0, 1, 3, 4],
            }))
            loaded = load_test_labels_after_pretest(2, pretest, label_dir=root)
            self.assertEqual(set(loaded), {"key"})

    def test_visual_logits_map_back_to_candidate_order(self):
        permutation = [5, 2, 0, 1, 3, 4, 6, 7, 8, 9, 10, 11]
        prediction = {
            "sample_key": "example",
            "display_to_candidate": permutation,
            "label_logits": list(range(12)),
            "label_probabilities": (np.arange(1, 13) / 78).tolist(),
        }
        logits, probabilities = original_visual_values(prediction)
        self.assertEqual(logits[5], 0)
        self.assertEqual(logits[2], 1)
        self.assertAlmostEqual(probabilities[0], 3 / 78)
        features = visual_features(prediction, fallback_index=2)
        self.assertEqual(features.shape, (12, 8))
        self.assertEqual(features[2, -1], 1.0)
        self.assertEqual(features[:, -1].sum(), 1.0)

    def test_weights_equalize_benchmarks_rows_and_active_arms(self):
        benchmarks = ("mind2web",) * 5 + ("screenspot_pro",) * 2
        row_ids = ("a", "a", "a", "b", "b", "c", "d")
        arms = ("C_uni", "C_cond", "C_rand", "C_uni", "C_cond", "C_uni", "C_uni")
        active = np.asarray((True, True, False, True, False, True, True))
        weights = assign_weights(benchmarks, row_ids, arms, active)
        self.assertAlmostEqual(weights[:5].sum(), 1.0)
        self.assertAlmostEqual(weights[5:].sum(), 1.0)
        self.assertAlmostEqual(weights[0], 0.25)
        self.assertAlmostEqual(weights[1], 0.25)
        self.assertAlmostEqual(weights[3], 0.5)
        self.assertEqual(weights[2], 0.0)
        self.assertEqual(weights[4], 0.0)


if __name__ == "__main__":
    unittest.main()
