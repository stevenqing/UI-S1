import unittest

from sklearn.metrics import roc_auc_score

from arm0_ceil_units import unclustered_bootstrap_auc


class ArmZeroTest(unittest.TestCase):
    def test_unclustered_point_matches_sklearn_with_ties(self):
        scores = [0.1, 0.4, 0.4, 0.9, 0.8]
        labels = [False, True, False, True, False]
        report = unclustered_bootstrap_auc(scores, labels, 100, 7, batch_size=10)
        self.assertAlmostEqual(report["point"], roc_auc_score(labels, scores))
        self.assertLessEqual(report["ci_99"][0], report["point"])
        self.assertGreaterEqual(report["ci_99"][1], report["point"])


if __name__ == "__main__":
    unittest.main()