import unittest

import numpy as np
from sklearn.metrics import roc_auc_score

from arm_b import group_pair_matrix


class GroupAUCTest(unittest.TestCase):
    def test_group_pair_matrix_matches_explicit_duplication(self):
        groups = [(0, "a"), (0, "b")]
        scores = {groups[0]: [0.8, 0.2], groups[1]: [0.6, 0.4, 0.4]}
        labels = {groups[0]: [True, False], groups[1]: [True, False, True]}
        matrix, positive, negative = group_pair_matrix(groups, scores, labels)
        counts = np.asarray([2.0, 1.0])
        calculated = float(counts @ matrix @ counts / ((counts @ positive) * (counts @ negative)))
        explicit_scores = scores[groups[0]] * 2 + scores[groups[1]]
        explicit_labels = labels[groups[0]] * 2 + labels[groups[1]]
        self.assertAlmostEqual(calculated, roc_auc_score(explicit_labels, explicit_scores))


if __name__ == "__main__":
    unittest.main()