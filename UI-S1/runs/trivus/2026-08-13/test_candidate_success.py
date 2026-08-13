import sys
import unittest
from pathlib import Path

import torch


sys.path.insert(0, str(Path(__file__).resolve().parent))

from candidate_success import (
    CandidateSuccessHead, candidate_success_loss, select_candidate,
)


class CandidateSuccessTest(unittest.TestCase):
    def test_loss_uses_all_valid_candidates_and_ignores_padding(self):
        logits = torch.tensor([
            [2.0, -2.0, 0.0, 100.0],
            [-2.0, 2.0, 2.0, -100.0],
        ], requires_grad=True)
        labels = torch.tensor([
            [True, False, False, False],
            [False, True, True, True],
        ])
        mask = torch.tensor([
            [True, True, True, False],
            [True, True, True, False],
        ])
        loss, report = candidate_success_loss(logits, labels, mask)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(report["pairwise_active_rows"], 2)
        self.assertEqual(logits.grad[:, 3].tolist(), [0.0, 0.0])

    def test_all_positive_and_all_negative_rows_still_train_bce(self):
        logits = torch.zeros((2, 3), requires_grad=True)
        labels = torch.tensor([
            [True, True, True],
            [False, False, False],
        ])
        mask = torch.ones((2, 3), dtype=torch.bool)
        loss, report = candidate_success_loss(logits, labels, mask)
        loss.backward()
        self.assertEqual(report["pairwise_active_rows"], 0)
        self.assertTrue(torch.any(logits.grad != 0))

    def test_row_normalization_equalizes_candidate_counts(self):
        logits = torch.zeros((2, 4))
        labels = torch.zeros((2, 4), dtype=torch.bool)
        mask = torch.tensor([
            [True, False, False, False],
            [True, True, True, True],
        ])
        loss, _ = candidate_success_loss(logits, labels, mask, pairwise_weight=0)
        self.assertAlmostEqual(float(loss), float(torch.log(torch.tensor(2.0))), places=6)

    def test_head_and_selection_respect_mask(self):
        torch.manual_seed(9)
        head = CandidateSuccessHead(8)
        representations = torch.randn(2, 4, 8)
        mask = torch.tensor([
            [True, True, False, False],
            [True, True, True, False],
        ])
        logits = head(representations, mask)
        selected = select_candidate(logits, mask)
        self.assertTrue(mask[torch.arange(2), selected].all())


if __name__ == "__main__":
    unittest.main()