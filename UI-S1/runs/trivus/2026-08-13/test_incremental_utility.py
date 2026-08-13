import sys
import unittest
from pathlib import Path

import torch


sys.path.insert(0, str(Path(__file__).resolve().parent))

from incremental_utility import (
    LOSS, TIE, WIN, IncrementalUtilityHead, apply_incremental_gate,
    incremental_labels, incremental_scores, incremental_utility_loss,
)


class IncrementalUtilityTest(unittest.TestCase):
    def test_labels_encode_win_loss_and_tie(self):
        direct = torch.tensor([True, False, True, False])
        baseline = torch.tensor([False, True, True, False])
        self.assertEqual(
            incremental_labels(direct, baseline).tolist(),
            [WIN, LOSS, TIE, TIE],
        )

    def test_scores_are_win_minus_loss_and_loss_risk(self):
        logits = torch.log(torch.tensor([
            [0.1, 0.8, 0.1],
            [0.1, 0.2, 0.7],
        ]))
        delta, loss = incremental_scores(logits)
        self.assertTrue(torch.allclose(delta, torch.tensor([0.7, -0.5])))
        self.assertTrue(torch.allclose(loss, torch.tensor([0.1, 0.7])))

    def test_gate_falls_back_when_delta_or_risk_fails(self):
        direct = torch.tensor([True, False, True])
        baseline = torch.tensor([False, True, False])
        output, override = apply_incremental_gate(
            direct,
            baseline,
            torch.tensor([0.4, 0.4, 0.1]),
            torch.tensor([0.1, 0.4, 0.1]),
            minimum_delta=0.2,
            maximum_loss_probability=0.2,
        )
        self.assertEqual(override.tolist(), [True, False, False])
        self.assertEqual(output.tolist(), [True, True, False])

    def test_head_and_weighted_loss_are_finite(self):
        torch.manual_seed(7)
        head = IncrementalUtilityHead(8)
        logits = head(torch.randn(6, 8))
        labels = torch.tensor([TIE, WIN, LOSS, TIE, WIN, LOSS])
        loss = incremental_utility_loss(
            logits, labels, torch.tensor([1.0, 2.0, 2.0, 1.0, 2.0, 2.0])
        )
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(
            parameter.grad is not None and torch.isfinite(parameter.grad).all()
            for parameter in head.parameters()
        ))


if __name__ == "__main__":
    unittest.main()