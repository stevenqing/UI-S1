import sys
import unittest
from pathlib import Path

import torch


sys.path.insert(0, str(Path(__file__).resolve().parent))

from sequential_policy import FALLBACK, sequential_select


class SequentialPolicyTest(unittest.TestCase):
    def test_accepts_first_candidate_that_passes_both_gates(self):
        result = sequential_select(
            torch.tensor([[2, 0, 1]]),
            torch.tensor([[0.55, 0.85, 0.90]]),
            torch.tensor([0.60]),
            budget=3,
            minimum_delta=0.10,
            maximum_loss_risk=0.10,
        )
        self.assertEqual(result["selected_candidate"].tolist(), [0])
        self.assertEqual(result["inspected_candidates"].tolist(), [2])

    def test_falls_back_when_no_candidate_passes(self):
        result = sequential_select(
            torch.tensor([[0, 1, 2]]),
            torch.tensor([[0.55, 0.65, 0.70]]),
            torch.tensor([0.60]),
            budget=2,
            minimum_delta=0.10,
            maximum_loss_risk=0.10,
        )
        self.assertEqual(result["selected_candidate"].tolist(), [FALLBACK])
        self.assertTrue(result["used_fallback"].item())
        self.assertEqual(result["inspected_candidates"].tolist(), [2])

    def test_stops_inspection_after_accept(self):
        result = sequential_select(
            torch.tensor([[0, 1, 2], [2, 1, 0]]),
            torch.tensor([[0.95, 0.20, 0.10], [0.20, 0.90, 0.10]]),
            torch.tensor([0.50, 0.50]),
            budget=3,
            minimum_delta=0.10,
            maximum_loss_risk=0.10,
        )
        self.assertEqual(result["selected_candidate"].tolist(), [0, 1])
        self.assertEqual(result["inspected_candidates"].tolist(), [1, 2])

    def test_rejects_non_permutation_order(self):
        with self.assertRaisesRegex(ValueError, "permutation"):
            sequential_select(
                torch.tensor([[0, 0, 1]]),
                torch.tensor([[0.9, 0.8, 0.7]]),
                torch.tensor([0.5]),
                budget=2,
                minimum_delta=0.1,
                maximum_loss_risk=0.2,
            )


if __name__ == "__main__":
    unittest.main()