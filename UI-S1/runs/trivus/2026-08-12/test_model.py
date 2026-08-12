import sys
import unittest
from pathlib import Path

import torch
import yaml


sys.path.insert(0, str(Path(__file__).resolve().parent))

from trivus_model import (
    MAX_CANDIDATES, TriVUSBatch, TriVUSSetRanker, permute_batch,
    restore_candidate_order, trivus_loss,
)


def make_batch(counts=(3, 12), input_dim=17):
    torch.manual_seed(20260812)
    rows = len(counts)
    features = torch.randn(rows, MAX_CANDIDATES, input_dim)
    mask = torch.zeros(rows, MAX_CANDIDATES, dtype=torch.bool)
    targets = torch.zeros(rows, MAX_CANDIDATES + 1)
    fallback = torch.zeros(rows, dtype=torch.long)
    for row, count in enumerate(counts):
        mask[row, :count] = True
        features[row, count:] = 0
        targets[row, min(1, count - 1)] = 1
    return TriVUSBatch(
        features=features,
        candidate_mask=mask,
        fallback_indices=fallback,
        target_distribution=targets,
        fallback_correct=torch.zeros(rows),
        weights=torch.ones(rows),
    )


class TriVUSModelTest(unittest.TestCase):
    def test_frozen_model_config(self):
        config = yaml.safe_load((Path(__file__).resolve().parent / "configs/model_prereg.yaml").read_text())
        self.assertEqual(config["status"], "FROZEN_BEFORE_R0_R1_AND_TRIVUS_RESULTS")
        self.assertEqual(config["candidate_contract"]["valid_counts"], [3, 12])
        self.assertEqual(config["loss"]["fallback_correct_bce"], 0.5)

    def test_padding_receives_no_probability(self):
        batch = make_batch()
        model = TriVUSSetRanker(batch.features.shape[-1], dropout=0).eval()
        with torch.no_grad():
            utility, _ = model(batch.features, batch.candidate_mask, batch.fallback_indices)
            probabilities = torch.softmax(utility, dim=-1)
        self.assertTrue(torch.all(probabilities[:, :MAX_CANDIDATES][~batch.candidate_mask] == 0))
        self.assertTrue(torch.allclose(probabilities.sum(dim=-1), torch.ones(len(batch.features))))

    def test_exact_counts_zero_padding_and_finite_features_are_enforced(self):
        model = TriVUSSetRanker(17, dropout=0)
        invalid_count = make_batch(counts=(4,))
        with self.assertRaises(ValueError):
            model(invalid_count.features, invalid_count.candidate_mask, invalid_count.fallback_indices)
        nonzero_padding = make_batch(counts=(3,))
        nonzero_padding.features[0, 4, 0] = 1
        with self.assertRaises(ValueError):
            model(nonzero_padding.features, nonzero_padding.candidate_mask, nonzero_padding.fallback_indices)
        nonfinite = make_batch(counts=(3,))
        nonfinite.features[0, 0, 0] = float("nan")
        with self.assertRaises(ValueError):
            model(nonfinite.features, nonfinite.candidate_mask, nonfinite.fallback_indices)

    def test_variable_set_permutation_equivariance(self):
        batch = make_batch()
        model = TriVUSSetRanker(batch.features.shape[-1], dropout=0).eval()
        permutations = torch.stack([torch.randperm(MAX_CANDIDATES) for _ in range(len(batch.features))])
        changed = permute_batch(batch, permutations)
        self.assertTrue(torch.equal(
            batch.candidate_mask,
            restore_candidate_order(changed.candidate_mask, permutations),
        ))
        self.assertTrue(torch.allclose(
            batch.target_distribution[:, :MAX_CANDIDATES],
            restore_candidate_order(changed.target_distribution[:, :MAX_CANDIDATES], permutations),
        ))
        self.assertTrue(torch.equal(
            batch.target_distribution[:, -1], changed.target_distribution[:, -1]
        ))
        with torch.no_grad():
            original, original_fallback = model(
                batch.features, batch.candidate_mask, batch.fallback_indices
            )
            moved, moved_fallback = model(
                changed.features, changed.candidate_mask, changed.fallback_indices
            )
        self.assertTrue(torch.allclose(
            original[:, :MAX_CANDIDATES],
            restore_candidate_order(moved[:, :MAX_CANDIDATES], permutations),
            atol=1e-5, rtol=1e-5,
        ))
        self.assertTrue(torch.allclose(original[:, -1], moved[:, -1], atol=1e-5, rtol=1e-5))
        self.assertTrue(torch.allclose(original_fallback, moved_fallback, atol=1e-5, rtol=1e-5))

    def test_fallback_must_point_to_valid_candidate(self):
        batch = make_batch()
        fallback = batch.fallback_indices.clone()
        fallback[0] = 5
        model = TriVUSSetRanker(batch.features.shape[-1], dropout=0)
        with self.assertRaises(ValueError):
            model(batch.features, batch.candidate_mask, fallback)

    def test_masked_s2_loss_is_finite_and_learns(self):
        batch = make_batch(counts=(3, 3, 12, 12))
        model = TriVUSSetRanker(
            batch.features.shape[-1], width=32, heads=4, layers=1, dropout=0
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
        before = float(trivus_loss(model, batch)[0].detach())
        for _ in range(20):
            optimizer.zero_grad(set_to_none=True)
            loss, _ = trivus_loss(model, batch)
            loss.backward()
            optimizer.step()
        after = float(trivus_loss(model, batch)[0].detach())
        self.assertTrue(torch.isfinite(torch.tensor(after)))
        self.assertLess(after, before)

    def test_loss_rejects_invalid_target_contracts(self):
        model = TriVUSSetRanker(17, dropout=0)
        batch = make_batch(counts=(3,))
        bad_mass = TriVUSBatch(
            **{**batch.__dict__, "target_distribution": batch.target_distribution * 0.5}
        )
        with self.assertRaises(ValueError):
            trivus_loss(model, bad_mass)
        padded_target = batch.target_distribution.clone()
        padded_target[0] = 0
        padded_target[0, 5] = 1
        bad_padding = TriVUSBatch(**{**batch.__dict__, "target_distribution": padded_target})
        with self.assertRaises(ValueError):
            trivus_loss(model, bad_padding)
        bad_binary = TriVUSBatch(
            **{**batch.__dict__, "fallback_correct": torch.full((1,), 0.5)}
        )
        with self.assertRaises(ValueError):
            trivus_loss(model, bad_binary)


if __name__ == "__main__":
    unittest.main()