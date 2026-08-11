import unittest

import torch

from set_ranker_model import RankerBatch, VisualLogitSetRanker, permute_batch, ranker_loss
from set_ranker_train import fallback_wrong_scores


def synthetic_batch(rows=8, features=10):
    torch.manual_seed(20260811)
    values = torch.randn(rows, 12, features)
    values[:, 4, 0] += 4.0
    targets = torch.zeros(rows, 13)
    targets[:, 4] = 1.0
    advantages = torch.full((rows, 13), -0.25)
    advantages[:, 4] = 1.0
    advantages[:, 12] = 0.0
    return RankerBatch(
        features=values,
        fallback_indices=torch.zeros(rows, dtype=torch.long),
        target_distribution=targets,
        fallback_correct=torch.zeros(rows),
        grpo_advantage=advantages,
        weights=torch.ones(rows),
    )


class SetRankerTest(unittest.TestCase):
    def test_fallback_wrong_score_direction_and_s1_bypass(self):
        logits = torch.tensor([-4.0, 0.0, 4.0])
        self.assertTrue(torch.equal(fallback_wrong_scores(logits, "S1"), torch.ones(3)))
        values = fallback_wrong_scores(logits, "S2")
        self.assertGreater(values[0].item(), values[1].item())
        self.assertGreater(values[1].item(), values[2].item())
        self.assertTrue(torch.allclose(values, fallback_wrong_scores(logits, "S3")))

    def test_eval_is_candidate_permutation_equivariant(self):
        batch = synthetic_batch(rows=3)
        model = VisualLogitSetRanker(batch.features.shape[-1], dropout=0.0).eval()
        permutations = torch.stack([torch.randperm(12) for _ in range(3)])
        permuted = permute_batch(batch, permutations)
        with torch.no_grad():
            original, original_aux = model(batch.features, batch.fallback_indices)
            changed, changed_aux = model(permuted.features, permuted.fallback_indices)
        restored = torch.gather(changed[:, :12], 1, torch.argsort(permutations, dim=1))
        self.assertTrue(torch.allclose(original[:, :12], restored, atol=1e-5, rtol=1e-5))
        self.assertTrue(torch.allclose(original[:, 12], changed[:, 12], atol=1e-5, rtol=1e-5))
        self.assertTrue(torch.allclose(original_aux, changed_aux, atol=1e-5, rtol=1e-5))

    def test_s3_learns_positive_expected_utility(self):
        batch = synthetic_batch()
        model = VisualLogitSetRanker(batch.features.shape[-1], width=32, heads=4, layers=1, dropout=0.0)
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
        with torch.no_grad():
            before = torch.softmax(model(batch.features, batch.fallback_indices)[0], dim=-1)[:, 4].mean().item()
        for _ in range(40):
            optimizer.zero_grad(set_to_none=True)
            loss, _ = ranker_loss(model, batch, "S3")
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            after = torch.softmax(model(batch.features, batch.fallback_indices)[0], dim=-1)[:, 4].mean().item()
        self.assertGreater(after, before + 0.5)

    def test_permutation_updates_fallback_and_targets(self):
        batch = synthetic_batch(rows=1)
        batch = RankerBatch(
            features=batch.features,
            fallback_indices=torch.tensor([3]),
            target_distribution=batch.target_distribution,
            fallback_correct=batch.fallback_correct,
            grpo_advantage=batch.grpo_advantage,
            weights=batch.weights,
        )
        permutation = torch.tensor([[3, 4, 0, 1, 2, 5, 6, 7, 8, 9, 10, 11]])
        changed = permute_batch(batch, permutation)
        self.assertEqual(changed.fallback_indices.item(), 0)
        self.assertEqual(torch.argmax(changed.target_distribution).item(), 1)


if __name__ == "__main__":
    unittest.main()
