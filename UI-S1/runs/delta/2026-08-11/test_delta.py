import sys
import tempfile
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from delta_model import CHANNELS, DeltaBatch, DeltaLateFusion, channel_mask, delta_loss, permute_batch, restore_candidate_order
from delta_train import CONFIG_PATH, load_test_after_pretest, validate_config
from finalize_delta import statistically_noninferior

import yaml


def batch(rows=4, base_dim=11):
    torch.manual_seed(20260811)
    targets = torch.zeros(rows, 13)
    targets[:, 3] = 1.0
    advantages = torch.full((rows, 13), -0.2)
    advantages[:, 3] = 1.0
    return DeltaBatch(
        base_features=torch.randn(rows, 12, base_dim),
        channel_features=torch.randn(rows, 12, len(CHANNELS), 7),
        fallback_indices=torch.zeros(rows, dtype=torch.long),
        target_distribution=targets,
        fallback_correct=torch.zeros(rows),
        grpo_advantage=advantages,
        weights=torch.ones(rows),
    )


class DeltaTest(unittest.TestCase):
    def test_frozen_config_matches_code_contract(self):
        validate_config(yaml.safe_load(CONFIG_PATH.read_text()))

    def test_noninferiority_uses_ci_lower_bound(self):
        self.assertTrue(statistically_noninferior({"ci_99": [-0.0069, -0.001]}, 0.007))
        self.assertFalse(statistically_noninferior({"ci_99": [-0.0071, 0.02]}, 0.007))

    def test_outer_labels_sealed_before_pretest(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(PermissionError):
                load_test_after_pretest(2, Path(directory) / "outer-2.pretest.json")

    def test_masked_channels_have_zero_gate_mass(self):
        value = batch()
        model = DeltaLateFusion(value.base_features.shape[-1], dropout=0).eval()
        mask = channel_mask("VUS_GLOBAL")
        with torch.no_grad():
            _, _, gate = model(value.base_features, value.channel_features, value.fallback_indices, mask)
        self.assertTrue(torch.all(gate[..., ~mask] == 0))
        self.assertTrue(torch.allclose(gate.sum(dim=-1), torch.ones_like(gate[..., 0])))

    def test_masked_channel_values_cannot_change_outputs(self):
        value = batch()
        model = DeltaLateFusion(value.base_features.shape[-1], dropout=0).eval()
        mask = channel_mask("VUS_GLOBAL")
        changed_channels = value.channel_features.clone()
        changed_channels[..., ~mask, :] = changed_channels[..., ~mask, :] * 1000 + 500
        with torch.no_grad():
            original = model(value.base_features, value.channel_features, value.fallback_indices, mask)
            changed = model(value.base_features, changed_channels, value.fallback_indices, mask)
        for original_value, changed_value in zip(original, changed):
            self.assertTrue(torch.equal(original_value, changed_value))

    def test_candidate_permutation_equivariance(self):
        value = batch()
        model = DeltaLateFusion(value.base_features.shape[-1], dropout=0).eval()
        permutations = torch.stack([torch.randperm(12) for _ in range(len(value.weights))])
        changed = permute_batch(value, permutations)
        mask = channel_mask("FULL")
        with torch.no_grad():
            original, original_aux, original_gate = model(value.base_features, value.channel_features, value.fallback_indices, mask)
            moved, moved_aux, moved_gate = model(changed.base_features, changed.channel_features, changed.fallback_indices, mask)
        self.assertTrue(torch.allclose(original[:, :12], restore_candidate_order(moved[:, :12], permutations), atol=1e-5, rtol=1e-5))
        self.assertTrue(torch.allclose(original[:, 12], moved[:, 12], atol=1e-5, rtol=1e-5))
        self.assertTrue(torch.allclose(original_aux, moved_aux, atol=1e-5, rtol=1e-5))
        self.assertTrue(torch.allclose(original_gate, restore_candidate_order(moved_gate, permutations), atol=1e-5, rtol=1e-5))

    def test_full_loss_is_finite_and_learns(self):
        value = batch(rows=12)
        model = DeltaLateFusion(value.base_features.shape[-1], channel_width=16, gate_width=16, candidate_width=32, layers=1, dropout=0)
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
        permutations = torch.stack([torch.randperm(12) for _ in range(len(value.weights))])
        before = float(delta_loss(model, value, "FULL", permutations)[0].detach())
        for _ in range(20):
            optimizer.zero_grad(set_to_none=True)
            loss, _ = delta_loss(model, value, "FULL", permutations)
            loss.backward()
            optimizer.step()
        after = float(delta_loss(model, value, "FULL", permutations)[0].detach())
        self.assertTrue(torch.isfinite(torch.tensor(after)))
        self.assertLess(after, before)


if __name__ == "__main__":
    unittest.main()
