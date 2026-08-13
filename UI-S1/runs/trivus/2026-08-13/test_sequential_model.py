import sys
import unittest
from pathlib import Path

import torch
import yaml


RUN_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(RUN_DIR))

from sequential_fit import (
    fit_with_checkpoint, require_real_data_optimizer_authorization, train_epoch,
)
from sequential_model import (
    SequentialCandidateVerifier, augment_verifier_features, cheap_oof_features,
)


class SequentialModelTest(unittest.TestCase):
    @staticmethod
    def batch(rows=8, dimension=115):
        mask = torch.zeros((rows, 12), dtype=torch.bool)
        mask[: rows // 2, :3] = True
        mask[rows // 2 :, :] = True
        features = torch.randn(rows, 12, dimension)
        features[~mask] = 0
        labels = torch.zeros((rows, 12), dtype=torch.bool)
        labels[:, 0] = True
        labels[~mask] = False
        weights = torch.ones(rows)
        fallback = torch.zeros(rows, dtype=torch.long)
        return features, mask, labels, weights, fallback

    def test_cheap_oof_features_and_augmentation(self):
        features, mask, _, _, fallback = self.batch()
        model = SequentialCandidateVerifier(115, dropout=0.0)
        logits, _ = model(features, mask)
        extras, order = cheap_oof_features(logits, mask, fallback)
        augmented = augment_verifier_features(features, extras, mask)
        self.assertEqual(augmented.shape, (8, 12, 120))
        self.assertEqual(order.shape, (8, 12))
        self.assertTrue(torch.all(augmented[~mask] == 0))

    def test_synthetic_checkpoint_fit(self):
        config = yaml.safe_load((RUN_DIR / "configs/sequential_training_prereg.yaml").read_text())
        config["optimizer"]["maximum_epochs"] = 2
        config["optimizer"]["patience"] = 1
        train = self.batch()[:4]
        checkpoint = self.batch()[:4]
        model, report = fit_with_checkpoint(
            train, checkpoint, 115, config, 17, torch.device("cpu")
        )
        self.assertIsInstance(model, SequentialCandidateVerifier)
        self.assertGreaterEqual(report["selected_epoch"], 1)

    def test_real_data_optimizer_fails_closed(self):
        config = yaml.safe_load((RUN_DIR / "configs/sequential_training_prereg.yaml").read_text())
        with self.assertRaisesRegex(PermissionError, "not authorized"):
            require_real_data_optimizer_authorization(config)

    def test_zero_weight_rows_never_form_empty_weight_batch(self):
        features, mask, labels, weights, _ = self.batch(rows=8)
        weights[:] = 0
        weights[[1, 6]] = 1
        model = SequentialCandidateVerifier(115, dropout=0.0)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        value = train_epoch(
            model, features, mask, labels, weights, optimizer,
            batch_size=1, pairwise_weight=0.5,
            gradient_clip_norm=1.0, seed=19,
        )
        self.assertTrue(torch.isfinite(torch.tensor(value)))


if __name__ == "__main__":
    unittest.main()