import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch


RUN_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(RUN_DIR))
sys.path.insert(0, str(RUN_DIR.parent / "2026-08-12"))

from sequential_verifier_oof_runner import (
    augment_data, load_cheap_rows, run_one, verifier_split,
)
from sequential_oof_runner import family_data
from test_trivus_data import make_data


class SequentialVerifierOOFRunnerTest(unittest.TestCase):
    def test_verifier_split_has_independent_checkpoint(self):
        split = verifier_split(0, 1)
        self.assertEqual(split, {
            "fit_folds": (3, 4),
            "checkpoint_fold": 2,
            "holdout_fold": 1,
        })

    def test_cheap_rows_and_augmentation_align_by_context(self):
        full, standardizer = family_data(make_data(), "mind2web")
        fold = int(full.folds[0])
        data = full.subset(full.folds == fold)
        rows = []
        for index, context in enumerate(data.context_keys):
            count = int(data.candidate_mask[index].sum())
            rows.append({
                "schema_version": 1,
                "context_key": context,
                "sample_key": data.sample_keys[index],
                "family": "mind2web",
                "cell": data.cells[index],
                "fold": int(data.folds[index]),
                "candidate_logits": [float(value) for value in range(count)],
                "candidate_probabilities": [0.5] * count,
                "candidate_order": list(reversed(range(count))),
            })
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rows.jsonl"
            path.write_text("".join(json.dumps(row) + "\n" for row in rows))
            loaded = load_cheap_rows(path, fold, "mind2web")
        augmented = augment_data(data, loaded, torch.device("cpu"))
        self.assertEqual(augmented[0].shape[-1], 120)
        self.assertTrue(torch.all(augmented[0][~augmented[1]] == 0))

    def test_authorization_fails_before_cheap_artifact_load(self):
        with patch("sequential_verifier_oof_runner.load_fold_data") as loader:
            with self.assertRaisesRegex(PermissionError, "not authorized"):
                run_one(0, 1, "mind2web", torch.device("cpu"))
        loader.assert_not_called()


if __name__ == "__main__":
    unittest.main()