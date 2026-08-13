import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch


RUN_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(RUN_DIR))
sys.path.insert(0, str(RUN_DIR.parent / "2026-08-12"))

from sequential_model import SequentialCandidateVerifier
from sequential_oof_runner import OOF_FIELDS, family_data, predict_rows, run_one
from test_trivus_data import make_data


class SequentialOOFRunnerTest(unittest.TestCase):
    def test_family_data_filters_before_training(self):
        data = make_data()
        selected, standardizer = family_data(data, "mind2web")
        self.assertEqual(set(selected.families), {"mind2web"})
        self.assertTrue(np.all(selected.weights > 0))
        self.assertEqual(standardizer.variant, "TARGET_ONLY")

    def test_prediction_schema_has_no_private_labels(self):
        data = make_data()
        selected, _ = family_data(data, "mind2web")
        model = SequentialCandidateVerifier(115, dropout=0.0)
        rows = predict_rows(model, selected, 8, torch.device("cpu"))
        self.assertTrue(rows)
        self.assertTrue(all(set(row) == OOF_FIELDS for row in rows))
        self.assertTrue(all("success" not in key for row in rows for key in row))

    def test_authorization_fails_before_public_or_label_load(self):
        with patch("sequential_oof_runner.load_locked_public_inputs") as loader:
            with self.assertRaisesRegex(PermissionError, "not authorized"):
                run_one(0, 1, "mind2web", torch.device("cpu"))
        loader.assert_not_called()


if __name__ == "__main__":
    unittest.main()