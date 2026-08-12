import sys
import tempfile
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent))

from representation_gate import (
    hash_random_index, load_config, majority_public_index,
    probability_by_public_candidate, validate_authorization,
)


class RepresentationGateTest(unittest.TestCase):
    def test_frozen_config_and_committed_private_seal(self):
        config, manifest = load_config(require_authorization=False)
        self.assertEqual(config["repair_auroc_threshold"], 0.55)
        self.assertEqual(config["noninferiority_margin"], 0.01)
        self.assertEqual(manifest["records"], 4000)

    def test_probability_restoration(self):
        prediction = {"display_to_candidate": [2, 0, 1], "label_probabilities": [0.1, 0.7, 0.2]}
        self.assertEqual(probability_by_public_candidate(prediction), [0.7, 0.2, 0.1])

    def test_hash_random_is_deterministic(self):
        self.assertEqual(hash_random_index("row", 20260812), hash_random_index("row", 20260812))
        self.assertIn(hash_random_index("row", 20260812), range(3))

    def test_majority_maps_canonical_priority_to_public_index(self):
        config, _ = load_config(require_authorization=False)
        row = {
            "sample_key": "androidcontrol/low/ac_1",
            "candidates": [
                {"action": "type", "parse_ok": True},
                {"action": "click", "parse_ok": True},
                {"action": "click", "parse_ok": True},
            ],
        }
        selected = majority_public_index(row, [0.9, 0.5, 0.4], config)
        self.assertIn(selected, range(3))
        self.assertEqual(row["candidates"][selected]["action"], "click")

    def test_execution_is_fail_closed_without_authorization(self):
        config, _ = load_config(require_authorization=False)
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(PermissionError):
                validate_authorization(config, Path(directory) / "missing.json")

    def test_raw_hash_and_full_schema_validation_are_mandatory(self):
        source = (Path(__file__).resolve().parent / "representation_gate.py").read_text()
        self.assertIn("load_locked_public_predictions(config)", source)
        self.assertIn("audit_public_record(row)", source)
        self.assertIn("validate_prediction(predictions[sample_key], row, selector_config)", source)


if __name__ == "__main__":
    unittest.main()