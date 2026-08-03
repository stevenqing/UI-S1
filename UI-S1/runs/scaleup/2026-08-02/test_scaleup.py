import unittest
from unittest.mock import patch
import json
import tempfile
from pathlib import Path

from g1_lineage_gate import adjudicate_gate, cohen_kappa
from g2_prepare_regions import clear_singleton_torchrun_environment, normalize_existing_manifest, required_region_indices


GATE = {
    "pass_requires": {
        "minimum_pass_at_3": 0.78,
        "at_least_one_pairwise_kappa_below": 0.45,
    },
    "cancel_g2_if_pass_at_3_below": 0.75,
    "lineage_concentrated_if_all_pairwise_kappa_at_least": 0.55,
    "default_g2_threshold": 0.731,
    "concentrated_g2_effective_threshold": 0.704,
    "stretch_threshold": 0.731,
}


class ScaleUpGateTest(unittest.TestCase):
    def test_p1_n8_required_region_union(self):
        perturbations = {"1": [2, 8, 10], "2": [3, 8, 11], "3": [2, 9, 11]}
        self.assertEqual(required_region_indices(perturbations, "GTA1-72B", 8), list(range(12)))
        self.assertEqual(required_region_indices(perturbations, "UI-Venus-Ground-72B", 8), [0, 1, 2, 3, 8, 9, 10, 11])

    def test_existing_manifest_normalization_changes_only_requirements(self):
        row = {
            "id": "row",
            "regions_sha256": "frozen",
            "regions": [{"region_index": index} for index in range(12)],
            "perturbed_region_indices": {"1": [2, 4, 6], "2": [3, 5, 7], "3": [2, 5, 7]},
            "required_region_indices_by_model": {
                "GTA1-72B": list(range(12)),
                "UI-Venus-Ground-72B": list(range(8)),
                "Qwen3.5-122B-A10B": list(range(8)),
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.jsonl"
            path.write_text(json.dumps(row) + "\n")
            self.assertEqual(normalize_existing_manifest(path, 8), 1)
            normalized = json.loads(path.read_text())
            self.assertEqual(normalized["required_region_indices_by_model"]["GTA1-72B"], list(range(8)))
            self.assertEqual(normalized["regions_sha256"], "frozen")
            self.assertEqual(normalized["regions"], row["regions"])

    def test_singleton_host_world_size_is_not_torchrun(self):
        with patch.dict("os.environ", {"WORLD_SIZE": "1", "RANK": "0"}, clear=True):
            clear_singleton_torchrun_environment()
            import os
            self.assertNotIn("WORLD_SIZE", os.environ)
            self.assertNotIn("RANK", os.environ)

    def test_real_torchrun_environment_is_preserved(self):
        with patch.dict("os.environ", {"WORLD_SIZE": "8", "RANK": "0", "LOCAL_RANK": "0"}, clear=True):
            clear_singleton_torchrun_environment()
            import os
            self.assertEqual(os.environ["WORLD_SIZE"], "8")
            self.assertEqual(os.environ["LOCAL_RANK"], "0")

    def test_kappa_identical_and_opposed(self):
        self.assertAlmostEqual(cohen_kappa([0, 0, 1, 1], [0, 0, 1, 1]), 1.0)
        self.assertAlmostEqual(cohen_kappa([0, 0, 1, 1], [1, 1, 0, 0]), -1.0)

    def test_gate_standard_pass(self):
        result = adjudicate_gate(0.80, [0.40, 0.58, 0.62], GATE)
        self.assertTrue(result["G1_pass"])
        self.assertFalse(result["G2_cancelled"])
        self.assertEqual(result["G2_effective_threshold"], 0.731)

    def test_gate_concentrated_relaxes_threshold(self):
        result = adjudicate_gate(0.79, [0.60, 0.61, 0.62], GATE)
        self.assertFalse(result["G1_pass"])
        self.assertTrue(result["lineage_concentrated"])
        self.assertEqual(result["G2_effective_threshold"], 0.704)

    def test_gate_common_failure_cancels(self):
        result = adjudicate_gate(0.74, [0.40, 0.50, 0.60], GATE)
        self.assertTrue(result["G2_cancelled"])
        self.assertIsNone(result["G2_effective_threshold"])

    def test_gate_marginal_runs_without_pass(self):
        result = adjudicate_gate(0.76, [0.40, 0.50, 0.60], GATE)
        self.assertFalse(result["G1_pass"])
        self.assertFalse(result["G2_cancelled"])
        self.assertEqual(result["G2_action"], "RUN_G2_MARGINAL_GATE_STANDARD_THRESHOLD")


if __name__ == "__main__":
    unittest.main()