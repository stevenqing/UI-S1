import copy
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image
import yaml


sys.path.insert(0, str(Path(__file__).resolve().parent))

from civa_data import CHANNELS, EXPERTS, attach_labels, audit_public_record, build_base_data, text_features, validate_config
from civa_model import fit_uplift_model
from civa_train import CONFIG_PATH, load_test_after_pretest, matched_random, select_cell_threshold


def prediction(sample_key, candidate_index):
    permutation = [5, 2, 8, 1, 9, 0, 11, 4, 7, 3, 10, 6]
    probabilities = np.full(12, 0.01, dtype=np.float32)
    display_index = permutation.index(candidate_index)
    probabilities[display_index] = 0.89
    probabilities /= probabilities.sum()
    return {
        "sample_key": sample_key,
        "display_to_candidate": permutation,
        "label_logits": np.log(probabilities).tolist(),
        "label_probabilities": probabilities.tolist(),
    }


class CivaDataTest(unittest.TestCase):
    def test_frozen_config_matches_code_contract(self):
        validate_config(yaml.safe_load(CONFIG_PATH.read_text()))

    def test_outer_labels_are_sealed_before_pretest(self):
        with self.assertRaises(PermissionError):
            load_test_after_pretest(2, Path(self.directory.name) / "outer-2.pretest.json")

    def test_conservative_threshold_and_matched_random_coverage(self):
        neutral = [
            {
                "row_id": f"row-{index}", "changed": True, "score": float(index + 1),
                "baseline_success": index % 2 == 0, "expert_success": index % 2 == 0,
                "expert": "global_semantic",
            }
            for index in range(20)
        ]
        threshold, report = select_cell_threshold(neutral)
        self.assertTrue(np.isinf(threshold))
        self.assertEqual(report["switches"], 0)

        records = self.records()
        base = build_base_data(records, self.channels(records), text_dimensions=16)
        labels = {
            key: {"candidate_success": [index in (2, 3, 4, 5) for index in range(12)]}
            for key in records
        }
        labeled = attach_labels(base, labels)
        indices = [
            index for index, (benchmark, arm) in enumerate(zip(base.benchmarks, base.arms))
            if benchmark == "mind2web" and arm == "C_uni"
        ]
        output, random_report = matched_random(labeled, indices, 3, 20260811)
        self.assertEqual(len(output), len(indices))
        self.assertEqual(random_report["switches"], 3)
        self.assertEqual(sum(random_report["expert_counts"].values()), 3)

    def test_uplift_model_separates_rescue_and_harm(self):
        generator = np.random.default_rng(20260811)
        features = generator.normal(size=(600, 4)).astype(np.float32)
        delta = np.zeros((600, 1), dtype=np.int8)
        delta[features[:, 0] > 0.6, 0] = 1
        delta[features[:, 0] < -0.6, 0] = -1
        learner = {
            "learning_rate": 0.1,
            "max_iter": 80,
            "max_leaf_nodes": 7,
            "max_depth": None,
            "min_samples_leaf": 20,
            "l2_regularization": 0.1,
            "early_stopping": False,
        }
        model = fit_uplift_model(features, delta, np.ones(len(features)), learner, 20260811)
        score, rescue, harm = model.predict(features)
        self.assertGreater(float(score[features[:, 0] > 1.0].mean()), 0.5)
        self.assertLess(float(score[features[:, 0] < -1.0].mean()), -0.5)
        self.assertGreater(float(rescue[features[:, 0] > 1.0].mean()), float(harm[features[:, 0] > 1.0].mean()))

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.image_path = Path(self.directory.name) / "image.png"
        Image.new("RGB", (120, 80), "white").save(self.image_path)

    def tearDown(self):
        self.directory.cleanup()

    def records(self):
        output = {}
        for benchmark in ("mind2web", "screenspot_pro"):
            for row_number in range(4):
                for arm in ("C_uni", "C_cond", "C_rand", "C_self"):
                    row_id = f"row-{benchmark}-{row_number}"
                    key = f"{benchmark}/{arm}/{row_id}"
                    output[key] = {
                        "schema_version": 1,
                        "sample_key": key,
                        "benchmark": benchmark,
                        "arm": arm,
                        "row_id": row_id,
                        "fold": row_number,
                        "group": f"group-{benchmark}-{row_number}",
                        "image_path": str(self.image_path),
                        "image_sha256": "unused",
                        "instruction": "open advanced settings",
                        "history": [],
                        "candidates": [
                            {
                                "action": "CLICK",
                                "coordinate": [index / 12.0, (11 - index) / 12.0],
                                "parameter": "",
                                "parse_ok": True,
                            }
                            for index in range(12)
                        ],
                    }
        return output

    def channels(self, records):
        output = {name: {} for name in CHANNELS}
        selected = {"vus_binding": 2, "global_semantic": 3, "fine_local": 4, "context_local": 5, "random_placebo": 6}
        for name in CHANNELS:
            for key in records:
                output[name][key] = prediction(key, selected[name])
        return output

    def test_policy_mapping_and_rescue_harm_targets(self):
        records = self.records()
        base = build_base_data(records, self.channels(records), text_dimensions=16)
        labels = {
            key: {"candidate_success": [index == 3 for index in range(12)]}
            for key in records
        }
        labeled = attach_labels(base, labels)
        self.assertTrue(np.all(base.baseline_indices == 2))
        self.assertTrue(np.all(base.expert_indices[:, 0] == 3))
        self.assertTrue(np.all(labeled.delta[:, 0] == 1))
        self.assertTrue(np.all(labeled.delta[:, 1:] == 0))

        labels = {
            key: {"candidate_success": [index == 2 for index in range(12)]}
            for key in records
        }
        labeled = attach_labels(base, labels)
        self.assertTrue(np.all(labeled.delta == -1))

    def test_expert_values_cannot_change_admission_features(self):
        records = self.records()
        channels = self.channels(records)
        original = build_base_data(records, channels, text_dimensions=16)
        changed = copy.deepcopy(channels)
        for name in EXPERTS:
            for key in records:
                changed[name][key] = prediction(key, 11)
        counterfactual = build_base_data(records, changed, text_dimensions=16)
        self.assertTrue(np.array_equal(original.full_features, counterfactual.full_features))
        self.assertTrue(np.array_equal(original.no_text_features, counterfactual.no_text_features))
        self.assertTrue(np.array_equal(original.text_only_features, counterfactual.text_only_features))
        self.assertFalse(np.array_equal(original.expert_indices, counterfactual.expert_indices))

    def test_prohibited_field_is_rejected(self):
        record = next(iter(self.records().values()))
        record["target_bbox"] = [1, 2, 3, 4]
        with self.assertRaises(ValueError):
            audit_public_record(record)

    def test_instruction_hash_is_deterministic(self):
        records = list(self.records().values())
        first = text_features(records, dimensions=32)
        second = text_features(records, dimensions=32)
        self.assertTrue(np.array_equal(first, second))


if __name__ == "__main__":
    unittest.main()