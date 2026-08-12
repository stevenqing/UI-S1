import sys
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch


sys.path.insert(0, str(Path(__file__).resolve().parent))

from trivus_data import (
    INPUT_DIMENSION, MAX_CANDIDATES, TriVUSData, assign_weights, base_features,
    canonical_action, fit_standardizer, pair_kernel, pseudo_identity_permutation,
    restore_visual_values, structural_features, target_values, variant_features,
    torch_batch, validate_trivus_data, visual_features, whitespace_token_f1,
)


def candidates(count):
    output = []
    for index in range(count):
        output.append({
            "action": "click" if index < 2 else "type",
            "coordinate": [0.1 + 0.01 * index, 0.2] if index < 2 else None,
            "parameter": "" if index < 2 else "hello world",
            "parse_ok": index != count - 1,
        })
    return output


def prediction(count):
    permutation = list(reversed(range(count)))
    logits = np.arange(count, dtype=np.float64)
    probabilities = np.exp(logits - logits.max())
    probabilities /= probabilities.sum()
    return {
        "display_to_candidate": permutation,
        "label_logits": logits.tolist(),
        "label_probabilities": probabilities.tolist(),
    }


def make_data():
    rows = 10
    features = np.zeros((rows, MAX_CANDIDATES, INPUT_DIMENSION), dtype=np.float32)
    mask = np.zeros((rows, MAX_CANDIDATES), dtype=np.bool_)
    fallback = np.zeros(rows, dtype=np.int64)
    targets = np.zeros((rows, MAX_CANDIDATES + 1), dtype=np.float32)
    labels = np.zeros((rows, MAX_CANDIDATES), dtype=np.bool_)
    families = (
        "mind2web", "mind2web", "mind2web", "mind2web",
        "screenspot_pro", "screenspot_pro", "screenspot_pro", "screenspot_pro",
        "androidcontrol", "androidcontrol",
    )
    cells = ("C_uni", "C_cond", "C_rand", "C_self") * 2 + ("low", "high")
    for row, family in enumerate(families):
        count = 3 if family == "androidcontrol" else 12
        mask[row, :count] = True
        features[row] = base_features(
            candidates(count), prediction(count), 0, family, cells[row]
        )
        targets[row, 1] = 1.0
        labels[row, 1] = True
    active = np.ones(rows, dtype=np.bool_)
    weights = assign_weights(
        families, cells, active,
        ("mind2web", "screenspot_pro", "androidcontrol"),
    )
    return TriVUSData(
        features=features,
        candidate_mask=mask,
        fallback_indices=fallback,
        target_distribution=targets,
        fallback_correct=np.zeros(rows, dtype=np.float32),
        weights=weights,
        active=active,
        labels=labels,
        context_keys=tuple(f"context-{row}" for row in range(rows)),
        sample_keys=tuple(f"row-{row}" for row in range(rows)),
        families=families,
        cells=cells,
        row_ids=tuple(f"id-{row}" for row in range(rows)),
        folds=np.arange(rows, dtype=np.int8) % 5,
        groups=tuple(f"group-{row}" for row in range(rows)),
    )


class TriVUSDataTest(unittest.TestCase):
    def test_action_canonicalization_contract(self):
        self.assertEqual(canonical_action("POINT"), "POINT")
        self.assertEqual(canonical_action("open_app"), "OPEN")
        self.assertEqual(canonical_action("press_back"), "BACK")
        self.assertEqual(canonical_action("swipe"), "SCROLL")
        self.assertEqual(canonical_action("long_press"), "LONG_PRESS")
        self.assertEqual(canonical_action("UNKNOWN"), "OTHER")

    def test_token_f1_and_pair_kernel(self):
        self.assertEqual(whitespace_token_f1("hello world", "hello"), 2 / 3)
        left = {"action": "click", "coordinate": [0.1, 0.2], "parameter": ""}
        same = {"action": "click", "coordinate": [0.1, 0.2], "parameter": ""}
        other = {"action": "type", "coordinate": [0.1, 0.2], "parameter": ""}
        missing = {"action": "click", "coordinate": None, "parameter": ""}
        self.assertEqual(pair_kernel(left, same), 1.0)
        self.assertEqual(pair_kernel(left, other), 0.0)
        self.assertEqual(pair_kernel(left, missing), 0.0)

    def test_structural_feature_width_and_nonself_denominator(self):
        values = structural_features(candidates(3))
        self.assertEqual(values.shape, (3, 85))
        self.assertTrue(np.isfinite(values).all())
        self.assertEqual(values[0, 80], 0.5)
        self.assertGreater(values[0, 81], 0)
        self.assertEqual(values[0, 84], 0)
        self.assertTrue(np.allclose(np.linalg.norm(values[2, 16:80]), 1.0))

    def test_public_candidate_schema_is_fail_closed(self):
        malformed = candidates(3)
        malformed[0] = {**malformed[0], "coordinate": [float("inf"), 0.2]}
        with self.assertRaisesRegex(ValueError, "coordinate"):
            structural_features(malformed)
        outside = candidates(3)
        outside[0] = {**outside[0], "coordinate": [1.05, -0.01]}
        self.assertTrue(np.isfinite(structural_features(outside)).all())
        malformed = candidates(3)
        malformed[0] = {**malformed[0], "parameter": "x" * 257}
        with self.assertRaisesRegex(ValueError, "schema"):
            structural_features(malformed)
        malformed = candidates(3)
        malformed[0] = {**malformed[0], "parse_ok": 1}
        with self.assertRaisesRegex(ValueError, "schema"):
            structural_features(malformed)
        malformed = candidates(3)
        malformed[0] = {**malformed[0], "source": "private-model"}
        with self.assertRaisesRegex(ValueError, "schema"):
            structural_features(malformed)
        for value in (True, "0.1"):
            malformed = candidates(3)
            malformed[0] = {**malformed[0], "coordinate": [value, 0.2]}
            with self.assertRaisesRegex(ValueError, "coordinate"):
                structural_features(malformed)

    def test_visual_restoration_and_exact_layout(self):
        pred = prediction(3)
        logits, probabilities = restore_visual_values(pred, 3)
        self.assertTrue(np.array_equal(logits, [2, 1, 0]))
        values = visual_features(pred, fallback_index=1, count=3)
        self.assertEqual(values.shape, (3, 7))
        self.assertTrue(np.allclose(values[:, 5], values[:, 0] - values[1, 0]))
        features = base_features(candidates(3), pred, 1, "androidcontrol", "low")
        self.assertEqual(features.shape, (12, 115))
        self.assertEqual(features[1, 92], 1)
        self.assertTrue(np.all(features[:3, 93] == 0.25))
        self.assertTrue(np.all(features[3:] == 0))
        self.assertTrue(np.all(features[:, 103:] == 0))

    def test_target_keep_repair_and_activity(self):
        target, fallback_correct, active = target_values([False, True, True], 0)
        self.assertTrue(np.allclose(target[:3], [0, 0.5, 0.5]))
        self.assertEqual(fallback_correct, 0)
        self.assertTrue(active)
        target, fallback_correct, active = target_values([True, False, True], 0)
        self.assertEqual(target[12], 1)
        self.assertEqual(fallback_correct, 1)
        self.assertTrue(active)
        target, _, active = target_values([False, False, False], 0)
        self.assertEqual(target[12], 1)
        self.assertFalse(active)

    def test_equal_family_and_cell_weights(self):
        families = (
            "mind2web", "mind2web", "mind2web", "mind2web",
            "screenspot_pro", "screenspot_pro", "screenspot_pro", "screenspot_pro",
            "androidcontrol", "androidcontrol",
        )
        cells = ("C_uni", "C_cond", "C_rand", "C_self") * 2 + ("low", "high")
        weights = assign_weights(
            families, cells, np.ones(10, dtype=np.bool_),
            ("mind2web", "screenspot_pro", "androidcontrol"),
        )
        self.assertAlmostEqual(weights[:4].sum(), 1)
        self.assertAlmostEqual(weights[4:8].sum(), 1)
        self.assertAlmostEqual(weights[8:].sum(), 1)
        self.assertAlmostEqual(weights.sum(), 3)
        joint2 = assign_weights(
            families, cells, np.ones(10, dtype=np.bool_),
            ("mind2web", "screenspot_pro"),
        )
        self.assertAlmostEqual(joint2.sum(), 2)
        self.assertTrue(np.all(joint2[8:] == 0))

    def test_variants_mask_and_placebo_without_padding_leakage(self):
        data = make_data()
        no_visual = variant_features(data.features, data.candidate_mask, data.context_keys, "NO_VISUAL")
        self.assertTrue(np.all(no_visual[:, :, 85:92] == 0))
        self.assertTrue(np.all(no_visual[:, :, 103:115] == 0))
        placebo = variant_features(data.features, data.candidate_mask, data.context_keys, "RANDOM_ID_PLACEBO")
        for row, mask in enumerate(data.candidate_mask):
            count = int(mask.sum())
            self.assertTrue(np.all(placebo[row, :count, 103:115].sum(axis=1) == 1))
            self.assertTrue(np.all(placebo[row, count:] == 0))
            expected = pseudo_identity_permutation(data.context_keys[row], count)
            self.assertTrue(np.array_equal(placebo[row, :count, 103:115].argmax(axis=1), expected))
        first = pseudo_identity_permutation("outer-0/final/same-sample", 12)
        second = pseudo_identity_permutation("outer-0/inner-1/same-sample", 12)
        self.assertFalse(np.array_equal(first, second))

    def test_train_only_standardizer_rezeros_padding(self):
        data = make_data()
        train = data.subset(data.folds != 4)
        standardizer = fit_standardizer(train, "JOINT3")
        transformed = standardizer.transform(data.subset(data.folds == 4))
        self.assertTrue(np.isfinite(transformed.features).all())
        self.assertTrue(np.all(transformed.features[~transformed.candidate_mask] == 0))
        self.assertEqual(standardizer.mean.shape, (115,))
        self.assertEqual(standardizer.scale.shape, (115,))

    def test_unified_validator_and_checked_torch_batch(self):
        data = make_data()
        self.assertTrue(validate_trivus_data(
            data, ("mind2web", "screenspot_pro", "androidcontrol")
        ))
        batch = torch_batch(data, np.arange(len(data)), torch.device("cpu"))
        self.assertEqual(batch.features.shape, (10, 12, 115))
        self.assertEqual(batch.candidate_mask.dtype, torch.bool)
        self.assertEqual(batch.fallback_indices.dtype, torch.long)

    def test_validator_rejects_target_identity_mask_and_inactive_weight_defects(self):
        data = make_data()
        bad_targets = data.target_distribution.copy()
        bad_targets[0] = 0
        bad_targets[0, 2] = 1
        with self.assertRaisesRegex(ValueError, "target/label"):
            validate_trivus_data(replace(data, target_distribution=bad_targets))

        duplicate_keys = (data.context_keys[0], data.context_keys[0], *data.context_keys[2:])
        with self.assertRaisesRegex(ValueError, "context keys"):
            validate_trivus_data(replace(data, context_keys=duplicate_keys))

        scattered = data.candidate_mask.copy()
        scattered[8, 0] = False
        scattered[8, 3] = True
        with self.assertRaisesRegex(ValueError, "prefix-valid"):
            validate_trivus_data(replace(data, candidate_mask=scattered))

        inactive = data.active.copy()
        inactive[0] = False
        with self.assertRaisesRegex(ValueError, "inactive rows"):
            validate_trivus_data(replace(data, active=inactive))


if __name__ == "__main__":
    unittest.main()