import sys
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
import yaml


sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_trivus_data import make_data
from trivus_fit import (
    MODEL_SPECS, epoch_order, epoch_permutations, half_up_median, model_spec,
    train_epoch, train_with_checkpoint,
)
from trivus_model import TriVUSSetRanker
from trivus_thresholds import (
    FAMILY_CELLS, apply_selected_thresholds, apply_threshold,
    compose_target_only, select_thresholds,
    select_cell_threshold, validate_threshold_rows,
)


class CountingAdam(torch.optim.AdamW):
    def __init__(self, parameters):
        super().__init__(parameters, lr=1e-3)
        self.steps = 0

    def step(self, closure=None):
        self.steps += 1
        return super().step(closure)


def threshold_rows(count=210):
    rows = []
    for family, cells in FAMILY_CELLS.items():
        for cell in cells:
            for index in range(count):
                fallback = index % 5 != 0
                direct = index % 7 != 0
                rows.append({
                    "context_key": f"{family}/{cell}/{index}",
                    "family": family,
                    "cell": cell,
                    "changed": True,
                    "margin": 0.1 + (index % 10) / 100,
                    "wrong_score": 0.2 + (index % 10) / 100,
                    "direct_success": direct,
                    "fallback_success": fallback,
                    "direct_index": 1,
                    "fallback_index": 0,
                })
    return rows


class FormalPrimitiveTest(unittest.TestCase):
    def test_formal_config_has_no_placeholders(self):
        path = Path(__file__).resolve().parent / "configs/formal_runner_prereg.yaml"
        self.assertNotIn("TO_BE_FROZEN", path.read_text())
        config = yaml.safe_load(path.read_text())
        self.assertEqual(config["status"], "FROZEN_AFTER_REAL_DATA_SMOKE_BEFORE_ANY_TRIVUS_OPTIMIZER_STEP")

    def test_exact_seven_specs_and_target_only_families(self):
        expected = {
            "JOINT3": ("JOINT3", None, ("mind2web", "screenspot_pro", "androidcontrol")),
            "TARGET_ONLY_MIND2WEB": ("TARGET_ONLY", "mind2web", ("mind2web",)),
            "TARGET_ONLY_SCREENSPOT_PRO": ("TARGET_ONLY", "screenspot_pro", ("screenspot_pro",)),
            "TARGET_ONLY_ANDROIDCONTROL": ("TARGET_ONLY", "androidcontrol", ("androidcontrol",)),
            "JOINT2_NO_ANDROID": ("JOINT2_NO_ANDROID", None, ("mind2web", "screenspot_pro")),
            "NO_VISUAL": ("NO_VISUAL", None, ("mind2web", "screenspot_pro", "androidcontrol")),
            "RANDOM_ID_PLACEBO": ("RANDOM_ID_PLACEBO", None, ("mind2web", "screenspot_pro", "androidcontrol")),
        }
        self.assertEqual(MODEL_SPECS, expected)

    def test_epoch_order_and_permutations_are_deterministic(self):
        keys = tuple(f"context-{index}" for index in range(12))
        indices = np.arange(12)
        first = epoch_order(keys, indices, 20260812, 1)
        second = epoch_order(keys, indices, 20260812, 1)
        self.assertTrue(np.array_equal(first, second))
        self.assertFalse(np.array_equal(first, epoch_order(keys, indices, 20260812, 2)))
        permutations = epoch_permutations(keys, 20260812, 1)
        self.assertEqual(permutations.shape, (12, 12))
        self.assertTrue(all(sorted(row.tolist()) == list(range(12)) for row in permutations))
        sample_keys = ("same-sample", "other-sample")
        self.assertTrue(np.array_equal(
            epoch_permutations(sample_keys, 20260812, 1),
            epoch_permutations(sample_keys, 20260812, 1),
        ))

    def test_one_optimizer_step_per_epoch(self):
        data = make_data()
        model = TriVUSSetRanker(115, width=16, heads=4, layers=1, dropout=0)
        optimizer = CountingAdam(model.parameters())
        train_epoch(
            model, data, optimizer, seed=20260812, epoch=1,
            batch_size=2, gradient_clip=1.0, device=torch.device("cpu"),
        )
        self.assertEqual(optimizer.steps, 1)

    def test_half_up_median(self):
        self.assertEqual(half_up_median([1, 2, 3, 4]), 3)
        self.assertEqual(half_up_median([2, 2, 3, 3]), 3)
        with self.assertRaises(ValueError):
            half_up_median([1, 2, 3])

    def test_train_checkpoint_overlap_is_rejected_before_optimizer(self):
        data = make_data()
        config = {
            "optimizer": {
                "learning_rate": 3e-4,
                "weight_decay": 1e-3,
                "maximum_epochs": 1,
                "batch_size": 2,
                "evaluation_batch_size": 4,
                "gradient_clip_norm": 1.0,
                "minimum_improvement": 1e-5,
                "patience": 1,
            }
        }
        with self.assertRaisesRegex(ValueError, "disjoint"):
            train_with_checkpoint(
                data, data, "JOINT3", config, 20260812, torch.device("cpu")
            )

    def test_threshold_safety_ties_and_full_coverage(self):
        rows = threshold_rows()
        selected = select_thresholds(
            rows,
            {"mind2web": 0.006106589385659482, "screenspot_pro": 0.007, "androidcontrol": 0.01},
            minimum_opportunities=200,
        )
        output, reports = apply_selected_thresholds(rows, selected)
        self.assertEqual(len(output), len(rows))
        self.assertEqual(set(reports), set(FAMILY_CELLS))
        for family, cells in FAMILY_CELLS.items():
            for cell in cells:
                selection = selected["families"][family]["cells"][cell]["selection"]
                self.assertGreaterEqual(
                    selection["point_delta"],
                    -0.5 * {"mind2web": 0.006106589385659482, "screenspot_pro": 0.007, "androidcontrol": 0.01}[family] - 1e-15,
                )
                self.assertEqual(
                    selected["families"][family]["cells"][cell]["threshold_source"],
                    "cell",
                )
            family_selection = selected["families"][family]["family_selection"]
            self.assertGreaterEqual(
                family_selection["equal_cell_delta"],
                -0.25 * {"mind2web": 0.006106589385659482, "screenspot_pro": 0.007, "androidcontrol": 0.01}[family] - 1e-15,
            )

    def test_cell_threshold_ties_prefer_wrong_then_margin(self):
        rows = [
            {
                "context_key": f"row-{index}",
                "family": "androidcontrol",
                "cell": "low",
                "changed": True,
                "margin": margin,
                "wrong_score": wrong,
                "direct_success": True,
                "fallback_success": False,
                "direct_index": 1,
                "fallback_index": 0,
            }
            for index, (margin, wrong) in enumerate(((0.1, 0.2), (0.2, 0.3)))
        ]
        threshold, _ = select_cell_threshold(rows, 0.01)
        self.assertGreaterEqual(threshold[1], 0.2)

    def test_threshold_rows_reject_duplicates_nan_and_changed_mismatch(self):
        row = threshold_rows(count=1)[0]
        with self.assertRaisesRegex(ValueError, "identity"):
            validate_threshold_rows([row, dict(row)])
        with self.assertRaisesRegex(ValueError, "numeric"):
            validate_threshold_rows([{**row, "margin": float("nan")}])
        with self.assertRaisesRegex(ValueError, "changed/index"):
            validate_threshold_rows([{
                **row, "direct_index": 1, "fallback_index": 1, "changed": True,
            }])
        incomplete = dict(row)
        incomplete.pop("direct_index")
        with self.assertRaisesRegex(ValueError, "schema"):
            validate_threshold_rows([incomplete])

    def test_infinite_threshold_is_fallback(self):
        rows = threshold_rows(count=1)
        output, report = apply_threshold(rows, (float("inf"), float("inf")))
        self.assertEqual(report["overrides"], 0)
        self.assertTrue(all(
            output[row["context_key"]] == row["fallback_success"] for row in rows
        ))

    def test_target_only_composition(self):
        values = {}
        for family, spec in (
            ("mind2web", "TARGET_ONLY_MIND2WEB"),
            ("screenspot_pro", "TARGET_ONLY_SCREENSPOT_PRO"),
            ("androidcontrol", "TARGET_ONLY_ANDROIDCONTROL"),
        ):
            values[spec] = [{"context_key": family, "family": family}]
        expected = {family: {family} for family in FAMILY_CELLS}
        output = compose_target_only(values, expected)
        self.assertEqual({row["family"] for row in output}, set(FAMILY_CELLS))
        values["TARGET_ONLY_MIND2WEB"] = []
        with self.assertRaisesRegex(ValueError, "coverage"):
            compose_target_only(values, expected)


if __name__ == "__main__":
    unittest.main()