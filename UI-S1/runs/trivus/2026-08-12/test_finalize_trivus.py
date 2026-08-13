import sys
import unittest
from pathlib import Path

import yaml


sys.path.insert(0, str(Path(__file__).resolve().parent))

from finalize_trivus import (
    CELL_ORDER, POLICIES, adjudicate, compare_policy, paired_samples,
    validate_outer_result,
)


class FinalizeTriVUSTest(unittest.TestCase):
    @staticmethod
    def outer_result(key="row-0", value=True):
        return {
            "schema_version": 1,
            "status": "PASS_TRIVUS_OUTER_COMPLETE",
            "outer_fold": 0,
            "pretest_sha256": "b" * 64,
            "inner_epochs": {},
            "final_epochs": {},
            "thresholds": {},
            "opened_outer_label_sha256": {},
            "reports": {},
            "outputs": {
                policy: {
                    method: {key: value}
                    for method in ("safe", "direct", "fallback")
                }
                for policy in POLICIES
            },
        }

    def test_paired_bootstrap_exact_identity(self):
        public = {
            f"row-{index}": {"fold": index % 5, "group": f"group-{index // 2}"}
            for index in range(20)
        }
        keys = sorted(public)
        left = {key: True for key in keys}
        right = {key: False for key in keys}
        report, samples = paired_samples(public, keys, left, right, 100, 7)
        self.assertEqual(report["point_delta"], 1.0)
        self.assertTrue((samples == 1).all())
        with self.assertRaises(ValueError):
            paired_samples(public, keys, {key: True for key in keys[:-1]}, right, 10, 1)
        with self.assertRaises(ValueError):
            paired_samples(public, keys, {**left, "extra": True}, right, 10, 1)

    def test_cell_order_has_ten_cells(self):
        self.assertEqual(len(CELL_ORDER), 10)
        self.assertEqual(CELL_ORDER[-2:], (("androidcontrol", "low"), ("androidcontrol", "high")))

    def test_formal_statistics_and_control_offsets_are_frozen(self):
        path = Path(__file__).resolve().parent / "configs/formal_runner_prereg.yaml"
        config = yaml.safe_load(path.read_text())
        self.assertEqual(config["statistics"]["resamples"], 10000)
        self.assertEqual(
            config["statistics"]["control_offsets"],
            {"primary": 0, "target_only": 100, "no_visual": 200, "strongest": 300},
        )

    def test_outer_result_requires_marker_fold_and_boolean_outputs(self):
        public = {"row-0": {"fold": 0}}
        marker = {
            "schema_version": 1,
            "status": "TRIVUS_OUTER_COMPLETE",
            "outer_fold": 0,
            "result_sha256": "a" * 64,
        }
        self.assertTrue(validate_outer_result(
            self.outer_result(), marker, public, 0, "a" * 64, "b" * 64
        ))
        with self.assertRaisesRegex(ValueError, "held-out"):
            validate_outer_result(
                self.outer_result(value=1), marker, public, 0,
                "a" * 64, "b" * 64,
            )
        with self.assertRaisesRegex(ValueError, "held-out"):
            validate_outer_result(
                self.outer_result(), marker, {"row-0": {"fold": 1}}, 0,
                "a" * 64, "b" * 64,
            )


if __name__ == "__main__":
    unittest.main()