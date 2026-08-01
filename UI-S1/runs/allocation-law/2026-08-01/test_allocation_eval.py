import tempfile
import unittest
import json
from pathlib import Path

import yaml

import numpy as np

from allocation_eval import (
    MDE,
    build_pool,
    cohen_kappa_from_counts,
    canonical_hash,
    failure_statistics,
    l1_predictions,
    l2_units,
    load_l1_units,
    load_model_views,
    matched_marginal_permutation,
)
from run_l2 import group_sufficient_statistics, weighted_kappa


class AllocationEvaluationTest(unittest.TestCase):
    def setUp(self):
        self.run_dir = Path(__file__).parent

    def test_frozen_pool_configs(self):
        l1 = load_l1_units(self.run_dir / "configs/l1_pools.yaml")
        self.assertEqual({budget: len(units) for budget, units in l1.items()}, {4: 4, 8: 8, 12: 12, 16: 16, 24: 24})
        l2 = l2_units(self.run_dir / "configs/l2_pools.yaml")
        self.assertEqual(len(l2), 8)
        self.assertTrue(all(len(units) == len(set(units)) == 12 for units in l2.values()))

    def test_explicit_prefix_mismatch_fails_closed(self):
        source = yaml.safe_load((self.run_dir / "configs/l1_pools.yaml").read_text())
        source["budget_prefixes"][4][0] = "Qwen3-VL-8B-Instruct/view7"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.yaml"
            path.write_text(yaml.safe_dump(source))
            with self.assertRaisesRegex(ValueError, "prefix mismatch"):
                load_l1_units(path)

    def test_prefix_token_mismatch_fails_closed(self):
        source = yaml.safe_load((self.run_dir / "configs/l1_pools.yaml").read_text())
        source["budget_prefixes"][12] = "first_16_allocation_sequence"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.yaml"
            path.write_text(yaml.safe_dump(source))
            with self.assertRaisesRegex(ValueError, "token mismatch"):
                load_l1_units(path)

    def test_duplicate_units_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "duplication"):
            build_pool({}, {}, [("GTA1-7B", 0), ("GTA1-7B", 0)])

    def test_l1_strict_mde_and_all_budget_contract(self):
        curves = {"v_only": {}, "mixed": {}}
        for budget in (4, 8, 12, 16):
            curves["v_only"][budget] = {"B3_mvp": 0.5, "M1_ccm": 0.5}
            curves["mixed"][budget] = {"B3_mvp": 0.52, "M1_ccm": 0.52}
        curves["mixed"][8]["B3_mvp"] = curves["mixed"][4]["B3_mvp"] + MDE
        result = l1_predictions(curves)
        first = result["P-L1a"]["increments"]["B3_mvp"][0]
        self.assertFalse(first["satisfied"])
        self.assertEqual(result["P-L1a"]["status"], "FAIL")
        self.assertEqual(result["P-L1b"]["status"], "PASS")
        self.assertFalse(result["kill_conditions"]["L-K2"])

        curves["mixed"][12]["M1_ccm"] = 0.48
        result = l1_predictions(curves)
        self.assertTrue(result["kill_conditions"]["L-K2"])

    def test_kappa_and_matched_marginal_null(self):
        self.assertEqual(cohen_kappa_from_counts(10, 5, 5, 5), 1.0)
        self.assertEqual(cohen_kappa_from_counts(10, 5, 5, 0), -1.0)
        self.assertIsNone(cohen_kappa_from_counts(10, 10, 10, 10))
        statistics = {
            "rows": 100,
            "mean_pairwise_kappa": 1.0,
            "pair_counts": [(50, 50, 50)],
        }
        result = matched_marginal_permutation(statistics, np.random.default_rng(1), permutations=1000)
        self.assertLess(abs(result["null_mean"]), 0.02)
        self.assertEqual(result["p_greater_equal_observed"], 1 / 1001)

    def test_model_trace_hash_and_target_integrity(self):
        model = "Qwen3-VL-8B-Instruct"
        revision = "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"
        regions = [[index, 0, index + 1, 1] for index in range(12)]
        manifest = {"row": {
            "stable_index": 0,
            "shared_region_candidate_sha256": "n12-hash",
            "regions": regions,
        }}
        old_predictions = [
            {"view_index": index, "region": regions[index], "point": [index, 0]}
            for index in range(4)
        ]
        new_predictions = [
            {"view_index": index, "region": regions[index], "point": [index, 0]}
            for index in range(4, 12)
        ]
        base = {
            "id": "row", "stable_index": 0, "model_id": model,
            "model_revision": revision, "num_shards": 4, "shard_index": 0,
        }
        old = {**base, "predictions": old_predictions, "prediction_sha256": canonical_hash(old_predictions)}
        new = {
            **base, "predictions": new_predictions, "prediction_sha256": canonical_hash(new_predictions),
            "shared_region_candidate_sha256": "n12-hash",
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            old_root = root / "old"; old_root.mkdir()
            old_path = old_root / "shard-0.jsonl"
            new_path = root / "new.jsonl"
            old_path.write_text(json.dumps(old) + "\n")
            new_path.write_text(json.dumps(new) + "\n")
            result = load_model_views(old_root, [new_path], manifest, model, expected_rows=1)
            self.assertEqual(len(result["row"]), 12)
            new["predictions"][0]["point"] = [999, 999]
            new_path.write_text(json.dumps(new) + "\n")
            with self.assertRaisesRegex(ValueError, "prediction hash mismatch"):
                load_model_views(old_root, [new_path], manifest, model, expected_rows=1)

    def test_vectorized_weighted_kappa_matches_direct(self):
        rows = []
        outputs = {method: {} for method in ("pass_at_n", "B3_mvp", "M1_ccm")}
        for index in range(20):
            row_id = f"row-{index}"
            candidates = [
                {"point": [0, 0] if (index + view) % 3 == 0 else [100, 100]}
                for view in range(12)
            ]
            rows.append({
                "id": row_id,
                "application": f"app-{index % 2}",
                "target_bbox": [-1, -1, 1, 1],
                "candidates": candidates,
            })
            for method in outputs:
                outputs[method][row_id] = index % 2 == 0
        direct = failure_statistics(rows)["mean_pairwise_kappa"]
        statistics = group_sufficient_statistics(rows, outputs, ["app-0", "app-1"])
        vectorized, total = weighted_kappa(np.ones((1, 2), dtype=np.int64), statistics, np.ones(2, dtype=np.int64))
        self.assertEqual(total.tolist(), [20])
        self.assertAlmostEqual(vectorized[0], direct)


if __name__ == "__main__":
    unittest.main()
