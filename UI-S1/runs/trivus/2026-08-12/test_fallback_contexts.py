import ast
import json
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent))

from context_common import (
    ARMS, BENCHMARKS, ContextCandidate, ContextRow, android_majority_index,
    apply_vus_policies, checkpoint_and_fit_folds, context_record,
    committed_file, fit_final_vus_policies, fit_inner_vus_policies, load_prereg,
    load_sealed_rows, mind_layout, publish_directory, safe_child_path,
    require_commit_order, screen_layout, sha256_file, staging_directory,
)
from prepare_fallback_contexts import validate_context_coverage, write_authorization_receipt
from representation_gate import majority_public_index


class FallbackContextTest(unittest.TestCase):
    def test_frozen_prereg_and_context_arithmetic(self):
        config = load_prereg()
        self.assertEqual(config["expected"]["contexts"], 391524)
        self.assertEqual(
            config["expected"]["contexts"],
            config["expected"]["total_records"] * 21,
        )

    def test_exact_two_train_one_checkpoint_one_oof_split(self):
        for outer in range(5):
            development = set(range(5)) - {outer}
            for holdout in sorted(development):
                checkpoint, fit_folds = checkpoint_and_fit_folds(outer, holdout)
                candidates = development - {holdout}
                expected_checkpoint = next(
                    fold for offset in range(1, 6)
                    if (fold := (holdout + offset) % 5) in candidates
                )
                self.assertEqual(checkpoint, expected_checkpoint)
                self.assertEqual(len(fit_folds), 2)
                self.assertEqual(set(fit_folds) | {checkpoint, holdout}, development)

    def test_source_layouts_match_frozen_slot_budget(self):
        self.assertEqual(len(mind_layout()), 12)
        self.assertEqual(mind_layout()[0], ("stage1_TongUI-7B_view0", "TongUI-7B"))
        self.assertEqual(mind_layout()[-1], ("stage2_UI-TARS-7B_crop1", "UI-TARS-7B"))
        record = {"sample_key": "screenspot_pro/C_cond/r", "arm": "C_cond"}
        region = {
            "stage1_actions": [
                ["GTA1-7B", 0], ["Qwen3-VL-8B-Instruct", 0], ["UI-TARS-7B-SFT", 0],
                ["GTA1-7B", 1], ["Qwen3-VL-8B-Instruct", 1], ["UI-TARS-7B-SFT", 1],
            ],
        }
        layout = screen_layout(record, region)
        self.assertEqual(len(layout), 12)
        self.assertEqual(layout[6], ("GTA1-7B_C_cond_crop0", "GTA1-7B"))

    def test_android_policy_matches_frozen_representation_semantics(self):
        config = load_prereg()
        representation_config = {"seed": config["seed"]}
        row = {
            "sample_key": "androidcontrol/low/ac_1",
            "candidates": [
                {"action": "type", "parse_ok": True},
                {"action": "click", "parse_ok": True},
                {"action": "click", "parse_ok": True},
            ],
        }
        reliability = (0.9, 0.5, 0.4)
        self.assertEqual(
            android_majority_index(row, reliability, config["seed"]),
            majority_public_index(row, reliability, representation_config),
        )
        failed = {
            **row,
            "candidates": [{"action": "", "parse_ok": False} for _ in range(3)],
        }
        self.assertEqual(
            android_majority_index(failed, reliability, config["seed"]),
            majority_public_index(failed, reliability, representation_config),
        )

    def test_sealed_loader_opens_only_requested_folds(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            folds = {}
            for fold in range(5):
                path = root / f"fold-{fold}.jsonl"
                path.write_text(json.dumps({"sample_key": f"row-{fold}", "candidate_success": [True]}) + "\n")
                folds[str(fold)] = {"path": path.name, "rows": 1, "sha256": sha256_file(path)}
            manifest = {"folds": folds}
            (root / "fold-0.jsonl").unlink()
            (root / "fold-1.jsonl").unlink()
            (root / "fold-4.jsonl").unlink()
            rows, opened = load_sealed_rows(
                manifest, (2, 3), root, {fold: 1 for fold in range(5)}, {"row-2", "row-3"},
            )
            self.assertEqual(set(rows), {"row-2", "row-3"})
            self.assertEqual({Path(path).name for path in opened}, {"fold-2.jsonl", "fold-3.jsonl"})
            with self.assertRaises(ValueError):
                safe_child_path(root, "../outside.jsonl")
            with self.assertRaises(ValueError):
                safe_child_path(root, root / "fold-2.jsonl")

    def test_context_output_has_only_preregistered_fields(self):
        config = load_prereg()
        record = context_record(0, "inner", 1, (3, 4), "mind2web/C_uni/r", 2)
        self.assertEqual(set(record), set(config["context_schema"]))
        self.assertEqual(record["context_key"], "outer-0/inner-1/mind2web/C_uni/r")
        self.assertNotIn("success", record)
        self.assertNotIn("source", record)

    def test_generation_entrypoint_is_one_shot(self):
        source = (Path(__file__).resolve().parent / "prepare_fallback_contexts.py").read_text()
        self.assertNotIn("audit_outer", source)
        self.assertNotIn("selected_outer", source)
        self.assertNotIn("argparse", source)
        self.assertIn("for outer_fold in range(5):", source)

    def test_transaction_failure_leaves_no_published_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            destination = parent / "sealed"
            with self.assertRaisesRegex(RuntimeError, "injected"):
                with staging_directory(destination) as staging:
                    (staging / "partial.jsonl").write_text("partial\n")
                    raise RuntimeError("injected")
            self.assertFalse(destination.exists())
            self.assertEqual(list(parent.glob(".sealed.staging-*")), [])
            with staging_directory(destination) as staging:
                (staging / "data.jsonl").write_text("data\n")
                (staging / "MANIFEST.json").write_text("{}\n")
                publish_directory(staging, destination)
            self.assertTrue((destination / "data.jsonl").is_file())
            self.assertTrue((destination / "MANIFEST.json").is_file())

    def test_authorization_receipt_is_exclusive(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nonce.json"
            schema = ["schema_version", "status"]
            value = {"schema_version": 1, "status": "CONSUMED"}
            write_authorization_receipt(path, value, schema)
            with self.assertRaises(FileExistsError):
                write_authorization_receipt(path, value, schema)

    def test_authorization_commit_order_is_strict_and_untracked_is_rejected(self):
        with self.assertRaisesRegex(PermissionError, "not strict"):
            require_commit_order("same", "same", "test")
        run_dir = Path(__file__).resolve().parent
        with tempfile.TemporaryDirectory(prefix=".untracked-auth-", dir=run_dir) as directory:
            path = Path(directory) / "authorization.json"
            path.write_text("{}\n")
            with self.assertRaisesRegex(PermissionError, "not committed"):
                committed_file(path)

    def test_exact_per_sample_coverage_is_enforced(self):
        coverage = {
            "mind2web/C_uni/r": Counter(final=5, inner=16),
            "androidcontrol/low/r": Counter(final=5, inner=16),
        }
        self.assertTrue(validate_context_coverage(coverage, coverage))
        invalid = dict(coverage)
        invalid["mind2web/C_uni/r"] = Counter(final=5, inner=15)
        with self.assertRaisesRegex(ValueError, "coverage"):
            validate_context_coverage(invalid, coverage)

    def test_vus_fitters_never_read_nonfit_success(self):
        banks = {arm: {benchmark: {} for benchmark in BENCHMARKS} for arm in ARMS}
        public = {}
        for arm in ARMS:
            for benchmark in BENCHMARKS:
                for fold in range(5):
                    row_id = f"{benchmark}-{fold}"
                    candidates = tuple(
                        ContextCandidate(
                            source=f"source-{index}",
                            lineage=f"lineage-{index}",
                            action="CLICK" if benchmark == "mind2web" else "POINT",
                            baseline_coordinate=(0.1 + index * 0.01, 0.2 + index * 0.01),
                            parameter="",
                            parse_ok=True,
                            order=index,
                            success=(fold + index) % 2 == 0 if fold in {3, 4} else None,
                        )
                        for index in range(2)
                    )
                    banks[arm][benchmark][row_id] = ContextRow(
                        row_id=row_id,
                        benchmark=benchmark,
                        fold=fold,
                        group=f"group-{fold}",
                        candidates=candidates,
                    )
                    sample_key = f"{benchmark}/{arm}/{row_id}"
                    public[sample_key] = {
                        "sample_key": sample_key,
                        "benchmark": benchmark,
                        "arm": arm,
                        "row_id": row_id,
                        "fold": fold,
                    }
        scales = {
            "mind2web-3": (0.1, 0.1),
            "mind2web-4": (0.1, 0.1),
        }
        checkpoint, fit_folds = checkpoint_and_fit_folds(0, 1)
        self.assertEqual((checkpoint, fit_folds), (2, (3, 4)))
        inner = fit_inner_vus_policies(banks, fit_folds, checkpoint, scales)
        self.assertTrue(all(
            policy.config_validation_fold == 3
            for by_arm in inner.values() for policy in by_arm.values()
            if policy.benchmark == "mind2web"
        ))
        predictions = apply_vus_policies(banks, public, inner, (1, 2, 3, 4))
        self.assertEqual(len(predictions), 4 * len(ARMS) * len(BENCHMARKS))

        final_banks = {arm: {benchmark: {} for benchmark in BENCHMARKS} for arm in ARMS}
        for arm in ARMS:
            for benchmark in BENCHMARKS:
                for row_id, row in banks[arm][benchmark].items():
                    candidates = tuple(
                        ContextCandidate(**{
                            **candidate.__dict__,
                            "success": (
                                (row.fold + candidate.order) % 2 == 0
                                if row.fold in {1, 2, 3, 4} else None
                            ),
                        })
                        for candidate in row.candidates
                    )
                    final_banks[arm][benchmark][row_id] = ContextRow(
                        **{**row.__dict__, "candidates": candidates}
                    )
        final_scales = {f"mind2web-{fold}": (0.1, 0.1) for fold in (1, 2, 3, 4)}
        final = fit_final_vus_policies(final_banks, 0, final_scales)
        predictions = apply_vus_policies(final_banks, public, final, range(5))
        self.assertEqual(len(predictions), 5 * len(ARMS) * len(BENCHMARKS))

    def test_private_scale_seal_hashes_images_and_emits_no_target_coordinates(self):
        source = (Path(__file__).resolve().parent / "prepare_private_scales.py").read_text()
        self.assertIn('sha256_file(image_path) != public["image_sha256"]', source)
        tree = ast.parse(source)
        emitted = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Dict):
                continue
            keys = {
                key.value for key in node.keys
                if isinstance(key, ast.Constant) and isinstance(key.value, str)
            }
            if "normalized_width" in keys or "normalized_height" in keys:
                emitted.append(keys)
        self.assertEqual(emitted, [{
            "schema_version", "row_id", "normalized_width", "normalized_height",
        }])


if __name__ == "__main__":
    unittest.main()