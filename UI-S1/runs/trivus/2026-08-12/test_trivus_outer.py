import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch


sys.path.insert(0, str(Path(__file__).resolve().parent))

from context_common import sha256_file
from test_trivus_data import make_data
from trivus_data import TriVUSStandardizer
from trivus_fit import make_model
from test_formal_primitives import threshold_rows
from trivus_outer import (
    MODEL_SPECS, POLICY_SPECS, acquire_outer_lock, load_final_artifact,
    load_outer_after_pretest, pretest_allowed_fields, reload_before_outer_labels,
    validate_pretest, write_final_artifact,
    recompute_thresholds_from_pretest, rows_sha256, write_jsonl_artifact,
)
from formal_authorization import validate_worker_receipt


class FakeRecovery:
    pass


class TriVUSOuterTest(unittest.TestCase):
    @staticmethod
    def threshold_tree(families):
        cells = {
            "mind2web": ("C_uni", "C_cond", "C_rand", "C_self"),
            "screenspot_pro": ("C_uni", "C_cond", "C_rand", "C_self"),
            "androidcontrol": ("low", "high"),
        }
        return {"families": {
            family: {
                "family_threshold": [float("inf"), float("inf")],
                "family_selection": {},
                "cells": {
                    cell: {
                        "threshold": [float("inf"), float("inf")],
                        "threshold_source": "family_backoff",
                        "changed_opportunities": 0,
                        "selection": {},
                    }
                    for cell in cells[family]
                },
            }
            for family in families
        }}

    def test_final_artifact_round_trip_and_hash_guard(self):
        model = make_model()
        standardizer = TriVUSStandardizer(
            mean=np.zeros(115, dtype=np.float32),
            scale=np.ones(115, dtype=np.float32),
            variant="JOINT3",
        )
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            path = Path(directory) / "model.pt"
            item = write_final_artifact(
                path, Path(directory), model, standardizer, "JOINT3", 3, 7
            )
            loaded, restored = load_final_artifact(
                item, Path(directory), "JOINT3", 3, 7, torch.device("cpu")
            )
            self.assertEqual(restored.variant, "JOINT3")
            for left, right in zip(model.parameters(), loaded.parameters()):
                self.assertTrue(torch.equal(left, right))
            path.write_bytes(path.read_bytes() + b"drift")
            with self.assertRaisesRegex(PermissionError, "hash"):
                load_final_artifact(
                    item, Path(directory), "JOINT3", 3, 7,
                    torch.device("cpu"),
                )

    def test_pretest_schema_and_outer_fold_seal(self):
        self.assertIn("outer_labels_opened", pretest_allowed_fields())
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            root = Path(directory)
            artifacts = {}
            for spec_id in MODEL_SPECS:
                path = root / f"{spec_id}.pt"
                path.write_bytes(spec_id.encode())
                artifacts[spec_id] = {
                    "path": str(path.relative_to(Path.cwd())),
                    "sha256": sha256_file(path),
                }
            record = {
                "schema_version": 1,
                "status": "PASS_TRIVUS_PRETEST_FROZEN_BEFORE_OUTER_LABEL_ACCESS",
                "outer_fold": 0,
                "development_folds": [1, 2, 3, 4],
                "sealed_outer_fold": 0,
                "opened_development_label_sha256": {},
                "sealed_outer_label_sha256": {},
                "code_and_data_sha256": {},
                "thresholds": {},
                "inner_epochs": {},
                "final_epochs": {},
                "final_seed": 0,
                "final_artifacts": artifacts,
                "optimizer_steps_per_epoch": 1,
                "outer_labels_opened": False,
                "training_complete": True,
            }
            import trivus_outer
            original = trivus_outer.assert_protected_process
            trivus_outer.assert_protected_process = lambda _: True
            try:
                with self.assertRaises(PermissionError):
                    validate_pretest(record, 0, {"seed": 20260812}, {}, FakeRecovery())
                with self.assertRaises(PermissionError):
                    validate_pretest(
                        {**record, "outer_labels_opened": True},
                        0, {"seed": 20260812}, {}, FakeRecovery(),
                    )
                with self.assertRaises(PermissionError):
                    validate_pretest(
                        {**record, "development_folds": [0, 1, 2, 3]},
                        0, {"seed": 20260812}, {}, FakeRecovery(),
                    )
            finally:
                trivus_outer.assert_protected_process = original

    def test_complete_pretest_record_passes_deep_validation(self):
        import trivus_outer

        model = make_model()
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            original_root = trivus_outer.OUTPUT_ROOT
            trivus_outer.OUTPUT_ROOT = Path(directory) / "formal"
            outer_dir = trivus_outer.OUTPUT_ROOT / "outer-0"
            artifacts = {}
            epochs = {spec_id: [1, 2, 3, 4] for spec_id in MODEL_SPECS}
            final_epochs = {spec_id: 3 for spec_id in MODEL_SPECS}
            checkpoints = {
                spec_id: [
                    {
                        "holdout_fold": fold,
                        "selected_epoch": epoch,
                        "selected_checkpoint_loss": 1.0,
                        "epochs_run": epoch,
                        "history": [],
                    }
                    for fold, epoch in zip((1, 2, 3, 4), (1, 2, 3, 4))
                ]
                for spec_id in MODEL_SPECS
            }
            for spec_id in MODEL_SPECS:
                variant = trivus_outer.model_spec(spec_id)["variant"]
                standardizer = TriVUSStandardizer(
                    mean=np.zeros(115, dtype=np.float32),
                    scale=np.ones(115, dtype=np.float32),
                    variant=variant,
                )
                artifacts[spec_id] = write_final_artifact(
                    outer_dir / f"{spec_id}.pt", outer_dir, model, standardizer,
                    spec_id, 3, 20261811,
                )
            thresholds = {
                policy: self.threshold_tree(families)
                for policy, (_, families) in POLICY_SPECS.items()
            }
            oof_artifacts = {}
            oof_hashes = {}
            for spec_id in MODEL_SPECS:
                rows = [{
                    "context_key": f"context/{spec_id}",
                    "family": "mind2web",
                }]
                oof_artifacts[spec_id] = write_jsonl_artifact(
                    outer_dir / "oof" / f"{spec_id}.jsonl", outer_dir, rows
                )
                oof_hashes[spec_id] = rows_sha256(rows)
            observed = {"vus": {str(fold): "a" * 64 for fold in (1, 2, 3, 4)}, "android": {str(fold): "b" * 64 for fold in (1, 2, 3, 4)}}
            sealed = {"vus": {"0": "c" * 64}, "android": {"0": "d" * 64}}
            record = {
                "schema_version": 1,
                "status": "PASS_TRIVUS_PRETEST_FROZEN_BEFORE_OUTER_LABEL_ACCESS",
                "outer_fold": 0,
                "development_folds": [1, 2, 3, 4],
                "sealed_outer_fold": 0,
                "opened_development_label_sha256": observed,
                "sealed_outer_label_sha256": sealed,
                "code_and_data_sha256": {"code": "e" * 64},
                "thresholds": thresholds,
                "inner_epochs": epochs,
                "inner_checkpoints": checkpoints,
                "final_epochs": final_epochs,
                "data_sha256": {
                    **{
                        str(fold): {
                            "model_training": "f" * 64,
                            "checkpoint": "1" * 64,
                            "oof": "2" * 64,
                        }
                        for fold in (1, 2, 3, 4)
                    },
                    "final_training": "3" * 64,
                },
                "oof_prediction_sha256": oof_hashes,
                "oof_artifacts": oof_artifacts,
                "final_seed": 20261811,
                "final_artifacts": artifacts,
                "optimizer_steps_per_epoch": 1,
                "outer_labels_opened": False,
                "training_complete": True,
            }
            originals = (
                trivus_outer.private_fold_hashes,
                trivus_outer.code_and_data_hashes,
                trivus_outer.assert_protected_process,
                trivus_outer.recompute_thresholds_from_pretest,
                trivus_outer.recompute_data_hashes,
                trivus_outer.load_locked_public_inputs,
                trivus_outer.expected_oof_contexts,
            )
            trivus_outer.private_fold_hashes = lambda _, folds: observed if tuple(folds) == (1, 2, 3, 4) else sealed
            trivus_outer.code_and_data_hashes = lambda *_: {"code": "e" * 64}
            trivus_outer.assert_protected_process = lambda _: True
            trivus_outer.recompute_thresholds_from_pretest = lambda *_: thresholds
            trivus_outer.recompute_data_hashes = lambda *_: record["data_sha256"]
            trivus_outer.load_locked_public_inputs = lambda *_: ({}, {})
            trivus_outer.expected_oof_contexts = lambda _, __, spec_id: {
                f"context/{spec_id}"
            }
            try:
                self.assertTrue(validate_pretest(
                    record, 0, {"seed": 20260812}, {}, FakeRecovery(),
                    trivus_outer.OUTPUT_ROOT,
                ))
            finally:
                (
                    trivus_outer.private_fold_hashes,
                    trivus_outer.code_and_data_hashes,
                    trivus_outer.assert_protected_process,
                    trivus_outer.recompute_thresholds_from_pretest,
                    trivus_outer.recompute_data_hashes,
                    trivus_outer.load_locked_public_inputs,
                    trivus_outer.expected_oof_contexts,
                ) = originals
                trivus_outer.OUTPUT_ROOT = original_root

    def test_outer_loader_requires_pretest_before_callback(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "missing.pretest.json"
            with self.assertRaisesRegex(PermissionError, "sealed"):
                load_outer_after_pretest(path, 0, {}, {}, {}, {}, None)

    def test_artifacts_reload_before_outer_label_callback(self):
        events = []
        loaded, labels = reload_before_outer_labels(
            [
                lambda: events.append("reload-1") or "model-1",
                lambda: events.append("reload-2") or "model-2",
            ],
            lambda: events.append("labels") or "outer-labels",
        )
        self.assertEqual(events, ["reload-1", "reload-2", "labels"])
        self.assertEqual(loaded, ["model-1", "model-2"])
        self.assertEqual(labels, "outer-labels")

    def test_outer_worker_lock_is_exclusive(self):
        import trivus_outer

        with tempfile.TemporaryDirectory() as directory:
            original = trivus_outer.OUTPUT_ROOT
            trivus_outer.OUTPUT_ROOT = Path(directory)
            try:
                acquire_outer_lock(3, Path(directory))
                with self.assertRaises(FileExistsError):
                    acquire_outer_lock(3, Path(directory))
            finally:
                trivus_outer.OUTPUT_ROOT = original

    def test_worker_rejects_missing_or_noncanonical_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "receipt.json"
            with self.assertRaises(PermissionError):
                validate_worker_receipt(path, 0)
            path.write_text(json.dumps({
                "schema_version": 1,
                "status": "CONSUMED_TRIVUS_FORMAL_AUTHORIZATION",
                "authorization_sha256": "a" * 64,
                "authorization_nonce": "b" * 64,
                "implementation_commit": "c" * 40,
                "authorization_commit": "d" * 40,
                "outer_folds": list(range(5)),
                "training_authorized": True,
            }))
            with self.assertRaises(PermissionError):
                validate_worker_receipt(path, 4)

    def test_oof_artifacts_round_trip_and_recompute_thresholds(self):
        import trivus_outer

        raw_rows = threshold_rows(count=3)
        public = {}
        rows = []
        for index, source in enumerate(raw_rows):
            fold = 1 + index % 4
            sample_key = source["context_key"]
            context_key = f"outer-0/inner-{fold}/{sample_key}"
            public[sample_key] = {
                "benchmark": source["family"],
                "fold": fold,
            }
            rows.append({**source, "context_key": context_key})
        by_spec = {}
        for spec_id in MODEL_SPECS:
            families = set(trivus_outer.model_spec(spec_id)["families"])
            by_spec[spec_id] = [
                row for row in rows if row["family"] in families
            ]
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            root = Path(directory)
            artifacts = {
                spec_id: write_jsonl_artifact(
                    root / "oof" / f"{spec_id}.jsonl", root, values
                )
                for spec_id, values in by_spec.items()
            }
            record = {"oof_artifacts": artifacts}
            original = trivus_outer.expected_oof_contexts
            trivus_outer.expected_oof_contexts = lambda _, __, spec_id: {
                row["context_key"] for row in by_spec[spec_id]
            }
            try:
                config = {
                    "thresholds": {
                        "mde": {
                            "mind2web": 0.006106589385659482,
                            "screenspot_pro": 0.007,
                            "androidcontrol": 0.01,
                        },
                        "minimum_cell_opportunities": 200,
                    }
                }
                thresholds = recompute_thresholds_from_pretest(
                    record, config, root, 0, public
                )
                restored = json.loads(json.dumps(thresholds, sort_keys=True))
                self.assertEqual(restored, thresholds)
                self.assertEqual(set(thresholds), set(POLICY_SPECS))
            finally:
                trivus_outer.expected_oof_contexts = original


if __name__ == "__main__":
    unittest.main()