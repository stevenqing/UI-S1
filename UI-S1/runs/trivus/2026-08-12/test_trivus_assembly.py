import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parent))

from context_common import sha256_file
from trivus_assembly import (
    PhaseContext, assemble_data, audit_prediction, audit_public_row,
    included_families_for_variant, legal_requested_folds, load_context_phase,
    phase_contract, select_phase_contexts, with_model_weights,
)
from trivus_data import validate_trivus_data


SCHEMAS = {
    "vus_public": ["schema_version", "sample_key", "benchmark", "arm", "row_id", "fold", "group", "image_path", "image_sha256", "instruction", "history", "candidates"],
    "android_public": ["schema_version", "sample_key", "benchmark", "setting", "row_id", "fold", "group", "image_path", "image_sha256", "instruction", "history", "candidates"],
    "candidate": ["action", "coordinate", "parameter", "parse_ok"],
    "vus_prediction": ["schema_version", "sample_key", "benchmark", "arm", "row_id", "fold", "group", "display_to_candidate", "selected_label", "selected_candidate_index", "label_logits", "label_probabilities", "prompt_sha256", "image_sha256", "model_index_sha256"],
    "android_prediction": ["schema_version", "sample_key", "benchmark", "setting", "row_id", "fold", "group", "display_to_candidate", "selected_label", "selected_candidate_index", "label_logits", "label_probabilities", "prompt_sha256", "overlay_sha256", "image_sha256", "model_index_sha256"],
}


def candidate(index):
    return {
        "action": "click",
        "coordinate": [0.1 + index * 0.01, 0.2],
        "parameter": "",
        "parse_ok": True,
    }


def public_row(family, cell, row_id, fold):
    count = 3 if family == "androidcontrol" else 12
    row = {
        "schema_version": 1,
        "sample_key": f"{family}/{cell}/{row_id}",
        "benchmark": family,
        "row_id": row_id,
        "fold": fold,
        "group": f"group-{fold}",
        "image_path": "image.png",
        "image_sha256": "a" * 64,
        "instruction": "task",
        "history": "" if family == "androidcontrol" else [],
        "candidates": [candidate(index) for index in range(count)],
    }
    row["setting" if family == "androidcontrol" else "arm"] = cell
    return row


def prediction(row):
    count = len(row["candidates"])
    probabilities = np.arange(1, count + 1, dtype=np.float64)
    probabilities /= probabilities.sum()
    output = {
        "schema_version": 1,
        "sample_key": row["sample_key"],
        "benchmark": row["benchmark"],
        "row_id": row["row_id"],
        "fold": row["fold"],
        "group": row["group"],
        "display_to_candidate": list(range(count)),
        "selected_label": ("C" if count == 3 else "L"),
        "selected_candidate_index": count - 1,
        "label_logits": list(np.arange(count, dtype=float)),
        "label_probabilities": probabilities.tolist(),
        "prompt_sha256": "b" * 64,
        "image_sha256": row["image_sha256"],
        "model_index_sha256": "c" * 64,
    }
    key = "setting" if row["benchmark"] == "androidcontrol" else "arm"
    output[key] = row[key]
    if row["benchmark"] == "androidcontrol":
        output["overlay_sha256"] = "d" * 64
    return output


def public_bank():
    rows = []
    for fold in range(5):
        for cell in ("C_uni", "C_cond", "C_rand", "C_self"):
            rows.append(public_row("mind2web", cell, f"m-{cell}-{fold}", fold))
            rows.append(public_row("screenspot_pro", cell, f"s-{cell}-{fold}", fold))
        for cell in ("low", "high"):
            rows.append(public_row("androidcontrol", cell, f"a-{cell}-{fold}", fold))
    return {row["sample_key"]: row for row in rows}


def context_rows(public, outer, holdout):
    contract = phase_contract(outer, "inner", holdout)
    rows = []
    for sample_key, row in public.items():
        if row["fold"] == outer:
            continue
        rows.append({
            "schema_version": 1,
            "context_key": f"outer-{outer}/inner-{holdout}/{sample_key}",
            "sample_key": sample_key,
            "outer_fold": outer,
            "role": "inner",
            "holdout_fold": holdout,
            "fit_folds": list(contract["fit_folds"]),
            "fallback_index": 0,
        })
    return sorted(rows, key=lambda row: row["context_key"])


def complete_context_bank(public):
    rows = []
    splits = []
    for outer in range(5):
        final = phase_contract(outer, "final")
        final_rows = []
        for sample_key, row in public.items():
            final_rows.append({
                "schema_version": 1,
                "context_key": f"outer-{outer}/final/{sample_key}",
                "sample_key": sample_key,
                "outer_fold": outer,
                "role": "final",
                "holdout_fold": None,
                "fit_folds": list(final["fit_folds"]),
                "fallback_index": 0,
            })
        rows.extend(final_rows)
        splits.append({
            "outer_fold": outer,
            "role": "final",
            "holdout_fold": None,
            "fit_folds": list(final["fit_folds"]),
            "checkpoint_fold": None,
            "applied_folds": list(final["applied_folds"]),
            "contexts": len(final_rows),
        })
        for holdout in range(5):
            if holdout == outer:
                continue
            inner = phase_contract(outer, "inner", holdout)
            inner_rows = context_rows(public, outer, holdout)
            rows.extend(inner_rows)
            splits.append({
                "outer_fold": outer,
                "role": "inner",
                "holdout_fold": holdout,
                "fit_folds": list(inner["fit_folds"]),
                "checkpoint_fold": inner["checkpoint_fold"],
                "applied_folds": list(inner["applied_folds"]),
                "contexts": len(inner_rows),
            })
    rows.sort(key=lambda row: row["context_key"])
    manifest = {
        "records": len(rows),
        "public_records": len(public),
        "record_schema": [
            "schema_version", "context_key", "sample_key", "outer_fold",
            "role", "holdout_fold", "fit_folds", "fallback_index",
        ],
        "splits": splits,
    }
    return rows, manifest


class TriVUSAssemblyTest(unittest.TestCase):
    def setUp(self):
        self.config = {"schemas": SCHEMAS}
        self.public = public_bank()
        self.predictions = {key: prediction(row) for key, row in self.public.items()}

    def test_public_and_prediction_schemas_are_exact(self):
        for key, row in self.public.items():
            self.assertTrue(audit_public_row(row, self.config))
            self.assertTrue(audit_prediction(
                self.predictions[key], row, self.config, "c" * 64
            ))
        row = dict(next(iter(self.public.values())), source="private")
        with self.assertRaisesRegex(ValueError, "schema"):
            audit_public_row(row, self.config)
        key = next(iter(self.predictions))
        bad = dict(self.predictions[key], selected_candidate_index=0)
        with self.assertRaisesRegex(ValueError, "selected-candidate"):
            audit_prediction(bad, self.public[key], self.config, "c" * 64)
        bad = dict(self.predictions[key], model_index_sha256="e" * 64)
        with self.assertRaisesRegex(ValueError, "values"):
            audit_prediction(bad, self.public[key], self.config, "c" * 64)
        android = next(row for row in self.public.values() if row["benchmark"] == "androidcontrol")
        self.assertTrue(audit_public_row({**android, "instruction": ""}, self.config))

    def test_phase_contract_and_illegal_fold_requests(self):
        contract = phase_contract(0, "inner", 1)
        self.assertEqual(contract["checkpoint_fold"], 2)
        self.assertEqual(contract["fit_folds"], (3, 4))
        self.assertEqual(legal_requested_folds(contract, "inner", (4, 3), 0), (3, 4))
        self.assertEqual(legal_requested_folds(contract, "inner", (2,), 0), (2,))
        self.assertEqual(legal_requested_folds(contract, "inner", (1,), 0), (1,))
        with self.assertRaises(PermissionError):
            legal_requested_folds(contract, "inner", (2, 3), 0)

    def test_context_slice_has_exact_requested_fold_identity(self):
        rows, manifest = complete_context_bank(self.public)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "contexts.jsonl"
            path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
            phase = load_context_phase(
                path, manifest, self.public, 0, "inner",
                {fold: 10 for fold in range(5)}, holdout_fold=1,
            )
            selected, requested = select_phase_contexts(phase, self.public, (3, 4))
        self.assertEqual({self.public[row["sample_key"]]["fold"] for row in selected}, {3, 4})
        self.assertEqual(len(selected), 20)
        self.assertEqual(phase.fit_folds, (3, 4))
        self.assertEqual(requested, (3, 4))
        with self.assertRaises(PermissionError):
            select_phase_contexts(phase, self.public, (2, 3))

        selected_row = next(
            row for row in rows if row["context_key"].startswith("outer-0/inner-1/")
        )
        selected_row["success"] = True
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "contexts.jsonl"
            path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
            with self.assertRaisesRegex(ValueError, "schema"):
                load_context_phase(
                    path, manifest, self.public, 0, "inner",
                    {fold: 10 for fold in range(5)}, holdout_fold=1,
                )

    def test_context_manifest_count_drift_is_rejected(self):
        rows, manifest = complete_context_bank(self.public)
        manifest["splits"][1]["contexts"] += 1
        manifest["records"] += 1
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "contexts.jsonl"
            path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
            with self.assertRaisesRegex(ValueError, "phase-count"):
                load_context_phase(
                    path, manifest, self.public, 0, "inner",
                    {fold: 10 for fold in range(5)}, holdout_fold=1,
                )

    def test_fabricated_phase_context_is_rejected(self):
        with self.assertRaisesRegex(PermissionError, "validated bank scan"):
            PhaseContext(
                outer_fold=0,
                role="inner",
                holdout_fold=1,
                fit_folds=(3, 4),
                checkpoint_fold=2,
                applied_folds=(1, 2, 3, 4),
                rows=(),
                expected_fold_counts=tuple((fold, 10) for fold in range(5)),
                _validation_token=object(),
            )

    def test_assembly_weights_and_source_free_features(self):
        contexts = [
            row for row in context_rows(self.public, 0, 1)
            if self.public[row["sample_key"]]["fold"] in {3, 4}
        ]
        labels = {}
        for row in contexts:
            count = len(self.public[row["sample_key"]]["candidates"])
            labels[row["sample_key"]] = {
                "schema_version": 1,
                "sample_key": row["sample_key"],
                "candidate_success": [False, True, *([False] * (count - 2))],
            }
        data = assemble_data(contexts, self.public, self.predictions, labels)
        self.assertTrue(validate_trivus_data(data))
        self.assertTrue(np.all(data.weights == 0))
        joint = with_model_weights(data, "JOINT3")
        self.assertAlmostEqual(joint.weights.sum(), 3)
        joint2 = with_model_weights(data, "JOINT2_NO_ANDROID")
        self.assertAlmostEqual(joint2.weights.sum(), 2)
        self.assertTrue(np.all(joint2.weights[np.asarray(joint2.families) == "androidcontrol"] == 0))
        target = with_model_weights(data, "TARGET_ONLY", "androidcontrol")
        self.assertAlmostEqual(target.weights.sum(), 1)
        self.assertTrue(np.all(target.features[:, :, 103:115] == 0))

        duplicate = [contexts[0], {**contexts[0], "context_key": contexts[0]["context_key"] + "/copy"}]
        duplicate_labels = {contexts[0]["sample_key"]: labels[contexts[0]["sample_key"]]}
        with self.assertRaisesRegex(ValueError, "context/sample"):
            assemble_data(duplicate, self.public, self.predictions, duplicate_labels)

        extra = {key: dict(value) for key, value in labels.items()}
        first = next(iter(extra))
        extra[first]["source"] = "private"
        with self.assertRaisesRegex(ValueError, "label row"):
            assemble_data(contexts, self.public, self.predictions, extra)

    def test_physical_loader_can_ignore_absent_unrequested_folds(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            folds = {}
            for fold in range(5):
                path = root / f"fold-{fold}.jsonl"
                path.write_text(json.dumps({"sample_key": f"row-{fold}", "candidate_success": [False, True, False]}) + "\n")
                folds[str(fold)] = {"path": path.name, "rows": 1, "sha256": sha256_file(path)}
            for fold in (0, 1, 2):
                (root / f"fold-{fold}.jsonl").unlink()
            from context_common import load_sealed_rows
            labels, opened = load_sealed_rows(
                {"folds": folds}, (3, 4), root,
                {fold: 1 for fold in range(5)}, {"row-3", "row-4"},
            )
        self.assertEqual(set(labels), {"row-3", "row-4"})
        self.assertEqual({Path(path).name for path in opened}, {"fold-3.jsonl", "fold-4.jsonl"})


if __name__ == "__main__":
    unittest.main()