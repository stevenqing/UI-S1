import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import yaml


RUN_DIR = Path(__file__).resolve().parent
ROOT = RUN_DIR.parents[2]
SOURCEBIAS_DIR = ROOT / "runs/sourcebias/2026-08-03"
H3_DIR = ROOT / "runs/ccm-h2h/2026-07-31/h3"
sys.path.insert(0, str(RUN_DIR))
sys.path.insert(0, str(SOURCEBIAS_DIR))
sys.path.insert(0, str(H3_DIR))

from common import MODELS, load_context, paired_group_bootstrap, write_json
from h3_eval import ccm_select, fit_ccm
from sourcebias_common import b3_select_index, point_in_bbox, split_ids


ARMS = ("C_cond", "C_rand", "C_self")
EXPECTED_MODELS = {
    "GTA1-7B": {
        "directory": "q1-gta1",
        "revision": "701bedc80b447863bd60e3318ae44f6cbbfafd78",
        "index_sha256": "3067e9b0f35596ff3426a0d0ec8c982a51fa1e110c4fc30dcf3be9ea37409df6",
    },
    "Qwen3-VL-8B-Instruct": {
        "directory": "q1-qwen3",
        "revision": "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b",
        "index_sha256": "520b2e05079402e9468a8701d03d1154d14b2599593afb6effa7fb60c1bff070",
    },
    "UI-TARS-7B-SFT": {
        "directory": "q1-uitars",
        "revision": "3434901a9dd04dd3625617d839a5724fe5e2db20",
        "index_sha256": "25b162a0f0f47af097d6a49b7da3d5c7d9c2b352490131c8cde5ca59d285f18b",
    },
}


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def load_unique(path):
    rows = {}
    for shard in sorted(path.glob("shard-*.jsonl")):
        for line in shard.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row["id"] in rows:
                raise ValueError(f"Q1 duplicate model prediction: {row['id']}")
            rows[row["id"]] = row
    if len(rows) != 1581:
        raise ValueError(f"Q1 requires 1,581 predictions in {path}, found {len(rows)}")
    return rows


def validate_predictions(regions, model_id, rows, spec):
    if set(rows) != set(regions):
        raise ValueError(f"Q1 identity mismatch: {model_id}")
    for row_id, row in rows.items():
        source = regions[row_id]
        if row["model_id"] != model_id or row["model_revision"] != spec["revision"]:
            raise ValueError(f"Q1 model provenance mismatch: {model_id}/{row_id}")
        if row["model_index_sha256"] != spec["index_sha256"]:
            raise ValueError(f"Q1 model index mismatch: {model_id}/{row_id}")
        if row["stable_index"] != source["stable_index"] or row["arms_sha256"] != source["arms_sha256"]:
            raise ValueError(f"Q1 region provenance mismatch: {model_id}/{row_id}")
        if canonical_hash(row["predictions"]) != row["predictions_sha256"]:
            raise ValueError(f"Q1 prediction hash mismatch: {model_id}/{row_id}")
        if set(row["predictions"]) != set(ARMS):
            raise ValueError(f"Q1 arm coverage mismatch: {model_id}/{row_id}")
        for arm in ARMS:
            values = row["predictions"][arm]
            if [value["crop_index"] for value in values] != [0, 1]:
                raise ValueError(f"Q1 crop order mismatch: {model_id}/{row_id}/{arm}")
            if [value["region"] for value in values] != source["arms"][arm]:
                raise ValueError(f"Q1 crop geometry mismatch: {model_id}/{row_id}/{arm}")


def candidate_from_existing(context, action, row_id):
    return dict(context["bank"][action][row_id])


def candidate_from_q1(model_id, crop_index, prediction):
    return {
        "model": model_id,
        "view_index": crop_index + 2,
        "point": list(map(float, prediction["point"])),
        "region": list(prediction["region"]),
        "coverage": 0.0,
    }


def build_arm_rows(context, regions, predictions, arm):
    rows = []
    for row_id in context["row_ids"]:
        source = regions[row_id]
        candidates = [
            candidate_from_existing(context, (model, view), row_id)
            for view in range(2) for model in MODELS
        ]
        for crop_index in range(2):
            for model_id in MODELS:
                candidates.append(candidate_from_q1(
                    model_id, crop_index, predictions[model_id][row_id]["predictions"][arm][crop_index]
                ))
        if len(candidates) != 12:
            raise ValueError("Q1 arm candidate budget mismatch")
        metadata = context["metadata"][row_id]
        rows.append({
            "id": row_id,
            "application": metadata["application"],
            "target_bbox": metadata["target_bbox"],
            "outer_fold": context["fold_for_group"][metadata["application"]],
            "candidates": candidates,
            "arms_sha256": source["arms_sha256"],
        })
    return rows


def evaluate_rows(context, rows):
    by_id = {row["id"]: row for row in rows}
    outputs = {"B3_mvp": {}, "M1_ccm": {}, "pass_at_n": {}}
    folds = []
    for fold in range(5):
        dev_ids, test_ids = split_ids(context, fold)
        dev_rows = [by_id[row_id] for row_id in dev_ids]
        test_rows = [by_id[row_id] for row_id in test_ids]
        tables, priors = fit_ccm(dev_rows)
        counts = Counter()
        for row in test_rows:
            b3_index, _ = b3_select_index(row["candidates"])
            m1_index = ccm_select(row, tables, priors)
            values = {
                "B3_mvp": point_in_bbox(row["candidates"][b3_index]["point"], row["target_bbox"]),
                "M1_ccm": point_in_bbox(row["candidates"][m1_index]["point"], row["target_bbox"]),
                "pass_at_n": any(point_in_bbox(candidate["point"], row["target_bbox"]) for candidate in row["candidates"]),
            }
            for metric, value in values.items():
                outputs[metric][row["id"]] = bool(value)
                counts[metric] += int(value)
        folds.append({
            "fold": fold,
            "dev_rows": len(dev_rows),
            "test_rows": len(test_rows),
            "accuracy": {metric: counts[metric] / len(test_rows) for metric in outputs},
            "source_priors": priors,
        })
    if any(len(values) != 1581 for values in outputs.values()):
        raise ValueError("Q1 evaluation output coverage mismatch")
    return {
        "rows": 1581,
        "folds": folds,
        "accuracy": {metric: sum(values.values()) / len(values) for metric, values in outputs.items()},
        "outputs": outputs,
        "row_metadata": {
            row["id"]: {"application": row["application"], "outer_fold": row["outer_fold"]}
            for row in rows
        },
    }


def main():
    config_path = RUN_DIR / "configs/q1_arms.yaml"
    config = yaml.safe_load(config_path.read_text())
    if config["status"] != "result_blind_design_freeze" or config["mandatory_controls"] != ["C_rand", "C_self"]:
        raise ValueError("Q1 config freeze mismatch")
    context = load_context()
    region_rows = [json.loads(line) for line in (RUN_DIR / "raw/q1_regions.jsonl").read_text().splitlines() if line.strip()]
    regions = {row["id"]: row for row in region_rows}
    if len(regions) != 1581 or set(regions) != set(context["row_ids"]):
        raise ValueError("Q1 prepared-region coverage mismatch")
    if any("target" in key or "bbox" in key for row in region_rows for key in row):
        raise ValueError("Q1 prepared-region target leak")

    predictions = {}
    for model_id, spec in EXPECTED_MODELS.items():
        rows = load_unique(RUN_DIR / "raw" / spec["directory"])
        validate_predictions(regions, model_id, rows, spec)
        predictions[model_id] = rows

    baseline_rows = []
    for row_id in context["row_ids"]:
        metadata = context["metadata"][row_id]
        baseline_rows.append({
            "id": row_id,
            "application": metadata["application"],
            "target_bbox": metadata["target_bbox"],
            "outer_fold": context["fold_for_group"][metadata["application"]],
            "candidates": [
                candidate_from_existing(context, (model, view), row_id)
                for view in range(4) for model in MODELS
            ],
        })
    evaluations = {"C_uni": evaluate_rows(context, baseline_rows)}
    for arm in ARMS:
        evaluations[arm] = evaluate_rows(context, build_arm_rows(context, regions, predictions, arm))

    expected = {"B3_mvp": 0.6369386464263125, "M1_ccm": 0.6382036685641999}
    for metric, value in expected.items():
        if abs(evaluations["C_uni"]["accuracy"][metric] - value) > 1e-15:
            raise ValueError(f"Q1 C_uni anchor mismatch: {metric}")

    comparisons = {}
    metadata = evaluations["C_uni"]["row_metadata"]
    for reference in ("C_uni", "C_rand", "C_self"):
        comparisons[f"C_cond_minus_{reference}"] = {
            metric: {
                **paired_group_bootstrap(
                    metadata,
                    evaluations["C_cond"]["outputs"][metric],
                    evaluations[reference]["outputs"][metric],
                ),
                "C_cond_accuracy": evaluations["C_cond"]["accuracy"][metric],
                "reference_accuracy": evaluations[reference]["accuracy"][metric],
            }
            for metric in ("B3_mvp", "M1_ccm", "pass_at_n")
        }
    primary = comparisons["C_cond_minus_C_uni"]["B3_mvp"]
    success = primary["point_delta"] > config["primary"]["mde"] and primary["ci_99"][0] > 0
    q_k1 = comparisons["C_cond_minus_C_rand"]["B3_mvp"]["point_delta"] <= 0
    q_k2 = comparisons["C_cond_minus_C_self"]["B3_mvp"]["point_delta"] <= 0
    result = {
        "schema_version": 1,
        "status": "PASS",
        "config": str(config_path.relative_to(ROOT)),
        "rows": 1581,
        "forward_budget": 12,
        "candidate_order": "stage1_view_major_then_stage2_crop_major_model_minor",
        "new_candidate_coverage_tie_break": 0.0,
        "evaluations": {
            arm: {"accuracy": evaluation["accuracy"], "folds": evaluation["folds"]}
            for arm, evaluation in evaluations.items()
        },
        "comparisons": comparisons,
        "primary_success": success,
        "Q_K1": q_k1,
        "Q_K2": q_k2,
        "claim": (
            "CROSS_LINEAGE_CONSENSUS_ROI_SUPPORTED"
            if success and not q_k1 and not q_k2
            else "CONSENSUS_ROI_SUPPORTED_NOT_CROSS_LINEAGE_SPECIFIC"
            if success and not q_k1 and q_k2
            else "SEQUENTIAL_CONDITIONING_NOT_SUPPORTED"
        ),
        "sources": {
            "regions_sha256": hashlib.sha256((RUN_DIR / "raw/q1_regions.jsonl").read_bytes()).hexdigest(),
            **{
                model_id: {
                    "directory": spec["directory"],
                    "model_revision": spec["revision"],
                    "model_index_sha256": spec["index_sha256"],
                }
                for model_id, spec in EXPECTED_MODELS.items()
            },
        },
    }
    write_json(RUN_DIR / "q1_sequential.json", result)
    print(json.dumps({
        "accuracies": {arm: value["accuracy"] for arm, value in evaluations.items()},
        "primary": primary,
        "primary_success": success,
        "Q_K1": q_k1,
        "Q_K2": q_k2,
        "claim": result["claim"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()