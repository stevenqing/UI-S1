import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import yaml


RUN_DIR = Path(__file__).resolve().parent
ROOT = RUN_DIR.parents[2]
SOURCEBIAS_DIR = ROOT / "runs/sourcebias/2026-08-03"
sys.path.insert(0, str(RUN_DIR))
sys.path.insert(0, str(SOURCEBIAS_DIR))

from common import MODELS, evaluate_actions, load_context, paired_group_bootstrap, write_json
from sourcebias_common import b3_select_index, point_in_bbox


EXPECTED_MODELS = {
    "GTA1-7B": ("q2b-gta1", "701bedc80b447863bd60e3318ae44f6cbbfafd78", "3067e9b0f35596ff3426a0d0ec8c982a51fa1e110c4fc30dcf3be9ea37409df6"),
    "Qwen3-VL-8B-Instruct": ("q2b-qwen3", "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b", "520b2e05079402e9468a8701d03d1154d14b2599593afb6effa7fb60c1bff070"),
    "UI-TARS-7B-SFT": ("q2b-uitars", "3434901a9dd04dd3625617d839a5724fe5e2db20", "25b162a0f0f47af097d6a49b7da3d5c7d9c2b352490131c8cde5ca59d285f18b"),
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
                raise ValueError(f"Q2b duplicate verifier row: {row['id']}")
            rows[row["id"]] = row
    if len(rows) != 1581:
        raise ValueError(f"Q2b requires 1,581 rows in {path}, found {len(rows)}")
    return rows


def main():
    config_path = RUN_DIR / "configs/q2b_verification.yaml"
    config = yaml.safe_load(config_path.read_text())
    if config["status"] != "result_blind_design_freeze":
        raise ValueError("Q2b config mismatch")
    context = load_context()
    crop_rows = [json.loads(line) for line in (RUN_DIR / "raw/q2b_crops.jsonl").read_text().splitlines() if line.strip()]
    crops = {row["id"]: row for row in crop_rows}
    if len(crops) != 1581 or set(crops) != set(context["row_ids"]):
        raise ValueError("Q2b crop coverage mismatch")
    if any("target" in key or "bbox" in key for row in crop_rows for key in row):
        raise ValueError("Q2b crop target leak")

    verifier_rows = {}
    decisions = {row_id: {} for row_id in context["row_ids"]}
    parse_counts = Counter()
    for model_id, (directory, revision, index_hash) in EXPECTED_MODELS.items():
        rows = load_unique(RUN_DIR / "raw" / directory)
        verifier_rows[model_id] = rows
        if set(rows) != set(crops):
            raise ValueError(f"Q2b verifier identity mismatch: {model_id}")
        for row_id, row in rows.items():
            if row["verifier_model"] != model_id or row["model_revision"] != revision or row["model_index_sha256"] != index_hash:
                raise ValueError(f"Q2b verifier provenance mismatch: {model_id}/{row_id}")
            if row["checks_sha256"] != crops[row_id]["checks_sha256"] or canonical_hash(row["checks"]) != row["outputs_sha256"]:
                raise ValueError(f"Q2b verifier hash mismatch: {model_id}/{row_id}")
            for value in row["checks"]:
                index = value["check_index"]
                source_check = crops[row_id]["checks"][index]
                expected = {
                    "candidate_model": source_check["candidate_model"],
                    "candidate_view": source_check["candidate_view"],
                    "verification_crop": source_check["verification_crop"],
                }
                actual = {key: value[key] for key in expected}
                if actual != expected or source_check["verifier_model"] != model_id:
                    raise ValueError(f"Q2b check provenance mismatch: {model_id}/{row_id}/{index}")
                if index in decisions[row_id]:
                    raise ValueError(f"Q2b duplicate check decision: {row_id}/{index}")
                decisions[row_id][index] = value
                parse_counts["parsed" if value["parse_ok"] else "failed"] += 1
    if any(set(values) != set(range(6)) for values in decisions.values()):
        raise ValueError("Q2b check coverage mismatch")

    stage1_actions = tuple((model, view) for view in range(2) for model in MODELS)
    baseline = evaluate_actions(context, tuple((model, view) for view in range(4) for model in MODELS))
    stage1_rows = []
    verified_outputs = {}
    random_outputs = {}
    labels = []
    predicted = []
    by_verifier = {model: {"labels": [], "predicted": []} for model in MODELS}
    rng = np.random.default_rng(config["evaluation"]["random_reference_seed"])
    for row_id in context["row_ids"]:
        metadata = context["metadata"][row_id]
        candidates = [context["bank"][action][row_id] for action in stage1_actions]
        target_bbox = metadata["target_bbox"]
        center = ((target_bbox[0] + target_bbox[2]) / 2, (target_bbox[1] + target_bbox[3]) / 2)
        row_labels = []
        row_predictions = []
        for check in crops[row_id]["checks"]:
            left, top, right, bottom = check["verification_crop"]
            label = left <= center[0] <= right and top <= center[1] <= bottom
            value = decisions[row_id][check["check_index"]]
            decision = bool(value["decision"]) if value["parse_ok"] else False
            row_labels.append(label)
            row_predictions.append(decision)
            labels.append(label)
            predicted.append(decision)
            verifier = check["verifier_model"]
            by_verifier[verifier]["labels"].append(label)
            by_verifier[verifier]["predicted"].append(decision)
        positive = [index for index, decision in enumerate(row_predictions) if decision]
        selected_pool = positive if positive else list(range(6))
        selected, _ = b3_select_index([candidates[index] for index in selected_pool])
        chosen = candidates[selected_pool[selected]]
        verified_outputs[row_id] = bool(point_in_bbox(chosen["point"], target_bbox))

        random_positive = [index for index, decision in enumerate(rng.random(6) < 0.5) if decision]
        random_pool = random_positive if random_positive else list(range(6))
        random_selected, _ = b3_select_index([candidates[index] for index in random_pool])
        random_chosen = candidates[random_pool[random_selected]]
        random_outputs[row_id] = bool(point_in_bbox(random_chosen["point"], target_bbox))
        stage1_rows.append({"id": row_id, "application": metadata["application"], "outer_fold": context["fold_for_group"][metadata["application"]]})

    def binary_report(targets, outputs):
        tp = sum(left and right for left, right in zip(targets, outputs))
        tn = sum((not left) and (not right) for left, right in zip(targets, outputs))
        fp = sum((not left) and right for left, right in zip(targets, outputs))
        fn = sum(left and (not right) for left, right in zip(targets, outputs))
        return {
            "rows": len(targets), "tp": tp, "tn": tn, "fp": fp, "fn": fn,
            "accuracy": (tp + tn) / len(targets),
            "yes_precision": tp / (tp + fp) if tp + fp else None,
            "yes_recall": tp / (tp + fn) if tp + fn else None,
            "positive_rate": sum(outputs) / len(outputs),
            "label_positive_rate": sum(targets) / len(targets),
        }

    metadata = {row["id"]: {"application": row["application"], "outer_fold": row["outer_fold"]} for row in stage1_rows}
    comparison = {
        **paired_group_bootstrap(metadata, verified_outputs, baseline["outputs"]["B3_mvp"]),
        "verified_accuracy": sum(verified_outputs.values()) / len(verified_outputs),
        "baseline_accuracy": baseline["accuracy"]["B3_mvp"],
    }
    random_comparison = {
        **paired_group_bootstrap(metadata, verified_outputs, random_outputs),
        "verified_accuracy": sum(verified_outputs.values()) / len(verified_outputs),
        "random_reference_accuracy": sum(random_outputs.values()) / len(random_outputs),
    }
    success = comparison["point_delta"] > config["primary"]["mde"] and comparison["ci_99"][0] > 0
    verification = binary_report(labels, predicted)
    result = {
        "schema_version": 1,
        "status": "PASS",
        "config": str(config_path.relative_to(ROOT)),
        "rows": 1581,
        "verification_checks": len(labels),
        "parse_counts": dict(parse_counts),
        "verification": verification,
        "verification_by_model": {
            model: binary_report(values["labels"], values["predicted"])
            for model, values in by_verifier.items()
        },
        "verified_B3_vs_Uniform_N12": comparison,
        "verified_B3_vs_random_50_percent": random_comparison,
        "primary_success": success,
        "Q_K3": verification["accuracy"] <= 0.5,
        "claim": "CROSS_LINEAGE_VERIFICATION_SUPPORTED" if success else "CROSS_LINEAGE_VERIFICATION_NOT_SUPPORTED",
        "sources": {
            "crops_sha256": hashlib.sha256((RUN_DIR / "raw/q2b_crops.jsonl").read_bytes()).hexdigest(),
            **{
                model: {"directory": values[0], "revision": values[1], "index_sha256": values[2]}
                for model, values in EXPECTED_MODELS.items()
            },
        },
    }
    write_json(RUN_DIR / "q2b_verification.json", result)
    print(json.dumps({
        "verification": verification,
        "comparison": comparison,
        "random_comparison": random_comparison,
        "primary_success": success,
        "Q_K3": result["Q_K3"],
        "claim": result["claim"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()