import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np


RUN_DIR = Path(__file__).resolve().parent
ROOT = RUN_DIR.parents[2]
PRIOR_DIR = RUN_DIR.parent / "2026-08-12"
VUS_DIR = ROOT / "runs/visual-utility-selector/2026-08-11"
sys.path.insert(0, str(RUN_DIR))
sys.path.insert(0, str(PRIOR_DIR))

from context_common import atomic_json_file, sha256_file
from finalize_trivus import CELL_ORDER, frozen_baselines, load_configs, load_public
from headroom_atlas import load_candidate_labels


OUTPUT_PATH = RUN_DIR / "SEQUENTIAL_ATLAS.json"


def load_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def load_blind_predictions(public):
    rows = load_jsonl(VUS_DIR / "zero_shot/predictions.jsonl")
    manifest_path = PRIOR_DIR / "selector/BLIND_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text())
    android_rows = []
    for shard, item in manifest["shards"].items():
        path = PRIOR_DIR / item["path"]
        if sha256_file(path) != item["sha256"]:
            raise PermissionError(f"Android blind prediction hash mismatch: {shard}")
        incoming = load_jsonl(path)
        if len(incoming) != item["rows"]:
            raise ValueError(f"Android blind prediction count mismatch: {shard}")
        android_rows.extend(incoming)
    rows.extend(android_rows)
    predictions = {row["sample_key"]: row for row in rows}
    if len(predictions) != len(rows) or set(predictions) != set(public):
        raise ValueError("Blind prediction public coverage mismatch")
    return predictions, {
        "vus_predictions_sha256": sha256_file(
            VUS_DIR / "zero_shot/predictions.jsonl"
        ),
        "android_blind_manifest_sha256": sha256_file(manifest_path),
    }


def candidate_scores(prediction, count):
    permutation = [int(value) for value in prediction["display_to_candidate"]]
    scores = np.asarray(prediction["label_probabilities"], dtype=np.float64)
    if (
        sorted(permutation) != list(range(count))
        or scores.shape != (count,)
        or not np.isfinite(scores).all()
        or np.any(scores < 0)
        or not np.isclose(scores.sum(), 1.0, atol=1e-6, rtol=0)
    ):
        raise ValueError(f"Blind candidate-score mismatch: {prediction.get('sample_key')}")
    restored = np.empty(count, dtype=np.float64)
    for display_index, candidate_index in enumerate(permutation):
        restored[candidate_index] = scores[display_index]
    return restored


def ranked_success(labels, scores):
    labels = np.asarray(labels, dtype=np.bool_)
    scores = np.asarray(scores, dtype=np.float64)
    if labels.ndim != 1 or scores.shape != labels.shape or not np.isfinite(scores).all():
        raise ValueError("Sequential ranked-success input mismatch")
    order = np.argsort(-scores, kind="stable")
    return labels[order]


def budget_curve(rows, maximum_candidates):
    if not rows or any(len(values) != maximum_candidates for values in rows):
        raise ValueError("Sequential budget rows mismatch")
    budgets = tuple(range(1, maximum_candidates + 1))
    hits = {
        budget: np.asarray([bool(np.any(values[:budget])) for values in rows])
        for budget in budgets
    }
    oracle = hits[maximum_candidates]
    oracle_rate = float(oracle.mean())
    recovery = {
        budget: float(hits[budget].sum() / oracle.sum()) if oracle.any() else 1.0
        for budget in budgets
    }
    first = []
    for values in rows:
        positions = np.flatnonzero(values)
        first.append(int(positions[0] + 1) if len(positions) else None)
    target_budget = next((budget for budget in budgets if recovery[budget] >= 0.9), maximum_candidates)
    return {
        "rows": len(rows),
        "oracle_rate": oracle_rate,
        "hit_at_k": {str(budget): float(hits[budget].mean()) for budget in budgets},
        "oracle_recovery_at_k": {str(budget): recovery[budget] for budget in budgets},
        "marginal_gain_at_k": {
            str(budget): float(
                hits[budget].mean() - (hits[budget - 1].mean() if budget > 1 else 0.0)
            )
            for budget in budgets
        },
        "first_success_rank_counts": {
            str(key): value for key, value in sorted(
                Counter("none" if value is None else value for value in first).items(),
                key=lambda item: (item[0] == "none", str(item[0])),
            )
        },
        "mean_first_success_rank_given_success": float(np.mean([
            value for value in first if value is not None
        ])) if oracle.any() else None,
        "minimum_budget_for_90_percent_oracle_recovery": target_budget,
    }


def build_atlas(public, labels, predictions, strongest):
    output = {}
    for family, cell in CELL_ORDER:
        keys = sorted(
            key for key, row in public.items()
            if row["benchmark"] == family
            and (row["setting"] if family == "androidcontrol" else row["arm"]) == cell
        )
        count = 3 if family == "androidcontrol" else 12
        ranked = [
            ranked_success(labels[key], candidate_scores(predictions[key], count))
            for key in keys
        ]
        curve = budget_curve(ranked, count)
        curve["strongest_accuracy"] = float(np.mean([strongest[key] for key in keys]))
        curve["full_oracle_minus_strongest"] = (
            curve["oracle_rate"] - curve["strongest_accuracy"]
        )
        output[f"{family}/{cell}"] = curve
    return output


def main():
    public = load_public()
    labels, label_manifests = load_candidate_labels(public)
    predictions, prediction_manifests = load_blind_predictions(public)
    _, training_config = load_configs()
    _, strongest = frozen_baselines(public, training_config)
    result = {
        "schema_version": 1,
        "status": "PASS_POSTHOC_SEQUENTIAL_BUDGET_ATLAS",
        "outcome": "EXPLORATORY_ONLY_NO_PROMOTION",
        "ordering": "frozen_blind_visual_probability_descending",
        "label_manifests": label_manifests,
        "prediction_manifests": prediction_manifests,
        "cells": build_atlas(public, labels, predictions, strongest),
        "interpretation": {
            "hit_at_k_uses_labels_only_for_evaluation": True,
            "runtime_success_is_unobserved": True,
            "deployable_stop_requires_calibrated_verifier": True,
            "promotion_allowed": False,
        },
    }
    if OUTPUT_PATH.exists():
        raise FileExistsError(OUTPUT_PATH)
    atomic_json_file(OUTPUT_PATH, result)
    print(json.dumps({
        cell: {
            "oracle": value["oracle_rate"],
            "hit_at_1": value["hit_at_k"]["1"],
            "budget_90": value["minimum_budget_for_90_percent_oracle_recovery"],
            "mean_first_success_rank": value["mean_first_success_rank_given_success"],
        }
        for cell, value in result["cells"].items()
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()