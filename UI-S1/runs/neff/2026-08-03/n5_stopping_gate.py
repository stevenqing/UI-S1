import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np


RUN_DIR = Path(__file__).resolve().parent
ROOT = RUN_DIR.parents[2]
DIVERSITY = ROOT / "runs/diversity-axis/2026-08-02"
ALLOCATION = ROOT / "runs/allocation-law/2026-08-01"
sys.path.insert(0, str(DIVERSITY))
sys.path.insert(0, str(ALLOCATION))
from x3_curve_stats import load_sources
from x7_safeground_port import compute_uncertainty
from allocation_eval import build_pool, compact_evaluation, group_folds
from run_l2 import stratified_group_sample_counts


SEED = 20260803
RESAMPLES = 10000
BUDGETS = (4, 8, 12)


def sha256_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def bootstrap_high_increment(all_rows, high_ids, pass4, pass12):
    mapping, fold_rows = group_folds(all_rows)
    groups = sorted(mapping)
    group_index = {group: index for index, group in enumerate(groups)}
    row_counts = np.zeros(len(groups), dtype=np.int64)
    deltas = np.zeros(len(groups), dtype=np.int64)
    by_id = {row["id"]: row for row in all_rows}
    for row_id in high_ids:
        index = group_index[by_id[row_id]["application"]]
        row_counts[index] += 1
        deltas[index] += int(pass12[row_id]) - int(pass4[row_id])
    sample_counts = stratified_group_sample_counts(groups, mapping, RESAMPLES, np.random.default_rng(SEED))
    denominators = sample_counts @ row_counts
    if np.any(denominators == 0):
        raise ValueError("N5 bootstrap empty highest-disagreement replicate")
    values = (sample_counts @ deltas) / denominators
    point = (sum(pass12[row_id] for row_id in high_ids) - sum(pass4[row_id] for row_id in high_ids)) / len(high_ids)
    return {
        "point_delta": point,
        "ci_99": [float(np.quantile(values, .005)), float(np.quantile(values, .995))],
        "bootstrap_mean": float(np.mean(values)),
        "p_delta_nonpositive": float((1 + np.sum(values <= 0)) / (RESAMPLES + 1)),
        "resamples": RESAMPLES,
        "seed": SEED,
        "groups": len(groups),
        "fold_rows": fold_rows,
    }


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--output", type=Path, required=True); args = parser.parse_args()
    gta1, generated, units = load_sources()
    actions = [tuple(unit.rsplit("/view", 1)) for unit in units[12]]
    actions = [(model, int(view)) for model, view in actions]
    rows_by_budget = {budget: build_pool(gta1, generated, actions[:budget]) for budget in BUDGETS}
    evaluations = {budget: compact_evaluation(rows) for budget, rows in rows_by_budget.items()}
    rows12 = rows_by_budget[12]
    uncertainty = {}
    for row in rows12:
        width, height = gta1[row["id"]]["img_size"]
        points = [candidate["point"] for candidate in row["candidates"]]
        uncertainty[row["id"]] = compute_uncertainty(points, width, height, patch_size=28, activation_threshold=0.0)["combined"]
    ordered = sorted(rows12, key=lambda row: (uncertainty[row["id"]], row["id"]))
    bins = np.array_split(np.asarray(ordered, dtype=object), 5)
    records = []
    for index, members in enumerate(bins):
        ids = [row["id"] for row in members]
        records.append({
            "bin": index,
            "label": "highest_disagreement" if index == 4 else "ordered_uncertainty_quintile",
            "rows": len(ids),
            "uncertainty_min": min(uncertainty[row_id] for row_id in ids),
            "uncertainty_max": max(uncertainty[row_id] for row_id in ids),
            "pass_at_n": {
                str(budget): sum(evaluations[budget]["outputs"]["pass_at_n"][row_id] for row_id in ids) / len(ids)
                for budget in BUDGETS
            },
        })
    high_ids = [row["id"] for row in bins[-1]]
    bootstrap = bootstrap_high_increment(
        rows12, high_ids,
        evaluations[4]["outputs"]["pass_at_n"], evaluations[12]["outputs"]["pass_at_n"],
    )
    result = {
        "schema_version": 1, "status": "PASS", "rows": 1581,
        "score": {"method": "SafeGround_official_code_transfer", "patch_size": 28, "activation_threshold": 0.0, "higher_is_more_disagreement": True},
        "bins": records,
        "highest_disagreement_increment": bootstrap,
        "gate": {"point_increment_positive": bootstrap["point_delta"] > 0, "NOA_stop_mode": "accuracy_plus_compute_saving" if bootstrap["point_delta"] > 0 else "pure_compute_saving"},
        "sources": {"X7_sha256": sha256_file(DIVERSITY / "x7_confidence.json"), "L1_sha256": sha256_file(ALLOCATION / "L1_RESULTS.json")},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(result, indent=2, sort_keys=True)+"\n")
    print(json.dumps({"bins": records, "highest_increment": bootstrap, "gate": result["gate"]}, indent=2, sort_keys=True))


if __name__ == "__main__": main()