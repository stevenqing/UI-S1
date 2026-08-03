import hashlib
import json
import sys
from pathlib import Path

import numpy as np


RUN_DIR = Path(__file__).resolve().parent
ROOT = RUN_DIR.parents[2]
ALLOCATION = ROOT / "runs/allocation-law/2026-08-01"
DIVERSITY = ROOT / "runs/diversity-axis/2026-08-02"
sys.path.insert(0, str(ALLOCATION))
sys.path.insert(0, str(DIVERSITY))
from allocation_eval import build_pool, compact_evaluation, group_folds
from run_l2 import stratified_group_sample_counts
from x3_curve_stats import load_sources
from x7_safeground_port import compute_uncertainty


SEED = 20260803
RESAMPLES = 10000
MIXED_BUDGETS = (4, 8, 12, 16, 24)


def sha256_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_pools():
    gta1, generated, units = load_sources()
    actions = [tuple(unit.rsplit("/view", 1)) for unit in units[24]]
    actions = [(model, int(view)) for model, view in actions]
    mixed = {budget: build_pool(gta1, generated, actions[:budget]) for budget in MIXED_BUDGETS}
    v_only = build_pool(gta1, generated, [("GTA1-7B", view) for view in range(12)])
    evaluations = {"mixed": {budget: compact_evaluation(rows) for budget, rows in mixed.items()}, "v_only_N12": compact_evaluation(v_only)}
    reference_mapping, fold_rows = group_folds(mixed[12])
    for rows in [*mixed.values(), v_only]:
        mapping, _ = group_folds(rows)
        if mapping != reference_mapping:
            raise ValueError("reallocation fold mapping mismatch")
    return {"gta1": gta1, "actions": actions, "mixed": mixed, "v_only_N12": v_only, "evaluations": evaluations, "fold_for_group": reference_mapping, "fold_rows": fold_rows}


def uncertainty_scores(rows, image_sizes):
    output = {}
    for row in rows:
        width, height = image_sizes[row["id"]]
        points = [candidate["point"] for candidate in row["candidates"]]
        output[row["id"]] = compute_uncertainty(points, width, height, patch_size=28, activation_threshold=0.0)["combined"]
    return output


def ordered_bins(rows, scores, count):
    ordered = sorted(rows, key=lambda row: (scores[row["id"]], row["id"]))
    return [[row["id"] for row in values] for values in np.array_split(np.asarray(ordered, dtype=object), count)]


def subset_bootstrap(all_rows, selected_ids, left, right, fold_for_group):
    groups = sorted(fold_for_group)
    group_index = {group: index for index, group in enumerate(groups)}
    by_id = {row["id"]: row for row in all_rows}
    row_counts = np.zeros(len(groups), dtype=np.int64)
    deltas = np.zeros(len(groups), dtype=np.int64)
    for row_id in selected_ids:
        index = group_index[by_id[row_id]["application"]]
        row_counts[index] += 1
        deltas[index] += int(left[row_id]) - int(right[row_id])
    sample_counts = stratified_group_sample_counts(groups, fold_for_group, RESAMPLES, np.random.default_rng(SEED))
    denominators = sample_counts @ row_counts
    if np.any(denominators == 0):
        raise ValueError("subset bootstrap produced empty replicate")
    values = (sample_counts @ deltas) / denominators
    point = (sum(left[row_id] for row_id in selected_ids) - sum(right[row_id] for row_id in selected_ids)) / len(selected_ids)
    return {
        "rows": len(selected_ids), "point_delta": point,
        "ci_99": [float(np.quantile(values, .005)), float(np.quantile(values, .995))],
        "bootstrap_mean": float(np.mean(values)),
        "p_delta_nonpositive": float((1 + np.sum(values <= 0)) / (RESAMPLES + 1)),
        "resamples": RESAMPLES, "seed": SEED, "groups": len(groups),
    }
