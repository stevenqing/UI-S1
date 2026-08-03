import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.stats import chisquare


RUN_DIR = Path(__file__).resolve().parent
ROOT = RUN_DIR.parents[2]
CALA = ROOT / "runs/cala/2026-08-03"
H1 = ROOT / "runs/ccm-h2h/2026-07-31/h1"
ALLOCATION = ROOT / "runs/allocation-law/2026-08-01"
sys.path.insert(0, str(CALA))
sys.path.insert(0, str(H1))
sys.path.insert(0, str(ALLOCATION))
from cala_common import UNIFORM_SEQUENCE, load_bank
from cala_transfer_72b import UNIFORM_N8, load_context as load_72
from aggregators_coord import official_groups
from allocation_eval import group_folds, point_in_bbox
from run_l2 import stratified_group_sample_counts


SEED = 20260803
RESAMPLES = 10000


def rows_7b():
    context = load_bank()
    rows = []
    for row_id in context["row_ids"]:
        rows.append({"id": row_id, "application": context["metadata"][row_id]["application"], "img_size": context["metadata"][row_id]["img_size"], "target_bbox": context["metadata"][row_id]["target_bbox"], "candidates": [context["bank"][action][row_id] for action in UNIFORM_SEQUENCE[:8]]})
    return rows


def rows_72b():
    context = load_72()
    rows = []
    for row_id in context["row_ids"]:
        rows.append({"id": row_id, "application": context["metadata"][row_id]["application"], "img_size": context["metadata"][row_id]["img_size"], "target_bbox": context["metadata"][row_id]["target_bbox"], "candidates": [context["bank"][action][row_id] for action in UNIFORM_N8]})
    return rows


def winner_group(candidates):
    points = [candidate["point"] for candidate in candidates]
    groups = official_groups(points)
    scored = []
    for group_index, group in enumerate(groups):
        coverage = sum(candidates[index].get("coverage", 0) for index in group) / len(group)
        scored.append((len(group) + coverage / 1000, -group_index, group))
    winner = max(scored)[2]
    selected = max(winner, key=lambda index: (candidates[index].get("coverage", 0), -index))
    return winner, selected


def row_diagnostics(row):
    candidates = row["candidates"]
    bbox = row["target_bbox"]
    width, height = row["img_size"]
    diagonal = math.hypot(width, height)
    failures = [index for index, candidate in enumerate(candidates) if not point_in_bbox(candidate["point"], bbox)]
    distances = [math.dist(candidates[left]["point"], candidates[right]["point"]) / diagonal for left_index, left in enumerate(failures) for right in failures[left_index + 1:]]
    if len(failures) >= 2:
        failed_groups = official_groups([candidates[index]["point"] for index in failures])
        largest_failed_fraction = len(failed_groups[0]) / len(failures)
    elif failures:
        largest_failed_fraction = 1.0
    else:
        largest_failed_fraction = 0.0
    winner, selected = winner_group(candidates)
    b3_correct = bool(point_in_bbox(candidates[selected]["point"], bbox))
    return {
        "id": row["id"], "application": row["application"], "failed_candidates": len(failures),
        "mean_failed_pair_distance": float(np.mean(distances)) if distances else None,
        "median_failed_pair_distance": float(np.median(distances)) if distances else None,
        "largest_failed_cluster_fraction": largest_failed_fraction,
        "B3_correct": b3_correct,
        "winner_group_size": len(winner),
        "winner_group_models": [candidates[index]["model"] for index in winner],
        "selected_model": candidates[selected]["model"],
    }


def summarize(records):
    distance = [record["mean_failed_pair_distance"] for record in records if record["mean_failed_pair_distance"] is not None]
    wrong = [record for record in records if not record["B3_correct"]]
    group_models = Counter(model for record in wrong for model in record["winner_group_models"])
    selected_models = Counter(record["selected_model"] for record in wrong)
    observed = np.asarray(list(group_models.values()), dtype=np.float64)
    test = chisquare(observed) if len(observed) > 1 and observed.sum() > 0 else None
    return {
        "rows": len(records),
        "rows_with_failed_pairs": len(distance),
        "mean_failed_pair_distance": float(np.mean(distance)),
        "median_row_mean_failed_pair_distance": float(np.median(distance)),
        "mean_largest_failed_cluster_fraction": float(np.mean([record["largest_failed_cluster_fraction"] for record in records])),
        "B3_accuracy": sum(record["B3_correct"] for record in records) / len(records),
        "B3_wrong_rows": len(wrong),
        "wrong_winner_group_model_members": dict(sorted(group_models.items())),
        "wrong_selected_model": dict(sorted(selected_models.items())),
        "winner_group_model_uniformity": None if test is None else {"chi_square": float(test.statistic), "p_value": float(test.pvalue)},
    }


def paired_bootstrap(rows, left, right):
    mapping, fold_rows = group_folds(rows)
    groups = sorted(mapping); group_index = {group: index for index, group in enumerate(groups)}
    counts = np.zeros(len(groups), dtype=np.int64); sums = np.zeros(len(groups), dtype=np.float64)
    for row in rows:
        row_id = row["id"]
        if left[row_id] is None or right[row_id] is None:
            continue
        index = group_index[row["application"]]; counts[index] += 1; sums[index] += left[row_id] - right[row_id]
    samples = stratified_group_sample_counts(groups, mapping, RESAMPLES, np.random.default_rng(SEED)); denominators = samples @ counts
    if np.any(denominators == 0): raise ValueError("R5 bootstrap empty replicate")
    values = (samples @ sums) / denominators
    point = sums.sum() / counts.sum()
    return {"point_delta_72B_minus_7B": point, "ci_99": [float(np.quantile(values,.005)),float(np.quantile(values,.995))], "bootstrap_mean": float(values.mean()), "p_delta_nonnegative": float((1+np.sum(values>=0))/(RESAMPLES+1)), "resamples": RESAMPLES, "seed": SEED, "paired_rows": int(counts.sum()), "fold_rows": fold_rows}


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--output", type=Path, required=True); args = parser.parse_args()
    source7 = rows_7b(); source72 = rows_72b()
    if [row["id"] for row in source7] != [row["id"] for row in source72]: raise ValueError("R5 identity mismatch")
    records7 = [row_diagnostics(row) for row in source7]; records72 = [row_diagnostics(row) for row in source72]
    by7 = {record["id"]: record["mean_failed_pair_distance"] for record in records7}; by72 = {record["id"]: record["mean_failed_pair_distance"] for record in records72}
    bootstrap = paired_bootstrap(source7, by72, by7)
    summaries = {"7B_Uniform_Mixed_N8": summarize(records7), "72B_Uniform_Mixed_N8": summarize(records72)}
    composition = summaries["72B_Uniform_Mixed_N8"]["winner_group_model_uniformity"]
    tighter = bool(bootstrap["point_delta_72B_minus_7B"] < 0)
    nonuniform = bool(composition is not None and composition["p_value"] < .01)
    result = {
        "schema_version": 1, "status": "PASS", "rows": 1581, "budget": 8,
        "summaries": summaries, "failed_distance_bootstrap": bootstrap,
        "hypothesis": {"72B_failed_candidates_tighter_point_estimate": tighter, "72B_wrong_winner_group_nonuniform_p_lt_0_01": nonuniform, "pollution_hypothesis_supported": tighter and nonuniform},
        "records_sha256": {"7B": __import__('hashlib').sha256(json.dumps(records7, sort_keys=True, separators=(',',':')).encode()).hexdigest(), "72B": __import__('hashlib').sha256(json.dumps(records72, sort_keys=True, separators=(',',':')).encode()).hexdigest()},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"summaries": summaries, "bootstrap": bootstrap, "hypothesis": result["hypothesis"]}, indent=2, sort_keys=True))


if __name__ == "__main__": main()