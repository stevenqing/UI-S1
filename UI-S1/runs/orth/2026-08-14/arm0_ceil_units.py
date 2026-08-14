import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score


RUN_DIR = Path(__file__).resolve().parent
ROOT = RUN_DIR.parents[2]
CEIL_DIR = ROOT / "runs/ceil/2026-08-14"
INPUT_PATH = CEIL_DIR / "ARM_B_ROWS.jsonl"
CEIL_RESULT_PATH = CEIL_DIR / "ARM_B.json"
OUTPUT_PATH = RUN_DIR / "ARM0.json"
sys.path.insert(0, str(CEIL_DIR))

from arm_b import grouped_auc


SEEDS = {"screenspot_pro": 20261014, "mind2web": 20261114}


def candidate_arrays(rows):
    scores = []
    labels = []
    group_keys = []
    row_keys = []
    folds = {}
    row_folds = {}
    for row in rows:
        labels_array = np.asarray(row["candidate_labels"], dtype=np.bool_)
        group = (int(row["fold"]), str(row["group"]))
        base_row = (int(row["fold"]), row["sample_key"].split("/", 2)[2])
        for probabilities in row["cheap_probabilities_by_context"]:
            scores.extend(map(float, probabilities))
            labels.extend(labels_array.tolist())
            group_keys.extend([group] * len(labels_array))
            row_keys.extend([base_row] * len(labels_array))
        folds[group] = group[0]
        row_folds[base_row] = base_row[0]
    return scores, labels, group_keys, row_keys, folds, row_folds


def unclustered_bootstrap_auc(scores, labels, resamples, seed, batch_size=20):
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.bool_)
    if len(scores) != len(labels) or not labels.any() or labels.all():
        raise ValueError("ORTH Arm 0 invalid unclustered AUROC input")
    order = np.argsort(scores, kind="stable")
    ordered_scores = scores[order]
    ordered_labels = labels[order]
    starts = np.empty(len(scores), dtype=np.int64)
    ends = np.empty(len(scores), dtype=np.int64)
    boundaries = np.r_[0, np.flatnonzero(np.diff(ordered_scores) != 0) + 1, len(scores)]
    for left, right in zip(boundaries[:-1], boundaries[1:]):
        starts[left:right] = left
        ends[left:right] = right - 1
    generator = np.random.default_rng(seed)
    probabilities = np.full(len(scores), 1 / len(scores), dtype=np.float64)
    collected = []
    draws = 0
    discarded = 0
    while len(collected) < resamples:
        if draws >= 10 * resamples:
            raise ValueError("ORTH Arm 0 excessive single-class replicates")
        count = min(batch_size, resamples - len(collected))
        weights = generator.multinomial(len(scores), probabilities, size=count)[:, order]
        draws += count
        positive = weights * ordered_labels[None, :]
        negative = weights * (~ordered_labels)[None, :]
        cumulative = np.cumsum(negative, axis=1)
        before = cumulative[:, starts] - negative[:, starts]
        tied = cumulative[:, ends] - before
        numerator = np.sum(positive * (before + 0.5 * tied), axis=1)
        denominator = positive.sum(axis=1) * negative.sum(axis=1)
        valid = denominator > 0
        discarded += int((~valid).sum())
        collected.extend((numerator[valid] / denominator[valid]).tolist())
    values = np.asarray(collected[:resamples], dtype=np.float64)
    return {
        "point": float(roc_auc_score(labels, scores)),
        "ci_99": [float(np.quantile(values, 0.005)), float(np.quantile(values, 0.995))],
        "resamples": resamples,
        "discarded_single_class_replicates": discarded,
        "seed": seed,
        "unit": "context_candidate_pair_IID",
    }


def arm_name(sample_key):
    return sample_key.split("/", 2)[1]


def base_row_id(sample_key):
    return sample_key.split("/", 2)[2]


def main():
    if OUTPUT_PATH.exists():
        raise FileExistsError(OUTPUT_PATH)
    ceil = json.loads(CEIL_RESULT_PATH.read_text())
    all_rows = [json.loads(line) for line in INPUT_PATH.read_text().splitlines() if line.strip()]
    reports = {}
    for family in ("mind2web", "screenspot_pro"):
        rows = [row for row in all_rows if row["family"] == family]
        scores, labels, groups, row_groups, folds, row_folds = candidate_arrays(rows)
        seed = SEEDS[family]
        group_report, _, _ = grouped_auc(scores, labels, groups, folds, 10000, seed)
        row_report, _, _ = grouped_auc(scores, labels, row_groups, row_folds, 10000, seed + 1000)
        iid_report = unclustered_bootstrap_auc(scores, labels, 10000, seed + 2000)
        original = ceil["reports"][family]["cheap_candidate_AUROC"]
        if group_report != original:
            raise ValueError(f"ORTH Arm 0 CEIL grouped reproduction mismatch: {family}")
        arm_counts = Counter(arm_name(row["sample_key"]) for row in rows)
        base_rows_by_arm = {
            arm: len({base_row_id(row["sample_key"]) for row in rows if arm_name(row["sample_key"]) == arm})
            for arm in sorted(arm_counts)
        }
        unique_base_rows = len({base_row_id(row["sample_key"]) for row in rows})
        report = {
            "family": family,
            "arm_expanded_family_samples": int(ceil["reports"][family]["family_samples"]),
            "arm_expanded_recoverable_sample_keys": len(rows),
            "recoverable_sample_keys_by_arm": dict(sorted(arm_counts.items())),
            "recoverable_base_rows_by_arm": base_rows_by_arm,
            "recoverable_unique_base_row_union": unique_base_rows,
            "context_candidate_pairs": len(scores),
            "positive_context_candidates": int(sum(labels)),
            "negative_context_candidates": len(labels) - int(sum(labels)),
            "group_clustered": group_report,
            "row_clustered": row_report,
            "unclustered_context_candidate": iid_report,
            "lower_bound_margin_over_0_65": {
                "group_clustered": group_report["ci_99"][0] - 0.65,
                "row_clustered": row_report["ci_99"][0] - 0.65,
                "unclustered_context_candidate": iid_report["ci_99"][0] - 0.65,
            },
            "ceil_decision_unchanged_automatically": True,
        }
        reports[family] = report
    result = {
        "schema_version": 1,
        "status": "PASS_ORTH_ARM0_COMPLETE",
        "reports": reports,
        "fact_correction": {
            "ceil_recoverable_is_unique_sample_key_not_candidate_count": True,
            "ceil_primary_CI_was_already_group_clustered": True,
            "base_row_and_arm_expanded_sample_are_distinct_units": True,
        },
        "claim_boundary": {
            "descriptive_scoping_only": True,
            "automatic_C_D2_change": False,
        },
    }
    OUTPUT_PATH.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()