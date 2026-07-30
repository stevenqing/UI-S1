import argparse
import json
import math
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.stats import kendalltau, mannwhitneyu

from build_rows import AC_MODELS, ac_cross_setting_map
from common import load_rows, pivot_rows
from scoring import GROUNDING_ACTIONS, read_jsonl


def confusion_for_wait(setting):
    rows = load_rows("androidcontrol", setting)
    identities, models, pivot = pivot_rows(rows)
    output = {}
    for model in models:
        gt_wait = [pivot[row_id][model] for row_id in identities if pivot[row_id][model]["gt_action"] == "wait"]
        pred_wait = [pivot[row_id][model] for row_id in identities if pivot[row_id][model]["pred_action"] == "wait"]
        output[model] = {
            "gt_wait_rows": len(gt_wait),
            "gt_wait_pred_action_counts": dict(sorted(Counter(row["pred_action"] or "<parse_failure>" for row in gt_wait).items())),
            "pred_wait_rows": len(pred_wait),
            "pred_wait_precision": sum(row["gt_action"] == "wait" for row in pred_wait) / len(pred_wait) if pred_wait else None,
            "pred_wait_base_rate": len(pred_wait) / len(identities),
            "gt_wait_base_rate": len(gt_wait) / len(identities),
        }
    return output


def cohen_kappa(left, right):
    left = np.asarray(left, dtype=np.int8)
    right = np.asarray(right, dtype=np.int8)
    observed = np.mean(left == right)
    left_rate, right_rate = left.mean(), right.mean()
    expected = left_rate * right_rate + (1 - left_rate) * (1 - right_rate)
    return float((observed - expected) / (1 - expected)) if expected < 1 else 1.0


def chance_corrected_pairs(bench, setting, permutations=1000):
    rows = load_rows(bench, setting)
    identities, models, pivot = pivot_rows(rows)
    rng = np.random.default_rng(20260730)
    output = []
    for left, right in combinations(models, 2):
        left_failure = np.asarray([not pivot[row_id][left]["success"] for row_id in identities], dtype=np.int8)
        right_failure = np.asarray([not pivot[row_id][right]["success"] for row_id in identities], dtype=np.int8)
        observed = cohen_kappa(left_failure, right_failure)
        null = np.asarray([cohen_kappa(left_failure, rng.permutation(right_failure)) for _ in range(permutations)])
        output.append({
            "left": left, "right": right, "observed_kappa": observed,
            "null_mean": float(null.mean()), "null_sd": float(null.std()),
            "p_greater_equal": float((1 + np.count_nonzero(null >= observed)) / (permutations + 1)),
            "permutations": permutations,
        })
    return output


def threshold_sensitivity(setting):
    rows = load_rows("androidcontrol", setting)
    identities, models, pivot = pivot_rows(rows)
    thresholds = [round(value, 2) for value in np.arange(0.06, 0.301, 0.02)]
    curves = {model: [] for model in models}
    oracle = []
    rankings = []
    for threshold in thresholds:
        success_by_model = {}
        for model in models:
            successes = []
            for row_id in identities:
                row = pivot[row_id][model]
                if row["gt_action"] in GROUNDING_ACTIONS:
                    success = row["pred_action"] == row["gt_action"] and row["ground_dist"] < threshold
                else:
                    success = row["success"]
                successes.append(bool(success))
            success_by_model[model] = successes
            curves[model].append(sum(successes) / len(successes))
        oracle.append(sum(any(success_by_model[model][index] for model in models) for index in range(len(identities))) / len(identities))
        rankings.append(sorted(models, key=lambda model: (-curves[model][-1], model)))
    base_index = thresholds.index(0.14)
    base_rank = {model: rank for rank, model in enumerate(rankings[base_index])}
    rank_tau = {}
    for index, threshold in enumerate(thresholds):
        current = {model: rank for rank, model in enumerate(rankings[index])}
        tau = kendalltau([base_rank[m] for m in models], [current[m] for m in models]).statistic
        rank_tau[str(threshold)] = float(tau)
    return {
        "thresholds": thresholds, "model_step_sr": curves, "oracle_step_sr": oracle,
        "rankings": {str(threshold): ranking for threshold, ranking in zip(thresholds, rankings)},
        "kendall_tau_vs_0.14_ranking": rank_tau,
    }


def high_only_diagnostic(root: Path):
    artifact_root = root / "runs/androidcontrol-rft/2026-07-29/artifacts"
    source_low = read_jsonl(artifact_root / "ui-agile-3b/low/predictions.jsonl")
    source_high = read_jsonl(artifact_root / "ui-agile-3b/high/predictions.jsonl")
    mapping, conflict_low, _ = ac_cross_setting_map(source_low, source_high)
    output = {}
    for model in AC_MODELS:
        low_rows = read_jsonl(artifact_root / model / "low/predictions.jsonl")
        high_rows = read_jsonl(artifact_root / model / "high/predictions.jsonl")
        tidy_low = {int(row["row_id"]): row for row in load_rows("androidcontrol", "low", include_quarantine=True) if row["model"] == model}
        tidy_high = {int(row["row_id"]): row for row in load_rows("androidcontrol", "high", include_quarantine=True) if row["model"] == model}
        high_only_low_indices = [
            low_index for low_index, high_index in mapping.items()
            if low_index not in conflict_low and not tidy_low[low_index]["success"] and tidy_high[high_index]["success"]
        ]
        all_low_fail_indices = [index for index, row in tidy_low.items() if not row["quarantine"] and not row["success"]]
        high_only_dist = [tidy_low[index]["ground_dist"] for index in high_only_low_indices if not math.isnan(tidy_low[index]["ground_dist"])]
        all_fail_dist = [tidy_low[index]["ground_dist"] for index in all_low_fail_indices if not math.isnan(tidy_low[index]["ground_dist"])]
        low_success_rate = sum(not row["quarantine"] and row["success"] for row in tidy_low.values()) / 7650
        high_success_rate = sum(not row["quarantine"] and row["success"] for row in tidy_high.values()) / 7650
        expected = 7650 * (1 - low_success_rate) * high_success_rate
        test = mannwhitneyu(high_only_dist, all_fail_dist, alternative="two-sided") if high_only_dist and all_fail_dist else None
        output[model] = {
            "high_only_count": len(high_only_low_indices),
            "independent_matched_marginal_expected": expected,
            "observed_over_expected": len(high_only_low_indices) / expected if expected else None,
            "low_grounding_distances": {
                "available": len(high_only_dist),
                "under_0.14": sum(value < 0.14 for value in high_only_dist),
                "0.14_to_0.28": sum(0.14 <= value < 0.28 for value in high_only_dist),
                "at_least_0.28": sum(value >= 0.28 for value in high_only_dist),
                "median": float(np.median(high_only_dist)) if high_only_dist else None,
            },
            "all_low_failure_grounding_median": float(np.median(all_fail_dist)) if all_fail_dist else None,
            "mann_whitney_u": float(test.statistic) if test else None,
            "mann_whitney_p": float(test.pvalue) if test else None,
        }
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    result = {
        "status": "PASS",
        "d1_wait": {setting: confusion_for_wait(setting) for setting in ("low", "high")},
        "d2_select": {"status": "PENDING_HUMAN_LABELS", "rows": 79, "labels_shared_with_e4": True},
        "d3_chance_corrected": {
            f"{bench}/{setting}": chance_corrected_pairs(bench, setting)
            for bench, setting in (("androidcontrol", "low"), ("androidcontrol", "high"), ("mind2web", "visual"))
        },
        "d4_threshold_sensitivity": {setting: threshold_sensitivity(setting) for setting in ("low", "high")},
        "d5_high_only": high_only_diagnostic(root),
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": result["status"],
        "wait_pred_base_rate_low": {m: v["pred_wait_base_rate"] for m, v in result["d1_wait"]["low"].items()},
        "wait_pred_base_rate_high": {m: v["pred_wait_base_rate"] for m, v in result["d1_wait"]["high"].items()},
        "high_only": {m: v["high_only_count"] for m, v in result["d5_high_only"].items()},
    }, indent=2))


if __name__ == "__main__":
    main()