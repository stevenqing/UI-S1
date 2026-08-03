import argparse
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.stats import binomtest
from sklearn.metrics import roc_auc_score

from aggregators import pka_medoid, pka_medoid_leave_one_out, plurality_then_density
from ccm import collision_calibrated_mode, fit_calibration, score_candidates
from w1_run import (
    POOLS,
    deployable_models,
    dev_priority,
    fold_map,
    load_pool,
    prediction_from_row,
    score_prediction,
    split_rows,
)


BASELINES = ("A0_heldout_best", "A2_sequential_density", "A3_joint_original", "A5a_LOO")
VARIANTS = ("A5b_MAP_pooled_LR", "A5c_MAP_nine_LR", "A5d_MAP_nine_LR_family", "A5d_risk")
POOLWISE_BEST = {
    "androidcontrol/low": "A0_heldout_best",
    "androidcontrol/high": "A0_heldout_best",
    "mind2web/visual": "A3_joint_original",
}


def predictions_for(models, model_rows):
    return [prediction_from_row(model_rows[model]) for model in models]


def calibration_rows(identities, models, pivot):
    output = []
    for row_id in identities:
        reference = next(iter(pivot[row_id].values()))
        predictions = predictions_for(models, pivot[row_id])
        output.append((predictions, [score_prediction(reference, prediction) for prediction in predictions]))
    return output


def ccm_decision(calibration, predictions, best_source, family_dedup=False):
    scores, backoffs = score_candidates(calibration, predictions, family_dedup)
    if not scores:
        return None, float("-inf"), float("inf"), backoffs
    winner = max(range(len(scores)), key=lambda index: (scores[index][0], -index))
    winner_score, _, prediction = scores[winner]
    baseline_scores = [score for score, _, candidate in scores if candidate.source == best_source]
    gap = winner_score - baseline_scores[0] if baseline_scores else float("inf")
    return prediction, winner_score, gap, backoffs


def aggregate_baselines(bench, predictions, priority):
    by_source = {prediction.source: prediction for prediction in predictions}
    return {
        "A0_heldout_best": by_source[priority[0]],
        "A2_sequential_density": plurality_then_density(bench, predictions, priority).prediction,
        "A3_joint_original": pka_medoid(bench, predictions).prediction,
        "A5a_LOO": pka_medoid_leave_one_out(bench, predictions).prediction,
    }


def choose_risk_threshold(bench, calibration, identities, models, pivot, best_source):
    rows = []
    finite_gaps = set()
    for row_id in identities:
        reference = next(iter(pivot[row_id].values()))
        predictions = predictions_for(models, pivot[row_id])
        baseline = next(prediction for prediction in predictions if prediction.source == best_source)
        winner, _, gap, _ = ccm_decision(calibration, predictions, best_source, family_dedup=True)
        rows.append((reference, baseline, winner, gap))
        if math.isfinite(gap) and gap >= 0:
            finite_gaps.add(gap)
    baseline_successes = sum(score_prediction(reference, baseline) for reference, baseline, _, _ in rows)
    for threshold in sorted(finite_gaps) + [float("inf")]:
        successes = 0
        for reference, baseline, winner, gap in rows:
            use_winner = math.isfinite(threshold) and gap >= threshold
            successes += int(score_prediction(reference, winner if use_winner else baseline))
        if successes >= baseline_successes:
            return threshold, {
                "threshold_dev_rows": len(rows),
                "best_single_successes": baseline_successes,
                "selected_successes": successes,
                "candidate_thresholds": len(finite_gaps) + 1,
            }
    raise AssertionError("infinity must reproduce best single")


def safe_auc(labels, scores):
    finite = [(label, score) for label, score in zip(labels, scores) if math.isfinite(score)]
    if not finite or len({label for label, _ in finite}) < 2:
        return None
    return float(roc_auc_score([label for label, _ in finite], [score for _, score in finite]))


def override_curve(rows):
    finite = sorted(row["s_gap"] for row in rows if math.isfinite(row["s_gap"]))
    if not finite:
        return []
    thresholds = sorted({finite[round((len(finite) - 1) * quantile / 10)] for quantile in range(11)})
    output = []
    for threshold in thresholds:
        selected = [row for row in rows if row["override"] and row["s_gap"] >= threshold]
        output.append({
            "threshold": threshold,
            "override_rows": len(selected),
            "override_rate": len(selected) / len(rows),
            "conditional_step_sr": (
                sum(row["success"] for row in selected) / len(selected) if selected else None
            ),
        })
    return output


def paired_counts(left, right):
    wins = sum(left_value and not right_value for left_value, right_value in zip(left, right))
    losses = sum(right_value and not left_value for left_value, right_value in zip(left, right))
    total = wins + losses
    return {
        "wins": wins,
        "losses": losses,
        "superiority_p_one_sided": float(binomtest(wins, total, 0.5, alternative="greater").pvalue) if total else 1.0,
        "inferiority_p_one_sided": float(binomtest(losses, total, 0.5, alternative="greater").pvalue) if total else 1.0,
    }


def holm_adjust(values):
    ordered = sorted(values, key=lambda item: item[1])
    adjusted = {}
    running = 0.0
    total = len(ordered)
    for rank, (key, value) in enumerate(ordered):
        running = max(running, min(1.0, value * (total - rank)))
        adjusted[key] = running
    return adjusted


def evaluate_pool(bench, setting):
    pool = f"{bench}/{setting}"
    identities, available_models, pivot = load_pool(bench, setting)
    models = deployable_models(identities, available_models, pivot)
    mapping = fold_map(pool)
    folds = []
    method_outputs = {method: {} for method in BASELINES + VARIANTS}
    diagnostics = {variant: [] for variant in VARIANTS}
    for test_fold in range(5):
        outer_dev, test_ids = split_rows(identities, pivot, mapping, test_fold)
        priority = dev_priority(outer_dev, models, pivot)
        best_source = priority[0]
        pooled = fit_calibration(bench, calibration_rows(outer_dev, models, pivot), "pooled")
        nine = fit_calibration(bench, calibration_rows(outer_dev, models, pivot), "nine")

        threshold_fold = (test_fold + 1) % 5
        threshold_ids = [
            row_id for row_id in outer_dev
            if mapping[next(iter(pivot[row_id].values()))["group_key"]] == threshold_fold
        ]
        threshold_set = set(threshold_ids)
        risk_train = [row_id for row_id in outer_dev if row_id not in threshold_set]
        risk_priority = dev_priority(risk_train, models, pivot)
        risk_best_source = risk_priority[0]
        risk_calibration = fit_calibration(bench, calibration_rows(risk_train, models, pivot), "nine")
        risk_threshold, risk_report = choose_risk_threshold(
            bench, risk_calibration, threshold_ids, models, pivot, risk_best_source
        )

        backoff_totals = {variant: Counter() for variant in VARIANTS}
        fold_successes = Counter()
        for row_id in test_ids:
            reference = next(iter(pivot[row_id].values()))
            predictions = predictions_for(models, pivot[row_id])
            methods = aggregate_baselines(bench, predictions, priority)
            decisions = {}
            for variant, calibration, family_dedup, source in (
                ("A5b_MAP_pooled_LR", pooled, False, best_source),
                ("A5c_MAP_nine_LR", nine, False, best_source),
                ("A5d_MAP_nine_LR_family", nine, True, best_source),
            ):
                prediction, score, gap, backoffs = ccm_decision(
                    calibration, predictions, source, family_dedup
                )
                decisions[variant] = (prediction, score, gap, source)
                backoff_totals[variant].update(backoffs)
            risk_prediction, risk_score, risk_gap, risk_backoffs = ccm_decision(
                risk_calibration, predictions, risk_best_source, True
            )
            risk_baseline = next(
                prediction for prediction in predictions if prediction.source == risk_best_source
            )
            use_risk = math.isfinite(risk_threshold) and risk_gap >= risk_threshold
            decisions["A5d_risk"] = (
                risk_prediction if use_risk else risk_baseline,
                risk_score,
                risk_gap,
                risk_best_source,
            )
            backoff_totals["A5d_risk"].update(risk_backoffs)

            for method, prediction in methods.items():
                success = bool(score_prediction(reference, prediction))
                method_outputs[method][row_id] = success
                fold_successes[method] += int(success)
            for variant, (prediction, score, gap, source) in decisions.items():
                baseline = next(item for item in predictions if item.source == source)
                success = bool(score_prediction(reference, prediction))
                method_outputs[variant][row_id] = success
                fold_successes[variant] += int(success)
                diagnostics[variant].append({
                    "row_id": row_id,
                    "fold": test_fold,
                    "s_gap": gap,
                    "selected_score": score,
                    "success": success,
                    "override": prediction != baseline,
                    "selected_source": prediction.source if prediction else None,
                    "best_source": source,
                })
        folds.append({
            "fold": test_fold,
            "dev_rows": len(outer_dev),
            "test_rows": len(test_ids),
            "models": models,
            "best_source": best_source,
            "step_sr": {
                method: sum(method_outputs[method][row_id] for row_id in test_ids) / len(test_ids)
                for method in BASELINES + VARIANTS
            },
            "calibration": {
                "A5b": pooled.table_report,
                "A5c_A5d": nine.table_report,
                "A5d_risk": risk_calibration.table_report,
            },
            "risk": {
                "train_rows": len(risk_train),
                "threshold_fold": threshold_fold,
                "best_source": risk_best_source,
                "threshold": risk_threshold,
                **risk_report,
            },
            "backoff_counts": {
                variant: dict(counts) for variant, counts in backoff_totals.items()
            },
        })

    aggregate = {
        method: sum(method_outputs[method].values()) / len(identities)
        for method in BASELINES + VARIANTS
    }
    comparisons = {
        variant: {
            baseline: paired_counts(
                [method_outputs[variant][row_id] for row_id in identities],
                [method_outputs[baseline][row_id] for row_id in identities],
            )
            for baseline in BASELINES
        }
        for variant in VARIANTS
    }
    variant_diagnostics = {}
    for variant, rows in diagnostics.items():
        variant_diagnostics[variant] = {
            "override_rows": sum(row["override"] for row in rows),
            "override_rate": sum(row["override"] for row in rows) / len(rows),
            "override_conditional_step_sr": (
                sum(row["success"] for row in rows if row["override"])
                / max(1, sum(row["override"] for row in rows))
            ),
            "s_gap_correctness_auroc": safe_auc(
                [row["success"] for row in rows], [row["s_gap"] for row in rows]
            ),
            "override_curve": override_curve(rows),
        }
    return {
        "rows": len(identities),
        "models": models,
        "folds": folds,
        "aggregate_step_sr": aggregate,
        "comparisons": comparisons,
        "diagnostics": variant_diagnostics,
    }


def add_global_tests(result):
    for variant in VARIANTS:
        superiority = []
        inferiority = []
        for pool, values in result["pools"].items():
            baseline = POOLWISE_BEST[pool]
            comparison = values["comparisons"][variant][baseline]
            superiority.append((pool, comparison["superiority_p_one_sided"]))
            inferiority.append((pool, comparison["inferiority_p_one_sided"]))
        superiority_adjusted = holm_adjust(superiority)
        inferiority_adjusted = holm_adjust(inferiority)
        for pool, values in result["pools"].items():
            baseline = POOLWISE_BEST[pool]
            comparison = values["comparisons"][variant][baseline]
            comparison["superiority_p_holm"] = superiority_adjusted[pool]
            comparison["inferiority_p_holm"] = inferiority_adjusted[pool]
    a5c = "A5c_MAP_nine_LR"
    mind2web_gain = (
        result["pools"]["mind2web/visual"]["aggregate_step_sr"][a5c]
        - result["pools"]["mind2web/visual"]["aggregate_step_sr"]["A5a_LOO"]
    )
    ac_inferior = {
        pool: result["pools"][pool]["comparisons"][a5c][POOLWISE_BEST[pool]]["inferiority_p_holm"] < 0.05
        for pool in ("androidcontrol/low", "androidcontrol/high")
    }
    result["k4"] = {
        "mind2web_a5c_minus_a5a": mind2web_gain,
        "mind2web_strict_gain": mind2web_gain > 0,
        "androidcontrol_significantly_inferior": ac_inferior,
        "triggered": mind2web_gain <= 0 or any(ac_inferior.values()),
    }
    success = {}
    for variant in VARIANTS:
        inferior = 0
        superior = 0
        for pool, values in result["pools"].items():
            comparison = values["comparisons"][variant][POOLWISE_BEST[pool]]
            inferior += comparison["inferiority_p_holm"] < 0.05
            superior += comparison["superiority_p_holm"] < 0.05
        success[variant] = {
            "significantly_inferior_pools": inferior,
            "significantly_superior_pools": superior,
            "discovery_success": inferior == 0 and superior >= 2,
        }
    result["success_criteria"] = success
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = {
        "status": "PASS",
        "protocol": "AMENDMENT_007_CCM_CONFIRMATION.md",
        "pools": {},
    }
    for bench, setting in POOLS:
        result["pools"][f"{bench}/{setting}"] = evaluate_pool(bench, setting)
        print(f"completed {bench}/{setting}", flush=True)
    add_global_tests(result)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": result["status"],
        "aggregate_step_sr": {
            pool: values["aggregate_step_sr"] for pool, values in result["pools"].items()
        },
        "k4": result["k4"],
        "success_criteria": result["success_criteria"],
    }, indent=2))


if __name__ == "__main__":
    main()