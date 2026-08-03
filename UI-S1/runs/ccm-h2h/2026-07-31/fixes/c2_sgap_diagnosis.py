import argparse
import json
import math
from pathlib import Path

from scipy.stats import hypergeom

from common import UPSTREAM, load_module, write_json


w1 = load_module(UPSTREAM / "w1_run.py", "c2_w1")
a5 = load_module(UPSTREAM / "a5_ccm_run.py", "c2_a5")
ccm = load_module(UPSTREAM / "ccm.py", "c2_ccm")


def order_statistic(values, quantile):
    ordered = sorted(values)
    if not ordered:
        raise ValueError("quantile requires non-empty values")
    return ordered[math.ceil(quantile * len(ordered)) - 1]


def run_pool(bench, setting):
    pool = f"{bench}/{setting}"
    identities, available_models, pivot = w1.load_pool(bench, setting)
    models = w1.deployable_models(identities, available_models, pivot)
    mapping = w1.fold_map(pool)
    records = []
    fold_reports = []
    for test_fold in range(5):
        outer_dev, test_ids = w1.split_rows(identities, pivot, mapping, test_fold)
        threshold_fold = (test_fold + 1) % 5
        threshold_ids = [
            row_id for row_id in outer_dev
            if mapping[next(iter(pivot[row_id].values()))["group_key"]] == threshold_fold
        ]
        threshold_set = set(threshold_ids)
        train_ids = [row_id for row_id in outer_dev if row_id not in threshold_set]
        priority = w1.dev_priority(train_ids, models, pivot)
        best_source = priority[0]
        calibration = ccm.fit_calibration(
            bench, a5.calibration_rows(train_ids, models, pivot), "nine"
        )
        risk_threshold, report = a5.choose_risk_threshold(
            bench, calibration, threshold_ids, models, pivot, best_source
        )
        fold_reports.append({
            "fold": test_fold,
            "train_rows": len(train_ids),
            "threshold_rows": len(threshold_ids),
            "test_rows": len(test_ids),
            "risk_threshold": risk_threshold,
            "best_source": best_source,
            **report,
        })
        for row_id in test_ids:
            reference = next(iter(pivot[row_id].values()))
            predictions = a5.predictions_for(models, pivot[row_id])
            baseline = next(prediction for prediction in predictions if prediction.source == best_source)
            winner, _, gap, _ = a5.ccm_decision(
                calibration, predictions, best_source, family_dedup=True
            )
            use_winner = math.isfinite(risk_threshold) and gap >= risk_threshold
            selected = winner if use_winner else baseline
            success = bool(w1.score_prediction(reference, selected))
            hard_core = not any(pivot[row_id][model]["success"] for model in models)
            records.append({
                "row_id": row_id,
                "fold": test_fold,
                "s_gap": gap,
                "selected_success": success,
                "hard_core": hard_core,
                "override": selected != baseline,
            })
    finite = [record["s_gap"] for record in records if math.isfinite(record["s_gap"])]
    threshold = order_statistic(finite, 0.9)
    diagnostic = [
        record for record in records
        if math.isfinite(record["s_gap"])
        and record["s_gap"] >= threshold
        and not record["selected_success"]
    ]
    hard_total = sum(record["hard_core"] for record in records)
    overlap = sum(record["hard_core"] for record in diagnostic)
    population = len(records)
    draws = len(diagnostic)
    expected = draws * hard_total / population
    p = float(hypergeom.sf(overlap - 1, population, hard_total, draws)) if draws else 1.0
    base_rate = hard_total / population
    diagnostic_rate = overlap / draws if draws else None
    return {
        "rows": population,
        "models": models,
        "folds": fold_reports,
        "high_gap_quantile": 0.9,
        "high_gap_threshold": threshold,
        "finite_gap_rows": len(finite),
        "diagnostic_failed_high_gap_rows": draws,
        "hard_core_rows": hard_total,
        "hard_core_base_rate": base_rate,
        "diagnostic_hard_core_overlap": overlap,
        "diagnostic_hard_core_rate": diagnostic_rate,
        "random_expected_overlap": expected,
        "enrichment_ratio": diagnostic_rate / base_rate if draws and base_rate else None,
        "hypergeometric_p_greater_equal": p,
        "prediction_satisfied": bool(draws and diagnostic_rate > base_rate and p < 0.01),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    pools = {
        f"{bench}/{setting}": run_pool(bench, setting)
        for bench, setting in w1.POOLS
    }
    result = {
        "status": "PASS",
        "definition": "inclusive OOF S_gap 90th percentile; failed selected rows; exact hard-core enrichment",
        "pools": pools,
        "summary": {
            "directional_passes": sum(value["prediction_satisfied"] for value in pools.values()),
            "pools": len(pools),
        },
    }
    write_json(args.output, result)
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
