import argparse
import json
import math
from pathlib import Path

from ccm import calibration_to_dict, fit_calibration, score_candidates
from w1_run import deployable_models, fold_map, load_pool, model_step_sr, prediction_from_row, score_prediction, split_rows


SETTINGS = ("low", "high")


def prediction_rows(identities, models, pivot):
    rows = []
    for row_id in identities:
        reference = next(iter(pivot[row_id].values()))
        predictions = [prediction_from_row(pivot[row_id][model]) for model in models]
        rows.append((predictions, [score_prediction(reference, prediction) for prediction in predictions]))
    return rows


def select_threshold(records):
    baseline_successes = sum(record["baseline_success"] for record in records)
    candidates = sorted({record["s_gap"] for record in records if math.isfinite(record["s_gap"]) and record["s_gap"] >= 0})
    for threshold in candidates + [None]:
        successes = 0
        overrides = 0
        for record in records:
            use_winner = threshold is not None and record["s_gap"] >= threshold
            successes += record["winner_success"] if use_winner else record["baseline_success"]
            overrides += int(use_winner and record["winner_source"] != record["best_source"])
        if successes >= baseline_successes:
            return threshold, {
                "rows": len(records),
                "candidate_thresholds": len(candidates) + 1,
                "best_source_successes": baseline_successes,
                "selected_successes": successes,
                "overrides": overrides,
            }
    raise AssertionError("infinity threshold must reproduce best source")


def freeze_setting(setting):
    identities, available_models, pivot = load_pool("androidcontrol", setting)
    models = deployable_models(identities, available_models, pivot)
    best_source = max(models, key=lambda model: (model_step_sr(identities, model, pivot), model))
    mapping = fold_map(f"androidcontrol/{setting}")
    oof = []
    fold_reports = []
    for test_fold in range(5):
        dev_ids, test_ids = split_rows(identities, pivot, mapping, test_fold)
        calibration = fit_calibration(
            "androidcontrol", prediction_rows(dev_ids, models, pivot), "nine"
        )
        for row_id in test_ids:
            reference = next(iter(pivot[row_id].values()))
            predictions = [prediction_from_row(pivot[row_id][model]) for model in models]
            scores, _ = score_candidates(calibration, predictions, family_dedup=True)
            winner_position = max(range(len(scores)), key=lambda index: (scores[index][0], -index))
            winner_score, _, winner = scores[winner_position]
            baseline = next(prediction for prediction in predictions if prediction.source == best_source)
            baseline_score = next(score for score, _, prediction in scores if prediction.source == best_source)
            oof.append({
                "s_gap": winner_score - baseline_score,
                "winner_success": int(score_prediction(reference, winner)),
                "baseline_success": int(score_prediction(reference, baseline)),
                "winner_source": winner.source,
                "best_source": best_source,
            })
        fold_reports.append({
            "fold": test_fold,
            "dev_rows": len(dev_ids),
            "test_rows": len(test_ids),
            "calibration": calibration.table_report,
        })
    threshold, threshold_report = select_threshold(oof)
    final_calibration = fit_calibration(
        "androidcontrol", prediction_rows(identities, models, pivot), "nine"
    )
    return {
        "models": models,
        "fixed_best_source": best_source,
        "discovery_rows": len(identities),
        "oof_threshold": threshold,
        "oof_threshold_report": threshold_report,
        "folds": fold_reports,
        "final_calibration": calibration_to_dict(final_calibration),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = {
        "status": "PASS",
        "protocol": "AMENDMENT_008_CCM_CONFIRMATION_DEPLOYMENT.md",
        "w4_labels_read": False,
        "settings": {setting: freeze_setting(setting) for setting in SETTINGS},
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": result["status"],
        "settings": {
            setting: {
                "models": value["models"],
                "fixed_best_source": value["fixed_best_source"],
                "oof_threshold": value["oof_threshold"],
                "oof_threshold_report": value["oof_threshold_report"],
            }
            for setting, value in result["settings"].items()
        },
    }, indent=2))


if __name__ == "__main__":
    main()