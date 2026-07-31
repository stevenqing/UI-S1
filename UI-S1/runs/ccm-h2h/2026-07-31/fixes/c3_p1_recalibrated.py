import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from common import UPSTREAM, load_module, write_json


w1 = load_module(UPSTREAM / "w1_run.py", "c3_w1")
pka = load_module(UPSTREAM / "pka.py", "c3_pka")


def stratum(bench, action):
    if bench == "mind2web":
        return "mind2web_click" if action == "CLICK" else "mind2web_select_type"
    if action in w1.GROUNDING_ACTIONS:
        return "androidcontrol_coordinate"
    if action in w1.TEXT_ACTIONS:
        return "androidcontrol_string"
    return "androidcontrol_parameterless"


def evaluate_pool(bench, setting):
    identities, available_models, pivot = w1.load_pool(bench, setting)
    models = w1.deployable_models(identities, available_models, pivot)
    mapping = w1.fold_map(f"{bench}/{setting}")
    rows = []
    for test_fold in range(5):
        dev_ids, test_ids = w1.split_rows(identities, pivot, mapping, test_fold)
        priority = w1.dev_priority(dev_ids, models, pivot)
        best_source = priority[0]
        for row_id in test_ids:
            model_rows = pivot[row_id]
            successes = [bool(model_rows[model]["success"]) for model in models]
            if not any(successes) or all(successes):
                continue
            reference = next(iter(model_rows.values()))
            predictions = [w1.prediction_from_row(model_rows[model]) for model in models]
            parsed = [(index, prediction) for index, prediction in enumerate(predictions) if prediction.parse_ok]
            failed_masses = []
            for candidate_position, (candidate_index, candidate) in enumerate(parsed):
                if successes[candidate_index]:
                    continue
                values = [
                    pka.pair_kernel(bench, candidate, voter)
                    for voter_position, (_, voter) in enumerate(parsed)
                    if voter_position != candidate_position
                ]
                if values:
                    failed_masses.append(sum(values) / len(values))
            if not failed_masses:
                continue
            a3 = w1.pka_medoid(bench, predictions).prediction
            rows.append({
                "row_id": row_id,
                "fold": test_fold,
                "stratum": stratum(bench, reference["gt_action"]),
                "collision_mass": sum(failed_masses) / len(failed_masses),
                "a0_success": bool(model_rows[best_source]["success"]),
                "a3_success": bool(w1.score_prediction(reference, a3)),
            })
    return models, rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    grouped = defaultdict(list)
    pool_reports = {}
    for bench, setting in w1.POOLS:
        models, rows = evaluate_pool(bench, setting)
        pool = f"{bench}/{setting}"
        pool_reports[pool] = {"models": models, "disagreement_rows": len(rows)}
        for row in rows:
            grouped[row["stratum"]].append({"pool": pool, **row})
    summaries = {}
    for name, rows in sorted(grouped.items()):
        count = len(rows)
        a0 = sum(row["a0_success"] for row in rows) / count
        a3 = sum(row["a3_success"] for row in rows) / count
        summaries[name] = {
            "rows": count,
            "mean_error_conditional_agreement_mass": float(np.mean([row["collision_mass"] for row in rows])),
            "a0_step_sr": a0,
            "a3_step_sr": a3,
            "a3_gain_over_a0": a3 - a0,
            "pools": sorted({row["pool"] for row in rows}),
        }
    collision = [value["mean_error_conditional_agreement_mass"] for value in summaries.values()]
    gains = [value["a3_gain_over_a0"] for value in summaries.values()]
    correlation = spearmanr(collision, gains)
    result = {
        "status": "PASS",
        "definition": "mean leave-one-out evaluator-kernel mass of failed candidates on disagreement rows",
        "pools": pool_reports,
        "strata": summaries,
        "p1_recalibrated": {
            "spearman_collision_vs_gain": float(correlation.statistic),
            "p_value_descriptive": float(correlation.pvalue),
            "prediction_satisfied": bool(correlation.statistic < 0),
            "retroactively_rescues_original_p1": False,
        },
    }
    write_json(args.output, result)
    print(json.dumps(result["p1_recalibrated"], indent=2))


if __name__ == "__main__":
    main()
