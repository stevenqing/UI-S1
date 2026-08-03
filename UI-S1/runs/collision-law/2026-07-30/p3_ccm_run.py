import argparse
import json
from pathlib import Path

from ccm import collision_calibrated_mode, fit_calibration
from w1_run import fold_map, score_prediction, split_rows
from w2_analyze import P3_MODELS, P3_REPRESENTATIVE, P3_SEED, VIEWS, build_p3_pool


POOLS = (("androidcontrol", "low"), ("androidcontrol", "high"), ("mind2web", "visual"))


def calibration_rows(identities, pivot, units, selected):
    rows = []
    for row_id in identities:
        reference = next(iter(pivot[row_id].values()))
        predictions = [units[key][row_id] for key in selected]
        successes = [score_prediction(reference, prediction) for prediction in predictions]
        rows.append((predictions, successes))
    return rows


def fit_and_evaluate(bench, train_ids, eval_ids, pivot, units, selected):
    calibration = fit_calibration(
        bench, calibration_rows(train_ids, pivot, units, selected), "nine"
    )
    successes = 0
    for row_id in eval_ids:
        reference = next(iter(pivot[row_id].values()))
        predictions = [units[key][row_id] for key in selected]
        aggregate = collision_calibrated_mode(
            calibration, predictions, family_dedup=True
        ).prediction
        successes += int(score_prediction(reference, aggregate))
    return successes / len(eval_ids), calibration.table_report


def standalone_rates(identities, pivot, units):
    return {
        key: sum(
            score_prediction(next(iter(pivot[row_id].values())), predictions[row_id])
            for row_id in identities
        ) / len(identities)
        for key, predictions in units.items()
    }


def greedy_allocation(bench, dev_ids, pivot, units, budget=5):
    rates = standalone_rates(dev_ids, pivot, units)
    selected = [max(rates, key=lambda key: (rates[key], key))]
    trace = [{
        "step": 1,
        "added_unit": selected[0],
        "selected": selected.copy(),
        "standalone_step_sr": rates[selected[0]],
        "simulated_ccm_step_sr": rates[selected[0]],
        "increment": None,
    }]
    current = rates[selected[0]]
    while len(selected) < budget:
        candidates = []
        for key in sorted(set(units) - set(selected)):
            trial = selected + [key]
            score, _ = fit_and_evaluate(bench, dev_ids, dev_ids, pivot, units, trial)
            candidates.append((score, rates[key], key))
        score, standalone, key = max(candidates)
        selected.append(key)
        trace.append({
            "step": len(selected),
            "added_unit": key,
            "selected": selected.copy(),
            "standalone_step_sr": standalone,
            "simulated_ccm_step_sr": score,
            "increment": score - current,
        })
        current = score
    return selected, trace


def run_pool(bench, setting, original):
    pool = build_p3_pool(bench, setting)
    if pool is None:
        raise ValueError(f"incomplete P3 pool: {bench}/{setting}")
    identities, pivot, units = pool
    representative = P3_REPRESENTATIVE[bench]
    c1 = [f"{representative}/{view}" for view in VIEWS]
    c2 = [f"{model}/full" for model in P3_MODELS[bench]]
    folds = []
    total_rows = 0
    totals = {method: 0.0 for method in ("C1_views_CCM", "C2_lineages_CCM", "C3_greedy_CCM", "C4_random_CCM")}
    for test_fold in range(5):
        dev_ids, test_ids = split_rows(
            identities, pivot, fold_map(f"{bench}/{setting}"), test_fold
        )
        c3, trace = greedy_allocation(bench, dev_ids, pivot, units)
        c4 = original["folds"][test_fold]["selections"]["C4_random_mixed"]
        selections = {
            "C1_views_CCM": c1,
            "C2_lineages_CCM": c2,
            "C3_greedy_CCM": c3,
            "C4_random_CCM": c4,
        }
        scores = {}
        reports = {}
        for method, selected in selections.items():
            score, report = fit_and_evaluate(
                bench, dev_ids, test_ids, pivot, units, selected
            )
            scores[method] = score
            reports[method] = report
            totals[method] += score * len(test_ids)
        total_rows += len(test_ids)
        folds.append({
            "fold": test_fold,
            "dev_rows": len(dev_ids),
            "test_rows": len(test_ids),
            "selections": selections,
            "c3_selection_trace": trace,
            "step_sr": scores,
            "calibration": reports,
        })
    aggregate = {method: value / total_rows for method, value in totals.items()}
    return {
        "rows": total_rows,
        "candidate_units": sorted(units),
        "folds": folds,
        "aggregate_step_sr": aggregate,
        "p3_ccm_prediction_satisfied": aggregate["C3_greedy_CCM"] > max(
            aggregate["C1_views_CCM"], aggregate["C2_lineages_CCM"]
        ),
        "c3_exceeds_random": aggregate["C3_greedy_CCM"] > aggregate["C4_random_CCM"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    original = json.loads(args.original.read_text())
    result = {
        "status": "PASS",
        "protocol": "AMENDMENT_007_CCM_CONFIRMATION.md",
        "contract": {
            "budget": 5,
            "aggregator": "A5d_MAP_nine_LR_family",
            "selection": "greedy development simulated CCM Step SR increment",
            "ties": "higher standalone development Step SR then unit key",
            "test_label_tuning": False,
            "original_p3_remains_negative": True,
        },
        "pools": {},
    }
    for bench, setting in POOLS:
        key = f"{bench}/{setting}"
        result["pools"][key] = run_pool(bench, setting, original["pools"][key])
        print(f"completed {key}", flush=True)
    result["summary"] = {
        "p3_ccm_directional_passes": sum(
            values["p3_ccm_prediction_satisfied"] for values in result["pools"].values()
        ),
        "c3_exceeds_random_pools": sum(
            values["c3_exceeds_random"] for values in result["pools"].values()
        ),
        "pools": len(result["pools"]),
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": result["status"],
        "summary": result["summary"],
        "aggregate_step_sr": {
            pool: values["aggregate_step_sr"] for pool, values in result["pools"].items()
        },
    }, indent=2))


if __name__ == "__main__":
    main()