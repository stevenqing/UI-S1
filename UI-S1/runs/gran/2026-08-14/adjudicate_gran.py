import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

from gran_common import attach_reliability, density_select, source_reliability
from run_tau_sweep import load_mind_rows, load_module, load_screen_rows


RUN_DIR = Path(__file__).resolve().parent
SWEEP_PATH = RUN_DIR / "TAU_SWEEP.json"
OUTPUT_PATH = RUN_DIR / "GRAN_ADJUDICATION.json"
RESAMPLES = 10000
CONFIDENCE_QUANTILES = (0.005, 0.995)
SEED = 20260814


def spearman(left, right):
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if len(left) < 2 or np.std(left) == 0 or np.std(right) == 0:
        return 0.0
    return float(np.corrcoef(rankdata(left), rankdata(right))[0, 1])


def fixed_strata(rows, counts):
    ordered = sorted(rows, key=lambda row: (row["p_hat"], row["row_id"]))
    if sum(counts) != len(ordered):
        raise ValueError("GRAN fixed stratum counts mismatch")
    output = []
    start = 0
    for index, count in enumerate(counts):
        selected = ordered[start:start + count]
        output.append({
            "index": index,
            "rows": selected,
            "count": len(selected),
            "p_min": float(min(row["p_hat"] for row in selected)),
            "p_max": float(max(row["p_hat"] for row in selected)),
            "p_mean": float(np.mean([row["p_hat"] for row in selected])),
            "q_max_mean": float(np.mean([row["q_max_hat"] for row in selected])),
            "margin": float(np.mean([row["margin"] for row in selected])),
        })
        start += count
    return output


def grouped_bootstrap(rows, statistic, seed):
    by_fold_group = defaultdict(lambda: defaultdict(list))
    for row in rows:
        by_fold_group[int(row["fold"])][str(row["group"])].append(row)
    generator = np.random.default_rng(seed)
    values = np.empty(RESAMPLES, dtype=np.float64)
    for index in range(RESAMPLES):
        selected = []
        for fold in sorted(by_fold_group):
            groups = sorted(by_fold_group[fold])
            for group in generator.choice(groups, size=len(groups), replace=True):
                selected.extend(by_fold_group[fold][str(group)])
        values[index] = statistic(selected)
    point = statistic(rows)
    return {
        "point": float(point),
        "ci_99": [
            float(np.quantile(values, CONFIDENCE_QUANTILES[0])),
            float(np.quantile(values, CONFIDENCE_QUANTILES[1])),
        ],
        "resamples": RESAMPLES,
        "seed": seed,
    }


def mechanism_rows(values):
    return [dict(row, row_id=row_id) for row_id, row in values.items()]


def p1(rows):
    report = grouped_bootstrap(
        rows,
        lambda selected: spearman(
            [row["p_hat"] for row in selected],
            [row["margin"] for row in selected],
        ),
        SEED + 1,
    )
    report["pass"] = report["ci_99"][0] > 0
    return report


def p2(rows, strata):
    highest_ids = {row["row_id"] for row in strata[-1]["rows"]}
    selected = [row for row in rows if row["row_id"] in highest_ids]
    report = grouped_bootstrap(
        selected,
        lambda values: float(np.mean([row["margin"] for row in values])),
        SEED + 2,
    )
    report["rows"] = len(selected)
    report["p_range"] = [strata[-1]["p_min"], strata[-1]["p_max"]]
    report["pass"] = report["ci_99"][0] > 0
    return report


def independent_stratum_difference(left, right, seed):
    left_groups = defaultdict(list)
    right_groups = defaultdict(list)
    for row in left:
        left_groups[(int(row["fold"]), str(row["group"]))].append(row)
    for row in right:
        right_groups[(int(row["fold"]), str(row["group"]))].append(row)
    generator = np.random.default_rng(seed)
    values = np.empty(RESAMPLES, dtype=np.float64)
    left_keys = sorted(left_groups)
    right_keys = sorted(right_groups)
    for index in range(RESAMPLES):
        left_selected = [
            row for key in generator.choice(len(left_keys), size=len(left_keys), replace=True)
            for row in left_groups[left_keys[int(key)]]
        ]
        right_selected = [
            row for key in generator.choice(len(right_keys), size=len(right_keys), replace=True)
            for row in right_groups[right_keys[int(key)]]
        ]
        values[index] = np.mean([row["margin"] for row in left_selected]) - np.mean([
            row["margin"] for row in right_selected
        ])
    point = float(np.mean([row["margin"] for row in left]) - np.mean([
        row["margin"] for row in right
    ]))
    return {
        "point": point,
        "ci_99": [float(np.quantile(values, 0.005)), float(np.quantile(values, 0.995))],
    }


def p3(screen_strata, mind_strata):
    reports = []
    for index, (screen, mind) in enumerate(zip(screen_strata, mind_strata)):
        report = independent_stratum_difference(
            screen["rows"], mind["rows"], SEED + 30 + index
        )
        report["stratum"] = index
        report["pass"] = report["ci_99"][0] <= 0 <= report["ci_99"][1]
        reports.append(report)
    return {"strata": reports, "pass": all(report["pass"] for report in reports)}


def endpoint_outputs(rows, benchmark):
    exact = {}
    single = {}
    metadata = {}
    for outer_fold in range(5):
        development = [row_id for row_id, row in rows.items() if row["fold"] != outer_fold]
        test = [row_id for row_id, row in rows.items() if row["fold"] == outer_fold]
        reliability = source_reliability(rows, development)
        for row_id in test:
            candidates = attach_reliability(rows[row_id]["candidates"], reliability)
            exact_candidate, _ = density_select(candidates, benchmark, "exact")
            single_candidate, _ = density_select(candidates, benchmark, "single")
            exact[row_id] = bool(exact_candidate is not None and exact_candidate.correct)
            single[row_id] = bool(single_candidate is not None and single_candidate.correct)
            metadata[row_id] = {
                "fold": outer_fold,
                "group": rows[row_id]["group"],
            }
    return exact, single, metadata


def p6():
    e1 = load_module(
        RUN_DIR.parents[2] / "runs/close/2026-08-08/e1_arm_aggregator_matrix.py",
        "gran_adjudication_e1",
    )
    datasets = {
        "screenspot_pro": load_screen_rows(e1),
        "mind2web": load_mind_rows(e1)["C_uni"],
    }
    mde = {"screenspot_pro": 0.007, "mind2web": 0.006106589385659482}
    reports = {}
    for index, (benchmark, rows) in enumerate(datasets.items()):
        exact, single, metadata = endpoint_outputs(rows, benchmark)
        values = [
            {
                "row_id": row_id,
                "fold": metadata[row_id]["fold"],
                "group": metadata[row_id]["group"],
                "difference": int(exact[row_id]) - int(single[row_id]),
            }
            for row_id in sorted(rows)
        ]
        report = grouped_bootstrap(
            values,
            lambda selected: float(np.mean([row["difference"] for row in selected])),
            SEED + 60 + index,
        )
        report["mde"] = mde[benchmark]
        report["pass"] = (
            report["ci_99"][0] >= -mde[benchmark]
            and report["ci_99"][1] <= mde[benchmark]
        )
        report["exact_accuracy"] = float(np.mean(list(exact.values())))
        report["single_accuracy"] = float(np.mean(list(single.values())))
        reports[benchmark] = report
    return {"benchmarks": reports, "pass": all(row["pass"] for row in reports.values())}


def strip_rows(strata):
    return [{key: value for key, value in row.items() if key != "rows"} for row in strata]


def main():
    if OUTPUT_PATH.exists():
        raise FileExistsError(OUTPUT_PATH)
    sweep = json.loads(SWEEP_PATH.read_text())
    if sweep.get("status") != "PASS_GRAN_NESTED_TAU_SWEEP":
        raise PermissionError("GRAN raw sweep is not locked")
    screen_rows = mechanism_rows(sweep["screenspot_pro"]["mechanisms"])
    mind_rows = mechanism_rows(sweep["mind2web"]["C_uni"]["mechanisms"])
    screen_strata = fixed_strata(screen_rows, [396, 395, 395, 395])
    mind_strata = fixed_strata(mind_rows, [444, 444, 443, 443])
    reports = {
        "G_P1": p1(screen_rows),
        "G_P2": p2(mind_rows, mind_strata),
        "G_P3": p3(screen_strata, mind_strata),
        "G_P4": {
            "status": "NOT_ADJUDICABLE_PREREG_UNDERDEFINED",
            "reason": "zero_point_interpolation_and_observed_crossing_rule_not_frozen_before_sweep",
        },
        "G_P5": {
            "status": "NOT_ADJUDICABLE_PREREG_UNDERDEFINED",
            "reason": "second_difference_plateau_tolerance_not_frozen_before_sweep",
        },
        "G_P6": p6(),
        "G_P7": {
            "status": "NOT_ADJUDICABLE_PREREG_UNDERDEFINED",
            "reason": "common_36_action_A1_A2_B3_output_scope_not_frozen",
        },
        "G_P8": {
            "status": "NOT_ADJUDICABLE_PREREG_UNDERDEFINED",
            "reason": "margin_z_numerator_standard_error_and_dependence_formula_not_frozen",
        },
    }
    kill_conditions = {
        "G_K1": not reports["G_P1"]["pass"],
        "G_K2": not reports["G_P2"]["pass"],
        "G_K3": not reports["G_P6"]["pass"],
        "G_K4": None,
        "G_K5": False,
        "G_K6": any(sweep["kill_conditions"].values()),
    }
    result = {
        "schema_version": 1,
        "status": "PASS_GRAN_ADJUDICATION_COMPLETE",
        "outcome": (
            "GRAN_PRIMARY_SUPPORTED_BUT_UNIFICATION_FAILED_G_K3_AND_GRID_FAILED_G_K6"
            if reports["G_P2"]["pass"] and kill_conditions["G_K3"] and kill_conditions["G_K6"]
            else "GRAN_PRIMARY_SUPPORTED_BUT_UNIFICATION_FAILED_G_K3"
            if reports["G_P2"]["pass"] and kill_conditions["G_K3"]
            else "GRAN_PRIMARY_SUPPORTED_WITH_GRID_FAILURE_G_K6"
            if reports["G_P2"]["pass"] and kill_conditions["G_K6"]
            else "GRAN_PRIMARY_SUPPORTED"
            if reports["G_P2"]["pass"]
            else "GRAN_PRIMARY_FAILED"
        ),
        "primary_test": "G_P2",
        "screen_strata": strip_rows(screen_strata),
        "mind2web_click_strata": strip_rows(mind_strata),
        "predictions": reports,
        "kill_conditions": kill_conditions,
        "underdefined_secondary_tests": ["G_P4", "G_P5", "G_P7", "G_P8"],
        "claim_boundary": {
            "explanatory_only": True,
            "method_claim_allowed": False,
            "gt_free_selector_claim_allowed": False,
            "changes_trivus_or_vus_sr": False,
        },
    }
    OUTPUT_PATH.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "outcome": result["outcome"],
        "primary": reports["G_P2"],
        "G_P1": reports["G_P1"],
        "G_P3": reports["G_P3"],
        "G_P6": reports["G_P6"],
        "kill_conditions": kill_conditions,
        "underdefined": result["underdefined_secondary_tests"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()