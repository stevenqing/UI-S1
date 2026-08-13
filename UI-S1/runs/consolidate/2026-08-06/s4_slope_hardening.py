import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


RUN_DIR = Path(__file__).resolve().parent
ROOT = RUN_DIR.parents[2]
sys.path.insert(0, str(RUN_DIR))

from common import MODELS, SEED, evaluate_actions, load_context, mixed_sequence, paired_group_bootstrap, write_json


BUDGETS = tuple(range(2, 17))
METHODS = ("B3_mvp", "M1_ccm")
RESAMPLES = 10000


def slope(values):
    x = np.asarray(BUDGETS, dtype=np.float64)
    y = np.asarray(values, dtype=np.float64)
    centered = x - x.mean()
    return float(centered @ y / (centered @ centered))


def group_bootstrap_slopes(evaluations, pool):
    metadata = evaluations[pool][2]["row_metadata"]
    by_fold_group = defaultdict(lambda: defaultdict(list))
    for row_id, row in metadata.items():
        by_fold_group[row["outer_fold"]][row["application"]].append(row_id)
    rng = np.random.default_rng(SEED)
    reports = {}
    for method in METHODS:
        samples = []
        for _ in range(RESAMPLES):
            selected = []
            for fold in sorted(by_fold_group):
                groups = sorted(by_fold_group[fold])
                for group in rng.choice(groups, size=len(groups), replace=True):
                    selected.extend(by_fold_group[fold][group])
            values = [
                np.mean([evaluations[pool][budget]["outputs"][method][row_id] for row_id in selected])
                for budget in BUDGETS
            ]
            samples.append(slope(values))
        point_values = [evaluations[pool][budget]["accuracy"][method] for budget in BUDGETS]
        reports[method] = {
            "point_slope_per_forward": slope(point_values),
            "ci_99": [float(np.quantile(samples, 0.005)), float(np.quantile(samples, 0.995))],
            "p_nonnegative": float(np.mean(np.asarray(samples) >= 0)),
            "resamples": RESAMPLES,
            "seed": SEED,
        }
    return reports


def page_permutation(evaluations, pool, method):
    matrix = np.asarray([
        [evaluations[pool][budget]["outputs"][method][row_id] for budget in BUDGETS]
        for row_id in evaluations[pool][2]["row_metadata"]
    ], dtype=np.float64)
    scores = np.arange(1, len(BUDGETS) + 1, dtype=np.float64)
    observed = float(np.sum(matrix @ scores))
    rng = np.random.default_rng(SEED)
    null = np.empty(RESAMPLES, dtype=np.float64)
    for index in range(RESAMPLES):
        null[index] = float(np.sum(matrix @ rng.permutation(scores)))
    direction = "decreasing" if observed < float(np.mean(null)) else "increasing"
    p_value = float((1 + np.sum(null <= observed)) / (RESAMPLES + 1)) if direction == "decreasing" else float((1 + np.sum(null >= observed)) / (RESAMPLES + 1))
    return {
        "direction": direction,
        "observed": observed,
        "null_mean": float(np.mean(null)),
        "p_one_sided_plus_one": p_value,
        "resamples": RESAMPLES,
        "seed": SEED,
        "note": "Page-style global budget-label permutation preserving each row's repeated outcomes",
    }


def main():
    context = load_context()
    sequence = mixed_sequence()
    evaluations = {"v_only": {}, "mixed": {}}
    for budget in BUDGETS:
        evaluations["v_only"][budget] = evaluate_actions(context, [("GTA1-7B", view) for view in range(budget)])
        evaluations["mixed"][budget] = evaluate_actions(context, sequence[:budget])

    anchors = json.loads((ROOT / "runs/allocation-law/2026-08-01/L1_RESULTS.json").read_text())
    for pool in ("v_only", "mixed"):
        for budget in (4, 8, 12, 16):
            for metric in (*METHODS, "pass_at_n"):
                expected = anchors["evaluations"][pool][str(budget)]["accuracy"][metric]
                actual = evaluations[pool][budget]["accuracy"][metric]
                if abs(actual - expected) > 1e-15:
                    raise ValueError(f"S4 anchor mismatch: {pool}/{budget}/{metric}")

    slopes = {pool: group_bootstrap_slopes(evaluations, pool) for pool in evaluations}
    paired = {
        pool: {
            method: {
                **paired_group_bootstrap(
                    evaluations[pool][4]["row_metadata"],
                    evaluations[pool][16]["outputs"][method],
                    evaluations[pool][4]["outputs"][method],
                ),
                "N4": evaluations[pool][4]["accuracy"][method],
                "N16": evaluations[pool][16]["accuracy"][method],
            }
            for method in METHODS
        }
        for pool in evaluations
    }
    page = {pool: {method: page_permutation(evaluations, pool, method) for method in METHODS} for pool in evaluations}
    s_k3 = any(slopes["v_only"][method]["ci_99"][0] <= 0 <= slopes["v_only"][method]["ci_99"][1] for method in METHODS)
    result = {
        "schema_version": 1,
        "status": "PASS",
        "budgets": list(BUDGETS),
        "curves": {
            pool: {str(budget): evaluation["accuracy"] for budget, evaluation in values.items()}
            for pool, values in evaluations.items()
        },
        "dense_slopes": slopes,
        "paired_N16_minus_N4": paired,
        "page_trend": page,
        "S_K3": s_k3,
        "primary_writing_statistic": "paired_N16_minus_N4",
        "slope_role": "supplementary",
    }
    write_json(RUN_DIR / "s4_slope_hardening.json", result)
    print(json.dumps({"S_K3": s_k3, "slopes": slopes, "paired": paired, "page": page}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()