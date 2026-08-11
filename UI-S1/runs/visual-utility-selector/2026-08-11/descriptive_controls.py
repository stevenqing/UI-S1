import json
import sys
from pathlib import Path

import numpy as np

from adjudicate_anchor import ARMS, BENCHMARKS
from finalize_set_ranker import paired_samples
from set_ranker_data import keyed


RUN_DIR = Path(__file__).resolve().parent
UTILITY_DIR = RUN_DIR.parents[2] / "runs/lsa-utility/2026-08-11"
sys.path.insert(0, str(UTILITY_DIR))

from utility_adjudicate import correctness_outputs


def anchor_outputs(public, anchor):
    output = {benchmark: {arm: {} for arm in ARMS} for benchmark in BENCHMARKS}
    for sample_key, success in anchor["outputs"]["safe"].items():
        row = public[sample_key]
        values = output[row["benchmark"]][row["arm"]]
        if row["row_id"] in values:
            raise ValueError(f"duplicate anchor row: {sample_key}")
        values[row["row_id"]] = bool(success)
    return output


def formal_override_totals(outers):
    output = {benchmark: {} for benchmark in BENCHMARKS}
    for benchmark in BENCHMARKS:
        for arm in ARMS:
            reports = [outer["test"][benchmark][arm] for outer in outers]
            rows = sum(report["rows"] for report in reports)
            overrides = sum(report["overrides"] for report in reports)
            wins = sum(report["wins"] for report in reports)
            losses = sum(report["losses"] for report in reports)
            output[benchmark][arm] = {
                "rows": rows,
                "overrides": overrides,
                "override_rate": overrides / rows,
                "wins": wins,
                "losses": losses,
                "net_wins": wins - losses,
                "precision_among_decisive_overrides": wins / (wins + losses) if wins + losses else None,
            }
        output[benchmark]["equal_arm_mean_override_rate"] = float(np.mean([
            output[benchmark][arm]["override_rate"] for arm in ARMS
        ]))
        output[benchmark]["total_wins"] = sum(output[benchmark][arm]["wins"] for arm in ARMS)
        output[benchmark]["total_losses"] = sum(output[benchmark][arm]["losses"] for arm in ARMS)
    return output


def main():
    public = keyed(RUN_DIR / "data/public_records.jsonl")
    formal = json.loads((RUN_DIR / "set_ranker_adjudication.json").read_text())
    anchor = json.loads((RUN_DIR / "zero_shot/anchor_adjudication.json").read_text())
    anchor_safe = anchor_outputs(public, anchor)
    correctness = correctness_outputs()
    comparisons = {benchmark: {} for benchmark in BENCHMARKS}
    for benchmark in BENCHMARKS:
        samples_by_arm = []
        points = []
        seed = 20261391 if benchmark == "mind2web" else 20261392
        for arm_index, arm in enumerate(ARMS):
            result, samples = paired_samples(
                public, benchmark, arm,
                formal["outputs"][benchmark][arm]["safe"],
                anchor_safe[benchmark][arm],
                10000, seed + arm_index,
            )
            comparisons[benchmark][f"{arm}_minus_zero_shot_anchor"] = result
            samples_by_arm.append(samples)
            points.append(result["point_delta"])
        equal_arm = np.mean(np.stack(samples_by_arm), axis=0)
        comparisons[benchmark]["equal_arm_mean_minus_zero_shot_anchor"] = {
            "point_delta": float(np.mean(points)),
            "ci_99": [float(np.quantile(equal_arm, 0.005)), float(np.quantile(equal_arm, 0.995))],
        }
        correctness_samples = []
        correctness_points = []
        for arm_index, arm in enumerate(ARMS):
            result, samples = paired_samples(
                public, benchmark, arm,
                formal["outputs"][benchmark][arm]["safe"],
                correctness[benchmark][arm],
                10000, seed + 100 + arm_index,
            )
            comparisons[benchmark][f"{arm}_minus_correctness_LSA"] = result
            correctness_samples.append(samples)
            correctness_points.append(result["point_delta"])
        equal_correctness = np.mean(np.stack(correctness_samples), axis=0)
        comparisons[benchmark]["equal_arm_mean_minus_correctness_LSA"] = {
            "point_delta": float(np.mean(correctness_points)),
            "ci_99": [float(np.quantile(equal_correctness, 0.005)), float(np.quantile(equal_correctness, 0.995))],
        }
    outers = [json.loads((RUN_DIR / f"set_ranker/outer-{fold}.json").read_text()) for fold in range(5)]
    result = {
        "schema_version": 1,
        "status": "PASS_POST_GATE_DESCRIPTIVE",
        "gate_effect": "NONE",
        "comparisons": comparisons,
        "formal_override_behavior": formal_override_totals(outers),
    }
    path = RUN_DIR / "set_ranker_descriptive.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
