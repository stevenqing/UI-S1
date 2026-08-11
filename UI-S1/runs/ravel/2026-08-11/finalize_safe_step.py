import argparse
import json
import sys
from pathlib import Path

import numpy as np


RUN_DIR = Path(__file__).resolve().parent
ROOT = RUN_DIR.parents[2]
VUS = ROOT / "runs/visual-utility-selector/2026-08-11"
sys.path.insert(0, str(VUS))

from finalize_set_ranker import paired_samples
from set_ranker_data import keyed


ARMS = ("C_uni", "C_cond", "C_rand", "C_self")
BENCHMARKS = ("mind2web", "screenspot_pro")


def load_mode(mode):
    values = []
    for fold in range(5):
        path = RUN_DIR / f"safe_step/{mode}/outer-{fold}.json"
        pretest = RUN_DIR / f"safe_step/{mode}/outer-{fold}.pretest.json"
        if not path.is_file() or not pretest.is_file():
            raise FileNotFoundError(path)
        value = json.loads(path.read_text())
        seal = json.loads(pretest.read_text())
        if (
            value["status"] != "PASS_OUTER_COMPLETE"
            or value.get("ravel_evidence_mode") != mode
            or seal["status"] != "PASS_SELECTION_FROZEN_BEFORE_OUTER_LABEL_ACCESS"
            or fold in seal["opened_development_label_folds"]
        ):
            raise ValueError(f"RAVEL invalid safe-step outer: {mode}/{fold}")
        values.append(value)
    return values


def merge_outputs(outers):
    output = {
        benchmark: {arm: {method: {} for method in ("safe", "direct", "fallback")} for arm in ARMS}
        for benchmark in BENCHMARKS
    }
    for outer in outers:
        for benchmark in BENCHMARKS:
            for arm in ARMS:
                for method in output[benchmark][arm]:
                    values = outer["outputs"][benchmark][arm][method]
                    if set(output[benchmark][arm][method]) & set(values):
                        raise ValueError(f"RAVEL duplicate row: {benchmark}/{arm}/{method}")
                    output[benchmark][arm][method].update(values)
    expected = {"mind2web": 2080, "screenspot_pro": 1581}
    for benchmark in BENCHMARKS:
        for arm in ARMS:
            if any(len(output[benchmark][arm][method]) != expected[benchmark] for method in output[benchmark][arm]):
                raise ValueError(f"RAVEL safe-step coverage mismatch: {benchmark}/{arm}")
    return output


def compare(public, left, right, seed_offset):
    comparisons = {benchmark: {} for benchmark in BENCHMARKS}
    samples_by_benchmark = {}
    for benchmark_index, benchmark in enumerate(BENCHMARKS):
        samples_by_arm = []
        points = []
        seed = (20260891 if benchmark == "mind2web" else 20260892) + seed_offset
        for arm_index, arm in enumerate(ARMS):
            result, samples = paired_samples(
                public, benchmark, arm,
                left[benchmark][arm]["safe"], right[benchmark][arm]["safe"],
                10000, seed + arm_index,
            )
            comparisons[benchmark][arm] = result
            samples_by_arm.append(samples)
            points.append(result["point_delta"])
        equal_arm = np.mean(np.stack(samples_by_arm), axis=0)
        comparisons[benchmark]["equal_arm"] = {
            "point_delta": float(np.mean(points)),
            "ci_99": [float(np.quantile(equal_arm, 0.005)), float(np.quantile(equal_arm, 0.995))],
        }
        samples_by_benchmark[benchmark] = equal_arm
    return comparisons, samples_by_benchmark


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("local", "random"), required=True)
    args = parser.parse_args()
    public = keyed(VUS / "data/public_records.jsonl")
    outputs = merge_outputs(load_mode(args.mode))
    vus = json.loads((VUS / "set_ranker_adjudication.json").read_text())["outputs"]
    comparisons, samples = compare(public, outputs, vus, 700 if args.mode == "local" else 800)
    result = {
        "schema_version": 1,
        "status": "PASS_RAVEL_SAFE_STEP_ADJUDICATED",
        "mode": args.mode,
        "comparisons_vs_VUS_SR": comparisons,
        "outputs": outputs,
        "outer_selections": [
            {
                "outer_fold": outer["outer_fold"],
                "config_id": outer["selected"]["config_id"],
                "final_epochs": outer["final_epochs"],
                "predictions_sha256": outer["ravel_predictions_sha256"],
            }
            for outer in load_mode(args.mode)
        ],
    }
    if args.mode == "local" and (RUN_DIR / "safe_step/random/outer-0.json").is_file():
        random_outputs = merge_outputs(load_mode("random"))
        random_comparisons, _ = compare(public, outputs, random_outputs, 900)
        result["comparisons_vs_random_center"] = random_comparisons
    path = RUN_DIR / f"safe_step/{args.mode}_adjudication.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "mode": args.mode,
        "vs_VUS_SR": comparisons,
        "vs_random_center": result.get("comparisons_vs_random_center"),
        "outer_selections": result["outer_selections"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
