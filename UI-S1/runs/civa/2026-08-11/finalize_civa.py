import json
import sys
from pathlib import Path

import numpy as np
import yaml


RUN_DIR = Path(__file__).resolve().parent
ROOT = RUN_DIR.parents[2]
VUS = ROOT / "runs/visual-utility-selector/2026-08-11"
sys.path.insert(0, str(VUS))

from finalize_set_ranker import paired_samples
from set_ranker_data import keyed
from civa_data import ARMS, BENCHMARKS, validate_config
from civa_train import ALL_VARIANTS, CONFIG_PATH


MDE = {"mind2web": 0.006106589385659482, "screenspot_pro": 0.007}


def load_outers():
    values = []
    for fold in range(5):
        path = RUN_DIR / f"outer/outer-{fold}.json"
        pretest = RUN_DIR / f"outer/outer-{fold}.pretest.json"
        if not path.is_file() or not pretest.is_file():
            raise FileNotFoundError(path)
        value = json.loads(path.read_text())
        seal = json.loads(pretest.read_text())
        if (
            value.get("status") != "PASS_CIVA_OUTER_COMPLETE"
            or value.get("outer_fold") != fold
            or seal.get("status") != "PASS_CIVA_SELECTION_FROZEN"
            or fold in seal.get("opened_development_folds", [])
        ):
            raise ValueError(f"invalid CIVA outer: {fold}")
        values.append(value)
    return values


def merge(outers):
    output = {
        variant: {
            benchmark: {arm: {method: {} for method in ("policy", "baseline")} for arm in ARMS}
            for benchmark in BENCHMARKS
        }
        for variant in ALL_VARIANTS
    }
    for outer in outers:
        for variant in ALL_VARIANTS:
            for benchmark in BENCHMARKS:
                for arm in ARMS:
                    for method in ("policy", "baseline"):
                        values = outer["outputs"][variant][benchmark][arm][method]
                        if set(output[variant][benchmark][arm][method]) & set(values):
                            raise ValueError("duplicate CIVA row")
                        output[variant][benchmark][arm][method].update(values)
    expected = {"mind2web": 2080, "screenspot_pro": 1581}
    for variant in ALL_VARIANTS:
        for benchmark in BENCHMARKS:
            for arm in ARMS:
                if any(len(output[variant][benchmark][arm][method]) != expected[benchmark] for method in ("policy", "baseline")):
                    raise ValueError(f"CIVA coverage mismatch: {variant}/{benchmark}/{arm}")
    return output


def comparisons(public, left, right, seed_offset):
    report = {benchmark: {} for benchmark in BENCHMARKS}
    samples_by_benchmark = {}
    for benchmark in BENCHMARKS:
        arm_samples = []
        points = []
        seed = (20260911 if benchmark == "mind2web" else 20260912) + seed_offset
        for arm_index, arm in enumerate(ARMS):
            value, samples = paired_samples(
                public, benchmark, arm, left[benchmark][arm], right[benchmark][arm],
                10000, seed + arm_index,
            )
            report[benchmark][arm] = value
            arm_samples.append(samples)
            points.append(value["point_delta"])
        equal_arm = np.mean(np.stack(arm_samples), axis=0)
        report[benchmark]["equal_arm"] = {
            "point_delta": float(np.mean(points)),
            "ci_99": [float(np.quantile(equal_arm, 0.005)), float(np.quantile(equal_arm, 0.995))],
        }
        samples_by_benchmark[benchmark] = equal_arm
    balanced = np.mean(np.stack([
        samples_by_benchmark[benchmark] / MDE[benchmark] for benchmark in BENCHMARKS
    ]), axis=0)
    return report, {
        "point": float(np.mean(balanced)),
        "ci_99": [float(np.quantile(balanced, 0.005)), float(np.quantile(balanced, 0.995))],
    }


def main():
    config = yaml.safe_load(CONFIG_PATH.read_text())
    validate_config(config)
    if config["mde"] != MDE:
        raise ValueError("CIVA MDE mismatch")
    outputs = merge(load_outers())
    public = keyed(VUS / "data/public_records.jsonl")
    full = {benchmark: {arm: outputs["REAL_FULL"][benchmark][arm]["policy"] for arm in ARMS} for benchmark in BENCHMARKS}
    controls = {
        "BASELINE": {benchmark: {arm: outputs["REAL_FULL"][benchmark][arm]["baseline"] for arm in ARMS} for benchmark in BENCHMARKS},
        "MATCHED_RANDOM": {benchmark: {arm: outputs["MATCHED_RANDOM"][benchmark][arm]["policy"] for arm in ARMS} for benchmark in BENCHMARKS},
        "PLACEBO_FULL": {benchmark: {arm: outputs["PLACEBO_FULL"][benchmark][arm]["policy"] for arm in ARMS} for benchmark in BENCHMARKS},
        "REAL_NO_TEXT": {benchmark: {arm: outputs["REAL_NO_TEXT"][benchmark][arm]["policy"] for arm in ARMS} for benchmark in BENCHMARKS},
        "REAL_TEXT_ONLY": {benchmark: {arm: outputs["REAL_TEXT_ONLY"][benchmark][arm]["policy"] for arm in ARMS} for benchmark in BENCHMARKS},
    }
    comparison = {}
    balanced = {}
    for index, (name, control) in enumerate(controls.items()):
        comparison[name], balanced[name] = comparisons(public, full, control, 1000 + index * 100)

    baseline = comparison["BASELINE"]
    positive_benchmarks = [benchmark for benchmark in BENCHMARKS if baseline[benchmark]["equal_arm"]["ci_99"][0] > 0]
    other_noninferior = all(
        baseline[benchmark][arm]["ci_99"][0] > -MDE[benchmark]
        for benchmark in BENCHMARKS if benchmark not in positive_benchmarks
        for arm in ARMS
    )
    gates = {
        "CIVA_1_balanced_ci_positive_vs_baseline": balanced["BASELINE"]["ci_99"][0] > 0,
        "CIVA_2_one_benchmark_positive_other_cells_noninferior": bool(positive_benchmarks) and other_noninferior,
        "CIVA_3_balanced_ci_positive_vs_matched_random": balanced["MATCHED_RANDOM"]["ci_99"][0] > 0,
        "CIVA_4_balanced_ci_positive_vs_placebo": balanced["PLACEBO_FULL"]["ci_99"][0] > 0,
        "CIVA_5_balanced_ci_positive_vs_no_text": balanced["REAL_NO_TEXT"]["ci_99"][0] > 0,
        "CIVA_6_all_cells_noninferior_vs_baseline": all(
            baseline[benchmark][arm]["ci_99"][0] > -MDE[benchmark]
            for benchmark in BENCHMARKS for arm in ARMS
        ),
    }
    result = {
        "schema_version": 1,
        "status": "PASS_CIVA_ADJUDICATED",
        "outcome": "CIVA_ADMISSION_LEARNABLE" if all(gates.values()) else "CIVA_ADMISSION_NOT_SUPPORTED",
        "gates": gates,
        "comparisons": comparison,
        "balanced": balanced,
        "positive_benchmarks": positive_benchmarks,
        "outer_thresholds": [outer["thresholds"] for outer in load_outers()],
        "outer_test": [outer["test"] for outer in load_outers()],
        "outputs": outputs,
    }
    path = RUN_DIR / "civa_adjudication.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "outcome": result["outcome"], "gates": gates,
        "vs_baseline": comparison["BASELINE"], "balanced": balanced,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()