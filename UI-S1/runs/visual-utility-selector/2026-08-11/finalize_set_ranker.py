import json
from pathlib import Path

import numpy as np
import yaml

from adjudicate_anchor import ARMS, BENCHMARKS
from set_ranker_data import keyed


RUN_DIR = Path(__file__).resolve().parent
UTILITY_PATH = RUN_DIR.parents[2] / "runs/lsa-utility/2026-08-11/utility_main.json"


def load_outers():
    paths = [RUN_DIR / f"set_ranker/outer-{fold}.json" for fold in range(5)]
    pretest_paths = [RUN_DIR / f"set_ranker/outer-{fold}.pretest.json" for fold in range(5)]
    if not all(path.is_file() for path in paths):
        missing = [str(path) for path in paths if not path.is_file()]
        raise FileNotFoundError(missing)
    values = [json.loads(path.read_text()) for path in paths]
    if {value["outer_fold"] for value in values} != set(range(5)):
        raise ValueError("VUS-SR outer fold coverage mismatch")
    if any(value["status"] != "PASS_OUTER_COMPLETE" for value in values):
        raise ValueError("VUS-SR incomplete outer result")
    if not all(path.is_file() for path in pretest_paths):
        raise FileNotFoundError([str(path) for path in pretest_paths if not path.is_file()])
    for fold, path in enumerate(pretest_paths):
        record = json.loads(path.read_text())
        if (
            record.get("status") != "PASS_SELECTION_FROZEN_BEFORE_OUTER_LABEL_ACCESS"
            or record.get("outer_fold") != fold
            or fold in record.get("opened_development_label_folds", [])
            or len(record.get("opened_development_label_folds", [])) != 4
            or len(record.get("opened_development_label_sha256", {})) != 4
        ):
            raise ValueError(f"VUS-SR invalid fold-sealed pretest record: {path}")
    return sorted(values, key=lambda value: value["outer_fold"])


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
                    overlap = set(output[benchmark][arm][method]) & set(values)
                    if overlap:
                        raise ValueError(f"VUS-SR duplicate held-out rows: {benchmark}/{arm}/{method}/{len(overlap)}")
                    output[benchmark][arm][method].update(values)
    expected = {"mind2web": 2080, "screenspot_pro": 1581}
    for benchmark in BENCHMARKS:
        for arm in ARMS:
            keys = [set(output[benchmark][arm][method]) for method in output[benchmark][arm]]
            if any(len(values) != expected[benchmark] for values in keys) or len({frozenset(values) for values in keys}) != 1:
                raise ValueError(f"VUS-SR merged coverage mismatch: {benchmark}/{arm}")
    return output


def paired_samples(public, benchmark, arm, left, right, resamples, seed):
    metadata = {
        row["row_id"]: row
        for row in public.values()
        if row["benchmark"] == benchmark and row["arm"] == arm
    }
    if set(left) != set(right) or set(left) != set(metadata):
        raise ValueError(f"VUS-SR paired coverage mismatch: {benchmark}/{arm}")
    by_fold_group = {}
    for row_id, row in metadata.items():
        by_fold_group.setdefault(row["fold"], {}).setdefault(row["group"], []).append(row_id)
    differences = {row_id: int(left[row_id]) - int(right[row_id]) for row_id in left}
    generator = np.random.default_rng(seed)
    samples = np.empty(resamples, dtype=np.float64)
    for index in range(resamples):
        selected = []
        for fold in sorted(by_fold_group):
            groups = sorted(by_fold_group[fold])
            for group in generator.choice(groups, size=len(groups), replace=True):
                selected.extend(by_fold_group[fold][group])
        samples[index] = np.mean([differences[row_id] for row_id in selected])
    return {
        "point_delta": float(np.mean(list(differences.values()))),
        "ci_99": [float(np.quantile(samples, 0.005)), float(np.quantile(samples, 0.995))],
        "wins": sum(value > 0 for value in differences.values()),
        "losses": sum(value < 0 for value in differences.values()),
        "rows": len(differences),
        "resamples": resamples,
        "seed": seed,
    }, samples


def main():
    config = yaml.safe_load((RUN_DIR / "configs/set_ranker_prereg.yaml").read_text())
    public = keyed(RUN_DIR / "data/public_records.jsonl")
    outers = load_outers()
    outputs = merge_outputs(outers)
    utility = json.loads(UTILITY_PATH.read_text())["main"]["outputs"]
    controls = {
        "CEV_A": {
            benchmark: {arm: outputs[benchmark][arm]["fallback"] for arm in ARMS}
            for benchmark in BENCHMARKS
        },
        "Utility_LSA": {
            benchmark: {arm: utility[benchmark][arm]["safe"] for arm in ARMS}
            for benchmark in BENCHMARKS
        },
    }
    comparisons = {benchmark: {} for benchmark in BENCHMARKS}
    sample_cache = {control: {} for control in controls}
    for control_index, (control, values) in enumerate(controls.items()):
        for benchmark_index, benchmark in enumerate(BENCHMARKS):
            arm_samples = []
            points = []
            seed = (20260891 if benchmark == "mind2web" else 20260892) + control_index * 100
            for arm_index, arm in enumerate(ARMS):
                comparison, samples = paired_samples(
                    public, benchmark, arm,
                    outputs[benchmark][arm]["safe"], values[benchmark][arm],
                    10000, seed + arm_index,
                )
                comparisons[benchmark][f"{arm}_minus_{control}"] = comparison
                arm_samples.append(samples)
                points.append(comparison["point_delta"])
            equal_arm_samples = np.mean(np.stack(arm_samples), axis=0)
            comparisons[benchmark][f"equal_arm_mean_minus_{control}"] = {
                "point_delta": float(np.mean(points)),
                "ci_99": [float(np.quantile(equal_arm_samples, 0.005)), float(np.quantile(equal_arm_samples, 0.995))],
            }
            sample_cache[control][benchmark] = equal_arm_samples
    accuracy = {
        benchmark: {
            arm: {
                method: float(np.mean(list(values.values())))
                for method, values in outputs[benchmark][arm].items()
            }
            for arm in ARMS
        }
        for benchmark in BENCHMARKS
    }
    mde = config["mde"]
    cell_safety = {
        benchmark: {
            arm: (
                comparisons[benchmark][f"{arm}_minus_CEV_A"]["ci_99"][1] >= 0
                or abs(comparisons[benchmark][f"{arm}_minus_CEV_A"]["point_delta"]) < mde[benchmark]
            )
            for arm in ARMS
        }
        for benchmark in BENCHMARKS
    }
    standardized_utility = np.mean(np.stack([
        sample_cache["Utility_LSA"][benchmark] / mde[benchmark]
        for benchmark in BENCHMARKS
    ]), axis=0)
    one_benchmark_gain = any(
        comparisons[benchmark]["equal_arm_mean_minus_CEV_A"]["point_delta"] >= 0.01
        for benchmark in BENCHMARKS
    )
    other_noninferior = all(
        comparisons[benchmark]["equal_arm_mean_minus_CEV_A"]["ci_99"][1] >= 0
        and comparisons[benchmark]["equal_arm_mean_minus_CEV_A"]["point_delta"] >= -mde[benchmark]
        for benchmark in BENCHMARKS
    )
    gates = {
        "SR1_all_cells_safe": all(value for benchmark in cell_safety.values() for value in benchmark.values()),
        "SR1_cells": cell_safety,
        "SR2_one_benchmark_gain_at_least_1pp": one_benchmark_gain,
        "SR3_other_benchmark_noninferior": other_noninferior,
        "SR4_vs_Utility_LSA_balanced_ci_positive": float(np.quantile(standardized_utility, 0.005)) > 0,
        "SR4_vs_Utility_LSA": {
            "point": float(np.mean(standardized_utility)),
            "ci_99": [float(np.quantile(standardized_utility, 0.005)), float(np.quantile(standardized_utility, 0.995))],
        },
    }
    promoted = gates["SR1_all_cells_safe"] and gates["SR2_one_benchmark_gain_at_least_1pp"] and gates["SR3_other_benchmark_noninferior"] and gates["SR4_vs_Utility_LSA_balanced_ci_positive"]
    result = {
        "schema_version": 1,
        "status": "PASS_ADJUDICATED",
        "outcome": "VUS_SET_RANKER_METHOD_CANDIDATE" if promoted else "PROCEED_TO_FULL_LORA",
        "accuracy": accuracy,
        "comparisons": comparisons,
        "gates": gates,
        "outer_selections": [
            {
                "outer_fold": outer["outer_fold"],
                "config_id": outer["selected"]["config_id"],
                "selection_objective": outer["selected"]["selection_objective"],
                "inner_epochs": outer["selected_inner_epochs"],
                "final_epochs": outer["final_epochs"],
            }
            for outer in outers
        ],
        "outputs": outputs,
    }
    path = RUN_DIR / "set_ranker_adjudication.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: result[key] for key in ("outcome", "gates", "accuracy", "outer_selections")}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
