import json
from pathlib import Path

import numpy as np
import yaml

from utility_common import ARMS, BENCHMARKS, load_banks


RUN_DIR = Path(__file__).resolve().parent
ROOT = RUN_DIR.parents[2]
CEV_PATH = ROOT / "runs/cev/2026-08-09/cev_main.json"
LSA_PATH = ROOT / "runs/lsa/2026-08-10/lsa_variants.json"
CONFIRM_PATH = ROOT / "runs/lsa-confirm/2026-08-10/confirmation.json"


def paired_samples(rows, left, right, resamples, seed):
    differences = {row_id: int(left[row_id]) - int(right[row_id]) for row_id in left}
    by_fold_group = {}
    for row_id in differences:
        row = rows[row_id]
        by_fold_group.setdefault(row.fold, {}).setdefault(row.group, []).append(row_id)
    rng = np.random.default_rng(seed)
    samples = np.empty(resamples, dtype=np.float64)
    for index in range(resamples):
        selected = []
        for fold in sorted(by_fold_group):
            groups = sorted(by_fold_group[fold])
            for group in rng.choice(groups, size=len(groups), replace=True):
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


def equal_arm(samples_by_arm, points_by_arm):
    values = np.mean(np.stack([samples_by_arm[arm] for arm in ARMS]), axis=0)
    return {
        "point_delta": float(np.mean([points_by_arm[arm] for arm in ARMS])),
        "ci_99": [float(np.quantile(values, 0.005)), float(np.quantile(values, 0.995))],
    }, values


def correctness_outputs():
    discovery = json.loads(LSA_PATH.read_text())["variants"]["no_action"]["outputs"]
    confirmation = json.loads(CONFIRM_PATH.read_text())["outputs"]
    output = {benchmark: {} for benchmark in BENCHMARKS}
    for benchmark in BENCHMARKS:
        output[benchmark]["C_uni"] = discovery[benchmark]["safe"]
        for arm in ARMS[1:]:
            output[benchmark][arm] = confirmation[benchmark][arm]["safe"]
    return output


def main():
    config = yaml.safe_load((RUN_DIR / "configs/utility_prereg.yaml").read_text())
    main_result = json.loads((RUN_DIR / "utility_main.json").read_text())["main"]
    cev = json.loads(CEV_PATH.read_text())
    banks = load_banks()
    prior = correctness_outputs()
    comparisons = {benchmark: {} for benchmark in BENCHMARKS}
    sample_cache = {benchmark: {} for benchmark in BENCHMARKS}
    controls = ("CEV_A", "dev_selection", "correctness_LSA")
    for benchmark in BENCHMARKS:
        seed = config["statistics"]["mind2web_seed" if benchmark == "mind2web" else "screenspot_seed"]
        for control_index, control in enumerate(controls):
            arm_samples = {}
            arm_points = {}
            for arm_index, arm in enumerate(ARMS):
                left = main_result["outputs"][benchmark][arm]["safe"]
                right = prior[benchmark][arm] if control == "correctness_LSA" else cev["outputs"][benchmark][arm][control]
                result, samples = paired_samples(
                    banks[arm][benchmark], left, right,
                    config["statistics"]["resamples"], seed + control_index * 100 + arm_index,
                )
                comparisons[benchmark][f"{arm}_minus_{control}"] = result
                arm_samples[arm] = samples
                arm_points[arm] = result["point_delta"]
            mean, samples = equal_arm(arm_samples, arm_points)
            comparisons[benchmark][f"equal_arm_mean_minus_{control}"] = mean
            sample_cache[benchmark][control] = samples

        direct_arm_samples = {}
        direct_arm_points = {}
        for arm_index, arm in enumerate(ARMS):
            result, samples = paired_samples(
                banks[arm][benchmark],
                main_result["outputs"][benchmark][arm]["safe"],
                main_result["outputs"][benchmark][arm]["direct"],
                config["statistics"]["resamples"], seed + 400 + arm_index,
            )
            comparisons[benchmark][f"{arm}_safe_minus_direct"] = result
            direct_arm_samples[arm] = samples
            direct_arm_points[arm] = result["point_delta"]
        comparisons[benchmark]["equal_arm_mean_safe_minus_direct"], _ = equal_arm(direct_arm_samples, direct_arm_points)

    mde = config["mde"]
    cell_safety = {
        benchmark: {
            arm: comparisons[benchmark][f"{arm}_minus_CEV_A"]["ci_99"][1] >= 0
            or abs(comparisons[benchmark][f"{arm}_minus_CEV_A"]["point_delta"]) < mde[benchmark]
            for arm in ARMS
        }
        for benchmark in BENCHMARKS
    }
    ur1 = all(value for benchmark in cell_safety.values() for value in benchmark.values())
    ur2 = comparisons["mind2web"]["equal_arm_mean_minus_CEV_A"]["ci_99"][0] > 0
    screen = comparisons["screenspot_pro"]["equal_arm_mean_minus_CEV_A"]
    ur3 = screen["ci_99"][1] >= 0 and screen["point_delta"] >= -mde["screenspot_pro"]
    balanced_dev = np.mean(np.stack([
        sample_cache[benchmark]["dev_selection"] / mde[benchmark]
        for benchmark in BENCHMARKS
    ]), axis=0)
    no_mde_loss = all(
        comparisons[benchmark][f"{arm}_minus_dev_selection"]["point_delta"] >= -mde[benchmark]
        for benchmark in BENCHMARKS for arm in ARMS
    )
    ur4 = float(np.quantile(balanced_dev, 0.005)) > 0 and no_mde_loss
    balanced_prior = np.mean(np.stack([
        sample_cache[benchmark]["correctness_LSA"] / mde[benchmark]
        for benchmark in BENCHMARKS
    ]), axis=0)
    prior_better = float(np.quantile(balanced_prior, 0.005)) > 0
    prior_ur2_failed = True
    ur5 = prior_better or (ur2 and prior_ur2_failed)
    selected_objectives = [fold["selected"]["objective"] for fold in main_result["folds"]]
    infinity_folds = sum(not np.isfinite(fold["selected"]["threshold"]) for fold in main_result["folds"])
    gates = {
        "UR1": ur1,
        "UR1_cells": cell_safety,
        "UR2": ur2,
        "UR3": ur3,
        "UR4": ur4,
        "UR4_balanced": {
            "point": float(np.mean(balanced_dev)),
            "ci_99": [float(np.quantile(balanced_dev, 0.005)), float(np.quantile(balanced_dev, 0.995))],
            "no_MDE_loss": no_mde_loss,
        },
        "UR5": ur5,
        "UR5_vs_correctness_LSA": {
            "point": float(np.mean(balanced_prior)),
            "ci_99": [float(np.quantile(balanced_prior, 0.005)), float(np.quantile(balanced_prior, 0.995))],
        },
        "UR_K1": False,
        "UR_K2": not ur1,
        "UR_K3": infinity_folds >= 3,
        "UR_K4_pending_objective_OOF": selected_objectives.count("U_GRPO") == 0,
        "UR_K5": None,
        "selected_objectives": selected_objectives,
        "infinity_threshold_folds": infinity_folds,
    }
    if ur1 and ur2 and ur3 and ur4 and ur5:
        outcome = "UTILITY_METHOD_CANDIDATE"
    elif ur1 and ur2 and ur3:
        outcome = "UTILITY_IMPROVES_CEV_NOT_DEVSEL"
    elif ur1 and ur3:
        outcome = "SAFE_EXPLORATORY_OVERRIDE"
    else:
        outcome = "FAILED_UTILITY_TRAINING"
    result = {
        "schema_version": 1,
        "status": "PASS_ADJUDICATED",
        "accuracy": main_result["accuracy"],
        "comparisons": comparisons,
        "gates": gates,
        "outcome": outcome,
        "fold_selections": [fold["selected"] for fold in main_result["folds"]],
    }
    (RUN_DIR / "utility_adjudication.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"outcome": outcome, "gates": gates, "accuracy": main_result["accuracy"], "equal_arm": {benchmark: {key: value for key, value in comparisons[benchmark].items() if key.startswith("equal_arm")} for benchmark in BENCHMARKS}}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()