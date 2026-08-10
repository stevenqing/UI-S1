import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml


RUN_DIR = Path(__file__).resolve().parent
ROOT = RUN_DIR.parents[2]
LSA = ROOT / "runs/lsa/2026-08-10"
CEV = ROOT / "runs/cev/2026-08-09"
sys.path.insert(0, str(LSA))
sys.path.insert(0, str(CEV))

from cev import Candidate as CEVCandidate
from cev import select as cev_select
from lsa_common import evaluation_rows, feature_names, load_rows, reliability_statistics
from lsa_train import fit_estimator


ARMS = ("C_cond", "C_rand", "C_self")
BENCHMARKS = ("mind2web", "screenspot_pro")


def no_action_indices():
    return [index for index, name in enumerate(feature_names()) if "action" not in name]


def transferred_reliability(cuni_rows, train_ids, target_rows):
    sums, counts = reliability_statistics(cuni_rows, train_ids)
    lineage_sums = defaultdict(float)
    lineage_counts = defaultdict(int)
    source_lineage = {}
    for row_id in train_ids:
        for candidate in cuni_rows[row_id].candidates:
            source_lineage[candidate.source] = candidate.lineage
    for source, value in sums.items():
        lineage = source_lineage[source]
        lineage_sums[lineage] += value
        lineage_counts[lineage] += counts[source]
    target_sources = {}
    for row in target_rows.values():
        for candidate in row.candidates:
            target_sources[candidate.source] = candidate.lineage
    for source, lineage in target_sources.items():
        if source not in sums:
            sums[source] = lineage_sums[lineage]
            counts[source] = lineage_counts[lineage]
    return sums, counts


def exact_fallback_index(row, target_sums, target_counts, fold_record):
    candidates = []
    for candidate in row.candidates:
        reliability = target_sums[candidate.source] / target_counts[candidate.source]
        candidates.append(CEVCandidate(
            action=candidate.action,
            coordinate=candidate.baseline_coordinate,
            parameter=candidate.parameter,
            source=candidate.source,
            reliability=reliability,
            order=candidate.order,
            payload=candidate.order,
            parse_ok=candidate.parse_ok,
            lineage=candidate.lineage,
        ))
    configuration = fold_record["global_configuration"]
    if row.benchmark == "screenspot_pro":
        threshold = configuration["coordinate_tolerance"]
    else:
        scale = fold_record["outer_refit_scale"]
        multiplier = configuration.get("coordinate_multiplier", 1.0)
        threshold = (scale[0] * multiplier, scale[1] * multiplier)
    prediction, _ = cev_select(
        candidates,
        configuration["granularity"],
        threshold,
        configuration.get("parameter_threshold", 1.0),
    )
    return int(prediction.payload)


def paired_samples(rows, differences, resamples, seed):
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
        "resamples": resamples,
        "seed": seed,
        "rows": len(differences),
    }, samples


def main():
    config = yaml.safe_load((RUN_DIR / "configs/confirm.yaml").read_text())
    if config["status"] != "FROZEN_BEFORE_CONFIRMATION_RESULTS":
        raise ValueError("confirmation protocol is not frozen")
    cuni = load_rows("C_uni")
    target = {arm: load_rows(arm) for arm in ARMS}
    cev = json.loads((CEV / "cev_main.json").read_text())
    feature_indices = no_action_indices()
    outputs = {
        benchmark: {arm: {"safe": {}, "direct": {}, "fallback": {}, "dev_selection": {}} for arm in ARMS}
        for benchmark in BENCHMARKS
    }
    folds = []
    for outer_fold in range(5):
        development_folds = [fold for fold in range(5) if fold != outer_fold]
        train_ids = {
            benchmark: [row_id for row_id, row in cuni[benchmark].items() if row.fold in development_folds]
            for benchmark in BENCHMARKS
        }
        model, _, training = fit_estimator(
            config["model"]["estimator_by_outer_fold"][outer_fold],
            yaml.safe_load((LSA / "configs/lsa_prereg.yaml").read_text()),
            cuni,
            train_ids,
            feature_indices,
        )
        threshold = config["model"]["threshold_by_outer_fold"][outer_fold]
        fold_report = {
            "outer_fold": outer_fold,
            "model_id": config["model"]["estimator_by_outer_fold"][outer_fold],
            "threshold": threshold,
            "training": training,
            "arms": {},
        }
        for arm in ARMS:
            fold_report["arms"][arm] = {}
            for benchmark in BENCHMARKS:
                target_rows = target[arm][benchmark]
                test_ids = [row_id for row_id, row in target_rows.items() if row.fold == outer_fold]
                feature_reliability = transferred_reliability(cuni[benchmark], train_ids[benchmark], target_rows)
                evaluation = evaluation_rows(
                    {benchmark: target_rows},
                    {benchmark: test_ids},
                    {benchmark: feature_reliability},
                    feature_indices,
                )[benchmark]
                target_dev_ids = [row_id for row_id, row in target_rows.items() if row.fold != outer_fold]
                target_sums, target_counts = reliability_statistics(target_rows, target_dev_ids)
                fold_record = cev[benchmark]["folds"][outer_fold]["arms"][arm]
                wins = losses = overrides = 0
                for row_id, values in evaluation.items():
                    probabilities = model.predict_proba(values["features"])[:, 1]
                    learned_index = int(np.argmax(probabilities))
                    fallback_index = exact_fallback_index(target_rows[row_id], target_sums, target_counts, fold_record)
                    frozen_fallback = cev["outputs"][benchmark][arm]["CEV_A"][row_id]
                    if bool(values["labels"][fallback_index]) != frozen_fallback:
                        raise ValueError(f"LT-K1 fallback mismatch: {benchmark}/{arm}/{row_id}")
                    margin = float(probabilities[learned_index] - probabilities[fallback_index])
                    override = learned_index != fallback_index and margin >= threshold
                    safe = bool(values["labels"][learned_index] if override else values["labels"][fallback_index])
                    direct = bool(values["labels"][learned_index])
                    outputs[benchmark][arm]["safe"][row_id] = safe
                    outputs[benchmark][arm]["direct"][row_id] = direct
                    outputs[benchmark][arm]["fallback"][row_id] = frozen_fallback
                    outputs[benchmark][arm]["dev_selection"][row_id] = cev["outputs"][benchmark][arm]["dev_selection"][row_id]
                    overrides += int(override)
                    wins += int(override and safe and not frozen_fallback)
                    losses += int(override and frozen_fallback and not safe)
                fold_report["arms"][arm][benchmark] = {
                    "rows": len(test_ids),
                    "safe_accuracy": float(np.mean(list(outputs[benchmark][arm]["safe"][row_id] for row_id in test_ids))),
                    "fallback_accuracy": float(np.mean(list(outputs[benchmark][arm]["fallback"][row_id] for row_id in test_ids))),
                    "overrides": overrides,
                    "override_rate": overrides / len(test_ids),
                    "wins": wins,
                    "losses": losses,
                }
        folds.append(fold_report)
        print(f"completed no-action transfer outer_fold={outer_fold}", flush=True)

    comparisons = {benchmark: {} for benchmark in BENCHMARKS}
    arm_samples = {benchmark: {"CEV_A": {}, "dev_selection": {}} for benchmark in BENCHMARKS}
    for benchmark_index, benchmark in enumerate(BENCHMARKS):
        seed_key = "mind2web_seed" if benchmark == "mind2web" else "screenspot_seed"
        seed = config["statistics"][seed_key]
        for arm_index, arm in enumerate(ARMS):
            for control_index, control in enumerate(("CEV_A", "dev_selection")):
                key = "fallback" if control == "CEV_A" else "dev_selection"
                differences = {
                    row_id: int(outputs[benchmark][arm]["safe"][row_id]) - int(outputs[benchmark][arm][key][row_id])
                    for row_id in outputs[benchmark][arm]["safe"]
                }
                result, samples = paired_samples(target[arm][benchmark], differences, config["statistics"]["resamples"], seed + arm_index * 10 + control_index)
                comparisons[benchmark][f"{arm}_minus_{control}"] = result
                arm_samples[benchmark][control][arm] = samples
        for control in ("CEV_A", "dev_selection"):
            values = np.mean(np.stack([arm_samples[benchmark][control][arm] for arm in ARMS]), axis=0)
            points = [comparisons[benchmark][f"{arm}_minus_{control}"]["point_delta"] for arm in ARMS]
            comparisons[benchmark][f"equal_arm_mean_minus_{control}"] = {
                "point_delta": float(np.mean(points)),
                "ci_99": [float(np.quantile(values, 0.005)), float(np.quantile(values, 0.995))],
                "resamples": config["statistics"]["resamples"],
            }

    mde = config["mde"]
    cell_safety = {
        benchmark: {
            arm: comparisons[benchmark][f"{arm}_minus_CEV_A"]["ci_99"][1] >= 0
            or abs(comparisons[benchmark][f"{arm}_minus_CEV_A"]["point_delta"]) < mde[benchmark]
            for arm in ARMS
        }
        for benchmark in BENCHMARKS
    }
    t1 = all(value for benchmark in cell_safety.values() for value in benchmark.values())
    t2 = comparisons["mind2web"]["equal_arm_mean_minus_CEV_A"]["ci_99"][0] > 0
    screen_mean = comparisons["screenspot_pro"]["equal_arm_mean_minus_CEV_A"]
    t3 = screen_mean["ci_99"][1] >= 0 and screen_mean["point_delta"] >= -mde["screenspot_pro"]
    standardized = np.mean(np.stack([
        np.mean(np.stack([arm_samples[benchmark]["dev_selection"][arm] for arm in ARMS]), axis=0) / mde[benchmark]
        for benchmark in BENCHMARKS
    ]), axis=0)
    no_mde_loss = all(comparisons[benchmark][f"{arm}_minus_dev_selection"]["point_delta"] >= -mde[benchmark] for benchmark in BENCHMARKS for arm in ARMS)
    t4 = float(np.quantile(standardized, 0.005)) > 0 and no_mde_loss
    outcome = "CONFIRMED_SAFE_LEARNED_AGGREGATOR" if t1 and t2 and t3 and t4 else "CONFIRMED_VS_CEV_ONLY" if t1 and t2 and t3 else "PARTIAL_TRANSFER" if t1 else "FAILED_CROSS_ARM_CONFIRMATION"
    result = {
        "schema_version": 1,
        "status": "PASS_CONFIRMATION_COMPLETE",
        "config": "configs/confirm.yaml",
        "accuracy": {
            benchmark: {
                arm: {method: float(np.mean(list(values.values()))) for method, values in methods.items()}
                for arm, methods in arms.items()
            }
            for benchmark, arms in outputs.items()
        },
        "comparisons": comparisons,
        "gates": {
            "T1": t1,
            "T1_cells": cell_safety,
            "T2": t2,
            "T3": t3,
            "T4": t4,
            "T4_balanced_standardized": {
                "point": float(np.mean(standardized)),
                "ci_99": [float(np.quantile(standardized, 0.005)), float(np.quantile(standardized, 0.995))],
                "no_MDE_loss": no_mde_loss,
            },
            "LT_K1": False,
            "LT_K2": not t1,
            "LT_K3": not t2,
            "LT_K4": not t3,
        },
        "outcome": outcome,
        "folds": folds,
        "outputs": outputs,
    }
    (RUN_DIR / "confirmation.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"outcome": outcome, "gates": result["gates"], "accuracy": result["accuracy"], "comparisons": comparisons}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()