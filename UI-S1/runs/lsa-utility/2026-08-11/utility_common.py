import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


RUN_DIR = Path(__file__).resolve().parent
ROOT = RUN_DIR.parents[2]
LSA = ROOT / "runs/lsa/2026-08-10"
CEV = ROOT / "runs/cev/2026-08-09"
sys.path.insert(0, str(LSA))
sys.path.insert(0, str(CEV))

from cev import Candidate as CEVCandidate
from cev import select as cev_select
from lsa_common import feature_names, load_rows, reliability_statistics, row_features
from behavior_policy import apply_policy


ARMS = ("C_uni", "C_cond", "C_rand", "C_self")
BENCHMARKS = ("mind2web", "screenspot_pro")
OBJECTIVES = ("U_RAW", "U_GRPO", "U_HYBRID")


def load_banks():
    return {arm: load_rows(arm) for arm in ARMS}


def no_action_indices():
    return [index for index, name in enumerate(feature_names()) if "action" not in name]


def feature_indices_for_mode(feature_mode):
    names = feature_names()
    indices = no_action_indices()
    if feature_mode == "no_mvp":
        indices = [
            index for index in indices
            if not any(token in names[index] for token in (
                "coordinate_", "lineage_support", "row_coordinate_dispersion"
            ))
        ]
    return indices


def exact_fallback_index(row, sums, counts, fold_record, leave_one=False):
    candidates = []
    for candidate in row.candidates:
        total = sums[candidate.source] - (float(candidate.success) if leave_one else 0.0)
        count = counts[candidate.source] - (1 if leave_one else 0)
        reliability = total / count if count else 0.0
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


def reliability_by_arm(banks, ids_by_benchmark):
    return {
        arm: {
            benchmark: reliability_statistics(banks[arm][benchmark], ids)
            for benchmark, ids in ids_by_benchmark.items()
        }
        for arm in ARMS
    }


def pair_features(row, sums, counts, fallback_index, leave_one):
    indices = no_action_indices()
    values = row_features(row, sums, counts, leave_one=leave_one)[:, indices]
    fallback = values[fallback_index]
    repeated = np.repeat(fallback[None, :], len(values), axis=0)
    return np.concatenate([values, repeated, values - repeated, np.abs(values - repeated)], axis=1)


def transformed_features(row, sums, counts, fallback_index, leave_one, feature_mode):
    all_no_action = no_action_indices()
    selected = feature_indices_for_mode(feature_mode)
    values = row_features(row, sums, counts, leave_one=leave_one)[:, all_no_action]
    positions = [all_no_action.index(index) for index in selected]
    values = values[:, positions]
    if feature_mode == "absolute":
        return values
    fallback = values[fallback_index]
    repeated = np.repeat(fallback[None, :], len(values), axis=0)
    return np.concatenate([values, repeated, values - repeated, np.abs(values - repeated)], axis=1)


def utility_targets(row, fallback_index, objective):
    success = np.asarray([candidate.success for candidate in row.candidates], dtype=np.float32)
    utility = success - success[fallback_index]
    if objective == "U_RAW":
        target = utility
    else:
        std = float(np.std(utility, ddof=1))
        advantage = (utility - float(np.mean(utility))) / (std + 1e-4)
        target = advantage if objective == "U_GRPO" else 0.5 * utility + 0.5 * advantage
    return utility, target.astype(np.float32)


def training_matrix(banks, ids_by_benchmark, reliability, policies, objective, feature_mode="pair"):
    arrays = []
    targets = []
    weights = []
    report = {benchmark: {arm: 0 for arm in ARMS} for benchmark in BENCHMARKS}
    for benchmark in BENCHMARKS:
        active_rows = []
        for row_id in ids_by_benchmark[benchmark]:
            row_groups = []
            for arm in ARMS:
                row = banks[arm][benchmark][row_id]
                sums, counts = reliability[arm][benchmark]
                fallback = apply_policy(row, policies[benchmark][arm])
                utility, target = utility_targets(row, fallback, objective)
                if float(np.std(utility)) == 0.0:
                    continue
                features = transformed_features(row, sums, counts, fallback, leave_one=True, feature_mode=feature_mode)
                row_groups.append((arm, features, target))
                report[benchmark][arm] += 1
            if row_groups:
                active_rows.append(row_groups)
        benchmark_row_weight = 1.0 / max(1, len(active_rows))
        for row_groups in active_rows:
            arm_weight = benchmark_row_weight / len(row_groups)
            for arm, features, target in row_groups:
                arrays.append(features)
                targets.append(target)
                weights.append(np.full(len(target), arm_weight / len(target), dtype=np.float64))
    if not arrays:
        raise ValueError("no active utility groups")
    return np.concatenate(arrays), np.concatenate(targets), np.concatenate(weights), report


def evaluation_rows(banks, ids_by_benchmark, reliability, policies, feature_mode="pair"):
    output = {benchmark: {arm: {} for arm in ARMS} for benchmark in BENCHMARKS}
    for benchmark in BENCHMARKS:
        for arm in ARMS:
            sums, counts = reliability[arm][benchmark]
            for row_id in ids_by_benchmark[benchmark]:
                row = banks[arm][benchmark][row_id]
                fallback = apply_policy(row, policies[benchmark][arm])
                features = transformed_features(row, sums, counts, fallback, leave_one=False, feature_mode=feature_mode)
                output[benchmark][arm][row_id] = {
                    "features": features,
                    "labels": np.asarray([candidate.success for candidate in row.candidates], dtype=np.int8),
                    "fallback_index": fallback,
                }
    return output


def ids_for_folds(banks, folds):
    return {
        benchmark: [row_id for row_id, row in banks["C_uni"][benchmark].items() if row.fold in folds]
        for benchmark in BENCHMARKS
    }


def metadata(banks):
    return {
        benchmark: {
            row_id: {"fold": row.fold, "group": row.group}
            for row_id, row in banks["C_uni"][benchmark].items()
        }
        for benchmark in BENCHMARKS
    }


def load_cev():
    return json.loads((CEV / "cev_main.json").read_text())
