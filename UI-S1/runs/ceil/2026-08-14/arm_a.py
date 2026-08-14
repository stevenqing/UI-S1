import importlib.util
import json
import math
import multiprocessing as mp
import os
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml
from PIL import Image
from scipy.optimize import least_squares
from scipy.stats import spearmanr
from sklearn.isotonic import IsotonicRegression


RUN_DIR = Path(__file__).resolve().parent
ROOT = RUN_DIR.parents[2]
MASK_DIR = ROOT / "runs/mask/2026-08-14"
CLOSE_DIR = ROOT / "runs/close/2026-08-08"
XFER_DIR = ROOT / "runs/xfer/2026-08-07"
CONFIG_PATH = RUN_DIR / "configs/ceil_prereg.yaml"
PREFLIGHT_PATH = RUN_DIR / "PREFLIGHT.json"
OUTPUT_PATH = RUN_DIR / "ARM_A.json"
FIGURE_PATH = RUN_DIR / "ARM_A_CURVES.pdf"
CACHE_PATH = RUN_DIR / "ARM_A_PANEL_CACHE.npz"
sys.path.insert(0, str(MASK_DIR))
sys.path.insert(0, str(XFER_DIR))

from mask_common import b3_correct, load_rows as load_screen_rows, source_reliability


MASKS = np.arange(1, 1 << 12, dtype=np.int64)
SUBSET_INDICES = tuple(tuple(index for index in range(12) if mask & (1 << index)) for mask in MASKS)
K_VALUES = np.asarray([len(indices) for indices in SUBSET_INDICES], dtype=np.float64)
PAIR_INDICATORS = np.asarray([
    [float(left in indices and right in indices) for left in range(12) for right in range(left + 1, 12)]
    for indices in SUBSET_INDICES
], dtype=np.float64)
FULL_OFFSET = len(MASKS) - 1
IDEAL_INCREMENT = 3 / 2.2
PANEL_ORDER = (
    "screenspot_pro/C_uni", "mind2web/C_uni", "mind2web/C_cond",
    "mind2web/C_rand", "mind2web/C_self",
)
AGGREGATOR_ORDER = ("density", "majority")
_WORKER = {}
_BOOTSTRAP = {}


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def cohen_kappa_from_counts(total, left_fail, right_fail, both_fail):
    if total <= 0:
        raise ValueError("CEIL Arm A empty development bootstrap")
    observed = (both_fail + total - left_fail - right_fail + both_fail) / total
    left_rate = left_fail / total
    right_rate = right_fail / total
    expected = left_rate * right_rate + (1 - left_rate) * (1 - right_rate)
    if math.isclose(expected, 1.0):
        return 0.0, True
    return float((observed - expected) / (1 - expected)), False


def cross_fitted_neff(statistics, counts):
    fold_values = []
    fold_weights = []
    undefined = 0
    for outer_fold in range(5):
        development = statistics["group_folds"] != outer_fold
        selected = counts * development
        total = float(selected @ statistics["group_sizes"])
        failures = np.tensordot(selected, statistics["group_failures"], axes=1)
        joints = np.tensordot(selected, statistics["group_joints"], axes=1)
        kappas = []
        for left in range(12):
            for right in range(left + 1, 12):
                value, missing = cohen_kappa_from_counts(
                    total, failures[left], failures[right], joints[left, right]
                )
                kappas.append(value)
                undefined += int(missing)
        denominator = K_VALUES + 2 * (PAIR_INDICATORS @ np.asarray(kappas, dtype=np.float64))
        values = np.where(denominator > 0, np.square(K_VALUES) / denominator, np.nan)
        test_weight = float((counts * (statistics["group_folds"] == outer_fold)) @ statistics["group_sizes"])
        fold_values.append(values)
        fold_weights.append(test_weight)
    fold_values = np.asarray(fold_values)
    fold_weights = np.asarray(fold_weights)
    if np.any(fold_weights <= 0):
        raise ValueError("CEIL Arm A empty held-out fold bootstrap")
    valid = np.all(np.isfinite(fold_values), axis=0)
    values = np.sum(fold_values * fold_weights[:, None], axis=0) / fold_weights.sum()
    values[~valid] = np.nan
    return values, undefined


def unique_curve(x_values, y_values):
    grouped = defaultdict(list)
    for x_value, y_value in zip(x_values, y_values):
        if np.isfinite(x_value):
            grouped[float(x_value)].append(float(y_value))
    x_values = np.asarray(sorted(grouped), dtype=np.float64)
    y_values = np.asarray([np.mean(grouped[value]) for value in x_values], dtype=np.float64)
    return x_values, y_values


def parametric_fit(x_values, y_values):
    maximum = float(np.max(y_values))
    starts_a = (maximum, min(1.0, maximum + 0.05), 1.0)
    solutions = []
    def residual(parameters):
        a_value, b_value, c_value = parameters
        return a_value - b_value * np.exp(-c_value * x_values) - y_values
    for a_value in starts_a:
        for b_value in (0.01, 0.1, 0.5):
            for c_value in (0.01, 0.1, 1.0, 10.0):
                initial = np.asarray([a_value, b_value, c_value], dtype=np.float64)
                initial = np.minimum(np.maximum(initial, [0.0, 0.0, 1e-8]), [1.0, 1.0, 100.0])
                try:
                    fit = least_squares(
                        residual, initial, bounds=([0.0, 0.0, 1e-8], [1.0, 1.0, 100.0]),
                        xtol=1e-12, ftol=1e-12, gtol=1e-12, max_nfev=100000,
                    )
                except (ValueError, RuntimeError, FloatingPointError):
                    continue
                if fit.success and np.all(np.isfinite(fit.x)):
                    sse = float(np.sum(np.square(fit.fun)))
                    solutions.append((sse, *map(float, fit.x)))
    if not solutions:
        return None
    minimum = min(value[0] for value in solutions)
    return min((value for value in solutions if value[0] <= minimum + 1e-12), key=lambda value: value[1:])


def curve_summary(neff, accuracy):
    valid = np.isfinite(neff)
    x_values, y_values = unique_curve(neff[valid], accuracy[valid])
    full_x = float(neff[FULL_OFFSET])
    full_accuracy = float(accuracy[FULL_OFFSET])
    target_x = full_x + IDEAL_INCREMENT
    isotonic = IsotonicRegression(increasing=True, out_of_bounds="clip").fit(x_values, y_values)
    thresholds_x = np.asarray(isotonic.X_thresholds_, dtype=np.float64)
    thresholds_y = np.asarray(isotonic.y_thresholds_, dtype=np.float64)
    if len(thresholds_x) >= 2:
        slope = max(0.0, float(
            (thresholds_y[-1] - thresholds_y[-2]) / (thresholds_x[-1] - thresholds_x[-2])
        ))
        insufficient = False
    else:
        slope = 0.0
        insufficient = True
    finite_isotonic = float(np.clip(thresholds_y[-1] + slope * (target_x - thresholds_x[-1]), 0, 1))
    fit = parametric_fit(x_values, y_values)
    if fit is None:
        parametric = None
        delta_infinity = None
        finite_parametric = None
    else:
        sse, a_value, b_value, c_value = fit
        finite_parametric = float(a_value - b_value * math.exp(-c_value * target_x))
        delta_infinity = float(a_value - full_accuracy)
        parametric = {"SSE": sse, "a": a_value, "b": b_value, "c": c_value}
    differences = np.diff(y_values)
    rho = spearmanr(x_values, y_values).statistic if len(x_values) > 1 else 0.0
    return {
        "full_neff": full_x,
        "support_minimum": float(x_values.min()),
        "support_maximum": float(x_values.max()),
        "full_accuracy": full_accuracy,
        "target_x": target_x,
        "isotonic": {
            "thresholds_x": thresholds_x.tolist(),
            "thresholds_y": thresholds_y.tolist(),
            "boundary_slope": slope,
            "insufficient_thresholds": insufficient,
            "finite_target_prediction": finite_isotonic,
            "finite_target_gain": finite_isotonic - full_accuracy,
        },
        "parametric": parametric,
        "Delta_infinity": delta_infinity,
        "finite_parametric_prediction": finite_parametric,
        "finite_parametric_gain": None if finite_parametric is None else finite_parametric - full_accuracy,
        "raw_nonmonotonicity": {
            "unique_x": len(x_values),
            "adjacent_decrease_fraction": float(np.mean(differences < 0)) if len(differences) else 0.0,
            "maximum_adjacent_decrease": float(max(0.0, -np.min(differences))) if len(differences) else 0.0,
            "spearman_rho": float(rho),
        },
    }


def counts_for_bootstrap(statistics, resamples, seed):
    groups_by_fold = defaultdict(list)
    for index, fold in enumerate(statistics["group_folds"]):
        groups_by_fold[int(fold)].append(index)
    generator = np.random.default_rng(seed)
    output = np.zeros((resamples, len(statistics["group_folds"])), dtype=np.int16)
    for replicate in range(resamples):
        for fold in sorted(groups_by_fold):
            indices = np.asarray(groups_by_fold[fold], dtype=np.int64)
            selected = generator.choice(indices, size=len(indices), replace=True)
            output[replicate] += np.bincount(selected, minlength=output.shape[1]).astype(np.int16)
    return output


def summarize_counts(statistics, counts, aggregator_index):
    neff, _ = cross_fitted_neff(statistics, counts)
    total = float(counts @ statistics["group_sizes"])
    accuracy = np.tensordot(counts, statistics["group_outputs"][:, aggregator_index, :], axes=1) / total
    return curve_summary(neff, accuracy)


def bootstrap_worker(task):
    counts_batch, aggregator_index = task
    statistics = _BOOTSTRAP["statistics"]
    output = []
    for counts in counts_batch:
        try:
            summary = summarize_counts(statistics, counts.astype(np.float64), aggregator_index)
            output.append([
                summary["Delta_infinity"], summary["full_neff"], summary["support_maximum"],
                summary["isotonic"]["finite_target_gain"], summary["finite_parametric_gain"],
            ])
        except (ValueError, RuntimeError, FloatingPointError):
            output.append([None, None, None, None, None])
    return output


def percentile_report(values):
    array = np.asarray([value for value in values if value is not None and np.isfinite(value)], dtype=np.float64)
    failures = len(values) - len(array)
    if failures > 0.01 * len(values):
        return {"status": "NA_BOOTSTRAP_FIT_FAILURE", "failures": failures, "replicates": len(values)}
    return {
        "status": "PASS",
        "ci_99": [float(np.quantile(array, 0.005)), float(np.quantile(array, 0.995))],
        "failures": failures,
        "replicates": len(values),
    }


def bootstrap_panel(statistics, panel_index, processes):
    _BOOTSTRAP["statistics"] = statistics
    reports = {}
    for aggregator_index, aggregator in enumerate(AGGREGATOR_ORDER):
        seed = 20260814 + panel_index * 100 + aggregator_index
        counts = counts_for_bootstrap(statistics, 10000, seed)
        batches = [batch for batch in np.array_split(counts, processes) if len(batch)]
        context = mp.get_context("fork")
        with context.Pool(processes=min(processes, len(batches))) as pool:
            nested = pool.map(bootstrap_worker, [(batch, aggregator_index) for batch in batches])
        rows = [row for batch in nested for row in batch]
        metric_names = (
            "Delta_infinity", "full_neff", "support_maximum",
            "finite_isotonic_gain", "finite_parametric_gain",
        )
        reports[aggregator] = {
            "seed": seed,
            **{name: percentile_report([row[index] for row in rows]) for index, name in enumerate(metric_names)},
        }
    return reports


def group_statistics(metadata, errors, outputs):
    group_rows = defaultdict(list)
    for index, row in enumerate(metadata):
        group_rows[(int(row["fold"]), str(row["group"]))].append(index)
    groups = sorted(group_rows)
    sizes = np.zeros(len(groups), dtype=np.float64)
    failures = np.zeros((len(groups), 12), dtype=np.float64)
    joints = np.zeros((len(groups), 12, 12), dtype=np.float64)
    successes = np.zeros((len(groups), 2, len(MASKS)), dtype=np.float64)
    folds = np.zeros(len(groups), dtype=np.int64)
    for offset, group in enumerate(groups):
        indices = np.asarray(group_rows[group], dtype=np.int64)
        values = errors[indices].astype(np.float64)
        sizes[offset] = len(indices)
        failures[offset] = values.sum(axis=0)
        joints[offset] = values.T @ values
        successes[offset] = outputs[:, indices, :].sum(axis=1)
        folds[offset] = group[0]
    return {
        "group_sizes": sizes,
        "group_failures": failures,
        "group_joints": joints,
        "group_outputs": successes,
        "group_folds": folds,
        "groups": [list(group) for group in groups],
    }


def screen_worker(index):
    row_id = _WORKER["row_ids"][index]
    row = _WORKER["rows"][row_id]
    density = np.zeros(len(MASKS), dtype=np.uint8)
    majority = np.zeros(len(MASKS), dtype=np.uint8)
    selected_by_mask = _WORKER["majority_by_fold"][row["fold"]]
    for offset, indices in enumerate(SUBSET_INDICES):
        density[offset] = b3_correct([row["candidates"][value] for value in indices], row["target_bbox"])
        majority[offset] = row["candidates"][selected_by_mask[offset]]["correct"]
    return index, density, majority


def build_screen_panel(processes):
    rows = load_screen_rows()
    row_ids = tuple(sorted(rows))
    majority_by_fold = {}
    source_order = [candidate.source for candidate in next(iter(rows.values()))["gran_candidates"]]
    for fold in range(5):
        development = [row_id for row_id in row_ids if rows[row_id]["fold"] != fold]
        reliability = source_reliability(rows, development)
        priority = sorted(range(12), key=lambda index: (-reliability[source_order[index]], index))
        majority_by_fold[fold] = np.asarray([
            next(index for index in priority if index in indices) for indices in SUBSET_INDICES
        ], dtype=np.int16)
    _WORKER.update({"rows": rows, "row_ids": row_ids, "majority_by_fold": majority_by_fold})
    outputs = np.zeros((2, len(row_ids), len(MASKS)), dtype=np.uint8)
    with mp.get_context("fork").Pool(processes) as pool:
        for index, density, majority in pool.imap_unordered(screen_worker, range(len(row_ids)), chunksize=4):
            outputs[0, index] = density
            outputs[1, index] = majority
    errors = np.asarray([[not candidate["correct"] for candidate in rows[row_id]["candidates"]] for row_id in row_ids], dtype=np.uint8)
    metadata = [{"fold": rows[row_id]["fold"], "group": rows[row_id]["application"]} for row_id in row_ids]
    return group_statistics(metadata, errors, outputs)


def mind_worker(index):
    row_id = _WORKER["row_ids"][index]
    fold = _WORKER["folds"][row_id]
    source = _WORKER["slots"][row_id]
    priority = _WORKER["priorities"][fold]
    singleton = {row_id: None}
    density = np.zeros(len(MASKS), dtype=np.uint8)
    majority = np.zeros(len(MASKS), dtype=np.uint8)
    for offset, indices in enumerate(SUBSET_INDICES):
        subset = [source[value] for value in indices]
        singleton[row_id] = subset
        density[offset] = _WORKER["e1"].evaluate_mind_method(
            "ours", row_id, singleton, priority, {}, _WORKER["task_rows"],
            _WORKER["image_sizes"], _WORKER["model_order"],
        )
        majority[offset] = _WORKER["e1"].evaluate_mind_method(
            "majority", row_id, singleton, priority, {}, _WORKER["task_rows"],
            _WORKER["image_sizes"], _WORKER["model_order"],
        )
    return index, density, majority


def load_mind_base():
    e1 = load_module(CLOSE_DIR / "e1_arm_aggregator_matrix.py", "ceil_arm_a_e1")
    task_rows = [json.loads(line) for line in (XFER_DIR / "data/mind2web/mind2web_test_task.jsonl").read_text().splitlines() if line.strip()]
    rows_by_id = {str(row["id"]): row for row in task_rows}
    image_sizes = {str(row["id"]): Image.open(ROOT / row["image"]).size for row in task_rows}
    full = {model: e1.load_unique(e1.XFER / "raw/stage1" / directory) for model, directory in e1.MODEL_DIRS.items()}
    view1 = {model: e1.load_unique(e1.XFER / "raw/stage1/view1" / directory) for model, directory in e1.MODEL_DIRS.items()}
    stage2 = {model: e1.load_unique(e1.XFER / "raw/stage2" / directory) for model, directory in e1.MODEL_DIRS.items()}
    fold_map = json.loads((ROOT / "runs/complementarity/2026-07-30/folds.json").read_text())["pools"]["mind2web/visual"]["group_to_fold"]
    folds = {row_id: int(fold_map[row["website"]]) for row_id, row in rows_by_id.items()}
    return e1, rows_by_id, image_sizes, full, view1, stage2, folds


def build_mind_panel(arm, base, processes):
    e1, rows_by_id, image_sizes, full, view1, stage2, folds = base
    slots = e1.mind_slots(rows_by_id, full, view1, stage2, arm)
    row_ids = tuple(sorted(rows_by_id))
    priorities = {}
    for fold in range(5):
        development = [row_id for row_id in row_ids if folds[row_id] != fold]
        priorities[fold], _ = e1.dev_mind_statistics(development, slots, rows_by_id, image_sizes)
    _WORKER.clear()
    _WORKER.update({
        "e1": e1, "task_rows": rows_by_id, "image_sizes": image_sizes,
        "slots": slots, "folds": folds, "priorities": priorities,
        "row_ids": row_ids, "model_order": list(e1.MODEL_DIRS),
    })
    outputs = np.zeros((2, len(row_ids), len(MASKS)), dtype=np.uint8)
    with mp.get_context("fork").Pool(processes) as pool:
        for index, density, majority in pool.imap_unordered(mind_worker, range(len(row_ids)), chunksize=2):
            outputs[0, index] = density
            outputs[1, index] = majority
    errors = np.zeros((len(row_ids), 12), dtype=np.uint8)
    for row_index, row_id in enumerate(row_ids):
        for candidate_index, (_, _, prediction) in enumerate(slots[row_id]):
            errors[row_index, candidate_index] = not e1.score_prediction(rows_by_id[row_id], prediction, image_sizes[row_id])
    metadata = [{"fold": folds[row_id], "group": rows_by_id[row_id]["episode_id"]} for row_id in row_ids]
    return group_statistics(metadata, errors, outputs)


def serializable_statistics(statistics):
    return {key: value.tolist() if isinstance(value, np.ndarray) else value for key, value in statistics.items()}


def plot_panels(results):
    figure, axes = plt.subplots(5, 2, figsize=(11, 18), squeeze=False)
    for panel_index, panel in enumerate(PANEL_ORDER):
        for aggregator_index, aggregator in enumerate(AGGREGATOR_ORDER):
            report = results[panel]["point"][aggregator]
            axis = axes[panel_index, aggregator_index]
            x_values = report["isotonic"]["thresholds_x"]
            y_values = report["isotonic"]["thresholds_y"]
            axis.step(x_values, y_values, where="post", label="isotonic")
            if report["parametric"] is not None:
                grid = np.linspace(report["support_minimum"], report["target_x"], 300)
                parameters = report["parametric"]
                axis.plot(grid, parameters["a"] - parameters["b"] * np.exp(-parameters["c"] * grid), label="saturating")
            axis.axvline(report["full_neff"], color="black", linestyle="--", linewidth=1, label="full")
            axis.axvline(report["support_maximum"], color="red", linestyle=":", linewidth=1, label="support max")
            axis.set_title(f"{panel} / {aggregator}")
            axis.set_xlabel("generalized N_eff")
            axis.set_ylabel("accuracy")
            axis.legend(fontsize=7)
    figure.tight_layout()
    figure.savefig(FIGURE_PATH)
    plt.close(figure)


def main():
    if OUTPUT_PATH.exists() or FIGURE_PATH.exists() or CACHE_PATH.exists():
        raise FileExistsError("CEIL Arm A outputs already exist")
    config = yaml.safe_load(CONFIG_PATH.read_text())
    preflight = json.loads(PREFLIGHT_PATH.read_text())
    if config.get("status") != "FROZEN_BEFORE_ANY_CEIL_RESULT" or preflight.get("status") != "PASS_CEIL_INPUT_PREFLIGHT":
        raise PermissionError("CEIL Arm A boundary mismatch")
    processes = min(48, max(1, os.cpu_count() or 1))
    panel_statistics = {"screenspot_pro/C_uni": build_screen_panel(processes)}
    mind_base = load_mind_base()
    for arm in ("C_uni", "C_cond", "C_rand", "C_self"):
        panel_statistics[f"mind2web/{arm}"] = build_mind_panel(arm, mind_base, processes)
    cache_values = {}
    for panel, statistics in panel_statistics.items():
        prefix = panel.replace("/", "__")
        for key, value in statistics.items():
            if isinstance(value, np.ndarray):
                cache_values[f"{prefix}__{key}"] = value
    np.savez_compressed(CACHE_PATH, **cache_values)
    results = {}
    for panel_index, panel in enumerate(PANEL_ORDER):
        statistics = panel_statistics[panel]
        unit_counts = np.ones(len(statistics["group_folds"]), dtype=np.float64)
        point = {
            aggregator: summarize_counts(statistics, unit_counts, aggregator_index)
            for aggregator_index, aggregator in enumerate(AGGREGATOR_ORDER)
        }
        bootstrap = bootstrap_panel(statistics, panel_index, processes)
        results[panel] = {
            "groups": len(statistics["group_folds"]),
            "rows": int(statistics["group_sizes"].sum()),
            "point": point,
            "bootstrap": bootstrap,
        }
    if abs(results["screenspot_pro/C_uni"]["point"]["density"]["full_neff"] - 1.5936767669403409) > 1e-12:
        raise ValueError("CEIL Arm A MASK N_eff anchor mismatch")
    if abs(results["screenspot_pro/C_uni"]["point"]["density"]["support_maximum"] - 1.7073149168564605) > 1e-12:
        raise ValueError("CEIL Arm A MASK support anchor mismatch")
    plot_panels(results)
    output = {
        "schema_version": 1,
        "status": "PASS_CEIL_ARM_A_POST_HOC_COMPLETE",
        "evidence_status": "POST_HOC_DESCRIPTIVE",
        "panels": results,
        "cache": CACHE_PATH.relative_to(ROOT).as_posix(),
        "figure": FIGURE_PATH.relative_to(ROOT).as_posix(),
        "claim_boundary": {
            "universal_neff_law": False,
            "cross_benchmark_pooling": False,
            "changes_existing_statuses": False,
        },
    }
    OUTPUT_PATH.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": output["status"],
        "panels": {
            panel: {
                aggregator: {
                    "full_neff": report["full_neff"],
                    "support_maximum": report["support_maximum"],
                    "Delta_infinity": report["Delta_infinity"],
                }
                for aggregator, report in value["point"].items()
            }
            for panel, value in results.items()
        },
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()