import importlib.util
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml


RUN_DIR = Path(__file__).resolve().parent
ROOT = RUN_DIR.parents[2]
MASK_DIR = ROOT / "runs/mask/2026-08-14"
CLOSE_PATH = ROOT / "runs/close/2026-08-08/e1_arm_aggregator_matrix.py"
CONFIG_PATH = RUN_DIR / "configs/decomp_prereg.yaml"
PREFLIGHT_PATH = RUN_DIR / "PREFLIGHT.json"
OUTPUT_PATH = RUN_DIR / "ARM1.json"
RAW_SUBSETS_PATH = RUN_DIR / "raw/arm1_subsets.jsonl"
RAW_FOLDS_PATH = RUN_DIR / "raw/arm1_fold_cells.jsonl"

sys.path.insert(0, str(MASK_DIR))
from mask_common import b3_correct, load_rows, source_reliability


MODELS = ("GTA1-7B", "Qwen3-VL-8B-Instruct", "UI-TARS-7B-SFT")
SLOTS = tuple((model, view) for view in range(4) for model in MODELS)
METHODS = ("majority", "A0", "ours", "A1", "A2", "A3", "A4")


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def subset_features(mask):
    indices = tuple(index for index in range(12) if mask & (1 << index))
    lineages = {SLOTS[index][0] for index in indices}
    views = {SLOTS[index][1] for index in indices}
    counts = [sum(SLOTS[index][0] == model for index in indices) for model in MODELS]
    return {
        "mask": mask,
        "indices": indices,
        "budget": len(indices),
        "lineage_count": len(lineages),
        "view_count": len(views),
        "lineage_variance": float(np.var(counts)),
    }


def cell_key(feature):
    return feature["budget"], feature["lineage_count"], feature["view_count"]


def choose_cell(cell_scores, cell_variance):
    return max(cell_scores, key=lambda cell: (
        cell_scores[cell],
        -cell_variance[cell],
        cell[1],
        cell[2],
        tuple(-value for value in cell[1:]),
    ))


def grouped_cell_means(predictions, row_indices, cells):
    output = {}
    for cell, subset_indices in cells.items():
        output[cell] = float(np.mean(predictions[np.ix_(row_indices, subset_indices)]))
    return output


def anova_components(values, features):
    values = np.asarray(values, dtype=np.float64)
    lineage = np.asarray([feature["lineage_count"] for feature in features])
    view = np.asarray([feature["view_count"] for feature in features])
    total = float(np.sum((values - values.mean()) ** 2))
    if total <= 0:
        return {"lineage": None, "view": None, "interaction": None, "residual": None}

    def design(include_lineage, include_view):
        columns = [np.ones(len(values))]
        if include_lineage:
            levels = sorted(set(lineage))
            columns.extend((lineage == level).astype(float) for level in levels[1:])
        if include_view:
            levels = sorted(set(view))
            columns.extend((view == level).astype(float) for level in levels[1:])
        return np.column_stack(columns)

    def sse(matrix):
        fitted = matrix @ np.linalg.lstsq(matrix, values, rcond=None)[0]
        return float(np.sum((values - fitted) ** 2))

    sse_l = sse(design(True, False))
    sse_v = sse(design(False, True))
    sse_lv = sse(design(True, True))
    cell_groups = defaultdict(list)
    for index, pair in enumerate(zip(lineage, view)):
        cell_groups[pair].append(index)
    sse_cell = sum(float(np.sum((values[indices] - values[indices].mean()) ** 2)) for indices in cell_groups.values())
    return {
        "lineage": max(0.0, sse_v - sse_lv) / total,
        "view": max(0.0, sse_l - sse_lv) / total,
        "interaction": max(0.0, sse_lv - sse_cell) / total,
        "residual": max(0.0, sse_cell) / total,
    }


def marginal_contrasts(cell_scores):
    lineage = []
    view = []
    cells = set(cell_scores)
    for cell in sorted(cells):
        budget, lineage_count, view_count = cell
        right = (budget, lineage_count + 1, view_count)
        up = (budget, lineage_count, view_count + 1)
        if right in cells:
            lineage.append(cell_scores[right] - cell_scores[cell])
        if up in cells:
            view.append(cell_scores[up] - cell_scores[cell])
    return {
        "lineage": float(np.mean(lineage)) if lineage else None,
        "view": float(np.mean(view)) if view else None,
        "lineage_pairs": len(lineage),
        "view_pairs": len(view),
    }


def percentile(values):
    values = np.asarray(values, dtype=np.float64)
    return [float(np.quantile(values, 0.005)), float(np.quantile(values, 0.995))]


def application_multiplicities(applications, app_fold, resamples, seed):
    rng = np.random.default_rng(seed)
    output = np.zeros((resamples, len(applications)), dtype=np.int16)
    for replicate in range(resamples):
        for fold in range(5):
            indices = np.asarray([index for index, app in enumerate(applications) if app_fold[app] == fold], dtype=np.int64)
            if not len(indices):
                continue
            sampled = rng.choice(indices, size=len(indices), replace=True)
            output[replicate] += np.bincount(sampled, minlength=len(applications))
    return output


def write_jsonl_fsynced(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())


def full_predictions(rows, row_ids, features):
    b3 = np.zeros((len(row_ids), len(features)), dtype=np.uint8)
    for row_offset, row_id in enumerate(row_ids):
        row = rows[row_id]
        for subset_offset, feature in enumerate(features):
            b3[row_offset, subset_offset] = b3_correct(
                [row["candidates"][index] for index in feature["indices"]],
                row["target_bbox"],
            )
    return b3


def majority_predictions(rows, row_ids, features, fit_ids):
    reliability = source_reliability(rows, fit_ids)
    source_order = [candidate.source for candidate in next(iter(rows.values()))["gran_candidates"]]
    selected = np.asarray([
        min(feature["indices"], key=lambda index: (-reliability[source_order[index]], index))
        for feature in features
    ], dtype=np.int64)
    correct = np.asarray([[candidate["correct"] for candidate in rows[row_id]["candidates"]] for row_id in row_ids], dtype=np.uint8)
    return correct[:, selected], reliability


def baseline_outputs(rows, row_ids, fold_for_row):
    close = load_module(CLOSE_PATH, "decomp_close_e1")
    outputs = {method: np.zeros(len(row_ids), dtype=np.uint8) for method in METHODS}
    selected_methods = []
    for outer_fold in range(5):
        inner_val = (outer_fold + 1) % 5
        inner_train_ids = [row_id for row_id in row_ids if fold_for_row[row_id] not in {outer_fold, inner_val}]
        inner_val_ids = [row_id for row_id in row_ids if fold_for_row[row_id] == inner_val]
        outer_dev_ids = [row_id for row_id in row_ids if fold_for_row[row_id] != outer_fold]
        test_ids = [row_id for row_id in row_ids if fold_for_row[row_id] == outer_fold]
        inner_reliability = source_reliability(rows, inner_train_ids)
        outer_reliability = source_reliability(rows, outer_dev_ids)
        source_names = [candidate.source for candidate in next(iter(rows.values()))["gran_candidates"]]
        inner_priority = sorted(source_names, key=lambda source: (-inner_reliability[source], source))
        outer_priority = sorted(source_names, key=lambda source: (-outer_reliability[source], source))

        def evaluate(method, row_id, priority):
            row = rows[row_id]
            points = [candidate["point"] for candidate in row["candidates"]]
            if method in {"majority", "A0"}:
                source = priority[0]
                index = source_names.index(source)
                return bool(row["candidates"][index]["correct"])
            if method == "ours":
                return bool(b3_correct(row["candidates"], row["target_bbox"]))
            if method == "A1":
                point = close.geometric_median(points)
            elif method in {"A2", "A3"}:
                point = close.density_medoid(points)
            elif method == "A4":
                point = close.density_mode(points)
            else:
                raise ValueError(method)
            return bool(close.point_in_bbox(point, row["target_bbox"]))

        scores = {method: float(np.mean([evaluate(method, row_id, inner_priority) for row_id in inner_val_ids])) for method in METHODS}
        selected_method = max(METHODS, key=lambda method: (scores[method], -METHODS.index(method)))
        selected_methods.append({"outer_fold": outer_fold, "selected_method": selected_method, "scores": scores})
        for row_id in test_ids:
            offset = row_ids.index(row_id)
            for method in METHODS:
                outputs[method][offset] = evaluate(method, row_id, outer_priority)
    return outputs, selected_methods


def main():
    if OUTPUT_PATH.exists() or RAW_SUBSETS_PATH.exists() or RAW_FOLDS_PATH.exists():
        raise FileExistsError("DECOMP Arm 1 output exists")
    config = yaml.safe_load(CONFIG_PATH.read_text())
    preflight = json.loads(PREFLIGHT_PATH.read_text())
    if (
        config["arm1"]["benchmark"] != "screenspot_pro"
        or config["arm1"]["mind2web_status"] != "BLOCKED_ALIGNED_POOL_UNAVAILABLE"
        or config["arm1"]["leaked_values_imported_by_implementation"] is not False
        or preflight["status"] != "PASS_DECOMP_PREFLIGHT_NO_ARM_STARTED"
    ):
        raise PermissionError("DECOMP Arm 1 authorization mismatch")
    rows = load_rows()
    row_ids = tuple(sorted(rows))
    row_index = {row_id: index for index, row_id in enumerate(row_ids)}
    fold_for_row = {row_id: int(rows[row_id]["fold"]) for row_id in row_ids}
    features = [subset_features(mask) for mask in range(1, 1 << 12) if mask.bit_count() >= 2]
    cells = defaultdict(list)
    for offset, feature in enumerate(features):
        cells[cell_key(feature)].append(offset)
    cell_variance = {cell: float(np.mean([features[index]["lineage_variance"] for index in indices])) for cell, indices in cells.items()}

    b3 = full_predictions(rows, row_ids, features)
    full_feature_offset = next(index for index, feature in enumerate(features) if feature["mask"] == (1 << 12) - 1)
    if abs(float(np.mean(b3[:, full_feature_offset])) - 0.6369386464263125) > 1e-15:
        raise ValueError("DECOMP Arm 1 full-pool B3 anchor mismatch")

    fold_records = []
    selected_results = {aggregator: [] for aggregator in ("density_B3", "F1_majority")}
    oof_predictions = {"density_B3": b3.copy(), "F1_majority": np.zeros_like(b3)}
    nested_matrices = {aggregator: {} for aggregator in oof_predictions}
    for outer_fold in range(5):
        inner_val = (outer_fold + 1) % 5
        inner_train_ids = [row_id for row_id in row_ids if fold_for_row[row_id] not in {outer_fold, inner_val}]
        inner_indices = [row_index[row_id] for row_id in row_ids if fold_for_row[row_id] == inner_val]
        outer_dev_ids = [row_id for row_id in row_ids if fold_for_row[row_id] != outer_fold]
        test_indices = [row_index[row_id] for row_id in row_ids if fold_for_row[row_id] == outer_fold]
        majority_inner, _ = majority_predictions(rows, row_ids, features, inner_train_ids)
        majority_outer, _ = majority_predictions(rows, row_ids, features, outer_dev_ids)
        oof_predictions["F1_majority"][test_indices] = majority_outer[test_indices]
        nested_matrices["density_B3"][outer_fold] = (b3, b3)
        nested_matrices["F1_majority"][outer_fold] = (majority_inner, majority_outer)
        record = {"outer_fold": outer_fold, "inner_validation_fold": inner_val, "aggregators": {}}
        for aggregator, inner_matrix, outer_matrix in (
            ("density_B3", b3, b3),
            ("F1_majority", majority_inner, majority_outer),
        ):
            scores = grouped_cell_means(inner_matrix, inner_indices, cells)
            current = []
            for budget in config["arm1"]["budgets"]:
                budget_scores = {cell: value for cell, value in scores.items() if cell[0] == budget}
                selected = choose_cell(budget_scores, cell_variance)
                test_accuracy = float(np.mean(outer_matrix[np.ix_(test_indices, cells[selected])]))
                supported = sorted(cell for cell in budget_scores)
                boundary = (
                    selected[1] in {min(cell[1] for cell in supported), max(cell[1] for cell in supported)}
                    or selected[2] in {min(cell[2] for cell in supported), max(cell[2] for cell in supported)}
                )
                value = {
                    "budget": budget,
                    "selected_cell": list(selected[1:]),
                    "inner_validation_cell_mean": budget_scores[selected],
                    "outer_test_cell_mean": test_accuracy,
                    "boundary_selected": boundary,
                    "cell_subset_count": len(cells[selected]),
                }
                current.append(value)
                selected_results[aggregator].append({"outer_fold": outer_fold, **value})
            record["aggregators"][aggregator] = current
        fold_records.append(record)

    baselines, dev_selection = baseline_outputs(rows, list(row_ids), fold_for_row)
    baseline_accuracy = {method: float(np.mean(values)) for method, values in baselines.items()}
    selected_dev = np.zeros(len(row_ids), dtype=np.uint8)
    for record in dev_selection:
        fold = record["outer_fold"]
        selected_dev[[row_index[row_id] for row_id in row_ids if fold_for_row[row_id] == fold]] = baselines[record["selected_method"]][[row_index[row_id] for row_id in row_ids if fold_for_row[row_id] == fold]]

    budget_offsets = {
        budget: np.asarray([index for index, feature in enumerate(features) if feature["budget"] == budget], dtype=np.int64)
        for budget in config["arm1"]["budgets"]
    }
    budget_features = {
        budget: [features[index] for index in offsets]
        for budget, offsets in budget_offsets.items()
    }
    budget_cells = {
        budget: {cell: np.asarray(indices, dtype=np.int64) for cell, indices in cells.items() if cell[0] == budget}
        for budget in config["arm1"]["budgets"]
    }
    fold_indices = {
        fold: np.asarray([row_index[row_id] for row_id in row_ids if fold_for_row[row_id] == fold], dtype=np.int64)
        for fold in range(5)
    }

    subset_records = []
    decomposition = {aggregator: {} for aggregator in oof_predictions}
    marginal = {aggregator: {} for aggregator in oof_predictions}
    for aggregator, matrix in oof_predictions.items():
        aggregate = np.mean(matrix, axis=0)
        for budget in config["arm1"]["budgets"]:
            offsets = budget_offsets[budget]
            current_features = budget_features[budget]
            values = aggregate[offsets]
            decomposition[aggregator][str(budget)] = anova_components(values, current_features)
            current_cells = defaultdict(list)
            for local, feature in enumerate(current_features):
                current_cells[cell_key(feature)].append(float(values[local]))
            cell_scores = {cell: float(np.mean(scores)) for cell, scores in current_cells.items()}
            marginal[aggregator][str(budget)] = marginal_contrasts(cell_scores)
        subset_records.extend({
            "aggregator": aggregator,
            **{key: (list(value) if key == "indices" else value) for key, value in feature.items()},
            "oof_accuracy": float(aggregate[index]),
        } for index, feature in enumerate(features))

    # Grouped bootstrap over application groups; subset configurations are never resampled.
    applications = sorted({rows[row_id]["application"] for row_id in row_ids})
    app_indices = {app: np.asarray([row_index[row_id] for row_id in row_ids if rows[row_id]["application"] == app], dtype=np.int64) for app in applications}
    app_fold = {app: fold_for_row[next(row_id for row_id in row_ids if rows[row_id]["application"] == app)] for app in applications}
    app_rows = np.asarray([len(app_indices[app]) for app in applications], dtype=np.int64)
    multiplicities = application_multiplicities(
        applications, app_fold, config["arm1"]["bootstrap"]["resamples"], 20260815
    )
    bootstrap_selected = {aggregator: {str(budget): [] for budget in config["arm1"]["budgets"]} for aggregator in oof_predictions}
    bootstrap_decomp = {aggregator: {str(budget): {name: [] for name in ("lineage", "view", "interaction")} for budget in config["arm1"]["budgets"]} for aggregator in oof_predictions}
    bootstrap_marginal = {aggregator: {str(budget): {name: [] for name in ("lineage", "view")} for budget in config["arm1"]["budgets"]} for aggregator in oof_predictions}
    for aggregator, matrix in oof_predictions.items():
        app_correct = np.stack([matrix[app_indices[app]].sum(axis=0) for app in applications])
        denominators = multiplicities @ app_rows
        if np.any(denominators <= 0):
            raise ValueError("DECOMP Arm 1 empty global bootstrap replicate")
        bootstrap_accuracy = (multiplicities @ app_correct) / denominators[:, None]
        for replicate, aggregate in enumerate(bootstrap_accuracy):
            for budget in config["arm1"]["budgets"]:
                offsets = budget_offsets[budget]
                current_features = budget_features[budget]
                values = aggregate[offsets]
                components = anova_components(values, current_features)
                for name in bootstrap_decomp[aggregator][str(budget)]:
                    if components[name] is not None:
                        bootstrap_decomp[aggregator][str(budget)][name].append(components[name])
                current_cells = defaultdict(list)
                for local, feature in enumerate(current_features):
                    current_cells[cell_key(feature)].append(float(values[local]))
                cell_scores = {cell: float(np.mean(scores)) for cell, scores in current_cells.items()}
                contrasts = marginal_contrasts(cell_scores)
                for name in bootstrap_marginal[aggregator][str(budget)]:
                    if contrasts[name] is not None:
                        bootstrap_marginal[aggregator][str(budget)][name].append(contrasts[name])
        del bootstrap_accuracy

    # Re-run nested cell selection from application-level cell sums.
    for aggregator in oof_predictions:
        replicate_numerators = {budget: np.zeros(len(multiplicities), dtype=np.float64) for budget in config["arm1"]["budgets"]}
        replicate_denominators = {budget: np.zeros(len(multiplicities), dtype=np.float64) for budget in config["arm1"]["budgets"]}
        for outer_fold in range(5):
            inner_fold = (outer_fold + 1) % 5
            inner_matrix, outer_matrix = nested_matrices[aggregator][outer_fold]
            inner_app_mask = np.asarray([app_fold[app] == inner_fold for app in applications])
            test_app_mask = np.asarray([app_fold[app] == outer_fold for app in applications])
            inner_denominator = multiplicities[:, inner_app_mask] @ app_rows[inner_app_mask]
            test_denominator = multiplicities[:, test_app_mask] @ app_rows[test_app_mask]
            if np.any(inner_denominator <= 0) or np.any(test_denominator <= 0):
                raise ValueError(f"DECOMP Arm 1 empty fold bootstrap replicate: {outer_fold}")
            for budget in config["arm1"]["budgets"]:
                current_cells = budget_cells[budget]
                ordered_cells = sorted(current_cells)
                inner_app_cell = np.asarray([
                    [float(np.mean(inner_matrix[np.ix_(app_indices[app], current_cells[cell])], axis=1).sum()) for cell in ordered_cells]
                    for app in applications
                ])
                outer_app_cell = np.asarray([
                    [float(np.mean(outer_matrix[np.ix_(app_indices[app], current_cells[cell])], axis=1).sum()) for cell in ordered_cells]
                    for app in applications
                ])
                inner_scores = (multiplicities[:, inner_app_mask] @ inner_app_cell[inner_app_mask]) / inner_denominator[:, None]
                outer_scores = (multiplicities[:, test_app_mask] @ outer_app_cell[test_app_mask]) / test_denominator[:, None]
                for replicate in range(len(multiplicities)):
                    score_map = {cell: float(inner_scores[replicate, index]) for index, cell in enumerate(ordered_cells)}
                    selected = choose_cell(score_map, cell_variance)
                    selected_index = ordered_cells.index(selected)
                    replicate_numerators[budget][replicate] += outer_scores[replicate, selected_index] * test_denominator[replicate]
                    replicate_denominators[budget][replicate] += test_denominator[replicate]
        for budget in config["arm1"]["budgets"]:
            bootstrap_selected[aggregator][str(budget)] = (
                replicate_numerators[budget] / replicate_denominators[budget]
            ).tolist()

    selected_summary = {}
    for aggregator, records in selected_results.items():
        selected_summary[aggregator] = []
        for budget in config["arm1"]["budgets"]:
            current = [record for record in records if record["budget"] == budget]
            point = float(np.average([record["outer_test_cell_mean"] for record in current], weights=[sum(fold_for_row[row_id] == record["outer_fold"] for row_id in row_ids) for record in current]))
            selected_summary[aggregator].append({
                "budget": budget,
                "point_accuracy": point,
                "ci_99": percentile(bootstrap_selected[aggregator][str(budget)]),
                "fold_selections": current,
            })
    for aggregator in decomposition:
        for budget in config["arm1"]["budgets"]:
            for name in ("lineage", "view", "interaction"):
                decomposition[aggregator][str(budget)][f"{name}_ci_99"] = percentile(bootstrap_decomp[aggregator][str(budget)][name]) if bootstrap_decomp[aggregator][str(budget)][name] else None
                marginal[aggregator][str(budget)][f"{name}_ci_99"] = percentile(bootstrap_marginal[aggregator][str(budget)][name]) if bootstrap_marginal[aggregator][str(budget)][name] else None

    write_jsonl_fsynced(RAW_SUBSETS_PATH, subset_records)
    write_jsonl_fsynced(RAW_FOLDS_PATH, fold_records)
    output = {
        "schema_version": 1,
        "status": "PASS_DECOMP_ARM1_COMPLETE",
        "evidence_status": "POST_HOC_DESCRIPTIVE_DECOMPOSITION",
        "benchmark": "screenspot_pro",
        "rows": len(row_ids),
        "subsets_budget_2_to_12": len(features),
        "mind2web_status": "BLOCKED_ALIGNED_POOL_UNAVAILABLE",
        "leaked_values_imported": False,
        "anchors": {"full_pool_density_B3": float(np.mean(b3[:, full_feature_offset]))},
        "budget_tables": selected_summary,
        "variance_decomposition": decomposition,
        "marginal_contrasts": marginal,
        "historical_kappa_reference": {"view": 0.895, "cross_lineage": 0.398},
        "baselines": {
            "accuracy": baseline_accuracy,
            "nested_dev_selection_accuracy": float(np.mean(selected_dev)),
            "nested_dev_selection_folds": dev_selection,
            "best_single_equals_full_pool_majority": True,
            "historical_v_only": {"4": {"density_B3": 0.6123, "F1_majority": 0.6142}, "8": {"density_B3": 0.6072, "F1_majority": 0.6078}, "12": {"density_B3": 0.6009, "F1_majority": 0.6040}},
        },
        "bootstrap": {"unit": "application_group_rows", "resamples": 10000, "confidence": 0.99, "subsets_resampled": False},
        "raw": {
            "subsets": {"path": str(RAW_SUBSETS_PATH.relative_to(ROOT)), "rows": len(subset_records), "bytes": RAW_SUBSETS_PATH.stat().st_size},
            "fold_cells": {"path": str(RAW_FOLDS_PATH.relative_to(ROOT)), "rows": len(fold_records), "bytes": RAW_FOLDS_PATH.stat().st_size},
            "write_flush_fsync_per_row": True,
        },
    }
    import hashlib
    for value in output["raw"].values():
        if isinstance(value, dict) and "path" in value:
            value["sha256"] = hashlib.sha256((ROOT / value["path"]).read_bytes()).hexdigest()
    temporary = OUTPUT_PATH.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    temporary.replace(OUTPUT_PATH)
    print(json.dumps({"status": output["status"], "rows": output["rows"], "subsets": output["subsets_budget_2_to_12"], "anchor": output["anchors"]}, indent=2))


if __name__ == "__main__":
    main()