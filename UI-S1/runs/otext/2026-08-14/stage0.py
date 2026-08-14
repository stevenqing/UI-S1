import importlib.util
import json
import math
import multiprocessing as mp
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml


RUN_DIR = Path(__file__).resolve().parent
ROOT = RUN_DIR.parents[2]
CONFIG_PATH = RUN_DIR / "configs/otext_prereg.yaml"
PREFLIGHT_PATH = RUN_DIR / "PREFLIGHT.json"
OUTPUT_PATH = RUN_DIR / "STAGE0.json"
SELECTED_PATH = RUN_DIR / "SELECTED_PARAMETERS.json"
sys.path.insert(0, str(RUN_DIR))
sys.path.insert(0, str(ROOT / "runs/mask/2026-08-14"))

from mask_common import load_rows, source_reliability
from otext_common import best_ocr_box, load_raw, weighted_b3


EXTRACTORS = ("quoted", "caps_camel", "full_normalized")
MATCHERS = ("exact", "normalized", "edit")
METHODS = ("majority", "A0", "ours", "A1", "A2", "A3", "A4")
WEIGHTS = (0.0, 0.125, 0.25, 0.5, 1.0, 1.5936767669403409, 2.0, 4.0)
ENGINES = ("easyocr", "rapidocr")
VISUAL_WEIGHT = 1.5936767669403409 / 12
_WORKER = {}


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def point_correct(point, bbox):
    return bool(point is not None and bbox[0] <= point[0] <= bbox[2] and bbox[1] <= point[1] <= bbox[3])


def ocr_worker(row_id):
    instruction = _WORKER["metadata"][row_id]["instruction"]
    output = {}
    for engine in ENGINES:
        boxes = _WORKER["raw"][engine][row_id]["boxes"]
        output[engine] = {
            f"{extractor}/{matcher}": best_ocr_box(boxes, instruction, extractor, matcher)
            for extractor in EXTRACTORS for matcher in MATCHERS
        }
    return row_id, output


def theta_grid(scores):
    values = np.asarray([value for value in scores if value > 0], dtype=np.float64)
    if not len(values):
        return [0.0] * 11 + [float("inf")]
    return [float(value) for value in np.quantile(values, np.linspace(0, 1, 11), method="linear")] + [float("inf")]


def output_with_ocr(row_ids, baseline, ocr_values, theta, rows):
    output = {}
    for row_id in row_ids:
        value = ocr_values[row_id]
        gated = value["point"] is not None and value["score"] >= theta
        output[row_id] = point_correct(value["point"], rows[row_id]["target_bbox"]) if gated else baseline[row_id]
    return output


def weighted_density_outputs(row_ids, ocr_values, theta, weight, rows):
    output = {}
    for row_id in row_ids:
        row = rows[row_id]; value = ocr_values[row_id]
        points = [candidate["point"] for candidate in row["candidates"]]
        weights = [VISUAL_WEIGHT] * 12
        if value["point"] is not None and value["score"] >= theta:
            points.append(tuple(value["point"])); weights.append(float(weight))
        selected, _ = weighted_b3(points, weights)
        output[row_id] = point_correct(points[selected], row["target_bbox"])
    return output


def gains(output, baselines, row_ids):
    return {
        name: float(np.mean([int(output[row_id]) - int(values[row_id]) for row_id in row_ids]))
        for name, values in baselines.items()
    }


def conditional_report(row_ids, baseline, ocr_values, rows):
    values = {True: [], False: []}
    for row_id in row_ids:
        correct = point_correct(ocr_values[row_id]["point"], rows[row_id]["target_bbox"])
        values[bool(baseline[row_id])].append(correct)
    return {
        "OCR_correct_given_baseline_correct": float(np.mean(values[True])) if values[True] else None,
        "OCR_correct_given_baseline_wrong": float(np.mean(values[False])) if values[False] else None,
        "baseline_correct_rows": len(values[True]),
        "baseline_wrong_rows": len(values[False]),
    }


def decile_table(row_ids, baseline, ocr_values, rows):
    ordered = sorted(row_ids, key=lambda row_id: (ocr_values[row_id]["score"], row_id))
    output = []
    for index, selected in enumerate(np.array_split(np.asarray(ordered, dtype=object), 10)):
        table = {"pool_correct_OCR_correct": 0, "pool_correct_OCR_wrong": 0, "pool_wrong_OCR_correct": 0, "pool_wrong_OCR_wrong": 0}
        scores = []
        for row_id in selected.tolist():
            pool = bool(baseline[row_id]); ocr = point_correct(ocr_values[row_id]["point"], rows[row_id]["target_bbox"])
            table[f"pool_{'correct' if pool else 'wrong'}_OCR_{'correct' if ocr else 'wrong'}"] += 1
            scores.append(ocr_values[row_id]["score"])
        output.append({"decile": index, "rows": len(selected), "score_range": [min(scores), max(scores)] if scores else None, **table})
    return output


def main():
    if OUTPUT_PATH.exists() or SELECTED_PATH.exists():
        raise FileExistsError("OTEXT Stage0 outputs exist")
    config = yaml.safe_load(CONFIG_PATH.read_text())
    preflight = json.loads(PREFLIGHT_PATH.read_text())
    if config.get("status") != "FROZEN_BEFORE_ANY_OTEXT_OCR_OR_LABEL_STATISTIC" or preflight.get("stage0_label_statistics_computed") is not False:
        raise PermissionError("OTEXT Stage0 boundary mismatch")
    rows = load_rows(); row_ids = tuple(sorted(rows))
    metadata = {}
    for path in sorted((ROOT / "runs/ccm-h2h/2026-07-31/h1/shards/top18").glob("shard-*.jsonl")):
        for line in path.read_text().splitlines():
            if line.strip():
                value = json.loads(line); metadata[value["id"]] = value
    raw = {engine: load_raw(engine) for engine in ENGINES}
    _WORKER.update({"metadata": metadata, "raw": raw})
    ocr = {}
    with mp.get_context("fork").Pool(min(48, os.cpu_count() or 1)) as pool:
        for row_id, value in pool.imap_unordered(ocr_worker, row_ids, chunksize=2):
            ocr[row_id] = value
    e1 = load_module(ROOT / "runs/close/2026-08-08/e1_arm_aggregator_matrix.py", "otext_e1")
    slots = {row_id: [(candidate["source"], dict(candidate)) for candidate in rows[row_id]["candidates"]] for row_id in row_ids}
    targets = {row_id: rows[row_id]["target_bbox"] for row_id in row_ids}
    reports = {engine: [] for engine in ENGINES}
    selected_manifest = {"schema_version": 1, "status": None, "engines": {}}
    for engine in ENGINES:
        engine_selected = []
        for outer_fold in range(5):
            inner_validation_fold = (outer_fold + 1) % 5
            inner_train = [row_id for row_id in row_ids if rows[row_id]["fold"] not in {outer_fold, inner_validation_fold}]
            inner_validation = [row_id for row_id in row_ids if rows[row_id]["fold"] == inner_validation_fold]
            outer_development = [row_id for row_id in row_ids if rows[row_id]["fold"] != outer_fold]
            outer_test = [row_id for row_id in row_ids if rows[row_id]["fold"] == outer_fold]
            inner_priority, _ = e1.screen_dev_priority(inner_train, slots, targets)
            validation_method_scores = {
                method: float(np.mean([e1.evaluate_screen_method(method, row_id, slots, inner_priority, targets[row_id]) for row_id in inner_validation]))
                for method in METHODS
            }
            selected_method = max(METHODS, key=lambda method: (validation_method_scores[method], -METHODS.index(method)))
            majority = {row_id: bool(e1.evaluate_screen_method("majority", row_id, slots, inner_priority, targets[row_id])) for row_id in inner_validation}
            dev_selection = {row_id: bool(e1.evaluate_screen_method(selected_method, row_id, slots, inner_priority, targets[row_id])) for row_id in inner_validation}
            baselines = {"majority": majority, "dev_selection": dev_selection}
            candidates = []
            for extractor in EXTRACTORS:
                for matcher in MATCHERS:
                    key = f"{extractor}/{matcher}"
                    values = {row_id: ocr[row_id][engine][key] for row_id in row_ids}
                    grid = theta_grid([values[row_id]["score"] for row_id in inner_train])
                    for theta_index, theta in enumerate(grid):
                        output = output_with_ocr(inner_validation, majority, values, theta, rows)
                        gain = gains(output, baselines, inner_validation)
                        candidates.append({
                            "extractor": extractor, "matcher": matcher,
                            "theta_index": theta_index, "theta": theta,
                            "gains": gain, "objective": min(gain.values()),
                        })
            selected = max(candidates, key=lambda value: (
                value["objective"], value["theta_index"],
                -EXTRACTORS.index(value["extractor"]), -MATCHERS.index(value["matcher"]),
            ))
            key = f"{selected['extractor']}/{selected['matcher']}"
            values = {row_id: ocr[row_id][engine][key] for row_id in row_ids}
            weight_scores = []
            for weight in WEIGHTS:
                output = weighted_density_outputs(inner_validation, values, selected["theta"], weight, rows)
                gain = gains(output, baselines, inner_validation)
                weight_scores.append({"weight": weight, "gains": gain, "objective": min(gain.values())})
            selected_weight = max(weight_scores, key=lambda value: (value["objective"], -WEIGHTS.index(value["weight"])))
            selected_output = output_with_ocr(inner_validation, majority, values, selected["theta"], rows)
            selected_gains = gains(selected_output, baselines, inner_validation)
            curve = [value for value in candidates if value["extractor"] == selected["extractor"] and value["matcher"] == selected["matcher"]]
            fold_report = {
                "outer_fold": outer_fold, "inner_validation_fold": inner_validation_fold,
                "inner_train_rows": len(inner_train), "inner_validation_rows": len(inner_validation),
                "outer_development_rows": len(outer_development), "outer_test_rows": len(outer_test),
                "selected_method": selected_method, "dev_selection_scores": validation_method_scores,
                "selected": selected, "selected_weight": selected_weight,
                "all_gate_validation_scores": candidates,
                "selected_matcher_net_curve": curve,
                "weight_validation_scores": weight_scores,
                "conditional_majority": conditional_report(inner_validation, majority, values, rows),
                "conditional_dev_selection": conditional_report(inner_validation, dev_selection, values, rows),
                "score_deciles_majority": decile_table(inner_validation, majority, values, rows),
                "score_deciles_dev_selection": decile_table(inner_validation, dev_selection, values, rows),
                "selected_validation_gains": selected_gains,
                "theta_boundary": selected["theta_index"] in {0, 11},
            }
            reports[engine].append(fold_report)
            engine_selected.append({
                "outer_fold": outer_fold, "extractor": selected["extractor"],
                "matcher": selected["matcher"], "theta_index": selected["theta_index"],
                "theta": selected["theta"], "weight": selected_weight["weight"],
                "selected_method": selected_method,
            })
        selected_manifest["engines"][engine] = engine_selected
    weighted = {}
    for engine in ENGINES:
        numerator = {"majority": 0.0, "dev_selection": 0.0}; denominator = 0
        for fold in reports[engine]:
            weight = fold["outer_test_rows"]; denominator += weight
            for name in numerator:
                numerator[name] += weight * fold["selected_validation_gains"][name]
        gains_value = {name: value / denominator for name, value in numerator.items()}
        weighted[engine] = {"gains": gains_value, "objective": min(gains_value.values()), "pass_O_G1": min(gains_value.values()) >= 0.007}
    primary_pass = weighted["easyocr"]["pass_O_G1"]
    selected_manifest["status"] = "PASS_OTEXT_STAGE0_SELECTED_PARAMETERS" if primary_pass else "STOP_OTEXT_O_K1"
    selected_manifest["O_G1"] = weighted
    result = {
        "schema_version": 1, "status": "PASS_OTEXT_STAGE0_COMPLETE",
        "evidence_status": "POST_SELECTION_VALIDATION",
        "reports": reports, "O_G1": weighted,
        "primary_engine": "easyocr", "proceed_stage1": primary_pass,
        "kill_conditions": {
            "O_K1": not primary_pass,
            "O_K7": any(fold["theta_boundary"] for fold in reports["easyocr"]),
        },
    }
    OUTPUT_PATH.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    SELECTED_PATH.write_text(json.dumps(selected_manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": result["status"], "O_G1": weighted, "proceed_stage1": primary_pass, "O_K7": result["kill_conditions"]["O_K7"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()