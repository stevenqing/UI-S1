import hashlib
import importlib.util
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


RUN_DIR = Path(__file__).resolve().parent
ROOT = RUN_DIR.parents[2]
STAGE0_PATH = RUN_DIR / "STAGE0.json"
SELECTED_PATH = RUN_DIR / "SELECTED_PARAMETERS.json"
OUTPUT_PATH = RUN_DIR / "STAGE1.json"
ROWS_PATH = RUN_DIR / "STAGE1_ROWS.jsonl"
sys.path.insert(0, str(RUN_DIR))
sys.path.insert(0, str(ROOT / "runs/mask/2026-08-14"))

from mask_common import load_rows
from otext_common import best_ocr_box, load_raw, weighted_b3


ENGINES = ("easyocr", "rapidocr")
VISUAL_WEIGHT = 1.5936767669403409 / 12


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def point_correct(point, bbox):
    return bool(point is not None and bbox[0] <= point[0] <= bbox[2] and bbox[1] <= point[1] <= bbox[3])


def hash_int(value):
    return int.from_bytes(hashlib.sha256(value.encode()).digest()[:8], "big")


def grouped_bootstrap(rows, left_key, right_key, seed, resamples=10000):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["fold"], row["application"])].append(row)
    by_fold = defaultdict(list)
    for group in groups:
        by_fold[group[0]].append(group)
    generator = np.random.default_rng(seed); samples = np.empty(resamples)
    for replicate in range(resamples):
        selected = []
        for fold in sorted(by_fold):
            options = sorted(by_fold[fold])
            for group in generator.choice(len(options), size=len(options), replace=True):
                selected.extend(groups[options[int(group)]])
        samples[replicate] = np.mean([int(row[left_key]) - int(row[right_key]) for row in selected])
    point = float(np.mean([int(row[left_key]) - int(row[right_key]) for row in rows]))
    return {"point": point, "ci_99": [float(np.quantile(samples, 0.005)), float(np.quantile(samples, 0.995))], "seed": seed, "resamples": resamples}


def complete_output(points, weights, bbox):
    selected, _ = weighted_b3(points, weights)
    return point_correct(points[selected], bbox)


def main():
    if OUTPUT_PATH.exists() or ROWS_PATH.exists():
        raise FileExistsError("OTEXT Stage1 outputs exist")
    stage0 = json.loads(STAGE0_PATH.read_text()); selected = json.loads(SELECTED_PATH.read_text())
    if not stage0.get("proceed_stage1") or selected.get("status") != "PASS_OTEXT_STAGE0_SELECTED_PARAMETERS":
        raise PermissionError("OTEXT Stage1 not authorized by O-G1")
    rows = load_rows(); row_ids = tuple(sorted(rows))
    metadata = {}
    for path in sorted((ROOT / "runs/ccm-h2h/2026-07-31/h1/shards/top18").glob("shard-*.jsonl")):
        for line in path.read_text().splitlines():
            if line.strip():
                value = json.loads(line); metadata[value["id"]] = value
    raw = {engine: load_raw(engine) for engine in ENGINES}
    e1 = load_module(ROOT / "runs/close/2026-08-08/e1_arm_aggregator_matrix.py", "otext_stage1_e1")
    slots = {row_id: [(candidate["source"], dict(candidate)) for candidate in rows[row_id]["candidates"]] for row_id in row_ids}
    targets = {row_id: rows[row_id]["target_bbox"] for row_id in row_ids}
    derived = []
    for engine in ENGINES:
        fold_parameters = {value["outer_fold"]: value for value in selected["engines"][engine]}
        for outer_fold in range(5):
            parameters = fold_parameters[outer_fold]
            outer_development = [row_id for row_id in row_ids if rows[row_id]["fold"] != outer_fold]
            test = [row_id for row_id in row_ids if rows[row_id]["fold"] == outer_fold]
            priority, reliability = e1.screen_dev_priority(outer_development, slots, targets)
            ocr_values = {
                row_id: best_ocr_box(raw[engine][row_id]["boxes"], metadata[row_id]["instruction"], parameters["extractor"], parameters["matcher"])
                for row_id in outer_development + test
            }
            stage0_fold = next(value for value in stage0["reports"][engine] if value["outer_fold"] == outer_fold)
            blind_theta = next(value["theta"] for value in stage0_fold["selected_matcher_net_curve"] if value["theta_index"] == 0)
            ocr_available_dev = [row_id for row_id in outer_development if ocr_values[row_id]["point"] is not None and ocr_values[row_id]["score"] >= parameters["theta"]]
            ocr_reliability = float(np.mean([point_correct(ocr_values[row_id]["point"], targets[row_id]) for row_id in ocr_available_dev])) if ocr_available_dev else -1.0
            gate_test = [row_id for row_id in test if ocr_values[row_id]["point"] is not None and ocr_values[row_id]["score"] >= parameters["theta"]]
            random_rows = set(sorted(test, key=lambda row_id: hash_int(f"{row_id}|{outer_fold}|20260814|random_gate"))[:len(gate_test)])
            for row_id in test:
                row = rows[row_id]; value = ocr_values[row_id]
                majority = bool(e1.evaluate_screen_method("majority", row_id, slots, priority, targets[row_id]))
                dev_selection = bool(e1.evaluate_screen_method(parameters["selected_method"], row_id, slots, priority, targets[row_id]))
                gate = row_id in gate_test
                ocr_correct = point_correct(value["point"], targets[row_id])
                arm_o_majority = ocr_correct if gate else majority
                arm_o_dev = ocr_correct if gate else dev_selection
                blind_gate = value["point"] is not None and value["score"] >= blind_theta
                blind_majority = ocr_correct if blind_gate else majority
                blind_dev = ocr_correct if blind_gate else dev_selection
                if row_id in random_rows:
                    candidate_index = hash_int(f"{row_id}|{outer_fold}|20260814|random_candidate") % 12
                    random_correct = bool(row["candidates"][candidate_index]["correct"])
                else:
                    random_correct = None
                random_majority = random_correct if random_correct is not None else majority
                random_dev = random_correct if random_correct is not None else dev_selection
                points = [candidate["point"] for candidate in row["candidates"]]; weights = [VISUAL_WEIGHT] * 12
                if gate:
                    points.append(tuple(value["point"])); weights.append(parameters["weight"])
                f_density = complete_output(points, weights, targets[row_id])
                candidate_by_source = {candidate["source"]: candidate for candidate in row["candidates"]}
                frozen_order = {candidate["source"]: index for index, candidate in enumerate(row["candidates"])}
                available_sources = [
                    (reliability[source], -frozen_order[source], bool(candidate_by_source[source]["correct"]))
                    for source in priority
                ]
                if gate:
                    available_sources.append((ocr_reliability, -12, ocr_correct))
                f_majority = max(available_sources)[2]
                derived.append({
                    "schema_version": 1, "engine": engine, "row_id": row_id,
                    "fold": outer_fold, "application": row["application"],
                    "ui_type": metadata[row_id]["ui_type"], "gate": gate,
                    "score": value["score"], "ocr_correct": ocr_correct,
                    "majority": majority, "dev_selection": dev_selection,
                    "arm_o_majority": arm_o_majority, "arm_o_dev_selection": arm_o_dev,
                    "f_density": f_density, "f_majority": f_majority,
                    "blind_majority": blind_majority, "blind_dev_selection": blind_dev,
                    "random_majority": random_majority, "random_dev_selection": random_dev,
                })
    temporary = ROWS_PATH.with_suffix(ROWS_PATH.suffix + ".tmp")
    with temporary.open("w", buffering=1) as handle:
        for row in sorted(derived, key=lambda value: (value["engine"], value["row_id"])):
            handle.write(json.dumps(row, sort_keys=True) + "\n"); handle.flush(); os.fsync(handle.fileno())
    temporary.replace(ROWS_PATH)
    reports = {}
    offsets = {
        "O_P1_majority": ("arm_o_majority", "majority", 101),
        "O_P1_dev_selection": ("arm_o_dev_selection", "dev_selection", 102),
        "O_P2_density_majority": ("f_density", "majority", 201),
        "O_P2_density_dev_selection": ("f_density", "dev_selection", 202),
        "O_P2_majority_majority": ("f_majority", "majority", 203),
        "O_P2_majority_dev_selection": ("f_majority", "dev_selection", 204),
        "random_majority": ("arm_o_majority", "random_majority", 301),
        "random_dev_selection": ("arm_o_dev_selection", "random_dev_selection", 302),
        "blind_majority": ("arm_o_majority", "blind_majority", 401),
        "blind_dev_selection": ("arm_o_dev_selection", "blind_dev_selection", 402),
    }
    for engine_index, engine in enumerate(ENGINES):
        values = [row for row in derived if row["engine"] == engine]
        comparisons = {name: grouped_bootstrap(values, left, right, 20260814 + engine_index * 1000 + offset) for name, (left, right, offset) in offsets.items()}
        gate_values = [row for row in values if row["gate"]]
        conditional = {}
        for baseline in ("majority", "dev_selection"):
            correct = [row for row in gate_values if row[baseline]]; wrong = [row for row in gate_values if not row[baseline]]
            conditional[baseline] = {
                "OCR_correct_given_baseline_correct": float(np.mean([row["ocr_correct"] for row in correct])) if correct else None,
                "OCR_correct_given_baseline_wrong": float(np.mean([row["ocr_correct"] for row in wrong])) if wrong else None,
                "gate_rows": len(gate_values),
                "arm_o_gate_accuracy": float(np.mean([row[f"arm_o_{'dev_selection' if baseline == 'dev_selection' else 'majority'}"] for row in gate_values])) if gate_values else None,
                "baseline_gate_accuracy": float(np.mean([row[baseline] for row in gate_values])) if gate_values else None,
            }
        by_type = {
            ui_type: {
                "rows": len(selected_rows),
                "arm_o_majority_accuracy": float(np.mean([row["arm_o_majority"] for row in selected_rows])),
                "majority_accuracy": float(np.mean([row["majority"] for row in selected_rows])),
                "gate_rate": float(np.mean([row["gate"] for row in selected_rows])),
            }
            for ui_type in ("text", "icon")
            for selected_rows in [[row for row in values if row["ui_type"] == ui_type]]
        }
        reports[engine] = {"rows": len(values), "gate_rows": len(gate_values), "comparisons": comparisons, "conditional": conditional, "ui_type": by_type}
    easy = reports["easyocr"]["comparisons"]
    o_p1 = easy["O_P1_majority"]["ci_99"][0] > 0 and easy["O_P1_dev_selection"]["ci_99"][0] > 0
    o_p2 = all(easy[name]["ci_99"][0] > 0 for name in ("O_P2_density_majority", "O_P2_density_dev_selection", "O_P2_majority_majority", "O_P2_majority_dev_selection"))
    result = {
        "schema_version": 1, "status": "PASS_OTEXT_STAGE1_COMPLETE",
        "evidence_status": "POST_SELECTION_VALIDATION", "reports": reports,
        "endpoints": {"O_P1": o_p1, "O_P2": o_p2, "O_P3": "DESCRIPTIVE", "O_P4": "DESCRIPTIVE", "O_P5": "DESCRIPTIVE"},
        "kill_conditions": {
            "O_K2": not o_p1,
            "O_K3": not (easy["random_majority"]["ci_99"][0] > 0 and easy["random_dev_selection"]["ci_99"][0] > 0),
            "O_K4": not (easy["blind_majority"]["ci_99"][0] > 0 and easy["blind_dev_selection"]["ci_99"][0] > 0),
            "O_K5": all(easy[name]["ci_99"][0] > 0 for name in ("O_P2_density_majority", "O_P2_density_dev_selection")) and not all(easy[name]["ci_99"][0] > 0 for name in ("O_P2_majority_majority", "O_P2_majority_dev_selection")),
            "O_K6": o_p1 != (reports["rapidocr"]["comparisons"]["O_P1_majority"]["ci_99"][0] > 0 and reports["rapidocr"]["comparisons"]["O_P1_dev_selection"]["ci_99"][0] > 0),
        },
    }
    OUTPUT_PATH.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"endpoints": result["endpoints"], "kill_conditions": result["kill_conditions"], "easy_O_P1": {name: easy[name] for name in ("O_P1_majority", "O_P1_dev_selection")}}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()