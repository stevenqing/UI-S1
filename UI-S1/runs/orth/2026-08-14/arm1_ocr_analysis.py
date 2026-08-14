import json
import math
import multiprocessing as mp
import os
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


RUN_DIR = Path(__file__).resolve().parent
ROOT = RUN_DIR.parents[2]
RAW_ROOT = RUN_DIR / "raw"
OUTPUT_PATH = RUN_DIR / "ARM1.json"
ROWS_PATH = RUN_DIR / "ARM1_ROWS.jsonl"
sys.path.insert(0, str(ROOT / "runs/mask/2026-08-14"))

from mask_common import b3_correct, load_rows


ENGINES = ("easyocr", "rapidocr")
MIN_LENGTHS = (1, 2, 3, 4, 5)
EDIT_THRESHOLDS = (0.5, 0.6, 0.7, 0.8, 0.9)
SETTINGS = (
    [("exact", length, None) for length in MIN_LENGTHS]
    + [("normalized", length, None) for length in MIN_LENGTHS]
    + [("edit", length, threshold) for length in MIN_LENGTHS for threshold in EDIT_THRESHOLDS]
)
_WORKER = {}


def normalize_text(value):
    value = unicodedata.normalize("NFKC", str(value)).casefold()
    value = " ".join(value.split())
    return value.strip(" \t\n\r.,;:!?()[]{}<>\"'`~_-|/\\")


def edit_distance(left, right):
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for left_index, left_value in enumerate(left, 1):
        current = [left_index]
        for right_index, right_value in enumerate(right, 1):
            current.append(min(
                current[-1] + 1,
                previous[right_index] + 1,
                previous[right_index - 1] + (left_value != right_value),
            ))
        previous = current
    return previous[-1]


def edit_similarity(text, instruction):
    normalized_text = normalize_text(text)
    tokens = normalize_text(instruction).split()
    return edit_similarity_prepared(normalized_text, tokens)


def edit_similarity_prepared(normalized_text, tokens):
    if not normalized_text or not tokens:
        return 0.0
    target_length = len(normalized_text)
    minimum = max(1, math.floor(0.8 * target_length))
    maximum = math.ceil(1.2 * target_length)
    best = 0.0
    for left in range(len(tokens)):
        value = ""
        for right in range(left, len(tokens)):
            value = tokens[right] if not value else value + " " + tokens[right]
            if len(value) > maximum:
                break
            if len(value) >= minimum:
                best = max(best, 1 - edit_distance(normalized_text, value) / max(len(normalized_text), len(value)))
    return best


def box_center(polygon):
    values = np.asarray(polygon, dtype=np.float64)
    return float(values[:, 0].mean()), float(values[:, 1].mean())


def point_in_bbox(point, bbox):
    return bbox[0] <= point[0] <= bbox[2] and bbox[1] <= point[1] <= bbox[3]


def match_box(boxes, instruction, matcher, minimum_length, threshold=None):
    return select_prepared(prepare_boxes(boxes, instruction), matcher, minimum_length, threshold)


def prepare_boxes(boxes, instruction):
    normalized_instruction = normalize_text(instruction)
    instruction_tokens = normalized_instruction.split()
    output = []
    for box in boxes:
        raw = str(box["text"]).strip()
        normalized = normalize_text(raw)
        output.append({
            "box": box,
            "raw": raw,
            "normalized": normalized,
            "exact_matched": bool(raw and raw in instruction),
            "normalized_matched": bool(normalized and normalized in normalized_instruction),
            "edit_score": edit_similarity_prepared(normalized, instruction_tokens),
        })
    return output


def select_prepared(prepared, matcher, minimum_length, threshold=None):
    candidates = []
    for value in prepared:
        box = value["box"]
        raw = value["raw"]
        normalized = value["normalized"]
        if len(normalized) < minimum_length:
            continue
        if matcher == "exact":
            matched = value["exact_matched"]
            score = len(raw) if matched else 0.0
        elif matcher == "normalized":
            matched = value["normalized_matched"]
            score = len(normalized) if matched else 0.0
        elif matcher == "edit":
            score = value["edit_score"]
            matched = score >= float(threshold)
        else:
            raise ValueError(matcher)
        if matched:
            top = min(point[1] for point in box["polygon"])
            left = min(point[0] for point in box["polygon"])
            candidates.append((
                float(score), len(normalized), float(box["confidence"]),
                -top, -left, -int(box["engine_order"]), box,
            ))
    return max(candidates, default=None)[-1] if candidates else None


def cohen_kappa(left, right):
    left = np.asarray(left, dtype=np.bool_)
    right = np.asarray(right, dtype=np.bool_)
    observed = float(np.mean(left == right))
    left_rate = float(np.mean(left)); right_rate = float(np.mean(right))
    expected = left_rate * right_rate + (1 - left_rate) * (1 - right_rate)
    return None if math.isclose(expected, 1.0) else float((observed - expected) / (1 - expected))


def load_raw(engine):
    output = {}
    for path in sorted((RAW_ROOT / engine).glob("shard-*.jsonl")):
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row["row_id"] in output:
                raise ValueError(f"ORTH duplicate OCR row: {engine}/{row['row_id']}")
            output[row["row_id"]] = row
    if len(output) != 1581:
        raise ValueError(f"ORTH OCR coverage mismatch: {engine}/{len(output)}")
    return output


def row_classes(rows):
    output = {}
    for row_id, row in rows.items():
        selected = b3_correct(row["candidates"], row["target_bbox"])
        oracle = any(candidate["correct"] for candidate in row["candidates"])
        output[row_id] = "selected_correct" if selected else "recoverable" if oracle else "zero_coverage"
    return output


def summarize(setting_rows):
    total = len(setting_rows)
    matched = [row for row in setting_rows if row["matched"]]
    table = {
        class_name: {
            "matched": sum(row["matched"] for row in setting_rows if row["row_class"] == class_name),
            "unmatched": sum(not row["matched"] for row in setting_rows if row["row_class"] == class_name),
        }
        for class_name in ("selected_correct", "recoverable", "zero_coverage")
    }
    by_type = {}
    for ui_type in ("text", "icon"):
        selected = [row for row in setting_rows if row["ui_type"] == ui_type]
        selected_matched = [row for row in selected if row["matched"]]
        by_type[ui_type] = {
            "rows": len(selected),
            "match_rate": float(np.mean([row["matched"] for row in selected])),
            "all_row_accuracy": float(np.mean([row["correct"] for row in selected])),
            "matched_only_accuracy": float(np.mean([row["correct"] for row in selected_matched])) if selected_matched else None,
        }
    kappas = {}
    for class_name in ("all", "selected_correct", "recoverable", "zero_coverage"):
        selected = setting_rows if class_name == "all" else [row for row in setting_rows if row["row_class"] == class_name]
        if class_name != "all":
            kappas[class_name] = "UNDEFINED_STRATIFIED_POOL_ERROR_CONSTANT"
        else:
            value = cohen_kappa([not row["correct"] for row in selected], [row["pool_error"] for row in selected])
            kappas[class_name] = value if value is not None else "UNDEFINED_DEGENERATE"
    return {
        "rows": total,
        "matched_rows": len(matched),
        "match_rate": len(matched) / total,
        "candidate_box_distribution": {
            "mean": float(np.mean([row["matching_boxes"] for row in setting_rows])),
            "median": float(np.median([row["matching_boxes"] for row in setting_rows])),
            "maximum": max(row["matching_boxes"] for row in setting_rows),
        },
        "class_match_table": table,
        "ui_type": by_type,
        "all_row_accuracy": float(np.mean([row["correct"] for row in setting_rows])),
        "matched_only_accuracy": float(np.mean([row["correct"] for row in matched])) if matched else None,
        "error_kappa": kappas,
        "same_error_outcome_as_pool": float(np.mean([(not row["correct"]) == row["pool_error"] for row in setting_rows])),
    }


def analyze_row(row_id):
    row = _WORKER["rows"][row_id]
    source = _WORKER["raw"][row_id]
    meta = _WORKER["metadata"][row_id]
    instruction = meta["instruction"]
    prepared = prepare_boxes(source["boxes"], instruction)
    output = []
    for matcher, minimum_length, threshold in SETTINGS:
        matched_box = select_prepared(prepared, matcher, minimum_length, threshold)
        matching_boxes = sum(
            select_prepared([value], matcher, minimum_length, threshold) is not None
            for value in prepared
        )
        correct = bool(
            matched_box is not None
            and point_in_bbox(box_center(matched_box["polygon"]), row["target_bbox"])
        )
        output.append({
            "schema_version": 1,
            "engine": _WORKER["engine"],
            "row_id": row_id,
            "matcher": matcher,
            "minimum_length": minimum_length,
            "edit_threshold": threshold,
            "row_class": _WORKER["classes"][row_id],
            "ui_type": meta["ui_type"],
            "matched": matched_box is not None,
            "matching_boxes": matching_boxes,
            "selected_box_order": matched_box["engine_order"] if matched_box else None,
            "selected_text": matched_box["text"] if matched_box else None,
            "selected_center": list(box_center(matched_box["polygon"])) if matched_box else None,
            "correct": correct,
            "pool_error": _WORKER["classes"][row_id] != "selected_correct",
        })
    return output


def main():
    if OUTPUT_PATH.exists() or ROWS_PATH.exists():
        raise FileExistsError("ORTH Arm 1 outputs exist")
    rows = load_rows()
    classes = row_classes(rows)
    metadata = {}
    for path in sorted((ROOT / "runs/ccm-h2h/2026-07-31/h1/shards/top18").glob("shard-*.jsonl")):
        for line in path.read_text().splitlines():
            if line.strip():
                row = json.loads(line); metadata[row["id"]] = row
    if len(metadata) != 1581:
        raise ValueError("ORTH SSPro metadata mismatch")
    all_setting_rows = []
    summaries = {}
    for engine in ENGINES:
        raw = load_raw(engine)
        engine_summaries = {}
        _WORKER.clear()
        _WORKER.update({
            "engine": engine, "raw": raw, "rows": rows,
            "metadata": metadata, "classes": classes,
        })
        with mp.get_context("fork").Pool(min(48, os.cpu_count() or 1)) as pool:
            nested = pool.map(analyze_row, sorted(rows), chunksize=2)
        engine_rows = [value for row_values in nested for value in row_values]
        by_setting = defaultdict(list)
        for value in engine_rows:
            setting_id = f"{value['matcher']}/min{value['minimum_length']}" + (
                f"/threshold{value['edit_threshold']:.1f}"
                if value["edit_threshold"] is not None else ""
            )
            by_setting[setting_id].append(value)
        for matcher, minimum_length, threshold in SETTINGS:
            setting_id = f"{matcher}/min{minimum_length}" + (f"/threshold{threshold:.1f}" if threshold is not None else "")
            engine_summaries[setting_id] = summarize(by_setting[setting_id])
        all_setting_rows.extend(engine_rows)
        summaries[engine] = engine_summaries
    engine_overlap = {}
    overlap_rows = defaultdict(dict)
    for value in all_setting_rows:
        setting_id = f"{value['matcher']}/min{value['minimum_length']}" + (
            f"/threshold{value['edit_threshold']:.1f}"
            if value["edit_threshold"] is not None else ""
        )
        overlap_rows[(setting_id, value["row_id"])][value["engine"]] = value["matched"]
    for matcher, minimum_length, threshold in SETTINGS:
        setting_id = f"{matcher}/min{minimum_length}" + (f"/threshold{threshold:.1f}" if threshold is not None else "")
        values = [(row_id, overlap_rows[(setting_id, row_id)]) for row_id in sorted(rows)]
        engine_overlap[setting_id] = {
            "union_matched": sum(any(flags.values()) for _, flags in values),
            "intersection_matched": sum(all(flags.get(engine, False) for engine in ENGINES) for _, flags in values),
            "union_match_rate": sum(any(flags.values()) for _, flags in values) / len(values),
            "intersection_match_rate": sum(all(flags.get(engine, False) for engine in ENGINES) for _, flags in values) / len(values),
            "union_by_class": {
                class_name: sum(any(flags.values()) for row_id, flags in values if classes[row_id] == class_name)
                for class_name in ("selected_correct", "recoverable", "zero_coverage")
            },
        }
    temporary = ROWS_PATH.with_suffix(ROWS_PATH.suffix + ".tmp")
    with temporary.open("w", buffering=1) as handle:
        for row in all_setting_rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n"); handle.flush(); os.fsync(handle.fileno())
    temporary.replace(ROWS_PATH)
    result = {
        "schema_version": 1,
        "status": "PASS_ORTH_ARM1_SCOPING_COMPLETE",
        "engines": summaries,
        "engine_overlap": engine_overlap,
        "row_class_counts": dict(sorted(Counter(classes.values()).items())),
        "settings_per_engine": len(next(iter(summaries.values()))),
        "rows_jsonl": ROWS_PATH.relative_to(ROOT).as_posix(),
        "claim_boundary": {
            "exploratory_scoping_only": True,
            "best_setting_selection_prohibited": True,
            "label_dependent_metrics_evaluation_side_only": True,
        },
    }
    OUTPUT_PATH.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": result["status"],
        "row_classes": result["row_class_counts"],
        "settings_per_engine": result["settings_per_engine"],
        "coverage_ranges": {
            engine: [min(value["match_rate"] for value in settings.values()), max(value["match_rate"] for value in settings.values())]
            for engine, settings in summaries.items()
        },
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()