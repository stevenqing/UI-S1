import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import yaml
from qwen_vl_utils import smart_resize
from transformers import AutoProcessor


RUN_DIR = Path(__file__).resolve().parent
ROOT = RUN_DIR.parents[2]
GRAN_DIR = ROOT / "runs/gran/2026-08-14"
CONFIG_PATH = RUN_DIR / "configs/split_prereg.yaml"
GATE_PATH = RUN_DIR / "ZERO_GPU_GATE.json"
PREFLIGHT_PATH = RUN_DIR / "PREFLIGHT.json"
OUTPUT_PATH = RUN_DIR / "GEOMETRY_AUDIT.json"
sys.path.insert(0, str(GRAN_DIR))


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def contains(window, point):
    left, top, right, bottom = window
    return left <= point[0] < right and top <= point[1] < bottom


def clamp_origin(value, side, extent):
    return min(max(int(value), 0), extent - side)


def target_window(target, other, side, width, height):
    delta_x = abs(target[0] - other[0])
    delta_y = abs(target[1] - other[1])
    axis = 0 if delta_x >= delta_y else 1
    origin = [math.floor(target[0] - side / 2), math.floor(target[1] - side / 2)]
    if target[axis] < other[axis]:
        origin[axis] = math.ceil(target[axis] - side + 1)
    else:
        origin[axis] = math.floor(target[axis] - 1)
    origin[0] = clamp_origin(origin[0], side, width)
    origin[1] = clamp_origin(origin[1], side, height)
    window = [origin[0], origin[1], origin[0] + side, origin[1] + side]
    return window, axis


def empty_window(center1, center2, side, width, height):
    corners = (
        ("top_left", [0, 0, side, side]),
        ("top_right", [width - side, 0, width, side]),
        ("bottom_left", [0, height - side, side, height]),
        ("bottom_right", [width - side, height - side, width, height]),
    )
    scored = []
    for index, (name, window) in enumerate(corners):
        center = ((window[0] + window[2]) / 2, (window[1] + window[3]) / 2)
        distance = min(math.dist(center, center1), math.dist(center, center2))
        scored.append((distance, -index, name, window))
    _, _, name, window = max(scored)
    return window, name


def mode_geometry(candidates, members, diagonal):
    points = np.asarray([
        [
            candidates[order].coordinate[0] * diagonal,
            candidates[order].coordinate[1] * diagonal,
        ]
        for order in members
    ], dtype=np.float64)
    center = points.mean(axis=0)
    sigma = float(np.sqrt(np.mean(np.sum((points - center) ** 2, axis=1))))
    return (float(center[0]), float(center[1])), sigma


def load_resize_specs(preflight):
    output = {}
    for model_id in ("Qwen3-VL-8B-Instruct", "GTA1-7B"):
        model_path = ROOT / preflight["models"][model_id]["path"]
        processor = AutoProcessor.from_pretrained(
            model_path, min_pixels=1_000_000, max_pixels=4_000_000, use_fast=False
        )
        image_processor = processor.image_processor
        output[model_id] = {
            "factor": int(image_processor.patch_size * image_processor.merge_size),
            "patch_size": int(image_processor.patch_size),
            "merge_size": int(image_processor.merge_size),
        }
    return output


def main():
    if OUTPUT_PATH.exists():
        raise FileExistsError(OUTPUT_PATH)
    config = yaml.safe_load(CONFIG_PATH.read_text())
    gate = json.loads(GATE_PATH.read_text())
    preflight = json.loads(PREFLIGHT_PATH.read_text())
    if (
        gate.get("status") != "PASS_Z_G1_PROCEED_TO_GEOMETRY"
        or gate.get("Z_G1_pass") is not True
        or gate.get("probe_forward_started") is not False
    ):
        raise PermissionError("SPLIT geometry requires passed zero-GPU gate")
    if config["window_geometry"].get("integer_side") != "ceil_continuous_side_capped_at_min_image_side":
        raise PermissionError("SPLIT integerization amendment missing")
    gran_runner = load_module(GRAN_DIR / "run_tau_sweep.py", "split_geometry_gran")
    e1 = gran_runner.load_module(
        gran_runner.CLOSE_DIR / "e1_arm_aggregator_matrix.py", "split_geometry_e1"
    )
    rows = gran_runner.load_screen_rows(e1)
    resize_specs = load_resize_specs(preflight)
    audits = {}
    for row_id, gate_row in sorted(gate["heldout_rows"].items()):
        if not gate_row["gate"]:
            continue
        image_record = preflight["images"][row_id]
        width, height = map(int, image_record["declared_size"])
        diagonal = math.hypot(width, height)
        candidates = rows[row_id]["candidates"]
        center1, sigma1 = mode_geometry(candidates, gate_row["M1_members"], diagonal)
        center2, sigma2 = mode_geometry(candidates, gate_row["M2_members"], diagonal)
        continuous_side = min(
            min(width, height), max(512.0, 2.5 * max(sigma1, sigma2))
        )
        side = min(min(width, height), math.ceil(continuous_side))
        window1, axis1 = target_window(center1, center2, side, width, height)
        window2, axis2 = target_window(center2, center1, side, width, height)
        window0, corner0 = empty_window(center1, center2, side, width, height)
        include_exclude = {
            "W1_includes_M1": contains(window1, center1),
            "W1_excludes_M2": not contains(window1, center2),
            "W2_includes_M2": contains(window2, center2),
            "W2_excludes_M1": not contains(window2, center1),
            "W0_excludes_M1": not contains(window0, center1),
            "W0_excludes_M2": not contains(window0, center2),
        }
        resized = {}
        for model_id, spec in resize_specs.items():
            resized_height, resized_width = smart_resize(
                side,
                side,
                factor=spec["factor"],
                min_pixels=int(config["preprocessing"]["crop_min_pixels"]),
                max_pixels=int(config["preprocessing"]["crop_max_pixels"]),
            )
            resized[model_id] = {
                "W1": [resized_width, resized_height],
                "W2": [resized_width, resized_height],
                "W0": [resized_width, resized_height],
                "exact_match": True,
            }
        valid = all(include_exclude.values()) and all(
            value["exact_match"] for value in resized.values()
        )
        audits[row_id] = {
            "fold": gate_row["fold"],
            "application": gate_row["application"],
            "positive": gate_row["positive"],
            "negative": gate_row["negative"],
            "image_size": [width, height],
            "center1": list(center1),
            "center2": list(center2),
            "sigma1": sigma1,
            "sigma2": sigma2,
            "continuous_side": continuous_side,
            "side": side,
            "W1": window1,
            "W2": window2,
            "W0": window0,
            "W1_separation_axis": axis1,
            "W2_separation_axis": axis2,
            "W0_corner": corner0,
            "include_exclude": include_exclude,
            "area_ratio_range": [1.0, 1.0],
            "aspect_ratio_difference": 0.0,
            "resized": resized,
            "valid": valid,
        }
    valid_rows = sum(row["valid"] for row in audits.values())
    geometry_failed = len(audits) - valid_rows
    positive_total = sum(row["positive"] for row in audits.values())
    positive_valid = sum(row["positive"] and row["valid"] for row in audits.values())
    negative_total = sum(row["negative"] for row in audits.values())
    negative_valid = sum(row["negative"] and row["valid"] for row in audits.values())
    failure_rate = float(geometry_failed / len(audits))
    maximum = float(config["window_geometry"]["maximum_failure_rate"])
    result = {
        "schema_version": 1,
        "status": "PASS_GEOMETRY_AUDIT" if failure_rate <= maximum else "STOP_Z_K6_GEOMETRY",
        "zero_gpu": True,
        "model_forward_started": False,
        "resize_specs": resize_specs,
        "gate_rows": len(audits),
        "valid_rows": valid_rows,
        "geometry_failed_rows": geometry_failed,
        "geometry_failure_rate": failure_rate,
        "maximum_failure_rate": maximum,
        "positive_rows_before_geometry": positive_total,
        "positive_rows_after_geometry": positive_valid,
        "negative_rows_before_geometry": negative_total,
        "negative_rows_after_geometry": negative_valid,
        "Z_K6_geometry_triggered": failure_rate > maximum,
        "Z_K7_observational_only": positive_valid < int(
            config["balanced_subset"]["minimum_positive_rows_for_decision"]
        ),
        "rows": audits,
    }
    OUTPUT_PATH.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "rows"}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()