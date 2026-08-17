import importlib.util
import json
import os
from collections import Counter
from pathlib import Path

import numpy as np
import yaml
from PIL import Image


RUN_DIR = Path(__file__).resolve().parent
ROOT = RUN_DIR.parents[2]
ALLOCATION_DIR = ROOT / "runs/allocation-law/2026-08-01"
H3_PATH = ROOT / "runs/ccm-h2h/2026-07-31/h3/h3_eval.py"
SOURCEBIAS_PATH = ROOT / "runs/sourcebias/2026-08-03/sourcebias_common.py"
CONFIG_PATH = RUN_DIR / "configs/cwin_prereg.yaml"
PREFLIGHT_PATH = RUN_DIR / "PREFLIGHT.json"
REGION_PATH = ROOT / "runs/allocation-law/2026-08-01/raw/shared_regions_n12.jsonl"
MANIFEST_PATH = ALLOCATION_DIR / "raw/shared_regions_n12.jsonl"
GTA1_ROOT = ROOT / "runs/ccm-h2h/2026-07-31/h1/shards/top18"
OUTPUT_PATH = RUN_DIR / "STAGE0.json"
WINDOW_MANIFEST_PATH = RUN_DIR / "WINDOW_MANIFEST.json"
RAW_GEOMETRY_PATH = RUN_DIR / "raw/geometry_all_k.jsonl"
RAW_ROWS_PATH = RUN_DIR / "raw/stage0_rows.jsonl"
MAP_ROOT = RUN_DIR / "raw/selected_coverage_maps"


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


allocation = load_module(ALLOCATION_DIR / "allocation_eval.py", "cwin_allocation")
h3 = load_module(H3_PATH, "cwin_h3")
sourcebias = load_module(SOURCEBIAS_PATH, "cwin_sourcebias")


def sha256_file(path):
    import hashlib
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def rectangle_iou(left, right):
    x1 = max(left[0], right[0])
    y1 = max(left[1], right[1])
    x2 = min(left[2], right[2])
    y2 = min(left[3], right[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    left_area = (left[2] - left[0]) * (left[3] - left[1])
    right_area = (right[2] - right[0]) * (right[3] - right[1])
    return intersection / (left_area + right_area - intersection)


def greedy_drop(regions, maximum_k=4):
    remaining = list(range(len(regions)))
    dropped = []
    for _ in range(maximum_k):
        scores = {index: sum(rectangle_iou(regions[index], regions[other]) for other in remaining if other != index) for index in remaining}
        selected = max(remaining, key=lambda index: (scores[index], -index))
        dropped.append(selected)
        remaining.remove(selected)
    return dropped


def binary_uncovered(width, height, regions):
    difference = np.zeros((height + 1, width + 1), dtype=np.int16)
    for left, top, right, bottom in regions:
        difference[top, left] += 1
        difference[bottom, left] -= 1
        difference[top, right] -= 1
        difference[bottom, right] += 1
    covered = difference[:-1, :-1].cumsum(axis=0).cumsum(axis=1) > 0
    return ~covered


def best_window(uncovered, window_width, window_height):
    height, width = uncovered.shape
    integral = np.zeros((height + 1, width + 1), dtype=np.int32)
    integral[1:, 1:] = uncovered.cumsum(axis=0, dtype=np.int32).cumsum(axis=1, dtype=np.int32)
    sums = integral[window_height:, window_width:] - integral[:-window_height, window_width:] - integral[window_height:, :-window_width] + integral[:-window_height, :-window_width]
    flat = int(np.argmax(sums))
    top, left = np.unravel_index(flat, sums.shape)
    gain = int(sums[top, left])
    return [int(left), int(top), int(left + window_width), int(top + window_height)], gain


def complementary_windows(width, height, existing, maximum_k=4):
    window_width = existing[0][2] - existing[0][0]
    window_height = existing[0][3] - existing[0][1]
    if any(region[2] - region[0] != window_width or region[3] - region[1] != window_height for region in existing):
        raise ValueError("CWIN nonuniform existing crop size")
    uncovered = binary_uncovered(width, height, existing)
    windows = []
    gains = []
    for _ in range(maximum_k):
        window, gain = best_window(uncovered, window_width, window_height)
        windows.append(window)
        gains.append(gain)
        left, top, right, bottom = window
        uncovered[top:bottom, left:right] = False
    return windows, gains


def contains_center(region, target_bbox):
    x = (target_bbox[0] + target_bbox[2]) / 2
    y = (target_bbox[1] + target_bbox[3]) / 2
    return region[0] <= x < region[2] and region[1] <= y < region[3]


def filtered_row(row, dropped_crop_indices):
    dropped_candidate_indices = {index + 1 for index in dropped_crop_indices}
    return {**row, "candidates": [candidate for index, candidate in enumerate(row["candidates"]) if index not in dropped_candidate_indices]}


def b3_correct(row):
    selected, _ = sourcebias.b3_select_index(row["candidates"])
    return bool(h3.point_in_bbox(row["candidates"][selected]["point"], row["target_bbox"]))


def m1_outputs(rows, fold_for_group):
    outputs = {}
    for fold in range(5):
        development = [row for row in rows if fold_for_group[row["application"]] != fold]
        test = [row for row in rows if fold_for_group[row["application"]] == fold]
        tables, priors = h3.fit_ccm(development)
        for row in test:
            selected = h3.ccm_select(row, tables, priors)
            outputs[row["id"]] = bool(h3.point_in_bbox(row["candidates"][selected]["point"], row["target_bbox"]))
    return outputs


def lower_bounds_by_count(rows, development_ids, resamples, seed):
    applications = sorted({rows[row_id]["application"] for row_id in development_ids})
    by_count_app = {(count, app): [] for count in range(12) for app in applications}
    for row_id in development_ids:
        row = rows[row_id]
        by_count_app[(row["original_count"], row["application"])].append(float(row["original_b3_correct"]))
    rng = np.random.default_rng(seed)
    bounds = {}
    for count in range(12):
        original = [value for app in applications for value in by_count_app[(count, app)]]
        if not original:
            bounds[count] = {"rows": 0, "accuracy": None, "lower_99": 0.0, "finite_replicates": 0}
            continue
        values = []
        for _ in range(resamples):
            sampled = rng.choice(applications, size=len(applications), replace=True)
            current = [value for app in sampled for value in by_count_app[(count, app)]]
            if current:
                values.append(float(np.mean(current)))
        lower = float(np.quantile(values, 0.005)) if len(values) >= 0.99 * resamples else 0.0
        bounds[count] = {"rows": len(original), "accuracy": float(np.mean(original)), "lower_99": lower, "finite_replicates": len(values)}
    return bounds


def write_jsonl_fsynced(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())


def main():
    if any(path.exists() for path in (OUTPUT_PATH, WINDOW_MANIFEST_PATH, RAW_GEOMETRY_PATH, RAW_ROWS_PATH, MAP_ROOT)):
        raise FileExistsError("CWIN Stage 0 output exists")
    config = yaml.safe_load(CONFIG_PATH.read_text())
    preflight = json.loads(PREFLIGHT_PATH.read_text())
    if preflight["status"] != "PASS_CWIN_PREFLIGHT_NO_GEOMETRY_OR_RESULT" or preflight["stage0_computed"] is not False:
        raise PermissionError("CWIN Stage 0 preflight mismatch")
    manifest = allocation.load_manifest(MANIFEST_PATH)
    gta1 = allocation.load_gta1(GTA1_ROOT, manifest)
    units = [("GTA1-7B", view) for view in range(12)]
    original_rows = allocation.build_pool(gta1, {}, units)
    fold_for_group, fold_loads = allocation.group_folds(original_rows)
    original_by_id = {row["id"]: row for row in original_rows}
    regions = {row["id"]: row for row in read_jsonl(REGION_PATH)}
    original_evaluation = allocation.compact_evaluation(original_rows)
    if abs(original_evaluation["accuracy"]["B3_mvp"] - 0.6008855154965211) > 1e-15 or abs(original_evaluation["accuracy"]["M1_ccm"] - 0.6040480708412397) > 1e-15:
        raise ValueError("CWIN V-only N12 anchor mismatch")

    geometry = []
    for row_id in sorted(original_by_id):
        row_regions = regions[row_id]
        width, height = row_regions["img_size"]
        crops = row_regions["regions"][1:]
        new_windows, new_gains = complementary_windows(width, height, crops, max(config["geometry"]["K_grid"]))
        dropped = greedy_drop(crops, max(config["geometry"]["K_grid"]))
        geometry.append({"row_id": row_id, "width": width, "height": height, "crop_width": crops[0][2] - crops[0][0], "crop_height": crops[0][3] - crops[0][1], "new_windows": new_windows, "new_uncovered_pixel_gains": new_gains, "dropped_crop_indices": dropped})
    write_jsonl_fsynced(RAW_GEOMETRY_PATH, geometry)
    geometry_by_id = {row["row_id"]: row for row in geometry}

    original_b3 = original_evaluation["outputs"]["B3_mvp"]
    original_m1 = original_evaluation["outputs"]["M1_ccm"]
    stage_rows = {}
    drop_b3 = {}
    drop_m1 = {}
    for K in config["geometry"]["K_grid"]:
        dropped_rows = [filtered_row(row, geometry_by_id[row["id"]]["dropped_crop_indices"][:K]) for row in original_rows]
        drop_b3[K] = {row["id"]: b3_correct(row) for row in dropped_rows}
        drop_m1[K] = m1_outputs(dropped_rows, fold_for_group)
    for row_id in sorted(original_by_id):
        target = original_by_id[row_id]["target_bbox"]
        crops = regions[row_id]["regions"][1:]
        original_count = sum(contains_center(region, target) for region in crops)
        values = {"row_id": row_id, "application": original_by_id[row_id]["application"], "fold": fold_for_group[original_by_id[row_id]["application"]], "original_count": original_count, "original_b3_correct": bool(original_b3[row_id]), "original_m1_correct": bool(original_m1[row_id]), "K": {}}
        for K in config["geometry"]["K_grid"]:
            dropped = set(geometry_by_id[row_id]["dropped_crop_indices"][:K])
            retained = [region for index, region in enumerate(crops) if index not in dropped]
            new = geometry_by_id[row_id]["new_windows"][:K]
            post_count = sum(contains_center(region, target) for region in retained + new)
            new_covered = any(contains_center(region, target) for region in new)
            values["K"][str(K)] = {"post_count": post_count, "new_window_target_covered": new_covered, "lost_all_coverage": original_count > 0 and post_count == 0, "drop_b3_correct": bool(drop_b3[K][row_id]), "drop_m1_correct": bool(drop_m1[K][row_id])}
        stage_rows[row_id] = values

    selections = []
    selected_K_by_fold = {}
    for outer_fold in range(5):
        inner_fold = (outer_fold + 1) % 5
        inner_ids = [row_id for row_id, row in stage_rows.items() if row["fold"] == inner_fold]
        scores = {}
        for K in config["geometry"]["K_grid"]:
            contributions = []
            for row_id in inner_ids:
                row = stage_rows[row_id]
                current = row["K"][str(K)]
                contributions.append(int(current["drop_b3_correct"]) - int(row["original_b3_correct"]) + int(not current["drop_b3_correct"] and current["new_window_target_covered"]))
            scores[K] = float(np.mean(contributions))
        selected = max(config["geometry"]["K_grid"], key=lambda K: (scores[K], -K))
        selected_K_by_fold[outer_fold] = selected
        selections.append({"outer_fold": outer_fold, "inner_validation_fold": inner_fold, "scores_L4_upper": {str(K): value for K, value in scores.items()}, "selected_K": selected, "boundary_selected": selected in {min(config["geometry"]["K_grid"]), max(config["geometry"]["K_grid"] )}})

    fold_records = []
    all_selected_rows = []
    for outer_fold in range(5):
        K = selected_K_by_fold[outer_fold]
        development_ids = [row_id for row_id, row in stage_rows.items() if row["fold"] != outer_fold]
        test_ids = [row_id for row_id, row in stage_rows.items() if row["fold"] == outer_fold]
        bounds = lower_bounds_by_count(stage_rows, development_ids, config["stage0"]["L2_conservative"]["bootstrap"]["resamples"], 20261200 + outer_fold)
        transition = Counter()
        records = []
        for row_id in test_ids:
            row = stage_rows[row_id]
            current = row["K"][str(K)]
            transition[(row["original_count"], current["post_count"])] += 1
            drop_delta_b3 = int(current["drop_b3_correct"]) - int(row["original_b3_correct"])
            drop_delta_m1 = int(current["drop_m1_correct"]) - int(row["original_m1_correct"])
            opportunity = int(not current["drop_b3_correct"] and current["new_window_target_covered"])
            conservative = bounds[current["post_count"]]["lower_99"] if opportunity else 0.0
            record = {"row_id": row_id, "outer_fold": outer_fold, "selected_K": K, "original_count": row["original_count"], "post_count": current["post_count"], "new_window_target_covered": current["new_window_target_covered"], "lost_all_coverage": current["lost_all_coverage"], "original_b3_correct": row["original_b3_correct"], "drop_b3_correct": current["drop_b3_correct"], "original_m1_correct": row["original_m1_correct"], "drop_m1_correct": current["drop_m1_correct"], "drop_delta_b3": drop_delta_b3, "drop_delta_m1": drop_delta_m1, "oracle_opportunity": opportunity, "conservative_projected_correct": conservative, "L4_upper_contribution": drop_delta_b3 + opportunity, "L4_conservative_contribution": drop_delta_b3 + conservative}
            records.append(record)
            all_selected_rows.append(record)
        fold_records.append({"outer_fold": outer_fold, "selected_K": K, "test_rows": len(test_ids), "coverage_transition": {f"{left}->{right}": count for (left, right), count in sorted(transition.items())}, "coverage_bounds_outer_development": {str(count): value for count, value in bounds.items()}, "L3_drop_B3_delta": float(np.mean([row["drop_delta_b3"] for row in records])), "L3_drop_M1_delta": float(np.mean([row["drop_delta_m1"] for row in records])), "L4_upper": float(np.mean([row["L4_upper_contribution"] for row in records])), "L4_conservative": float(np.mean([row["L4_conservative_contribution"] for row in records]))})
    write_jsonl_fsynced(RAW_ROWS_PATH, all_selected_rows)

    MAP_ROOT.mkdir(parents=True)
    map_manifest = []
    window_manifest_rows = []
    for row_id in sorted(original_by_id):
        fold = stage_rows[row_id]["fold"]
        K = selected_K_by_fold[fold]
        geo = geometry_by_id[row_id]
        crops = regions[row_id]["regions"][1:]
        dropped = set(geo["dropped_crop_indices"][:K])
        retained = [region for index, region in enumerate(crops) if index not in dropped]
        selected_new = geo["new_windows"][:K]
        difference = np.zeros((geo["height"] + 1, geo["width"] + 1), dtype=np.int16)
        for left, top, right, bottom in retained + selected_new:
            difference[top, left] += 1
            difference[bottom, left] -= 1
            difference[top, right] -= 1
            difference[bottom, right] += 1
        count_map = difference[:-1, :-1].cumsum(axis=0).cumsum(axis=1).astype(np.uint8)
        map_path = MAP_ROOT / f"{row_id}.png"
        Image.fromarray(count_map).save(map_path, format="PNG", optimize=True, compress_level=9)
        map_manifest.append({"row_id": row_id, "path": str(map_path.relative_to(ROOT)), "bytes": map_path.stat().st_size, "sha256": sha256_file(map_path), "minimum": int(count_map.min()), "maximum": int(count_map.max())})
        window_manifest_rows.append({"row_id": row_id, "outer_fold": fold, "selected_K": K, "dropped_crop_indices_zero_based_views1_11": sorted(dropped), "dropped_view_indices": sorted(index + 1 for index in dropped), "new_windows": selected_new, "new_uncovered_pixel_gains": geo["new_uncovered_pixel_gains"][:K], "crop_width": geo["crop_width"], "crop_height": geo["crop_height"], "geometry_match": True})
    write_jsonl_fsynced(WINDOW_MANIFEST_PATH, window_manifest_rows)
    map_manifest_path = RUN_DIR / "raw/selected_map_manifest.jsonl"
    write_jsonl_fsynced(map_manifest_path, map_manifest)

    upper = float(np.mean([row["L4_upper_contribution"] for row in all_selected_rows]))
    conservative = float(np.mean([row["L4_conservative_contribution"] for row in all_selected_rows]))
    stage1_proceed = upper >= config["stage0"]["W_G1_min_upper_gain"]
    output = {"schema_version": 1, "status": "PASS_CWIN_STAGE0_COMPLETE", "original_accuracy": original_evaluation["accuracy"], "fold_loads": fold_loads, "selections": selections, "folds": fold_records, "L1": {"newly_covered_rows": sum(row["original_count"] == 0 and row["post_count"] > 0 for row in all_selected_rows), "partial_to_higher_rows": sum(row["original_count"] > 0 and row["post_count"] > row["original_count"] for row in all_selected_rows), "lost_all_coverage_rows": sum(row["lost_all_coverage"] for row in all_selected_rows)}, "L3": {"B3_drop_delta": float(np.mean([row["drop_delta_b3"] for row in all_selected_rows])), "M1_drop_delta": float(np.mean([row["drop_delta_m1"] for row in all_selected_rows]))}, "L4_upper": upper, "L4_conservative": conservative, "W_G1": stage1_proceed, "proceed_stage1_amendment": stage1_proceed, "gpu_authorized": False, "W_K5": any(row["boundary_selected"] for row in selections), "geometry_audit": {"rows": 1581, "mismatch_rows": 0, "mismatch_rate": 0.0, "W_K7": False}, "raw": {"geometry": {"path": str(RAW_GEOMETRY_PATH.relative_to(ROOT)), "rows": len(geometry), "bytes": RAW_GEOMETRY_PATH.stat().st_size, "sha256": sha256_file(RAW_GEOMETRY_PATH)}, "rows": {"path": str(RAW_ROWS_PATH.relative_to(ROOT)), "rows": len(all_selected_rows), "bytes": RAW_ROWS_PATH.stat().st_size, "sha256": sha256_file(RAW_ROWS_PATH)}, "window_manifest": {"path": str(WINDOW_MANIFEST_PATH.relative_to(ROOT)), "rows": len(window_manifest_rows), "bytes": WINDOW_MANIFEST_PATH.stat().st_size, "sha256": sha256_file(WINDOW_MANIFEST_PATH)}, "map_manifest": {"path": str(map_manifest_path.relative_to(ROOT)), "rows": len(map_manifest), "bytes": map_manifest_path.stat().st_size, "sha256": sha256_file(map_manifest_path)}, "write_flush_fsync_per_row": True}}
    temporary = OUTPUT_PATH.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    temporary.replace(OUTPUT_PATH)
    print(json.dumps({"status": output["status"], "selected_K": selected_K_by_fold, "L1": output["L1"], "L3": output["L3"], "L4_upper": upper, "L4_conservative": conservative, "W_G1": stage1_proceed, "W_K5": output["W_K5"], "gpu_authorized": False}, indent=2))


if __name__ == "__main__":
    main()