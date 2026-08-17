import hashlib
import json
import math
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import yaml
from PIL import Image


RUN_DIR = Path(__file__).resolve().parent
ROOT = RUN_DIR.parents[2]
MASK_DIR = ROOT / "runs/mask/2026-08-14"
SOURCEBIAS_DIR = ROOT / "runs/sourcebias/2026-08-03"
CONFIG_PATH = RUN_DIR / "configs/cover_prereg.yaml"
FEASIBILITY_PATH = RUN_DIR / "ARM_B_FEASIBILITY.json"
REGION_PATH = ROOT / "runs/allocation-law/2026-08-01/raw/shared_regions_n12.jsonl"
OUTPUT_PATH = RUN_DIR / "ARM_A.json"
RAW_ROWS_PATH = RUN_DIR / "raw/arm_a_rows.jsonl"
RAW_MAP_MANIFEST_PATH = RUN_DIR / "raw/arm_a_map_manifest.jsonl"
MAP_ROOT = RUN_DIR / "raw/coverage_maps"

sys.path.insert(0, str(MASK_DIR))
sys.path.insert(0, str(SOURCEBIAS_DIR))
from mask_common import load_rows
from sourcebias_common import b3_select_index


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def coverage_map(width, height, regions):
    difference = np.zeros((height + 1, width + 1), dtype=np.int16)
    for left, top, right, bottom in regions:
        left = max(0, min(width, int(left)))
        right = max(0, min(width, int(right)))
        top = max(0, min(height, int(top)))
        bottom = max(0, min(height, int(bottom)))
        if right <= left or bottom <= top:
            raise ValueError("COVER invalid crop rectangle")
        difference[top, left] += 1
        difference[bottom, left] -= 1
        difference[top, right] -= 1
        difference[bottom, right] += 1
    values = difference[:-1, :-1].cumsum(axis=0).cumsum(axis=1)
    if values.min() < 0 or values.max() > len(regions):
        raise ValueError("COVER invalid coverage count")
    return values.astype(np.uint8)


def summarize(values):
    array = np.asarray(values, dtype=np.float64)
    return {"minimum": float(array.min()), "q1": float(np.quantile(array, 0.25, method="linear")), "median": float(np.quantile(array, 0.5, method="linear")), "mean": float(array.mean()), "q3": float(np.quantile(array, 0.75, method="linear")), "maximum": float(array.max())}


def target_stratum(count):
    if count == 11:
        return "common_11"
    if count == 0:
        return "uncovered_0"
    return "partial_1_10"


def row_class(candidates, selected_index):
    selected = bool(candidates[selected_index]["correct"])
    if selected:
        return "selected_correct"
    if any(candidate["correct"] for candidate in candidates):
        return "recoverable"
    return "zero_candidate_success_coverage"


def grouped_bootstrap(rows, resamples, seed):
    common = [row for row in rows if row["target_stratum"] == "common_11"]
    low = [row for row in rows if row["target_stratum"] != "common_11"]
    if not common or not low:
        return {"point_delta": None, "ci_99": None, "resamples": 0, "reason": "empty_stratum"}
    applications = sorted({row["application"] for row in rows})
    common_by_app = {app: [row["b3_correct"] for row in common if row["application"] == app] for app in applications}
    low_by_app = {app: [row["b3_correct"] for row in low if row["application"] == app] for app in applications}
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(resamples):
        sampled = rng.choice(applications, size=len(applications), replace=True)
        common_values = [value for app in sampled for value in common_by_app[app]]
        low_values = [value for app in sampled for value in low_by_app[app]]
        if common_values and low_values:
            values.append(float(np.mean(common_values) - np.mean(low_values)))
    if len(values) < 0.99 * resamples:
        raise ValueError("COVER Arm A insufficient finite bootstrap replicates")
    point = float(np.mean([row["b3_correct"] for row in common]) - np.mean([row["b3_correct"] for row in low]))
    return {"point_delta": point, "ci_99": [float(np.quantile(values, 0.005)), float(np.quantile(values, 0.995))], "resamples": len(values), "unit": "application_group"}


def write_jsonl_fsynced(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())


def main():
    if OUTPUT_PATH.exists() or RAW_ROWS_PATH.exists() or RAW_MAP_MANIFEST_PATH.exists() or MAP_ROOT.exists():
        raise FileExistsError("COVER Arm A output exists")
    config = yaml.safe_load(CONFIG_PATH.read_text())
    feasibility = json.loads(FEASIBILITY_PATH.read_text())
    if feasibility["status"] != "PASS_COVER_ARM_B_FEASIBILITY_AND_INPUT_LOCK" or feasibility["screenspot_regions"]["sha256"] != sha256_file(REGION_PATH):
        raise PermissionError("COVER Arm A input lock mismatch")
    regions = {row["id"]: row for row in read_jsonl(REGION_PATH)}
    rows = load_rows()
    if set(regions) != set(rows) or len(rows) != 1581:
        raise ValueError("COVER Arm A row identity mismatch")
    MAP_ROOT.mkdir(parents=True)
    row_records = []
    map_records = []
    for row_id in sorted(rows):
        region = regions[row_id]
        width, height = region["img_size"]
        if region["regions"][0] != [0, 0, width, height] or len(region["regions"]) != 12:
            raise ValueError(f"COVER full-image/crop anchor mismatch: {row_id}")
        values = coverage_map(width, height, region["regions"][1:])
        histogram = np.bincount(values.ravel(), minlength=12)
        if int(histogram.sum()) != width * height:
            raise ValueError(f"COVER map histogram mismatch: {row_id}")
        map_path = MAP_ROOT / f"{row_id}.png"
        image = Image.fromarray(values, mode="L")
        image.save(map_path, format="PNG", optimize=True, compress_level=9)
        center_x = (rows[row_id]["target_bbox"][0] + rows[row_id]["target_bbox"][2]) / 2
        center_y = (rows[row_id]["target_bbox"][1] + rows[row_id]["target_bbox"][3]) / 2
        pixel_x = max(0, min(width - 1, math.floor(center_x)))
        pixel_y = max(0, min(height - 1, math.floor(center_y)))
        count = int(values[pixel_y, pixel_x])
        selected_index, selected_group = b3_select_index(rows[row_id]["candidates"])
        record = {
            "row_id": row_id,
            "application": rows[row_id]["application"],
            "fold": rows[row_id]["fold"],
            "width": width,
            "height": height,
            "coverage_histogram": histogram.tolist(),
            "intersection_fraction": float(histogram[11] / (width * height)),
            "union_fraction": float((width * height - histogram[0]) / (width * height)),
            "uncovered_fraction": float(histogram[0] / (width * height)),
            "target_center_pixel": [pixel_x, pixel_y],
            "target_coverage_count": count,
            "target_stratum": target_stratum(count),
            "b3_selected_index": selected_index,
            "b3_group": list(selected_group),
            "b3_correct": bool(rows[row_id]["candidates"][selected_index]["correct"]),
            "row_class": row_class(rows[row_id]["candidates"], selected_index),
        }
        row_records.append(record)
        map_records.append({"row_id": row_id, "path": str(map_path.relative_to(ROOT)), "bytes": map_path.stat().st_size, "sha256": sha256_file(map_path), "width": width, "height": height, "dtype": "uint8", "minimum": int(values.min()), "maximum": int(values.max())})
    b3_accuracy = float(np.mean([row["b3_correct"] for row in row_records]))
    if abs(b3_accuracy - 0.6369386464263125) > 1e-15:
        raise ValueError("COVER Arm A B3 anchor mismatch")
    write_jsonl_fsynced(RAW_ROWS_PATH, row_records)
    write_jsonl_fsynced(RAW_MAP_MANIFEST_PATH, map_records)
    strata = {}
    for stratum in config["arm_a"]["target_strata"]:
        selected = [row for row in row_records if row["target_stratum"] == stratum]
        strata[stratum] = {"rows": len(selected), "fraction": len(selected) / len(row_records), "b3_accuracy": float(np.mean([row["b3_correct"] for row in selected])) if selected else None}
    low_fraction = strata["partial_1_10"]["fraction"] + strata["uncovered_0"]["fraction"]
    conditional = grouped_bootstrap(row_records, config["arm_a"]["bootstrap"]["resamples"], 20261001)
    cross = {stratum: Counter() for stratum in config["arm_a"]["target_strata"]}
    for row in row_records:
        cross[row["target_stratum"]][row["row_class"]] += 1
    gates = {
        "A_G1": strata["common_11"]["fraction"] >= config["arm_a"]["gates"]["A_G1_common_fraction"],
        "A_G2": low_fraction < config["arm_a"]["gates"]["A_G2_min_low_fraction"],
        "A_G3": bool(low_fraction >= config["arm_a"]["gates"]["A_G2_min_low_fraction"] and conditional["point_delta"] is not None and conditional["point_delta"] >= config["arm_a"]["gates"]["A_G3_min_accuracy_gap"]),
    }
    output = {
        "schema_version": 1,
        "status": "PASS_COVER_ARM_A_COMPLETE_AWAITING_HUMAN_DECISION",
        "evidence_status": "POST_SELECTION_DIAGNOSTIC",
        "geometry": {"crop_views": list(range(1, 12)), "full_image_view": 0, "lineage_regions_identical": True, "map_count": len(map_records), "map_bytes": sum(row["bytes"] for row in map_records)},
        "area": {"intersection_fraction": summarize([row["intersection_fraction"] for row in row_records]), "union_fraction": summarize([row["union_fraction"] for row in row_records]), "uncovered_fraction": summarize([row["uncovered_fraction"] for row in row_records]), "full_image_baseline_fraction": 1.0},
        "target_strata": strata,
        "low_coverage_fraction": low_fraction,
        "conditional_accuracy": conditional,
        "row_class_cross_table": {stratum: {name: cross[stratum][name] for name in config["arm_a"]["row_classes"]} for stratum in config["arm_a"]["target_strata"]},
        "b3_accuracy": b3_accuracy,
        "gates": gates,
        "followup_gpu_authorized": False,
        "raw": {"rows": {"path": str(RAW_ROWS_PATH.relative_to(ROOT)), "rows": len(row_records), "bytes": RAW_ROWS_PATH.stat().st_size, "sha256": sha256_file(RAW_ROWS_PATH)}, "map_manifest": {"path": str(RAW_MAP_MANIFEST_PATH.relative_to(ROOT)), "rows": len(map_records), "bytes": RAW_MAP_MANIFEST_PATH.stat().st_size, "sha256": sha256_file(RAW_MAP_MANIFEST_PATH)}, "write_flush_fsync_per_row": True},
    }
    OUTPUT_PATH.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": output["status"], "strata": strata, "conditional": conditional, "gates": gates, "map_bytes": output["geometry"]["map_bytes"]}, indent=2))


if __name__ == "__main__":
    main()