import bisect
import hashlib
import json
import os
from pathlib import Path

import numpy as np


RUN_DIR = Path(__file__).resolve().parent
ROOT = RUN_DIR.parents[2]
N_GRID = (4, 5, 6, 8, 11)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl_fsynced(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def atomic_text(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def contains_center(rectangle, target_bbox):
    center_x = (target_bbox[0] + target_bbox[2]) / 2
    center_y = (target_bbox[1] + target_bbox[3]) / 2
    return rectangle[0] <= center_x < rectangle[2] and rectangle[1] <= center_y < rectangle[3]


def eccentricity(rectangle, target_bbox):
    center_x = (target_bbox[0] + target_bbox[2]) / 2
    center_y = (target_bbox[1] + target_bbox[3]) / 2
    window_x = (rectangle[0] + rectangle[2]) / 2
    window_y = (rectangle[1] + rectangle[3]) / 2
    return float(np.hypot((center_x - window_x) / 644.0, (center_y - window_y) / 364.0))


def fit_curve(pairs, row_areas):
    if not pairs or not row_areas:
        raise ValueError("TILE empty curve fit")
    boundaries = [float(value) for value in np.quantile([row["eccentricity"] for row in pairs], np.arange(0.1, 1.0, 0.1), method="linear")]
    area_median = float(np.median(list(row_areas.values())))
    pooled_all = float(np.mean([row["correct"] for row in pairs]))
    output = {"boundaries": boundaries, "area_median": area_median, "scales": {}}
    for scale in ("small", "large"):
        current = [row for row in pairs if (row_areas[row["row_id"]] <= area_median) == (scale == "small")]
        scale_fallback = not current
        pooled_scale = float(np.mean([row["correct"] for row in current])) if current else pooled_all
        bins = []
        for index in range(10):
            selected = [row for row in current if bisect.bisect_right(boundaries, row["eccentricity"]) == index]
            bins.append({"bin": index, "pairs": len(selected), "correctness": float(np.mean([row["correct"] for row in selected])) if selected else pooled_scale, "fallback": "NONE" if selected else ("EMPTY_SCALE_FALLBACK" if scale_fallback else "EMPTY_BIN_FALLBACK")})
        output["scales"][scale] = {"pairs": len(current), "pooled_correctness": pooled_scale, "bins": bins}
    return output


def curve_probability(curve, target_area, value):
    scale = "small" if target_area <= curve["area_median"] else "large"
    index = bisect.bisect_right(curve["boundaries"], value)
    return curve["scales"][scale]["bins"][index]["correctness"], scale, index


def score_layout(rectangles, target_bbox, target_area, curve):
    values = []
    for index, rectangle in enumerate(rectangles):
        if contains_center(rectangle, target_bbox):
            value = eccentricity(rectangle, target_bbox)
            probability, scale, bin_index = curve_probability(curve, target_area, value)
            values.append({"tile_index": index, "eccentricity": value, "probability": probability, "scale": scale, "bin": bin_index})
    if not values:
        return {"covering_tiles": [], "minimum_e_tile": None, "minimum_e_probability": 0.0, "p_hat": 0.0}
    minimum = min(values, key=lambda row: (row["eccentricity"], row["tile_index"]))
    return {"covering_tiles": values, "minimum_e_tile": minimum["tile_index"], "minimum_e_probability": minimum["probability"], "p_hat": max(row["probability"] for row in values)}


def ledger_record(probability, correct):
    return {"expected_repair": (1 - int(correct)) * probability, "expected_damage": int(correct) * (1 - probability), "expected_net": probability - int(correct), "hard_below_0_5": probability < 0.5}


def select_n(scores):
    return max(sorted(scores), key=lambda value: (scores[value], -value))