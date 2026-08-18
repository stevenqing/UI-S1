import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np


RUN_DIR = Path(__file__).resolve().parent
ROOT = RUN_DIR.parents[2]
ASPECT_RATIO = 1288 / 728
TAU_BY_FOLD = (0.0022908676527677724, 0.0015135612484362087, 0.012022644346174132, 0.0034673685045253167, 0.0034673685045253167)


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


def confrontation_window(width, height, centroids):
    if len(centroids) < 2:
        raise ValueError("LOOK confrontation requires at least two centroids")
    x_values = [float(point[0]) for point in centroids]
    y_values = [float(point[1]) for point in centroids]
    minimum_x, maximum_x = min(x_values), max(x_values)
    minimum_y, maximum_y = min(y_values), max(y_values)
    delta_x = maximum_x - minimum_x
    delta_y = maximum_y - minimum_y
    short_edge = max(1.0, min(max(delta_x, 1.0), max(delta_y, 1.0)))
    padding = 0.25 * short_edge
    left = minimum_x - padding
    right = maximum_x + padding
    top = minimum_y - padding
    bottom = maximum_y + padding
    padded = [left, top, right, bottom]
    current_width = right - left
    current_height = bottom - top
    if current_width / current_height < ASPECT_RATIO:
        target_width = current_height * ASPECT_RATIO
        expansion = (target_width - current_width) / 2
        left -= expansion
        right += expansion
    else:
        target_height = current_width / ASPECT_RATIO
        expansion = (target_height - current_height) / 2
        top -= expansion
        bottom += expansion
    aspect_adjusted = [left, top, right, bottom]
    integer = [math.floor(left), math.floor(top), math.ceil(right), math.ceil(bottom)]
    window_width = integer[2] - integer[0]
    window_height = integer[3] - integer[1]
    if window_width > width or window_height > height:
        return {"status": "INFEASIBLE_TOO_LARGE", "centroids": [list(point) for point in centroids], "short_edge": short_edge, "padding": padding, "padded_bounds": padded, "aspect_bounds": aspect_adjusted, "integer_bounds": integer, "dimensions": [window_width, window_height]}
    final_left = min(max(integer[0], 0), width - window_width)
    final_top = min(max(integer[1], 0), height - window_height)
    final = [final_left, final_top, final_left + window_width, final_top + window_height]
    contains = [final[0] <= point[0] < final[2] and final[1] <= point[1] < final[3] for point in centroids]
    if not all(contains):
        raise ValueError("LOOK translated window lost a requested centroid")
    return {"status": "FEASIBLE", "centroids": [list(point) for point in centroids], "short_edge": short_edge, "padding": padding, "padded_bounds": padded, "aspect_bounds": aspect_adjusted, "integer_bounds": integer, "translation": [final_left - integer[0], final_top - integer[1]], "final_window": final, "dimensions": [window_width, window_height], "area": window_width * window_height, "area_fraction": window_width * window_height / (width * height), "centroids_contained": contains}


def null_seed(row_id, attempt):
    payload = f"LOOK|20260818|NULL|{row_id}|{attempt}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big", signed=False)


def find_null_window(row_id, width, height, m1_centroid, candidate_points, main_area, attempts=10000):
    failure_counts = {"candidate_proximity": 0, "geometry": 0, "area_ratio": 0}
    for attempt in range(attempts):
        seed = null_seed(row_id, attempt)
        generator = np.random.Generator(np.random.PCG64(seed))
        point = [int(generator.integers(0, width)), int(generator.integers(0, height))]
        if any(math.dist(point, candidate) <= 14 for candidate in candidate_points):
            failure_counts["candidate_proximity"] += 1
            continue
        window = confrontation_window(width, height, [m1_centroid, point])
        if window["status"] != "FEASIBLE":
            failure_counts["geometry"] += 1
            continue
        ratio = window["area"] / main_area
        if not 0.9 <= ratio <= 1.1:
            failure_counts["area_ratio"] += 1
            continue
        return {"status": "FEASIBLE", "attempt": attempt, "seed": seed, "random_point": point, "minimum_candidate_distance": min(math.dist(point, candidate) for candidate in candidate_points), "area_ratio": ratio, "failure_counts_before_selection": failure_counts, "window": window}
    return {"status": "INFEASIBLE_NO_MATCHED_NULL", "attempts": attempts, "failure_counts": failure_counts}


def allocate_counts(populations, target):
    applications = sorted(populations)
    if not applications:
        return {}
    if sum(populations.values()) <= target:
        return dict(populations)
    if target < len(applications):
        raise ValueError("LOOK target smaller than nonempty applications")
    allocations = {application: 1 for application in applications}
    capacities = {application: populations[application] - 1 for application in applications}
    remaining = target - len(applications)
    while remaining:
        total_capacity = sum(capacities.values())
        quotas = {application: remaining * capacities[application] / total_capacity for application in applications}
        floors = {application: min(capacities[application], int(quotas[application])) for application in applications}
        assigned = sum(floors.values())
        for application in applications:
            allocations[application] += floors[application]
            capacities[application] -= floors[application]
        remaining -= assigned
        if not remaining:
            break
        for application in sorted(applications, key=lambda value: (-(quotas[value] - floors[value]), value)):
            if not remaining:
                break
            if capacities[application]:
                allocations[application] += 1
                capacities[application] -= 1
                remaining -= 1
    if sum(allocations.values()) != target:
        raise ValueError("LOOK allocation mismatch")
    return allocations


def sample_hash(stratum, application, row_id):
    return hashlib.sha256(f"LOOK|20260818|SAMPLE|{stratum}|{application}|{row_id}".encode()).hexdigest()


def nearest_choice(point, centroids):
    if point is None or len(point) != 2 or not all(math.isfinite(float(value)) for value in point):
        return None
    return min(range(len(centroids)), key=lambda index: (math.dist(point, centroids[index]), index))