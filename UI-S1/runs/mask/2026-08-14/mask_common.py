import importlib.util
import math
import sys
from pathlib import Path

import numpy as np


RUN_DIR = Path(__file__).resolve().parent
ROOT = RUN_DIR.parents[2]
GRAN_DIR = ROOT / "runs/gran/2026-08-14"
CONSOLIDATE_DIR = ROOT / "runs/consolidate/2026-08-06"
SOURCEBIAS_DIR = ROOT / "runs/sourcebias/2026-08-03"
sys.path.insert(0, str(GRAN_DIR))
sys.path.insert(0, str(SOURCEBIAS_DIR))

from gran_common import GranCandidate, attach_reliability, partition
from sourcebias_common import b3_select_index, point_in_bbox


MODELS = ("GTA1-7B", "Qwen3-VL-8B-Instruct", "UI-TARS-7B-SFT")
CUNI_ACTIONS = tuple((model, view) for view in range(4) for model in MODELS)


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_rows():
    common = load_module(CONSOLIDATE_DIR / "common.py", "mask_consolidate_common")
    context = common.load_context()
    rows = {}
    for row_id in context["row_ids"]:
        metadata = context["metadata"][row_id]
        width, height = map(int, metadata["img_size"])
        diagonal = math.hypot(width, height)
        candidates = []
        gran_candidates = []
        for order, action in enumerate(CUNI_ACTIONS):
            value = context["bank"][action][row_id]
            point = tuple(float(coordinate) for coordinate in value["point"])
            source = f"{action[0]}/view{action[1]}"
            correct = bool(point_in_bbox(point, metadata["target_bbox"]))
            candidates.append({**value, "point": point, "source": source, "correct": correct})
            gran_candidates.append(GranCandidate(
                source=source,
                lineage=action[0],
                action="POINT",
                coordinate=(point[0] / diagonal, point[1] / diagonal),
                parameter="",
                parse_ok=True,
                order=order,
                correct=correct,
            ))
        fold = int(context["fold_for_group"][metadata["application"]])
        rows[row_id] = {
            "fold": fold,
            "application": str(metadata["application"]),
            "image_size": [width, height],
            "target_bbox": list(map(float, metadata["target_bbox"])),
            "candidates": tuple(candidates),
            "gran_candidates": tuple(gran_candidates),
        }
    if len(rows) != 1581 or any(len(row["candidates"]) != 12 for row in rows.values()):
        raise ValueError("MASK C-uni row coverage mismatch")
    return rows


def source_reliability(rows, row_ids):
    values = {candidate.source: [] for candidate in next(iter(rows.values()))["gran_candidates"]}
    for row_id in row_ids:
        for candidate in rows[row_id]["gran_candidates"]:
            values[candidate.source].append(float(candidate.correct))
    return {source: float(np.mean(correct)) for source, correct in values.items()}


def ranked_modes(candidates, reliability, tau):
    attached = attach_reliability(candidates, reliability)
    parsed, classes = partition(attached, "screenspot_pro", "finite", float(tau))
    ranked = sorted(classes, key=lambda members: (
        len(members),
        sum(parsed[index].reliability for index in members),
        max(parsed[index].reliability for index in members),
        -min(parsed[index].order for index in members),
    ), reverse=True)
    output = []
    for members in ranked:
        representative = min(members, key=lambda index: (
            -parsed[index].reliability,
            parsed[index].source,
            parsed[index].order,
        ))
        output.append({
            "members": [parsed[index].order for index in members],
            "correct": any(parsed[index].correct for index in members),
            "representative_order": parsed[representative].order,
            "representative_correct": parsed[representative].correct,
        })
    return output


def mode_center(candidates, members):
    points = np.asarray([candidates[index]["point"] for index in members], dtype=np.float64)
    center = points.mean(axis=0)
    return float(center[0]), float(center[1])


def informative_mask_pixels(width, height, center, radius):
    selected = []
    minimum_y = max(0, math.ceil(center[1] - radius - 0.5))
    maximum_y = min(height - 1, math.floor(center[1] + radius - 0.5))
    radius_squared = radius * radius
    for y_value in range(minimum_y, maximum_y + 1):
        vertical = y_value + 0.5 - center[1]
        horizontal = math.sqrt(max(0.0, radius_squared - vertical * vertical))
        minimum_x = max(0, math.ceil(center[0] - horizontal - 0.5))
        maximum_x = min(width - 1, math.floor(center[0] + horizontal - 0.5))
        if maximum_x >= minimum_x:
            selected.extend(y_value * width + x_value for x_value in range(minimum_x, maximum_x + 1))
    return np.asarray(selected, dtype=np.int64)


def nearest_pixels(width, height, center, count):
    if count <= 0 or count > width * height:
        raise ValueError("MASK invalid empty-mask area")
    radius = max(2, math.ceil(math.sqrt(count / math.pi)) + 3)
    while True:
        minimum_x = max(0, math.floor(center[0] - radius))
        maximum_x = min(width - 1, math.ceil(center[0] + radius))
        minimum_y = max(0, math.floor(center[1] - radius))
        maximum_y = min(height - 1, math.ceil(center[1] + radius))
        x_values = np.arange(minimum_x, maximum_x + 1, dtype=np.int64)
        y_values = np.arange(minimum_y, maximum_y + 1, dtype=np.int64)
        grid_x, grid_y = np.meshgrid(x_values, y_values)
        indices = (grid_y * width + grid_x).ravel()
        if len(indices) >= count:
            distances = (
                np.square(grid_x.ravel() + 0.5 - center[0])
                + np.square(grid_y.ravel() + 0.5 - center[1])
            )
            order = np.lexsort((indices, distances))[:count]
            selected = np.sort(indices[order])
            touches_box = any(
                value in {minimum_x, maximum_x}
                for value in (selected % width).tolist()
            ) or any(
                value in {minimum_y, maximum_y}
                for value in (selected // width).tolist()
            )
            if not touches_box or (
                minimum_x == 0 and maximum_x == width - 1
                and minimum_y == 0 and maximum_y == height - 1
            ):
                return selected
        radius *= 2


def empty_mask(width, height, information_count, m1_center, mode_centers):
    image_center = ((width - 1) / 2, (height - 1) / 2)
    radial_distance = math.dist(image_center, m1_center)
    mode_pixels = {
        min(height - 1, max(0, math.floor(center[1]))) * width
        + min(width - 1, max(0, math.floor(center[0])))
        for center in mode_centers
    }
    for angle in range(360):
        radians = math.radians(angle)
        center = (
            image_center[0] + radial_distance * math.cos(radians),
            image_center[1] + radial_distance * math.sin(radians),
        )
        if not (0 <= center[0] < width and 0 <= center[1] < height):
            continue
        pixels = nearest_pixels(width, height, center, information_count)
        if mode_pixels.isdisjoint(set(pixels.tolist())):
            return {"angle_degrees": angle, "center": center, "pixels": pixels}
    return None


def cohen_kappa(left, right):
    left = np.asarray(left, dtype=np.bool_)
    right = np.asarray(right, dtype=np.bool_)
    observed = float(np.mean(left == right))
    left_rate = float(np.mean(left))
    right_rate = float(np.mean(right))
    expected = left_rate * right_rate + (1 - left_rate) * (1 - right_rate)
    if math.isclose(expected, 1.0):
        return None
    return float((observed - expected) / (1 - expected))


def failure_matrix(rows, row_ids, indices):
    failures = np.asarray([
        [not rows[row_id]["candidates"][index]["correct"] for index in indices]
        for row_id in row_ids
    ], dtype=np.bool_)
    matrix = np.eye(len(indices), dtype=np.float64)
    undefined = 0
    for left in range(len(indices)):
        for right in range(left + 1, len(indices)):
            value = cohen_kappa(failures[:, left], failures[:, right])
            if value is None:
                value = 0.0
                undefined += 1
            matrix[left, right] = matrix[right, left] = value
    return matrix, undefined


def generalized_neff(matrix):
    count = len(matrix)
    denominator = float(np.ones(count) @ matrix @ np.ones(count))
    return None if denominator <= 0 else float(count * count / denominator)


def b3_correct(candidates, target_bbox):
    selected, _ = b3_select_index(candidates)
    return bool(point_in_bbox(candidates[selected]["point"], target_bbox))
