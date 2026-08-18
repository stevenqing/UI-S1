import hashlib
import json
import math
import os
from fractions import Fraction
from pathlib import Path

import numpy as np


RUN_DIR = Path(__file__).resolve().parent
ROOT = RUN_DIR.parents[2]
WINDOW_WIDTH = 1288
WINDOW_HEIGHT = 728
R_GRID = (109.2, 150, 200, 250, 300, 340)


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


def round_half_away(value):
    return math.floor(value + 0.5) if value >= 0 else math.ceil(value - 0.5)


def round_fraction_half_up(value):
    quotient, remainder = divmod(value.numerator, value.denominator)
    return quotient + int(2 * remainder >= value.denominator)


def jitter_offsets(radius):
    return [
        [
            round_half_away(radius * index / 10 * math.cos(2 * math.pi * index / 11)),
            round_half_away(radius * index / 10 * math.sin(2 * math.pi * index / 11)),
        ]
        for index in range(11)
    ]


def nearest_in_interval(value, lower, upper):
    if lower > upper:
        raise ValueError("OWIN empty feasible interval")
    return min(max(value, lower), upper)


def oracle_window(width, height, target_bbox, offset):
    if width < WINDOW_WIDTH or height < WINDOW_HEIGHT:
        raise ValueError("OWIN image smaller than crop")
    center_x = (target_bbox[0] + target_bbox[2]) / 2
    center_y = (target_bbox[1] + target_bbox[3]) / 2
    initial_left = math.floor(center_x + offset[0] - WINDOW_WIDTH / 2)
    initial_top = math.floor(center_y + offset[1] - WINDOW_HEIGHT / 2)
    left_lower = max(0, math.floor(center_x - WINDOW_WIDTH) + 1)
    left_upper = min(width - WINDOW_WIDTH, math.floor(center_x))
    top_lower = max(0, math.floor(center_y - WINDOW_HEIGHT) + 1)
    top_upper = min(height - WINDOW_HEIGHT, math.floor(center_y))
    left = nearest_in_interval(initial_left, left_lower, left_upper)
    top = nearest_in_interval(initial_top, top_lower, top_upper)
    window = [left, top, left + WINDOW_WIDTH, top + WINDOW_HEIGHT]
    return {
        "requested_offset": list(offset),
        "initial_window": [initial_left, initial_top, initial_left + WINDOW_WIDTH, initial_top + WINDOW_HEIGHT],
        "final_window": window,
        "translation": [left - initial_left, top - initial_top],
        "target_center_contained": contains_center(window, target_bbox),
        "target_bbox_contained": contains_bbox(window, target_bbox),
    }


def rectangle_iou(left, right):
    intersection = rectangle_intersection_area(left, right)
    left_area = (left[2] - left[0]) * (left[3] - left[1])
    right_area = (right[2] - right[0]) * (right[3] - right[1])
    return intersection / (left_area + right_area - intersection)


def rectangle_intersection_area(left, right):
    return max(0, min(left[2], right[2]) - max(left[0], right[0])) * max(
        0, min(left[3], right[3]) - max(left[1], right[1])
    )


def median_pairwise_iou(rectangles):
    values = [
        rectangle_iou(rectangles[left], rectangles[right])
        for left in range(len(rectangles))
        for right in range(left + 1, len(rectangles))
    ]
    return float(np.median(values))


def contains_center(rectangle, target_bbox):
    center_x = (target_bbox[0] + target_bbox[2]) / 2
    center_y = (target_bbox[1] + target_bbox[3]) / 2
    return rectangle[0] <= center_x < rectangle[2] and rectangle[1] <= center_y < rectangle[3]


def contains_bbox(rectangle, target_bbox):
    return (
        rectangle[0] <= target_bbox[0]
        and rectangle[1] <= target_bbox[1]
        and target_bbox[2] <= rectangle[2]
        and target_bbox[3] <= rectangle[3]
    )


def positive_bbox_intersection(rectangle, target_bbox):
    return rectangle_intersection_area(rectangle, target_bbox) > 0


def uniform_anchors(extent, count):
    if count == 1:
        return [round_fraction_half_up(Fraction(extent, 2))]
    return [round_fraction_half_up(Fraction(index * extent, count - 1)) for index in range(count)]


def union_area(rectangles):
    x_values = sorted({rectangle[0] for rectangle in rectangles} | {rectangle[2] for rectangle in rectangles})
    area = 0
    for left, right in zip(x_values, x_values[1:]):
        intervals = sorted(
            (rectangle[1], rectangle[3])
            for rectangle in rectangles
            if rectangle[0] < right and rectangle[2] > left
        )
        covered = 0
        if intervals:
            start, end = intervals[0]
            for current_start, current_end in intervals[1:]:
                if current_start > end:
                    covered += end - start
                    start, end = current_start, current_end
                else:
                    end = max(end, current_end)
            covered += end - start
        area += (right - left) * covered
    return area


def tiling_layout(width, height, window_count):
    if width < WINDOW_WIDTH or height < WINDOW_HEIGHT:
        raise ValueError("OWIN image smaller than crop")
    candidates = []
    for row_count in range(1, window_count + 1):
        base, remainder = divmod(window_count, row_count)
        counts = [base] * row_count
        center_order = sorted(range(row_count), key=lambda index: (abs(index - (row_count - 1) / 2), index))
        for index in center_order[:remainder]:
            counts[index] += 1
        tops = uniform_anchors(height - WINDOW_HEIGHT, row_count)
        rectangles = []
        for top, count in zip(tops, counts):
            for left in uniform_anchors(width - WINDOW_WIDTH, count):
                rectangles.append([left, top, left + WINDOW_WIDTH, top + WINDOW_HEIGHT])
        rectangles.sort(key=lambda rectangle: (rectangle[1], rectangle[0]))
        pairwise = [
            rectangle_intersection_area(rectangles[left], rectangles[right])
            for left in range(window_count)
            for right in range(left + 1, window_count)
        ]
        candidates.append(
            {
                "row_count": row_count,
                "rectangles": rectangles,
                "union_area": union_area(rectangles),
                "sum_pairwise_overlap": sum(pairwise),
                "maximum_pairwise_overlap": max(pairwise, default=0),
            }
        )
    return min(
        candidates,
        key=lambda value: (
            -value["union_area"],
            value["sum_pairwise_overlap"],
            value["maximum_pairwise_overlap"],
            value["row_count"],
            value["rectangles"],
        ),
    )


def summarize(values):
    array = np.asarray(values, dtype=np.float64)
    return {
        "minimum": float(array.min()),
        "q1": float(np.quantile(array, 0.25, method="linear")),
        "median": float(np.quantile(array, 0.5, method="linear")),
        "mean": float(array.mean()),
        "q3": float(np.quantile(array, 0.75, method="linear")),
        "maximum": float(array.max()),
    }