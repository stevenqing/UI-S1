import hashlib
import math

import numpy as np


SOURCE_COMMIT = "2c1125067958df2468663004b2b4b7c50557da25"
POINT_BOX_PIXELS = 50
GATING_THRESHOLD = 1.5
SIGMA_SCALE = 2.5
MIN_CROP_PIXELS = 512


def deterministic_seed(row_id, cell, model, chain, slot, seed_base=20260802):
    payload = f"{row_id}|{cell}|{model}|{chain}|{slot}|{seed_base}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**31 - 1)


def point_to_box(point, width, height, size=POINT_BOX_PIXELS):
    if point is None or len(point) != 2 or not all(math.isfinite(float(value)) for value in point):
        return None
    half = size / 2
    x_value, y_value = map(float, point)
    return [
        max(0.0, x_value - half) / width,
        max(0.0, y_value - half) / height,
        min(float(width), x_value + half) / width,
        min(float(height), y_value + half) / height,
    ]


def box_iou(left, right):
    if left is None or right is None:
        return 0.0
    left_x1, left_x2 = sorted((left[0], left[2]))
    left_y1, left_y2 = sorted((left[1], left[3]))
    right_x1, right_x2 = sorted((right[0], right[2]))
    right_y1, right_y2 = sorted((right[1], right[3]))
    intersection = max(0.0, min(left_x2, right_x2) - max(left_x1, right_x1)) * max(
        0.0, min(left_y2, right_y2) - max(left_y1, right_y1)
    )
    union = (
        (left_x2 - left_x1) * (left_y2 - left_y1)
        + (right_x2 - right_x1) * (right_y2 - right_y1)
        - intersection
    )
    return intersection / union if union > 1e-6 else 0.0


def spatial_consistency(boxes):
    if len(boxes) <= 1:
        return 1.0
    values = [box_iou(left, right) for left_index, left in enumerate(boxes) for right_index, right in enumerate(boxes) if left_index != right_index]
    return sum(values) / len(values) if values else 0.0


def gate(candidates, threshold=GATING_THRESHOLD):
    valid = [candidate for candidate in candidates if candidate["box"] is not None]
    if not valid:
        return {"reliable": False, "spatial_consistency": 0.0, "mean_confidence": 0.0, "score": 0.0, "valid_candidates": 0}
    consistency = spatial_consistency([candidate["box"] for candidate in valid])
    confidence = sum(candidate["confidence"] for candidate in valid) / len(valid)
    score = consistency + confidence
    return {
        "reliable": score > threshold,
        "spatial_consistency": consistency,
        "mean_confidence": confidence,
        "score": score,
        "valid_candidates": len(valid),
    }


def adaptive_crop(candidates, width, height, sigma_scale=SIGMA_SCALE, minimum=MIN_CROP_PIXELS):
    valid = []
    for candidate in candidates:
        box = candidate.get("box")
        if box is None:
            continue
        x1, y1, x2, y2 = box
        x1, x2 = x1 * width, x2 * width
        y1, y2 = y1 * height, y2 * height
        valid.append({
            "center": [(x1 + x2) / 2, (y1 + y2) / 2],
            "size": [x2 - x1, y2 - y1],
        })
    if not valid:
        return None
    centers = np.asarray([item["center"] for item in valid], dtype=np.float64)
    median = np.median(centers, axis=0)
    distances = np.linalg.norm(centers - median, axis=1)
    keep = max(1, int(len(valid) * 0.75))
    indices = np.argsort(distances)[:keep]
    filtered_centers = centers[indices]
    filtered_sizes = np.asarray([valid[index]["size"] for index in indices], dtype=np.float64)
    total_variance = np.var(filtered_centers, axis=0) + np.mean(np.square(filtered_sizes / 4.0), axis=0)
    sigma = np.sqrt(total_variance)
    center = np.mean(filtered_centers, axis=0)
    final_size = max(2 * sigma_scale * sigma[0], 2 * sigma_scale * sigma[1], minimum)
    half = final_size / 2
    left, top, right, bottom = center[0] - half, center[1] - half, center[0] + half, center[1] + half
    if left < 0:
        right -= left
        left = 0
    if top < 0:
        bottom -= top
        top = 0
    if right > width:
        left -= right - width
        right = width
    if bottom > height:
        top -= bottom - height
        bottom = height
    return [
        max(0, int(left)),
        max(0, int(top)),
        min(width, int(right)),
        min(height, int(bottom)),
    ]