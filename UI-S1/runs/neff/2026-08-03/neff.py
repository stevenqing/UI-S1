import math

import numpy as np


SIGMA = 14.0


def cohen_kappa(left, right):
    left = np.asarray(left, dtype=np.bool_)
    right = np.asarray(right, dtype=np.bool_)
    observed = float(np.mean(left == right))
    left_rate = float(np.mean(left))
    right_rate = float(np.mean(right))
    expected = left_rate * right_rate + (1 - left_rate) * (1 - right_rate)
    if math.isclose(expected, 1.0):
        return None
    return (observed - expected) / (1 - expected)


def effective_sample_size(count, rho):
    denominator = 1 + (count - 1) * rho
    if denominator <= 0:
        raise ValueError(f"nonpositive N_eff denominator: K={count}, rho={rho}")
    return count / denominator


def estimate_pool(rows):
    if not rows:
        raise ValueError("empty N_eff pool")
    count = len(rows[0]["candidates"])
    if count < 2 or any(len(row["candidates"]) != count for row in rows):
        raise ValueError("N_eff pool candidate-count mismatch")
    failures = np.asarray([
        [not point_in_bbox(candidate["point"], row["target_bbox"]) for candidate in row["candidates"]]
        for row in rows
    ], dtype=np.bool_)
    kappa_values = []
    geom_values = []
    conditional_values = []
    for left in range(count):
        for right in range(left + 1, count):
            value = cohen_kappa(failures[:, left], failures[:, right])
            if value is not None:
                kappa_values.append(value)
            similarities = np.asarray([
                math.exp(-math.dist(row["candidates"][left]["point"], row["candidates"][right]["point"]) ** 2 / (2 * SIGMA ** 2))
                for row in rows
            ], dtype=np.float64)
            geom_values.extend(similarities.tolist())
            both_fail = failures[:, left] & failures[:, right]
            conditional_values.extend(similarities[both_fail].tolist())
    if not kappa_values or not geom_values or not conditional_values:
        raise ValueError("N_eff estimator lacks finite pair events")
    quality_values = []
    for row in rows:
        bbox = row["target_bbox"]
        for candidate in row["candidates"]:
            region = candidate["region"]
            quality_values.append(region[0] <= bbox[0] and region[1] <= bbox[1] and region[2] >= bbox[2] and region[3] >= bbox[3])
    rho = {
        "failure_kappa": float(np.mean(kappa_values)),
        "rho_geom": float(np.mean(geom_values)),
        "rho_cond": float(np.mean(conditional_values)),
    }
    return {
        "K": count,
        "rho": rho,
        "N_eff": {name: effective_sample_size(count, value) for name, value in rho.items()},
        "mean_proposal_full_bbox_containment": float(np.mean(quality_values)),
        "rows": len(rows),
        "finite_kappa_pairs": len(kappa_values),
        "kernel_pair_row_events": len(geom_values),
        "conditional_both_failure_events": len(conditional_values),
    }


def point_in_bbox(point, bbox):
    return bbox[0] <= point[0] <= bbox[2] and bbox[1] <= point[1] <= bbox[3]


def linear_fit(points, x_key):
    x = np.asarray([point[x_key] for point in points], dtype=np.float64)
    y = np.asarray([point["accuracy"] for point in points], dtype=np.float64)
    design = np.column_stack((np.ones(len(x)), x))
    coefficients = np.linalg.lstsq(design, y, rcond=None)[0]
    predicted = design @ coefficients
    residual = y - predicted
    sse = float(np.sum(residual ** 2))
    total = float(np.sum((y - y.mean()) ** 2))
    return {
        "observations": len(points),
        "intercept": float(coefficients[0]),
        "coefficient": float(coefficients[1]),
        "r_squared": float(1 - sse / total) if total > 0 else 0.0,
        "residual_sd": math.sqrt(sse / (len(points) - 2)),
        "sse": sse,
    }


def two_factor_fit(points, neff_key):
    x = np.asarray([[point[neff_key], point["quality"]] for point in points], dtype=np.float64)
    y = np.asarray([point["accuracy"] for point in points], dtype=np.float64)
    design = np.column_stack((np.ones(len(x)), x))
    coefficients = np.linalg.lstsq(design, y, rcond=None)[0]
    predicted = design @ coefficients
    residual = y - predicted
    sse = float(np.sum(residual ** 2))
    total = float(np.sum((y - y.mean()) ** 2))
    r_squared = float(1 - sse / total) if total > 0 else 0.0
    n, parameters = len(points), 3
    adjusted = 1 - (1 - r_squared) * (n - 1) / (n - parameters) if n > parameters else float("nan")
    return {
        "observations": n,
        "intercept": float(coefficients[0]),
        "coefficient_N_eff": float(coefficients[1]),
        "coefficient_quality": float(coefficients[2]),
        "r_squared": r_squared,
        "adjusted_r_squared": adjusted,
        "residual_sd": math.sqrt(sse / (n - parameters)),
        "sse": sse,
    }
