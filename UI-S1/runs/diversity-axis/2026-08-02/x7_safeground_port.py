import math

import numpy as np


OFFICIAL_COMMIT = "5e8fca7ef091bc6751cad9703ca430e775aa4433"
DEFAULT_WEIGHTS = {"margin": 0.2, "entropy": 0.2, "concentration": 0.6}


def region_scores(points, width, height, patch_size=28, activation_threshold=0.0):
    n_width = width // patch_size
    n_height = height // patch_size
    if n_width < 1 or n_height < 1:
        raise ValueError("SafeGround patch grid must be nonempty")
    heatmap = np.zeros((n_height, n_width), dtype=np.float64)
    for point in points:
        if point is None or len(point) != 2 or not all(math.isfinite(float(value)) for value in point):
            continue
        x_value, y_value = map(float, point)
        patch_x = min(max(int((x_value / width) * n_width), 0), n_width - 1)
        patch_y = min(max(int((y_value / height) * n_height), 0), n_height - 1)
        heatmap[patch_y, patch_x] += 1
    total = heatmap.sum()
    if total == 0:
        return []
    probability = heatmap / total
    mask = probability > probability.max() * activation_threshold
    components = []
    visited = set()
    for y_value in range(n_height):
        for x_value in range(n_width):
            if not mask[y_value, x_value] or (y_value, x_value) in visited:
                continue
            component = []
            queue = [(y_value, x_value)]
            visited.add((y_value, x_value))
            while queue:
                current_y, current_x = queue.pop(0)
                component.append((current_y, current_x))
                for delta_y, delta_x in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    next_y, next_x = current_y + delta_y, current_x + delta_x
                    if (
                        0 <= next_y < n_height
                        and 0 <= next_x < n_width
                        and mask[next_y, next_x]
                        and (next_y, next_x) not in visited
                    ):
                        visited.add((next_y, next_x))
                        queue.append((next_y, next_x))
            components.append(component)
    scores = [float(np.mean([probability[y_value, x_value] for y_value, x_value in component])) for component in components]
    return sorted(scores, reverse=True)


def uncertainty_from_scores(scores):
    if not scores:
        margin = entropy = concentration = 1.0
    elif len(scores) == 1:
        margin = max(0.1, 1.0 - scores[0])
        entropy = 0.5
        concentration = 0.1
    else:
        margin = 1.0 - (scores[0] - scores[1]) / (scores[0] + 1e-8)
        values = np.asarray(scores, dtype=np.float64)
        probabilities = values / (values.sum() + 1e-8)
        entropy = float(-np.sum(probabilities * np.log(probabilities + 1e-10)) / np.log(len(scores)))
        concentration = float(1.0 - np.sum(probabilities**2))
    margin = min(max(float(margin), 0.0), 1.0)
    entropy = min(max(float(entropy), 0.0), 1.0)
    concentration = min(max(float(concentration), 0.0), 1.0)
    combined = (
        DEFAULT_WEIGHTS["margin"] * margin
        + DEFAULT_WEIGHTS["entropy"] * entropy
        + DEFAULT_WEIGHTS["concentration"] * concentration
    )
    return {
        "margin": margin,
        "entropy": entropy,
        "concentration": concentration,
        "combined": combined,
        "regions": len(scores),
        "region_scores": scores,
    }


def compute_uncertainty(points, width, height, patch_size=28, activation_threshold=0.0):
    return uncertainty_from_scores(region_scores(points, width, height, patch_size, activation_threshold))