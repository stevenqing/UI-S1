import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class KDEResult:
    coordinate: tuple[float, float] | None
    candidate_index: int | None
    bandwidth: float | None
    scores: tuple[float, ...]


def scott_bandwidth(points: list[tuple[float, float]]) -> float:
    if len(points) <= 1:
        return 1e-6
    values = np.asarray(points, dtype=np.float64)
    scale = float(np.sqrt(np.mean(np.var(values, axis=0, ddof=1))))
    factor = len(points) ** (-1 / 6)
    return max(scale * factor, 1e-6)


def kde_candidate_peak(points: list[tuple[float, float]], weights=None) -> KDEResult:
    if not points:
        return KDEResult(None, None, None, ())
    weights = [1.0] * len(points) if weights is None else list(weights)
    if len(weights) != len(points) or sum(weights) <= 0:
        raise ValueError("invalid KDE weights")
    bandwidth = scott_bandwidth(points)
    scores = []
    for candidate in points:
        score = sum(
            weight * math.exp(-(math.dist(point, candidate) ** 2) / (2 * bandwidth**2))
            for point, weight in zip(points, weights)
        )
        scores.append(score)
    winner = max(range(len(points)), key=lambda index: (scores[index], -index))
    return KDEResult(points[winner], winner, bandwidth, tuple(scores))


def reguide_two_stage(
    first_stage_points: list[tuple[float, float]],
    roi_stage_points: list[tuple[float, float]],
) -> dict:
    first = kde_candidate_peak(first_stage_points)
    if first.coordinate is None:
        return {"roi_center": None, "prediction": None, "first_stage": first, "second_stage": None}
    second = kde_candidate_peak(roi_stage_points)
    return {
        "roi_center": first.coordinate,
        "prediction": second.coordinate if second.coordinate is not None else first.coordinate,
        "first_stage": first,
        "second_stage": second,
    }