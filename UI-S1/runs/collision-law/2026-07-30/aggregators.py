import math
from collections import Counter
from dataclasses import dataclass
from typing import Callable

from pka import (
    AggregateResult,
    Prediction,
    coordinate_density_medoid,
    coordinate_density_mode,
    pair_kernel,
    pka_joint,
    pka_joint_leave_one_out,
    pka_joint_continuous,
    requires_coordinate,
    requires_string,
)
from scoring import token_f1


@dataclass(frozen=True)
class Aggregator:
    id: str
    function: Callable


def _parsed(predictions):
    return [(index, prediction) for index, prediction in enumerate(predictions) if prediction.parse_ok]


def best_single(predictions: list[Prediction], priority_sources: list[str]) -> AggregateResult:
    by_source = {prediction.source: (index, prediction) for index, prediction in enumerate(predictions)}
    for source in priority_sources:
        if source in by_source and by_source[source][1].parse_ok:
            index, prediction = by_source[source]
            return AggregateResult(prediction, index, (), 1)
    return AggregateResult(None, None, (), 0)


def _plurality_action(predictions: list[Prediction], priority_sources: list[str]) -> str | None:
    parsed = [prediction for prediction in predictions if prediction.parse_ok]
    if not parsed:
        return None
    counts = Counter(prediction.action for prediction in parsed)
    highest = max(counts.values())
    tied = {action for action, count in counts.items() if count == highest}
    for source in priority_sources:
        for prediction in parsed:
            if prediction.source == source and prediction.action in tied:
                return prediction.action
    return next(prediction.action for prediction in parsed if prediction.action in tied)


def _text_medoid(predictions: list[Prediction], action: str) -> str:
    candidates = [prediction.parameter for prediction in predictions if prediction.parse_ok and prediction.action == action]
    if not candidates:
        return ""
    scores = [sum(token_f1(value.lower(), other.lower()) for other in candidates) for value in candidates]
    return candidates[max(range(len(candidates)), key=lambda index: (scores[index], -index))]


def _geometric_median(points, weights=None, iterations=100):
    if not points:
        return None
    weights = [1.0] * len(points) if weights is None else list(weights)
    estimate = (
        sum(weight * point[0] for point, weight in zip(points, weights)) / sum(weights),
        sum(weight * point[1] for point, weight in zip(points, weights)) / sum(weights),
    )
    for _ in range(iterations):
        distances = [math.dist(point, estimate) for point in points]
        if any(distance < 1e-12 for distance in distances):
            return points[distances.index(min(distances))]
        adjusted = [weight / distance for weight, distance in zip(weights, distances)]
        updated = (
            sum(weight * point[0] for point, weight in zip(points, adjusted)) / sum(adjusted),
            sum(weight * point[1] for point, weight in zip(points, adjusted)) / sum(adjusted),
        )
        if math.dist(updated, estimate) < 1e-9:
            return updated
        estimate = updated
    return estimate


def plurality_then_median(
    bench: str,
    predictions: list[Prediction],
    priority_sources: list[str],
    source_weights: dict[str, float] | None = None,
) -> AggregateResult:
    action = _plurality_action(predictions, priority_sources)
    if action is None:
        return AggregateResult(None, None, (), 0)
    selected = [prediction for prediction in predictions if prediction.parse_ok and prediction.action == action]
    coordinate = None
    if requires_coordinate(bench, action):
        coordinate_predictions = [prediction for prediction in selected if prediction.coordinate is not None]
        points = [prediction.coordinate for prediction in coordinate_predictions]
        weights = [source_weights.get(prediction.source, 1.0) for prediction in coordinate_predictions] if source_weights else None
        coordinate = _geometric_median(points, weights)
    parameter = _text_medoid(selected, action) if requires_string(bench, action) else ""
    prediction = Prediction(
        action=action,
        x=coordinate[0] if coordinate else None,
        y=coordinate[1] if coordinate else None,
        parameter=parameter,
        source="plurality_median",
    )
    return AggregateResult(prediction, None, (), len(selected))


def plurality_then_density(
    bench: str,
    predictions: list[Prediction],
    priority_sources: list[str],
) -> AggregateResult:
    action = _plurality_action(predictions, priority_sources)
    if action is None:
        return AggregateResult(None, None, (), 0)
    selected = [prediction for prediction in predictions if prediction.parse_ok and prediction.action == action]
    coordinate = coordinate_density_medoid(bench, selected, action) if requires_coordinate(bench, action) else None
    parameter = _text_medoid(selected, action) if requires_string(bench, action) else ""
    prediction = Prediction(
        action=action,
        x=coordinate[0] if coordinate else None,
        y=coordinate[1] if coordinate else None,
        parameter=parameter,
        source="plurality_density",
    )
    return AggregateResult(prediction, None, (), len(selected))


def pka_medoid(bench: str, predictions: list[Prediction]) -> AggregateResult:
    return pka_joint(bench, predictions)


def pka_medoid_leave_one_out(bench: str, predictions: list[Prediction]) -> AggregateResult:
    return pka_joint_leave_one_out(bench, predictions)


def pka_continuous(bench: str, predictions: list[Prediction]) -> AggregateResult:
    return pka_joint_continuous(bench, predictions)


AGGREGATORS = {
    "A1_plurality_median": Aggregator("A1_plurality_median", plurality_then_median),
    "A2_plurality_density": Aggregator("A2_plurality_density", plurality_then_density),
    "A3_pka_joint": Aggregator("A3_pka_joint", pka_medoid),
    "A4_pka_continuous": Aggregator("A4_pka_continuous", pka_continuous),
}