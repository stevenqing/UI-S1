import math
from dataclasses import dataclass
from typing import Iterable

from kernels import (
    android_coord_kernel_normalized,
    mind2web_coord_inference,
    string_kernel,
    type_kernel,
)
from scoring import GROUNDING_ACTIONS, SIMPLE_ACTIONS, TEXT_ACTIONS


@dataclass(frozen=True)
class Prediction:
    action: str
    x: float | None = None
    y: float | None = None
    parameter: str = ""
    source: str = ""
    parse_ok: bool = True

    @property
    def coordinate(self) -> tuple[float, float] | None:
        if self.x is None or self.y is None:
            return None
        if math.isnan(self.x) or math.isnan(self.y):
            return None
        return self.x, self.y


@dataclass(frozen=True)
class AggregateResult:
    prediction: Prediction | None
    candidate_index: int | None
    candidate_scores: tuple[float, ...]
    parsed_inputs: int


def requires_coordinate(bench: str, action: str) -> bool:
    if bench == "androidcontrol":
        return action in GROUNDING_ACTIONS
    if bench == "mind2web":
        return action in {"CLICK", "SELECT", "TYPE"}
    raise ValueError(f"unknown benchmark: {bench}")


def requires_string(bench: str, action: str) -> bool:
    if bench == "androidcontrol":
        return action in TEXT_ACTIONS
    if bench == "mind2web":
        return action in {"SELECT", "TYPE"}
    raise ValueError(f"unknown benchmark: {bench}")


def pair_kernel(bench: str, left: Prediction, right: Prediction) -> float:
    score = type_kernel(left.action, right.action)
    if score == 0.0:
        return 0.0
    if requires_coordinate(bench, left.action):
        left_coordinate = left.coordinate
        right_coordinate = right.coordinate
        if left_coordinate is None or right_coordinate is None:
            return 0.0
        if bench == "androidcontrol":
            score *= android_coord_kernel_normalized(left_coordinate, right_coordinate)
        else:
            score *= mind2web_coord_inference(left_coordinate, right_coordinate)
    if requires_string(bench, left.action):
        score *= string_kernel(left.parameter, right.parameter)
    return score


def _validated_inputs(predictions: Iterable[Prediction], weights=None):
    original = list(predictions)
    if weights is None:
        original_weights = [1.0] * len(original)
    else:
        original_weights = list(weights)
        if len(original_weights) != len(original):
            raise ValueError("weights and predictions differ in length")
    parsed = []
    parsed_weights = []
    original_indices = []
    for index, (prediction, weight) in enumerate(zip(original, original_weights)):
        if not prediction.parse_ok:
            continue
        if weight < 0 or not math.isfinite(weight):
            raise ValueError(f"invalid PKA weight: {weight}")
        parsed.append(prediction)
        parsed_weights.append(weight)
        original_indices.append(index)
    if parsed and sum(parsed_weights) <= 0:
        raise ValueError("PKA weights sum to zero")
    return original, parsed, parsed_weights, original_indices


def pka_joint(bench: str, predictions: Iterable[Prediction], weights=None) -> AggregateResult:
    original, parsed, parsed_weights, original_indices = _validated_inputs(predictions, weights)
    if not parsed:
        return AggregateResult(None, None, (), 0)
    scores = tuple(
        sum(weight * pair_kernel(bench, voter, candidate) for voter, weight in zip(parsed, parsed_weights))
        for candidate in parsed
    )
    winner = max(range(len(parsed)), key=lambda index: (scores[index], -index))
    return AggregateResult(parsed[winner], original_indices[winner], scores, len(parsed))


def pka_joint_leave_one_out(bench: str, predictions: Iterable[Prediction], weights=None) -> AggregateResult:
    original, parsed, parsed_weights, original_indices = _validated_inputs(predictions, weights)
    if not parsed:
        return AggregateResult(None, None, (), 0)
    scores = tuple(
        sum(
            weight * pair_kernel(bench, voter, candidate)
            for voter_index, (voter, weight) in enumerate(zip(parsed, parsed_weights))
            if voter_index != candidate_index
        )
        for candidate_index, candidate in enumerate(parsed)
    )
    winner = max(range(len(parsed)), key=lambda index: (scores[index], -index))
    return AggregateResult(parsed[winner], original_indices[winner], scores, len(parsed))


def coordinate_density_mode(
    bench: str,
    predictions: Iterable[Prediction],
    action: str,
    weights=None,
    iterations: int = 8,
) -> tuple[float, float] | None:
    if iterations != 8:
        raise ValueError("main continuous-mode iteration count is fixed at 8")
    original, parsed, parsed_weights, _ = _validated_inputs(predictions, weights)
    selected = [
        (prediction.coordinate, weight)
        for prediction, weight in zip(parsed, parsed_weights)
        if prediction.action == action and prediction.coordinate is not None
    ]
    if not selected:
        return None
    points = [point for point, _ in selected]
    point_weights = [weight for _, weight in selected]
    candidates = []
    for seed in points:
        estimate = seed
        for _ in range(iterations):
            local_weights = []
            for point, weight in zip(points, point_weights):
                kernel = (
                    android_coord_kernel_normalized(point, estimate)
                    if bench == "androidcontrol"
                    else mind2web_coord_inference(point, estimate)
                )
                local_weights.append(weight * kernel)
            total = sum(local_weights)
            if total <= 0:
                break
            updated = (
                sum(weight * point[0] for point, weight in zip(points, local_weights)) / total,
                sum(weight * point[1] for point, weight in zip(points, local_weights)) / total,
            )
            if math.dist(updated, estimate) < 1e-9:
                estimate = updated
                break
            estimate = updated
        density = sum(
            weight * (
                android_coord_kernel_normalized(point, estimate)
                if bench == "androidcontrol"
                else mind2web_coord_inference(point, estimate)
            )
            for point, weight in zip(points, point_weights)
        )
        candidates.append((density, estimate))
    return max(candidates, key=lambda item: item[0])[1]


def coordinate_density_medoid(
    bench: str,
    predictions: Iterable[Prediction],
    action: str,
    weights=None,
) -> tuple[float, float] | None:
    original, parsed, parsed_weights, _ = _validated_inputs(predictions, weights)
    selected = [
        (prediction.coordinate, weight)
        for prediction, weight in zip(parsed, parsed_weights)
        if prediction.action == action and prediction.coordinate is not None
    ]
    if not selected:
        return None
    points = [point for point, _ in selected]
    point_weights = [weight for _, weight in selected]
    scores = []
    for candidate in points:
        scores.append(sum(
            weight * (
                android_coord_kernel_normalized(point, candidate)
                if bench == "androidcontrol"
                else mind2web_coord_inference(point, candidate)
            )
            for point, weight in zip(points, point_weights)
        ))
    winner = max(range(len(points)), key=lambda index: (scores[index], -index))
    return points[winner]


def pka_joint_continuous(bench: str, predictions: Iterable[Prediction], weights=None) -> AggregateResult:
    predictions = list(predictions)
    result = pka_joint(bench, predictions, weights)
    if result.prediction is None or not requires_coordinate(bench, result.prediction.action):
        return result
    coordinate = coordinate_density_mode(bench, predictions, result.prediction.action, weights)
    if coordinate is None:
        return result
    updated = Prediction(
        action=result.prediction.action,
        x=coordinate[0],
        y=coordinate[1],
        parameter=result.prediction.parameter,
        source="pka_continuous",
    )
    return AggregateResult(updated, result.candidate_index, result.candidate_scores, result.parsed_inputs)