from collections import Counter

from pka import Prediction, requires_coordinate, requires_string
from scoring import token_f1
from reguide_port import kde_candidate_peak


def self_consistency_product_space(predictions: list[Prediction]) -> Prediction | None:
    parsed = [prediction for prediction in predictions if prediction.parse_ok]
    if not parsed:
        return None
    counts = Counter(prediction.action for prediction in parsed)
    highest = max(counts.values())
    action = next(
        prediction.action for prediction in parsed
        if counts[prediction.action] == highest
    )
    selected = [prediction for prediction in parsed if prediction.action == action]
    coordinates = [prediction.coordinate for prediction in selected if prediction.coordinate is not None]
    coordinate = kde_candidate_peak(coordinates).coordinate if coordinates else None
    parameter = ""
    if selected and any(prediction.parameter for prediction in selected):
        values = [prediction.parameter for prediction in selected]
        scores = [sum(token_f1(value.lower(), other.lower()) for other in values) for value in values]
        parameter = values[max(range(len(values)), key=lambda index: (scores[index], -index))]
    return Prediction(
        action=action,
        x=coordinate[0] if coordinate else None,
        y=coordinate[1] if coordinate else None,
        parameter=parameter,
        source="self_consistency",
    )