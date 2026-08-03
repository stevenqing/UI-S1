import bisect
import math
from dataclasses import dataclass


BINS = 8
MIN_CLASS = 32
BANDWIDTH_PIXELS = 14.0


def similarity(left, right):
    distance = math.dist(left, right)
    return math.exp(-(distance * distance) / (2 * BANDWIDTH_PIXELS**2))


@dataclass(frozen=True)
class CoordinateLR:
    boundaries: tuple[float, ...]
    log_ratios: tuple[float, ...]
    successes: int
    failures: int

    def score(self, value):
        return self.log_ratios[bisect.bisect_right(self.boundaries, value)]


def fit(rows):
    observations = []
    for points, candidate_success in rows:
        for candidate_index, candidate in enumerate(points):
            label = bool(candidate_success[candidate_index])
            for voter_index, voter in enumerate(points):
                if voter_index != candidate_index:
                    observations.append((similarity(candidate, voter), label))
    labels = [label for _, label in observations]
    successes = sum(labels)
    failures = len(labels) - successes
    if successes < MIN_CLASS or failures < MIN_CLASS:
        raise ValueError("coordinate CCM calibration lacks both classes")
    ordered = sorted(value for value, _ in observations)
    boundaries = tuple(
        ordered[math.ceil(len(ordered) * index / BINS) - 1]
        for index in range(1, BINS)
    )
    positive = [0] * BINS
    negative = [0] * BINS
    for value, label in observations:
        index = bisect.bisect_right(boundaries, value)
        (positive if label else negative)[index] += 1
    ratios = tuple(
        math.log((positive[index] + 1) / (successes + BINS))
        - math.log((negative[index] + 1) / (failures + BINS))
        for index in range(BINS)
    )
    return CoordinateLR(boundaries, ratios, successes, failures)


def score_candidates(calibration, points):
    return [
        sum(
            calibration.score(similarity(candidate, voter))
            for voter_index, voter in enumerate(points)
            if voter_index != candidate_index
        )
        for candidate_index, candidate in enumerate(points)
    ]


def select(calibration, points):
    scores = score_candidates(calibration, points)
    winner = max(range(len(points)), key=lambda index: (scores[index], -index))
    return winner, scores
