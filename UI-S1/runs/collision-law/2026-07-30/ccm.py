import bisect
import math
from collections import Counter, defaultdict
from dataclasses import dataclass

from pka import AggregateResult, Prediction, pair_kernel, requires_coordinate, requires_string


PAIR_TYPES = ("same-model-diff-view", "same-family", "cross-family")
CANDIDATE_CLASSES = ("coordinate-bearing", "parameterless", "string-bearing")
BINS = 8
MIN_CLASS_OBSERVATIONS = 32
FAMILY_PREFIXES = (
    ("ui-r1-e", "ui-r1-e"),
    ("ui-agile", "ui-agile"),
    ("gui-r1", "gui-r1"),
    ("ui-tars", "ui-tars"),
    ("tongui", "tongui"),
    ("cogagent", "cogagent"),
)


def source_model(source: str) -> str:
    return source.split("/", 1)[0].split("__", 1)[0]


def source_view(source: str) -> str:
    if "/" in source:
        return source.split("/", 1)[1]
    if "__" in source:
        return source.split("__", 1)[1]
    return "full"


def source_family(source: str) -> str:
    model = source_model(source).lower()
    for prefix, family in FAMILY_PREFIXES:
        if model.startswith(prefix):
            return family
    return model


def pair_type(candidate: Prediction, voter: Prediction) -> str:
    candidate_model = source_model(candidate.source)
    voter_model = source_model(voter.source)
    if candidate_model == voter_model and source_view(candidate.source) != source_view(voter.source):
        return "same-model-diff-view"
    if source_family(candidate.source) == source_family(voter.source):
        return "same-family"
    return "cross-family"


def candidate_class(bench: str, prediction: Prediction) -> str:
    if requires_string(bench, prediction.action):
        return "string-bearing"
    if requires_coordinate(bench, prediction.action):
        return "coordinate-bearing"
    return "parameterless"


@dataclass(frozen=True)
class RankLikelihoodRatio:
    boundaries: tuple[float, ...]
    log_ratios: tuple[float, ...]
    successes: int
    failures: int

    @classmethod
    def fit(cls, observations, bins=BINS, minimum=MIN_CLASS_OBSERVATIONS):
        values = [float(value) for value, _ in observations]
        labels = [bool(label) for _, label in observations]
        successes = sum(labels)
        failures = len(labels) - successes
        if successes < minimum or failures < minimum:
            return None
        ordered = sorted(values)
        boundaries = tuple(ordered[math.ceil(len(ordered) * index / bins) - 1] for index in range(1, bins))
        success_counts = [0] * bins
        failure_counts = [0] * bins
        for value, label in zip(values, labels):
            bin_index = bisect.bisect_right(boundaries, value)
            (success_counts if label else failure_counts)[bin_index] += 1
        log_ratios = tuple(
            math.log((success_counts[index] + 1) / (successes + bins))
            - math.log((failure_counts[index] + 1) / (failures + bins))
            for index in range(bins)
        )
        return cls(boundaries, log_ratios, successes, failures)

    def score(self, value: float) -> float:
        return self.log_ratios[bisect.bisect_right(self.boundaries, value)]


@dataclass(frozen=True)
class CCMCalibration:
    bench: str
    source_priors: dict[str, float]
    global_table: RankLikelihoodRatio
    class_tables: dict[str, RankLikelihoodRatio]
    cell_tables: dict[tuple[str, str], RankLikelihoodRatio]
    table_report: dict


def table_to_dict(table: RankLikelihoodRatio) -> dict:
    return {
        "boundaries": list(table.boundaries),
        "log_ratios": list(table.log_ratios),
        "successes": table.successes,
        "failures": table.failures,
    }


def table_from_dict(value: dict) -> RankLikelihoodRatio:
    return RankLikelihoodRatio(
        tuple(value["boundaries"]), tuple(value["log_ratios"]),
        value["successes"], value["failures"],
    )


def calibration_to_dict(calibration: CCMCalibration) -> dict:
    return {
        "bench": calibration.bench,
        "source_priors": calibration.source_priors,
        "global_table": table_to_dict(calibration.global_table),
        "class_tables": {
            key: table_to_dict(table) for key, table in calibration.class_tables.items()
        },
        "cell_tables": {
            f"{pair_kind}|{classification}": table_to_dict(table)
            for (pair_kind, classification), table in calibration.cell_tables.items()
        },
        "table_report": calibration.table_report,
    }


def calibration_from_dict(value: dict) -> CCMCalibration:
    return CCMCalibration(
        value["bench"],
        value["source_priors"],
        table_from_dict(value["global_table"]),
        {key: table_from_dict(table) for key, table in value["class_tables"].items()},
        {
            tuple(key.split("|", 1)): table_from_dict(table)
            for key, table in value["cell_tables"].items()
        },
        value["table_report"],
    )


def fit_calibration(bench: str, rows, mode: str) -> CCMCalibration:
    if mode not in {"pooled", "nine"}:
        raise ValueError(mode)
    source_counts = defaultdict(Counter)
    global_observations = []
    class_observations = defaultdict(list)
    cell_observations = defaultdict(list)
    for predictions, candidate_successes in rows:
        parsed = [(index, prediction) for index, prediction in enumerate(predictions) if prediction.parse_ok]
        for prediction, success in zip(predictions, candidate_successes):
            source_counts[prediction.source]["rows"] += 1
            source_counts[prediction.source]["successes"] += int(success)
        for candidate_position, (candidate_index, candidate) in enumerate(parsed):
            success = candidate_successes[candidate_index]
            classification = candidate_class(bench, candidate)
            for voter_position, (_, voter) in enumerate(parsed):
                if candidate_position == voter_position:
                    continue
                observation = (pair_kernel(bench, candidate, voter), success)
                global_observations.append(observation)
                class_observations[classification].append(observation)
                cell_observations[(pair_type(candidate, voter), classification)].append(observation)
    global_table = RankLikelihoodRatio.fit(global_observations)
    if global_table is None:
        raise ValueError("pool-wide CCM table lacks both success classes")
    class_tables = {
        classification: table
        for classification, observations in class_observations.items()
        if (table := RankLikelihoodRatio.fit(observations)) is not None
    }
    cell_tables = {
        key: table
        for key, observations in cell_observations.items()
        if (table := RankLikelihoodRatio.fit(observations)) is not None
    } if mode == "nine" else {}
    priors = {
        source: (counts["successes"] + 1) / (counts["rows"] + 2)
        for source, counts in source_counts.items()
    }
    report = {
        "mode": mode,
        "bins": BINS,
        "minimum_successes_and_failures": MIN_CLASS_OBSERVATIONS,
        "global": {"successes": global_table.successes, "failures": global_table.failures},
        "classes": {
            classification: {"successes": table.successes, "failures": table.failures}
            for classification, table in class_tables.items()
        },
        "cells": {
            f"{kind}/{classification}": {"successes": table.successes, "failures": table.failures}
            for (kind, classification), table in cell_tables.items()
        },
    }
    return CCMCalibration(bench, priors, global_table, class_tables, cell_tables, report)


def _table_for(calibration: CCMCalibration, candidate: Prediction, voter: Prediction):
    classification = candidate_class(calibration.bench, candidate)
    key = (pair_type(candidate, voter), classification)
    if key in calibration.cell_tables:
        return calibration.cell_tables[key], "cell"
    if classification in calibration.class_tables:
        return calibration.class_tables[classification], "class"
    return calibration.global_table, "global"


def score_candidates(calibration: CCMCalibration, predictions, family_dedup=False):
    parsed = [(index, prediction) for index, prediction in enumerate(predictions) if prediction.parse_ok]
    scores = []
    backoff_counts = Counter()
    for candidate_position, (original_index, candidate) in enumerate(parsed):
        prior = calibration.source_priors[candidate.source]
        score = math.log(prior / (1 - prior))
        evidence = []
        by_family = defaultdict(list)
        for voter_position, (_, voter) in enumerate(parsed):
            if candidate_position == voter_position:
                continue
            table, level = _table_for(calibration, candidate, voter)
            value = table.score(pair_kernel(calibration.bench, candidate, voter))
            backoff_counts[level] += 1
            if family_dedup:
                by_family[source_family(voter.source)].append(value)
            else:
                evidence.append(value)
        if family_dedup:
            evidence = [sum(values) / len(values) for values in by_family.values()]
        scores.append((score + sum(evidence), original_index, candidate))
    return scores, dict(backoff_counts)


def collision_calibrated_mode(calibration: CCMCalibration, predictions, family_dedup=False) -> AggregateResult:
    predictions = list(predictions)
    scores, _ = score_candidates(calibration, predictions, family_dedup)
    if not scores:
        return AggregateResult(None, None, (), 0)
    winner = max(range(len(scores)), key=lambda index: (scores[index][0], -index))
    score, original_index, prediction = scores[winner]
    return AggregateResult(prediction, original_index, tuple(item[0] for item in scores), len(scores))