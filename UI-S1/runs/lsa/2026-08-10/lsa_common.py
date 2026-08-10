import importlib.util
import json
import math
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


RUN_DIR = Path(__file__).resolve().parent
ROOT = RUN_DIR.parents[2]
CLOSE = ROOT / "runs/close/2026-08-08"
CEV = ROOT / "runs/cev/2026-08-09"
sys.path.insert(0, str(CEV))

from cev import Candidate as CEVCandidate
from cev import select as cev_select
from cev import token_set_f1


RADII = (0.01, 0.03, 0.07, 0.14)
ACTION_CATEGORIES = ("POINT", "CLICK", "TYPE", "SELECT", "OTHER")


@dataclass(frozen=True)
class Candidate:
    source: str
    lineage: str
    action: str
    coordinate: tuple[float, float] | None
    baseline_coordinate: tuple[float, float] | None
    parameter: str
    parse_ok: bool
    stage: str
    success: bool
    order: int


@dataclass(frozen=True)
class Row:
    row_id: str
    benchmark: str
    fold: int
    group: str
    candidates: tuple[Candidate, ...]


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def normalize_action(action):
    value = str(action or "").upper()
    return value if value in {"POINT", "CLICK", "TYPE", "SELECT"} else "OTHER"


def load_rows(arm="C_uni"):
    e1 = load_module(CLOSE / "e1_arm_aggregator_matrix.py", "lsa_e1")
    output = {"mind2web": {}, "screenspot_pro": {}}

    mind_rows = [json.loads(line) for line in (e1.XFER / "data/mind2web/mind2web_test_task.jsonl").read_text().splitlines() if line.strip()]
    mind_by_id = {row["id"]: row for row in mind_rows}
    image_sizes = {row["id"]: Image.open(ROOT / row["image"]).size for row in mind_rows}
    full = {model: e1.load_unique(e1.XFER / "raw/stage1" / directory) for model, directory in e1.MODEL_DIRS.items()}
    view1 = {model: e1.load_unique(e1.XFER / "raw/stage1/view1" / directory) for model, directory in e1.MODEL_DIRS.items()}
    stage2 = {model: e1.load_unique(e1.XFER / "raw/stage2" / directory) for model, directory in e1.MODEL_DIRS.items()}
    slots = e1.mind_slots(mind_by_id, full, view1, stage2, arm)
    fold_map = json.loads((ROOT / "runs/complementarity/2026-07-30/folds.json").read_text())["pools"]["mind2web/visual"]["group_to_fold"]
    for row_id, row in mind_by_id.items():
        candidates = []
        for order, (source, lineage, prediction) in enumerate(slots[row_id]):
            coordinate = prediction.get("position")
            success = bool(e1.score_prediction(row, prediction, image_sizes[row_id]))
            candidates.append(Candidate(
                source=source,
                lineage=lineage,
                action=normalize_action(prediction.get("action")),
                coordinate=tuple(coordinate) if coordinate is not None else None,
                baseline_coordinate=tuple(coordinate) if coordinate is not None else None,
                parameter=str(prediction.get("value") or ""),
                parse_ok=bool(prediction.get("parse_ok")),
                stage="stage2" if source.startswith("stage2_") else "stage1",
                success=success,
                order=order,
            ))
        output["mind2web"][row_id] = Row(
            row_id=row_id,
            benchmark="mind2web",
            fold=fold_map[row["website"]],
            group=row["episode_id"],
            candidates=tuple(candidates),
        )

    common = e1.load_module(e1.CONSOLIDATE / "common.py", "lsa_screen_common")
    context = common.load_context()
    regions = {row["id"]: row for row in map(json.loads, (e1.CONSOLIDATE / "raw/q1_regions.jsonl").read_text().splitlines())}
    q1 = {model: e1.load_screen_q1(model) for model in e1.SCREEN_MODELS}
    for row_id in context["row_ids"]:
        metadata = context["metadata"][row_id]
        width, height = metadata["img_size"]
        candidates = []
        slots = e1.screen_slots(context, regions, q1, arm, row_id)
        for order, (source, prediction) in enumerate(slots):
            lineage = prediction["model"]
            point = tuple(prediction["point"])
            candidates.append(Candidate(
                source=source,
                lineage=lineage,
                action="POINT",
                coordinate=(point[0] / width, point[1] / height),
                baseline_coordinate=point,
                parameter="",
                parse_ok=True,
                stage="stage2" if "_crop" in source else "stage1",
                success=bool(e1.point_in_bbox(point, metadata["target_bbox"])),
                order=order,
            ))
        output["screenspot_pro"][row_id] = Row(
            row_id=row_id,
            benchmark="screenspot_pro",
            fold=context["fold_for_group"][metadata["application"]],
            group=metadata["application"],
            candidates=tuple(candidates),
        )
    return output


def reliability_statistics(rows, row_ids):
    sums = defaultdict(float)
    counts = defaultdict(int)
    for row_id in row_ids:
        for candidate in rows[row_id].candidates:
            sums[candidate.source] += float(candidate.success)
            counts[candidate.source] += 1
    return dict(sums), dict(counts)


def candidate_reliability(candidate, sums, counts, leave_one=False):
    total = sums.get(candidate.source, 0.0) - (float(candidate.success) if leave_one else 0.0)
    count = counts.get(candidate.source, 0) - (1 if leave_one else 0)
    return (total + 1.0) / (count + 2.0)


def fallback_index(row, sums, counts):
    candidates = [
        CEVCandidate(
            action=candidate.action,
            coordinate=candidate.baseline_coordinate,
            parameter=candidate.parameter,
            source=candidate.source,
            reliability=candidate_reliability(candidate, sums, counts),
            order=candidate.order,
            payload=candidate.order,
            parse_ok=candidate.parse_ok,
            lineage=candidate.lineage,
        )
        for candidate in row.candidates
    ]
    granularity = "G0" if row.benchmark == "mind2web" else "G4"
    threshold = (1.0, 1.0) if row.benchmark == "mind2web" else 14.0
    prediction, _ = cev_select(candidates, granularity, threshold)
    return int(prediction.payload)


def entropy(values):
    counts = Counter(values)
    total = sum(counts.values())
    return -sum((count / total) * math.log(count / total) for count in counts.values()) if total else 0.0


def feature_names():
    names = [
        "parse_ok",
        *[f"action_{action}" for action in ACTION_CATEGORIES],
        "coordinate_present", "parameter_present", "log_parameter_length",
        "stage1", "stage2", "source_reliability",
        "candidate_count", "parsed_fraction", "action_support_fraction",
        "action_vote_margin", "distinct_action_rate", "action_entropy",
        "coordinate_min_distance", "coordinate_mean_distance", "coordinate_median_distance",
        "coordinate_medoid_distance",
    ]
    names.extend(f"coordinate_support_r{radius}" for radius in RADII)
    names.extend(f"same_action_coordinate_support_r{radius}" for radius in RADII)
    names.extend(f"lineage_support_r{radius}" for radius in RADII)
    names.extend([
        "exact_parameter_support", "mean_parameter_f1", "max_parameter_f1",
        "same_lineage_fraction", "total_lineages", "row_coordinate_dispersion",
        "parameter_bearing_fraction",
    ])
    return names


def row_features(row, sums, counts, leave_one=False):
    candidates = row.candidates
    parsed = [candidate for candidate in candidates if candidate.parse_ok]
    parsed_count = len(parsed)
    actions = [candidate.action for candidate in parsed]
    action_counts = Counter(actions)
    votes = sorted(action_counts.values(), reverse=True)
    action_margin = ((votes[0] - (votes[1] if len(votes) > 1 else 0)) / parsed_count) if parsed_count else 0.0
    coordinates = [candidate.coordinate for candidate in parsed if candidate.coordinate is not None]
    pair_distances = [math.dist(left, right) for index, left in enumerate(coordinates) for right in coordinates[index + 1:]]
    coordinate_dispersion = float(np.median(pair_distances)) if pair_distances else 0.0
    if coordinates:
        medoid = min(coordinates, key=lambda point: sum(math.dist(point, other) for other in coordinates))
    else:
        medoid = None
    lineages = {candidate.lineage for candidate in parsed}
    parameter_fraction = sum(bool(candidate.parameter) for candidate in parsed) / parsed_count if parsed_count else 0.0
    output = []
    for candidate in candidates:
        action = candidate.action if candidate.action in ACTION_CATEGORIES else "OTHER"
        same_action = [other for other in parsed if other.action == candidate.action]
        coordinate_peers = [other for other in parsed if other.coordinate is not None and candidate.coordinate is not None]
        distances = [math.dist(candidate.coordinate, other.coordinate) for other in coordinate_peers] if candidate.coordinate is not None else []
        nonself_distances = [distance for other, distance in zip(coordinate_peers, distances) if other.order != candidate.order]
        parameter_peers = [other for other in same_action if other.order != candidate.order]
        parameter_scores = [token_set_f1(candidate.parameter, other.parameter) for other in parameter_peers]
        values = [
            float(candidate.parse_ok),
            *[float(action == category) for category in ACTION_CATEGORIES],
            float(candidate.coordinate is not None),
            float(bool(candidate.parameter)),
            math.log1p(len(candidate.parameter)),
            float(candidate.stage == "stage1"),
            float(candidate.stage == "stage2"),
            candidate_reliability(candidate, sums, counts, leave_one),
            float(len(candidates)),
            parsed_count / len(candidates),
            action_counts.get(candidate.action, 0) / parsed_count if parsed_count else 0.0,
            action_margin,
            len(action_counts) / parsed_count if parsed_count else 0.0,
            entropy(actions),
            min(nonself_distances) if nonself_distances else 0.0,
            float(np.mean(nonself_distances)) if nonself_distances else 0.0,
            float(np.median(nonself_distances)) if nonself_distances else 0.0,
            math.dist(candidate.coordinate, medoid) if candidate.coordinate is not None and medoid is not None else 0.0,
        ]
        for radius in RADII:
            values.append(sum(distance <= radius for distance in distances) / max(1, len(coordinates)))
        for radius in RADII:
            support = sum(
                other.coordinate is not None and candidate.coordinate is not None
                and math.dist(candidate.coordinate, other.coordinate) <= radius
                for other in same_action
            )
            values.append(support / max(1, len(same_action)))
        for radius in RADII:
            supported = {
                other.lineage for other in parsed
                if other.coordinate is not None and candidate.coordinate is not None
                and math.dist(candidate.coordinate, other.coordinate) <= radius
            }
            values.append(len(supported) / max(1, len(lineages)))
        values.extend([
            sum(other.parameter == candidate.parameter for other in same_action) / max(1, len(same_action)),
            float(np.mean(parameter_scores)) if parameter_scores else 0.0,
            max(parameter_scores) if parameter_scores else 0.0,
            sum(other.lineage == candidate.lineage for other in parsed) / max(1, parsed_count),
            float(len(lineages)),
            coordinate_dispersion,
            parameter_fraction,
        ])
        output.append(values)
    values = np.asarray(output, dtype=np.float32)
    if values.shape[1] != len(feature_names()):
        raise ValueError(f"feature width mismatch: {values.shape[1]} != {len(feature_names())}")
    return values


def training_matrix(rows_by_benchmark, ids_by_benchmark, reliability_by_benchmark, feature_indices=None):
    arrays = []
    labels = []
    weights = []
    row_refs = []
    mixed_counts = {}
    for benchmark, ids in ids_by_benchmark.items():
        rows = rows_by_benchmark[benchmark]
        sums, counts = reliability_by_benchmark[benchmark]
        mixed = []
        for row_id in ids:
            successes = [candidate.success for candidate in rows[row_id].candidates]
            if 0 < sum(successes) < len(successes):
                mixed.append(row_id)
        mixed_counts[benchmark] = len(mixed)
        benchmark_mass = 1.0 / max(1, len(mixed))
        for row_id in mixed:
            row = rows[row_id]
            features = row_features(row, sums, counts, leave_one=True)
            if feature_indices is not None:
                features = features[:, feature_indices]
            y = np.asarray([candidate.success for candidate in row.candidates], dtype=np.int8)
            positives = int(y.sum())
            negatives = len(y) - positives
            sample_weight = np.where(y == 1, 0.5 / positives, 0.5 / negatives) * benchmark_mass
            arrays.append(features)
            labels.append(y)
            weights.append(sample_weight)
            row_refs.extend((benchmark, row_id, index) for index in range(len(y)))
    if not arrays:
        raise ValueError("LSA-K1: no mixed-label training rows")
    return np.concatenate(arrays), np.concatenate(labels), np.concatenate(weights), row_refs, mixed_counts


def evaluation_rows(rows_by_benchmark, ids_by_benchmark, reliability_by_benchmark, feature_indices=None):
    output = {}
    for benchmark, ids in ids_by_benchmark.items():
        rows = rows_by_benchmark[benchmark]
        sums, counts = reliability_by_benchmark[benchmark]
        output[benchmark] = {}
        for row_id in ids:
            row = rows[row_id]
            features = row_features(row, sums, counts, leave_one=False)
            if feature_indices is not None:
                features = features[:, feature_indices]
            output[benchmark][row_id] = {
                "features": features,
                "labels": np.asarray([candidate.success for candidate in row.candidates], dtype=np.int8),
                "fallback_index": fallback_index(row, sums, counts),
            }
    return output
