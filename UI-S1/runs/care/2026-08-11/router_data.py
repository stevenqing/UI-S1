import hashlib
import json
import math
import re
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import torch

from router_model import RouterBatch


RUN_DIR = Path(__file__).resolve().parent
ROOT = RUN_DIR.parents[2]
VUS = ROOT / "runs/visual-utility-selector/2026-08-11"
ARMS = ("C_uni", "C_cond", "C_rand", "C_self")
BENCHMARKS = ("mind2web", "screenspot_pro")
ACTION_CATEGORIES = ("POINT", "CLICK", "TYPE", "SELECT", "OTHER")
RADII = (0.01, 0.03, 0.07, 0.14)


@dataclass(frozen=True)
class RouterData:
    features: np.ndarray
    targets: np.ndarray
    target_distribution: np.ndarray
    listwise_active: np.ndarray
    weights: np.ndarray
    row_ids: tuple[str, ...]
    benchmarks: tuple[str, ...]
    folds: np.ndarray
    groups: tuple[str, ...]

    def __len__(self):
        return len(self.row_ids)


@dataclass(frozen=True)
class Standardizer:
    mean: np.ndarray
    scale: np.ndarray

    def transform(self, data):
        values = (data.features - self.mean[None, None, :]) / self.scale[None, None, :]
        return replace(data, features=values.astype(np.float32))


def load_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def load_public(path=VUS / "data/public_records.jsonl"):
    rows = load_jsonl(path)
    output = {row["sample_key"]: row for row in rows}
    if len(output) != len(rows):
        raise ValueError("duplicate CARE public sample keys")
    return output


def load_label_folds(folds, label_dir=VUS / "data"):
    rows = []
    for fold in sorted(set(int(value) for value in folds)):
        path = Path(label_dir) / f"private_labels_fold-{fold}.jsonl"
        if not path.is_file():
            raise FileNotFoundError(path)
        rows.extend(load_jsonl(path))
    output = {row["sample_key"]: row for row in rows}
    if len(output) != len(rows):
        raise ValueError("duplicate CARE private label keys")
    return output


def token_set_f1(left, right):
    left_tokens = set(re.findall(r"[a-z0-9]+", str(left).lower()))
    right_tokens = set(re.findall(r"[a-z0-9]+", str(right).lower()))
    if not left_tokens and not right_tokens:
        return 1.0
    if not left_tokens or not right_tokens:
        return 0.0
    overlap = len(left_tokens & right_tokens)
    precision = overlap / len(left_tokens)
    recall = overlap / len(right_tokens)
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def entropy(values):
    counts = Counter(values)
    total = len(values)
    return -sum((count / total) * math.log(count / total) for count in counts.values()) if total else 0.0


def load_source_metadata(path=RUN_DIR / "data/stage1_sources.jsonl"):
    rows = load_jsonl(path)
    output = {(row["benchmark"], row["row_id"]): row for row in rows}
    if len(output) != len(rows):
        raise ValueError("duplicate CARE stage1 source metadata")
    if any(len(row["sources"]) != 6 for row in rows):
        raise ValueError("CARE source metadata width mismatch")
    return output


def fit_source_statistics(source_metadata, labels, folds):
    fold_set = set(int(value) for value in folds)
    sums = Counter()
    counts = Counter()
    for (benchmark, row_id), metadata in source_metadata.items():
        if metadata["fold"] not in fold_set:
            continue
        key = f"{benchmark}/C_uni/{row_id}"
        if key not in labels:
            raise KeyError(key)
        success = labels[key]["candidate_success"][:6]
        for source, value in zip(metadata["sources"], success):
            source_key = (benchmark, source)
            sums[source_key] += float(value)
            counts[source_key] += 1
    if not counts:
        raise ValueError("CARE source reliability has no fit rows")
    return dict(sums), dict(counts)


def source_reliability_values(benchmark, row_id, source_metadata, labels, statistics, leave_one):
    metadata = source_metadata[(benchmark, row_id)]
    sums, counts = statistics
    success = labels[f"{benchmark}/C_uni/{row_id}"]["candidate_success"][:6] if leave_one else [0.0] * 6
    output = []
    for source, value in zip(metadata["sources"], success):
        source_key = (benchmark, source)
        total = sums[source_key] - (float(value) if leave_one else 0.0)
        count = counts[source_key] - (1 if leave_one else 0)
        if count < 0:
            raise ValueError(f"CARE negative source count: {source_key}")
        output.append((total + 1.0) / (count + 2.0))
    return output


def stage1_features(candidates, benchmark, reliability):
    if len(candidates) != 6:
        raise ValueError("CARE A1 requires exactly six candidates")
    if len(reliability) != 6:
        raise ValueError("CARE A1 reliability width mismatch")
    parsed = [candidate for candidate in candidates if candidate["parse_ok"]]
    actions = [candidate["action"] if candidate["action"] in ACTION_CATEGORIES else "OTHER" for candidate in parsed]
    action_counts = Counter(actions)
    votes = sorted(action_counts.values(), reverse=True)
    action_margin = ((votes[0] - (votes[1] if len(votes) > 1 else 0)) / len(parsed)) if parsed else 0.0
    coordinates = [candidate["coordinate"] for candidate in parsed if candidate["coordinate"] is not None]
    pair_distances = [math.dist(left, right) for index, left in enumerate(coordinates) for right in coordinates[index + 1:]]
    dispersion = float(np.median(pair_distances)) if pair_distances else 0.0
    medoid = min(coordinates, key=lambda point: sum(math.dist(point, other) for other in coordinates)) if coordinates else None
    parameter_fraction = sum(bool(candidate["parameter"]) for candidate in parsed) / len(parsed) if parsed else 0.0
    output = []
    for index, candidate in enumerate(candidates):
        action = candidate["action"] if candidate["action"] in ACTION_CATEGORIES else "OTHER"
        coordinate = candidate["coordinate"]
        peers = [other for other in parsed if other["coordinate"] is not None and coordinate is not None]
        distances = [math.dist(coordinate, other["coordinate"]) for other in peers] if coordinate is not None else []
        nonself = [distance for other, distance in zip(peers, distances) if other is not candidate]
        same_action = [other for other in parsed if other["action"] == candidate["action"]]
        parameter_peers = [other for other in same_action if other is not candidate]
        parameter_scores = [token_set_f1(candidate["parameter"], other["parameter"]) for other in parameter_peers]
        x, y = coordinate if coordinate is not None else (0.0, 0.0)
        out_of_frame = coordinate is not None and not (0 <= x <= 1 and 0 <= y <= 1)
        values = [
            float(candidate["parse_ok"]),
            *[float(action == category) for category in ACTION_CATEGORIES],
            float(coordinate is not None), float(x), float(y), float(out_of_frame),
            float(bool(candidate["parameter"])), math.log1p(len(candidate["parameter"])),
            action_counts.get(action, 0) / max(1, len(parsed)), action_margin,
            len(action_counts) / max(1, len(parsed)), entropy(actions),
            min(nonself) if nonself else 0.0,
            float(np.mean(nonself)) if nonself else 0.0,
            float(np.median(nonself)) if nonself else 0.0,
            math.dist(coordinate, medoid) if coordinate is not None and medoid is not None else 0.0,
        ]
        values.extend(sum(distance <= radius for distance in distances) / max(1, len(coordinates)) for radius in RADII)
        for radius in RADII:
            support = sum(
                other["coordinate"] is not None and coordinate is not None
                and math.dist(coordinate, other["coordinate"]) <= radius
                for other in same_action
            )
            values.append(support / max(1, len(same_action)))
        values.extend([
            sum(other["parameter"] == candidate["parameter"] for other in same_action) / max(1, len(same_action)),
            float(np.mean(parameter_scores)) if parameter_scores else 0.0,
            max(parameter_scores) if parameter_scores else 0.0,
            dispersion, parameter_fraction,
            float(reliability[index]),
            float(benchmark == "mind2web"), float(benchmark == "screenspot_pro"),
        ])
        output.append(values)
    return np.asarray(output, dtype=np.float32)


def verify_shared_stage1(public, benchmark, row_id):
    blocks = [public[f"{benchmark}/{arm}/{row_id}"]["candidates"][:6] for arm in ARMS]
    if any(block != blocks[0] for block in blocks[1:]):
        raise ValueError(f"CARE-K1 first-six mismatch: {benchmark}/{row_id}")
    return blocks[0]


def build_router_data(public, labels, folds, source_metadata, source_statistics, leave_one=False):
    fold_set = set(int(value) for value in folds)
    rows = []
    for benchmark in BENCHMARKS:
        candidates = [row for row in public.values() if row["benchmark"] == benchmark and row["arm"] == "C_uni" and row["fold"] in fold_set]
        for record in sorted(candidates, key=lambda row: row["row_id"]):
            row_id = record["row_id"]
            targets = []
            for arm in ARMS:
                key = f"{benchmark}/{arm}/{row_id}"
                if key not in labels:
                    raise KeyError(key)
                targets.append(float(any(labels[key]["candidate_success"])))
            reliability = source_reliability_values(
                benchmark, row_id, source_metadata, labels, source_statistics, leave_one
            )
            rows.append((record, stage1_features(verify_shared_stage1(public, benchmark, row_id), benchmark, reliability), targets))
    features = np.stack([row[1] for row in rows])
    targets = np.asarray([row[2] for row in rows], dtype=np.float32)
    positive_counts = targets.sum(axis=1)
    distribution = np.divide(targets, positive_counts[:, None], out=np.zeros_like(targets), where=positive_counts[:, None] > 0)
    active = (positive_counts > 0).astype(np.float32)
    benchmarks = tuple(row[0]["benchmark"] for row in rows)
    weights = np.zeros(len(rows), dtype=np.float64)
    for benchmark in BENCHMARKS:
        indices = [index for index, value in enumerate(benchmarks) if value == benchmark]
        for index in indices:
            weights[index] = 1.0 / len(indices)
    return RouterData(
        features=features,
        targets=targets,
        target_distribution=distribution,
        listwise_active=active,
        weights=weights,
        row_ids=tuple(row[0]["row_id"] for row in rows),
        benchmarks=benchmarks,
        folds=np.asarray([row[0]["fold"] for row in rows], dtype=np.int8),
        groups=tuple(row[0]["group"] for row in rows),
    )


def fit_standardizer(data):
    flat = data.features.reshape(-1, data.features.shape[-1]).astype(np.float64)
    mean = flat.mean(axis=0).astype(np.float32)
    scale = flat.std(axis=0).astype(np.float32)
    scale[scale < 1e-6] = 1.0
    return Standardizer(mean, scale)


def torch_batch(data, indices, device):
    return RouterBatch(
        features=torch.as_tensor(data.features[indices], device=device),
        targets=torch.as_tensor(data.targets[indices], device=device),
        target_distribution=torch.as_tensor(data.target_distribution[indices], device=device),
        listwise_active=torch.as_tensor(data.listwise_active[indices], device=device),
        weights=torch.as_tensor(data.weights[indices], dtype=torch.float32, device=device),
    )


def deterministic_permutations(row_ids, benchmark, epoch, seed):
    values = []
    for row_id, name in zip(row_ids, benchmark):
        digest = hashlib.sha256(f"{name}/{row_id}/{epoch}/{seed}".encode()).digest()
        values.append(np.random.default_rng(int.from_bytes(digest[:8], "big")).permutation(6))
    return np.stack(values).astype(np.int64)
