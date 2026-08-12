import hashlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.feature_extraction.text import HashingVectorizer


RUN_DIR = Path(__file__).resolve().parent
ROOT = RUN_DIR.parents[2]
VUS = ROOT / "runs/visual-utility-selector/2026-08-11"
sys.path.insert(0, str(VUS))

from set_ranker_data import keyed, original_visual_values


CHANNELS = ("vus_binding", "global_semantic", "fine_local", "context_local", "random_placebo")
EXPERTS = CHANNELS[1:]
REAL_EXPERTS = EXPERTS[:3]
BENCHMARKS = ("mind2web", "screenspot_pro")
ARMS = ("C_uni", "C_cond", "C_rand", "C_self")
ACTIONS = ("POINT", "CLICK", "TYPE", "SELECT", "OTHER")
PROHIBITED_KEYS = {
    "application", "website", "target", "target_bbox", "target_area", "bbox",
    "candidate_success", "success", "label", "evaluator", "outer_fold",
}


@dataclass(frozen=True)
class CivaBaseData:
    full_features: np.ndarray
    no_text_features: np.ndarray
    text_only_features: np.ndarray
    baseline_indices: np.ndarray
    expert_indices: np.ndarray
    weights: np.ndarray
    sample_keys: tuple[str, ...]
    benchmarks: tuple[str, ...]
    arms: tuple[str, ...]
    row_ids: tuple[str, ...]
    folds: np.ndarray
    groups: tuple[str, ...]

    def __len__(self):
        return len(self.sample_keys)

    def subset(self, folds):
        allowed = set(int(fold) for fold in folds)
        indices = np.flatnonzero(np.asarray([fold in allowed for fold in self.folds]))
        return CivaBaseData(
            full_features=self.full_features[indices],
            no_text_features=self.no_text_features[indices],
            text_only_features=self.text_only_features[indices],
            baseline_indices=self.baseline_indices[indices],
            expert_indices=self.expert_indices[indices],
            weights=self.weights[indices],
            sample_keys=tuple(self.sample_keys[index] for index in indices),
            benchmarks=tuple(self.benchmarks[index] for index in indices),
            arms=tuple(self.arms[index] for index in indices),
            row_ids=tuple(self.row_ids[index] for index in indices),
            folds=self.folds[indices],
            groups=tuple(self.groups[index] for index in indices),
        )


@dataclass(frozen=True)
class CivaLabeledData:
    base: CivaBaseData
    baseline_success: np.ndarray
    expert_success: np.ndarray
    delta: np.ndarray

    def __len__(self):
        return len(self.base)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_array(value):
    value = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(json.dumps(value.shape).encode())
    digest.update(value.tobytes())
    return digest.hexdigest()


def validate_config(config):
    if config.get("status") != "PREREGISTERED_AFTER_DELTA_BEFORE_CIVA_FIT":
        raise ValueError("CIVA protocol is not frozen")
    if tuple(config.get("channels", {})) != CHANNELS:
        raise ValueError("CIVA channel order mismatch")
    if config.get("baseline") != CHANNELS[0]:
        raise ValueError("CIVA baseline mismatch")
    if tuple(config.get("real_experts", ())) != REAL_EXPERTS:
        raise ValueError("CIVA real experts mismatch")
    if config.get("placebo_expert") != EXPERTS[-1]:
        raise ValueError("CIVA placebo mismatch")
    prohibited = set(config["features"].get("prohibited", ()))
    expected = {
        "expert_channel_logits", "source_model_slot_identity", "website_application_identity",
        "target_bbox_area", "evaluator_outputs", "private_labels", "outer_fold_identity",
    }
    if prohibited != expected:
        raise ValueError("CIVA prohibited-feature contract mismatch")
    if config["features"].get("instruction_hash_dimensions") != 256:
        raise ValueError("CIVA instruction feature width mismatch")
    expected_learner = {
        "name": "HistGradientBoostingClassifier",
        "heads": ["rescue", "harm"],
        "learning_rate": 0.05,
        "max_iter": 200,
        "max_leaf_nodes": 15,
        "max_depth": None,
        "min_samples_leaf": 30,
        "l2_regularization": 0.1,
        "early_stopping": False,
    }
    if config.get("learner") != expected_learner:
        raise ValueError("CIVA learner contract mismatch")
    if tuple(config.get("variants", {})) != (
        "REAL_FULL", "REAL_NO_TEXT", "REAL_TEXT_ONLY", "PLACEBO_FULL", "MATCHED_RANDOM"
    ):
        raise ValueError("CIVA variant contract mismatch")
    nested = config["nested_protocol"]
    if (
        nested.get("outer_folds") != 5
        or nested.get("development_oof_folds") != 4
        or not nested.get("fold_sealed_labels")
        or not nested.get("atomic_pretest")
    ):
        raise ValueError("CIVA nested protocol mismatch")
    if config["statistics"] != {"resamples": 10000, "confidence": 0.99}:
        raise ValueError("CIVA statistical contract mismatch")


def load_inputs(config):
    public_path = VUS / "data/public_records.jsonl"
    public_manifest = json.loads((VUS / "data/MANIFEST.json").read_text())
    observed_public = sha256_file(public_path)
    if observed_public != public_manifest["public_sha256"]:
        raise ValueError(f"CIVA-K1 public hash mismatch: {observed_public}")
    public = keyed(public_path)
    channels = {}
    for name in CHANNELS:
        item = config["channels"][name]
        path = ROOT / item["path"]
        observed = sha256_file(path)
        if observed != item["sha256"]:
            raise ValueError(f"CIVA-K1 channel hash mismatch: {name}/{observed}")
        channels[name] = keyed(path)
        if set(channels[name]) != set(public):
            raise ValueError(f"CIVA-K1 channel identity mismatch: {name}")
    return public, channels


def audit_public_record(record):
    def visit(value):
        if isinstance(value, dict):
            for key, child in value.items():
                normalized = key.lower()
                if normalized in PROHIBITED_KEYS or normalized.startswith("gt_") or normalized.endswith("_bbox"):
                    raise ValueError(f"CIVA-K2 prohibited public field: {key}")
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(record)
    expected = {
        "schema_version", "sample_key", "benchmark", "arm", "row_id", "fold", "group",
        "image_path", "image_sha256", "instruction", "history", "candidates",
    }
    if set(record) != expected:
        raise ValueError(f"CIVA public schema mismatch: {set(record) ^ expected}")
    if len(record["candidates"]) != 12:
        raise ValueError("CIVA requires exactly 12 candidates")


def text_features(records, dimensions=256):
    vectorizer = HashingVectorizer(
        n_features=dimensions,
        alternate_sign=False,
        norm="l2",
        ngram_range=(1, 2),
        lowercase=True,
    )
    return vectorizer.transform([record["instruction"] for record in records]).toarray().astype(np.float32)


def _finite_coordinate(candidate):
    coordinate = candidate.get("coordinate")
    if coordinate is None:
        return None
    values = tuple(float(value) for value in coordinate)
    if len(values) != 2 or not all(math.isfinite(value) for value in values):
        raise ValueError("CIVA invalid public coordinate")
    return values


def _quantiles(values, count=5):
    if not values:
        return np.zeros(count, dtype=np.float32)
    return np.quantile(np.asarray(values, dtype=np.float64), np.linspace(0, 1, count)).astype(np.float32)


def _image_aspect(record, cache):
    path = record["image_path"]
    if path not in cache:
        with Image.open(path) as image:
            width, height = image.size
        cache[path] = (float(width) / max(float(height), 1.0), math.log1p(width * height))
    return cache[path]


def public_structure(record, baseline_index, image_cache):
    candidates = record["candidates"]
    action_counts = np.zeros(len(ACTIONS), dtype=np.float32)
    coordinates = []
    parse_count = 0
    parameter_lengths = []
    for candidate in candidates:
        action = candidate["action"] if candidate["action"] in ACTIONS else "OTHER"
        action_counts[ACTIONS.index(action)] += 1.0 / 12.0
        coordinate = _finite_coordinate(candidate)
        if coordinate is not None:
            coordinates.append(coordinate)
        parse_count += int(bool(candidate["parse_ok"]))
        parameter_lengths.append(len(candidate.get("parameter", "")))
    pair_distances = [
        math.dist(coordinates[left], coordinates[right])
        for left in range(len(coordinates))
        for right in range(left + 1, len(coordinates))
    ]
    if coordinates:
        coordinate_array = np.asarray(coordinates, dtype=np.float64)
        coordinate_summary = np.concatenate((
            coordinate_array.mean(axis=0), coordinate_array.std(axis=0),
            coordinate_array.min(axis=0), coordinate_array.max(axis=0),
        )).astype(np.float32)
    else:
        coordinate_summary = np.zeros(8, dtype=np.float32)
    baseline_candidate = candidates[int(baseline_index)]
    baseline_coordinate = _finite_coordinate(baseline_candidate)
    baseline_position = np.asarray(
        (*baseline_coordinate, 1.0) if baseline_coordinate is not None else (0.0, 0.0, 0.0),
        dtype=np.float32,
    )
    baseline_action = np.asarray([
        float((baseline_candidate["action"] if baseline_candidate["action"] in ACTIONS else "OTHER") == action)
        for action in ACTIONS
    ], dtype=np.float32)
    benchmark_arm = np.asarray(
        [float(record["benchmark"] == value) for value in BENCHMARKS]
        + [float(record["arm"] == value) for value in ARMS],
        dtype=np.float32,
    )
    aspect, log_pixels = _image_aspect(record, image_cache)
    return np.concatenate((
        action_counts,
        coordinate_summary,
        _quantiles(pair_distances),
        baseline_position,
        baseline_action,
        np.asarray([
            len(coordinates) / 12.0,
            parse_count / 12.0,
            np.mean(parameter_lengths) / 256.0,
            np.count_nonzero(parameter_lengths) / 12.0,
            aspect,
            log_pixels,
        ], dtype=np.float32),
        benchmark_arm,
    )), benchmark_arm


def baseline_state(prediction):
    logits, probabilities = original_visual_values(prediction)
    order = np.argsort(-probabilities, kind="stable")
    sorted_probabilities = probabilities[order]
    centered = logits - float(logits.mean())
    entropy = -float(np.sum(probabilities * np.log(np.maximum(probabilities, 1e-12))))
    return np.concatenate((
        sorted_probabilities.astype(np.float32),
        np.asarray([
            sorted_probabilities[0] - sorted_probabilities[1],
            sorted_probabilities[0] - sorted_probabilities[2],
            entropy,
            float(centered.std()),
            float(centered.max() - centered.min()),
        ], dtype=np.float32),
    )), int(order[0])


def policy_index(prediction):
    _, probabilities = original_visual_values(prediction)
    return int(np.argmax(probabilities))


def _weights(benchmarks, row_ids, arms):
    output = np.zeros(len(row_ids), dtype=np.float64)
    for benchmark in BENCHMARKS:
        unique_rows = sorted({row_id for name, row_id in zip(benchmarks, row_ids) if name == benchmark})
        for row_id in unique_rows:
            indices = [
                index for index, (name, value) in enumerate(zip(benchmarks, row_ids))
                if name == benchmark and value == row_id
            ]
            if sorted(arms[index] for index in indices) != sorted(ARMS):
                raise ValueError(f"CIVA arm coverage mismatch: {benchmark}/{row_id}")
            for index in indices:
                output[index] = 1.0 / len(unique_rows) / len(indices)
    return output


def build_base_data(public, channels, text_dimensions=256):
    keys = sorted(public)
    records = [public[key] for key in keys]
    for record in records:
        audit_public_record(record)
    text = text_features(records, text_dimensions)
    no_text_rows = []
    text_only_rows = []
    baseline_indices = []
    expert_indices = []
    image_cache = {}
    for key, record in zip(keys, records):
        state, baseline_index = baseline_state(channels["vus_binding"][key])
        structure, benchmark_arm = public_structure(record, baseline_index, image_cache)
        no_text_rows.append(np.concatenate((state, structure)))
        text_only_rows.append(np.concatenate((text[len(no_text_rows) - 1], benchmark_arm)))
        baseline_indices.append(baseline_index)
        expert_indices.append([policy_index(channels[name][key]) for name in EXPERTS])
    no_text = np.stack(no_text_rows).astype(np.float32)
    text_only = np.stack(text_only_rows).astype(np.float32)
    benchmarks = tuple(record["benchmark"] for record in records)
    arms = tuple(record["arm"] for record in records)
    row_ids = tuple(record["row_id"] for record in records)
    return CivaBaseData(
        full_features=np.concatenate((text, no_text), axis=1).astype(np.float32),
        no_text_features=no_text,
        text_only_features=text_only,
        baseline_indices=np.asarray(baseline_indices, dtype=np.int8),
        expert_indices=np.asarray(expert_indices, dtype=np.int8),
        weights=_weights(benchmarks, row_ids, arms),
        sample_keys=tuple(keys),
        benchmarks=benchmarks,
        arms=arms,
        row_ids=row_ids,
        folds=np.asarray([record["fold"] for record in records], dtype=np.int8),
        groups=tuple(record["group"] for record in records),
    )


def attach_labels(base, labels_by_key):
    if set(base.sample_keys) != set(labels_by_key):
        raise ValueError("CIVA label identity mismatch")
    baseline_success = []
    expert_success = []
    for index, key in enumerate(base.sample_keys):
        labels = np.asarray(labels_by_key[key]["candidate_success"], dtype=np.bool_)
        if labels.shape != (12,):
            raise ValueError(f"CIVA label width mismatch: {key}")
        baseline_success.append(labels[int(base.baseline_indices[index])])
        expert_success.append(labels[base.expert_indices[index]])
    baseline_values = np.asarray(baseline_success, dtype=np.bool_)
    expert_values = np.asarray(expert_success, dtype=np.bool_)
    delta = expert_values.astype(np.int8) - baseline_values[:, None].astype(np.int8)
    return CivaLabeledData(base, baseline_values, expert_values, delta)