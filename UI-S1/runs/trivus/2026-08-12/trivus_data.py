import hashlib
import json
import math
from dataclasses import dataclass, replace

import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer


MAX_CANDIDATES = 12
INPUT_DIMENSION = 115
GENERIC_DIMENSIONS = 103
ACTION_CATEGORIES = (
    "POINT", "CLICK", "TYPE", "SELECT", "OPEN", "BACK", "SCROLL",
    "WAIT", "LONG_PRESS", "OTHER",
)
FAMILIES = ("mind2web", "screenspot_pro", "androidcontrol")
CELLS = ("C_uni", "C_cond", "C_rand", "C_self", "low", "high")
VARIANTS = (
    "JOINT3", "TARGET_ONLY", "JOINT2_NO_ANDROID", "NO_VISUAL",
    "RANDOM_ID_PLACEBO",
)
VISUAL_SLICE = slice(85, 92)
PLACEBO_SLICE = slice(103, 115)


PARAMETER_VECTORIZER = HashingVectorizer(
    analyzer="char",
    ngram_range=(2, 4),
    n_features=64,
    lowercase=True,
    alternate_sign=False,
    norm="l2",
)


@dataclass(frozen=True)
class TriVUSData:
    features: np.ndarray
    candidate_mask: np.ndarray
    fallback_indices: np.ndarray
    target_distribution: np.ndarray
    fallback_correct: np.ndarray
    weights: np.ndarray
    active: np.ndarray
    labels: np.ndarray
    context_keys: tuple[str, ...]
    sample_keys: tuple[str, ...]
    families: tuple[str, ...]
    cells: tuple[str, ...]
    row_ids: tuple[str, ...]
    folds: np.ndarray
    groups: tuple[str, ...]

    def __len__(self):
        return len(self.sample_keys)

    def subset(self, indices):
        indices = np.asarray(indices)
        if indices.dtype == np.bool_:
            if indices.shape != (len(self),):
                raise ValueError("TriVUS subset mask shape mismatch")
            indices = np.flatnonzero(indices)
        indices = indices.astype(np.int64, copy=False)
        return TriVUSData(
            features=self.features[indices],
            candidate_mask=self.candidate_mask[indices],
            fallback_indices=self.fallback_indices[indices],
            target_distribution=self.target_distribution[indices],
            fallback_correct=self.fallback_correct[indices],
            weights=self.weights[indices],
            active=self.active[indices],
            labels=self.labels[indices],
            context_keys=tuple(self.context_keys[index] for index in indices),
            sample_keys=tuple(self.sample_keys[index] for index in indices),
            families=tuple(self.families[index] for index in indices),
            cells=tuple(self.cells[index] for index in indices),
            row_ids=tuple(self.row_ids[index] for index in indices),
            folds=self.folds[indices],
            groups=tuple(self.groups[index] for index in indices),
        )


@dataclass(frozen=True)
class TriVUSStandardizer:
    mean: np.ndarray
    scale: np.ndarray
    variant: str

    def transform(self, data):
        validate_trivus_data(data)
        features = variant_features(
            data.features, data.candidate_mask, data.context_keys, self.variant
        )
        values = (features - self.mean[None, None, :]) / self.scale[None, None, :]
        values[~data.candidate_mask] = 0.0
        if not np.isfinite(values).all():
            raise ValueError("TriVUS standardized features are non-finite")
        return replace(data, features=values.astype(np.float32))


def canonical_action(action):
    value = str(action or "").strip().lower()
    groups = {
        "POINT": {"point"},
        "CLICK": {"click", "doubleclick", "rightclick", "moveto"},
        "LONG_PRESS": {"long_press", "longpress"},
        "TYPE": {"type", "input_text"},
        "SELECT": {"select"},
        "OPEN": {"open", "open_app", "launch_app"},
        "BACK": {"back", "go_back", "navigate_back", "press_back"},
        "SCROLL": {"scroll", "swipe"},
        "WAIT": {"wait", "idle"},
    }
    return next((name for name, values in groups.items() if value in values), "OTHER")


def whitespace_token_f1(left, right):
    left_tokens = set(str(left or "").lower().split())
    right_tokens = set(str(right or "").lower().split())
    if not left_tokens and not right_tokens:
        return 1.0
    if not left_tokens or not right_tokens:
        return 0.0
    overlap = len(left_tokens & right_tokens)
    if not overlap:
        return 0.0
    precision = overlap / len(left_tokens)
    recall = overlap / len(right_tokens)
    return 2 * precision * recall / (precision + recall)


def pair_kernel(left, right, sigma=0.07):
    if canonical_action(left["action"]) != canonical_action(right["action"]):
        return 0.0
    value = 1.0
    left_coordinate = left.get("coordinate")
    right_coordinate = right.get("coordinate")
    if (left_coordinate is None) != (right_coordinate is None):
        return 0.0
    if left_coordinate is not None:
        distance = math.dist(left_coordinate, right_coordinate)
        value *= math.exp(-(distance * distance) / (2 * sigma * sigma))
    if left.get("parameter") or right.get("parameter"):
        value *= whitespace_token_f1(left.get("parameter"), right.get("parameter"))
    return value


def parameter_hashes(candidates):
    values = PARAMETER_VECTORIZER.transform([
        str(candidate.get("parameter") or "") for candidate in candidates
    ]).toarray()
    return values.astype(np.float32)


def structural_features(candidates):
    count = len(candidates)
    if count not in (3, 12):
        raise ValueError("TriVUS structural candidate count must be 3 or 12")
    expected_fields = {"action", "coordinate", "parameter", "parse_ok"}
    for candidate in candidates:
        if (
            not isinstance(candidate, dict)
            or set(candidate) != expected_fields
            or not isinstance(candidate["action"], str)
            or not isinstance(candidate["parameter"], str)
            or len(candidate["parameter"]) > 256
            or type(candidate["parse_ok"]) is not bool
        ):
            raise ValueError("TriVUS public candidate schema mismatch")
    hashes = parameter_hashes(candidates)
    output = np.zeros((count, 85), dtype=np.float32)
    for index, candidate in enumerate(candidates):
        action = canonical_action(candidate.get("action"))
        output[index, ACTION_CATEGORIES.index(action)] = 1.0
        coordinate = candidate.get("coordinate")
        if coordinate is not None:
            if (
                not isinstance(coordinate, (list, tuple))
                or len(coordinate) != 2
                or not all(
                    isinstance(value, (int, float, np.integer, np.floating))
                    and not isinstance(value, (bool, np.bool_))
                    and math.isfinite(float(value))
                    and 0.0 <= float(value) <= 1.0
                    for value in coordinate
                )
            ):
                raise ValueError("TriVUS invalid coordinate")
            output[index, 10:13] = (float(coordinate[0]), float(coordinate[1]), 1.0)
        parameter = str(candidate.get("parameter") or "")
        output[index, 13:16] = (
            float(bool(candidate.get("parse_ok"))),
            float(bool(parameter)),
            min(len(parameter), 256) / 256.0,
        )
        output[index, 16:80] = hashes[index]
        peers = [peer for peer in range(count) if peer != index]
        same_action = [
            peer for peer in peers
            if canonical_action(candidates[peer].get("action")) == action
        ]
        kernels = [pair_kernel(candidate, candidates[peer]) for peer in peers]
        neighborhoods = []
        for peer in peers:
            other_coordinate = candidates[peer].get("coordinate")
            neighborhoods.append(
                coordinate is not None and other_coordinate is not None
                and math.dist(coordinate, other_coordinate) < 0.14
            )
        parameter_scores = [
            whitespace_token_f1(parameter, candidates[peer].get("parameter"))
            for peer in same_action
            if parameter and candidates[peer].get("parameter")
        ]
        output[index, 80:85] = (
            len(same_action) / len(peers),
            float(np.mean(kernels)),
            max(kernels),
            float(np.mean(neighborhoods)),
            float(np.mean(parameter_scores)) if parameter_scores else 0.0,
        )
    return output


def restore_visual_values(prediction, count):
    permutation = [int(value) for value in prediction["display_to_candidate"]]
    if sorted(permutation) != list(range(count)):
        raise ValueError("TriVUS visual display permutation mismatch")
    display_logits = np.asarray(prediction["label_logits"], dtype=np.float64)
    display_probabilities = np.asarray(prediction["label_probabilities"], dtype=np.float64)
    if display_logits.shape != (count,) or display_probabilities.shape != (count,):
        raise ValueError("TriVUS visual width mismatch")
    if not np.isfinite(display_logits).all() or not np.isfinite(display_probabilities).all():
        raise ValueError("TriVUS visual values are non-finite")
    if np.any(display_probabilities < 0) or np.any(display_probabilities > 1) or not math.isclose(
        float(display_probabilities.sum()), 1.0, abs_tol=1e-6
    ):
        raise ValueError("TriVUS visual probabilities mismatch")
    logits = np.empty(count, dtype=np.float64)
    probabilities = np.empty(count, dtype=np.float64)
    for display_index, candidate_index in enumerate(permutation):
        logits[candidate_index] = display_logits[display_index]
        probabilities[candidate_index] = display_probabilities[display_index]
    return logits, probabilities


def visual_features(prediction, fallback_index, count):
    if not 0 <= fallback_index < count:
        raise ValueError("TriVUS visual fallback out of range")
    logits, probabilities = restore_visual_values(prediction, count)
    centered = logits - float(logits.mean())
    log_probabilities = np.log(np.maximum(probabilities, 1e-12))
    order = np.argsort(-probabilities, kind="stable")
    ranks = np.empty(count, dtype=np.float64)
    ranks[order] = np.arange(count, dtype=np.float64) / (count - 1)
    entropy = -float(np.sum(probabilities * log_probabilities)) / math.log(count)
    return np.column_stack((
        centered,
        log_probabilities,
        probabilities,
        ranks,
        np.full(count, entropy),
        centered - centered[fallback_index],
        probabilities - probabilities[fallback_index],
    )).astype(np.float32)


def base_features(candidates, prediction, fallback_index, family, cell):
    count = len(candidates)
    if family not in FAMILIES or cell not in CELLS:
        raise ValueError(f"TriVUS family/cell mismatch: {family}/{cell}")
    if family == "androidcontrol" and cell not in {"low", "high"}:
        raise ValueError("TriVUS Android cell mismatch")
    if family != "androidcontrol" and cell not in CELLS[:4]:
        raise ValueError("TriVUS VUS cell mismatch")
    features = np.zeros((MAX_CANDIDATES, INPUT_DIMENSION), dtype=np.float32)
    features[:count, :85] = structural_features(candidates)
    features[:count, VISUAL_SLICE] = visual_features(prediction, fallback_index, count)
    features[fallback_index, 92] = 1.0
    features[:count, 93] = count / MAX_CANDIDATES
    features[:count, 94:97] = np.asarray([
        float(family == value) for value in FAMILIES
    ], dtype=np.float32)
    features[:count, 97:103] = np.asarray([
        float(cell == value) for value in CELLS
    ], dtype=np.float32)
    return features


def pseudo_identity_permutation(context_key, count):
    digest = hashlib.sha256(f"{context_key}/20260812/random-id-placebo".encode()).digest()
    generator = np.random.default_rng(int.from_bytes(digest[:8], "big"))
    return generator.permutation(count)


def validate_prefix_mask(candidate_mask):
    mask = np.asarray(candidate_mask)
    if mask.dtype != np.bool_:
        raise ValueError("TriVUS candidate mask must be boolean")
    if mask.ndim != 2 or mask.shape[1] != MAX_CANDIDATES:
        raise ValueError("TriVUS candidate-mask shape mismatch")
    counts = mask.sum(axis=1)
    if not np.all((counts == 3) | (counts == 12)):
        raise ValueError("TriVUS valid candidate count must be 3 or 12")
    for row, count in enumerate(counts):
        if not mask[row, :count].all() or mask[row, count:].any():
            raise ValueError("TriVUS raw candidate mask must be prefix-valid")
    return mask, counts.astype(np.int64)


def variant_features(features, candidate_mask, context_keys, variant):
    if variant not in VARIANTS:
        raise ValueError(f"TriVUS unknown variant: {variant}")
    values = np.asarray(features, dtype=np.float32).copy()
    mask, counts = validate_prefix_mask(candidate_mask)
    if values.shape != (len(context_keys), MAX_CANDIDATES, INPUT_DIMENSION) or mask.shape != values.shape[:2]:
        raise ValueError("TriVUS variant feature shape mismatch")
    values[:, :, PLACEBO_SLICE] = 0.0
    if variant == "NO_VISUAL":
        values[:, :, VISUAL_SLICE] = 0.0
    if variant == "RANDOM_ID_PLACEBO":
        for row, context_key in enumerate(context_keys):
            valid = int(counts[row])
            permutation = pseudo_identity_permutation(context_key, valid)
            values[row, np.arange(valid), 103 + permutation] = 1.0
    values[~mask] = 0.0
    return values


def target_values(labels, fallback_index):
    labels = np.asarray(labels, dtype=np.bool_)
    count = len(labels)
    if count not in (3, 12) or not 0 <= fallback_index < count:
        raise ValueError("TriVUS target contract mismatch")
    target = np.zeros(MAX_CANDIDATES + 1, dtype=np.float32)
    positives = np.flatnonzero(labels)
    if not labels[fallback_index] and len(positives):
        target[positives] = 1.0 / len(positives)
    else:
        target[MAX_CANDIDATES] = 1.0
    active = bool(np.any(labels != labels[fallback_index]))
    return target, float(labels[fallback_index]), active


def validate_trivus_data(data, included_families=None):
    rows = len(data)
    if rows < 1:
        raise ValueError("TriVUS data must contain at least one row")
    if data.features.shape != (rows, MAX_CANDIDATES, INPUT_DIMENSION):
        raise ValueError("TriVUS data feature shape mismatch")
    if not np.issubdtype(data.features.dtype, np.floating) or not np.isfinite(data.features).all():
        raise ValueError("TriVUS data features must be finite floats")
    mask, counts = validate_prefix_mask(data.candidate_mask)
    if np.any(data.features[~mask] != 0):
        raise ValueError("TriVUS raw padding features must be zero")
    if data.fallback_indices.shape != (rows,) or not np.issubdtype(data.fallback_indices.dtype, np.integer):
        raise ValueError("TriVUS fallback-index shape/type mismatch")
    if np.any(data.fallback_indices < 0) or np.any(data.fallback_indices >= counts):
        raise ValueError("TriVUS fallback points outside valid candidates")
    if data.target_distribution.shape != (rows, MAX_CANDIDATES + 1):
        raise ValueError("TriVUS target shape mismatch")
    if (
        not np.issubdtype(data.target_distribution.dtype, np.floating)
        or not np.isfinite(data.target_distribution).all()
        or np.any(data.target_distribution < 0)
        or not np.allclose(data.target_distribution.sum(axis=1), 1.0, atol=1e-6, rtol=0)
    ):
        raise ValueError("TriVUS target simplex mismatch")
    if np.any(data.target_distribution[:, :MAX_CANDIDATES][~mask] != 0):
        raise ValueError("TriVUS target assigns mass to padding")
    if data.labels.shape != (rows, MAX_CANDIDATES) or data.labels.dtype != np.bool_:
        raise ValueError("TriVUS label shape/type mismatch")
    if np.any(data.labels[~mask]):
        raise ValueError("TriVUS padding labels must be false")
    if data.fallback_correct.shape != (rows,) or not np.issubdtype(data.fallback_correct.dtype, np.floating):
        raise ValueError("TriVUS fallback-correct shape/type mismatch")
    if not np.isfinite(data.fallback_correct).all() or not np.all(
        (data.fallback_correct == 0) | (data.fallback_correct == 1)
    ):
        raise ValueError("TriVUS fallback-correct must be binary")
    if data.active.shape != (rows,) or data.active.dtype != np.bool_:
        raise ValueError("TriVUS active shape/type mismatch")
    if data.weights.shape != (rows,) or not np.issubdtype(data.weights.dtype, np.floating):
        raise ValueError("TriVUS weight shape/type mismatch")
    if not np.isfinite(data.weights).all() or np.any(data.weights < 0):
        raise ValueError("TriVUS weights must be finite and nonnegative")
    if np.any(data.weights[~data.active] != 0):
        raise ValueError("TriVUS inactive rows must have zero weight")
    metadata = (
        data.context_keys, data.sample_keys, data.families, data.cells,
        data.row_ids, data.groups,
    )
    if any(len(values) != rows for values in metadata) or data.folds.shape != (rows,):
        raise ValueError("TriVUS metadata length mismatch")
    if len(set(data.context_keys)) != rows or any(
        not isinstance(value, str) or not value for value in data.context_keys
    ):
        raise ValueError("TriVUS context keys must be unique nonempty strings")
    if any(
        not isinstance(value, str) or not value
        for values in (data.sample_keys, data.row_ids, data.groups)
        for value in values
    ):
        raise ValueError("TriVUS row metadata must be nonempty strings")
    if not np.issubdtype(data.folds.dtype, np.integer) or np.any(data.folds < 0) or np.any(data.folds >= 5):
        raise ValueError("TriVUS fold metadata mismatch")
    for family, cell in zip(data.families, data.cells):
        if family not in FAMILIES:
            raise ValueError(f"TriVUS unknown family: {family}")
        valid_cells = CELLS[4:] if family == "androidcontrol" else CELLS[:4]
        if cell not in valid_cells:
            raise ValueError(f"TriVUS invalid family/cell pair: {family}/{cell}")
    for row, count in enumerate(counts):
        expected_target, expected_fallback, expected_active = target_values(
            data.labels[row, :count], int(data.fallback_indices[row])
        )
        if not np.allclose(data.target_distribution[row], expected_target, atol=1e-7, rtol=0):
            raise ValueError(f"TriVUS target/label mismatch: {data.context_keys[row]}")
        if data.fallback_correct[row] != expected_fallback or bool(data.active[row]) != expected_active:
            raise ValueError(f"TriVUS fallback/activity mismatch: {data.context_keys[row]}")
    if included_families is not None:
        expected_weights = assign_weights(
            data.families, data.cells, data.active, included_families
        )
        if not np.allclose(data.weights, expected_weights, atol=1e-12, rtol=0):
            raise ValueError("TriVUS assigned weights mismatch")
    return True


def assign_weights(families, cells, active, included_families):
    families = tuple(families)
    cells = tuple(cells)
    active = np.asarray(active, dtype=np.bool_)
    included = tuple(included_families)
    if len(families) != len(cells) or active.shape != (len(families),):
        raise ValueError("TriVUS weight metadata mismatch")
    if not included or len(set(included)) != len(included) or any(value not in FAMILIES for value in included):
        raise ValueError("TriVUS included family mismatch")
    weights = np.zeros(len(families), dtype=np.float64)
    family_cells = {
        "mind2web": CELLS[:4],
        "screenspot_pro": CELLS[:4],
        "androidcontrol": CELLS[4:],
    }
    for family in included:
        required = family_cells[family]
        for cell in required:
            indices = [
                index for index, (row_family, row_cell, row_active) in enumerate(zip(families, cells, active))
                if row_family == family and row_cell == cell and row_active
            ]
            if not indices:
                raise ValueError(f"TriVUS empty active weight cell: {family}/{cell}")
            weights[indices] = 1.0 / len(required) / len(indices)
    if not math.isclose(float(weights.sum()), float(len(included)), abs_tol=1e-12):
        raise ValueError("TriVUS family weight sum mismatch")
    return weights


def fit_standardizer(data, variant, included_families=None):
    validate_trivus_data(data, included_families)
    features = variant_features(
        data.features, data.candidate_mask, data.context_keys, variant
    )
    selected = data.candidate_mask & data.active[:, None] & (data.weights[:, None] > 0)
    values = features[selected].astype(np.float64)
    if not len(values):
        raise ValueError("TriVUS standardizer has no active valid candidates")
    mean = values.mean(axis=0).astype(np.float32)
    scale = values.std(axis=0).astype(np.float32)
    scale[scale < 1e-6] = 1.0
    if mean.shape != (INPUT_DIMENSION,) or scale.shape != (INPUT_DIMENSION,):
        raise AssertionError("TriVUS standardizer width mismatch")
    return TriVUSStandardizer(mean=mean, scale=scale, variant=variant)


def torch_batch(data, indices, device):
    import torch

    from trivus_model import TriVUSBatch

    selected = data.subset(indices)
    validate_trivus_data(selected)
    return TriVUSBatch(
        features=torch.as_tensor(selected.features, dtype=torch.float32, device=device),
        candidate_mask=torch.as_tensor(selected.candidate_mask, dtype=torch.bool, device=device),
        fallback_indices=torch.as_tensor(selected.fallback_indices, dtype=torch.long, device=device),
        target_distribution=torch.as_tensor(selected.target_distribution, dtype=torch.float32, device=device),
        fallback_correct=torch.as_tensor(selected.fallback_correct, dtype=torch.float32, device=device),
        weights=torch.as_tensor(selected.weights, dtype=torch.float32, device=device),
    )


def stable_row_seed(sample_key, seed, epoch):
    payload = json.dumps((int(seed), int(epoch), sample_key), separators=(",", ":")).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")