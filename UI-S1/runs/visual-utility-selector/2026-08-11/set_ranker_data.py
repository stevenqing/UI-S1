import json
import math
import sys
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import torch


RUN_DIR = Path(__file__).resolve().parent
ROOT = RUN_DIR.parents[2]
UTILITY_DIR = ROOT / "runs/lsa-utility/2026-08-11"
sys.path.insert(0, str(UTILITY_DIR))

from behavior_policy import apply_policy
from utility_common import ARMS, BENCHMARKS, transformed_features, utility_targets
from set_ranker_model import RankerBatch


ACTION_CATEGORIES = ("POINT", "CLICK", "TYPE", "SELECT", "OTHER")


@dataclass(frozen=True)
class SetData:
    features: np.ndarray
    fallback_indices: np.ndarray
    target_distribution: np.ndarray
    fallback_correct: np.ndarray
    grpo_advantage: np.ndarray
    weights: np.ndarray
    active: np.ndarray
    labels: np.ndarray
    sample_keys: tuple[str, ...]
    benchmarks: tuple[str, ...]
    arms: tuple[str, ...]
    row_ids: tuple[str, ...]
    folds: np.ndarray
    groups: tuple[str, ...]

    def __len__(self):
        return len(self.sample_keys)

    def subset(self, mask):
        indices = np.flatnonzero(np.asarray(mask, dtype=np.bool_))
        return SetData(
            features=self.features[indices],
            fallback_indices=self.fallback_indices[indices],
            target_distribution=self.target_distribution[indices],
            fallback_correct=self.fallback_correct[indices],
            grpo_advantage=self.grpo_advantage[indices],
            weights=self.weights[indices],
            active=self.active[indices],
            labels=self.labels[indices],
            sample_keys=tuple(self.sample_keys[index] for index in indices),
            benchmarks=tuple(self.benchmarks[index] for index in indices),
            arms=tuple(self.arms[index] for index in indices),
            row_ids=tuple(self.row_ids[index] for index in indices),
            folds=self.folds[indices],
            groups=tuple(self.groups[index] for index in indices),
        )


@dataclass(frozen=True)
class Standardizer:
    mean: np.ndarray
    scale: np.ndarray

    def transform(self, data):
        values = (data.features - self.mean[None, None, :]) / self.scale[None, None, :]
        return replace(data, features=values.astype(np.float32))


def load_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def keyed(path):
    rows = load_jsonl(path)
    output = {row["sample_key"]: row for row in rows}
    if len(output) != len(rows):
        raise ValueError(f"duplicate sample keys: {path}")
    return output


def load_public_predictions(
    public_path=RUN_DIR / "data/public_records.jsonl",
    predictions_path=RUN_DIR / "zero_shot/predictions.jsonl",
):
    public = keyed(public_path)
    predictions = keyed(predictions_path)
    if set(public) != set(predictions):
        raise ValueError("VUS-SR public/prediction coverage mismatch")
    return public, predictions


def load_label_folds(folds, label_dir=RUN_DIR / "data"):
    folds = tuple(sorted(set(int(fold) for fold in folds)))
    if not folds or any(not 0 <= fold < 5 for fold in folds):
        raise ValueError(f"invalid private label folds: {folds}")
    rows = []
    for fold in folds:
        path = Path(label_dir) / f"private_labels_fold-{fold}.jsonl"
        if not path.is_file():
            raise FileNotFoundError(path)
        rows.extend(load_jsonl(path))
    output = {row["sample_key"]: row for row in rows}
    if len(output) != len(rows):
        raise ValueError(f"duplicate keys across private label folds: {folds}")
    return output


def original_visual_values(prediction):
    permutation = prediction["display_to_candidate"]
    if sorted(permutation) != list(range(12)):
        raise ValueError(f"invalid visual permutation: {prediction['sample_key']}")
    display_logits = np.asarray(prediction["label_logits"], dtype=np.float32)
    display_probabilities = np.asarray(prediction["label_probabilities"], dtype=np.float32)
    if display_logits.shape != (12,) or display_probabilities.shape != (12,):
        raise ValueError(f"visual width mismatch: {prediction['sample_key']}")
    logits = np.empty(12, dtype=np.float32)
    probabilities = np.empty(12, dtype=np.float32)
    for display_index, candidate_index in enumerate(permutation):
        logits[candidate_index] = display_logits[display_index]
        probabilities[candidate_index] = display_probabilities[display_index]
    return logits, probabilities


def visual_features(prediction, fallback_index):
    logits, probabilities = original_visual_values(prediction)
    centered = logits - float(np.mean(logits))
    log_probabilities = np.log(np.maximum(probabilities, 1e-12))
    order = np.argsort(-probabilities, kind="stable")
    ranks = np.empty(12, dtype=np.float32)
    ranks[order] = np.arange(12, dtype=np.float32) / 11.0
    entropy = -float(np.sum(probabilities * log_probabilities))
    fallback_flag = np.zeros(12, dtype=np.float32)
    fallback_flag[fallback_index] = 1.0
    return np.column_stack((
        centered,
        log_probabilities,
        probabilities,
        ranks,
        centered - centered[fallback_index],
        probabilities - probabilities[fallback_index],
        np.full(12, entropy, dtype=np.float32),
        fallback_flag,
    )).astype(np.float32)


def categorical_features(row):
    output = []
    for candidate in row.candidates:
        action = candidate.action if candidate.action in ACTION_CATEGORIES else "OTHER"
        output.append([float(action == category) for category in ACTION_CATEGORIES])
    return np.asarray(output, dtype=np.float32)


def context_features(row, structural, visual, arm):
    benchmark_values = np.repeat(
        np.asarray([[float(row.benchmark == benchmark) for benchmark in BENCHMARKS]], dtype=np.float32),
        12, axis=0,
    )
    arm_values = np.repeat(
        np.asarray([[float(arm == value) for value in ARMS]], dtype=np.float32),
        12, axis=0,
    )
    return np.concatenate((structural, visual, categorical_features(row), benchmark_values, arm_values), axis=1)


def target_values(row, fallback_index):
    labels = np.asarray([candidate.success for candidate in row.candidates], dtype=np.bool_)
    utility, advantage = utility_targets(row, fallback_index, "U_GRPO")
    positive = np.flatnonzero(utility > 0)
    target = np.zeros(13, dtype=np.float32)
    if len(positive):
        target[positive] = 1.0 / len(positive)
    else:
        target[12] = 1.0
    grpo = np.zeros(13, dtype=np.float32)
    grpo[:12] = advantage
    return labels, utility, target, grpo


def assign_weights(benchmarks, row_ids, arms, active):
    weights = np.zeros(len(active), dtype=np.float64)
    for benchmark in BENCHMARKS:
        active_rows = sorted({
            row_id for row_id, value, name in zip(row_ids, active, benchmarks)
            if name == benchmark and value
        })
        if not active_rows:
            continue
        row_mass = 1.0 / len(active_rows)
        for row_id in active_rows:
            indices = [
                index for index, (name, value, candidate_row) in enumerate(zip(benchmarks, active, row_ids))
                if name == benchmark and value and candidate_row == row_id
            ]
            arm_names = [arms[index] for index in indices]
            if len(indices) != len(set(arm_names)):
                raise ValueError(f"duplicate active arm group: {benchmark}/{row_id}")
            for index in indices:
                weights[index] = row_mass / len(indices)
    return weights


def build_set_data(
    banks,
    ids_by_benchmark,
    reliability,
    policies,
    public,
    predictions,
    labels_by_key,
    leave_one_ids=None,
):
    leave_one_ids = leave_one_ids or {benchmark: set() for benchmark in BENCHMARKS}
    arrays = []
    fallbacks = []
    targets = []
    fallback_correct = []
    advantages = []
    active = []
    labels = []
    sample_keys = []
    benchmarks = []
    arms = []
    row_ids = []
    folds = []
    groups = []
    for benchmark in BENCHMARKS:
        for row_id in ids_by_benchmark[benchmark]:
            for arm in ARMS:
                row = banks[arm][benchmark][row_id]
                sample_key = f"{benchmark}/{arm}/{row_id}"
                if sample_key not in public or sample_key not in predictions or sample_key not in labels_by_key:
                    raise KeyError(sample_key)
                sums, counts = reliability[arm][benchmark]
                fallback = apply_policy(row, policies[benchmark][arm])
                leave_one = row_id in leave_one_ids.get(benchmark, set())
                structural = transformed_features(row, sums, counts, fallback, leave_one, "pair")
                visual = visual_features(predictions[sample_key], fallback)
                row_labels, utility, target, advantage = target_values(row, fallback)
                private_labels = np.asarray(labels_by_key[sample_key]["candidate_success"], dtype=np.bool_)
                if not np.array_equal(row_labels, private_labels):
                    raise ValueError(f"private label mismatch: {sample_key}")
                arrays.append(context_features(row, structural, visual, arm))
                fallbacks.append(fallback)
                targets.append(target)
                fallback_correct.append(float(row_labels[fallback]))
                advantages.append(advantage)
                active.append(bool(np.std(utility) > 0))
                labels.append(row_labels)
                sample_keys.append(sample_key)
                benchmarks.append(benchmark)
                arms.append(arm)
                row_ids.append(row_id)
                folds.append(row.fold)
                groups.append(row.group)
    active_values = np.asarray(active, dtype=np.bool_)
    weights = assign_weights(benchmarks, row_ids, arms, active_values)
    return SetData(
        features=np.stack(arrays).astype(np.float32),
        fallback_indices=np.asarray(fallbacks, dtype=np.int64),
        target_distribution=np.stack(targets).astype(np.float32),
        fallback_correct=np.asarray(fallback_correct, dtype=np.float32),
        grpo_advantage=np.stack(advantages).astype(np.float32),
        weights=weights,
        active=active_values,
        labels=np.stack(labels),
        sample_keys=tuple(sample_keys),
        benchmarks=tuple(benchmarks),
        arms=tuple(arms),
        row_ids=tuple(row_ids),
        folds=np.asarray(folds, dtype=np.int8),
        groups=tuple(groups),
    )


def fit_standardizer(data):
    selected = data.features[data.active]
    if not len(selected):
        raise ValueError("no active VUS-SR rows")
    flat = selected.reshape(-1, selected.shape[-1]).astype(np.float64)
    mean = flat.mean(axis=0).astype(np.float32)
    scale = flat.std(axis=0).astype(np.float32)
    scale[scale < 1e-6] = 1.0
    return Standardizer(mean, scale)


def torch_batch(data, indices, device):
    return RankerBatch(
        features=torch.as_tensor(data.features[indices], device=device),
        fallback_indices=torch.as_tensor(data.fallback_indices[indices], device=device),
        target_distribution=torch.as_tensor(data.target_distribution[indices], device=device),
        fallback_correct=torch.as_tensor(data.fallback_correct[indices], device=device),
        grpo_advantage=torch.as_tensor(data.grpo_advantage[indices], device=device),
        weights=torch.as_tensor(data.weights[indices], dtype=torch.float32, device=device),
    )


def deterministic_epoch_permutations(sample_keys, epoch, seed):
    values = []
    for sample_key in sample_keys:
        payload = json.dumps((sample_key, int(epoch), int(seed)), separators=(",", ":")).encode()
        derived = int.from_bytes(__import__("hashlib").sha256(payload).digest()[:8], "big")
        values.append(np.random.default_rng(derived).permutation(12))
    return np.stack(values).astype(np.int64)


def feature_dimension():
    return None
