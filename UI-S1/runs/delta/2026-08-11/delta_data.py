import hashlib
import json
import sys
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import torch


RUN_DIR = Path(__file__).resolve().parent
ROOT = RUN_DIR.parents[2]
VUS = ROOT / "runs/visual-utility-selector/2026-08-11"
UTILITY = ROOT / "runs/lsa-utility/2026-08-11"
sys.path.insert(0, str(VUS))
sys.path.insert(0, str(UTILITY))

from behavior_policy import apply_policy
from utility_common import ARMS, BENCHMARKS, transformed_features
from set_ranker_data import assign_weights, categorical_features, target_values
from delta_model import CHANNELS, DeltaBatch


CHANNEL_PATHS = {
    "vus_binding": VUS / "zero_shot/predictions.jsonl",
    "global_semantic": ROOT / "runs/ravel/2026-08-11/evidence/global_only/predictions.jsonl",
    "fine_local": ROOT / "runs/ravel/2026-08-11/evidence/fine_only/predictions.jsonl",
    "context_local": ROOT / "runs/ravel/2026-08-11/evidence/context_only/predictions.jsonl",
    "random_placebo": ROOT / "runs/ravel/2026-08-11/evidence/random/predictions.jsonl",
}


@dataclass(frozen=True)
class DeltaData:
    base_features: np.ndarray
    channel_features: np.ndarray
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


@dataclass(frozen=True)
class Standardizer:
    base_mean: np.ndarray
    base_scale: np.ndarray
    channel_mean: np.ndarray
    channel_scale: np.ndarray

    def transform(self, data):
        base = (data.base_features - self.base_mean[None, None, :]) / self.base_scale[None, None, :]
        channels = (
            data.channel_features - self.channel_mean[None, None, None, :]
        ) / self.channel_scale[None, None, None, :]
        return replace(data, base_features=base.astype(np.float32), channel_features=channels.astype(np.float32))


def load_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def keyed(path):
    rows = load_jsonl(path)
    output = {row["sample_key"]: row for row in rows}
    if len(output) != len(rows):
        raise ValueError(f"duplicate DELTA keys: {path}")
    return output


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_channels(config):
    output = {}
    for name in CHANNELS:
        path = CHANNEL_PATHS[name]
        expected = config["channels"][name]["sha256"]
        observed = sha256_file(path)
        if observed != expected:
            raise ValueError(f"DELTA-K1 channel hash mismatch: {name}/{observed}/{expected}")
        output[name] = keyed(path)
    identities = [set(values) for values in output.values()]
    if any(values != identities[0] for values in identities[1:]):
        raise ValueError("DELTA-K1 channel identity mismatch")
    return output


def original_channel_features(row, fallback_index):
    logits = np.empty(12, dtype=np.float32)
    probabilities = np.empty(12, dtype=np.float32)
    for display_index, candidate_index in enumerate(row["display_to_candidate"]):
        logits[candidate_index] = row["label_logits"][display_index]
        probabilities[candidate_index] = row["label_probabilities"][display_index]
    centered = logits - float(logits.mean())
    ranks = np.empty(12, dtype=np.float32)
    order = np.argsort(-probabilities, kind="stable")
    ranks[order] = np.arange(12, dtype=np.float32) / 11.0
    entropy = -float(np.sum(probabilities * np.log(np.maximum(probabilities, 1e-12))))
    return np.column_stack((
        centered,
        np.log(np.maximum(probabilities, 1e-12)),
        probabilities,
        ranks,
        np.full(12, entropy, dtype=np.float32),
        centered - centered[fallback_index],
        probabilities - probabilities[fallback_index],
    )).astype(np.float32)


def build_delta_data(
    banks,
    ids_by_benchmark,
    reliability,
    policies,
    public,
    channels,
    labels_by_key,
    leave_one_ids=None,
):
    leave_one_ids = leave_one_ids or {benchmark: set() for benchmark in BENCHMARKS}
    base_arrays = []
    channel_arrays = []
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
                if sample_key not in public or sample_key not in labels_by_key:
                    raise KeyError(sample_key)
                sums, counts = reliability[arm][benchmark]
                fallback = apply_policy(row, policies[benchmark][arm])
                structural = transformed_features(
                    row, sums, counts, fallback,
                    row_id in leave_one_ids.get(benchmark, set()), "pair",
                )
                fallback_flag = np.zeros((12, 1), dtype=np.float32)
                fallback_flag[fallback] = 1.0
                benchmark_values = np.repeat(np.asarray([[float(benchmark == value) for value in BENCHMARKS]], dtype=np.float32), 12, axis=0)
                arm_values = np.repeat(np.asarray([[float(arm == value) for value in ARMS]], dtype=np.float32), 12, axis=0)
                base = np.concatenate((
                    structural, categorical_features(row), benchmark_values, arm_values, fallback_flag,
                ), axis=1)
                channel = np.stack([
                    original_channel_features(channels[name][sample_key], fallback)
                    for name in CHANNELS
                ], axis=1)
                row_labels, utility, target, advantage = target_values(row, fallback)
                private = np.asarray(labels_by_key[sample_key]["candidate_success"], dtype=np.bool_)
                if not np.array_equal(row_labels, private):
                    raise ValueError(f"DELTA private-label mismatch: {sample_key}")
                base_arrays.append(base)
                channel_arrays.append(channel)
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
    return DeltaData(
        base_features=np.stack(base_arrays).astype(np.float32),
        channel_features=np.stack(channel_arrays).astype(np.float32),
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
    selected = data.active & (data.weights > 0)
    if not selected.any():
        raise ValueError("DELTA has no active training rows")
    base = data.base_features[selected].reshape(-1, data.base_features.shape[-1]).astype(np.float64)
    channels = data.channel_features[selected].reshape(-1, data.channel_features.shape[-1]).astype(np.float64)
    base_mean = base.mean(axis=0).astype(np.float32)
    base_scale = base.std(axis=0).astype(np.float32)
    channel_mean = channels.mean(axis=0).astype(np.float32)
    channel_scale = channels.std(axis=0).astype(np.float32)
    base_scale[base_scale < 1e-6] = 1.0
    channel_scale[channel_scale < 1e-6] = 1.0
    return Standardizer(base_mean, base_scale, channel_mean, channel_scale)


def torch_batch(data, indices, device):
    return DeltaBatch(
        base_features=torch.as_tensor(data.base_features[indices], device=device),
        channel_features=torch.as_tensor(data.channel_features[indices], device=device),
        fallback_indices=torch.as_tensor(data.fallback_indices[indices], device=device),
        target_distribution=torch.as_tensor(data.target_distribution[indices], device=device),
        fallback_correct=torch.as_tensor(data.fallback_correct[indices], device=device),
        grpo_advantage=torch.as_tensor(data.grpo_advantage[indices], device=device),
        weights=torch.as_tensor(data.weights[indices], dtype=torch.float32, device=device),
    )


def deterministic_permutations(sample_keys, epoch, seed, suffix):
    values = []
    for sample_key in sample_keys:
        digest = hashlib.sha256(f"{sample_key}/{epoch}/{seed}/{suffix}".encode()).digest()
        values.append(np.random.default_rng(int.from_bytes(digest[:8], "big")).permutation(12))
    return np.stack(values).astype(np.int64)
