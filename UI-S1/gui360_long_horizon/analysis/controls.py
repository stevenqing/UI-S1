"""Audit and shuffle-control utilities for long-horizon analysis."""

from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass, fields
from typing import Any, Callable, Dict, Mapping, Optional, Sequence


@dataclass(frozen=True)
class DependencyGateThresholds:
    """Pre-registered constants for the cross-step dependency gate."""

    min_value_len: int = 4
    min_unique_chars: int = 3
    min_alpha_numeric_chars: int = 3
    null_model_shuffles: int = 128
    null_model_seed: int = 41
    null_model_margin: float = 0.02
    routine_entropy_floor: float = 0.35
    routine_min_support: int = 3
    legal_forced_max_actions: int = 1
    long_horizon_min_distance: int = 3
    no_interference_min_distance: int = 3
    persistent_visible_fraction: float = 0.50
    missing_ocr_is_available: bool = True
    q1_battlefield_share_min: float = 0.15
    q1_no_battlefield_share_max: float = 0.10
    q2_min_distance_ge3_n: int = 30
    q3_failure_lift_eps: float = 0.0
    q3_min_memory_44_fraction: float = 0.01


DEFAULT_DEPENDENCY_THRESHOLDS = DependencyGateThresholds()


class ThresholdFreezeError(RuntimeError):
    """Raised when a diagnostic run tries to alter frozen gate thresholds."""


def thresholds_to_dict(thresholds: DependencyGateThresholds = DEFAULT_DEPENDENCY_THRESHOLDS) -> Dict[str, Any]:
    return asdict(thresholds)


def thresholds_from_dict(data: Mapping[str, Any]) -> DependencyGateThresholds:
    allowed = {item.name for item in fields(DependencyGateThresholds)}
    unknown = sorted(set(data) - allowed)
    if unknown:
        raise ThresholdFreezeError(f"unknown dependency gate threshold keys: {unknown}")
    values = thresholds_to_dict()
    values.update(dict(data))
    return DependencyGateThresholds(**values)


def assert_thresholds_frozen(candidate: Mapping[str, Any] | DependencyGateThresholds, reference: DependencyGateThresholds = DEFAULT_DEPENDENCY_THRESHOLDS) -> DependencyGateThresholds:
    """Reject threshold overrides so the gate cannot be loosened after inspection."""

    loaded = candidate if isinstance(candidate, DependencyGateThresholds) else thresholds_from_dict(candidate)
    if thresholds_to_dict(loaded) != thresholds_to_dict(reference):
        raise ThresholdFreezeError("dependency gate thresholds are pre-registered and read-only at run time")
    return loaded


@dataclass(frozen=True)
class NullModelResult:
    actual_reuse_rate: float
    null_mean_reuse_rate: float
    null_p95_reuse_rate: float
    shuffles: int
    passed: bool


def null_model_reuse_test(
    produced_by_episode: Mapping[str, Sequence[str]],
    consumed_by_episode: Mapping[str, Sequence[str]],
    *,
    shuffles: int = DEFAULT_DEPENDENCY_THRESHOLDS.null_model_shuffles,
    seed: int = DEFAULT_DEPENDENCY_THRESHOLDS.null_model_seed,
    margin: float = DEFAULT_DEPENDENCY_THRESHOLDS.null_model_margin,
) -> NullModelResult:
    """Shuffle produced values across episodes and estimate chance reuse.

    The statistic is intentionally simple: for each episode, count whether any
    produced value appears among later consumed values. The gate only treats a
    reuse signal as real if the observed rate exceeds the shuffled p95 plus the
    frozen margin.
    """

    episode_ids = sorted(set(produced_by_episode) | set(consumed_by_episode))
    if not episode_ids:
        return NullModelResult(0.0, 0.0, 0.0, shuffles, False)

    def reuse_rate(produced: Mapping[str, Sequence[str]]) -> float:
        hits = 0
        eligible = 0
        for episode_id in episode_ids:
            produced_values = {str(value) for value in produced.get(episode_id, ()) if value}
            consumed_values = {str(value) for value in consumed_by_episode.get(episode_id, ()) if value}
            if not produced_values or not consumed_values:
                continue
            eligible += 1
            if produced_values & consumed_values:
                hits += 1
        return float(hits / eligible) if eligible else 0.0

    eligible = 0
    for episode_id in episode_ids:
        produced_values = {str(value) for value in produced_by_episode.get(episode_id, ()) if value}
        consumed_values = {str(value) for value in consumed_by_episode.get(episode_id, ()) if value}
        if produced_values and consumed_values:
            eligible += 1
    actual = reuse_rate(produced_by_episode)
    if eligible < 2:
        return NullModelResult(actual, 0.0, 0.0, shuffles, True)
    rng = random.Random(seed)
    produced_lists = [list(produced_by_episode.get(episode_id, ())) for episode_id in episode_ids]
    null_rates = []
    for _ in range(max(1, int(shuffles))):
        shuffled = produced_lists[:]
        rng.shuffle(shuffled)
        shuffled_map = {episode_id: values for episode_id, values in zip(episode_ids, shuffled)}
        null_rates.append(reuse_rate(shuffled_map))
    null_rates.sort()
    p95_index = min(len(null_rates) - 1, max(0, math.ceil(0.95 * len(null_rates)) - 1))
    null_mean = float(sum(null_rates) / len(null_rates)) if null_rates else 0.0
    null_p95 = float(null_rates[p95_index]) if null_rates else 0.0
    return NullModelResult(actual, null_mean, null_p95, int(shuffles), bool(actual > null_p95 + margin))


def shuffle_test(experiment_fn: Callable[[Sequence[Any]], float], label: Sequence[Any]) -> bool:
    """Return True when a shuffled label control collapses the effect to chance."""

    labels = list(label)
    if not labels:
        raise ValueError("shuffle_test requires non-empty labels")
    original = abs(float(experiment_fn(labels)))
    rng = random.Random(41)
    shuffled = labels[:]
    rng.shuffle(shuffled)
    control = abs(float(experiment_fn(shuffled)))
    return control <= max(1e-12, original * 0.25)


def run_audits(*, divergence_bundle: Optional[Any] = None, recovery_steps: Optional[Sequence[Any]] = None, recovery_labels: Optional[Sequence[bool]] = None, recovery_precision_min: float = 0.90) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if divergence_bundle is not None:
        divergence_bundle.assert_reliable()
        out["divergence"] = {"n": len(divergence_bundle.items), "passed": True}
    if recovery_steps is not None and recovery_labels is not None:
        from gui360_long_horizon.recovery_oracle import audit_precision

        precision = audit_precision(zip(recovery_steps, recovery_labels), precision_min=recovery_precision_min)
        out["recovery"] = {"precision": precision, "passed": precision >= recovery_precision_min}
    return out
