"""Demoted observational diagnostic bound for difficulty-matched on/off rows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

from gui360_long_horizon.data.divergence import delta as delta_for_step
from gui360_long_horizon.harness.predict import query_step
from gui360_long_horizon.types import Row, depth_fraction, row_from_step, task_key_for_step

from .common import BucketFn, bucket_lookup, model_label


INTERPRETATION = "upper bound = real_visual_drift + selection_on_model_specific_difficulty"


@dataclass(frozen=True)
class DiagResult:
    rows: List[Row]
    identifies_drift: bool = False
    interpretation: str = INTERPRETATION

    @property
    def upper_bound(self) -> float:
        on = [row.self_offtrack for row in self.rows if row.manifold == "on"]
        off = [row.self_offtrack for row in self.rows if row.manifold == "off"]
        if not on or not off:
            return 0.0
        return (sum(off) / len(off)) - (sum(on) / len(on))

    @property
    def drift_effect(self) -> float:
        raise RuntimeError("m_diag is non-identifying; use upper_bound only")


def _match_pairs(on_steps: Iterable[Any], off_steps: Iterable[Any], bucket_fn: Optional[BucketFn], caliper: float) -> List[Tuple[Any, Any]]:
    on_list = list(on_steps)
    pairs: List[Tuple[Any, Any]] = []
    used_on: set[int] = set()
    for off in off_steps:
        off_key = task_key_for_step(off)
        off_bucket = bucket_lookup(off, bucket_fn)
        off_depth = depth_fraction(off)
        candidates = [
            (abs(depth_fraction(on) - off_depth), idx, on)
            for idx, on in enumerate(on_list)
            if idx not in used_on and task_key_for_step(on) == off_key and bucket_lookup(on, bucket_fn) == off_bucket and abs(depth_fraction(on) - off_depth) <= caliper
        ]
        if not candidates:
            continue
        _, idx, on = min(candidates, key=lambda item: item[0])
        used_on.add(idx)
        pairs.append((on, off))
    return pairs


def run(on_steps: Iterable[Any], off_steps: Iterable[Any], model: Any, *, t_star_by_exec: Optional[Dict[str, int]] = None, bucket_fn: Optional[BucketFn] = None, history_mode: str = "full", input_mode: str = "visual", caliper: float = 0.10, n: int = 8) -> DiagResult:
    rows: List[Row] = []
    label = model_label(model)
    t_star_by_exec = t_star_by_exec or {}
    for pair_id, (on, off) in enumerate(_match_pairs(on_steps, off_steps, bucket_fn, caliper)):
        for manifold, step in (("on", on), ("off", off)):
            pred = query_step(model, step, history_mode, input_mode, n=n)
            rows.append(
                row_from_step(
                    step,
                    manifold=manifold,
                    delta_value=None if manifold == "on" else delta_for_step(step, t_star_by_exec.get(getattr(step, "exec_id", ""))),
                    d_bucket=bucket_lookup(step, bucket_fn),
                    cond={"pair_id": pair_id, "history_mode": history_mode, "input_mode": input_mode, "plan": None, "injected_error": 0},
                    step_correct_value=None,
                    recovery_correct=None,
                    top1_conf=pred.top1_conf,
                    action_entropy=pred.action_entropy,
                    recovery_class=pred.recovery_class,
                    self_offtrack=pred.self_offtrack,
                    model=label,
                )
            )
    return DiagResult(rows=rows)
