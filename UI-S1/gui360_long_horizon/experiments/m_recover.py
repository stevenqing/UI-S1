"""Generic recovery experiment over oracle-labeled off-trajectory screens."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

from gui360_long_horizon.data.divergence import delta as delta_for_step
from gui360_long_horizon.harness.predict import Pred, query_step
from gui360_long_horizon.recovery_oracle import RecoveryTarget, recovery_oracle
from gui360_long_horizon.types import Row, row_from_step

from .common import BucketFn, bucket_lookup, model_label


@dataclass(frozen=True)
class RecoverResult:
    rows: List[Row]


def _matches(pred: Pred, target: RecoveryTarget, tol: float = 32.0) -> bool:
    action = target.correct_action
    if action.function == "click":
        if pred.function != "click" or pred.xy is None or action.xy is None:
            return False
        return math.hypot(pred.xy[0] - action.xy[0], pred.xy[1] - action.xy[1]) <= tol
    if action.function == "press_key":
        text = (pred.function + " " + pred.text).lower()
        return str(action.key or "").lower() in text or pred.recovery_class == "recover"
    return False


def run(off_steps: Iterable[Any], model: Any, *, t_star_by_exec: Optional[Dict[str, int]] = None, bucket_fn: Optional[BucketFn] = None, history_mode: str = "full", input_mode: str = "visual", n: int = 8) -> RecoverResult:
    rows: List[Row] = []
    label = model_label(model)
    t_star_by_exec = t_star_by_exec or {}
    for step in off_steps:
        target = recovery_oracle(step)
        if target is None:
            continue
        pred = query_step(model, step, history_mode, input_mode, n=n)
        rows.append(
            row_from_step(
                step,
                manifold="off",
                delta_value=delta_for_step(step, t_star_by_exec.get(getattr(step, "exec_id", ""))),
                d_bucket=bucket_lookup(step, bucket_fn),
                cond={"history_mode": history_mode, "input_mode": input_mode, "plan": None, "injected_error": 0, "target_kind": target.kind},
                step_correct_value=None,
                recovery_correct=_matches(pred, target),
                top1_conf=pred.top1_conf,
                action_entropy=pred.action_entropy,
                recovery_class=pred.recovery_class,
                self_offtrack=pred.self_offtrack,
                model=label,
            )
        )
    return RecoverResult(rows=rows)
