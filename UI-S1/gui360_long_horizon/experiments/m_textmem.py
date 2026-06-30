"""Text-memory gate experiment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, List, Optional, Sequence

from gui360_long_horizon.harness.correctness import step_correct
from gui360_long_horizon.harness.predict import query_step
from gui360_long_horizon.types import Row, row_from_step

from .common import BucketFn, bucket_lookup, model_label


HISTORY_MODES = ("full", "summary", "corrupt", "none")


@dataclass(frozen=True)
class TextMemResult:
    rows: List[Row]
    acc_by_mode: dict
    gate_eps: float
    gate_passed: bool


def _as_fraction(eps: float) -> float:
    return eps / 100.0 if eps > 0.5 else eps


def _accuracy(rows: Sequence[Row], mode: str) -> float:
    values = [row.step_correct for row in rows if row.cond.get("history_mode") == mode and row.step_correct is not None]
    return sum(bool(value) for value in values) / len(values) if values else 0.0


def run(steps: Iterable[Any], model: Any, *, bucket_fn: Optional[BucketFn] = None, input_mode: str = "visual", gate_eps: float = 0.01, n: int = 8) -> TextMemResult:
    rows: List[Row] = []
    label = model_label(model)
    for step in steps:
        for history_mode in HISTORY_MODES:
            pred = query_step(model, step, history_mode, input_mode, n=n)
            correct = step_correct(pred, step)
            rows.append(
                row_from_step(
                    step,
                    manifold="on",
                    d_bucket=bucket_lookup(step, bucket_fn),
                    cond={"history_mode": history_mode, "input_mode": input_mode, "plan": None, "injected_error": 0},
                    step_correct_value=correct,
                    top1_conf=pred.top1_conf,
                    action_entropy=pred.action_entropy,
                    recovery_class=pred.recovery_class,
                    self_offtrack=pred.self_offtrack,
                    model=label,
                )
            )
    acc = {mode: _accuracy(rows, mode) for mode in HISTORY_MODES}
    eps = _as_fraction(gate_eps)
    gate_passed = abs(acc.get("full", 0.0) - acc.get("none", 0.0)) <= eps
    return TextMemResult(rows=rows, acc_by_mode=acc, gate_eps=eps, gate_passed=gate_passed)
