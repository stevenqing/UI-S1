"""Oracle-plan paired experiment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, List, Optional, Sequence

from gui360_long_horizon.harness.correctness import step_correct
from gui360_long_horizon.harness.predict import query_step
from gui360_long_horizon.types import Row, row_from_step

from .common import BucketFn, bucket_lookup, model_label, oracle_plan


CAVEAT = "value of correct global decomposition (horizon construct), not single-step"


@dataclass(frozen=True)
class PlanResult:
    rows: List[Row]
    caveat: str = CAVEAT


def run(trajectories: Iterable[Sequence[Any]], model: Any, *, bucket_fn: Optional[BucketFn] = None, history_mode: str = "none", input_mode: str = "visual", n: int = 8) -> PlanResult:
    rows: List[Row] = []
    label = model_label(model)
    for traj in trajectories:
        plan_text = oracle_plan(traj)
        for step in traj:
            for plan_name, plan in (("none", None), ("oracle", plan_text)):
                pred = query_step(model, step, history_mode, input_mode, plan=plan, n=n)
                rows.append(
                    row_from_step(
                        step,
                        manifold="on",
                        d_bucket=bucket_lookup(step, bucket_fn),
                        cond={"history_mode": history_mode, "input_mode": input_mode, "plan": plan_name, "injected_error": 0},
                        step_correct_value=step_correct(pred, step),
                        top1_conf=pred.top1_conf,
                        action_entropy=pred.action_entropy,
                        recovery_class=pred.recovery_class,
                        self_offtrack=pred.self_offtrack,
                        model=label,
                    )
                )
    return PlanResult(rows=rows)
