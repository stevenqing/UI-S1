"""Core existence experiment runner for invariance decomposition."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence

from gui360_long_horizon.harness.correctness import step_correct
from gui360_long_horizon.harness.predict import query_step
from gui360_long_horizon.types import Row, row_from_step

from .common import BucketFn, bucket_lookup, model_label, oracle_plan, with_raw_updates


BLOCKS: Dict[str, bool] = {
    "base": True,
    "position": False,
    "history": True,
    "plan": True,
    "injected_error": True,
}


@dataclass(frozen=True)
class CoreResult:
    rows: List[Row]
    blocks: Dict[str, bool]


def run(trajectories: Iterable[Sequence[Any]], model: Any, *, bucket_fn: Optional[BucketFn] = None, input_mode: str = "visual", n: int = 8) -> CoreResult:
    rows: List[Row] = []
    label = model_label(model)
    for traj in trajectories:
        plan_text = oracle_plan(traj)
        for step in traj:
            conditions = [
                ("base", step, "none", None, 0),
                ("history", step, "full", None, 0),
                ("plan", step, "none", plan_text, 0),
                ("injected_error", with_raw_updates(step, {"corrupt_history": "Step 1: wrong prior action injected"}), "corrupt", None, 1),
            ]
            for block, query_step_obj, history_mode, plan, injected_error in conditions:
                pred = query_step(model, query_step_obj, history_mode, input_mode, plan=plan, n=n)
                rows.append(
                    row_from_step(
                        step,
                        manifold="on",
                        d_bucket=bucket_lookup(step, bucket_fn),
                        cond={"block": block, "history_mode": history_mode, "input_mode": input_mode, "plan": "oracle" if plan else None, "injected_error": injected_error},
                        step_correct_value=step_correct(pred, step),
                        top1_conf=pred.top1_conf,
                        action_entropy=pred.action_entropy,
                        recovery_class=pred.recovery_class,
                        self_offtrack=pred.self_offtrack,
                        model=label,
                    )
                )
    return CoreResult(rows=rows, blocks=dict(BLOCKS))
