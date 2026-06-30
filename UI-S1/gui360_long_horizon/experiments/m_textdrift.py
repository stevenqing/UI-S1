"""Randomized text-only drift injection experiment."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Iterable, List, Optional

from gui360_long_horizon.harness.correctness import step_correct
from gui360_long_horizon.harness.predict import query_step
from gui360_long_horizon.types import Row, row_from_step

from .common import BucketFn, bucket_lookup, fail_history_snippets, model_label, with_raw_updates


@dataclass(frozen=True)
class TextDriftResult:
    rows: List[Row]
    identified: bool = True
    invariant: str = "injected error is text only; screenshot is unchanged"


def run(success_steps: Iterable[Any], fail_steps: Iterable[Any], model: Any, *, bucket_fn: Optional[BucketFn] = None, input_mode: str = "visual", max_injected: int = 3, seed: int = 41, n: int = 8) -> TextDriftResult:
    rng = random.Random(seed)
    snippets = fail_history_snippets(fail_steps) or ["Step 1: wrong prior action"]
    rows: List[Row] = []
    label = model_label(model)
    for step in success_steps:
        base_pred = query_step(model, step, "none", input_mode, n=n)
        if not step_correct(base_pred, step):
            continue
        for injected_count in range(1, max_injected + 1):
            injected = "\n".join(rng.choice(snippets) for _ in range(injected_count))
            corrupted = with_raw_updates(step, {"corrupt_history": injected})
            pred = query_step(model, corrupted, "corrupt", input_mode, n=n)
            rows.append(
                row_from_step(
                    step,
                    manifold="on",
                    d_bucket=bucket_lookup(step, bucket_fn),
                    cond={"history_mode": "corrupt", "input_mode": input_mode, "plan": None, "injected_error": injected_count},
                    step_correct_value=step_correct(pred, step),
                    top1_conf=pred.top1_conf,
                    action_entropy=pred.action_entropy,
                    recovery_class=pred.recovery_class,
                    self_offtrack=pred.self_offtrack,
                    model=label,
                )
            )
    return TextDriftResult(rows=rows)
