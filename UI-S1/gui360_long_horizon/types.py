"""Shared row contract and identifiers for GUI-360 long-horizon experiments."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


def task_key_for(request: str, template: str, app: str) -> str:
    """Stable task key from the pre-registered request/template/app tuple."""

    payload = f"{request or ''}\n{template or ''}\n{app or ''}".encode("utf-8")
    return hashlib.sha1(payload).hexdigest()[:16]


def task_key_for_step(step: Any) -> str:
    return task_key_for(getattr(step, "request", ""), getattr(step, "template", ""), getattr(step, "app", ""))


def depth_fraction(step: Any) -> float:
    total_steps = int(getattr(step, "total_steps", 0) or 0)
    step_id = int(getattr(step, "step_id", 0) or 0)
    if total_steps <= 1:
        return 0.0
    return max(0.0, min(1.0, (step_id - 1) / max(total_steps - 1, 1)))


@dataclass(frozen=True)
class Row:
    exec_id: str
    app: str
    tag: str
    task_key: str
    step_id: int
    depth_frac: float
    manifold: str
    delta: Optional[int]
    d_bucket: int
    cond: Dict[str, Any] = field(default_factory=dict)
    step_correct: Optional[bool] = None
    recovery_correct: Optional[bool] = None
    top1_conf: float = 0.0
    action_entropy: float = 0.0
    recovery_class: str = "other"
    self_offtrack: float = 0.0
    model: str = ""

    def __post_init__(self) -> None:
        if self.manifold not in {"on", "off"}:
            raise ValueError(f"manifold must be 'on' or 'off', got {self.manifold!r}")
        if self.manifold == "off" and self.step_correct is not None:
            raise ValueError("off-manifold rows must keep step_correct=None")
        if self.manifold == "on" and self.delta is not None:
            raise ValueError("on-manifold rows must keep delta=None")
        if self.recovery_class not in {"recover", "progress", "other"}:
            raise ValueError(f"unknown recovery_class: {self.recovery_class!r}")


def row_from_step(
    step: Any,
    *,
    manifold: str,
    d_bucket: int,
    cond: Dict[str, Any],
    model: str,
    delta_value: Optional[int] = None,
    step_correct_value: Optional[bool] = None,
    recovery_correct: Optional[bool] = None,
    top1_conf: float = 0.0,
    action_entropy: float = 0.0,
    recovery_class: str = "other",
    self_offtrack: float = 0.0,
) -> Row:
    return Row(
        exec_id=str(getattr(step, "exec_id", "")),
        app=str(getattr(step, "app", "")),
        tag=str(getattr(step, "tag", "")),
        task_key=task_key_for_step(step),
        step_id=int(getattr(step, "step_id", 0) or 0),
        depth_frac=depth_fraction(step),
        manifold=manifold,
        delta=delta_value,
        d_bucket=int(d_bucket),
        cond=dict(cond),
        step_correct=step_correct_value,
        recovery_correct=recovery_correct,
        top1_conf=float(top1_conf),
        action_entropy=float(action_entropy),
        recovery_class=recovery_class,
        self_offtrack=float(self_offtrack),
        model=model,
    )
