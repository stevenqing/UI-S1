"""Shared helpers for long-horizon experiment runners."""

from __future__ import annotations

import copy
from dataclasses import is_dataclass, replace
from typing import Any, Callable, Dict, Iterable, List, Optional


BucketFn = Callable[[Any], int]


def model_label(model: Any) -> str:
    return str(getattr(model, "model_name", getattr(model, "cache_id", model.__class__.__name__)))


def bucket_lookup(step: Any, bucket_fn: Optional[BucketFn] = None, default: int = -1) -> int:
    if bucket_fn is not None:
        return int(bucket_fn(step))
    if hasattr(step, "d_bucket"):
        return int(getattr(step, "d_bucket"))
    raw = getattr(step, "raw", {}) or {}
    if "d_bucket" in raw:
        return int(raw["d_bucket"])
    return int(default)


def oracle_plan(steps: Iterable[Any]) -> str:
    lines: List[str] = []
    for step in sorted(steps, key=lambda item: int(getattr(item, "step_id", 0) or 0)):
        subtask = str(getattr(step, "subtask", "") or "").strip()
        if subtask:
            lines.append(f"Step {getattr(step, 'step_id', len(lines) + 1)}: {subtask}")
    return "\n".join(lines)


def with_raw_updates(step: Any, updates: Dict[str, Any]) -> Any:
    raw = dict(getattr(step, "raw", {}) or {})
    raw.update(updates)
    if is_dataclass(step):
        return replace(step, raw=raw)
    new_step = copy.copy(step)
    setattr(new_step, "raw", raw)
    return new_step


def fail_history_snippets(fail_steps: Iterable[Any]) -> List[str]:
    snippets = []
    for step in fail_steps:
        raw = getattr(step, "raw", {}) or {}
        for key in ("history_text", "thought", "observation"):
            text = str(raw.get(key) or getattr(step, key, "") or "").strip()
            if text:
                snippets.append(text)
                break
        else:
            snippets.append(f"Step {getattr(step, 'step_id', '?')}: wrong prior action")
    return snippets
