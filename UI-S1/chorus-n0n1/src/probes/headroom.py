from __future__ import annotations

from collections import Counter
from typing import Any, Dict, Iterable, List, Optional

from src.bench.har_odyssey_results import category_error_table, compact_step_for_queue


N0_PROBES = [
    "teacher_action_recovery",
    "teacher_error_attribution",
    "teacher_coordinate_recovery",
]


def build_headroom_summary(step_rows: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    rows = list(step_rows)
    total = len(rows)
    errors = [row for row in rows if row.get("baseline_error")]
    first_errors = [row for row in errors if row.get("is_first_error_step")]
    after_first_errors = [row for row in errors if row.get("after_first_error")]
    error_types = Counter(_error_kind(row) for row in errors)
    return {
        "n0_status": "OFFLINE_QUEUE_READY",
        "definition": "N0 raw headroom is the set of baseline-error steps queued for teacher probes.",
        "steps": total,
        "baseline_error_steps": len(errors),
        "baseline_error_percent": 100 * len(errors) / total if total else 0.0,
        "first_error_steps": len(first_errors),
        "after_first_error_steps": len(after_first_errors),
        "error_kind_counts": dict(sorted(error_types.items())),
        "by_category": category_error_table(rows),
        "teacher_probe_required": True,
        "teacher_probe_model_calls": "not_started",
        "model_call_requirement": "All N0 teacher calls must use src/infer/wrapper.py.",
        "truncation_gate": "Do not run N0 teacher probes if Phase A truncated_generation_percent exceeds 1%.",
        "probes": N0_PROBES,
    }


def build_headroom_probe_queue(
    step_rows: Iterable[Dict[str, Any]],
    include_correct: bool = False,
    limit: Optional[int] = None,
) -> List[Dict[str, Any]]:
    queue: List[Dict[str, Any]] = []
    for row in step_rows:
        if not include_correct and not row.get("baseline_error"):
            continue
        item = compact_step_for_queue(row)
        item.update(
            {
                "n0_probe_status": "queued",
                "n0_probe_family": "headroom",
                "required_probes": N0_PROBES,
                "expected_outputs": {
                    "teacher_action": "official-action-format prediction under teacher prompting",
                    "teacher_match": "official matcher result for teacher_action",
                    "recoverable": "whether the teacher probe fixes the baseline error",
                    "attribution": "vision, action_type, coordinate, history, or ambiguity",
                },
                "model_calls_must_use": "src/infer/wrapper.py",
            }
        )
        queue.append(item)
        if limit is not None and len(queue) >= limit:
            break
    return queue


def build_headroom_manifest(
    step_rows: Iterable[Dict[str, Any]],
    queue_path: str,
    include_correct: bool = False,
    limit: Optional[int] = None,
) -> Dict[str, Any]:
    rows = list(step_rows)
    queue = build_headroom_probe_queue(rows, include_correct=include_correct, limit=limit)
    summary = build_headroom_summary(rows)
    summary.update(
        {
            "queue_path": queue_path,
            "queue_items": len(queue),
            "queue_policy": "baseline_error_steps" if not include_correct else "all_steps",
            "queue_limit": limit,
            "sota_proxy": True,
        }
    )
    return summary


def _error_kind(row: Dict[str, Any]) -> str:
    if row.get("type_match") is False:
        return "action_type"
    gt_action = row.get("gt_action") or {}
    pred_action = row.get("pred_action") or {}
    if gt_action.get("action") in {"click", "swipe", "long_press"} or pred_action.get("action") in {
        "click",
        "swipe",
        "long_press",
    }:
        return "coordinate_or_target"
    if row.get("truncated"):
        return "truncation"
    if row.get("error"):
        return "parse_or_runtime"
    return "semantic_or_sequence"