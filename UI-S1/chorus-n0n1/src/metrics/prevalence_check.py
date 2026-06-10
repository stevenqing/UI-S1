from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, Optional


def build_prevalence_summary(step_rows: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    rows = list(step_rows)
    total = len(rows)
    errors = [row for row in rows if row.get("baseline_error")]
    correct = total - len(errors)
    error_action_counts = Counter(str(row.get("gt_action_type") or "unknown") for row in errors)
    by_category: dict[str, dict[str, int]] = defaultdict(lambda: {"steps": 0, "errors": 0, "type_errors": 0})
    for row in rows:
        category = str(row.get("category", ""))
        by_category[category]["steps"] += 1
        if row.get("baseline_error"):
            by_category[category]["errors"] += 1
        if row.get("type_match") is False:
            by_category[category]["type_errors"] += 1
    return {
        "n1_status": "PREVALENCE_CHECK_ONLY",
        "definition": "Counts/rates from scoring-side metadata only; not a reader-disagreement metric.",
        "steps": total,
        "correct_steps": correct,
        "baseline_error_steps": len(errors),
        "baseline_error_percent": 100 * len(errors) / total if total else 0.0,
        "correct_percent": 100 * correct / total if total else 0.0,
        "error_counts_by_gt_action": dict(sorted(error_action_counts.items())),
        "by_category": _category_rows(by_category),
        "reader_model_calls": "not_started",
        "model_call_requirement": "All reader calls must use src/infer/wrapper.py.",
        "gt_isolation_note": "Prevalence checks stay on the scoring side; reader inputs are built separately without GT fields.",
    }


def build_prevalence_manifest(
    step_rows: Iterable[Dict[str, Any]],
    queue_path: str,
    queue_items: int,
    limit: Optional[int] = None,
) -> Dict[str, Any]:
    summary = build_prevalence_summary(step_rows)
    summary.update({
        "queue_path": queue_path,
        "queue_items": queue_items,
        "queue_limit": limit,
        "sota_proxy": True,
    })
    return summary


def _category_rows(by_category: dict[str, dict[str, int]]) -> list[dict[str, Any]]:
    rows = []
    for category, values in sorted(by_category.items()):
        steps = values["steps"]
        errors = values["errors"]
        rows.append({
            "category": category,
            "steps": steps,
            "baseline_error_steps": errors,
            "baseline_error_percent": 100 * errors / steps if steps else 0.0,
            "type_errors": values["type_errors"],
        })
    return rows