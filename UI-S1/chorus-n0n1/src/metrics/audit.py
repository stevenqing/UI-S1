from __future__ import annotations

from typing import Any, Dict, Iterable


def truncation_summary(rows: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    total = 0
    truncated = 0
    finish_reasons: Dict[str, int] = {}
    for row in rows:
        total += 1
        if row.get("truncated"):
            truncated += 1
        reason = str(row.get("finish_reason", "unknown"))
        finish_reasons[reason] = finish_reasons.get(reason, 0) + 1
    rate = truncated / total if total else 0.0
    return {
        "total_generations": total,
        "truncated_generations": truncated,
        "truncation_rate": rate,
        "finish_reasons": finish_reasons,
    }


def is_truncation_valid(summary: Dict[str, Any], threshold: float = 0.01) -> bool:
    return float(summary.get("truncation_rate", 0.0)) <= threshold
