"""V1: OOD-repair check for history-trained arms.

V1 is necessary but not sufficient. It can certify that a history-bearing arm no
longer collapses when given its matched history representation. It must never be
used as a history-utilization verdict.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping

from gui360_long_horizon.analysis.guards import assert_format_match, assert_no_v1_utilization_claim


@dataclass(frozen=True)
class V1Result:
    arm: str
    history_format: str
    acc_none: float
    acc_matched: float
    matched_minus_none: float
    repaired: bool
    n_none: int
    n_matched: int
    caveat: str = "V1 only certifies OOD repair; it is not a history-utilization claim."


def _rows(rows: Iterable[Any]) -> List[Dict[str, Any]]:
    out = []
    for row in rows:
        if isinstance(row, Mapping):
            out.append(dict(row))
        elif hasattr(row, "to_dict"):
            out.append(row.to_dict())
        else:
            out.append(dict(row.__dict__))
    return out


def _history(row: Mapping[str, Any]) -> str:
    cond = row.get("cond") or {}
    return str(cond.get("history_format") or cond.get("history_mode") or cond.get("condition") or "")


def _acc(rows: List[Dict[str, Any]], history_format: str) -> tuple[float, int]:
    values = [bool(row.get("step_correct", row.get("correct"))) for row in rows if _history(row) == history_format and row.get("ok", True)]
    return (sum(values) / len(values), len(values)) if values else (0.0, 0)


def summarize(rows: Iterable[Any], *, arm: str, history_format: str, repair_eps: float = 0.01) -> V1Result:
    assert_format_match(arm, history_format)
    data = _rows(rows)
    acc_none, n_none = _acc(data, "none")
    acc_matched, n_matched = _acc(data, history_format)
    delta = acc_matched - acc_none
    repaired = bool(n_none and n_matched and abs(delta) <= repair_eps)
    return V1Result(arm=arm, history_format=history_format, acc_none=acc_none, acc_matched=acc_matched, matched_minus_none=delta, repaired=repaired, n_none=n_none, n_matched=n_matched)


def assert_v1_not_utilization() -> None:
    assert_no_v1_utilization_claim("V1")