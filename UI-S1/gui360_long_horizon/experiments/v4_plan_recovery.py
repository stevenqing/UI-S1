"""V4: plan-effect recovery cross-check for history-trained arms."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping

from gui360_long_horizon.analysis.guards import assert_format_match


@dataclass(frozen=True)
class V4Result:
    arm: str
    history_format: str
    acc_none: float
    acc_oracle: float
    oracle_minus_none: float
    n_none: int
    n_oracle: int


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


def _plan(row: Mapping[str, Any]) -> str:
    return str((row.get("cond") or {}).get("plan") or "")


def _acc(rows: List[Dict[str, Any]], plan: str) -> tuple[float, int]:
    values = [bool(row.get("step_correct", row.get("correct"))) for row in rows if _plan(row) == plan and row.get("ok", True)]
    return (sum(values) / len(values), len(values)) if values else (0.0, 0)


def summarize(rows: Iterable[Any], *, arm: str, history_format: str) -> V4Result:
    assert_format_match(arm, history_format)
    data = _rows(rows)
    none, n_none = _acc(data, "none")
    oracle, n_oracle = _acc(data, "oracle")
    return V4Result(arm=arm, history_format=history_format, acc_none=none, acc_oracle=oracle, oracle_minus_none=oracle - none, n_none=n_none, n_oracle=n_oracle)