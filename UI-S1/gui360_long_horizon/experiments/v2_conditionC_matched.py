"""V2: format-matched Condition-C drift probe."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping

from gui360_long_horizon.analysis.guards import assert_format_match


@dataclass(frozen=True)
class V2Result:
    arm: str
    history_format: str
    acc_clean: float
    acc_injected: float
    injected_minus_clean: float
    n_clean: int
    n_injected: int
    identified: bool = True


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


def _is_injected(row: Mapping[str, Any]) -> bool:
    cond = row.get("cond") or {}
    return bool(cond.get("injected_error") or cond.get("condition") == "injected")


def _acc(rows: List[Dict[str, Any]], injected: bool) -> tuple[float, int]:
    values = [bool(row.get("step_correct", row.get("correct"))) for row in rows if _is_injected(row) == injected and row.get("ok", True)]
    return (sum(values) / len(values), len(values)) if values else (0.0, 0)


def summarize(rows: Iterable[Any], *, arm: str, history_format: str) -> V2Result:
    assert_format_match(arm, history_format)
    data = _rows(rows)
    clean, n_clean = _acc(data, False)
    injected, n_injected = _acc(data, True)
    return V2Result(arm=arm, history_format=history_format, acc_clean=clean, acc_injected=injected, injected_minus_clean=injected - clean, n_clean=n_clean, n_injected=n_injected)