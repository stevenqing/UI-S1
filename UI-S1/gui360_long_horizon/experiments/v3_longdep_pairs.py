"""V3: long-dependency matched-pair memory probe."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Sequence

from gui360_long_horizon.analysis.guards import assert_format_match
from gui360_long_horizon.data.longdep_pairs import LongDepPair, assert_difficulty_balanced, assert_shuffle_collapse


@dataclass(frozen=True)
class V3Result:
    arm: str
    history_format: str
    near_acc: float
    far_acc: float
    near_minus_far: float
    shuffle_gap: float
    shuffle_clean: bool
    n_near: int
    n_far: int


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


def _distance(row: Mapping[str, Any]) -> str:
    cond = row.get("cond") or {}
    return str(cond.get("distance") or cond.get("pair_distance") or "")


def _acc(rows: List[Dict[str, Any]], distance: str) -> tuple[float, int]:
    values = [bool(row.get("step_correct", row.get("correct"))) for row in rows if _distance(row) == distance and row.get("ok", True)]
    return (sum(values) / len(values), len(values)) if values else (0.0, 0)


def summarize(rows: Iterable[Any], *, arm: str, history_format: str, pairs: Sequence[LongDepPair], shuffle_gap: float = 0.0, max_abs_bucket_gap: int = 0) -> V3Result:
    assert_format_match(arm, history_format)
    assert_difficulty_balanced(pairs, max_abs_bucket_gap=max_abs_bucket_gap)
    data = _rows(rows)
    near, n_near = _acc(data, "near")
    far, n_far = _acc(data, "far")
    gap = near - far
    assert_shuffle_collapse(gap, shuffle_gap)
    return V3Result(arm=arm, history_format=history_format, near_acc=near, far_acc=far, near_minus_far=gap, shuffle_gap=shuffle_gap, shuffle_clean=True, n_near=n_near, n_far=n_far)