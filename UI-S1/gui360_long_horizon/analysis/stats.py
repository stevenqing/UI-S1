"""Uniform statistics and pre-registered decision helpers for Row frames."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np


@dataclass(frozen=True)
class FitResult:
    dv: str
    fixed: Sequence[str]
    params: Dict[str, float]
    pvalues: Dict[str, float]
    n: int
    model: str


@dataclass(frozen=True)
class CI:
    estimate: float
    low: float
    high: float
    n: int


@dataclass(frozen=True)
class Verdict:
    label: str
    reason: str
    online_only_open_question: bool = False
    details: Dict[str, Any] | None = None


class DecisionAborted(RuntimeError):
    def __init__(self, verdict: Verdict):
        super().__init__(verdict.reason)
        self.verdict = verdict


class _MiniSeries:
    def __init__(self, values: Sequence[Any]):
        self.values = list(values)

    def mean(self) -> float:
        values = [float(value) for value in self.values if value is not None]
        return float(sum(values) / len(values)) if values else 0.0

    def dropna(self) -> "_MiniSeries":
        return _MiniSeries([value for value in self.values if value is not None])

    def unique(self) -> "_MiniSeries":
        out = []
        for value in self.values:
            if value not in out:
                out.append(value)
        return _MiniSeries(out)

    def tolist(self) -> List[Any]:
        return list(self.values)

    def __eq__(self, other: Any) -> List[bool]:
        return [value == other for value in self.values]


class _MiniFrame:
    def __init__(self, rows: Sequence[Mapping[str, Any]]):
        self.rows = [dict(row) for row in rows]
        columns = []
        for row in self.rows:
            for key in row:
                if key not in columns:
                    columns.append(key)
        self.columns = columns

    @property
    def empty(self) -> bool:
        return not self.rows

    def __len__(self) -> int:
        return len(self.rows)

    def __contains__(self, key: str) -> bool:
        return key in self.columns

    def __getitem__(self, key: Any) -> Any:
        if isinstance(key, str):
            return _MiniSeries([row.get(key) for row in self.rows])
        if isinstance(key, list) and all(isinstance(item, bool) for item in key):
            return _MiniFrame([row for row, keep in zip(self.rows, key) if keep])
        raise TypeError(f"unsupported mini-frame key: {key!r}")

    def assign(self, **kwargs: Sequence[Any]) -> "_MiniFrame":
        rows = [dict(row) for row in self.rows]
        for key, values in kwargs.items():
            for idx, value in enumerate(values):
                if idx < len(rows):
                    rows[idx][key] = value
        return _MiniFrame(rows)


def _rows_to_dicts(rows: Any) -> List[Dict[str, Any]]:
    if rows is None:
        return []
    if hasattr(rows, "rows"):
        rows = rows.rows
    if hasattr(rows, "to_dict") and not isinstance(rows, list):
        try:
            return rows.to_dict("records")
        except TypeError:
            pass
    out = []
    for row in rows:
        if hasattr(row, "to_dict"):
            out.append(row.to_dict())
        elif hasattr(row, "__dataclass_fields__"):
            out.append(asdict(row))
        elif isinstance(row, Mapping):
            out.append(dict(row))
        else:
            out.append(dict(row.__dict__))
    return out


def _dataframe(rows: Any):
    try:
        import pandas as pd
    except ImportError:
        return _MiniFrame(_rows_to_dicts(rows))

    return pd.DataFrame(_rows_to_dicts(rows))


def mixed_logit(df: Any, dv: str, fixed: Sequence[str], group: str = "exec_id") -> FitResult:
    """Fit a clustered binomial GLM for binary DVs.

    Continuous DVs are handled by Gaussian GLM so callers get the same compact
    result object. This keeps the API stable while avoiding heavy MixedLM setup
    for small audit/test samples.
    """

    import statsmodels.api as sm

    frame = _dataframe(df).dropna(subset=[dv])
    if frame.empty:
        raise ValueError(f"no non-null rows for dv={dv!r}")
    y = frame[dv].astype(float)
    x = frame[list(fixed)].astype(float) if fixed else frame.assign(intercept=1.0)[["intercept"]]
    x = sm.add_constant(x, has_constant="add")
    binary = set(y.dropna().unique()).issubset({0.0, 1.0})
    family = sm.families.Binomial() if binary else sm.families.Gaussian()
    model = sm.GLM(y, x, family=family)
    if group in frame:
        fit = model.fit(cov_type="cluster", cov_kwds={"groups": frame[group]})
    else:
        fit = model.fit()
    return FitResult(dv=dv, fixed=tuple(fixed), params={key: float(val) for key, val in fit.params.items()}, pvalues={key: float(val) for key, val in fit.pvalues.items()}, n=int(len(frame)), model="GLM-Binomial" if binary else "GLM-Gaussian")


def bootstrap_ci(df: Any, estimator: Callable[[Any], float], by: str = "task_key", B: int = 2000) -> CI:
    frame = _dataframe(df)
    if frame.empty:
        raise ValueError("cannot bootstrap an empty frame")
    estimate = float(estimator(frame))
    rng = np.random.default_rng(41)
    if isinstance(frame, _MiniFrame):
        clusters = frame[by].dropna().unique().tolist() if by in frame else list(range(len(frame)))
        if not clusters:
            clusters = list(range(len(frame)))
            frame = frame.assign(__rowid__=clusters)
            by = "__rowid__"
        boot = []
        for _ in range(B):
            sampled = rng.choice(clusters, size=len(clusters), replace=True)
            sampled_rows = []
            for cluster in sampled:
                sampled_rows.extend((frame[frame[by] == cluster]).rows)
            boot.append(float(estimator(_MiniFrame(sampled_rows))))
        low, high = np.percentile(boot, [2.5, 97.5]) if boot else (estimate, estimate)
        return CI(estimate=estimate, low=float(low), high=float(high), n=int(len(frame)))
    clusters = frame[by].dropna().unique().tolist() if by in frame else list(range(len(frame)))
    if not clusters:
        clusters = list(range(len(frame)))
        by = "__rowid__"
        frame = frame.assign(__rowid__=clusters)
    boot: List[float] = []
    for _ in range(B):
        sampled = rng.choice(clusters, size=len(clusters), replace=True)
        pieces = [frame[frame[by] == cluster] for cluster in sampled]
        boot.append(float(estimator(type(frame)(np.concatenate([piece.to_numpy() for piece in pieces], axis=0), columns=frame.columns))))
    low, high = np.percentile(boot, [2.5, 97.5]) if boot else (estimate, estimate)
    return CI(estimate=estimate, low=float(low), high=float(high), n=int(len(frame)))


def _mean_bool(rows: Iterable[Dict[str, Any]]) -> float:
    values = [bool(row["step_correct"]) for row in rows if row.get("step_correct") is not None]
    return sum(values) / len(values) if values else 0.0


def existence_verdict(m_core_out: Any) -> Dict[str, Any]:
    rows = _rows_to_dicts(m_core_out)
    blocks = getattr(m_core_out, "blocks", None) or {"history": True, "plan": True, "injected_error": True, "position": False}
    base_rows = [row for row in rows if (row.get("cond") or {}).get("block") == "base"]
    base_acc = _mean_bool(base_rows)
    effects = {}
    exists = False
    for block, identified in blocks.items():
        if block == "base":
            continue
        block_rows = [row for row in rows if (row.get("cond") or {}).get("block") == block]
        effect = _mean_bool(block_rows) - base_acc
        effects[block] = {"effect": effect, "identified": bool(identified)}
        if identified and abs(effect) > 1e-12:
            exists = True
    return {"exists": exists, "effects": effects}


def _metric(results: Mapping[str, Any], key: str, attr: str, default: float = 0.0) -> float:
    value = results.get(key)
    if value is None:
        return default
    if isinstance(value, Mapping):
        return float(value.get(attr, default))
    return float(getattr(value, attr, default))


def _capstone_source(results: Mapping[str, Any], *names: str) -> Any:
    for name in names:
        if name in results:
            return results[name]
    return None


def _capstone_attr(results: Mapping[str, Any], names: Sequence[str], attr: str, default: float = 0.0) -> float:
    value = _capstone_source(results, *names)
    if value is None:
        return default
    if isinstance(value, Mapping):
        return float(value.get(attr, default))
    return float(getattr(value, attr, default))


def _capstone_bool(results: Mapping[str, Any], names: Sequence[str], attr: str, default: bool = False) -> bool:
    value = _capstone_source(results, *names)
    if value is None:
        return default
    if isinstance(value, Mapping):
        return bool(value.get(attr, default))
    return bool(getattr(value, attr, default))


def capstone_decision(
    results: Dict[str, Any],
    *,
    memory_gap_eps: float = 0.01,
    drift_eps: float = 0.01,
    plan_jump_target: float = 0.13,
    plan_jump_tol: float = 0.04,
) -> Verdict:
    """Apply the pre-registered GUI-360 History-Utilization A/B decision table."""

    repaired = _capstone_bool(results, ("v1", "V1", "v1_ood_repair"), "repaired", False)
    o_minus_g = _capstone_attr(results, ("og_contrast", "O_minus_G"), "value", 0.0)
    if not repaired:
        return Verdict(
            label="NOT_REPAIRED_ESCALATE",
            reason="history consumption is not repaired even by O; SFT recipe did not fix the consumption artifact",
            details={"row": 4, "o_minus_g": o_minus_g},
        )

    memory_gap = _capstone_attr(results, ("v3", "V3", "v3_longdep"), "near_minus_far", 0.0)
    shuffle_clean = _capstone_bool(results, ("v3", "V3", "v3_longdep"), "shuffle_clean", False)
    drift = _capstone_attr(results, ("v2", "V2", "v2_conditionC"), "injected_minus_clean", 0.0)
    plan = _capstone_attr(results, ("v4", "V4", "v4_plan"), "oracle_minus_none", 0.0)
    details = {"memory_gap": memory_gap, "shuffle_clean": shuffle_clean, "drift": drift, "plan": plan, "o_minus_g": o_minus_g}

    if memory_gap > memory_gap_eps and shuffle_clean:
        return Verdict(label="REAL_HORIZON_MEMORY", reason="V3 long-dependency gap survives shuffle control; real horizon memory was hidden by the single-step consumption artifact", details=details)

    memory_null = abs(memory_gap) <= memory_gap_eps
    drift_null = abs(drift) <= drift_eps
    plan_jumps = abs(plan - plan_jump_target) <= plan_jump_tol or plan >= plan_jump_target - plan_jump_tol

    if memory_null and drift_null and plan_jumps:
        return Verdict(label="STRONG_NULL_MEMORY", reason="model consumes history after repair, but shuffle-clean long-horizon memory is approximately zero; planning effect returns", details=details)

    if memory_null and drift_null and not plan_jumps:
        return Verdict(label="PLAN_HISTORY_SEPARATE_DEFICITS", reason="history consumption is repaired but plan effect remains weak; planning and history are separate deficits", details=details)

    return Verdict(label="CAPSTONE_MIXED", reason="capstone probes do not match a pre-registered clean row; inspect V2/V3/V4 jointly", details=details)


def decision(results: Dict[str, Any]) -> Verdict:
    if any(key in results for key in ("v1", "V1", "v1_ood_repair", "v3", "V3", "v4", "V4")):
        return capstone_decision(results)

    textmem = results.get("m_textmem")
    if textmem is not None and not bool(getattr(textmem, "gate_passed", True)):
        verdict = Verdict(label="ABORT_TEXTMEM_GATE", reason="text-memory gate failed; visual-channel interpretation is invalid")
        raise DecisionAborted(verdict)

    core = results.get("m_core")
    core_verdict = existence_verdict(core) if core is not None else {"exists": False, "effects": {}}
    plan_beta = _metric(results, "m_plan", "beta", 0.0)
    textdrift_beta = _metric(results, "m_textdrift", "beta", 0.0)
    recover_recognition = _metric(results, "m_recover", "recognition", 0.0)
    diag_upper = _metric(results, "m_diag", "upper_bound", 0.0)
    online_open = diag_upper > 0.0

    if not core_verdict["exists"]:
        return Verdict(label="NO_IDENTIFIED_LONG_HORIZON_EFFECT", reason="no identified residual block has a non-zero effect", online_only_open_question=online_open, details=core_verdict)
    if abs(textdrift_beta) > max(abs(plan_beta), 1e-12):
        label = "TEXT_DRIFT_DOMINANT"
        reason = "randomized text-only drift is the largest identified component"
    elif plan_beta > 0.0:
        label = "PLANNING_COMPONENT"
        reason = "oracle global plan improves paired success-step correctness"
    elif recover_recognition > 0.5:
        label = "GENERIC_RECOVERY_COMPONENT"
        reason = "generic content-independent recovery is recognized on the oracle slice"
    else:
        label = "IDENTIFIED_EFFECT_MIXED"
        reason = "identified components exist but no single component dominates"
    return Verdict(label=label, reason=reason, online_only_open_question=online_open, details=core_verdict)
