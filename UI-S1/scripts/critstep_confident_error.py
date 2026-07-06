#!/usr/bin/env python3
"""Confident-error and external-disagreement diagnostic for critical steps.

Part A uses the all-step sampled pool to compare model confidence on critical
greedy errors, non-critical greedy-correct steps, and critical long-tail correct
samples. Part B uses verifier/external candidate scores only when they are
available for both critical and non-critical steps; otherwise it reports the
coverage limitation rather than filling missing external scores with a leaky
proxy.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy.stats import mannwhitneyu, rankdata

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.score_critstep_verifier_v2_cot_voting import candidate_distinct_key  # noqa: E402


DEFAULT_CANDIDATES = "outputs/verifier_e2e/slice200/candidates/per_step.jsonl"
DEFAULT_CRIT_TASKS = "outputs/critstep_eval/per_task.jsonl"
DEFAULT_IDENTIFY_SUMMARY = "outputs/critstep_identify/summary.json"
DEFAULT_OUTPUT_DIR = "outputs/critstep_confident_error"
BUDGETS = (0.10, 0.20, 0.30)


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def safe_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        value = float(value)
        return value if math.isfinite(value) else None
    except (TypeError, ValueError):
        return None


def auc_score(y: Sequence[int], scores: Sequence[float]) -> Optional[float]:
    y_arr = np.asarray(y, dtype=int)
    s_arr = np.asarray(scores, dtype=float)
    mask = np.isfinite(s_arr)
    y_arr = y_arr[mask]
    s_arr = s_arr[mask]
    n_pos = int(np.sum(y_arr == 1))
    n_neg = int(np.sum(y_arr == 0))
    if n_pos == 0 or n_neg == 0:
        return None
    ranks = rankdata(s_arr, method="average")
    pos_rank_sum = float(np.sum(ranks[y_arr == 1]))
    return (pos_rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def oriented_auc(y: Sequence[int], scores: Sequence[float]) -> Optional[float]:
    auc = auc_score(y, scores)
    return None if auc is None else max(float(auc), 1.0 - float(auc))


def entropy(values: Sequence[str]) -> float:
    vals = [str(value) for value in values if value not in (None, "")]
    if not vals:
        return 0.0
    total = len(vals)
    counts = Counter(vals)
    return -sum((count / total) * math.log(count / total, 2) for count in counts.values())


def normalized_entropy(values: Sequence[str]) -> float:
    vals = [str(value) for value in values if value not in (None, "")]
    if len(set(vals)) <= 1:
        return 0.0
    return entropy(vals) / math.log(len(set(vals)), 2)


def key_counter(candidates: Sequence[Mapping[str, Any]], key_fn) -> Counter[str]:
    return Counter(key_fn(candidate) for candidate in candidates)


def action_key(candidate: Mapping[str, Any]) -> str:
    return str(candidate.get("action_signature") or json.dumps(candidate.get("action") or {}, sort_keys=True, ensure_ascii=False))


def control_key(candidate: Mapping[str, Any]) -> str:
    control = candidate.get("control") if isinstance(candidate.get("control"), dict) else {}
    return str(control.get("key") or "NO_CONTROL")


def support_fraction(counter: Counter[str], key: str, total: int) -> float:
    return counter.get(key, 0) / total if total else 0.0


def candidate_external_score(candidate: Mapping[str, Any]) -> Optional[float]:
    for key in ("stage1_score_k8", "verifier_score", "verifier_margin", "stage1_score", "stage2_score"):
        value = safe_float(candidate.get(key))
        if value is not None:
            return value
    return None


def candidate_logprob(candidate: Mapping[str, Any]) -> Optional[float]:
    return safe_float(candidate.get("model_logprob_avg"))


def load_tasks(path: Path) -> Dict[str, Dict[str, Any]]:
    return {str(row.get("episode_id")): row for row in read_jsonl(path)}


def load_verifier_rows(paths: Sequence[str]) -> Dict[Tuple[str, int], Dict[str, Any]]:
    out: Dict[Tuple[str, int], Dict[str, Any]] = {}
    for raw_path in paths:
        if not raw_path:
            continue
        path = Path(raw_path)
        for row in read_jsonl(path):
            if row.get("episode_id") is None or row.get("step_idx") is None:
                continue
            key = (str(row.get("episode_id")), int(row.get("step_idx")))
            out[key] = row
    return out


def attach_verifier_scores(row: Dict[str, Any], verifier_row: Optional[Mapping[str, Any]]) -> None:
    if not verifier_row:
        return
    score_by_key: Dict[str, Dict[str, Any]] = {}
    for candidate in verifier_row.get("candidates", []) if isinstance(verifier_row.get("candidates"), list) else []:
        try:
            score_by_key[candidate_distinct_key(candidate)] = candidate
        except Exception:
            continue
    for candidate in row.get("candidates", []) if isinstance(row.get("candidates"), list) else []:
        try:
            match = score_by_key.get(candidate_distinct_key(candidate))
        except Exception:
            match = None
        if not match:
            continue
        for key in ("stage1_score_k8", "verifier_score", "verifier_margin", "verifier_logprob_correct", "verifier_logprob_incorrect"):
            if match.get(key) is not None:
                candidate[key] = match.get(key)


def build_rows(args: argparse.Namespace) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    raw_rows = read_jsonl(Path(args.candidates))
    tasks = load_tasks(Path(args.crit_tasks))
    verifier_rows = load_verifier_rows(args.verifier_scored)
    out = []
    coverage = Counter()
    for raw in raw_rows:
        row = dict(raw)
        episode_id = str(row.get("episode_id"))
        step_idx = int(row.get("step_idx") or 0)
        task = tasks.get(episode_id)
        if not task:
            coverage["missing_task"] += 1
            continue
        attach_verifier_scores(row, verifier_rows.get((episode_id, step_idx)))
        candidates = row.get("candidates") if isinstance(row.get("candidates"), list) else []
        if not candidates:
            coverage["no_candidates"] += 1
            continue
        bottom2 = {int(idx) for idx in task.get("bottom2_critical_indices", [])}
        per_p = task.get("per_step_p_heldout_cv") if isinstance(task.get("per_step_p_heldout_cv"), list) else []
        p_i = safe_float(per_p[step_idx]) if step_idx < len(per_p) else None
        action_counts = key_counter(candidates, action_key)
        control_counts = key_counter(candidates, control_key)
        total = len(candidates)
        greedy = candidates[0]
        greedy_action_key = action_key(greedy)
        greedy_control_key = control_key(greedy)
        greedy_action_support = support_fraction(action_counts, greedy_action_key, total)
        greedy_control_support = support_fraction(control_counts, greedy_control_key, total)
        action_norm_entropy = normalized_entropy([action_key(candidate) for candidate in candidates])
        control_norm_entropy = normalized_entropy([control_key(candidate) for candidate in candidates])
        greedy_same_logprobs = [candidate_logprob(candidate) for candidate in candidates[1:] if action_key(candidate) == greedy_action_key]
        greedy_same_logprobs = [value for value in greedy_same_logprobs if value is not None]
        external_scores = [candidate_external_score(candidate) for candidate in candidates]
        external_scores = [value for value in external_scores if value is not None]
        greedy_external = candidate_external_score(greedy)
        best_external = max(external_scores) if external_scores else None
        non_greedy_scores = [(idx, candidate_external_score(candidate)) for idx, candidate in enumerate(candidates) if idx > 0 and candidate_external_score(candidate) is not None]
        best_non_greedy_score = max((score for _, score in non_greedy_scores), default=None)
        best_external_idx = None
        if external_scores:
            scored = [(idx, candidate_external_score(candidate)) for idx, candidate in enumerate(candidates) if candidate_external_score(candidate) is not None]
            best_external_idx = max(scored, key=lambda item: (float(item[1]), -item[0]))[0]
        model_confidence = 0.5 * greedy_action_support + 0.5 * (1.0 - action_norm_entropy)
        external_gap = None if best_external is None or greedy_external is None else best_external - greedy_external
        external_gap_non_greedy = None if best_non_greedy_score is None or greedy_external is None else best_non_greedy_score - greedy_external
        model_conf_minus_external_doubt = None if greedy_external is None else model_confidence - greedy_external
        correct_candidates = [candidate for candidate in candidates[1:] if candidate.get("is_correct")]
        longtail_items = []
        for candidate in correct_candidates:
            longtail_items.append({
                "candidate_id": candidate.get("candidate_id"),
                "action_support": support_fraction(action_counts, action_key(candidate), total),
                "control_support": support_fraction(control_counts, control_key(candidate), total),
                "model_logprob_avg": candidate_logprob(candidate),
                "external_score": candidate_external_score(candidate),
            })
        item = {
            "target_id": row.get("target_id"),
            "episode_id": episode_id,
            "step_idx": step_idx,
            "critical": step_idx in bottom2,
            "p_i_heldout_label_only": p_i,
            "greedy_correct": bool(greedy.get("is_correct")),
            "recoverable_longtail": bool(longtail_items),
            "model_confidence": model_confidence,
            "greedy_action_support": greedy_action_support,
            "greedy_control_support": greedy_control_support,
            "one_minus_action_entropy_norm": 1.0 - action_norm_entropy,
            "one_minus_control_entropy_norm": 1.0 - control_norm_entropy,
            "greedy_same_sample_logprob_mean": float(np.mean(greedy_same_logprobs)) if greedy_same_logprobs else None,
            "greedy_same_sample_logprob_max": max(greedy_same_logprobs) if greedy_same_logprobs else None,
            "greedy_external_score": greedy_external,
            "best_external_score": best_external,
            "best_non_greedy_external_score": best_non_greedy_score,
            "external_best_minus_greedy": external_gap,
            "external_best_non_greedy_minus_greedy": external_gap_non_greedy,
            "external_prefers_non_greedy": None if best_external_idx is None else bool(best_external_idx != 0),
            "model_conf_minus_external_greedy": model_conf_minus_external_doubt,
            "external_score_available": greedy_external is not None and best_external is not None,
            "longtail_correct_items": longtail_items,
        }
        coverage["rows_out"] += 1
        coverage["external_available"] += int(item["external_score_available"])
        coverage["external_available_critical"] += int(item["external_score_available"] and item["critical"])
        coverage["external_available_noncritical"] += int(item["external_score_available"] and not item["critical"])
        out.append(item)
    return out, {
        "candidate_rows_in": len(raw_rows),
        "verifier_scored_rows_in": len(verifier_rows),
        "coverage": dict(coverage),
        "note": "Verifier/external AUC is computed only when external scores cover both critical and non-critical rows.",
    }


def describe(values: Sequence[Optional[float]]) -> Dict[str, Any]:
    vals = np.asarray([float(value) for value in values if value is not None and math.isfinite(float(value))], dtype=float)
    if len(vals) == 0:
        return {"n": 0, "mean": None, "median": None, "p25": None, "p75": None}
    return {
        "n": int(len(vals)),
        "mean": float(np.mean(vals)),
        "median": float(np.median(vals)),
        "p25": float(np.quantile(vals, 0.25)),
        "p75": float(np.quantile(vals, 0.75)),
    }


def mannwhitney_summary(a: Sequence[Optional[float]], b: Sequence[Optional[float]]) -> Dict[str, Any]:
    aa = [float(value) for value in a if value is not None and math.isfinite(float(value))]
    bb = [float(value) for value in b if value is not None and math.isfinite(float(value))]
    if not aa or not bb:
        return {"n_a": len(aa), "n_b": len(bb), "auc_a_greater_b": None, "p_value": None}
    result = mannwhitneyu(aa, bb, alternative="two-sided")
    auc = float(result.statistic) / (len(aa) * len(bb))
    return {"n_a": len(aa), "n_b": len(bb), "auc_a_greater_b": auc, "p_value": float(result.pvalue)}


def confidence_groups(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    critical_errors = [row for row in rows if row["critical"] and not row["greedy_correct"]]
    noncritical_correct = [row for row in rows if not row["critical"] and row["greedy_correct"]]
    longtail_conf = []
    longtail_logprob = []
    for row in rows:
        if not row["critical"]:
            continue
        for item in row.get("longtail_correct_items") or []:
            longtail_conf.append(0.5 * float(item.get("action_support") or 0.0) + 0.5 * float(item.get("control_support") or 0.0))
            if item.get("model_logprob_avg") is not None:
                longtail_logprob.append(float(item["model_logprob_avg"]))
    conf_i = [row["model_confidence"] for row in critical_errors]
    conf_ii = [row["model_confidence"] for row in noncritical_correct]
    action_i = [row["greedy_action_support"] for row in critical_errors]
    action_ii = [row["greedy_action_support"] for row in noncritical_correct]
    logprob_i = [row.get("greedy_same_sample_logprob_max") for row in critical_errors]
    summary = {
        "critical_greedy_errors": {
            "n_steps": len(critical_errors),
            "model_confidence": describe(conf_i),
            "greedy_action_support": describe(action_i),
            "greedy_same_sample_logprob_max": describe(logprob_i),
        },
        "noncritical_greedy_correct": {
            "n_steps": len(noncritical_correct),
            "model_confidence": describe(conf_ii),
            "greedy_action_support": describe(action_ii),
        },
        "critical_longtail_correct_samples": {
            "n_samples": len(longtail_conf),
            "support_confidence": describe(longtail_conf),
            "model_logprob_avg": describe(longtail_logprob),
        },
        "tests": {
            "critical_error_vs_noncritical_correct_model_conf": mannwhitney_summary(conf_i, conf_ii),
            "critical_error_vs_longtail_correct_support_conf": mannwhitney_summary(conf_i, longtail_conf),
            "critical_error_logprob_vs_longtail_logprob": mannwhitney_summary(logprob_i, longtail_logprob),
        },
    }
    med_i = summary["critical_greedy_errors"]["model_confidence"].get("median")
    med_ii = summary["noncritical_greedy_correct"]["model_confidence"].get("median")
    med_iii = summary["critical_longtail_correct_samples"]["support_confidence"].get("median")
    if med_i is not None and med_iii is not None and med_ii is not None and med_i >= med_iii - 0.05 and med_i >= med_ii - 0.10:
        verdict = "CONFIDENT ERROR CONFIRMED"
        reason = "Critical greedy errors have confidence comparable to long-tail correct samples and not much lower than non-critical correct steps."
    elif med_i is not None and med_ii is not None and med_i < med_ii - 0.15:
        verdict = "NOT CONFIDENT ERROR"
        reason = "Critical greedy errors are meaningfully lower-confidence than non-critical correct steps."
    else:
        verdict = "CONFIDENT ERROR WEAK/MIXED"
        reason = "Confidence separation is small or long-tail comparison is under-covered."
    summary["verdict"] = verdict
    summary["reason"] = reason
    return summary


def external_metrics(rows: Sequence[Mapping[str, Any]], internal_baseline_auc: float) -> Dict[str, Any]:
    external_rows = [row for row in rows if row.get("external_score_available")]
    y = [int(row["critical"]) for row in external_rows]
    signals = {
        "external_best_minus_greedy": [row.get("external_best_minus_greedy") for row in external_rows],
        "external_best_non_greedy_minus_greedy": [row.get("external_best_non_greedy_minus_greedy") for row in external_rows],
        "model_conf_minus_external_greedy": [row.get("model_conf_minus_external_greedy") for row in external_rows],
        "external_prefers_non_greedy": [1.0 if row.get("external_prefers_non_greedy") else 0.0 for row in external_rows],
    }
    signal_metrics = []
    for name, values in signals.items():
        valid = [(label, value) for label, value in zip(y, values) if value is not None]
        labels = [label for label, _ in valid]
        vals = [float(value) for _, value in valid]
        auc = oriented_auc(labels, vals) if valid else None
        signal_metrics.append({"signal": name, "n": len(valid), "oriented_auc": auc})
    signal_metrics.sort(key=lambda item: item["oriented_auc"] if item["oriented_auc"] is not None else -1.0, reverse=True)
    best = signal_metrics[0] if signal_metrics else {"signal": None, "oriented_auc": None, "n": 0}
    greedy_wrong_scores = [1.0 if not row["greedy_correct"] else 0.0 for row in rows]
    greedy_wrong_auc = oriented_auc([int(row["critical"]) for row in rows], greedy_wrong_scores)
    incremental = None
    within_wrong_auc = None
    if best.get("signal") is not None:
        best_name = str(best["signal"])
        valid_rows = [row for row in external_rows if row.get(best_name) is not None]
        labels = [int(row["critical"]) for row in valid_rows]
        values = [float(row[best_name]) for row in valid_rows]
        raw_auc = oriented_auc(labels, values) if valid_rows else None
        if raw_auc is not None and greedy_wrong_auc is not None:
            incremental = raw_auc - greedy_wrong_auc
        wrong_rows = [row for row in valid_rows if not row["greedy_correct"]]
        if wrong_rows:
            within_wrong_auc = oriented_auc([int(row["critical"]) for row in wrong_rows], [float(row[best_name]) for row in wrong_rows])
    coverage = {
        "rows_with_external": len(external_rows),
        "critical_with_external": sum(1 for row in external_rows if row["critical"]),
        "noncritical_with_external": sum(1 for row in external_rows if not row["critical"]),
        "all_rows": len(rows),
    }
    if coverage["critical_with_external"] == 0 or coverage["noncritical_with_external"] == 0:
        verdict = "EXTERNAL IDENTIFICATION NOT TESTABLE"
        reason = "Verifier/external scores do not cover both critical and non-critical steps, so raw AUC and circularity guard would be invalid."
    elif (best.get("oriented_auc") or 0.0) > internal_baseline_auc + 0.05 and incremental is not None and incremental > 0.03 and within_wrong_auc is not None and within_wrong_auc > 0.60:
        verdict = "GENUINE EXTERNAL SIGNAL"
        reason = "External disagreement beats internal uncertainty and retains signal beyond greedy correctness."
    elif (best.get("oriented_auc") or 0.0) > internal_baseline_auc + 0.05:
        verdict = "EXTERNAL SIGNAL CIRCULAR/WEAK"
        reason = "Raw external AUC is higher, but incremental power beyond greedy correctness is not established."
    else:
        verdict = "NO EXTERNAL IMPROVEMENT"
        reason = "External disagreement does not clearly beat the internal uncertainty baseline."
    return {
        "coverage": coverage,
        "signals": signal_metrics,
        "best_signal": best,
        "greedy_wrong_auc_reference": greedy_wrong_auc,
        "increment_over_greedy_wrong_auc": incremental,
        "within_greedy_wrong_auc": within_wrong_auc,
        "internal_baseline_auc": internal_baseline_auc,
        "verdict": verdict,
        "reason": reason,
    }


def triage(rows: Sequence[Mapping[str, Any]], score_name: str, budgets: Sequence[float]) -> List[Dict[str, Any]]:
    valid_rows = [row for row in rows if row.get(score_name) is not None]
    if not valid_rows:
        return []
    scores = np.asarray([float(row[score_name]) for row in valid_rows], dtype=float)
    y = np.asarray([int(row["critical"]) for row in valid_rows], dtype=int)
    order = np.argsort(-scores)
    total_pos = int(np.sum(y == 1))
    out = []
    for budget in budgets:
        k = max(1, int(round(len(valid_rows) * budget)))
        selected = order[:k]
        hits = int(np.sum(y[selected] == 1))
        out.append({
            "budget_fraction": budget,
            "selected_steps": k,
            "recall": hits / total_pos if total_pos else 0.0,
            "precision": hits / k if k else 0.0,
            "random_recall": budget,
            "coverage_rows": len(valid_rows),
        })
    return out


def identify_internal_baseline(path: Path, user_baseline: float) -> Dict[str, Any]:
    if not path.exists():
        return {"auc": user_baseline, "source": "user_spec_default"}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        auc = data.get("classifier", {}).get("auc")
        if auc is not None:
            return {"auc": float(auc), "source": str(path)}
    except Exception:
        pass
    return {"auc": user_baseline, "source": "user_spec_default"}


def fmt_pct(value: Optional[float]) -> str:
    if value is None:
        return "NA"
    return f"{100.0 * value:.2f}%"


def fmt_num(value: Optional[float], digits: int = 3) -> str:
    if value is None:
        return "NA"
    return f"{value:.{digits}f}"


def render_report(summary: Mapping[str, Any], output_dir: Path) -> str:
    lines = ["# Confident Errors and External-Disagreement Identification", ""]
    lines.append("Diagnostic only: zero training. Held-out p_i defines the critical label only; matcher correctness is used for grouping/guard analyses, not as an inference feature.")
    lines.append("")
    lines.append("## Scope")
    lines.append("")
    lines.append(f"- sampled pool: `{summary['inputs']['candidates']}`")
    lines.append(f"- verifier-scored inputs: `{summary['inputs']['verifier_scored']}`")
    lines.append(f"- rows analyzed: `{summary['dataset']['rows']}`")
    lines.append(f"- critical steps: `{summary['dataset']['critical_steps']}` (`{summary['dataset']['critical_prevalence']*100:.2f}%`)")
    lines.append(f"- internal uncertainty reference AUC to beat: `{summary['internal_reference_auc']*100:.2f}%` (spec baseline)")
    lines.append(f"- current covered-slice internal AUC: `{summary['internal_baseline']['auc']*100:.2f}%` from `{summary['internal_baseline']['source']}`")
    lines.append("")
    lines.append("## Part A: Confident Error")
    lines.append("")
    part_a = summary["part_a"]
    lines.append("| group | n | model confidence mean | median | p25 | p75 |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    ce = part_a["critical_greedy_errors"]["model_confidence"]
    nc = part_a["noncritical_greedy_correct"]["model_confidence"]
    lt = part_a["critical_longtail_correct_samples"]["support_confidence"]
    lines.append(f"| critical-step greedy errors | {ce['n']} | {fmt_num(ce['mean'])} | {fmt_num(ce['median'])} | {fmt_num(ce['p25'])} | {fmt_num(ce['p75'])} |")
    lines.append(f"| non-critical greedy correct | {nc['n']} | {fmt_num(nc['mean'])} | {fmt_num(nc['median'])} | {fmt_num(nc['p25'])} | {fmt_num(nc['p75'])} |")
    lines.append(f"| critical long-tail correct samples | {lt['n']} | {fmt_num(lt['mean'])} | {fmt_num(lt['median'])} | {fmt_num(lt['p25'])} | {fmt_num(lt['p75'])} |")
    lines.append("")
    lines.append("Tests:")
    lines.append("")
    lines.append("| comparison | AUC first > second | p-value |")
    lines.append("|---|---:|---:|")
    for name, test in part_a["tests"].items():
        lines.append(f"| `{name}` | {fmt_pct(test.get('auc_a_greater_b'))} | {fmt_num(test.get('p_value'), 4)} |")
    lines.append("")
    lines.append(f"**{part_a['verdict']}**")
    lines.append("")
    lines.append(part_a["reason"])
    lines.append("")
    lines.append("## Part B: External-Disagreement Identification")
    lines.append("")
    ext = summary["part_b"]
    cov = ext["coverage"]
    lines.append(f"External-score coverage: `{cov['rows_with_external']}/{cov['all_rows']}` rows; critical `{cov['critical_with_external']}`, non-critical `{cov['noncritical_with_external']}`.")
    lines.append("")
    lines.append("| signal | n | oriented AUC |")
    lines.append("|---|---:|---:|")
    for item in ext["signals"]:
        lines.append(f"| `{item['signal']}` | {item['n']} | {fmt_pct(item.get('oriented_auc'))} |")
    lines.append("")
    lines.append("Circularity guard:")
    lines.append("")
    lines.append(f"- greedy-wrong AUC reference: `{fmt_pct(ext.get('greedy_wrong_auc_reference'))}`")
    lines.append(f"- best external increment over greedy-wrong: `{fmt_pct(ext.get('increment_over_greedy_wrong_auc'))}`")
    lines.append(f"- within greedy-wrong external AUC: `{fmt_pct(ext.get('within_greedy_wrong_auc'))}`")
    lines.append("")
    lines.append(f"**{ext['verdict']}**")
    lines.append("")
    lines.append(ext["reason"])
    lines.append("")
    lines.append("## Part C: Operational Triage")
    lines.append("")
    if summary["part_c"]:
        lines.append("| budget | selected | recall | precision | random recall | coverage rows |")
        lines.append("|---:|---:|---:|---:|---:|---:|")
        for row in summary["part_c"]:
            lines.append(f"| {row['budget_fraction']*100:.0f}% | {row['selected_steps']} | {fmt_pct(row['recall'])} | {fmt_pct(row['precision'])} | {fmt_pct(row['random_recall'])} | {row['coverage_rows']} |")
    else:
        lines.append("External triage is not reported because external scores do not cover both critical and non-critical rows.")
    lines.append("")
    lines.append("## Gate")
    lines.append("")
    gate = summary["gate"]
    lines.append(f"**{gate['verdict']}**")
    lines.append("")
    lines.append(gate["reason"])
    lines.append("")
    lines.append("## Artifacts")
    lines.append("")
    lines.append(f"- `{output_dir / 'confident_error.md'}`")
    lines.append(f"- `{output_dir / 'summary.json'}`")
    lines.append(f"- `{output_dir / 'per_step.jsonl'}`")
    lines.append("")
    lines.append("STOP for review.")
    return "\n".join(lines) + "\n"


def decide_gate(part_a: Mapping[str, Any], part_b: Mapping[str, Any]) -> Dict[str, str]:
    if not str(part_a.get("verdict", "")).startswith("CONFIDENT ERROR"):
        return {"verdict": "NOT CONFIDENT ERROR", "reason": "Part A does not support the confident-error mechanism; the external-identification derivation is not established."}
    if part_b.get("verdict") == "GENUINE EXTERNAL SIGNAL":
        return {"verdict": "CONFIDENT ERROR + EXTERNAL IDENTIFICATION", "reason": "Critical steps look like confident errors and external disagreement identifies them beyond greedy correctness."}
    if part_b.get("verdict") == "EXTERNAL IDENTIFICATION NOT TESTABLE":
        return {"verdict": "CONFIDENT ERROR, EXTERNAL TEST BLOCKED", "reason": "Part A supports/mixes toward confident error, but all-step verifier scores are missing for non-critical rows, so the circularity-safe external test cannot be run honestly."}
    return {"verdict": "CONFIDENT ERROR, NO GENUINE IDENTIFICATION", "reason": "Part A supports/mixes toward confident error, but external disagreement is circular, weak, or not better than internal uncertainty."}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", default=DEFAULT_CANDIDATES)
    parser.add_argument("--crit-tasks", default=DEFAULT_CRIT_TASKS)
    parser.add_argument("--identify-summary", default=DEFAULT_IDENTIFY_SUMMARY)
    parser.add_argument("--internal-baseline-auc", type=float, default=0.634)
    parser.add_argument("--verifier-scored", nargs="*", default=[
        "outputs/critstep_verifier_v2/stage1_eval_200_k8_verdict/stage1_per_step.jsonl",
        "outputs/critstep_verifier/eval_overnight/per_step.jsonl",
    ])
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows, manifest = build_rows(args)
    if not rows:
        raise SystemExit("no rows available")
    part_a = confidence_groups(rows)
    internal = identify_internal_baseline(Path(args.identify_summary), args.internal_baseline_auc)
    part_b = external_metrics(rows, args.internal_baseline_auc)
    best_external = part_b.get("best_signal", {}).get("signal")
    part_c = []
    if best_external and part_b["coverage"]["critical_with_external"] > 0 and part_b["coverage"]["noncritical_with_external"] > 0:
        part_c = triage(rows, str(best_external), BUDGETS)
    gate = decide_gate(part_a, part_b)
    summary = {
        "inputs": {"candidates": args.candidates, "crit_tasks": args.crit_tasks, "verifier_scored": args.verifier_scored},
        "dataset": {
            "rows": len(rows),
            "critical_steps": sum(1 for row in rows if row["critical"]),
            "critical_prevalence": sum(1 for row in rows if row["critical"]) / len(rows),
            "greedy_wrong_steps": sum(1 for row in rows if not row["greedy_correct"]),
            "manifest": manifest,
        },
        "internal_reference_auc": args.internal_baseline_auc,
        "internal_baseline": internal,
        "part_a": part_a,
        "part_b": part_b,
        "part_c": part_c,
        "gate": gate,
    }
    per_step = []
    for row in rows:
        per_step.append({
            key: row.get(key)
            for key in [
                "target_id", "episode_id", "step_idx", "critical", "p_i_heldout_label_only", "greedy_correct", "recoverable_longtail",
                "model_confidence", "greedy_action_support", "greedy_control_support", "one_minus_action_entropy_norm", "one_minus_control_entropy_norm",
                "greedy_same_sample_logprob_mean", "greedy_same_sample_logprob_max", "greedy_external_score", "best_external_score",
                "best_non_greedy_external_score", "external_best_minus_greedy", "external_best_non_greedy_minus_greedy",
                "external_prefers_non_greedy", "model_conf_minus_external_greedy", "external_score_available", "longtail_correct_items",
            ]
        })
    write_jsonl(output_dir / "per_step.jsonl", per_step)
    write_json(output_dir / "summary.json", summary)
    (output_dir / "confident_error.md").write_text(render_report(summary, output_dir), encoding="utf-8")
    print(json.dumps({
        "output_dir": str(output_dir),
        "rows": len(rows),
        "part_a": part_a["verdict"],
        "external_coverage": part_b["coverage"],
        "part_b": part_b["verdict"],
        "gate": gate["verdict"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()