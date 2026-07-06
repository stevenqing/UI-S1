#!/usr/bin/env python3
"""Decode-distribution misalignment diagnostic for critical-step identification.

This is the final GT-free signal attempt. It asks whether the sampled decode
distribution contains a higher-confidence non-greedy alternative than the greedy
action, without using candidate correctness or matcher reward to choose that
alternative.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy.stats import rankdata

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.critstep_identifiability import logistic_cv  # noqa: E402
from scripts.score_critstep_verifier_v2_cot_voting import candidate_distinct_key  # noqa: E402


DEFAULT_CANDIDATES = "outputs/verifier_e2e/slice200/candidates/per_step.jsonl"
DEFAULT_CRIT_TASKS = "outputs/critstep_eval/per_task.jsonl"
DEFAULT_IDENTIFY_SUMMARY = "outputs/critstep_identify/summary.json"
DEFAULT_OUTPUT_DIR = "outputs/critstep_misalign"
BUDGETS = (0.10, 0.20, 0.30)
SPEC_INTERNAL_BASELINE_AUC = 0.634
SPEC_INTERNAL_TOP20_RECALL = 0.335


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
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
        number = float(value)
        return number if math.isfinite(number) else None
    except (TypeError, ValueError):
        return None


def auc_score(labels: Sequence[int], scores: Sequence[float]) -> Optional[float]:
    label_array = np.asarray(labels, dtype=int)
    score_array = np.asarray(scores, dtype=float)
    finite_mask = np.isfinite(score_array)
    label_array = label_array[finite_mask]
    score_array = score_array[finite_mask]
    positive_count = int(np.sum(label_array == 1))
    negative_count = int(np.sum(label_array == 0))
    if positive_count == 0 or negative_count == 0:
        return None
    ranks = rankdata(score_array, method="average")
    positive_rank_sum = float(np.sum(ranks[label_array == 1]))
    return (positive_rank_sum - positive_count * (positive_count + 1) / 2.0) / (positive_count * negative_count)


def oriented_auc(labels: Sequence[int], scores: Sequence[float]) -> Optional[Tuple[float, str]]:
    auc_value = auc_score(labels, scores)
    if auc_value is None:
        return None
    if auc_value >= 0.5:
        return float(auc_value), "high"
    return 1.0 - float(auc_value), "low"


def entropy_from_counts(counts: Counter[str]) -> float:
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    return -sum((count / total) * math.log(count / total, 2) for count in counts.values() if count > 0)


def normalized_entropy(counts: Counter[str]) -> float:
    if len(counts) <= 1:
        return 0.0
    return entropy_from_counts(counts) / math.log(len(counts), 2)


def modal_fraction(counts: Counter[str]) -> float:
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    return counts.most_common(1)[0][1] / total


def decode_key(candidate: Mapping[str, Any]) -> str:
    try:
        return candidate_distinct_key(candidate)
    except Exception:
        control = candidate.get("control") if isinstance(candidate.get("control"), dict) else {}
        payload = {
            "action_signature": candidate.get("action_signature"),
            "pred_type": candidate.get("pred_type"),
            "control_key": control.get("key"),
            "control_rect": control.get("rect"),
        }
        return json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def candidate_logprob(candidate: Mapping[str, Any]) -> Optional[float]:
    return safe_float(candidate.get("model_logprob_avg"))


def load_tasks(path: Path) -> Dict[str, Dict[str, Any]]:
    return {str(row.get("episode_id")): row for row in read_jsonl(path)}


def best_non_greedy_by_support(candidates: Sequence[Mapping[str, Any]], greedy_key: str, counts: Counter[str]) -> Tuple[Optional[str], float, int]:
    total = max(1, len(candidates))
    eligible = [(key, count) for key, count in counts.items() if key != greedy_key]
    if not eligible:
        return None, 0.0, 0
    best_key, best_count = max(eligible, key=lambda item: (item[1], item[0]))
    return best_key, best_count / total, best_count


def logprob_by_decode_key(candidates: Sequence[Mapping[str, Any]]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for candidate in candidates:
        logprob = candidate_logprob(candidate)
        if logprob is None:
            continue
        key = decode_key(candidate)
        current = out.get(key)
        if current is None or logprob > current:
            out[key] = logprob
    return out


def rank_descending(values: Sequence[float], target_value: float) -> int:
    sorted_values = sorted(values, reverse=True)
    for index, value in enumerate(sorted_values, 1):
        if value == target_value:
            return index
    return len(sorted_values) + 1


def build_rows(args: argparse.Namespace) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    candidate_rows = read_jsonl(Path(args.candidates))
    tasks = load_tasks(Path(args.crit_tasks))
    out: List[Dict[str, Any]] = []
    skipped = Counter()
    for row in candidate_rows:
        episode_id = str(row.get("episode_id"))
        step_idx = int(row.get("step_idx") or 0)
        task = tasks.get(episode_id)
        if task is None:
            skipped["missing_task"] += 1
            continue
        candidates = row.get("candidates") if isinstance(row.get("candidates"), list) else []
        if not candidates:
            skipped["missing_candidates"] += 1
            continue
        bottom2 = {int(index) for index in task.get("bottom2_critical_indices", [])}
        per_step_p = task.get("per_step_p_heldout_cv") if isinstance(task.get("per_step_p_heldout_cv"), list) else []
        p_i = safe_float(per_step_p[step_idx]) if step_idx < len(per_step_p) else None
        keys = [decode_key(candidate) for candidate in candidates]
        counts = Counter(keys)
        total = len(candidates)
        greedy_key = keys[0]
        greedy_support = counts[greedy_key] / total
        best_non_greedy_key, best_non_greedy_support, best_non_greedy_count = best_non_greedy_by_support(candidates, greedy_key, counts)
        support_values = [count / total for count in counts.values()]
        greedy_support_rank = rank_descending(support_values, greedy_support)
        non_greedy_mass = 1.0 - greedy_support
        support_misalignment = best_non_greedy_support - greedy_support
        nongreedy_exceeds_greedy_support = float(best_non_greedy_support > greedy_support)
        decode_entropy_norm = normalized_entropy(counts)
        one_minus_modal = 1.0 - modal_fraction(counts)
        logprobs = logprob_by_decode_key(candidates[1:])
        greedy_tail_logprob = logprobs.get(greedy_key)
        non_greedy_logprob_items = [(key, value) for key, value in logprobs.items() if key != greedy_key]
        if non_greedy_logprob_items:
            best_non_greedy_logprob_key, best_non_greedy_logprob = max(non_greedy_logprob_items, key=lambda item: (item[1], item[0]))
        else:
            best_non_greedy_logprob_key, best_non_greedy_logprob = None, None
        logprob_misalignment = None
        logprob_rank = None
        nongreedy_exceeds_greedy_logprob = None
        if greedy_tail_logprob is not None and best_non_greedy_logprob is not None:
            logprob_misalignment = best_non_greedy_logprob - greedy_tail_logprob
            logprob_rank = rank_descending(list(logprobs.values()), greedy_tail_logprob)
            nongreedy_exceeds_greedy_logprob = float(best_non_greedy_logprob > greedy_tail_logprob)
        features = {
            "support_misalignment": support_misalignment,
            "best_non_greedy_support": best_non_greedy_support,
            "greedy_support": greedy_support,
            "non_greedy_mass": non_greedy_mass,
            "greedy_support_rank": float(greedy_support_rank),
            "nongreedy_exceeds_greedy_support": nongreedy_exceeds_greedy_support,
            "distinct_decode_count": float(len(counts)),
            "decode_entropy_norm": decode_entropy_norm,
            "one_minus_modal_decode_frac": one_minus_modal,
            "logprob_misalignment": logprob_misalignment,
            "best_non_greedy_logprob": best_non_greedy_logprob,
            "greedy_tail_logprob": greedy_tail_logprob,
            "greedy_logprob_rank": float(logprob_rank) if logprob_rank is not None else None,
            "nongreedy_exceeds_greedy_logprob": nongreedy_exceeds_greedy_logprob,
        }
        out.append({
            "target_id": row.get("target_id"),
            "episode_id": episode_id,
            "step_idx": step_idx,
            "critical": step_idx in bottom2,
            "p_i_heldout_label_only": p_i,
            "n_candidates": total,
            "greedy_decode_key": greedy_key,
            "best_non_greedy_decode_key": best_non_greedy_key,
            "best_non_greedy_logprob_key": best_non_greedy_logprob_key,
            "best_non_greedy_count": best_non_greedy_count,
            "features": features,
        })
    manifest = {"candidate_rows_in": len(candidate_rows), "rows_out": len(out), "skipped": dict(skipped)}
    return out, manifest


def feature_values(rows: Sequence[Mapping[str, Any]], feature_name: str, fill_missing: Optional[float] = 0.0) -> List[float]:
    values = []
    for row in rows:
        features = row.get("features") if isinstance(row.get("features"), dict) else {}
        value = features.get(feature_name)
        if value is None:
            if fill_missing is None:
                values.append(float("nan"))
            else:
                values.append(float(fill_missing))
        else:
            values.append(float(value))
    return values


def signal_metrics(rows: Sequence[Mapping[str, Any]], feature_names: Sequence[str]) -> List[Dict[str, Any]]:
    labels = [int(row["critical"]) for row in rows]
    out = []
    for feature_name in feature_names:
        values = feature_values(rows, feature_name, fill_missing=None)
        valid = [(label, value) for label, value in zip(labels, values) if math.isfinite(value)]
        valid_labels = [label for label, _ in valid]
        valid_values = [value for _, value in valid]
        oriented = oriented_auc(valid_labels, valid_values) if valid else None
        auc_value, direction = (oriented if oriented is not None else (None, "NA"))
        critical_values = [value for label, value in valid if label == 1]
        noncritical_values = [value for label, value in valid if label == 0]
        out.append({
            "feature": feature_name,
            "n_valid": len(valid),
            "oriented_auc": auc_value,
            "direction": direction,
            "mean_critical": float(np.mean(critical_values)) if critical_values else None,
            "mean_noncritical": float(np.mean(noncritical_values)) if noncritical_values else None,
        })
    out.sort(key=lambda item: item["oriented_auc"] if item["oriented_auc"] is not None else -1.0, reverse=True)
    return out


def rows_for_logistic(rows: Sequence[Mapping[str, Any]], feature_names: Sequence[str]) -> List[Dict[str, Any]]:
    converted = []
    medians: Dict[str, float] = {}
    for feature_name in feature_names:
        values = [row["features"].get(feature_name) for row in rows]
        finite = [float(value) for value in values if value is not None and math.isfinite(float(value))]
        medians[feature_name] = float(np.median(finite)) if finite else 0.0
    for row in rows:
        features = {}
        for feature_name in feature_names:
            value = row["features"].get(feature_name)
            features[feature_name] = medians[feature_name] if value is None else float(value)
        converted.append({"critical": bool(row["critical"]), "features": features})
    return converted


def triage(rows: Sequence[Mapping[str, Any]], score_values: Sequence[float], budgets: Sequence[float]) -> List[Dict[str, Any]]:
    labels = np.asarray([int(row["critical"]) for row in rows], dtype=int)
    scores = np.asarray(score_values, dtype=float)
    order = np.argsort(-scores)
    positive_total = int(np.sum(labels == 1))
    out = []
    for budget in budgets:
        selected_count = max(1, int(round(len(rows) * budget)))
        selected = order[:selected_count]
        hits = int(np.sum(labels[selected] == 1))
        out.append({
            "budget_fraction": budget,
            "selected_steps": selected_count,
            "recall": hits / positive_total if positive_total else 0.0,
            "precision": hits / selected_count if selected_count else 0.0,
            "random_recall": budget,
        })
    return out


def load_internal_summary(path: Path, spec_auc: float, spec_top20: float) -> Dict[str, Any]:
    if not path.exists():
        return {"current_auc": None, "current_top20_recall": None, "source": None, "spec_auc": spec_auc, "spec_top20_recall": spec_top20}
    data = json.loads(path.read_text(encoding="utf-8"))
    current_auc = data.get("classifier", {}).get("auc")
    top20 = None
    for item in data.get("triage_primary", []):
        if abs(float(item.get("budget_fraction", -1.0)) - 0.20) < 1e-6:
            top20 = item.get("recall")
            break
    return {"current_auc": current_auc, "current_top20_recall": top20, "source": str(path), "spec_auc": spec_auc, "spec_top20_recall": spec_top20}


def rank_rows(rows: List[Dict[str, Any]], scores: Sequence[float]) -> None:
    order = np.argsort(-np.asarray(scores, dtype=float))
    for rank, index in enumerate(order, 1):
        rows[int(index)]["misalignment_score"] = float(scores[int(index)])
        rows[int(index)]["triage_rank"] = rank
        rows[int(index)]["triage_percentile"] = rank / len(rows)


def fmt_pct(value: Optional[float]) -> str:
    if value is None:
        return "NA"
    return f"{100.0 * value:.2f}%"


def fmt_num(value: Optional[float], digits: int = 3) -> str:
    if value is None:
        return "NA"
    return f"{value:.{digits}f}"


def decide_gate(primary_auc: Optional[float], combined_increment: Optional[float], triage20: Optional[float], internal: Mapping[str, Any]) -> Dict[str, str]:
    spec_auc = float(internal.get("spec_auc") or SPEC_INTERNAL_BASELINE_AUC)
    spec_top20 = float(internal.get("spec_top20_recall") or SPEC_INTERNAL_TOP20_RECALL)
    if primary_auc is not None and primary_auc >= 0.70 and primary_auc >= spec_auc + 0.05 and (combined_increment or 0.0) >= 0.03 and triage20 is not None and triage20 >= spec_top20 + 0.10:
        return {
            "verdict": "IDENTIFICATION SIGNAL FOUND",
            "reason": "Decode-distribution misalignment clearly beats the internal baseline, increments the internal set, and improves triage.",
        }
    return {
        "verdict": "IDENTIFICATION REMAINS WEAK",
        "reason": "Misalignment does not clearly beat the 0.634 internal baseline and/or does not provide a meaningful triage increment. Per convergence commitment, critical-step identification remains an open limitation.",
    }


def render_report(summary: Mapping[str, Any], output_dir: Path) -> str:
    lines = ["# Decode-Distribution Misalignment Signal", ""]
    lines.append("Diagnostic only: zero training. Critical bottom-2 held-out p_i is the label only; best non-greedy is chosen by confidence regardless of correctness.")
    lines.append("")
    lines.append("## Scope")
    lines.append("")
    lines.append(f"- sampled pool: `{summary['inputs']['candidates']}`")
    lines.append(f"- rows analyzed: `{summary['dataset']['rows']}` steps")
    lines.append(f"- critical prevalence: `{summary['dataset']['critical_prevalence']*100:.2f}%`")
    lines.append(f"- direct greedy logprob available: `{summary['coverage']['direct_greedy_logprob_available']}` rows; primary confidence is support-based decode mass.")
    lines.append("")
    lines.append("## Metric 1: Misalignment AUC")
    lines.append("")
    lines.append("| signal | valid rows | AUC | direction | critical mean | non-critical mean |")
    lines.append("|---|---:|---:|---|---:|---:|")
    for item in summary["signals"]:
        lines.append(
            f"| `{item['feature']}` | {item['n_valid']} | {fmt_pct(item.get('oriented_auc'))} | {item['direction']} | "
            f"{fmt_num(item.get('mean_critical'))} | {fmt_num(item.get('mean_noncritical'))} |"
        )
    lines.append("")
    lines.append("Primary signal: `support_misalignment = support(best non-greedy decode) - support(greedy decode)`. This is GT-free; candidate correctness/reward are not used to select the alternative.")
    lines.append("")
    lines.append("## Metric 2: Baseline Comparison and Increment")
    lines.append("")
    internal = summary["internal_baseline"]
    lines.append("| comparison | AUC / recall |")
    lines.append("|---|---:|")
    lines.append(f"| spec internal baseline AUC to beat | {fmt_pct(internal.get('spec_auc'))} |")
    lines.append(f"| current covered-slice internal AUC | {fmt_pct(internal.get('current_auc'))} |")
    lines.append(f"| primary misalignment AUC | {fmt_pct(summary['primary_signal']['auc'])} |")
    lines.append(f"| internal uncertainty/disagreement CV AUC | {fmt_pct(summary['combined']['internal_auc'])} |")
    lines.append(f"| internal + misalignment CV AUC | {fmt_pct(summary['combined']['combined_auc'])} |")
    lines.append(f"| increment from adding misalignment | {fmt_pct(summary['combined']['increment'])} |")
    lines.append("")
    lines.append("## Metric 3: Operational Triage")
    lines.append("")
    lines.append("| budget | selected steps | misalignment recall | precision | random recall | internal baseline recall |")
    lines.append("|---:|---:|---:|---:|---:|---:|")
    internal_top20 = internal.get("current_top20_recall") or internal.get("spec_top20_recall")
    for item in summary["triage"]:
        baseline = internal_top20 if abs(float(item["budget_fraction"]) - 0.20) < 1e-6 else None
        lines.append(
            f"| {item['budget_fraction']*100:.0f}% | {item['selected_steps']} | {fmt_pct(item['recall'])} | "
            f"{fmt_pct(item['precision'])} | {fmt_pct(item['random_recall'])} | {fmt_pct(baseline)} |"
        )
    lines.append("")
    lines.append("## Leakage Audit")
    lines.append("")
    lines.append("- best non-greedy candidate is selected by decode confidence/support, not by matcher correctness.")
    lines.append("- `reward`, `is_correct`, GT action, and p_i are not used as signal features.")
    lines.append(f"- suspicious AUC>=0.90 signals: `{summary['leakage_audit']['suspicious_high_auc_signals']}`")
    lines.append("")
    lines.append("## Gate")
    lines.append("")
    lines.append(f"**{summary['gate']['verdict']}**")
    lines.append("")
    lines.append(summary["gate"]["reason"])
    lines.append("")
    lines.append("Convergence commitment: this is the last critical-step identification signal attempt. If weak, the paper should frame inference-time critical-step identification as an honest open limitation and keep the method every-step or explicitly post-hoc.")
    lines.append("")
    lines.append("## Artifacts")
    lines.append("")
    lines.append(f"- `{output_dir / 'misalignment.md'}`")
    lines.append(f"- `{output_dir / 'summary.json'}`")
    lines.append(f"- `{output_dir / 'per_step.jsonl'}`")
    lines.append("")
    lines.append("STOP for review.")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", default=DEFAULT_CANDIDATES)
    parser.add_argument("--crit-tasks", default=DEFAULT_CRIT_TASKS)
    parser.add_argument("--identify-summary", default=DEFAULT_IDENTIFY_SUMMARY)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--internal-baseline-auc", type=float, default=SPEC_INTERNAL_BASELINE_AUC)
    parser.add_argument("--internal-top20-recall", type=float, default=SPEC_INTERNAL_TOP20_RECALL)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=43)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows, manifest = build_rows(args)
    if not rows:
        raise SystemExit("no rows available")
    labels = [int(row["critical"]) for row in rows]
    signal_names = [
        "support_misalignment",
        "best_non_greedy_support",
        "greedy_support",
        "non_greedy_mass",
        "greedy_support_rank",
        "nongreedy_exceeds_greedy_support",
        "distinct_decode_count",
        "decode_entropy_norm",
        "one_minus_modal_decode_frac",
        "logprob_misalignment",
        "best_non_greedy_logprob",
        "greedy_tail_logprob",
        "greedy_logprob_rank",
        "nongreedy_exceeds_greedy_logprob",
    ]
    signals = signal_metrics(rows, signal_names)
    primary_values = feature_values(rows, "support_misalignment", fill_missing=0.0)
    primary_auc = auc_score(labels, primary_values)
    # Keep the sign fixed by the mechanistic hypothesis: larger positive gap predicts critical.
    rank_rows(rows, primary_values)
    triage_rows = triage(rows, primary_values, BUDGETS)
    internal_features = [
        "distinct_decode_count",
        "decode_entropy_norm",
        "one_minus_modal_decode_frac",
        "non_greedy_mass",
        "greedy_support_rank",
    ]
    misalignment_features = ["support_misalignment", "best_non_greedy_support", "nongreedy_exceeds_greedy_support"]
    internal_cv = logistic_cv(rows_for_logistic(rows, internal_features), internal_features, args.folds, args.seed)
    combined_cv = logistic_cv(rows_for_logistic(rows, internal_features + misalignment_features), internal_features + misalignment_features, args.folds, args.seed)
    internal = load_internal_summary(Path(args.identify_summary), args.internal_baseline_auc, args.internal_top20_recall)
    gate = decide_gate(primary_auc, (combined_cv.get("auc") or 0.0) - (internal_cv.get("auc") or 0.0), next(item["recall"] for item in triage_rows if abs(item["budget_fraction"] - 0.20) < 1e-6), internal)
    suspicious = [item["feature"] for item in signals if (item.get("oriented_auc") or 0.0) >= 0.90]
    summary = {
        "inputs": {"candidates": args.candidates, "crit_tasks": args.crit_tasks, "identify_summary": args.identify_summary},
        "dataset": {
            "rows": len(rows),
            "critical_steps": int(sum(labels)),
            "critical_prevalence": float(sum(labels) / len(labels)),
            "manifest": manifest,
        },
        "coverage": {
            "direct_greedy_logprob_available": sum(1 for row in rows if row["features"].get("greedy_tail_logprob") is not None and row["features"].get("greedy_decode_key") is not None),
            "greedy_tail_logprob_available": sum(1 for row in rows if row["features"].get("greedy_tail_logprob") is not None),
            "logprob_misalignment_available": sum(1 for row in rows if row["features"].get("logprob_misalignment") is not None),
        },
        "signals": signals,
        "primary_signal": {"feature": "support_misalignment", "auc": primary_auc, "note": "fixed direction: larger best-non-greedy support minus greedy support predicts critical"},
        "internal_baseline": internal,
        "combined": {
            "internal_features": internal_features,
            "misalignment_features_added": misalignment_features,
            "internal_auc": internal_cv.get("auc"),
            "internal_balanced_accuracy": internal_cv.get("balanced_accuracy"),
            "combined_auc": combined_cv.get("auc"),
            "combined_balanced_accuracy": combined_cv.get("balanced_accuracy"),
            "increment": (combined_cv.get("auc") or 0.0) - (internal_cv.get("auc") or 0.0),
        },
        "triage": triage_rows,
        "leakage_audit": {
            "excluded_fields": ["candidate.reward", "candidate.is_correct", "row.gt_action", "task.per_step_p_heldout_cv as feature"],
            "suspicious_high_auc_signals": suspicious,
        },
        "gate": gate,
    }
    per_step_rows = []
    for row in rows:
        features = row["features"]
        per_step_rows.append({
            "target_id": row["target_id"],
            "episode_id": row["episode_id"],
            "step_idx": row["step_idx"],
            "critical": row["critical"],
            "p_i_heldout_label_only": row["p_i_heldout_label_only"],
            "greedy_conf_support": features["greedy_support"],
            "best_non_greedy_conf_support": features["best_non_greedy_support"],
            "support_misalignment": features["support_misalignment"],
            "greedy_support_rank": features["greedy_support_rank"],
            "nongreedy_exceeds_greedy_support": features["nongreedy_exceeds_greedy_support"],
            "non_greedy_mass": features["non_greedy_mass"],
            "greedy_tail_logprob": features["greedy_tail_logprob"],
            "best_non_greedy_logprob": features["best_non_greedy_logprob"],
            "logprob_misalignment": features["logprob_misalignment"],
            "misalignment_score": row.get("misalignment_score"),
            "triage_rank": row.get("triage_rank"),
            "triage_percentile": row.get("triage_percentile"),
            "greedy_decode_key": row["greedy_decode_key"],
            "best_non_greedy_decode_key": row["best_non_greedy_decode_key"],
            "best_non_greedy_logprob_key": row["best_non_greedy_logprob_key"],
        })
    write_jsonl(output_dir / "per_step.jsonl", per_step_rows)
    write_json(output_dir / "summary.json", summary)
    (output_dir / "misalignment.md").write_text(render_report(summary, output_dir), encoding="utf-8")
    print(json.dumps({
        "output_dir": str(output_dir),
        "rows": len(rows),
        "primary_auc": primary_auc,
        "combined_increment": summary["combined"]["increment"],
        "top20_recall": next(item["recall"] for item in triage_rows if abs(item["budget_fraction"] - 0.20) < 1e-6),
        "gate": gate["verdict"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()