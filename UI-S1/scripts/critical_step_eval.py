#!/usr/bin/env python3
"""Critical-step-aware evaluation for GUI-360 compound bottlenecks.

Pure diagnostic over baseline per-task compound proof outputs. Critical steps are
defined by held-out per-step p_i, not by the same instance's outcome.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple


EPS = 1e-12


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def median(values: Sequence[float]) -> float:
    return statistics.median(values) if values else 0.0


def quantile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return ordered[lo]
    return ordered[lo] * (hi - pos) + ordered[hi] * (pos - lo)


def product(values: Iterable[float]) -> float:
    out = 1.0
    for value in values:
        out *= max(EPS, min(1.0 - EPS, value))
    return out


def pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def wilson(success: int, total: int, z: float = 1.96) -> Tuple[float, float]:
    if total <= 0:
        return (0.0, 0.0)
    p = success / total
    denom = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denom
    margin = z * math.sqrt((p * (1 - p) + z * z / (4 * total)) / total) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))


def bottom_indices(ps: Sequence[float], k: int) -> List[int]:
    n = len(ps)
    keep = min(k, n)
    return [idx for idx, _ in sorted(enumerate(ps), key=lambda item: (item[1], item[0]))[:keep]]


def neg_log_share(ps: Sequence[float], indices: Sequence[int]) -> float:
    weights = [-math.log(max(EPS, min(1.0 - EPS, value))) for value in ps]
    denom = sum(weights)
    if denom <= 0:
        return 0.0
    return sum(weights[idx] for idx in indices) / denom


def normalized_position(idx: int, k: int) -> float:
    if k <= 1:
        return 0.0
    return idx / (k - 1)


def summarize_values(values: Sequence[float]) -> Dict[str, float]:
    return {
        "mean": mean(values),
        "median": median(values),
        "p25": quantile(values, 0.25),
        "p75": quantile(values, 0.75),
    }


def step_records(tasks: Sequence[Dict[str, Any]], critical_k: int) -> List[Dict[str, Any]]:
    rows = []
    for task in tasks:
        ps = [float(value) for value in task["per_step_p_heldout_cv"]]
        ys = [bool(value) for value in task["per_step_success"]]
        features = task.get("step_features") or [{} for _ in ps]
        k = len(ps)
        crit = set(bottom_indices(ps, critical_k))
        for idx, p in enumerate(ps):
            feature = features[idx] if idx < len(features) else {}
            rows.append(
                {
                    "episode_id": task["episode_id"],
                    "k": k,
                    "idx": idx,
                    "p": p,
                    "success": ys[idx],
                    "critical": idx in crit,
                    "norm_pos": normalized_position(idx, k),
                    "action_type": feature.get("action_type", "unknown"),
                    "bbox_area_bin": feature.get("bbox_area_bin", "unknown"),
                    "position_bin": feature.get("position_bin", "unknown"),
                    "step_phase": feature.get("step_phase", "unknown"),
                    "label_detail": feature.get("label_detail", "unknown"),
                }
            )
    return rows


def cssr_ncsr(tasks: Sequence[Dict[str, Any]], critical_k: int) -> Dict[str, Any]:
    critical_total = critical_correct = non_total = non_correct = 0
    for row in step_records(tasks, critical_k):
        if row["critical"]:
            critical_total += 1
            critical_correct += 1 if row["success"] else 0
        else:
            non_total += 1
            non_correct += 1 if row["success"] else 0
    cssr = critical_correct / critical_total if critical_total else 0.0
    ncsr = non_correct / non_total if non_total else 0.0
    return {
        "critical_correct": critical_correct,
        "critical_total": critical_total,
        "noncritical_correct": non_correct,
        "noncritical_total": non_total,
        "cssr": cssr,
        "ncsr": ncsr,
        "gap_cssr_minus_ncsr": cssr - ncsr,
        "cssr_ci95": wilson(critical_correct, critical_total),
        "ncsr_ci95": wilson(non_correct, non_total),
    }


def bootstrap_gap(tasks: Sequence[Dict[str, Any]], critical_k: int, n_boot: int, seed: int) -> Tuple[float, float]:
    rng = random.Random(seed)
    values = []
    task_list = list(tasks)
    for _ in range(n_boot):
        sample = [rng.choice(task_list) for _ in task_list]
        values.append(cssr_ncsr(sample, critical_k)["gap_cssr_minus_ncsr"])
    return (quantile(values, 0.025), quantile(values, 0.975))


def oracle_metrics(tasks: Sequence[Dict[str, Any]], critical_k: int) -> Dict[str, Any]:
    predicted_baseline = []
    oracle_predicted = []
    empirical_oracle_success = []
    log_shares = []
    failed = 0
    failed_any_critical = 0
    failed_first_critical = 0
    failed_only_critical = 0
    for task in tasks:
        ps = [float(value) for value in task["per_step_p_heldout_cv"]]
        ys = [bool(value) for value in task["per_step_success"]]
        crit = set(bottom_indices(ps, critical_k))
        predicted_baseline.append(product(ps))
        oracle_ps = [1.0 if idx in crit else value for idx, value in enumerate(ps)]
        oracle_predicted.append(product(oracle_ps))
        empirical_oracle_success.append(all(ys[idx] or idx in crit for idx in range(len(ys))))
        log_shares.append(neg_log_share(ps, sorted(crit)))
        actual_success = all(ys)
        if not actual_success:
            failed += 1
            failed_critical_indices = [idx for idx in crit if not ys[idx]]
            failed_noncritical_indices = [idx for idx in range(len(ys)) if idx not in crit and not ys[idx]]
            if failed_critical_indices:
                failed_any_critical += 1
            if failed_critical_indices and not failed_noncritical_indices:
                failed_only_critical += 1
            first_failed = next((idx for idx, value in enumerate(ys) if not value), None)
            if first_failed in crit:
                failed_first_critical += 1
    return {
        "predicted_baseline_tsr": mean(predicted_baseline),
        "oracle_critical_predicted_tsr": mean(oracle_predicted),
        "oracle_critical_predicted_lift": mean(oracle_predicted) - mean(predicted_baseline),
        "empirical_oracle_critical_tsr": mean([1.0 if value else 0.0 for value in empirical_oracle_success]),
        "log_failure_share_mean": mean(log_shares),
        "log_failure_share_median": median(log_shares),
        "log_failure_share_p25": quantile(log_shares, 0.25),
        "log_failure_share_p75": quantile(log_shares, 0.75),
        "failed_tasks": failed,
        "failed_any_critical_fraction": failed_any_critical / failed if failed else 0.0,
        "failed_first_critical_fraction": failed_first_critical / failed if failed else 0.0,
        "failed_only_critical_fraction": failed_only_critical / failed if failed else 0.0,
    }


def distribution(rows: Sequence[Dict[str, Any]], key: str, *, critical: bool, top_n: int = 12) -> List[Tuple[str, int, float]]:
    selected = [row for row in rows if row["critical"] == critical]
    total = len(selected)
    counts = Counter(str(row.get(key, "unknown")) for row in selected)
    return [(label, count, count / total if total else 0.0) for label, count in counts.most_common(top_n)]


def bucket_success(rows: Sequence[Dict[str, Any]], key: str, min_n: int) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, bool], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row.get(key, "unknown")), bool(row["critical"]))].append(row)
    labels = sorted({label for label, _ in grouped})
    out = []
    for label in labels:
        crit_rows = grouped.get((label, True), [])
        non_rows = grouped.get((label, False), [])
        if len(crit_rows) < min_n or len(non_rows) < min_n:
            continue
        cssr = mean([1.0 if row["success"] else 0.0 for row in crit_rows])
        ncsr = mean([1.0 if row["success"] else 0.0 for row in non_rows])
        out.append({"bucket": label, "critical_n": len(crit_rows), "noncritical_n": len(non_rows), "cssr": cssr, "ncsr": ncsr, "gap": cssr - ncsr})
    return sorted(out, key=lambda item: item["gap"])


def per_task_rows(tasks: Sequence[Dict[str, Any]], critical_ks: Sequence[int], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for task in tasks:
            ps = [float(value) for value in task["per_step_p_heldout_cv"]]
            ys = [bool(value) for value in task["per_step_success"]]
            payload: Dict[str, Any] = {
                "episode_id": task["episode_id"],
                "k": int(task["k"]),
                "actual_success": bool(task["actual_success"]),
                "actual_progress": float(task.get("actual_progress", 0.0)),
                "predicted_prob_heldout_cv": float(task["predicted_prob_heldout_cv"]),
                "per_step_p_heldout_cv": ps,
                "per_step_success": ys,
            }
            for critical_k in critical_ks:
                crit = bottom_indices(ps, critical_k)
                crit_set = set(crit)
                oracle_ps = [1.0 if idx in crit_set else value for idx, value in enumerate(ps)]
                payload[f"bottom{critical_k}_critical_indices"] = crit
                payload[f"bottom{critical_k}_critical_p"] = [ps[idx] for idx in crit]
                payload[f"bottom{critical_k}_critical_success"] = [ys[idx] for idx in crit]
                payload[f"bottom{critical_k}_critical_log_failure_share"] = neg_log_share(ps, crit)
                payload[f"bottom{critical_k}_oracle_critical_predicted_success"] = product(oracle_ps)
                payload[f"bottom{critical_k}_empirical_oracle_success"] = all(ys[idx] or idx in crit_set for idx in range(len(ys)))
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def gate(summary: Dict[int, Dict[str, Any]]) -> Dict[str, Any]:
    bottom2 = summary[2]
    gap_ci = bottom2["cssr_ncsr"]["gap_boot_ci95"]
    cssr_gap_negative = gap_ci[1] < 0
    share_high = bottom2["oracle"]["log_failure_share_mean"] >= 0.50
    oracle_lift_substantial = bottom2["oracle"]["oracle_critical_predicted_lift"] >= 0.05
    failure_attribution_high = bottom2["oracle"]["failed_any_critical_fraction"] >= 0.50
    if cssr_gap_negative and share_high and oracle_lift_substantial:
        verdict = "BOTTLENECK_IDENTIFIED"
        explanation = "Critical steps have significantly lower success, dominate the log-failure product, and oracle-critical fixing yields a substantial compound-model lift."
    else:
        verdict = "WEAK_BOTTLENECK"
        explanation = "The critical-step gap, concentration, or oracle lift is not strong enough under the preset thresholds."
    return {
        "verdict": verdict,
        "explanation": explanation,
        "checks": {
            "bottom2_gap_ci95": gap_ci,
            "cssr_gap_negative": cssr_gap_negative,
            "bottom2_log_share_mean": bottom2["oracle"]["log_failure_share_mean"],
            "log_share_high": share_high,
            "bottom2_oracle_lift": bottom2["oracle"]["oracle_critical_predicted_lift"],
            "oracle_lift_substantial": oracle_lift_substantial,
            "failed_any_critical_fraction": bottom2["oracle"]["failed_any_critical_fraction"],
            "failure_attribution_high": failure_attribution_high,
        },
    }


def table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(item) for item in row) + " |")
    return "\n".join(lines)


def write_report(path: Path, tasks: Sequence[Dict[str, Any]], fit: Dict[str, Any], summary: Dict[int, Dict[str, Any]], gate_data: Dict[str, Any]) -> None:
    lines = []
    lines.append("# Critical-Step-Aware GUI-360 Evaluation")
    lines.append("")
    lines.append("Date: 2026-06-30")
    lines.append("")
    lines.append("## Scope")
    lines.append("")
    lines.append("Regime: original SFT baseline `checkpoints/gui360-fullparam-sft-step250`, original GUI-360 template, GT-history teacher-forced, frozen `compute_step_reward` matcher.")
    lines.append("")
    lines.append("Critical-step definition: bottom-k by held-out per-step p_i within each task. Reported k: bottom-1 and bottom-2. The same held-out p_i from the compound proof is used; no outcome is used to choose critical steps.")
    lines.append("")
    lines.append(f"Dataset: {len(tasks)} tasks / {fit['n_steps']} steps. Actual TSR: {pct(fit['layer1']['actual_tsr'])}. StepSR: {pct(fit['step_sr'])}.")
    lines.append("")
    lines.append("## Metric 1 - Critical-Step Distribution")
    lines.append("")
    for critical_k, data in summary.items():
        lines.append(f"### Bottom-{critical_k}")
        lines.append("")
        lines.append(table(["quantity", "critical", "non-critical"], [
            ["step count", data["count_critical"], data["count_noncritical"]],
            ["mean p_i", f"{data['p_critical']['mean']:.4f}", f"{data['p_noncritical']['mean']:.4f}"],
            ["median p_i", f"{data['p_critical']['median']:.4f}", f"{data['p_noncritical']['median']:.4f}"],
            ["mean normalized position", f"{data['mean_norm_pos_critical']:.3f}", f"{data['mean_norm_pos_noncritical']:.3f}"],
        ]))
        lines.append("")
        lines.append("Critical action distribution:")
        lines.append(table(["action", "n", "share"], [[label, count, pct(share)] for label, count, share in data["critical_action_distribution"]]))
        lines.append("")
        lines.append("Critical phase distribution:")
        lines.append(table(["phase", "n", "share"], [[label, count, pct(share)] for label, count, share in data["critical_phase_distribution"]]))
        lines.append("")
    lines.append("## Metric 2 - Critical Step Success Rate")
    lines.append("")
    rows = []
    for critical_k, data in summary.items():
        metric = data["cssr_ncsr"]
        rows.append([
            f"bottom-{critical_k}",
            f"{metric['critical_correct']} / {metric['critical_total']}",
            pct(metric["cssr"]),
            f"{metric['noncritical_correct']} / {metric['noncritical_total']}",
            pct(metric["ncsr"]),
            pct(metric["gap_cssr_minus_ncsr"]),
            f"[{pct(metric['gap_boot_ci95'][0])}, {pct(metric['gap_boot_ci95'][1])}]",
        ])
    lines.append(table(["critical set", "critical correct", "CSSR", "non-critical correct", "NCSR", "CSSR-NCSR", "gap bootstrap CI"], rows))
    lines.append("")
    lines.append("Action-bucket gaps with n>=30 per side:")
    for critical_k, data in summary.items():
        lines.append("")
        lines.append(f"Bottom-{critical_k}:")
        lines.append(table(["action", "critical n", "non-critical n", "CSSR", "NCSR", "gap"], [[row["bucket"], row["critical_n"], row["noncritical_n"], pct(row["cssr"]), pct(row["ncsr"]), pct(row["gap"])] for row in data["action_bucket_success"]]))
    lines.append("")
    lines.append("## Metric 3 - Oracle-Critical Compound Ceiling")
    lines.append("")
    rows = []
    for critical_k, data in summary.items():
        oracle = data["oracle"]
        rows.append([
            f"bottom-{critical_k}",
            pct(oracle["predicted_baseline_tsr"]),
            pct(oracle["oracle_critical_predicted_tsr"]),
            pct(oracle["oracle_critical_predicted_lift"]),
            pct(oracle["empirical_oracle_critical_tsr"]),
        ])
    lines.append(table(["critical set", "compound baseline", "oracle-critical predicted TSR", "predicted lift", "observed-outcome critical-fix TSR"], rows))
    lines.append("")
    lines.append("Oracle-critical predicted TSR is a compound-model ceiling: it sets critical p_i to 1 in the held-out product. The observed-outcome critical-fix number is a descriptive counterfactual over observed failures, not an achieved model score.")
    lines.append("")
    lines.append("## Metric 4 - Failure Concentration")
    lines.append("")
    rows = []
    for critical_k, data in summary.items():
        oracle = data["oracle"]
        rows.append([
            f"bottom-{critical_k}",
            pct(oracle["log_failure_share_mean"]),
            pct(oracle["log_failure_share_median"]),
            f"{pct(oracle['log_failure_share_p25'])} - {pct(oracle['log_failure_share_p75'])}",
            pct(oracle["failed_any_critical_fraction"]),
            pct(oracle["failed_first_critical_fraction"]),
            pct(oracle["failed_only_critical_fraction"]),
        ])
    lines.append(table(["critical set", "mean log-failure share", "median", "IQR", "failed tasks with critical fail", "first failure critical", "only critical failed"], rows))
    lines.append("")
    lines.append("## P4/P5 Method Criteria Enabled")
    lines.append("")
    lines.append("P4 Critical-step targeting: future self-distillation runs should report Delta CSSR vs Delta NCSR. A useful method should concentrate improvement on CSSR.")
    lines.append("")
    lines.append("P5 Compound amplification: plug the measured critical-step p_i improvement into the held-out product to predict Delta TSR, then compare measured Delta TSR to the predicted lift.")
    lines.append("")
    lines.append("## Gate Verdict")
    lines.append("")
    lines.append(f"Verdict: **{gate_data['verdict']}**")
    lines.append("")
    lines.append(gate_data["explanation"])
    lines.append("")
    lines.append(table(["check", "value"], [[key, value] for key, value in gate_data["checks"].items()]))
    lines.append("")
    lines.append("## Artifacts")
    lines.append("")
    lines.append("- `outputs/critstep_eval/critstep_eval.md`")
    lines.append("- `outputs/critstep_eval/critstep_eval.json`")
    lines.append("- `outputs/critstep_eval/per_task.jsonl`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def summarize(tasks: Sequence[Dict[str, Any]], fit: Dict[str, Any], critical_ks: Sequence[int], n_boot: int, seed: int, min_bucket_n: int) -> Dict[int, Dict[str, Any]]:
    out = {}
    for critical_k in critical_ks:
        rows = step_records(tasks, critical_k)
        critical_rows = [row for row in rows if row["critical"]]
        non_rows = [row for row in rows if not row["critical"]]
        cssr = cssr_ncsr(tasks, critical_k)
        cssr["gap_boot_ci95"] = bootstrap_gap(tasks, critical_k, n_boot, seed + critical_k)
        out[critical_k] = {
            "count_critical": len(critical_rows),
            "count_noncritical": len(non_rows),
            "p_critical": summarize_values([row["p"] for row in critical_rows]),
            "p_noncritical": summarize_values([row["p"] for row in non_rows]),
            "mean_norm_pos_critical": mean([row["norm_pos"] for row in critical_rows]),
            "mean_norm_pos_noncritical": mean([row["norm_pos"] for row in non_rows]),
            "critical_action_distribution": distribution(rows, "action_type", critical=True),
            "critical_phase_distribution": distribution(rows, "step_phase", critical=True),
            "critical_bbox_distribution": distribution(rows, "bbox_area_bin", critical=True),
            "cssr_ncsr": cssr,
            "action_bucket_success": bucket_success(rows, "action_type", min_bucket_n),
            "phase_bucket_success": bucket_success(rows, "step_phase", min_bucket_n),
            "oracle": oracle_metrics(tasks, critical_k),
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Critical-step-aware GUI-360 evaluation")
    parser.add_argument("--per-task", default="outputs/compound_proof/per_task.jsonl")
    parser.add_argument("--compound-fit", default="outputs/compound_proof/compound_fit.json")
    parser.add_argument("--output-dir", default="outputs/critstep_eval")
    parser.add_argument("--critical-ks", default="1,2")
    parser.add_argument("--bootstrap", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260630)
    parser.add_argument("--min-bucket-n", type=int, default=30)
    args = parser.parse_args()

    tasks = load_jsonl(Path(args.per_task))
    fit = json.loads(Path(args.compound_fit).read_text(encoding="utf-8"))
    critical_ks = [int(value) for value in args.critical_ks.split(",") if value.strip()]
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = summarize(tasks, fit, critical_ks, args.bootstrap, args.seed, args.min_bucket_n)
    gate_data = gate(summary)
    per_task_rows(tasks, critical_ks, out_dir / "per_task.jsonl")
    payload = {"inputs": {"per_task": args.per_task, "compound_fit": args.compound_fit, "critical_ks": critical_ks}, "summary": summary, "gate": gate_data}
    (out_dir / "critstep_eval.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_report(out_dir / "critstep_eval.md", tasks, fit, summary, gate_data)
    print(json.dumps({"output_dir": str(out_dir), "verdict": gate_data["verdict"], "bottom2_cssr": summary[2]["cssr_ncsr"]["cssr"], "bottom2_ncsr": summary[2]["cssr_ncsr"]["ncsr"], "bottom2_oracle_lift": summary[2]["oracle"]["oracle_critical_predicted_lift"]}, indent=2))


if __name__ == "__main__":
    main()