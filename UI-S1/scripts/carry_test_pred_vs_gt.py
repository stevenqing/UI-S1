#!/usr/bin/env python3
"""Paired pred-history vs GT-history carry test for GUI-360.

This reuses stored full-step eval outputs. It tests the causal carry signature:
whether the GT-vs-pred step drop scales with the count of prior pred-history
errors, with all-correct-prefix steps as the distribution-shift control.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np


DEFAULT_GT_SUMMARY = "outputs/gui360_history_ab/original_sft_template_gt_history_merged_20260630/summary.json"
DEFAULT_PRED_RESULTS = "outputs/gui360_history_ab/original_sft_template_pred_history_full_20260701/eval_results_20260701_085620.json"
DEFAULT_CRIT_TASKS = "outputs/critstep_eval/per_task.jsonl"
DEFAULT_OUTPUT_DIR = "outputs/carry_test"


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_eval_results(paths: Sequence[str]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for raw_path in paths:
        data = json.loads(Path(raw_path).read_text(encoding="utf-8"))
        for key, value in data.items():
            episode_id = str(value.get("episode_id", key))
            if episode_id in out:
                raise ValueError(f"duplicate episode {episode_id} from {raw_path}")
            out[episode_id] = value
    return out


def gt_result_paths(summary_path: Path) -> List[str]:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    return [str(item["results"]) for item in summary.get("shards", []) if item.get("results")]


def load_tasks(path: Path) -> Dict[str, Dict[str, Any]]:
    return {str(row.get("episode_id")): row for row in read_jsonl(path)}


def bin_prefix_errors(count: int) -> str:
    if count <= 0:
        return "0"
    if count == 1:
        return "1"
    if count == 2:
        return "2"
    return "3+"


def position_bin(step_idx: int, task_k: int) -> str:
    if step_idx == 0:
        return "0"
    if step_idx <= 2:
        return "1-2"
    if step_idx <= 4:
        return "3-4"
    if step_idx <= 7:
        return "5-7"
    return "8+"


def difficulty_bin(p_i: Optional[float]) -> str:
    if p_i is None or not math.isfinite(float(p_i)):
        return "unknown"
    if p_i < 0.50:
        return "p_lt_0.50"
    if p_i < 0.60:
        return "p_0.50_0.60"
    if p_i < 0.70:
        return "p_0.60_0.70"
    return "p_ge_0.70"


def safe_mean(values: Sequence[float]) -> Optional[float]:
    return sum(values) / len(values) if values else None


def group_metrics(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    n = len(rows)
    if n == 0:
        return {"n": 0, "gt_step_sr": None, "pred_step_sr": None, "drop_gt_minus_pred": None, "drop_rate": None}
    gt = sum(1 for row in rows if row["gt_correct"])
    pred = sum(1 for row in rows if row["pred_correct"])
    drops = sum(1 for row in rows if row["gt_correct"] and not row["pred_correct"])
    improves = sum(1 for row in rows if row["pred_correct"] and not row["gt_correct"])
    return {
        "n": n,
        "gt_correct": gt,
        "pred_correct": pred,
        "gt_step_sr": gt / n,
        "pred_step_sr": pred / n,
        "drop_gt_minus_pred": (gt - pred) / n,
        "drop_steps_gt_only": drops,
        "pred_better_steps": improves,
        "drop_rate_gt_only": drops / n,
        "pred_better_rate": improves / n,
    }


def aggregate_by(rows: Sequence[Mapping[str, Any]], field: str) -> Dict[str, Dict[str, Any]]:
    grouped: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get(field))].append(row)
    order = sorted(grouped)
    if field == "prefix_error_bin":
        order = [key for key in ["0", "1", "2", "3+"] if key in grouped]
    if field == "position_bin":
        order = [key for key in ["0", "1-2", "3-4", "5-7", "8+"] if key in grouped]
    return {key: group_metrics(grouped[key]) for key in order}


def residualize(values: np.ndarray, controls: np.ndarray) -> np.ndarray:
    if controls.size == 0:
        return values - np.mean(values)
    design = np.column_stack([np.ones(len(values)), controls])
    coef, *_ = np.linalg.lstsq(design, values, rcond=None)
    return values - design @ coef


def one_hot(values: Sequence[str]) -> Tuple[np.ndarray, List[str]]:
    keys = sorted(set(values))
    if len(keys) <= 1:
        return np.zeros((len(values), 0)), []
    used = keys[1:]
    matrix = np.zeros((len(values), len(used)))
    index = {key: idx for idx, key in enumerate(used)}
    for row_idx, value in enumerate(values):
        if value in index:
            matrix[row_idx, index[value]] = 1.0
    return matrix, used


def controlled_prefix_effect(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    if not rows:
        return {}
    y = np.asarray([float(row["gt_correct"]) - float(row["pred_correct"]) for row in rows], dtype=float)
    prefix = np.asarray([float(row["prefix_error_count"]) for row in rows], dtype=float)
    pos_matrix, pos_keys = one_hot([str(row["position_bin"]) for row in rows])
    diff_matrix, diff_keys = one_hot([str(row["difficulty_bin"]) for row in rows])
    task_k = np.asarray([float(row.get("task_k") or 0.0) for row in rows], dtype=float).reshape(-1, 1)
    step_idx = np.asarray([float(row.get("step_idx") or 0.0) for row in rows], dtype=float).reshape(-1, 1)
    controls = np.concatenate([pos_matrix, diff_matrix, task_k, step_idx], axis=1)
    y_res = residualize(y, controls)
    prefix_res = residualize(prefix, controls)
    denom = float(prefix_res @ prefix_res)
    slope = float(prefix_res @ y_res / denom) if denom > 1e-12 else 0.0
    pred = slope * prefix_res
    residual = y_res - pred
    dof = max(1, len(rows) - controls.shape[1] - 2)
    se = math.sqrt(float(residual @ residual) / dof / denom) if denom > 1e-12 else None
    t_stat = slope / se if se and se > 0 else None
    return {
        "n": len(rows),
        "outcome": "gt_correct_minus_pred_correct",
        "prefix_error_count_slope_after_position_difficulty": slope,
        "standard_error": se,
        "t_stat": t_stat,
        "controls": {"position_bins": pos_keys, "difficulty_bins": diff_keys, "continuous": ["task_k", "step_idx"]},
    }


def render_table(lines: List[str], title: str, table: Mapping[str, Mapping[str, Any]]) -> None:
    lines.append(f"## {title}")
    lines.append("")
    lines.append("| group | n | GT StepSR | Pred StepSR | GT-Pred drop | GT-only drops | Pred-better |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for key, item in table.items():
        lines.append(
            f"| {key} | {item['n']} | {pct(item.get('gt_step_sr'))} | {pct(item.get('pred_step_sr'))} | "
            f"{pct(item.get('drop_gt_minus_pred'))} | {item.get('drop_steps_gt_only', 0)} | {item.get('pred_better_steps', 0)} |"
        )
    lines.append("")


def pct(value: Optional[float]) -> str:
    if value is None:
        return "NA"
    return f"{100.0 * value:.2f}%"


def decide_gate(summary: Mapping[str, Any]) -> Dict[str, str]:
    prefix = summary["by_prefix_error_bin"]
    drop0 = prefix.get("0", {}).get("drop_gt_minus_pred")
    drop3 = prefix.get("3+", {}).get("drop_gt_minus_pred")
    slope = (summary.get("controlled_prefix_effect") or {}).get("prefix_error_count_slope_after_position_difficulty")
    if drop0 is not None and drop3 is not None and slope is not None:
        if drop3 > drop0 + 0.08 and slope > 0.02:
            return {"verdict": "CARRY EXISTS", "reason": "GT-vs-pred drop grows strongly with prefix-error count beyond position/difficulty controls."}
        if abs(drop3 - drop0) <= 0.04 or slope <= 0.01:
            if drop0 > 0.03:
                return {"verdict": "NO CARRY", "reason": "Drop persists even with all-correct pred prefixes and does not materially scale with prefix-error count; this supports distribution shift rather than causal carry."}
            return {"verdict": "NO CARRY", "reason": "Drop does not materially scale with prefix-error count after controls."}
    return {"verdict": "MIXED / WEAK", "reason": "Prefix-error scaling is partial or underpowered; report subgroup structure rather than a clean carry/no-carry claim."}


def render_report(summary: Mapping[str, Any], output_dir: Path) -> str:
    lines: List[str] = ["# Definitive Cross-Step Carry Test", ""]
    lines.append("Paired step-level GT-history vs pred-history analysis. Frozen matcher and existing eval outputs only; no model inference was run.")
    lines.append("")
    lines.append("## Metric 1: Paired Step-Level Success")
    lines.append("")
    overall = summary["overall"]
    lines.append("| condition | TSR | StepSR | correct steps | total steps |")
    lines.append("|---|---:|---:|---:|---:|")
    lines.append(f"| GT-history | {pct(summary['gt_tsr'])} | {pct(overall['gt_step_sr'])} | {overall['gt_correct']} | {overall['n']} |")
    lines.append(f"| Pred-history | {pct(summary['pred_tsr'])} | {pct(overall['pred_step_sr'])} | {overall['pred_correct']} | {overall['n']} |")
    lines.append(f"| GT - Pred | {pct(summary['gt_tsr'] - summary['pred_tsr'])} | {pct(overall['drop_gt_minus_pred'])} | {overall['gt_correct'] - overall['pred_correct']} | {overall['n']} |")
    lines.append("")
    all_correct = summary["by_prefix_error_bin"].get("0", {})
    lines.append(f"All-correct-prefix control: `{all_correct.get('n', 0)}` steps, drop `{pct(all_correct.get('drop_gt_minus_pred'))}`.")
    lines.append("")
    render_table(lines, "Metric 2: Drop vs Prefix-Error Count", summary["by_prefix_error_bin"])
    controlled = summary.get("controlled_prefix_effect") or {}
    lines.append("## Position/Difficulty-Controlled Carry Slope")
    lines.append("")
    lines.append("| effect | value |")
    lines.append("|---|---:|")
    lines.append(f"| slope of GT-Pred drop per prefix error | {controlled.get('prefix_error_count_slope_after_position_difficulty', 0.0):.4f} |")
    lines.append(f"| standard error | {controlled.get('standard_error') if controlled.get('standard_error') is not None else 'NA'} |")
    lines.append(f"| t-stat | {controlled.get('t_stat') if controlled.get('t_stat') is not None else 'NA'} |")
    lines.append("")
    render_table(lines, "Metric 3: Carry vs Distribution Shift Control", summary["by_all_correct_prefix"])
    render_table(lines, "Metric 4: Critical vs Non-Critical Drops", summary["by_critical"])
    render_table(lines, "Drop by Step Position", summary["by_position_bin"])
    render_table(lines, "Drop by Difficulty", summary["by_difficulty_bin"])
    lines.append("## Exp B Reconciliation")
    lines.append("")
    rec = summary["exp_b_reconciliation"]
    lines.append(f"GT-history improves `{rec['gt_only_drop_steps']}` step predictions over pred-history, but TSR changes by only `{pct(summary['gt_tsr'] - summary['pred_tsr'])}`.")
    lines.append(f"GT-only drops on bottom-2 critical steps: `{rec['gt_only_critical_steps']}` / `{rec['gt_only_drop_steps']}` ({pct(rec['gt_only_critical_fraction'])}).")
    lines.append(f"GT-only drops on non-critical steps: `{rec['gt_only_noncritical_steps']}` / `{rec['gt_only_drop_steps']}` ({pct(rec['gt_only_noncritical_fraction'])}).")
    lines.append("This explains why pred-history loses StepSR while TSR remains nearly flat: many drops are off the task-deciding bottom-p_i set.")
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
    lines.append(f"- `{output_dir / 'carry.md'}`")
    lines.append(f"- `{output_dir / 'summary.json'}`")
    lines.append(f"- `{output_dir / 'per_step.jsonl'}`")
    lines.append("")
    lines.append("STOP for review.")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt-summary", default=DEFAULT_GT_SUMMARY)
    parser.add_argument("--pred-results", default=DEFAULT_PRED_RESULTS)
    parser.add_argument("--crit-tasks", default=DEFAULT_CRIT_TASKS)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    gt_results = load_eval_results(gt_result_paths(Path(args.gt_summary)))
    pred_results = load_eval_results([args.pred_results])
    tasks = load_tasks(Path(args.crit_tasks))
    rows: List[Dict[str, Any]] = []
    for episode_id, gt_episode in sorted(gt_results.items(), key=lambda item: int(item[0]) if item[0].isdigit() else item[0]):
        pred_episode = pred_results.get(episode_id)
        task = tasks.get(episode_id, {})
        if not pred_episode:
            continue
        gt_steps = gt_episode.get("steps") or []
        pred_steps = pred_episode.get("steps") or []
        task_k = int(gt_episode.get("num_steps") or len(gt_steps))
        per_p = task.get("per_step_p_heldout_cv") if isinstance(task.get("per_step_p_heldout_cv"), list) else []
        bottom2 = {int(index) for index in task.get("bottom2_critical_indices", [])}
        prefix_errors = 0
        for step_idx in range(min(len(gt_steps), len(pred_steps), task_k)):
            gt_success = bool(gt_steps[step_idx].get("success"))
            pred_success = bool(pred_steps[step_idx].get("success"))
            p_i = float(per_p[step_idx]) if step_idx < len(per_p) else None
            rows.append({
                "episode_id": episode_id,
                "step_idx": step_idx,
                "task_k": task_k,
                "gt_correct": gt_success,
                "pred_correct": pred_success,
                "gt_only_drop": bool(gt_success and not pred_success),
                "pred_better": bool(pred_success and not gt_success),
                "prefix_error_count": prefix_errors,
                "prefix_error_bin": bin_prefix_errors(prefix_errors),
                "all_correct_prefix": prefix_errors == 0,
                "position_bin": position_bin(step_idx, task_k),
                "p_i_heldout_compare_only": p_i,
                "difficulty_bin": difficulty_bin(p_i),
                "critical_bottom2": step_idx in bottom2,
                "gt_pred_type": gt_steps[step_idx].get("pred_type"),
                "pred_pred_type": pred_steps[step_idx].get("pred_type"),
                "gt_reward": gt_steps[step_idx].get("reward"),
                "pred_reward": pred_steps[step_idx].get("reward"),
            })
            if not pred_success:
                prefix_errors += 1
    gt_tsr = sum(1 for ep in gt_results.values() if ep.get("task_success")) / len(gt_results)
    pred_tsr = sum(1 for ep in pred_results.values() if ep.get("task_success")) / len(pred_results)
    overall = group_metrics(rows)
    gt_only = [row for row in rows if row["gt_only_drop"]]
    crit_gt_only = [row for row in gt_only if row["critical_bottom2"]]
    summary: Dict[str, Any] = {
        "inputs": {"gt_summary": args.gt_summary, "pred_results": args.pred_results, "crit_tasks": args.crit_tasks},
        "n_episodes_gt": len(gt_results),
        "n_episodes_pred": len(pred_results),
        "gt_tsr": gt_tsr,
        "pred_tsr": pred_tsr,
        "overall": overall,
        "by_prefix_error_bin": aggregate_by(rows, "prefix_error_bin"),
        "by_all_correct_prefix": aggregate_by([{**row, "all_correct_prefix_label": "all_correct_prefix" if row["all_correct_prefix"] else "prefix_has_error"} for row in rows], "all_correct_prefix_label"),
        "by_critical": aggregate_by([{**row, "critical_label": "critical_bottom2" if row["critical_bottom2"] else "noncritical"} for row in rows], "critical_label"),
        "by_position_bin": aggregate_by(rows, "position_bin"),
        "by_difficulty_bin": aggregate_by(rows, "difficulty_bin"),
        "controlled_prefix_effect": controlled_prefix_effect(rows),
        "exp_b_reconciliation": {
            "gt_only_drop_steps": len(gt_only),
            "gt_only_critical_steps": len(crit_gt_only),
            "gt_only_noncritical_steps": len(gt_only) - len(crit_gt_only),
            "gt_only_critical_fraction": len(crit_gt_only) / len(gt_only) if gt_only else 0.0,
            "gt_only_noncritical_fraction": (len(gt_only) - len(crit_gt_only)) / len(gt_only) if gt_only else 0.0,
        },
    }
    summary["gate"] = decide_gate(summary)
    write_jsonl(output_dir / "per_step.jsonl", rows)
    write_json(output_dir / "summary.json", summary)
    (output_dir / "carry.md").write_text(render_report(summary, output_dir), encoding="utf-8")
    print(json.dumps({
        "output_dir": str(output_dir),
        "steps": len(rows),
        "gt_step_sr": overall["gt_step_sr"],
        "pred_step_sr": overall["pred_step_sr"],
        "drop": overall["drop_gt_minus_pred"],
        "gate": summary["gate"]["verdict"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()