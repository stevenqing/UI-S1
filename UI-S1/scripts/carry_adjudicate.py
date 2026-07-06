#!/usr/bin/env python3
"""Adjudicate carry vs cumulative distribution shift.

Uses existing paired carry-test rows and reconstructs GT/pred history text
prefixes. Tests whether prefix-error count survives within position x length
cells and beyond text-divergence controls.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from v13_gui_360.eval_gui360_template import _format_action_for_history


DEFAULT_CARRY_PER_STEP = "outputs/carry_test/per_step.jsonl"
DEFAULT_TEST_DATA = "outputs/gui360_history_ab/original_eval/gui360_test_1000_balanced_uia.jsonl"
DEFAULT_PRED_RESULTS = "outputs/gui360_history_ab/original_sft_template_pred_history_full_20260701/eval_results_20260701_085620.json"
DEFAULT_OUTPUT_DIR = "outputs/carry_adjudicate"


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


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def task_length_bin(k: int) -> str:
    if k <= 1:
        return "len_1"
    if k <= 3:
        return "len_2_3"
    if k <= 5:
        return "len_4_5"
    if k <= 10:
        return "len_6_10"
    return "len_11_plus"


def text_divergence(a: str, b: str) -> float:
    if not a and not b:
        return 0.0
    return 1.0 - SequenceMatcher(None, a, b).ratio()


def token_jaccard_distance(a: str, b: str) -> float:
    aa = set(a.split())
    bb = set(b.split())
    if not aa and not bb:
        return 0.0
    return 1.0 - len(aa & bb) / len(aa | bb)


def action_type(action: Optional[Mapping[str, Any]]) -> str:
    if not isinstance(action, Mapping):
        return "none"
    return str(action.get("action") or "none")


def build_prefix_features(carry_rows: List[Dict[str, Any]], data_path: Path, pred_results_path: Path) -> None:
    episodes = {str(row["episode_id"]): row for row in read_jsonl(data_path)}
    pred_results = {str(value.get("episode_id", key)): value for key, value in load_json(pred_results_path).items()}
    by_ep: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in carry_rows:
        by_ep[str(row["episode_id"])].append(row)
    for episode_id, rows in by_ep.items():
        rows.sort(key=lambda item: int(item["step_idx"]))
        episode = episodes.get(episode_id, {})
        pred_episode = pred_results.get(episode_id, {})
        gt_steps = episode.get("steps") if isinstance(episode.get("steps"), list) else []
        pred_steps = pred_episode.get("steps") if isinstance(pred_episode.get("steps"), list) else []
        gt_history: List[str] = []
        pred_history: List[str] = []
        diff_count = 0
        type_diff_count = 0
        for row in rows:
            idx = int(row["step_idx"])
            gt_text = "\n".join(gt_history)
            pred_text = "\n".join(pred_history)
            row["task_length_bin"] = task_length_bin(int(row.get("task_k") or len(rows)))
            row["cell_id"] = f"{row.get('position_bin')}|{row['task_length_bin']}"
            row["prefix_text_divergence"] = text_divergence(gt_text, pred_text)
            row["prefix_token_jaccard_distance"] = token_jaccard_distance(gt_text, pred_text)
            row["prefix_action_text_diff_count"] = diff_count
            row["prefix_action_type_diff_count"] = type_diff_count
            row["prefix_text_len_delta_abs"] = abs(len(pred_text) - len(gt_text))
            row["drop_value"] = float(row["gt_correct"]) - float(row["pred_correct"])
            if idx < len(gt_steps):
                gt_action = gt_steps[idx].get("action") if isinstance(gt_steps[idx], Mapping) else None
            else:
                gt_action = None
            if idx < len(pred_steps):
                pred_action = pred_steps[idx].get("pred_action") if isinstance(pred_steps[idx], Mapping) else None
            else:
                pred_action = None
            gt_line = _format_action_for_history(gt_action, idx + 1)
            pred_line = _format_action_for_history(pred_action, idx + 1)
            gt_history.append(gt_line)
            pred_history.append(pred_line)
            diff_count += int(gt_line != pred_line)
            type_diff_count += int(action_type(gt_action) != action_type(pred_action))


def one_hot(values: Sequence[str]) -> Tuple[np.ndarray, List[str]]:
    keys = sorted(set(values))
    if len(keys) <= 1:
        return np.zeros((len(values), 0)), []
    used = keys[1:]
    index = {key: i for i, key in enumerate(used)}
    mat = np.zeros((len(values), len(used)))
    for row_idx, value in enumerate(values):
        if value in index:
            mat[row_idx, index[value]] = 1.0
    return mat, used


def ols(y: np.ndarray, x: np.ndarray, names: Sequence[str]) -> Dict[str, Any]:
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    design = np.column_stack([np.ones(len(y)), x])
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    resid = y - design @ coef
    dof = max(1, len(y) - design.shape[1])
    sigma2 = float(resid @ resid) / dof
    xtx_inv = np.linalg.pinv(design.T @ design)
    se = np.sqrt(np.diag(xtx_inv) * sigma2)
    out = {}
    all_names = ["intercept", *names]
    for idx, name in enumerate(all_names):
        out[name] = {"coef": float(coef[idx]), "se": float(se[idx]), "t": float(coef[idx] / se[idx]) if se[idx] > 0 else None}
    return {"n": int(len(y)), "terms": out, "r2": float(1.0 - (resid @ resid) / max(1e-12, ((y - np.mean(y)) @ (y - np.mean(y)))))}


def fixed_effect_slope(rows: Sequence[Mapping[str, Any]], x_field: str, y_field: str = "drop_value", cell_field: str = "cell_id") -> Dict[str, Any]:
    y_vals = []
    x_vals = []
    used_rows = []
    groups: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row[cell_field])].append(row)
    per_cell = []
    for cell, group in sorted(groups.items()):
        xs = np.asarray([float(row[x_field]) for row in group], dtype=float)
        ys = np.asarray([float(row[y_field]) for row in group], dtype=float)
        if len(group) < 20 or len(set(xs.tolist())) < 2:
            continue
        x_center = xs - float(np.mean(xs))
        y_center = ys - float(np.mean(ys))
        y_vals.extend(y_center.tolist())
        x_vals.extend(x_center.tolist())
        used_rows.extend(group)
        denom = float(x_center @ x_center)
        slope = float(x_center @ y_center / denom) if denom > 1e-12 else 0.0
        per_cell.append({"cell": cell, "n": len(group), "slope": slope, "x_mean": float(np.mean(xs)), "drop_mean": float(np.mean(ys))})
    if not x_vals:
        return {"n": 0, "slope": None, "se": None, "t": None, "per_cell": per_cell}
    x = np.asarray(x_vals, dtype=float)
    y = np.asarray(y_vals, dtype=float)
    fit = ols(y, x, [x_field])
    term = fit["terms"][x_field]
    return {"n": int(len(y)), "slope": term["coef"], "se": term["se"], "t": term["t"], "per_cell": per_cell[:80]}


def regression_with_controls(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    y = np.asarray([float(row["drop_value"]) for row in rows], dtype=float)
    prefix = np.asarray([float(row["prefix_error_count"]) for row in rows], dtype=float).reshape(-1, 1)
    divergence = np.asarray([float(row["prefix_text_divergence"]) for row in rows], dtype=float).reshape(-1, 1)
    diff_count = np.asarray([float(row["prefix_action_text_diff_count"]) for row in rows], dtype=float).reshape(-1, 1)
    rel_idx = np.asarray([float(row["step_idx"]) / max(1.0, float(row["task_k"]) - 1.0) for row in rows], dtype=float).reshape(-1, 1)
    task_k = np.asarray([float(row["task_k"]) for row in rows], dtype=float).reshape(-1, 1)
    pos, pos_names = one_hot([str(row["position_bin"]) for row in rows])
    length, len_names = one_hot([str(row["task_length_bin"]) for row in rows])
    diff, diff_names = one_hot([str(row["difficulty_bin"]) for row in rows])
    x = np.concatenate([prefix, divergence, diff_count, rel_idx, task_k, pos, length, diff], axis=1)
    names = ["prefix_error_count", "prefix_text_divergence", "prefix_action_text_diff_count", "relative_position", "task_k"] + [f"pos:{n}" for n in pos_names] + [f"len:{n}" for n in len_names] + [f"diff:{n}" for n in diff_names]
    return ols(y, x, names)


def quantile_label(values: Sequence[float], value: float, q: int = 3) -> str:
    arr = np.asarray(values, dtype=float)
    edges = np.quantile(arr, [i / q for i in range(1, q)]) if len(arr) else []
    label = 0
    for edge in edges:
        if value > edge:
            label += 1
    return f"q{label}"


def group_metrics(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    n = len(rows)
    if n == 0:
        return {"n": 0, "drop": None, "gt_sr": None, "pred_sr": None, "prefix_errors_mean": None, "divergence_mean": None}
    gt = sum(1 for row in rows if row["gt_correct"])
    pred = sum(1 for row in rows if row["pred_correct"])
    return {
        "n": n,
        "gt_sr": gt / n,
        "pred_sr": pred / n,
        "drop": (gt - pred) / n,
        "prefix_errors_mean": float(np.mean([row["prefix_error_count"] for row in rows])),
        "divergence_mean": float(np.mean([row["prefix_text_divergence"] for row in rows])),
        "action_text_diff_mean": float(np.mean([row["prefix_action_text_diff_count"] for row in rows])),
    }


def contrast_groups(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    divergence_values = [row["prefix_text_divergence"] for row in rows]
    for row in rows:
        row["divergence_q"] = quantile_label(divergence_values, row["prefix_text_divergence"], q=3)
        if row["prefix_error_count"] == 0:
            row["error_q"] = "err0"
        elif row["prefix_error_count"] <= 2:
            row["error_q"] = "err1_2"
        else:
            row["error_q"] = "err3plus"
    groups = {
        "high_divergence_low_error": [row for row in rows if row["divergence_q"] == "q2" and row["prefix_error_count"] <= 1],
        "low_divergence_high_error": [row for row in rows if row["divergence_q"] == "q0" and row["prefix_error_count"] >= 3],
        "high_divergence_zero_error": [row for row in rows if row["divergence_q"] == "q2" and row["prefix_error_count"] == 0],
        "low_divergence_3plus_error": [row for row in rows if row["divergence_q"] == "q0" and row["prefix_error_count"] >= 3],
    }
    return {name: group_metrics(group) for name, group in groups.items()}


def aggregate_by(rows: Sequence[Mapping[str, Any]], field: str) -> Dict[str, Any]:
    grouped: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row[field])].append(row)
    return {key: group_metrics(grouped[key]) for key in sorted(grouped)}


def pct(value: Optional[float]) -> str:
    return "NA" if value is None else f"{100.0 * value:.2f}%"


def fmt(value: Optional[float]) -> str:
    return "NA" if value is None else f"{value:.4f}"


def decide_gate(summary: Mapping[str, Any]) -> Dict[str, str]:
    within = summary["test_a_within_cell_prefix_error"].get("slope") or 0.0
    reg_terms = summary["test_b_regression"]["terms"]
    error_coef = reg_terms["prefix_error_count"]["coef"]
    error_t = abs(reg_terms["prefix_error_count"].get("t") or 0.0)
    div_coef = reg_terms["prefix_text_divergence"]["coef"]
    diff_count_coef = reg_terms["prefix_action_text_diff_count"]["coef"]
    if abs(within) < 0.01 and abs(error_coef) < 0.01:
        return {"verdict": "DISTRIBUTION SHIFT", "reason": "Within-cell prefix-error slope collapses and prefix-error adds little after text/position controls."}
    if within > 0.02 and error_coef > 0.015 and error_t > 3:
        return {"verdict": "CARRY", "reason": "Prefix-error effect survives within position-length cells and remains positive beyond text-divergence controls."}
    return {"verdict": "MIXED", "reason": "Prefix-error survives some controls but text divergence/position share the effect; report partial text-mediated carry rather than pure state carry."}


def render_report(summary: Mapping[str, Any], output_dir: Path) -> str:
    lines: List[str] = ["# Carry vs Cumulative Distribution Shift", ""]
    lines.append("Adjudicates whether the prior CARRY EXISTS result is true error-dependence or position/length + history-text distribution shift. Uses existing paired outcomes; no model inference.")
    lines.append("")
    lines.append("## Test A: Prefix-Error Within Position x Length Cells")
    lines.append("")
    a = summary["test_a_within_cell_prefix_error"]
    lines.append("| effect | value |")
    lines.append("|---|---:|")
    lines.append(f"| previous controlled slope | {summary['previous_controlled_slope']:.4f} |")
    lines.append(f"| within-cell slope | {fmt(a.get('slope'))} |")
    lines.append(f"| standard error | {fmt(a.get('se'))} |")
    lines.append(f"| t-stat | {fmt(a.get('t'))} |")
    lines.append(f"| rows used | {a.get('n')} |")
    lines.append("")
    lines.append("Top per-cell slopes:")
    lines.append("")
    lines.append("| cell | n | slope | mean errors | mean drop |")
    lines.append("|---|---:|---:|---:|---:|")
    for item in sorted(a.get("per_cell", []), key=lambda row: abs(row.get("slope") or 0.0), reverse=True)[:20]:
        lines.append(f"| `{item['cell']}` | {item['n']} | {fmt(item['slope'])} | {fmt(item['x_mean'])} | {pct(item['drop_mean'])} |")
    lines.append("")
    lines.append("## Test B: Text-Divergence vs Error Count")
    lines.append("")
    b = summary["test_b_regression"]
    lines.append("| term | coef | se | t |")
    lines.append("|---|---:|---:|---:|")
    for term in ["prefix_error_count", "prefix_text_divergence", "prefix_action_text_diff_count", "relative_position", "task_k"]:
        item = b["terms"].get(term, {})
        lines.append(f"| `{term}` | {fmt(item.get('coef'))} | {fmt(item.get('se'))} | {fmt(item.get('t'))} |")
    lines.append(f"Regression R2: `{fmt(b.get('r2'))}`")
    lines.append("")
    lines.append("Clean contrast:")
    lines.append("")
    lines.append("| contrast group | n | drop | mean errors | mean divergence |")
    lines.append("|---|---:|---:|---:|---:|")
    for name, item in summary["test_b_contrasts"].items():
        lines.append(f"| `{name}` | {item['n']} | {pct(item.get('drop'))} | {fmt(item.get('prefix_errors_mean'))} | {fmt(item.get('divergence_mean'))} |")
    lines.append("")
    lines.append("## Test C: Teacher-Forced Text Channel")
    lines.append("")
    lines.append("Because screens are GT-forced, prior errors can affect later prediction only through the textual history prefix. Text features therefore mediate the observed carry signal; no environment-state corruption is present in this test.")
    lines.append("")
    lines.append("| text divergence bin | n | drop | mean errors | mean divergence |")
    lines.append("|---|---:|---:|---:|---:|")
    for key, item in summary["by_text_divergence_bin"].items():
        lines.append(f"| {key} | {item['n']} | {pct(item.get('drop'))} | {fmt(item.get('prefix_errors_mean'))} | {fmt(item.get('divergence_mean'))} |")
    lines.append("")
    lines.append("## Gate")
    lines.append("")
    gate = summary["gate"]
    lines.append(f"**{gate['verdict']}**")
    lines.append("")
    lines.append(gate["reason"])
    lines.append("")
    lines.append("## Consequence")
    lines.append("")
    if gate["verdict"] == "CARRY":
        lines.append("Prefix errors retain an independent text-mediated effect. The no-coupling claim should be revised: under pred-history rollout, erroneous history text causally hurts later steps, even though the screen remains GT-forced.")
    elif gate["verdict"] == "DISTRIBUTION SHIFT":
        lines.append("The earlier carry signal is explained by position/length and text drift; state-level no-coupling remains intact.")
    else:
        lines.append("The effect is mixed: teacher-forced screens rule out state carry, but prefix-error count still has residual text-mediated predictive power beyond coarse divergence measures.")
    lines.append("")
    lines.append("## Artifacts")
    lines.append("")
    lines.append(f"- `{output_dir / 'adjudicate.md'}`")
    lines.append(f"- `{output_dir / 'summary.json'}`")
    lines.append(f"- `{output_dir / 'per_step.jsonl'}`")
    lines.append("")
    lines.append("STOP for review.")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--carry-per-step", default=DEFAULT_CARRY_PER_STEP)
    parser.add_argument("--test-data", default=DEFAULT_TEST_DATA)
    parser.add_argument("--pred-results", default=DEFAULT_PRED_RESULTS)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = read_jsonl(Path(args.carry_per_step))
    build_prefix_features(rows, Path(args.test_data), Path(args.pred_results))
    divergence_values = [row["prefix_text_divergence"] for row in rows]
    for row in rows:
        row["text_divergence_bin"] = quantile_label(divergence_values, row["prefix_text_divergence"], q=4)
    previous_slope = 0.040634193739564894
    summary: Dict[str, Any] = {
        "inputs": {"carry_per_step": args.carry_per_step, "test_data": args.test_data, "pred_results": args.pred_results},
        "n_steps": len(rows),
        "previous_controlled_slope": previous_slope,
        "test_a_within_cell_prefix_error": fixed_effect_slope(rows, "prefix_error_count"),
        "test_b_regression": regression_with_controls(rows),
        "test_b_contrasts": contrast_groups(rows),
        "by_text_divergence_bin": aggregate_by(rows, "text_divergence_bin"),
        "by_prefix_error_bin": aggregate_by(rows, "prefix_error_bin"),
        "by_cell": aggregate_by(rows, "cell_id"),
    }
    summary["gate"] = decide_gate(summary)
    write_jsonl(output_dir / "per_step.jsonl", rows)
    write_json(output_dir / "summary.json", summary)
    (output_dir / "adjudicate.md").write_text(render_report(summary, output_dir), encoding="utf-8")
    print(json.dumps({
        "output_dir": str(output_dir),
        "n_steps": len(rows),
        "within_cell_slope": summary["test_a_within_cell_prefix_error"].get("slope"),
        "error_coef": summary["test_b_regression"]["terms"]["prefix_error_count"]["coef"],
        "divergence_coef": summary["test_b_regression"]["terms"]["prefix_text_divergence"]["coef"],
        "gate": summary["gate"]["verdict"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()