#!/usr/bin/env python3
"""Separate positive carry (H2a) from task-level confounding (H2b).

Pure diagnostic over the baseline compound proof artifacts. No model calls.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple


EPS = 1e-12


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def variance(values: Sequence[float]) -> float:
    if len(values) <= 1:
        return 0.0
    avg = mean(values)
    return sum((value - avg) ** 2 for value in values) / (len(values) - 1)


def stddev(values: Sequence[float]) -> float:
    return math.sqrt(max(0.0, variance(values)))


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


def corr(x_values: Sequence[float], y_values: Sequence[float]) -> float:
    if len(x_values) != len(y_values) or len(x_values) < 2:
        return 0.0
    x_mean = mean(x_values)
    y_mean = mean(y_values)
    num = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_values, y_values))
    den_x = sum((x - x_mean) ** 2 for x in x_values)
    den_y = sum((y - y_mean) ** 2 for y in y_values)
    den = math.sqrt(den_x * den_y)
    return num / den if den > 0 else 0.0


def logit(value: float) -> float:
    value = max(1e-6, min(1 - 1e-6, value))
    return math.log(value / (1 - value))


def solve_linear(matrix: List[List[float]], vector: List[float]) -> List[float]:
    n = len(vector)
    aug = [row[:] + [vector[i]] for i, row in enumerate(matrix)]
    for i in range(n):
        aug[i][i] += 1e-8
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(aug[r][col]))
        aug[col], aug[pivot] = aug[pivot], aug[col]
        pivot_value = aug[col][col]
        if abs(pivot_value) < 1e-12:
            continue
        for j in range(col, n + 1):
            aug[col][j] /= pivot_value
        for r in range(n):
            if r == col:
                continue
            factor = aug[r][col]
            for j in range(col, n + 1):
                aug[r][j] -= factor * aug[col][j]
    return [aug[i][n] for i in range(n)]


def ols_residuals(y_values: Sequence[float], x_columns: Sequence[Sequence[float]]) -> List[float]:
    n = len(y_values)
    if n == 0:
        return []
    columns = [[1.0] * n] + [list(col) for col in x_columns]
    p = len(columns)
    xtx = [[sum(columns[a][i] * columns[b][i] for i in range(n)) for b in range(p)] for a in range(p)]
    xty = [sum(columns[a][i] * y_values[i] for i in range(n)) for a in range(p)]
    beta = solve_linear(xtx, xty)
    return [y_values[i] - sum(beta[j] * columns[j][i] for j in range(p)) for i in range(n)]


def fisher_ci(r: float, n: int) -> Tuple[float, float]:
    if n <= 3:
        return (0.0, 0.0)
    r = max(-0.999999, min(0.999999, r))
    z = 0.5 * math.log((1 + r) / (1 - r))
    se = 1.0 / math.sqrt(n - 3)
    return (math.tanh(z - 1.96 * se), math.tanh(z + 1.96 * se))


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def read_goals(path: Path) -> Dict[str, str]:
    goals = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            goals[str(item["episode_id"])] = str(item.get("goal") or "")
    return goals


def infer_app(goal: str) -> str:
    text = goal.lower()
    rules = [
        ("excel", ["excel", "worksheet", "spreadsheet", "cell", "row", "column", "formula", "chart", "pivot"]),
        ("word", ["word", "document", "paragraph", "font", "heading", "page", "proofread"]),
        ("powerpoint", ["powerpoint", "slide", "presentation"]),
        ("browser", ["browser", "web", "website", "url", "search", "page"]),
        ("file", ["file explorer", "folder", "file", "directory"]),
        ("settings", ["settings", "control panel", "toggle", "enable", "disable"]),
        ("outlook", ["outlook", "email", "mail", "calendar"]),
    ]
    for label, words in rules:
        if any(word in text for word in words):
            return label
    return "other"


def k_bin(k: int) -> str:
    return str(k) if k <= 7 else "8+"


def task_covariates(tasks: Sequence[Dict[str, Any]], goals: Dict[str, str]) -> Tuple[Dict[str, List[float]], List[str]]:
    apps = sorted({infer_app(goals.get(str(task["episode_id"]), "")) for task in tasks})
    k_bins = sorted({k_bin(int(task["k"])) for task in tasks}, key=lambda x: int(x.rstrip("+")) if x.rstrip("+").isdigit() else 99)
    covariates: Dict[str, List[float]] = defaultdict(list)
    for task in tasks:
        episode_id = str(task["episode_id"])
        goal = goals.get(episode_id, "")
        tokens = re.findall(r"\w+", goal.lower())
        ps = [float(v) for v in task["per_step_p_heldout_cv"]]
        action_counts = Counter(feature.get("action_type", "unknown") for feature in task.get("step_features", []))
        total_actions = max(1, sum(action_counts.values()))
        k = int(task["k"])
        covariates["k"].append(float(k))
        covariates["log_k"].append(math.log(max(1, k)))
        covariates["goal_tokens"].append(float(len(tokens)))
        covariates["goal_chars"].append(float(len(goal)))
        covariates["mean_p"].append(mean(ps))
        covariates["min_p"].append(min(ps) if ps else 0.0)
        covariates["std_p"].append(stddev(ps))
        covariates["bottom1_share"].append(float(task.get("bottom1_log_failure_share") or 0.0))
        for action in ("click", "type", "swipe", "drag"):
            covariates[f"frac_{action}"].append(action_counts[action] / total_actions)
        app = infer_app(goal)
        for label in apps[1:]:
            covariates[f"app_{label}"].append(1.0 if app == label else 0.0)
        kb = k_bin(k)
        for label in k_bins[1:]:
            covariates[f"kbin_{label}"].append(1.0 if kb == label else 0.0)
    return covariates, sorted(covariates)


def build_pair_rows(tasks: Sequence[Dict[str, Any]], covariates: Dict[str, List[float]], mode: str) -> List[Dict[str, Any]]:
    rows = []
    task_index = {str(task["episode_id"]): idx for idx, task in enumerate(tasks)}
    for task in tasks:
        episode_id = str(task["episode_id"])
        idx = task_index[episode_id]
        ps = [float(v) for v in task["per_step_p_heldout_cv"]]
        ys = [1.0 if value else 0.0 for value in task["per_step_success"]]
        k = int(task["k"])
        if mode == "adjacent":
            pairs = [(i, i + 1) for i in range(k - 1)]
        else:
            pairs = [(i, j) for i in range(k) for j in range(i + 1, k)]
        for i, j in pairs:
            row = {
                "episode_id": episode_id,
                "i": i,
                "j": j,
                "k": k,
                "distance": j - i,
                "p_i": ps[i],
                "p_j": ps[j],
                "correct_i": ys[i],
                "correct_j": ys[j],
            }
            for name, values in covariates.items():
                row[f"task_{name}"] = values[idx]
            rows.append(row)
    return rows


def partial_corr(rows: Sequence[Dict[str, Any]], cov_names: Sequence[str]) -> Dict[str, Any]:
    y_i = [float(row["correct_i"]) for row in rows]
    y_j = [float(row["correct_j"]) for row in rows]
    p_i = [float(row["p_i"]) for row in rows]
    p_j = [float(row["p_j"]) for row in rows]
    columns = [p_i, p_j, [logit(v) for v in p_i], [logit(v) for v in p_j]]
    for name in cov_names:
        columns.append([float(row[f"task_{name}"]) for row in rows])
    ri = ols_residuals(y_i, columns)
    rj = ols_residuals(y_j, columns)
    r = corr(ri, rj)
    return {"n_pairs": len(rows), "partial_corr": r, "ci95": fisher_ci(r, len(rows))}


def bootstrap_partial(tasks: Sequence[Dict[str, Any]], goals: Dict[str, str], mode: str, n_boot: int, seed: int) -> Tuple[float, float]:
    rng = random.Random(seed)
    values = []
    task_list = list(tasks)
    for _ in range(n_boot):
        sample = [rng.choice(task_list) for _ in task_list]
        covs, names = task_covariates(sample, goals)
        rows = build_pair_rows(sample, covs, mode)
        values.append(partial_corr(rows, names)["partial_corr"])
    return (quantile(values, 0.025), quantile(values, 0.975))


def prefix_rows(tasks: Sequence[Dict[str, Any]], out_path: Path) -> List[Dict[str, Any]]:
    rows = []
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for task in tasks:
            episode_id = str(task["episode_id"])
            ps = [float(v) for v in task["per_step_p_heldout_cv"]]
            ys = [1.0 if value else 0.0 for value in task["per_step_success"]]
            k = int(task["k"])
            prefix_ok = True
            suffix_all_correct = [False] * k
            suffix_ok = True
            for idx in range(k - 1, -1, -1):
                suffix_all_correct[idx] = suffix_ok
                suffix_ok = suffix_ok and bool(ys[idx])
            for t in range(k):
                row = {
                    "episode_id": episode_id,
                    "k": k,
                    "t": t,
                    "prefix_len": t,
                    "all_correct_prefix": prefix_ok,
                    "later_all_correct": suffix_all_correct[t],
                    "correct_t": ys[t],
                    "p_t": ps[t],
                    "position": t,
                    "position_frac": (t + 1) / k,
                }
                rows.append(row)
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                prefix_ok = prefix_ok and bool(ys[t])
    return rows


def lift(successes: Sequence[float], ps: Sequence[float]) -> float:
    return mean(successes) - mean(ps) if successes else 0.0


def test_e(rows: Sequence[Dict[str, Any]], min_n: int) -> Dict[str, Any]:
    forward_values = [row for row in rows if row["prefix_len"] > 0 and row["all_correct_prefix"]]
    backward_values = [row for row in rows if row["prefix_len"] < row["k"] - 1 and row["later_all_correct"]]
    all_nonfirst = [row for row in rows if row["prefix_len"] > 0]
    forward_lift = lift([row["correct_t"] for row in forward_values], [row["p_t"] for row in forward_values])
    backward_lift = lift([row["correct_t"] for row in backward_values], [row["p_t"] for row in backward_values])
    position_lift = lift([row["correct_t"] for row in all_nonfirst], [row["p_t"] for row in all_nonfirst])
    prefix_extra = forward_lift - position_lift
    by_prefix = []
    for prefix_len in sorted({int(row["prefix_len"]) for row in rows if row["prefix_len"] > 0}):
        selected = [row for row in rows if int(row["prefix_len"]) == prefix_len and row["all_correct_prefix"]]
        baseline = [row for row in rows if int(row["prefix_len"]) == prefix_len]
        if not selected:
            continue
        selected_lift = lift([row["correct_t"] for row in selected], [row["p_t"] for row in selected])
        baseline_lift = lift([row["correct_t"] for row in baseline], [row["p_t"] for row in baseline])
        by_prefix.append(
            {
                "prefix_len": prefix_len,
                "n_prefix_ok": len(selected),
                "n_position": len(baseline),
                "prefix_success_rate": mean([row["correct_t"] for row in selected]),
                "prefix_mean_p": mean([row["p_t"] for row in selected]),
                "prefix_lift": selected_lift,
                "position_lift": baseline_lift,
                "extra_lift_over_position": selected_lift - baseline_lift,
                "powered": len(selected) >= min_n,
            }
        )
    powered = [row for row in by_prefix if row["powered"]]
    slope = ols_slope([row["prefix_len"] for row in powered], [row["extra_lift_over_position"] for row in powered])
    return {
        "forward": {"n": len(forward_values), "lift": forward_lift},
        "backward": {"n": len(backward_values), "lift": backward_lift},
        "position_baseline": {"n": len(all_nonfirst), "lift": position_lift},
        "forward_minus_backward_lift": forward_lift - backward_lift,
        "forward_extra_over_position": prefix_extra,
        "prefix_by_length": by_prefix,
        "powered_prefix_slope_extra_lift": slope,
        "min_n": min_n,
    }


def ols_slope(xs: Sequence[float], ys: Sequence[float]) -> float:
    if len(xs) < 2:
        return 0.0
    x_mean = mean(xs)
    y_mean = mean(ys)
    den = sum((x - x_mean) ** 2 for x in xs)
    return sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys)) / den if den > 0 else 0.0


def gate(test_d: Dict[str, Any], test_e_data: Dict[str, Any]) -> Dict[str, Any]:
    baseline_adjacent = 0.21853158159900019
    partial = test_d["adjacent"]["partial_corr"]
    drop = baseline_adjacent - partial
    drop_fraction = drop / baseline_adjacent if baseline_adjacent else 0.0
    forward_minus_backward = test_e_data["forward_minus_backward_lift"]
    extra = test_e_data["forward_extra_over_position"]
    slope = test_e_data["powered_prefix_slope_extra_lift"]
    h2b_signature = (drop_fraction >= 0.5) or (abs(forward_minus_backward) <= 0.02 and abs(slope) <= 0.01)
    h2a_signature = (drop_fraction < 0.25 and forward_minus_backward > 0.03 and extra > 0.03 and slope > 0.005)
    if h2a_signature:
        verdict = "H2a_CONFIRMED"
        consequence = "No negative coupling still holds, but there is a positive forward carry; the paper claim should narrow accordingly."
    elif h2b_signature:
        verdict = "H2b_CONFIRMED"
        consequence = "Residual dependence is static task-level structure; no cross-step causal coupling claim is restored conditionally on task factors."
    else:
        verdict = "INCONCLUSIVE"
        consequence = "Residual positive task-level dependence remains, but static confounder vs causal carry is not cleanly separated."
    return {
        "verdict": verdict,
        "claim_consequence": consequence,
        "checks": {
            "baseline_adjacent_partial": baseline_adjacent,
            "task_covariate_adjacent_partial": partial,
            "partial_drop": drop,
            "partial_drop_fraction": drop_fraction,
            "forward_minus_backward_lift": forward_minus_backward,
            "forward_extra_over_position": extra,
            "prefix_extra_lift_slope": slope,
        },
    }


def pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(item) for item in row) + " |")
    return "\n".join(lines)


def write_report(path: Path, cov_names: Sequence[str], test_d: Dict[str, Any], test_e_data: Dict[str, Any], gate_data: Dict[str, Any]) -> None:
    lines = []
    lines.append("# H2a vs H2b Attribution")
    lines.append("")
    lines.append("Date: 2026-06-30")
    lines.append("")
    lines.append("## Scope")
    lines.append("")
    lines.append("Regime: original SFT baseline `checkpoints/gui360-fullparam-sft-step250`, original GUI-360 template, GT-history teacher-forced, frozen matcher. This is a zero-training diagnostic over existing baseline artifacts.")
    lines.append("")
    lines.append("Available task-level covariates: " + ", ".join(f"`{name}`" for name in cov_names) + ".")
    lines.append("")
    lines.append("## Test D - Task-Covariate Partial Correlation")
    lines.append("")
    d_rows = []
    for label in ("adjacent", "all_pairs"):
        stats = test_d[label]
        d_rows.append([label, stats["n_pairs"], f"{stats['partial_corr']:.4f}", f"[{stats['ci95'][0]:.4f}, {stats['ci95'][1]:.4f}]", f"[{stats['boot_ci95'][0]:.4f}, {stats['boot_ci95'][1]:.4f}]"])
    lines.append(table(["pair set", "n", "partial corr with task covariates", "analytic CI", "task-bootstrap CI"], d_rows))
    lines.append("")
    lines.append("Baseline adjacent partial corr without task covariates: `0.2185`. The table reports residualizing on p_i, p_j, logit p_i, logit p_j plus task covariates.")
    lines.append("")
    lines.append("## Test E - Directionality And Prefix-Cumulativity")
    lines.append("")
    lines.append(table(["quantity", "value"], [
        ["Forward lift: P(correct_t | prefix all correct) - p_t", pct(test_e_data["forward"]["lift"])],
        ["Forward n", test_e_data["forward"]["n"]],
        ["Backward lift: P(correct_t | later all correct) - p_t", pct(test_e_data["backward"]["lift"])],
        ["Backward n", test_e_data["backward"]["n"]],
        ["Position baseline lift", pct(test_e_data["position_baseline"]["lift"])],
        ["Forward minus backward", pct(test_e_data["forward_minus_backward_lift"])],
        ["Forward extra over position", pct(test_e_data["forward_extra_over_position"])],
        ["Powered prefix extra-lift slope", f"{test_e_data['powered_prefix_slope_extra_lift']:.4f}"],
    ]))
    lines.append("")
    lines.append("Prefix by length:")
    lines.append("")
    p_rows = []
    for row in test_e_data["prefix_by_length"]:
        p_rows.append([row["prefix_len"], row["n_prefix_ok"], row["n_position"], pct(row["prefix_lift"]), pct(row["position_lift"]), pct(row["extra_lift_over_position"]), row["powered"]])
    lines.append(table(["prefix len", "n prefix ok", "n position", "prefix lift", "position lift", "extra", "powered"], p_rows))
    lines.append("")
    lines.append("## Gate Verdict")
    lines.append("")
    lines.append(f"Verdict: **{gate_data['verdict']}**")
    lines.append("")
    lines.append(gate_data["claim_consequence"])
    lines.append("")
    lines.append(table(["check", "value"], [[k, v] for k, v in gate_data["checks"].items()]))
    lines.append("")
    lines.append("## Artifacts")
    lines.append("")
    lines.append("- `outputs/compound_proof/h2a_vs_h2b.md`")
    lines.append("- `outputs/compound_proof/h2a_vs_h2b.json`")
    lines.append("- `outputs/compound_proof/prefix_lift.jsonl`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Separate H2a causal carry from H2b task-level confounding")
    parser.add_argument("--per-task", default="outputs/compound_proof/per_task.jsonl")
    parser.add_argument("--test-data", default="outputs/gui360_history_ab/original_eval/gui360_test_1000_balanced_reconstructed.jsonl")
    parser.add_argument("--output-dir", default="outputs/compound_proof")
    parser.add_argument("--bootstrap", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260630)
    parser.add_argument("--min-prefix-n", type=int, default=30)
    args = parser.parse_args()

    tasks = load_jsonl(Path(args.per_task))
    goals = read_goals(Path(args.test_data))
    covs, cov_names = task_covariates(tasks, goals)
    adjacent_rows = build_pair_rows(tasks, covs, "adjacent")
    all_rows = build_pair_rows(tasks, covs, "all")
    test_d = {
        "adjacent": partial_corr(adjacent_rows, cov_names),
        "all_pairs": partial_corr(all_rows, cov_names),
    }
    test_d["adjacent"]["boot_ci95"] = bootstrap_partial(tasks, goals, "adjacent", args.bootstrap, args.seed)
    test_d["all_pairs"]["boot_ci95"] = test_d["all_pairs"]["ci95"]

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = prefix_rows(tasks, out_dir / "prefix_lift.jsonl")
    test_e_data = test_e(prefix, args.min_prefix_n)
    gate_data = gate(test_d, test_e_data)
    result = {"task_covariates": cov_names, "test_d": test_d, "test_e": test_e_data, "gate": gate_data}
    (out_dir / "h2a_vs_h2b.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_report(out_dir / "h2a_vs_h2b.md", cov_names, test_d, test_e_data, gate_data)
    print(json.dumps({"output_dir": str(out_dir), "verdict": gate_data["verdict"], "adjacent_partial_with_covariates": test_d["adjacent"]["partial_corr"], "forward_minus_backward_lift": test_e_data["forward_minus_backward_lift"], "prefix_extra_slope": test_e_data["powered_prefix_slope_extra_lift"]}, indent=2))


if __name__ == "__main__":
    main()