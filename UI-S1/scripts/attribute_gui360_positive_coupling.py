#!/usr/bin/env python3
"""Attribute positive real-vs-pseudo TSR deviation in GUI-360 compound proof.

This script is a pure diagnostic over outputs/compound_proof/per_task.jsonl and
outputs/compound_proof/pseudo_tasks.jsonl. It tests whether the positive residual
comes from task-level difficulty heterogeneity or from outcome coupling beyond
estimated per-step difficulty.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple


EPS = 1e-12


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def variance(values: Sequence[float]) -> float:
    if len(values) <= 1:
        return 0.0
    value_mean = mean(values)
    return sum((value - value_mean) ** 2 for value in values) / (len(values) - 1)


def stddev(values: Sequence[float]) -> float:
    return math.sqrt(max(0.0, variance(values)))


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


def corr(x_values: Sequence[float], y_values: Sequence[float]) -> float:
    if len(x_values) != len(y_values) or len(x_values) < 2:
        return 0.0
    mean_x = mean(x_values)
    mean_y = mean(y_values)
    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(x_values, y_values))
    den_x = sum((x - mean_x) ** 2 for x in x_values)
    den_y = sum((y - mean_y) ** 2 for y in y_values)
    den = math.sqrt(den_x * den_y)
    return num / den if den > 0 else 0.0


def ols_residuals(y_values: Sequence[float], x_columns: Sequence[Sequence[float]]) -> List[float]:
    """Return residuals from OLS with intercept using normal equations."""
    n = len(y_values)
    if n == 0:
        return []
    columns = [[1.0] * n] + [list(col) for col in x_columns]
    p = len(columns)
    xtx = [[sum(columns[a][i] * columns[b][i] for i in range(n)) for b in range(p)] for a in range(p)]
    xty = [sum(columns[a][i] * y_values[i] for i in range(n)) for a in range(p)]
    beta = solve_linear(xtx, xty)
    return [y_values[i] - sum(beta[j] * columns[j][i] for j in range(p)) for i in range(n)]


def solve_linear(matrix: List[List[float]], vector: List[float]) -> List[float]:
    n = len(vector)
    aug = [row[:] + [vector[i]] for i, row in enumerate(matrix)]
    ridge = 1e-8
    for i in range(n):
        aug[i][i] += ridge
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


def fisher_ci(r: float, n: int, alpha_z: float = 1.96) -> Tuple[float, float]:
    if n <= 3:
        return (0.0, 0.0)
    r = max(-0.999999, min(0.999999, r))
    z = 0.5 * math.log((1 + r) / (1 - r))
    se = 1.0 / math.sqrt(n - 3)
    lo = z - alpha_z * se
    hi = z + alpha_z * se
    return (math.tanh(lo), math.tanh(hi))


def product(values: Iterable[float]) -> float:
    out = 1.0
    for value in values:
        out *= max(EPS, min(1.0 - EPS, value))
    return out


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def p_spread(values: Sequence[float]) -> Dict[str, float]:
    return {
        "mean": mean(values),
        "std": stddev(values),
        "range": (max(values) - min(values)) if values else 0.0,
        "iqr": quantile(values, 0.75) - quantile(values, 0.25),
    }


def icc_by_task(tasks: Sequence[Dict[str, Any]]) -> Dict[str, float]:
    groups = [list(map(float, task["per_step_p_heldout_cv"])) for task in tasks if task.get("per_step_p_heldout_cv")]
    all_values = [value for group in groups for value in group]
    grand_mean = mean(all_values)
    between_ss = sum(len(group) * (mean(group) - grand_mean) ** 2 for group in groups)
    within_ss = sum(sum((value - mean(group)) ** 2 for value in group) for group in groups)
    total_ss = between_ss + within_ss
    return {
        "between_variance_share_eta2": between_ss / total_ss if total_ss > 0 else 0.0,
        "between_ss": between_ss,
        "within_ss": within_ss,
        "total_ss": total_ss,
        "total_steps": len(all_values),
        "tasks": len(groups),
    }


def test_a(tasks: Sequence[Dict[str, Any]], pseudo_rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    real_std = []
    real_range = []
    real_iqr = []
    real_means = []
    for task in tasks:
        values = list(map(float, task["per_step_p_heldout_cv"]))
        spread = p_spread(values)
        real_std.append(spread["std"])
        real_range.append(spread["range"])
        real_iqr.append(spread["iqr"])
        real_means.append(spread["mean"])

    pseudo_std = []
    pseudo_range = []
    pseudo_iqr = []
    for row in pseudo_rows:
        values = list(map(float, row["sampled_p_i"]))
        spread = p_spread(values)
        pseudo_std.append(spread["std"])
        pseudo_range.append(spread["range"])
        pseudo_iqr.append(spread["iqr"])

    icc = icc_by_task(tasks)
    low_spread_threshold = quantile(real_std, 0.25)
    return {
        "icc": icc,
        "real_within_std_mean": mean(real_std),
        "real_within_std_median": median(real_std),
        "pseudo_within_std_mean": mean(pseudo_std),
        "pseudo_within_std_median": median(pseudo_std),
        "real_within_range_mean": mean(real_range),
        "pseudo_within_range_mean": mean(pseudo_range),
        "real_within_iqr_mean": mean(real_iqr),
        "pseudo_within_iqr_mean": mean(pseudo_iqr),
        "homogeneous_threshold_std_p25": low_spread_threshold,
        "homogeneous_task_fraction": sum(value <= low_spread_threshold for value in real_std) / len(real_std) if real_std else 0.0,
        "task_mean_p_std": stddev(real_means),
    }


def build_pairs(tasks: Sequence[Dict[str, Any]], mode: str) -> List[Dict[str, Any]]:
    pairs = []
    for task in tasks:
        ps = list(map(float, task["per_step_p_heldout_cv"]))
        ys = [1.0 if value else 0.0 for value in task["per_step_success"]]
        episode_id = task["episode_id"]
        k = int(task["k"])
        if mode == "adjacent":
            indexes = [(i, i + 1) for i in range(k - 1)]
        elif mode == "all":
            indexes = [(i, j) for i in range(k) for j in range(i + 1, k)]
        else:
            raise ValueError(f"unknown pair mode: {mode}")
        for i, j in indexes:
            pairs.append(
                {
                    "episode_id": episode_id,
                    "k": k,
                    "i": i,
                    "j": j,
                    "distance": j - i,
                    "p_i": ps[i],
                    "p_j": ps[j],
                    "correct_i": ys[i],
                    "correct_j": ys[j],
                }
            )
    return pairs


def pair_correlation(pairs: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    y_i = [float(row["correct_i"]) for row in pairs]
    y_j = [float(row["correct_j"]) for row in pairs]
    p_i = [float(row["p_i"]) for row in pairs]
    p_j = [float(row["p_j"]) for row in pairs]
    logit_i = [logit(value) for value in p_i]
    logit_j = [logit(value) for value in p_j]
    raw = corr(y_i, y_j)
    residual_i = ols_residuals(y_i, [p_i, p_j, logit_i, logit_j])
    residual_j = ols_residuals(y_j, [p_i, p_j, logit_i, logit_j])
    partial = corr(residual_i, residual_j)
    raw_ci = fisher_ci(raw, len(pairs))
    partial_ci = fisher_ci(partial, len(pairs))
    return {
        "n_pairs": len(pairs),
        "raw_corr": raw,
        "raw_corr_ci95": raw_ci,
        "partial_corr_given_p_i_p_j": partial,
        "partial_corr_ci95": partial_ci,
    }


def logit(value: float) -> float:
    value = max(1e-6, min(1 - 1e-6, value))
    return math.log(value / (1 - value))


def bootstrap_partial(tasks: Sequence[Dict[str, Any]], *, mode: str, n_boot: int, seed: int) -> Dict[str, Any]:
    rng = random.Random(seed)
    raw_values = []
    partial_values = []
    task_list = list(tasks)
    for _ in range(n_boot):
        sample = [rng.choice(task_list) for _ in task_list]
        stats = pair_correlation(build_pairs(sample, mode))
        raw_values.append(stats["raw_corr"])
        partial_values.append(stats["partial_corr_given_p_i_p_j"])
    return {
        "raw_boot_ci95": (quantile(raw_values, 0.025), quantile(raw_values, 0.975)),
        "partial_boot_ci95": (quantile(partial_values, 0.025), quantile(partial_values, 0.975)),
    }


def test_b(tasks: Sequence[Dict[str, Any]], pair_out_path: Path, n_boot: int, seed: int) -> Dict[str, Any]:
    adjacent = build_pairs(tasks, "adjacent")
    all_pairs = build_pairs(tasks, "all")
    pair_out_path.parent.mkdir(parents=True, exist_ok=True)
    with pair_out_path.open("w", encoding="utf-8") as handle:
        for row in all_pairs:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    adjacent_stats = pair_correlation(adjacent)
    all_stats = pair_correlation(all_pairs)
    adjacent_stats.update(bootstrap_partial(tasks, mode="adjacent", n_boot=n_boot, seed=seed))
    all_stats.update(bootstrap_partial(tasks, mode="all", n_boot=n_boot, seed=seed + 1))
    return {"adjacent": adjacent_stats, "all_pairs": all_stats, "pair_outcomes": str(pair_out_path)}


def test_c(tasks: Sequence[Dict[str, Any]], pseudo_rows: Sequence[Dict[str, Any]], fit: Dict[str, Any]) -> Dict[str, Any]:
    actual = float(fit["layer1"]["actual_tsr"])
    product_pred = mean([float(task["predicted_prob_heldout_cv"]) for task in tasks])
    pseudo = mean([1.0 if row["success"] else 0.0 for row in pseudo_rows])
    difficulty_gap = product_pred - pseudo
    total_gap = actual - pseudo
    residual_gap = actual - product_pred
    return {
        "actual_real_tsr": actual,
        "independent_difficulty_preserving_tsr": product_pred,
        "pseudo_task_tsr": pseudo,
        "real_minus_pseudo_gap": total_gap,
        "difficulty_structure_gap_product_minus_pseudo": difficulty_gap,
        "residual_gap_actual_minus_product": residual_gap,
        "difficulty_explained_fraction_of_real_minus_pseudo": difficulty_gap / total_gap if abs(total_gap) > EPS else 0.0,
        "residual_fraction_of_real_minus_pseudo": residual_gap / total_gap if abs(total_gap) > EPS else 0.0,
    }


def gate(test_a_data: Dict[str, Any], test_b_data: Dict[str, Any], test_c_data: Dict[str, Any]) -> Dict[str, Any]:
    icc = test_a_data["icc"]["between_variance_share_eta2"]
    partial = test_b_data["adjacent"]["partial_corr_given_p_i_p_j"]
    partial_hi = test_b_data["adjacent"]["partial_boot_ci95"][1]
    residual_gap = test_c_data["residual_gap_actual_minus_product"]
    residual_fraction = test_c_data["residual_fraction_of_real_minus_pseudo"]
    difficulty_fraction = test_c_data["difficulty_explained_fraction_of_real_minus_pseudo"]
    tolerances = {
        "high_icc": 0.20,
        "partial_corr_abs_max": 0.03,
        "partial_corr_ci_hi_max": 0.05,
        "residual_gap_abs_max": 0.015,
        "difficulty_fraction_min": 0.70,
    }
    test_a_pass = icc >= tolerances["high_icc"]
    test_b_pass = abs(partial) <= tolerances["partial_corr_abs_max"] and partial_hi <= tolerances["partial_corr_ci_hi_max"]
    test_c_h1_pass = abs(residual_gap) <= tolerances["residual_gap_abs_max"] and difficulty_fraction >= tolerances["difficulty_fraction_min"]
    h2_detected = partial_hi > tolerances["partial_corr_ci_hi_max"] or residual_gap > tolerances["residual_gap_abs_max"]
    if test_a_pass and test_b_pass and test_c_h1_pass:
        verdict = "H1_CONFIRMED"
        explanation = "Difficulty is task-level, outcome dependence vanishes after p_i controls, and difficulty-preserving independence explains the real-vs-pseudo gap."
    elif h2_detected:
        verdict = "H2_DETECTED"
        explanation = "A positive residual remains after observed p_i controls and/or difficulty-preserving independence does not close the gap."
    else:
        verdict = "INCONCLUSIVE"
        explanation = "Tests do not cleanly agree; do not force H1."
    return {
        "verdict": verdict,
        "explanation": explanation,
        "tolerances": tolerances,
        "checks": {
            "test_a_icc": icc,
            "test_a_pass": test_a_pass,
            "test_b_adjacent_partial_corr": partial,
            "test_b_adjacent_partial_boot_ci95": test_b_data["adjacent"]["partial_boot_ci95"],
            "test_b_pass": test_b_pass,
            "test_c_residual_gap_actual_minus_product": residual_gap,
            "test_c_difficulty_explained_fraction": difficulty_fraction,
            "test_c_residual_fraction": residual_fraction,
            "test_c_h1_pass": test_c_h1_pass,
        },
    }


def pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def markdown_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---" for _ in headers]) + "|"]
    for row in rows:
        out.append("| " + " | ".join(str(item) for item in row) + " |")
    return "\n".join(out)


def write_report(path: Path, fit: Dict[str, Any], test_a_data: Dict[str, Any], test_b_data: Dict[str, Any], test_c_data: Dict[str, Any], gate_data: Dict[str, Any]) -> None:
    lines = []
    lines.append("# Positive Coupling Attribution")
    lines.append("")
    lines.append("Date: 2026-06-30")
    lines.append("")
    lines.append("## Scope")
    lines.append("")
    lines.append("Regime: original SFT baseline `checkpoints/gui360-fullparam-sft-step250`, original GUI-360 template, GT-history teacher-forced, full-history mode, frozen `compute_step_reward` matcher.")
    lines.append("")
    lines.append("Question: actual real-task TSR is higher than the held-out independent product and higher than matched pseudo-task TSR. This report attributes that positive deviation to task-level difficulty heterogeneity or residual positive outcome coupling.")
    lines.append("")
    lines.append("## Test A - Within-Task Difficulty Correlation")
    lines.append("")
    lines.append(markdown_table(["quantity", "value"], [
        ["ICC / between-task p_i variance share", f"{test_a_data['icc']['between_variance_share_eta2']:.4f}"],
        ["Mean within-task p_i std, real", f"{test_a_data['real_within_std_mean']:.4f}"],
        ["Mean within-task p_i std, pseudo", f"{test_a_data['pseudo_within_std_mean']:.4f}"],
        ["Mean within-task p_i range, real", f"{test_a_data['real_within_range_mean']:.4f}"],
        ["Mean within-task p_i range, pseudo", f"{test_a_data['pseudo_within_range_mean']:.4f}"],
        ["Mean within-task p_i IQR, real", f"{test_a_data['real_within_iqr_mean']:.4f}"],
        ["Mean within-task p_i IQR, pseudo", f"{test_a_data['pseudo_within_iqr_mean']:.4f}"],
        ["Homogeneous-task fraction", pct(test_a_data['homogeneous_task_fraction'])],
        ["Homogeneous threshold, std p25", f"{test_a_data['homogeneous_threshold_std_p25']:.4f}"],
    ]))
    lines.append("")
    lines.append("## Test B - Outcome Correlation Controlling For p_i")
    lines.append("")
    b_rows = []
    for label, stats in [("adjacent", test_b_data["adjacent"]), ("all pairs", test_b_data["all_pairs"] )]:
        b_rows.append([
            label,
            stats["n_pairs"],
            f"{stats['raw_corr']:.4f}",
            f"[{stats['raw_boot_ci95'][0]:.4f}, {stats['raw_boot_ci95'][1]:.4f}]",
            f"{stats['partial_corr_given_p_i_p_j']:.4f}",
            f"[{stats['partial_boot_ci95'][0]:.4f}, {stats['partial_boot_ci95'][1]:.4f}]",
        ])
    lines.append(markdown_table(["pair set", "n", "raw corr", "raw boot CI", "partial corr given p_i,p_j", "partial boot CI"], b_rows))
    lines.append("")
    lines.append("Partial correlation residualizes both outcomes on p_i, p_j, logit(p_i), and logit(p_j). A positive residual after this control is evidence for H2.")
    lines.append("")
    lines.append("## Test C - Difficulty-Preserving Independence")
    lines.append("")
    lines.append(markdown_table(["quantity", "value"], [
        ["Actual real-task TSR", pct(test_c_data["actual_real_tsr"])],
        ["Independent, real p_i structure TSR", pct(test_c_data["independent_difficulty_preserving_tsr"])],
        ["Matched pseudo-task TSR", pct(test_c_data["pseudo_task_tsr"])],
        ["Real minus pseudo gap", f"{test_c_data['real_minus_pseudo_gap']:+.4f}"],
        ["Difficulty structure gap, product minus pseudo", f"{test_c_data['difficulty_structure_gap_product_minus_pseudo']:+.4f}"],
        ["Residual gap, actual minus product", f"{test_c_data['residual_gap_actual_minus_product']:+.4f}"],
        ["Difficulty explained fraction", pct(test_c_data["difficulty_explained_fraction_of_real_minus_pseudo"])],
        ["Residual fraction", pct(test_c_data["residual_fraction_of_real_minus_pseudo"])],
    ]))
    lines.append("")
    lines.append("## Gate Verdict")
    lines.append("")
    lines.append(f"Verdict: **{gate_data['verdict']}**")
    lines.append("")
    lines.append(gate_data["explanation"])
    lines.append("")
    lines.append(markdown_table(["check", "value"], [[key, value] for key, value in gate_data["checks"].items()]))
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    if gate_data["verdict"] == "H1_CONFIRMED":
        lines.append("The positive real-vs-pseudo deviation is explained by task-difficulty heterogeneity. The no-causal-coupling claim stands.")
    elif gate_data["verdict"] == "H2_DETECTED":
        lines.append("The positive deviation is not explained by the observed p_i difficulty structure. There is residual positive dependence beyond the current p_i estimator: either a genuine positive cross-step carry or an unmodeled task-level factor not captured by the present difficulty buckets. This does not indicate negative coupling/error propagation, which remains absent in the compound proof, but it narrows the claim to no negative coupling plus a positive residual to explain.")
    else:
        lines.append("The attribution tests disagree or lack closure. Do not force H1.")
    lines.append("")
    lines.append("## Artifacts")
    lines.append("")
    lines.append("- `outputs/compound_proof/positive_coupling_attribution.md`")
    lines.append("- `outputs/compound_proof/positive_coupling_attribution.json`")
    lines.append("- `outputs/compound_proof/pair_outcomes.jsonl`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Attribute positive real-vs-pseudo TSR deviation")
    parser.add_argument("--per-task", default="outputs/compound_proof/per_task.jsonl")
    parser.add_argument("--pseudo-tasks", default="outputs/compound_proof/pseudo_tasks.jsonl")
    parser.add_argument("--compound-fit", default="outputs/compound_proof/compound_fit.json")
    parser.add_argument("--output-dir", default="outputs/compound_proof")
    parser.add_argument("--bootstrap", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260630)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tasks = load_jsonl(Path(args.per_task))
    pseudo_rows = load_jsonl(Path(args.pseudo_tasks))
    fit = json.loads(Path(args.compound_fit).read_text(encoding="utf-8"))

    a = test_a(tasks, pseudo_rows)
    b = test_b(tasks, out_dir / "pair_outcomes.jsonl", args.bootstrap, args.seed)
    c = test_c(tasks, pseudo_rows, fit)
    g = gate(a, b, c)
    result = {"test_a": a, "test_b": b, "test_c": c, "gate": g}
    (out_dir / "positive_coupling_attribution.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_report(out_dir / "positive_coupling_attribution.md", fit, a, b, c, g)
    print(json.dumps({"output_dir": str(out_dir), "verdict": g["verdict"], "test_b_adjacent_partial": b["adjacent"]["partial_corr_given_p_i_p_j"], "test_c_residual_gap": c["residual_gap_actual_minus_product"]}, indent=2))


if __name__ == "__main__":
    main()