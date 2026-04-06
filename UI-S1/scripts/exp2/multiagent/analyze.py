"""Multi-Agent Decomposition Experiment — Cross-Experiment Analysis (Exp2d).

Produces T7.1-T7.5:
  T7.1: Cross-experiment comparison table
  T7.2: Error type shift analysis
  T7.3: Step position analysis
  T7.4: Agent output quality vs downstream performance
  T7.5: Hypothesis validation summary
"""

import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TEXT_CONDITIONS = {"agent_v", "agent_h", "pass1", "f5_pass1"}
ACTION_CONDITIONS = {"f1", "f2", "f3", "f4", "f5", "f6"}
DOMAINS = ["excel", "ppt", "word"]
POSITIONS = ["early", "mid", "late"]

CONDITION_LABELS = {
    "c0": "C0 (SFT v2 baseline)",
    "c3": "C3 (Base model baseline)",
    "f1": "F1 (Two-pass serial)",
    "f2": "F2 (Visual agent)",
    "f3": "F3 (History agent)",
    "f4": "F4 (Full decomposition)",
    "f5": "F5 (Serial + decomposition)",
    "f6": "F6 (Ensemble x3)",
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_eval_results(results_dir):
    """Load all evaluation results."""
    path = os.path.join(results_dir, "eval_all.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def load_condition_jsonl(results_dir, condition_name):
    """Load raw JSONL results for a condition."""
    path = os.path.join(results_dir, f"{condition_name}.jsonl")
    if not os.path.exists(path):
        return None
    results = []
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            results.append(json.loads(line.strip()))
    return results


def load_c0_results(c0_path):
    """Load C0 baseline and flatten to step-level."""
    with open(c0_path) as f:
        data = json.load(f)
    steps = []
    for traj in data["detailed_results"]:
        for sr in traj["step_results"]:
            steps.append({
                "sample_id": sr["sample_id"],
                "trajectory_id": traj["trajectory_id"],
                "domain": traj["domain"],
                "success": sr.get("success", False),
                "function_match": sr.get("function_match", False),
                "args_match": sr.get("args_match", False),
                "status_match": sr.get("status_match", False),
            })
    return steps


# ---------------------------------------------------------------------------
# Statistical helpers
# ---------------------------------------------------------------------------

def proportion_ci(successes, n, z=1.96):
    """Wilson score 95% confidence interval for a proportion."""
    if n == 0:
        return 0, 0, 0
    p = successes / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    spread = z * np.sqrt((p * (1 - p) + z**2 / (4 * n)) / n) / denom
    return p, max(0, center - spread), min(1, center + spread)


def mcnemar_test(a_correct, b_correct):
    """McNemar's test for paired binary outcomes.

    Returns (chi2, p_value, n_discordant).
    """
    n01 = sum(1 for a, b in zip(a_correct, b_correct) if not a and b)
    n10 = sum(1 for a, b in zip(a_correct, b_correct) if a and not b)
    n_disc = n01 + n10
    if n_disc == 0:
        return 0.0, 1.0, 0
    chi2 = (abs(n01 - n10) - 1) ** 2 / n_disc  # continuity correction
    # Approximate p-value from chi2(1)
    from math import erfc, sqrt
    p_value = erfc(sqrt(chi2 / 2))
    return float(chi2), float(p_value), n_disc


def effect_size(acc_a, acc_b, n):
    """Cohen's h effect size for two proportions."""
    import math
    h = 2 * (math.asin(math.sqrt(acc_a)) - math.asin(math.sqrt(acc_b)))
    return float(h)


# ---------------------------------------------------------------------------
# T7.1: Cross-experiment comparison table
# ---------------------------------------------------------------------------

def build_t71_table(evaluations, c0_stats=None):
    """Build cross-experiment comparison table."""
    lines = []
    lines.append("## T7.1: Cross-Experiment Comparison\n")
    lines.append("| Condition | Step Acc | Func Match | Args Match | TSR | Thought-Hit |")
    lines.append("|-----------|----------|------------|------------|-----|-------------|")

    for cond in ["c0", "f1", "f2", "f3", "f4", "f5", "f6"]:
        ev = evaluations.get(cond)
        if not ev:
            continue
        label = CONDITION_LABELS.get(cond, cond)
        sm = ev.get("step_metrics", {})
        tm = ev.get("trajectory_metrics", {})
        lines.append(
            f"| {label} | {sm.get('step_accuracy', 0):.4f} | "
            f"{sm.get('function_match_rate', 0):.4f} | "
            f"{sm.get('args_match_rate', 0):.4f} | "
            f"{tm.get('tsr', 0):.4f} | --- |"
        )

    # Text conditions
    lines.append("")
    lines.append("### Agent Quality\n")
    lines.append("| Condition | Thought-Hit | Avg Chars | Avg Words |")
    lines.append("|-----------|-------------|-----------|-----------|")
    for cond in ["agent_v", "agent_h", "pass1", "f5_pass1"]:
        ev = evaluations.get(cond)
        if not ev:
            continue
        aq = ev.get("agent_quality", {})
        lines.append(
            f"| {cond} | {aq.get('thought_hit_rate', 0):.4f} | "
            f"{aq.get('avg_length_chars', 0):.0f} | "
            f"{aq.get('avg_length_words', 0):.0f} |"
        )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# T7.2: Error type shift analysis
# ---------------------------------------------------------------------------

def build_t72_table(evaluations):
    """Build error type shift analysis."""
    lines = []
    lines.append("## T7.2: Error Type Shift Analysis\n")
    lines.append("| Condition | CASCADE (%) | PLANNING (%) | GROUNDING (%) | NO_HIT (%) | Total Errors |")
    lines.append("|-----------|------------|-------------|--------------|-----------|-------------|")

    for cond in ["c0", "f1", "f2", "f3", "f4", "f5", "f6"]:
        ev = evaluations.get(cond)
        if not ev:
            continue
        ea = ev.get("error_analysis", {})
        dist = ea.get("error_distribution", {})
        total = ea.get("total_errors", 1)
        label = CONDITION_LABELS.get(cond, cond)

        pcts = {
            "CASCADE": dist.get("CASCADE", 0) / total * 100 if total else 0,
            "PLANNING": dist.get("PLANNING", 0) / total * 100 if total else 0,
            "GROUNDING": dist.get("GROUNDING", 0) / total * 100 if total else 0,
            "NO_HIT": dist.get("NO_HIT", 0) / total * 100 if total else 0,
        }

        lines.append(
            f"| {label} | {pcts['CASCADE']:.1f} | {pcts['PLANNING']:.1f} | "
            f"{pcts['GROUNDING']:.1f} | {pcts['NO_HIT']:.1f} | {total} |"
        )

    # Identify which condition best reduces each error type
    lines.append("\n### Best Reduction per Error Type\n")
    c0_ev = evaluations.get("c0")
    if c0_ev:
        c0_ea = c0_ev.get("error_analysis", {}).get("error_distribution", {})
        for err_type in ["CASCADE", "PLANNING", "GROUNDING", "NO_HIT"]:
            c0_count = c0_ea.get(err_type, 0)
            best_cond = None
            best_reduction = 0
            for cond in ["f1", "f2", "f3", "f4", "f5", "f6"]:
                ev = evaluations.get(cond)
                if not ev:
                    continue
                ea = ev.get("error_analysis", {}).get("error_distribution", {})
                reduction = c0_count - ea.get(err_type, 0)
                if reduction > best_reduction:
                    best_reduction = reduction
                    best_cond = cond
            if best_cond:
                lines.append(f"- **{err_type}**: Best reduced by {CONDITION_LABELS.get(best_cond, best_cond)} "
                             f"(C0: {c0_count} -> {c0_count - best_reduction}, delta={best_reduction})")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# T7.3: Step position analysis
# ---------------------------------------------------------------------------

def build_t73_table(evaluations):
    """Build step position analysis."""
    lines = []
    lines.append("## T7.3: Step Position Analysis\n")
    lines.append("### Step Accuracy by Position\n")
    lines.append("| Condition | Early | Mid | Late | Early-Late Gap |")
    lines.append("|-----------|-------|-----|------|---------------|")

    for cond in ["c0", "f1", "f2", "f3", "f4", "f5", "f6"]:
        ev = evaluations.get(cond)
        if not ev:
            continue
        bp = ev.get("step_metrics", {}).get("by_position", {})
        early = bp.get("early", {}).get("step_accuracy", 0)
        mid = bp.get("mid", {}).get("step_accuracy", 0)
        late = bp.get("late", {}).get("step_accuracy", 0)
        gap = early - late
        label = CONDITION_LABELS.get(cond, cond)
        lines.append(f"| {label} | {early:.4f} | {mid:.4f} | {late:.4f} | {gap:+.4f} |")

    # Analysis: where does each condition's improvement concentrate?
    c0_ev = evaluations.get("c0")
    if c0_ev:
        lines.append("\n### Improvement vs C0 by Position\n")
        lines.append("| Condition | Early Delta | Mid Delta | Late Delta |")
        lines.append("|-----------|------------|-----------|-----------|")
        c0_bp = c0_ev.get("step_metrics", {}).get("by_position", {})
        for cond in ["f1", "f2", "f3", "f4", "f5", "f6"]:
            ev = evaluations.get(cond)
            if not ev:
                continue
            bp = ev.get("step_metrics", {}).get("by_position", {})
            deltas = []
            for pos in POSITIONS:
                c0_acc = c0_bp.get(pos, {}).get("step_accuracy", 0)
                cond_acc = bp.get(pos, {}).get("step_accuracy", 0)
                deltas.append(cond_acc - c0_acc)
            label = CONDITION_LABELS.get(cond, cond)
            lines.append(f"| {label} | {deltas[0]:+.4f} | {deltas[1]:+.4f} | {deltas[2]:+.4f} |")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# T7.4: Agent output quality vs downstream performance
# ---------------------------------------------------------------------------

def build_t74_analysis(results_dir, evaluations):
    """Correlate agent quality with downstream F4 performance."""
    lines = []
    lines.append("## T7.4: Agent Quality vs Downstream Performance\n")

    # Load raw data
    agent_v_data = load_condition_jsonl(results_dir, "agent_v")
    agent_h_data = load_condition_jsonl(results_dir, "agent_h")
    f4_data = load_condition_jsonl(results_dir, "f4")

    if not all([agent_v_data, f4_data]):
        lines.append("Insufficient data for T7.4 analysis (need agent_v and f4).\n")
        return "\n".join(lines)

    # Build lookup by sample_id
    v_by_id = {r["sample_id"]: r for r in agent_v_data}
    f4_by_id = {r["sample_id"]: r for r in f4_data}
    h_by_id = {r["sample_id"]: r for r in agent_h_data} if agent_h_data else {}

    # Correlation: Agent V thought-hit -> F4 accuracy
    v_hit_f4_correct = 0
    v_hit_f4_wrong = 0
    v_miss_f4_correct = 0
    v_miss_f4_wrong = 0

    for sid, vr in v_by_id.items():
        f4r = f4_by_id.get(sid)
        if not f4r:
            continue
        v_hit = vr.get("control_test_hit", False)
        f4_ok = f4r.get("success", False)
        if v_hit and f4_ok:
            v_hit_f4_correct += 1
        elif v_hit and not f4_ok:
            v_hit_f4_wrong += 1
        elif not v_hit and f4_ok:
            v_miss_f4_correct += 1
        else:
            v_miss_f4_wrong += 1

    total_v = v_hit_f4_correct + v_hit_f4_wrong + v_miss_f4_correct + v_miss_f4_wrong

    lines.append("### Agent V (Visual) Quality → F4 Accuracy\n")
    lines.append("| Agent V Hit | F4 Correct | F4 Wrong | F4 Acc |")
    lines.append("|-------------|-----------|----------|--------|")
    if v_hit_f4_correct + v_hit_f4_wrong > 0:
        acc_when_hit = v_hit_f4_correct / (v_hit_f4_correct + v_hit_f4_wrong)
        lines.append(f"| Yes | {v_hit_f4_correct} | {v_hit_f4_wrong} | {acc_when_hit:.4f} |")
    if v_miss_f4_correct + v_miss_f4_wrong > 0:
        acc_when_miss = v_miss_f4_correct / (v_miss_f4_correct + v_miss_f4_wrong)
        lines.append(f"| No | {v_miss_f4_correct} | {v_miss_f4_wrong} | {acc_when_miss:.4f} |")

    if total_v > 0 and (v_hit_f4_correct + v_hit_f4_wrong) > 0 and (v_miss_f4_correct + v_miss_f4_wrong) > 0:
        acc_hit = v_hit_f4_correct / (v_hit_f4_correct + v_hit_f4_wrong)
        acc_miss = v_miss_f4_correct / (v_miss_f4_correct + v_miss_f4_wrong)
        lift = acc_hit - acc_miss
        lines.append(f"\nLift when Agent V identifies target: **{lift:+.4f}** ({acc_hit:.4f} vs {acc_miss:.4f})")
        if lift > 0.05:
            lines.append("→ Agent V quality IS a meaningful bottleneck for F4.")
        else:
            lines.append("→ Agent V quality has limited impact on F4 accuracy.")

    # Agent H correlation
    if h_by_id:
        lines.append("\n### Agent H (History) Quality → F4 Accuracy\n")
        h_hit_f4_correct = 0
        h_hit_f4_wrong = 0
        h_miss_f4_correct = 0
        h_miss_f4_wrong = 0

        for sid, hr in h_by_id.items():
            f4r = f4_by_id.get(sid)
            if not f4r:
                continue
            h_hit = hr.get("control_test_hit", False)
            f4_ok = f4r.get("success", False)
            if h_hit and f4_ok:
                h_hit_f4_correct += 1
            elif h_hit and not f4_ok:
                h_hit_f4_wrong += 1
            elif not h_hit and f4_ok:
                h_miss_f4_correct += 1
            else:
                h_miss_f4_wrong += 1

        lines.append("| Agent H Hit | F4 Correct | F4 Wrong | F4 Acc |")
        lines.append("|-------------|-----------|----------|--------|")
        if h_hit_f4_correct + h_hit_f4_wrong > 0:
            acc_h_hit = h_hit_f4_correct / (h_hit_f4_correct + h_hit_f4_wrong)
            lines.append(f"| Yes | {h_hit_f4_correct} | {h_hit_f4_wrong} | {acc_h_hit:.4f} |")
        if h_miss_f4_correct + h_miss_f4_wrong > 0:
            acc_h_miss = h_miss_f4_correct / (h_miss_f4_correct + h_miss_f4_wrong)
            lines.append(f"| No | {h_miss_f4_correct} | {h_miss_f4_wrong} | {acc_h_miss:.4f} |")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# T7.5: Hypothesis validation summary
# ---------------------------------------------------------------------------

def build_t75_summary(evaluations, results_dir, c0_path):
    """Hypothesis validation with statistical tests."""
    lines = []
    lines.append("## T7.5: Hypothesis Validation Summary\n")

    # Load raw data for McNemar tests
    c0_raw = load_c0_results(c0_path)
    c0_by_id = {s["sample_id"]: s["success"] for s in c0_raw}

    raw_data = {}
    for cond in ACTION_CONDITIONS:
        data = load_condition_jsonl(results_dir, cond)
        if data:
            raw_data[cond] = {r["sample_id"]: r.get("success", False) for r in data}

    def get_paired(cond_name):
        """Get paired outcomes vs C0."""
        if cond_name not in raw_data:
            return None, None
        common_ids = sorted(set(c0_by_id.keys()) & set(raw_data[cond_name].keys()))
        if not common_ids:
            return None, None
        c0_outcomes = [c0_by_id[sid] for sid in common_ids]
        cond_outcomes = [raw_data[cond_name][sid] for sid in common_ids]
        return c0_outcomes, cond_outcomes

    def format_hypothesis(name, description, cond_name, expected):
        c0_out, cond_out = get_paired(cond_name)
        if c0_out is None:
            return f"### {name}: {description}\n- Data not available for {cond_name}\n"

        n = len(c0_out)
        c0_acc = sum(c0_out) / n
        cond_acc = sum(cond_out) / n
        delta = cond_acc - c0_acc

        chi2, p_val, n_disc = mcnemar_test(c0_out, cond_out)
        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"

        result = "SUPPORTED" if (delta > 0 and p_val < 0.05) == expected else "NOT SUPPORTED"
        if not expected and delta <= 0:
            result = "SUPPORTED (no improvement as expected)" if p_val >= 0.05 else "PARTIALLY"

        text = f"### {name}: {description}\n"
        text += f"- C0 acc: {c0_acc:.4f} vs {cond_name} acc: {cond_acc:.4f} (delta: {delta:+.4f})\n"
        text += f"- McNemar chi2={chi2:.2f}, p={p_val:.4f} {sig}, n_discordant={n_disc}\n"
        text += f"- **{result}**\n"
        return text

    # H1: Processing order (F1 vs C0)
    lines.append(format_hypothesis(
        "H1", "Two-pass reasoning improves accuracy (F1 > C0)",
        "f1", True))

    # H2-visual: F2 vs C0
    lines.append(format_hypothesis(
        "H2-visual", "Visual agent description improves accuracy (F2 > C0)",
        "f2", True))

    # H2-history: F3 vs C0
    lines.append(format_hypothesis(
        "H2-history", "History agent analysis improves accuracy (F3 > C0)",
        "f3", True))

    # H2-full: F4 vs C0
    lines.append(format_hypothesis(
        "H2-full", "Full decomposition improves accuracy (F4 > C0)",
        "f4", True))

    # H2-complementarity: F4 vs max(F2, F3)
    ev_f2 = evaluations.get("f2", {}).get("step_metrics", {}).get("step_accuracy", 0)
    ev_f3 = evaluations.get("f3", {}).get("step_metrics", {}).get("step_accuracy", 0)
    ev_f4 = evaluations.get("f4", {}).get("step_metrics", {}).get("step_accuracy", 0)
    max_f2_f3 = max(ev_f2, ev_f3)
    lines.append(f"### H2-complementarity: F4 vs max(F2, F3)\n")
    lines.append(f"- F2 acc: {ev_f2:.4f}, F3 acc: {ev_f3:.4f}, max={max_f2_f3:.4f}")
    lines.append(f"- F4 acc: {ev_f4:.4f}")
    lines.append(f"- Delta F4 - max(F2,F3): {ev_f4 - max_f2_f3:+.4f}")
    if ev_f4 > max_f2_f3:
        lines.append("- **SUPPORTED**: Visual + History agents are complementary\n")
    else:
        lines.append("- **NOT SUPPORTED**: No complementarity benefit\n")

    # H1+H2 stacking: F5 vs max(F1, F4)
    ev_f1 = evaluations.get("f1", {}).get("step_metrics", {}).get("step_accuracy", 0)
    ev_f5 = evaluations.get("f5", {}).get("step_metrics", {}).get("step_accuracy", 0)
    max_f1_f4 = max(ev_f1, ev_f4)
    lines.append(f"### H1+H2 stacking: F5 vs max(F1, F4)\n")
    lines.append(f"- F1 acc: {ev_f1:.4f}, F4 acc: {ev_f4:.4f}, max={max_f1_f4:.4f}")
    lines.append(f"- F5 acc: {ev_f5:.4f}")
    lines.append(f"- Delta F5 - max(F1,F4): {ev_f5 - max_f1_f4:+.4f}")
    if ev_f5 > max_f1_f4:
        lines.append("- **SUPPORTED**: Stacking reasoning + decomposition helps\n")
    else:
        lines.append("- **NOT SUPPORTED**: No stacking benefit\n")

    # Compute effect: F4 vs F6
    ev_f6 = evaluations.get("f6", {}).get("step_metrics", {}).get("step_accuracy", 0)
    lines.append(f"### Compute effect: F4 (decomposition) vs F6 (ensemble)\n")
    lines.append(f"- F4 acc: {ev_f4:.4f}")
    lines.append(f"- F6 acc: {ev_f6:.4f}")
    lines.append(f"- Delta F4 - F6: {ev_f4 - ev_f6:+.4f}")
    if ev_f4 > ev_f6:
        lines.append("- Decomposition > ensemble (information > compute)\n")
    elif ev_f6 > ev_f4:
        lines.append("- Ensemble > decomposition (compute > information)\n")
    else:
        lines.append("- Comparable performance\n")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Multi-Agent Decomposition — Cross-Experiment Analysis")
    parser.add_argument("--results_dir", type=str, required=True,
                        help="Directory containing eval results and JSONL files")
    parser.add_argument("--c0_path", type=str, required=True,
                        help="Path to C0 baseline results JSON")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (default: results_dir)")
    args = parser.parse_args()

    output_dir = args.output_dir or args.results_dir
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("Multi-Agent Decomposition — Cross-Experiment Analysis")
    print(f"Results dir: {args.results_dir}")
    print("=" * 60)

    # Load evaluations
    evaluations = load_eval_results(args.results_dir)
    if not evaluations:
        print("ERROR: No evaluation results found. Run evaluate.py first.")
        return

    print(f"Loaded evaluations for: {list(evaluations.keys())}")

    # Build report sections
    report_sections = []
    report_sections.append("# Multi-Agent Decomposition Experiment Report (Exp2d)\n")
    report_sections.append(f"Generated: {__import__('datetime').datetime.now().isoformat()}\n")

    # T7.1
    print("\nBuilding T7.1: Cross-experiment comparison...")
    report_sections.append(build_t71_table(evaluations))

    # T7.2
    print("Building T7.2: Error type shift analysis...")
    report_sections.append(build_t72_table(evaluations))

    # T7.3
    print("Building T7.3: Step position analysis...")
    report_sections.append(build_t73_table(evaluations))

    # T7.4
    print("Building T7.4: Agent quality vs downstream performance...")
    report_sections.append(build_t74_analysis(args.results_dir, evaluations))

    # T7.5
    print("Building T7.5: Hypothesis validation...")
    report_sections.append(build_t75_summary(evaluations, args.results_dir, args.c0_path))

    # Write report
    report = "\n\n".join(report_sections)
    report_path = os.path.join(output_dir, "MULTIAGENT_EXPERIMENT_REPORT.md")
    with open(report_path, "w") as f:
        f.write(report)
    print(f"\nSaved report: {report_path}")

    # Save stats JSON
    stats = {
        "evaluations": evaluations,
        "conditions_found": list(evaluations.keys()),
    }

    # Add key comparisons
    c0_acc = evaluations.get("c0", {}).get("step_metrics", {}).get("step_accuracy", 0)
    comparisons = {}
    for cond in ACTION_CONDITIONS:
        ev = evaluations.get(cond, {})
        acc = ev.get("step_metrics", {}).get("step_accuracy", 0)
        comparisons[cond] = {
            "step_accuracy": acc,
            "delta_vs_c0": acc - c0_acc,
        }
    stats["comparisons"] = comparisons

    stats_path = os.path.join(output_dir, "multiagent_stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2, default=str)
    print(f"Saved stats: {stats_path}")

    # Print key findings
    print("\n" + "=" * 60)
    print("KEY FINDINGS")
    print("=" * 60)
    print(f"\nC0 baseline step accuracy: {c0_acc:.4f}")
    for cond in ["f1", "f2", "f3", "f4", "f5", "f6"]:
        comp = comparisons.get(cond, {})
        acc = comp.get("step_accuracy", 0)
        delta = comp.get("delta_vs_c0", 0)
        label = CONDITION_LABELS.get(cond, cond)
        marker = "+" if delta > 0 else ""
        print(f"  {label}: {acc:.4f} ({marker}{delta:.4f})")


if __name__ == "__main__":
    main()
