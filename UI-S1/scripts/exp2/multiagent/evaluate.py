"""Multi-Agent Decomposition Experiment — Evaluation (Exp2d).

Loads JSONL results from one or more conditions and computes metrics:
  - Step-level: accuracy, function/args/status match, by domain, by position
  - Trajectory-level: TSR, sequential/scattered progress
  - Agent quality: thought-hit rate, output length
  - Error type classification: CASCADE, PLANNING, GROUNDING, NO_HIT
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
ALL_CONDITIONS = sorted(TEXT_CONDITIONS | ACTION_CONDITIONS)
DOMAINS = ["excel", "ppt", "word"]
POSITIONS = ["early", "mid", "late"]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_condition_results(results_dir, condition_name):
    """Load JSONL results for a single condition."""
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
    """Load C0 baseline results and flatten to step-level list."""
    with open(c0_path) as f:
        data = json.load(f)

    steps = []
    for traj in data["detailed_results"]:
        for sr in traj["step_results"]:
            steps.append({
                "sample_id": sr["sample_id"],
                "trajectory_id": traj["trajectory_id"],
                "domain": traj["domain"],
                "category": traj["category"],
                "success": sr.get("success", False),
                "function_match": sr.get("function_match", False),
                "args_match": sr.get("args_match", False),
                "status_match": sr.get("status_match", False),
                "predicted_function": sr.get("predicted_function"),
                "predicted_args": sr.get("predicted_args"),
                "predicted_status": sr.get("predicted_status"),
                "gt_function": sr.get("ground_truth_function"),
                "gt_args": sr.get("ground_truth_args"),
                "gt_status": sr.get("ground_truth_status"),
                "gt_rect": sr.get("ground_truth_rect"),
            })
    return steps


# ---------------------------------------------------------------------------
# Step-level metrics
# ---------------------------------------------------------------------------

def compute_step_metrics(results):
    """Compute step-level accuracy metrics."""
    n = len(results)
    if n == 0:
        return {}

    metrics = {
        "n_steps": n,
        "step_accuracy": sum(1 for r in results if r.get("success")) / n,
        "function_match_rate": sum(1 for r in results if r.get("function_match")) / n,
        "args_match_rate": sum(1 for r in results if r.get("args_match")) / n,
        "status_match_rate": sum(1 for r in results if r.get("status_match")) / n,
    }

    # Domain breakdown
    by_domain = defaultdict(list)
    for r in results:
        by_domain[r.get("domain", "unknown")].append(r)

    metrics["by_domain"] = {}
    for domain in DOMAINS:
        dr = by_domain.get(domain, [])
        if dr:
            metrics["by_domain"][domain] = {
                "n": len(dr),
                "step_accuracy": sum(1 for r in dr if r.get("success")) / len(dr),
                "function_match": sum(1 for r in dr if r.get("function_match")) / len(dr),
                "args_match": sum(1 for r in dr if r.get("args_match")) / len(dr),
                "status_match": sum(1 for r in dr if r.get("status_match")) / len(dr),
            }

    # Position breakdown
    by_pos = defaultdict(list)
    for r in results:
        by_pos[r.get("position_bucket", "unknown")].append(r)

    metrics["by_position"] = {}
    for pos in POSITIONS:
        pr = by_pos.get(pos, [])
        if pr:
            metrics["by_position"][pos] = {
                "n": len(pr),
                "step_accuracy": sum(1 for r in pr if r.get("success")) / len(pr),
                "function_match": sum(1 for r in pr if r.get("function_match")) / len(pr),
                "args_match": sum(1 for r in pr if r.get("args_match")) / len(pr),
            }

    return metrics


# ---------------------------------------------------------------------------
# Trajectory-level metrics
# ---------------------------------------------------------------------------

def compute_trajectory_metrics(results):
    """Compute trajectory-level TSR, sequential/scattered progress."""
    by_traj = defaultdict(list)
    for r in results:
        by_traj[r.get("trajectory_id", "")].append(r)

    n_traj = len(by_traj)
    if n_traj == 0:
        return {}

    tsr_count = 0
    progress_rates = []
    scattered_rates = []

    for traj_id, steps in by_traj.items():
        # Sort by step_num
        steps.sort(key=lambda s: s.get("step_num", 0))
        n = len(steps)

        all_correct = all(s.get("success", False) for s in steps)
        if all_correct:
            tsr_count += 1

        # Sequential progress: fraction correct before first error
        first_error = n
        for i, s in enumerate(steps):
            if not s.get("success", False):
                first_error = i
                break
        progress_rates.append(first_error / n if n > 0 else 0)

        # Scattered progress: fraction correct regardless of order
        n_correct = sum(1 for s in steps if s.get("success", False))
        scattered_rates.append(n_correct / n if n > 0 else 0)

    return {
        "n_trajectories": n_traj,
        "tsr": tsr_count / n_traj,
        "avg_sequential_progress": float(np.mean(progress_rates)),
        "avg_scattered_progress": float(np.mean(scattered_rates)),
    }


# ---------------------------------------------------------------------------
# Agent quality metrics (for text conditions)
# ---------------------------------------------------------------------------

def compute_agent_quality(results):
    """Compute thought-hit rate and output stats for text conditions."""
    n = len(results)
    if n == 0:
        return {}

    texts = [r.get("text_output", "") for r in results]
    valid_texts = [t for t in texts if t]

    hit_count = sum(1 for r in results if r.get("control_test_hit", False))

    metrics = {
        "n_steps": n,
        "n_valid": len(valid_texts),
        "thought_hit_rate": hit_count / n if n > 0 else 0,
        "avg_length_chars": float(np.mean([len(t) for t in valid_texts])) if valid_texts else 0,
        "avg_length_words": float(np.mean([len(t.split()) for t in valid_texts])) if valid_texts else 0,
    }

    # Per-domain thought-hit rate
    by_domain = defaultdict(list)
    for r in results:
        by_domain[r.get("domain", "unknown")].append(r)

    metrics["by_domain"] = {}
    for domain in DOMAINS:
        dr = by_domain.get(domain, [])
        if dr:
            metrics["by_domain"][domain] = {
                "n": len(dr),
                "thought_hit_rate": sum(1 for r in dr if r.get("control_test_hit")) / len(dr),
            }

    return metrics


# ---------------------------------------------------------------------------
# Error type classification
# ---------------------------------------------------------------------------

def classify_errors(results, c0_results_by_id=None):
    """Classify errors into CASCADE, PLANNING, GROUNDING, NO_HIT.

    CASCADE: previous step in same trajectory was wrong (error propagated)
    PLANNING: function_match=False (wrong action type)
    GROUNDING: function_match=True but args_match=False (right action, wrong target)
    NO_HIT: function & args match but status_match=False (wrong completion signal)
    """
    # Sort by trajectory and step
    by_traj = defaultdict(list)
    for r in results:
        by_traj[r.get("trajectory_id", "")].append(r)

    error_types = defaultdict(int)
    error_details = []
    total_errors = 0

    for traj_id, steps in by_traj.items():
        steps.sort(key=lambda s: s.get("step_num", 0))
        prev_correct = True

        for s in steps:
            if s.get("success", False):
                prev_correct = True
                continue

            total_errors += 1
            error_type = None

            # Check cascade: was previous step wrong?
            if not prev_correct:
                error_type = "CASCADE"
            elif not s.get("function_match", False):
                error_type = "PLANNING"
            elif not s.get("args_match", False):
                error_type = "GROUNDING"
            elif not s.get("status_match", False):
                error_type = "NO_HIT"
            else:
                error_type = "UNKNOWN"

            error_types[error_type] += 1
            error_details.append({
                "sample_id": s.get("sample_id"),
                "error_type": error_type,
                "function_match": s.get("function_match"),
                "args_match": s.get("args_match"),
                "status_match": s.get("status_match"),
            })
            prev_correct = False

    return {
        "total_errors": total_errors,
        "error_distribution": dict(error_types),
        "error_rates": {
            k: v / total_errors if total_errors > 0 else 0
            for k, v in error_types.items()
        },
    }


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def evaluate_condition(results, condition_name, c0_by_id=None):
    """Full evaluation for one condition."""
    eval_result = {
        "condition": condition_name,
        "n_results": len(results),
    }

    if condition_name in TEXT_CONDITIONS:
        eval_result["agent_quality"] = compute_agent_quality(results)
    else:
        eval_result["step_metrics"] = compute_step_metrics(results)
        eval_result["trajectory_metrics"] = compute_trajectory_metrics(results)
        eval_result["error_analysis"] = classify_errors(results, c0_by_id)

    return eval_result


def format_summary_table(evaluations):
    """Format a markdown summary table across all conditions."""
    lines = []
    lines.append("# Multi-Agent Decomposition — Evaluation Summary\n")

    # Action conditions table
    action_evals = {k: v for k, v in evaluations.items() if k in ACTION_CONDITIONS or k == "c0"}
    if action_evals:
        lines.append("## Action Conditions\n")
        lines.append("| Condition | Step Acc | Func Match | Args Match | Status Match | TSR | Seq Progress | Scat Progress |")
        lines.append("|-----------|----------|------------|------------|--------------|-----|-------------|---------------|")

        for cond in ["c0", "f1", "f2", "f3", "f4", "f5", "f6"]:
            ev = action_evals.get(cond)
            if not ev:
                continue
            sm = ev.get("step_metrics", {})
            tm = ev.get("trajectory_metrics", {})
            lines.append(
                f"| {cond} | {sm.get('step_accuracy', 0):.4f} | "
                f"{sm.get('function_match_rate', 0):.4f} | "
                f"{sm.get('args_match_rate', 0):.4f} | "
                f"{sm.get('status_match_rate', 0):.4f} | "
                f"{tm.get('tsr', 0):.4f} | "
                f"{tm.get('avg_sequential_progress', 0):.4f} | "
                f"{tm.get('avg_scattered_progress', 0):.4f} |"
            )
        lines.append("")

    # Domain breakdown
    if action_evals:
        lines.append("## Step Accuracy by Domain\n")
        lines.append("| Condition | Excel | PPT | Word |")
        lines.append("|-----------|-------|-----|------|")
        for cond in ["c0", "f1", "f2", "f3", "f4", "f5", "f6"]:
            ev = action_evals.get(cond)
            if not ev:
                continue
            bd = ev.get("step_metrics", {}).get("by_domain", {})
            cells = []
            for d in DOMAINS:
                dd = bd.get(d, {})
                cells.append(f"{dd.get('step_accuracy', 0):.4f}")
            lines.append(f"| {cond} | {' | '.join(cells)} |")
        lines.append("")

    # Position breakdown
    if action_evals:
        lines.append("## Step Accuracy by Position\n")
        lines.append("| Condition | Early | Mid | Late |")
        lines.append("|-----------|-------|-----|------|")
        for cond in ["c0", "f1", "f2", "f3", "f4", "f5", "f6"]:
            ev = action_evals.get(cond)
            if not ev:
                continue
            bp = ev.get("step_metrics", {}).get("by_position", {})
            cells = []
            for p in POSITIONS:
                pp = bp.get(p, {})
                cells.append(f"{pp.get('step_accuracy', 0):.4f}")
            lines.append(f"| {cond} | {' | '.join(cells)} |")
        lines.append("")

    # Text conditions table
    text_evals = {k: v for k, v in evaluations.items() if k in TEXT_CONDITIONS}
    if text_evals:
        lines.append("## Agent Quality (Text Conditions)\n")
        lines.append("| Condition | Valid | Thought-Hit | Avg Chars | Avg Words |")
        lines.append("|-----------|-------|-------------|-----------|-----------|")
        for cond in ["agent_v", "agent_h", "pass1", "f5_pass1"]:
            ev = text_evals.get(cond)
            if not ev:
                continue
            aq = ev.get("agent_quality", {})
            lines.append(
                f"| {cond} | {aq.get('n_valid', 0)}/{aq.get('n_steps', 0)} | "
                f"{aq.get('thought_hit_rate', 0):.4f} | "
                f"{aq.get('avg_length_chars', 0):.0f} | "
                f"{aq.get('avg_length_words', 0):.0f} |"
            )
        lines.append("")

    # Error analysis
    error_evals = {k: v for k, v in evaluations.items()
                   if k in ACTION_CONDITIONS or k == "c0"}
    if error_evals:
        lines.append("## Error Type Distribution\n")
        lines.append("| Condition | CASCADE | PLANNING | GROUNDING | NO_HIT | Total Errors |")
        lines.append("|-----------|---------|----------|-----------|--------|-------------|")
        for cond in ["c0", "f1", "f2", "f3", "f4", "f5", "f6"]:
            ev = error_evals.get(cond)
            if not ev:
                continue
            ea = ev.get("error_analysis", {})
            dist = ea.get("error_distribution", {})
            total = ea.get("total_errors", 0)
            lines.append(
                f"| {cond} | {dist.get('CASCADE', 0)} | "
                f"{dist.get('PLANNING', 0)} | "
                f"{dist.get('GROUNDING', 0)} | "
                f"{dist.get('NO_HIT', 0)} | {total} |"
            )
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Multi-Agent Decomposition — Evaluation")
    parser.add_argument("--results_dir", type=str, required=True,
                        help="Directory containing condition JSONL files")
    parser.add_argument("--c0_path", type=str, required=True,
                        help="Path to C0 baseline results JSON")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory for eval results (default: results_dir)")
    args = parser.parse_args()

    output_dir = args.output_dir or args.results_dir
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("Multi-Agent Decomposition — Evaluation")
    print(f"Results dir: {args.results_dir}")
    print(f"C0 path: {args.c0_path}")
    print("=" * 60)

    # Load C0 baseline
    print("\nLoading C0 baseline...")
    c0_steps = load_c0_results(args.c0_path)
    c0_by_id = {s["sample_id"]: s for s in c0_steps}
    print(f"  {len(c0_steps)} steps")

    # Evaluate C0
    c0_eval = evaluate_condition(c0_steps, "c0", c0_by_id)

    evaluations = {"c0": c0_eval}

    # Discover and evaluate conditions
    for cond in ALL_CONDITIONS:
        results = load_condition_results(args.results_dir, cond)
        if results is None:
            print(f"\nSkipping {cond} (not found)")
            continue
        print(f"\nEvaluating {cond} ({len(results)} results)...")
        ev = evaluate_condition(results, cond, c0_by_id)
        evaluations[cond] = ev

        # Save per-condition eval
        cond_path = os.path.join(output_dir, f"eval_{cond}.json")
        with open(cond_path, "w") as f:
            json.dump(ev, f, indent=2, default=str)
        print(f"  Saved: {cond_path}")

    # Save all evaluations
    all_path = os.path.join(output_dir, "eval_all.json")
    with open(all_path, "w") as f:
        json.dump(evaluations, f, indent=2, default=str)
    print(f"\nSaved all evaluations: {all_path}")

    # Generate summary table
    summary = format_summary_table(evaluations)
    summary_path = os.path.join(output_dir, "EVALUATION_SUMMARY.md")
    with open(summary_path, "w") as f:
        f.write(summary)
    print(f"Saved summary: {summary_path}")

    # Print summary
    print(f"\n{summary}")


if __name__ == "__main__":
    main()
