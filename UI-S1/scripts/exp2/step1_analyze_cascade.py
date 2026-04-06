"""
Step 1 Analysis: Cross-dataset Error Cascade Analysis

Reads results from both AndroidControl and GUI-360 no-stop evaluations,
computes unified cascade metrics, and generates comparison tables.
"""

import argparse
import json
import os
from collections import defaultdict


def load_jsonl(path):
    results = []
    with open(path, 'r') as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))
    return results


def compute_cascade_metrics(results, dataset_name):
    """Compute cascade metrics for a set of trajectory results."""
    metrics = {
        'dataset': dataset_name,
        'total_trajectories': len(results),
        'total_steps': 0,
        'total_errors': 0,
    }

    # Step-level analysis
    step0_fail = 0
    step0_total = 0
    first_error_positions = []  # Normalized [0, 1]
    cascade_depths = []
    post_error_correct = 0
    post_error_total = 0
    step_success_by_position = defaultdict(lambda: {'correct': 0, 'total': 0})
    trajectory_success = 0

    # Survival analysis: P(survive | step k correct)
    survival_counts = defaultdict(lambda: {'survived': 0, 'total': 0})

    for r in results:
        steps = r.get('steps', [])
        n = r.get('num_steps', len(steps))
        metrics['total_steps'] += n

        if r.get('trajectory_success', False):
            trajectory_success += 1

        # Step-0 analysis
        if steps:
            step0_total += 1
            if not steps[0].get('success', False):
                step0_fail += 1

        # Per-step position accuracy
        for s in steps:
            sid = s['step_id']
            rel_pos = sid / n if n > 0 else 0
            bucket = int(rel_pos * 10) / 10  # 0.0, 0.1, ..., 0.9
            step_success_by_position[bucket]['total'] += 1
            if s.get('success', False):
                step_success_by_position[bucket]['correct'] += 1

        # First error position
        first_error = r.get('first_error_step', None)
        if first_error is not None:
            first_error_positions.append(first_error / n if n > 0 else 0)
            metrics['total_errors'] += sum(1 for s in steps if not s.get('success', False))

            # Cascade depth: consecutive correct steps after first error
            depth = 0
            for s in steps[first_error + 1:]:
                if s.get('success', False):
                    depth += 1
                else:
                    break
            cascade_depths.append(depth)

            # Post-error accuracy
            for s in steps[first_error + 1:]:
                post_error_total += 1
                if s.get('success', False):
                    post_error_correct += 1
        else:
            # No errors in this trajectory
            first_error_positions.append(1.0)

        # Survival analysis
        for i, s in enumerate(steps):
            if s.get('success', False):
                survival_counts[i]['total'] += 1
                # Check if next step also succeeds
                if i + 1 < len(steps) and steps[i + 1].get('success', False):
                    survival_counts[i]['survived'] += 1

    # Error type breakdown
    error_types = defaultdict(int)
    for r in results:
        for s in r.get('steps', []):
            if not s.get('success', False):
                if not s.get('type_match', False):
                    error_types['planning'] += 1
                else:
                    error_types['grounding'] += 1

    # Aggregate metrics
    metrics['trajectory_success_rate'] = trajectory_success / len(results) * 100 if results else 0
    metrics['step0_failure_rate'] = step0_fail / step0_total * 100 if step0_total > 0 else 0
    metrics['mean_first_error_position'] = sum(first_error_positions) / len(first_error_positions) if first_error_positions else 0
    metrics['mean_cascade_depth'] = sum(cascade_depths) / len(cascade_depths) if cascade_depths else 0
    metrics['post_error_accuracy'] = post_error_correct / post_error_total * 100 if post_error_total > 0 else 0
    metrics['step_accuracy_by_position'] = {
        f"{k:.1f}": v['correct'] / v['total'] * 100 if v['total'] > 0 else 0
        for k, v in sorted(step_success_by_position.items())
    }
    metrics['error_type_distribution'] = dict(error_types)
    metrics['survival_probability'] = {
        str(k): v['survived'] / v['total'] if v['total'] > 0 else 0
        for k, v in sorted(survival_counts.items()) if v['total'] >= 5
    }

    return metrics


def format_comparison_table(ac_metrics, gui360_metrics):
    """Generate a markdown comparison table."""
    lines = []
    lines.append("# Cross-Dataset Error Cascade Analysis")
    lines.append("")
    lines.append("## Key Metrics Comparison")
    lines.append("")
    lines.append("| Metric | AndroidControl | GUI-360 |")
    lines.append("|--------|---------------|---------|")

    def row(name, ac_val, g360_val, fmt=".2f"):
        lines.append(f"| {name} | {ac_val:{fmt}} | {g360_val:{fmt}} |")

    row("Trajectory Success Rate (%)", ac_metrics['trajectory_success_rate'], gui360_metrics['trajectory_success_rate'])
    row("Step-0 Failure Rate (%)", ac_metrics['step0_failure_rate'], gui360_metrics['step0_failure_rate'])
    row("Mean First Error Position", ac_metrics['mean_first_error_position'], gui360_metrics['mean_first_error_position'])
    row("Mean Cascade Depth", ac_metrics['mean_cascade_depth'], gui360_metrics['mean_cascade_depth'])
    row("Post-Error Accuracy (%)", ac_metrics['post_error_accuracy'], gui360_metrics['post_error_accuracy'])

    lines.append("")
    lines.append("## Error Type Distribution")
    lines.append("")
    lines.append("| Error Type | AndroidControl | GUI-360 |")
    lines.append("|-----------|---------------|---------|")

    ac_errors = ac_metrics['error_type_distribution']
    g360_errors = gui360_metrics['error_type_distribution']
    ac_total = sum(ac_errors.values()) or 1
    g360_total = sum(g360_errors.values()) or 1

    for etype in ['planning', 'grounding']:
        ac_pct = ac_errors.get(etype, 0) / ac_total * 100
        g360_pct = g360_errors.get(etype, 0) / g360_total * 100
        lines.append(f"| {etype.capitalize()} | {ac_pct:.1f}% ({ac_errors.get(etype, 0)}) | {g360_pct:.1f}% ({g360_errors.get(etype, 0)}) |")

    lines.append("")
    lines.append("## Step Position Accuracy")
    lines.append("")
    lines.append("| Position | AndroidControl | GUI-360 |")
    lines.append("|----------|---------------|---------|")

    all_positions = sorted(set(
        list(ac_metrics['step_accuracy_by_position'].keys()) +
        list(gui360_metrics['step_accuracy_by_position'].keys())
    ))
    for pos in all_positions:
        ac_acc = ac_metrics['step_accuracy_by_position'].get(pos, 'N/A')
        g360_acc = gui360_metrics['step_accuracy_by_position'].get(pos, 'N/A')
        ac_str = f"{ac_acc:.1f}%" if isinstance(ac_acc, (int, float)) else ac_acc
        g360_str = f"{g360_acc:.1f}%" if isinstance(g360_acc, (int, float)) else g360_acc
        lines.append(f"| {pos} | {ac_str} | {g360_str} |")

    lines.append("")
    lines.append("## Survival Probability P(step k+1 correct | step k correct)")
    lines.append("")
    lines.append("| Step k | AndroidControl | GUI-360 |")
    lines.append("|--------|---------------|---------|")

    ac_surv = ac_metrics.get('survival_probability', {})
    g360_surv = gui360_metrics.get('survival_probability', {})
    all_steps = sorted(set(list(ac_surv.keys()) + list(g360_surv.keys())), key=lambda x: int(x))
    for step in all_steps[:15]:  # Show first 15 steps
        ac_p = ac_surv.get(step, 'N/A')
        g360_p = g360_surv.get(step, 'N/A')
        ac_str = f"{ac_p:.3f}" if isinstance(ac_p, (int, float)) else ac_p
        g360_str = f"{g360_p:.3f}" if isinstance(g360_p, (int, float)) else g360_p
        lines.append(f"| {step} | {ac_str} | {g360_str} |")

    return "\n".join(lines)


def analyze_gui360_results(results_dir):
    """Parse GUI-360 AR evaluation results into unified format."""
    # GUI-360 evaluator outputs detailed JSON results
    # Look for trajectory-level results
    results = []

    # Check for JSONL results (from the AR evaluator)
    for fname in os.listdir(results_dir):
        if fname.endswith('.jsonl'):
            path = os.path.join(results_dir, fname)
            with open(path, 'r') as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        # Convert GUI-360 format to unified format
                        if 'steps' in data:
                            results.append(data)
                        elif 'trajectory_id' in data:
                            # Already in our format
                            results.append(data)

    # Also check for the summary JSON
    for fname in os.listdir(results_dir):
        if fname.endswith('_summary.json'):
            path = os.path.join(results_dir, fname)
            with open(path, 'r') as f:
                summary = json.load(f)
                print(f"GUI-360 Summary found: {fname}")
                print(json.dumps(summary, indent=2)[:500])

    return results


def main(args):
    print("=" * 60)
    print("Cross-Dataset Error Cascade Analysis")
    print("=" * 60)

    # Load AC results
    ac_results = []
    ac_file = os.path.join(args.ac_results_dir, f"ac_nostop_{args.mode}.jsonl")
    if os.path.exists(ac_file):
        ac_results = load_jsonl(ac_file)
        print(f"Loaded {len(ac_results)} AC trajectories from {ac_file}")
    else:
        print(f"WARNING: AC results not found at {ac_file}")

    # Load GUI-360 results
    gui360_results = []
    gui360_file = os.path.join(args.gui360_results_dir, f"gui360_nostop_results.jsonl")
    if os.path.exists(gui360_file):
        gui360_results = load_jsonl(gui360_file)
        print(f"Loaded {len(gui360_results)} GUI-360 trajectories from {gui360_file}")
    else:
        # Try to find any JSONL result files
        if os.path.isdir(args.gui360_results_dir):
            for fname in os.listdir(args.gui360_results_dir):
                if fname.endswith('.jsonl'):
                    fpath = os.path.join(args.gui360_results_dir, fname)
                    gui360_results = load_jsonl(fpath)
                    print(f"Loaded {len(gui360_results)} GUI-360 trajectories from {fpath}")
                    break
        if not gui360_results:
            print(f"WARNING: GUI-360 results not found in {args.gui360_results_dir}")

    # Compute metrics
    if ac_results:
        ac_metrics = compute_cascade_metrics(ac_results, 'AndroidControl')
        ac_metrics_path = os.path.join(args.output_dir, 'ac_cascade_metrics.json')
        with open(ac_metrics_path, 'w') as f:
            json.dump(ac_metrics, f, indent=2)
        print(f"\nAC Cascade Metrics saved to {ac_metrics_path}")
        print(f"  Step-0 Failure Rate: {ac_metrics['step0_failure_rate']:.2f}%")
        print(f"  Mean Cascade Depth: {ac_metrics['mean_cascade_depth']:.2f}")

    if gui360_results:
        gui360_metrics = compute_cascade_metrics(gui360_results, 'GUI-360')
        gui360_metrics_path = os.path.join(args.output_dir, 'gui360_cascade_metrics.json')
        with open(gui360_metrics_path, 'w') as f:
            json.dump(gui360_metrics, f, indent=2)
        print(f"\nGUI-360 Cascade Metrics saved to {gui360_metrics_path}")
        print(f"  Step-0 Failure Rate: {gui360_metrics['step0_failure_rate']:.2f}%")
        print(f"  Mean Cascade Depth: {gui360_metrics['mean_cascade_depth']:.2f}")

    # Cross-dataset comparison
    if ac_results and gui360_results:
        report = format_comparison_table(ac_metrics, gui360_metrics)
        report_path = os.path.join(args.output_dir, 'cross_dataset_cascade_report.md')
        with open(report_path, 'w') as f:
            f.write(report)
        print(f"\nCross-dataset report saved to {report_path}")
        print("\n" + report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ac_results_dir", type=str,
                        default="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/scripts/exp2/results/ac")
    parser.add_argument("--gui360_results_dir", type=str,
                        default="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/scripts/exp2/results/gui360")
    parser.add_argument("--output_dir", type=str,
                        default="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/scripts/exp2/results/analysis")
    parser.add_argument("--mode", type=str, default='natural_cascade')
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    main(args)
