"""Analyze prompt engineering experiment results on Pattern B subset.

Compares subtask_isolated baseline with progress/scene/intent prompt variants.
Produces comparison tables by overall, domain, and subtask count.
"""

import argparse
import json
import os
import re
from collections import defaultdict


def load_eval_results(filepath):
    """Load evaluation results and index by trajectory_id."""
    with open(filepath) as f:
        data = json.load(f)
    results = data.get("trajectory_results", data.get("results", []))
    return {r["trajectory_id"]: r for r in results}


def compute_metrics(results_list):
    """Compute aggregate metrics from a list of trajectory results."""
    if not results_list:
        return {}

    n = len(results_list)
    tsr = sum(1 for r in results_list if r.get("trajectory_success", False)) / n
    avg_progress = sum(r.get("progress_rate", 0) for r in results_list) / n
    avg_scattered = sum(r.get("scattered_progress_rate", 0) for r in results_list) / n

    total_steps = sum(r.get("num_steps", 0) for r in results_list)
    total_correct = 0
    for r in results_list:
        for sr in r.get("step_results", []):
            if sr.get("success", False):
                total_correct += 1
    step_acc = total_correct / total_steps if total_steps > 0 else 0

    return {
        "n_trajectories": n,
        "total_steps": total_steps,
        "tsr": tsr,
        "avg_progress_rate": avg_progress,
        "avg_scattered_progress_rate": avg_scattered,
        "step_accuracy": step_acc,
    }


def is_open_app_subtask(subtask_str):
    """Check if a subtask is just 'Open the application of ...' pattern."""
    s = subtask_str.lower().strip()
    return (
        s.startswith("open the application") or
        s.startswith("open the file") or
        s.startswith("open the powerpoint") or
        s.startswith("open the excel") or
        s.startswith("open the word") or
        s.startswith("launch the") or
        re.match(r"^open .*\.(pptx?|xlsx?|docx?|csv)", s) is not None
    )


def load_gui360_subtask_counts():
    """Load number of genuine subtasks per trajectory."""
    gui360_data = "/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/datasets/GUI-360/test/data"
    traj_subtask_count = {}

    for domain in ["ppt", "excel", "word"]:
        for category in ["in_app", "online", "search"]:
            data_dir = os.path.join(gui360_data, domain, category, "success")
            if not os.path.isdir(data_dir):
                continue

            for fname in sorted(os.listdir(data_dir)):
                if not fname.endswith(".jsonl"):
                    continue

                fpath = os.path.join(data_dir, fname)
                traj_id = f"{domain}_{category}_{fname.replace('.jsonl', '')}"

                steps = []
                with open(fpath) as f:
                    for line in f:
                        data = json.loads(line)
                        subtask = data.get("step", {}).get("subtask", "")
                        action_func = data.get("step", {}).get("action", {}).get("function", "")
                        if action_func == "drag":
                            continue
                        if not data.get("step", {}).get("action", {}).get("rectangle", {}):
                            continue
                        steps.append(subtask)

                if not steps:
                    continue

                # Count unique non-open-app subtask segments
                segments = []
                current = steps[0]
                for i in range(1, len(steps)):
                    if steps[i] != current and steps[i]:
                        segments.append(current)
                        current = steps[i]
                segments.append(current)

                non_open = [s for s in segments if not is_open_app_subtask(s)]
                traj_subtask_count[traj_id] = len(set(non_open))

    return traj_subtask_count


def print_comparison_table(conditions, all_traj_ids):
    """Print a formatted comparison table."""
    # Header
    print(f"  {'Prompt':<20} {'N':>5} {'TSR':>8} {'Scattered':>10} {'StepAcc':>9} {'dTSR':>8}")
    print(f"  {'-'*20} {'-'*5} {'-'*8} {'-'*10} {'-'*9} {'-'*8}")

    baseline_tsr = None
    for label, results_by_id in conditions:
        subset = [results_by_id[tid] for tid in all_traj_ids if tid in results_by_id]
        m = compute_metrics(subset)
        if not m:
            print(f"  {label:<20} {'N/A':>5}")
            continue

        if baseline_tsr is None:
            baseline_tsr = m["tsr"]
            delta = "-"
        else:
            delta = f"{(m['tsr'] - baseline_tsr)*100:+.2f}pp"

        print(f"  {label:<20} {m['n_trajectories']:>5} {m['tsr']*100:>7.2f}% {m['avg_scattered_progress_rate']*100:>9.2f}% {m['step_accuracy']*100:>8.2f}% {delta:>8}")


def main():
    parser = argparse.ArgumentParser(description="Analyze prompt engineering experiment results")
    parser.add_argument("--baseline", type=str, required=True,
                        help="Path to subtask_isolated baseline results JSON")
    parser.add_argument("--progress", type=str, required=True,
                        help="Path to progress prompt results JSON")
    parser.add_argument("--scene", type=str, required=True,
                        help="Path to scene prompt results JSON")
    parser.add_argument("--intent", type=str, required=True,
                        help="Path to intent prompt results JSON")
    parser.add_argument("--pattern_b_ids", type=str, default=None,
                        help="Path to pattern_b_ids.json (auto-detected if not set)")
    args = parser.parse_args()

    # Load results
    print("Loading results...")
    baseline = load_eval_results(args.baseline)
    progress = load_eval_results(args.progress)
    scene = load_eval_results(args.scene)
    intent = load_eval_results(args.intent)

    # Determine trajectory IDs to compare
    if args.pattern_b_ids:
        with open(args.pattern_b_ids) as f:
            all_ids = set(json.load(f))
    else:
        # Use intersection of all conditions
        all_ids = set(baseline.keys()) & set(progress.keys()) & set(scene.keys()) & set(intent.keys())

    print(f"Comparing {len(all_ids)} trajectories\n")

    conditions = [
        ("subtask_isolated", baseline),
        ("progress", progress),
        ("scene", scene),
        ("intent", intent),
    ]

    # Overall comparison
    print("=" * 70)
    print("OVERALL COMPARISON (Pattern B Subset)")
    print("=" * 70)
    print_comparison_table(conditions, all_ids)

    # Domain breakdown
    print(f"\n{'=' * 70}")
    print("BY DOMAIN")
    print(f"{'=' * 70}")

    domain_groups = defaultdict(set)
    for tid in all_ids:
        domain = tid.split("_")[0] if "_" in tid else "unknown"
        domain_groups[domain].add(tid)

    for domain in sorted(domain_groups.keys()):
        tids = domain_groups[domain]
        print(f"\n  --- {domain.upper()} (N={len(tids)}) ---")
        print_comparison_table(conditions, tids)

    # Subtask count breakdown
    print(f"\n{'=' * 70}")
    print("BY SUBTASK COUNT")
    print(f"{'=' * 70}")

    subtask_counts = load_gui360_subtask_counts()
    count_groups = defaultdict(set)
    for tid in all_ids:
        n_sub = subtask_counts.get(tid, 0)
        group = "2 subtasks" if n_sub == 2 else "3+ subtasks"
        count_groups[group].add(tid)

    for group in ["2 subtasks", "3+ subtasks"]:
        tids = count_groups.get(group, set())
        if tids:
            print(f"\n  --- {group} (N={len(tids)}) ---")
            print_comparison_table(conditions, tids)

    print(f"\n{'=' * 70}")
    print("DONE")


if __name__ == "__main__":
    main()
