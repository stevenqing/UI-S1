#!/usr/bin/env python3
"""
TODO 1: 进一步 spot 成功和失败案例，分析 error 的详细构成

分析内容:
1. Success vs Failure 案例对比
2. Error type 详细分解 (action error vs grounding error)
3. Per-action-type error pattern
4. First-error analysis (trajectory 中第一个错误的特征)
5. Error transition patterns (错误之间的关联)
6. Goal complexity 与 error 的关系
7. 典型成功/失败案例展示
"""

import json
import os
import sys
from collections import Counter, defaultdict
import re

# Paths
EVAL_A_DIR = "outputs/eval_a_ac/Qwen2.5-VL-7B"
C4C7_DIR = "outputs/eval_c4c7_ac/Qwen2.5-VL-7B"
OUTPUT_DIR = "outputs/todo1_error_composition"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_trajectory_results():
    results = []
    path = os.path.join(EVAL_A_DIR, "trajectory_results.jsonl")
    with open(path) as f:
        for line in f:
            results.append(json.loads(line.strip()))
    return results


def load_multisample_results():
    results = []
    path = os.path.join(C4C7_DIR, "multisample_results.jsonl")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        for line in f:
            results.append(json.loads(line.strip()))
    return results


def classify_error(step):
    """分类 step 的 error type"""
    if step["extract_match"]:
        return "correct"
    elif step["type_match"]:
        return "grounding_error"  # action type 对，但 target/coord 错
    else:
        return "action_error"  # action type 就错了


def extract_app_name(goal):
    """从 goal 中提取 app 名称"""
    patterns = [
        r"on the (\w[\w\s]*?) app",
        r"in the (\w[\w\s]*?) app",
        r"open (?:the )?(\w[\w\s]*?)(?:\s+and|\s+app|\s+to)",
        r"using (?:the )?(\w[\w\s]*?) app",
    ]
    for p in patterns:
        m = re.search(p, goal, re.IGNORECASE)
        if m:
            return m.group(1).strip()
    return "unknown"


def analyze_error_composition(trajectories):
    """分析 1: Error 的详细构成"""
    print("=" * 80)
    print("Analysis 1: Error Composition Overview")
    print("=" * 80)

    total_episodes = len(trajectories)
    success_episodes = [t for t in trajectories if t["task_success"]]
    fail_episodes = [t for t in trajectories if not t["task_success"]]

    # 收集所有 step 的 error type
    all_steps = []
    for t in trajectories:
        for s in t["step_results"]:
            s["episode_id"] = t["episode_id"]
            s["task_success"] = t["task_success"]
            s["num_steps"] = t["num_steps"]
            s["goal"] = t["goal"]
            s["error_type"] = classify_error(s)
            all_steps.append(s)

    error_counts = Counter(s["error_type"] for s in all_steps)
    total_steps = len(all_steps)

    print(f"\nTotal episodes: {total_episodes}")
    print(f"  Success: {len(success_episodes)} ({len(success_episodes)/total_episodes*100:.1f}%)")
    print(f"  Failure: {len(fail_episodes)} ({len(fail_episodes)/total_episodes*100:.1f}%)")
    print(f"\nTotal evaluated steps: {total_steps}")
    print(f"  Correct:         {error_counts['correct']:4d} ({error_counts['correct']/total_steps*100:.1f}%)")
    print(f"  Action error:    {error_counts['action_error']:4d} ({error_counts['action_error']/total_steps*100:.1f}%)")
    print(f"  Grounding error: {error_counts['grounding_error']:4d} ({error_counts['grounding_error']/total_steps*100:.1f}%)")

    # Error 在失败 episode 中的分布
    fail_steps = [s for s in all_steps if not s["task_success"]]
    fail_error_counts = Counter(s["error_type"] for s in fail_steps)
    print(f"\nError distribution in FAILED episodes ({len(fail_steps)} steps):")
    for et in ["correct", "action_error", "grounding_error"]:
        c = fail_error_counts.get(et, 0)
        print(f"  {et:20s}: {c:4d} ({c/len(fail_steps)*100:.1f}%)")

    return all_steps, error_counts


def analyze_first_error(trajectories):
    """分析 2: First-Error 特征分析"""
    print("\n" + "=" * 80)
    print("Analysis 2: First-Error Characteristics")
    print("=" * 80)

    fail_trajectories = [t for t in trajectories if not t["task_success"]]

    first_error_step = []
    first_error_type = []
    first_error_action_type = []  # GT action type at first error

    for t in fail_trajectories:
        for s in t["step_results"]:
            if not s["extract_match"]:
                first_error_step.append(s["step_num"])
                first_error_type.append(classify_error(s))
                first_error_action_type.append(s["gt_action_type"])
                break

    # First error step distribution
    step_counter = Counter(first_error_step)
    print(f"\nFirst error step distribution (N={len(fail_trajectories)} failed episodes):")
    for step in sorted(step_counter.keys()):
        c = step_counter[step]
        print(f"  Step {step}: {c:4d} ({c/len(fail_trajectories)*100:.1f}%)")

    # First error type
    type_counter = Counter(first_error_type)
    print(f"\nFirst error type distribution:")
    for et, c in type_counter.most_common():
        print(f"  {et:20s}: {c:4d} ({c/len(fail_trajectories)*100:.1f}%)")

    # First error GT action type
    action_counter = Counter(first_error_action_type)
    print(f"\nGT action type at first error:")
    for at, c in action_counter.most_common():
        print(f"  {at:15s}: {c:4d} ({c/len(fail_trajectories)*100:.1f}%)")

    # Cross-tabulation: first error step × first error type
    cross_tab = defaultdict(lambda: defaultdict(int))
    for step, etype in zip(first_error_step, first_error_type):
        cross_tab[step][etype] += 1

    print(f"\nCross-tabulation: First-error step × First-error type:")
    print(f"  {'Step':>5s} | {'action_error':>14s} | {'grounding_error':>16s} | {'Total':>6s}")
    print(f"  {'-'*5} | {'-'*14} | {'-'*16} | {'-'*6}")
    for step in sorted(cross_tab.keys())[:8]:
        ae = cross_tab[step].get("action_error", 0)
        ge = cross_tab[step].get("grounding_error", 0)
        total = ae + ge
        print(f"  {step:5d} | {ae:14d} | {ge:16d} | {total:6d}")

    return first_error_step, first_error_type


def analyze_per_action_error(all_steps):
    """分析 3: Per-Action-Type Error Pattern"""
    print("\n" + "=" * 80)
    print("Analysis 3: Per-Action-Type Error Pattern")
    print("=" * 80)

    action_stats = defaultdict(lambda: {"total": 0, "correct": 0, "action_error": 0, "grounding_error": 0})

    for s in all_steps:
        gt = s["gt_action_type"]
        et = s["error_type"]
        action_stats[gt]["total"] += 1
        action_stats[gt][et] += 1

    print(f"\n{'GT Action':15s} | {'Total':>6s} | {'Correct':>10s} | {'Action Err':>12s} | {'Grd Err':>10s} | {'Acc':>6s}")
    print(f"{'-'*15} | {'-'*6} | {'-'*10} | {'-'*12} | {'-'*10} | {'-'*6}")

    for at in sorted(action_stats.keys(), key=lambda x: -action_stats[x]["total"]):
        st = action_stats[at]
        acc = st["correct"] / st["total"] * 100 if st["total"] > 0 else 0
        ae_pct = st["action_error"] / st["total"] * 100 if st["total"] > 0 else 0
        ge_pct = st["grounding_error"] / st["total"] * 100 if st["total"] > 0 else 0
        print(f"{at:15s} | {st['total']:6d} | {st['correct']:4d} ({acc:4.1f}%) | {st['action_error']:4d} ({ae_pct:4.1f}%) | {st['grounding_error']:4d} ({ge_pct:4.1f}%) | {acc:5.1f}%")

    # Action confusion matrix: predicted action type vs GT action type
    confusion = defaultdict(lambda: defaultdict(int))
    for s in all_steps:
        gt = s["gt_action_type"]
        if s.get("pred_action") and isinstance(s["pred_action"], dict):
            pred = s["pred_action"].get("action", "unknown")
        else:
            pred = "unknown"
        confusion[gt][pred] += 1

    print(f"\nAction Confusion Matrix (GT rows, Pred cols, showing top confusions):")
    all_pred_types = set()
    for gt_dict in confusion.values():
        all_pred_types.update(gt_dict.keys())
    pred_types = sorted(all_pred_types, key=lambda x: -sum(confusion[gt].get(x, 0) for gt in confusion))[:8]

    header = f"{'GT\\Pred':>15s} |" + "".join(f" {p:>12s} |" for p in pred_types)
    print(header)
    print("-" * len(header))
    for gt in sorted(confusion.keys(), key=lambda x: -sum(confusion[x].values())):
        row = f"{gt:>15s} |"
        total = sum(confusion[gt].values())
        for p in pred_types:
            c = confusion[gt].get(p, 0)
            pct = c / total * 100 if total > 0 else 0
            row += f" {c:4d} ({pct:4.1f}%) |"
        print(row)

    return action_stats, confusion


def analyze_success_failure_patterns(trajectories):
    """分析 4: 成功 vs 失败 episode 的特征对比"""
    print("\n" + "=" * 80)
    print("Analysis 4: Success vs Failure Episode Characteristics")
    print("=" * 80)

    success = [t for t in trajectories if t["task_success"]]
    failure = [t for t in trajectories if not t["task_success"]]

    # Length distribution
    print(f"\nTrajectory length comparison:")
    for name, group in [("Success", success), ("Failure", failure)]:
        lengths = [t["num_steps"] for t in group]
        avg_len = sum(lengths) / len(lengths) if lengths else 0
        len_dist = Counter(t.get("length_bucket", "unknown") for t in group)
        print(f"  {name} (N={len(group)}):")
        print(f"    Avg length: {avg_len:.1f}")
        for bucket in ["short(1-3)", "medium(4-7)", "long(8-15)", "vlong(16+)"]:
            c = len_dist.get(bucket, 0)
            print(f"    {bucket}: {c} ({c/len(group)*100:.1f}%)")

    # Step 0 action type distribution
    print(f"\nStep 0 GT action type comparison:")
    for name, group in [("Success", success), ("Failure", failure)]:
        step0_types = Counter()
        for t in group:
            if t["step_results"]:
                step0_types[t["step_results"][0]["gt_action_type"]] += 1
        print(f"  {name}:")
        for at, c in step0_types.most_common():
            print(f"    {at}: {c} ({c/len(group)*100:.1f}%)")

    # App distribution
    print(f"\nApp distribution (top 10 in failures, showing success rate):")
    app_stats = defaultdict(lambda: {"success": 0, "failure": 0})
    for t in trajectories:
        app = extract_app_name(t["goal"])
        if t["task_success"]:
            app_stats[app]["success"] += 1
        else:
            app_stats[app]["failure"] += 1

    app_list = sorted(app_stats.items(), key=lambda x: -(x[1]["success"] + x[1]["failure"]))[:15]
    print(f"  {'App':>20s} | {'Success':>8s} | {'Failure':>8s} | {'Total':>6s} | {'Rate':>6s}")
    for app, st in app_list:
        total = st["success"] + st["failure"]
        rate = st["success"] / total * 100 if total > 0 else 0
        print(f"  {app:>20s} | {st['success']:8d} | {st['failure']:8d} | {total:6d} | {rate:5.1f}%")


def analyze_error_transitions(trajectories):
    """分析 5: Error Transition Patterns"""
    print("\n" + "=" * 80)
    print("Analysis 5: Error Transition Patterns")
    print("=" * 80)

    # Look at sequences of error types in failed trajectories
    # Step k error type → Step k+1 error type transition
    transitions = defaultdict(lambda: defaultdict(int))
    consecutive_correct = []  # runs of consecutive correct steps before first error

    for t in trajectories:
        if len(t["step_results"]) < 2:
            continue
        run_length = 0
        for i, s in enumerate(t["step_results"]):
            et = classify_error(s)
            if et == "correct":
                run_length += 1
            else:
                if i == 0 or classify_error(t["step_results"][i - 1]) == "correct":
                    consecutive_correct.append(run_length)
                run_length = 0

            if i > 0:
                prev_et = classify_error(t["step_results"][i - 1])
                transitions[prev_et][et] += 1

    print(f"\nError Transition Matrix (P(next | current)):")
    types = ["correct", "action_error", "grounding_error"]
    print(f"  {'Current\\Next':>20s} |" + "".join(f" {t:>18s} |" for t in types))
    for curr in types:
        total = sum(transitions[curr].values())
        row = f"  {curr:>20s} |"
        for nxt in types:
            c = transitions[curr].get(nxt, 0)
            p = c / total if total > 0 else 0
            row += f" {c:5d} ({p:5.1%}) |"
        print(row)

    # Consecutive correct runs
    if consecutive_correct:
        run_dist = Counter(consecutive_correct)
        print(f"\nConsecutive correct steps before error:")
        print(f"  Mean: {sum(consecutive_correct)/len(consecutive_correct):.2f}")
        for r in sorted(run_dist.keys())[:10]:
            c = run_dist[r]
            print(f"  Run={r}: {c} ({c/len(consecutive_correct)*100:.1f}%)")


def analyze_typical_cases(trajectories):
    """分析 6: 展示典型成功/失败案例"""
    print("\n" + "=" * 80)
    print("Analysis 6: Typical Success and Failure Cases")
    print("=" * 80)

    # Top success cases (longest successful trajectories)
    success = sorted(
        [t for t in trajectories if t["task_success"]],
        key=lambda t: -t["num_steps"]
    )

    print(f"\nTop 5 longest successful trajectories:")
    for t in success[:5]:
        print(f"  Episode {t['episode_id']}: {t['num_steps']} steps")
        print(f"    Goal: {t['goal'][:100]}")
        steps_str = " → ".join(
            f"{s['gt_action_type']}({'✓' if s['extract_match'] else '✗'})"
            for s in t["step_results"]
        )
        print(f"    Steps: {steps_str}")

    # Common failure patterns
    failure = [t for t in trajectories if not t["task_success"]]

    # Group by first error type and GT action type at first error
    first_error_patterns = Counter()
    for t in failure:
        for s in t["step_results"]:
            if not s["extract_match"]:
                pattern = f"step{s['step_num']}_{classify_error(s)}_{s['gt_action_type']}"
                if s.get("pred_action") and isinstance(s["pred_action"], dict):
                    pattern += f"→pred_{s['pred_action'].get('action', '?')}"
                first_error_patterns[pattern] += 1
                break

    print(f"\nTop 15 first-error patterns (step_errortype_GT→pred):")
    for pattern, count in first_error_patterns.most_common(15):
        print(f"  {count:4d} ({count/len(failure)*100:.1f}%): {pattern}")

    # Show some failure examples
    print(f"\nExample failure cases (step 0 action error → open):")
    open_failures = [
        t for t in failure
        if t["step_results"] and t["step_results"][0]["gt_action_type"] == "open"
        and not t["step_results"][0]["type_match"]
    ][:3]
    for t in open_failures:
        s = t["step_results"][0]
        print(f"  Episode {t['episode_id']} ({t['num_steps']} steps):")
        print(f"    Goal: {t['goal'][:120]}")
        print(f"    GT: {json.dumps(s['gt_action'])}")
        print(f"    Pred: {json.dumps(s.get('pred_action', 'N/A'))}")

    print(f"\nExample failure cases (step >0 grounding error):")
    late_grd_failures = [
        t for t in failure
        if len(t["step_results"]) > 1
        and t["step_results"][0]["extract_match"]
        and not t["step_results"][-1]["extract_match"]
        and t["step_results"][-1]["type_match"]
    ][:3]
    for t in late_grd_failures:
        s = t["step_results"][-1]  # last evaluated step (the error)
        print(f"  Episode {t['episode_id']} ({t['num_steps']} steps, error at step {s['step_num']}):")
        print(f"    Goal: {t['goal'][:120]}")
        print(f"    GT: {json.dumps(s['gt_action'])}")
        print(f"    Pred: {json.dumps(s.get('pred_action', 'N/A'))}")


def analyze_progress_distribution(trajectories):
    """分析 7: Progress distribution for failed episodes"""
    print("\n" + "=" * 80)
    print("Analysis 7: Failed Episode Progress Distribution")
    print("=" * 80)

    failure = [t for t in trajectories if not t["task_success"]]

    # Progress = final_step_id / num_steps
    progress_values = []
    for t in failure:
        if t["num_steps"] > 0:
            progress = t["final_step_id"] / t["num_steps"]
            progress_values.append(progress)

    # Bucket progress into bins
    bins = [(0, 0), (0, 0.1), (0.1, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0)]
    print(f"\nProgress distribution for failed episodes (N={len(failure)}):")
    print(f"  Progress = completed_steps / total_steps")

    # Actually use final_step_id directly
    completed_steps = [t["final_step_id"] for t in failure]
    step_dist = Counter(completed_steps)
    print(f"\n  Completed steps before failure:")
    for s in sorted(step_dist.keys())[:10]:
        c = step_dist[s]
        print(f"    {s} steps: {c} ({c/len(failure)*100:.1f}%)")

    # By length bucket
    print(f"\n  Progress by length bucket:")
    for bucket in ["short(1-3)", "medium(4-7)", "long(8-15)", "vlong(16+)"]:
        bucket_fails = [t for t in failure if t.get("length_bucket") == bucket]
        if bucket_fails:
            avg_progress = sum(t["final_step_id"] / t["num_steps"] for t in bucket_fails) / len(bucket_fails)
            avg_completed = sum(t["final_step_id"] for t in bucket_fails) / len(bucket_fails)
            print(f"    {bucket:15s}: N={len(bucket_fails):4d}, avg_progress={avg_progress:.3f}, avg_completed={avg_completed:.1f} steps")


def main():
    print("Loading trajectory results...")
    trajectories = load_trajectory_results()
    print(f"Loaded {len(trajectories)} trajectories")

    all_steps, error_counts = analyze_error_composition(trajectories)
    first_error_step, first_error_type = analyze_first_error(trajectories)
    action_stats, confusion = analyze_per_action_error(all_steps)
    analyze_success_failure_patterns(trajectories)
    analyze_error_transitions(trajectories)
    analyze_typical_cases(trajectories)
    analyze_progress_distribution(trajectories)

    # Save results
    results = {
        "total_episodes": len(trajectories),
        "success_count": sum(1 for t in trajectories if t["task_success"]),
        "total_steps": len(all_steps),
        "error_distribution": dict(error_counts),
        "first_error_step_distribution": dict(Counter(first_error_step)),
        "first_error_type_distribution": dict(Counter(first_error_type)),
        "action_stats": {k: dict(v) for k, v in action_stats.items()},
        "confusion_matrix": {k: dict(v) for k, v in confusion.items()},
    }
    with open(os.path.join(OUTPUT_DIR, "error_composition_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {OUTPUT_DIR}/error_composition_results.json")


if __name__ == "__main__":
    main()
