#!/usr/bin/env python3
"""
TODO 2: 切成 subtask 来进行进一步分析，看 agent 在 subtask 上是否表现更好

将 trajectory 切分为以下 subtask 类型:
1. App Launch: step 0 如果是 open action
2. Navigation: click/swipe/system_button 用于导航
3. Content Interaction: type/click 用于内容输入
4. Confirmation: wait/click 用于确认操作

分析:
- 每种 subtask 的准确率
- Subtask boundary 处的 error rate
- 连续相同 subtask 的准确率变化
- Subtask 长度与准确率的关系
"""

import json
import os
from collections import Counter, defaultdict

EVAL_A_DIR = "outputs/eval_a_ac/Qwen2.5-VL-7B"
OUTPUT_DIR = "outputs/todo2_subtask_analysis"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_trajectory_results():
    results = []
    path = os.path.join(EVAL_A_DIR, "trajectory_results.jsonl")
    with open(path) as f:
        for line in f:
            results.append(json.loads(line.strip()))
    return results


def classify_subtask(step, prev_step=None, goal=""):
    """
    将 step 分类为 subtask 类型:
    - app_launch: open action 或 step 0 的导航到 app
    - navigation: system_button, swipe, 以及部分 click (导航性质)
    - content_input: type action
    - target_interaction: click 用于操作目标元素
    - wait_confirm: wait action
    """
    gt_type = step["gt_action_type"]

    if gt_type == "open":
        return "app_launch"
    elif gt_type == "type":
        return "content_input"
    elif gt_type == "wait":
        return "wait_confirm"
    elif gt_type == "long_press":
        return "target_interaction"
    elif gt_type == "system_button":
        return "navigation"
    elif gt_type == "swipe":
        return "navigation"
    elif gt_type == "click":
        # Heuristic: step 0 click 可能是 app launch 相关
        if step["step_num"] == 0:
            return "app_launch"
        else:
            return "target_interaction"
    else:
        return "other"


def segment_trajectory_into_subtasks(trajectory):
    """
    将 trajectory 分段为 subtask 序列
    返回 subtask 列表，每个 subtask 包含连续的同类型 steps
    """
    subtasks = []
    current_subtask = None
    current_steps = []

    for i, step in enumerate(trajectory["step_results"]):
        prev_step = trajectory["step_results"][i - 1] if i > 0 else None
        st_type = classify_subtask(step, prev_step, trajectory["goal"])

        if st_type != current_subtask:
            if current_subtask is not None:
                subtasks.append({
                    "type": current_subtask,
                    "steps": current_steps,
                    "start_step": current_steps[0]["step_num"],
                    "end_step": current_steps[-1]["step_num"],
                    "length": len(current_steps),
                })
            current_subtask = st_type
            current_steps = [step]
        else:
            current_steps.append(step)

    if current_subtask is not None:
        subtasks.append({
            "type": current_subtask,
            "steps": current_steps,
            "start_step": current_steps[0]["step_num"],
            "end_step": current_steps[-1]["step_num"],
            "length": len(current_steps),
        })

    return subtasks


def analyze_subtask_accuracy(trajectories):
    """分析 1: 每种 subtask 类型的准确率"""
    print("=" * 80)
    print("Analysis 1: Per-Subtask-Type Accuracy")
    print("=" * 80)

    subtask_stats = defaultdict(lambda: {
        "total_steps": 0, "correct_steps": 0,
        "total_subtasks": 0, "perfect_subtasks": 0,
        "lengths": []
    })

    all_subtasks = []
    for t in trajectories:
        subtasks = segment_trajectory_into_subtasks(t)
        for st in subtasks:
            st_type = st["type"]
            n_correct = sum(1 for s in st["steps"] if s["extract_match"])
            n_total = len(st["steps"])
            is_perfect = (n_correct == n_total)

            subtask_stats[st_type]["total_steps"] += n_total
            subtask_stats[st_type]["correct_steps"] += n_correct
            subtask_stats[st_type]["total_subtasks"] += 1
            subtask_stats[st_type]["perfect_subtasks"] += int(is_perfect)
            subtask_stats[st_type]["lengths"].append(n_total)

            all_subtasks.append({
                "type": st_type,
                "length": n_total,
                "n_correct": n_correct,
                "is_perfect": is_perfect,
                "episode_id": t["episode_id"],
            })

    print(f"\n{'Subtask Type':>20s} | {'#Subtasks':>10s} | {'#Steps':>7s} | {'Step Acc':>9s} | {'Perfect%':>9s} | {'Avg Len':>8s}")
    print(f"{'-'*20} | {'-'*10} | {'-'*7} | {'-'*9} | {'-'*9} | {'-'*8}")

    for st_type in sorted(subtask_stats.keys(), key=lambda x: -subtask_stats[x]["total_steps"]):
        st = subtask_stats[st_type]
        step_acc = st["correct_steps"] / st["total_steps"] * 100 if st["total_steps"] > 0 else 0
        perfect_pct = st["perfect_subtasks"] / st["total_subtasks"] * 100 if st["total_subtasks"] > 0 else 0
        avg_len = sum(st["lengths"]) / len(st["lengths"]) if st["lengths"] else 0
        print(f"{st_type:>20s} | {st['total_subtasks']:>10d} | {st['total_steps']:>7d} | {step_acc:>8.1f}% | {perfect_pct:>8.1f}% | {avg_len:>7.1f}")

    return subtask_stats, all_subtasks


def analyze_subtask_boundary_errors(trajectories):
    """分析 2: Subtask boundary 处的 error rate"""
    print("\n" + "=" * 80)
    print("Analysis 2: Error Rate at Subtask Boundaries")
    print("=" * 80)

    boundary_errors = {"at_boundary": 0, "at_boundary_total": 0,
                       "within_subtask": 0, "within_subtask_total": 0}

    for t in trajectories:
        subtasks = segment_trajectory_into_subtasks(t)
        for i, st in enumerate(subtasks):
            for j, step in enumerate(st["steps"]):
                is_boundary = (j == 0 and i > 0)  # first step of non-first subtask
                correct = step["extract_match"]
                if is_boundary:
                    boundary_errors["at_boundary_total"] += 1
                    if not correct:
                        boundary_errors["at_boundary"] += 1
                else:
                    boundary_errors["within_subtask_total"] += 1
                    if not correct:
                        boundary_errors["within_subtask"] += 1

    b_rate = boundary_errors["at_boundary"] / boundary_errors["at_boundary_total"] * 100 if boundary_errors["at_boundary_total"] > 0 else 0
    w_rate = boundary_errors["within_subtask"] / boundary_errors["within_subtask_total"] * 100 if boundary_errors["within_subtask_total"] > 0 else 0

    print(f"\nError rate at subtask boundaries: {b_rate:.1f}% ({boundary_errors['at_boundary']}/{boundary_errors['at_boundary_total']})")
    print(f"Error rate within subtask:        {w_rate:.1f}% ({boundary_errors['within_subtask']}/{boundary_errors['within_subtask_total']})")
    print(f"Boundary/Within ratio:            {b_rate/w_rate:.2f}x" if w_rate > 0 else "")

    return boundary_errors


def analyze_subtask_sequence_patterns(trajectories):
    """分析 3: Subtask sequence patterns 与 success 的关系"""
    print("\n" + "=" * 80)
    print("Analysis 3: Subtask Sequence Patterns")
    print("=" * 80)

    sequence_patterns = Counter()
    seq_success = defaultdict(lambda: {"success": 0, "total": 0})

    for t in trajectories:
        subtasks = segment_trajectory_into_subtasks(t)
        seq = " → ".join(st["type"] for st in subtasks)
        sequence_patterns[seq] += 1
        seq_success[seq]["total"] += 1
        if t["task_success"]:
            seq_success[seq]["success"] += 1

    print(f"\nTop 15 subtask sequence patterns:")
    print(f"  {'Count':>6s} | {'Success':>8s} | {'Rate':>6s} | Sequence")
    for seq, count in sequence_patterns.most_common(15):
        s = seq_success[seq]
        rate = s["success"] / s["total"] * 100 if s["total"] > 0 else 0
        print(f"  {count:>6d} | {s['success']:>8d} | {rate:>5.1f}% | {seq}")


def analyze_subtask_position_accuracy(trajectories):
    """分析 4: Subtask 在 trajectory 中的位置对准确率的影响"""
    print("\n" + "=" * 80)
    print("Analysis 4: Subtask Position vs Accuracy")
    print("=" * 80)

    position_stats = defaultdict(lambda: {"correct": 0, "total": 0})

    for t in trajectories:
        subtasks = segment_trajectory_into_subtasks(t)
        num_subtasks = len(subtasks)
        for i, st in enumerate(subtasks):
            # Position buckets: first, middle, last
            if i == 0:
                pos = "first"
            elif i == num_subtasks - 1:
                pos = "last"
            else:
                pos = "middle"

            for step in st["steps"]:
                position_stats[f"{pos}_{st['type']}"]["total"] += 1
                if step["extract_match"]:
                    position_stats[f"{pos}_{st['type']}"]["correct"] += 1

            # Also track by pure position
            position_stats[f"pos_{pos}"]["total"] += len(st["steps"])
            position_stats[f"pos_{pos}"]["correct"] += sum(1 for s in st["steps"] if s["extract_match"])

    print(f"\nAccuracy by subtask position:")
    for pos in ["first", "middle", "last"]:
        key = f"pos_{pos}"
        st = position_stats[key]
        acc = st["correct"] / st["total"] * 100 if st["total"] > 0 else 0
        print(f"  {pos:>8s}: {acc:.1f}% ({st['correct']}/{st['total']})")

    print(f"\nAccuracy by position × subtask type:")
    print(f"  {'Position_Type':>30s} | {'Acc':>7s} | {'N':>5s}")
    for key in sorted(position_stats.keys()):
        if key.startswith("pos_"):
            continue
        st = position_stats[key]
        if st["total"] < 10:
            continue
        acc = st["correct"] / st["total"] * 100 if st["total"] > 0 else 0
        print(f"  {key:>30s} | {acc:>6.1f}% | {st['total']:>5d}")


def analyze_subtask_length_vs_accuracy(all_subtasks):
    """分析 5: Subtask length 与 accuracy 的关系"""
    print("\n" + "=" * 80)
    print("Analysis 5: Subtask Length vs Accuracy")
    print("=" * 80)

    length_stats = defaultdict(lambda: {"perfect": 0, "total": 0, "correct_steps": 0, "total_steps": 0})

    for st in all_subtasks:
        l = min(st["length"], 5)  # cap at 5+
        length_stats[l]["total"] += 1
        length_stats[l]["perfect"] += int(st["is_perfect"])
        length_stats[l]["correct_steps"] += st["n_correct"]
        length_stats[l]["total_steps"] += st["length"]

    print(f"\n{'Length':>8s} | {'#Subtasks':>10s} | {'Perfect%':>9s} | {'Step Acc':>9s}")
    print(f"{'-'*8} | {'-'*10} | {'-'*9} | {'-'*9}")
    for l in sorted(length_stats.keys()):
        st = length_stats[l]
        perfect_pct = st["perfect"] / st["total"] * 100 if st["total"] > 0 else 0
        step_acc = st["correct_steps"] / st["total_steps"] * 100 if st["total_steps"] > 0 else 0
        label = f"{l}+" if l == 5 else str(l)
        print(f"{label:>8s} | {st['total']:>10d} | {perfect_pct:>8.1f}% | {step_acc:>8.1f}%")


def analyze_single_step_subtask_accuracy(trajectories):
    """分析 6: 如果整个 task 只有 1 个 subtask，accuracy 对比"""
    print("\n" + "=" * 80)
    print("Analysis 6: Single-Step Task vs Multi-Step Task Accuracy")
    print("=" * 80)

    for bucket in ["short(1-3)", "medium(4-7)", "long(8-15)"]:
        bucket_trajs = [t for t in trajectories if t.get("length_bucket") == bucket]
        if not bucket_trajs:
            continue

        # Accuracy per step position
        step_acc = defaultdict(lambda: {"correct": 0, "total": 0})
        for t in bucket_trajs:
            for s in t["step_results"]:
                step_acc[s["step_num"]]["total"] += 1
                if s["extract_match"]:
                    step_acc[s["step_num"]]["correct"] += 1

        print(f"\n  {bucket} (N={len(bucket_trajs)}):")
        for step_num in sorted(step_acc.keys())[:8]:
            st = step_acc[step_num]
            acc = st["correct"] / st["total"] * 100 if st["total"] > 0 else 0
            print(f"    Step {step_num}: {acc:.1f}% ({st['correct']}/{st['total']})")


def main():
    print("Loading trajectory results...")
    trajectories = load_trajectory_results()
    print(f"Loaded {len(trajectories)} trajectories")

    subtask_stats, all_subtasks = analyze_subtask_accuracy(trajectories)
    boundary_errors = analyze_subtask_boundary_errors(trajectories)
    analyze_subtask_sequence_patterns(trajectories)
    analyze_subtask_position_accuracy(trajectories)
    analyze_subtask_length_vs_accuracy(all_subtasks)
    analyze_single_step_subtask_accuracy(trajectories)

    # Save results
    results = {
        "subtask_stats": {
            k: {kk: vv for kk, vv in v.items() if kk != "lengths"}
            for k, v in subtask_stats.items()
        },
        "boundary_errors": boundary_errors,
    }
    with open(os.path.join(OUTPUT_DIR, "subtask_analysis_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {OUTPUT_DIR}/subtask_analysis_results.json")


if __name__ == "__main__":
    main()
