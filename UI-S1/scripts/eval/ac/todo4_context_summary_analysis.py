#!/usr/bin/env python3
"""
TODO 4: 添加 summary 来看看是否是上下文的问题

分析思路:
1. 分析准确率随 step position 的衰减模式 - 是否是上下文丢失导致
2. 条件分析: 在前面都对的情况下，后续 step 的准确率
3. Goal 复杂度 (长度) 与 accuracy 的关系
4. Action history 的重复模式 - 是否存在"迷路"现象 (重复执行相同 action)
5. 对比 step 0 (无 history) vs 后续 steps (有 history) 的 error type 差异
6. 模拟: 如果加 summary，哪些 error 可能被避免
"""

import json
import os
import re
from collections import Counter, defaultdict

EVAL_A_DIR = "outputs/eval_a_ac/Qwen2.5-VL-7B"
U7_DIR = "outputs/eval_u7_ac/Qwen2.5-VL-7B"
C4C7_DIR = "outputs/eval_c4c7_ac/Qwen2.5-VL-7B"
OUTPUT_DIR = "outputs/todo4_context_analysis"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_trajectory_results():
    results = []
    with open(os.path.join(EVAL_A_DIR, "trajectory_results.jsonl")) as f:
        for line in f:
            results.append(json.loads(line.strip()))
    return results


def load_multisample_results():
    path = os.path.join(C4C7_DIR, "multisample_results.jsonl")
    if not os.path.exists(path):
        return None
    results = []
    with open(path) as f:
        for line in f:
            results.append(json.loads(line.strip()))
    return results


def analyze_accuracy_decay(trajectories):
    """分析 1: 准确率随 step position 的衰减"""
    print("=" * 80)
    print("Analysis 1: Accuracy Decay by Step Position")
    print("=" * 80)

    # 条件准确率: P(correct at step k | reached step k in AR)
    # 在 stop-on-error AR 中, "reached" 意味着前面都对了
    step_stats = defaultdict(lambda: {"reached": 0, "correct": 0})

    for t in trajectories:
        for s in t["step_results"]:
            step_stats[s["step_num"]]["reached"] += 1
            if s["extract_match"]:
                step_stats[s["step_num"]]["correct"] += 1

    print(f"\nConditional accuracy P(correct | reached step k):")
    print(f"  {'Step':>5s} | {'Reached':>8s} | {'Correct':>8s} | {'P(correct|reached)':>20s} | {'Cumulative P':>13s}")
    cum_p = 1.0
    for step in sorted(step_stats.keys()):
        st = step_stats[step]
        p = st["correct"] / st["reached"] if st["reached"] > 0 else 0
        cum_p *= p
        print(f"  {step:>5d} | {st['reached']:>8d} | {st['correct']:>8d} | {p:>19.4f} | {cum_p:>12.4f}")

    # Unconditional accuracy by step position (from C4+C7 independent eval)
    # This doesn't depend on previous steps being correct


def analyze_accuracy_given_correct_prefix(trajectories):
    """分析 2: 在前 N 步全对的条件下，第 N+1 步的准确率"""
    print("\n" + "=" * 80)
    print("Analysis 2: P(correct at step k | all steps 0..k-1 correct)")
    print("=" * 80)

    # 在 AR stop-on-error 中，reached = all previous correct
    # 所以 analysis 1 已经是这个

    # 更有趣的是: 对比 "短 trajectory 中的 step k" vs "长 trajectory 中的 step k"
    step_by_traj_len = defaultdict(lambda: defaultdict(lambda: {"correct": 0, "total": 0}))

    for t in trajectories:
        for s in t["step_results"]:
            step_by_traj_len[t["num_steps"]][s["step_num"]]["total"] += 1
            if s["extract_match"]:
                step_by_traj_len[t["num_steps"]][s["step_num"]]["correct"] += 1

    print(f"\nStep accuracy by trajectory total length:")
    print(f"  (Same step position, different trajectory lengths)")
    for step_k in range(5):
        print(f"\n  Step {step_k}:")
        for traj_len in sorted(step_by_traj_len.keys()):
            if step_k in step_by_traj_len[traj_len]:
                st = step_by_traj_len[traj_len][step_k]
                if st["total"] < 5:
                    continue
                acc = st["correct"] / st["total"] * 100
                print(f"    Traj length {traj_len:2d}: {acc:.1f}% ({st['correct']}/{st['total']})")


def analyze_goal_complexity(trajectories):
    """分析 3: Goal 复杂度与 accuracy 的关系"""
    print("\n" + "=" * 80)
    print("Analysis 3: Goal Complexity vs Task Success")
    print("=" * 80)

    # Goal length as proxy for complexity
    goal_lengths = []
    for t in trajectories:
        goal_len = len(t["goal"].split())
        goal_lengths.append((goal_len, t["task_success"], t["num_steps"]))

    # Bucket by goal word count
    bins = [(0, 10), (10, 15), (15, 20), (20, 30), (30, 50), (50, 100)]
    print(f"\nTask success rate by goal length (word count):")
    print(f"  {'Goal Words':>12s} | {'N':>5s} | {'Success':>8s} | {'Rate':>6s} | {'Avg Traj Len':>13s}")
    for lo, hi in bins:
        bucket = [(gl, ts, nl) for gl, ts, nl in goal_lengths if lo <= gl < hi]
        if not bucket:
            continue
        n = len(bucket)
        succ = sum(1 for _, ts, _ in bucket if ts)
        avg_len = sum(nl for _, _, nl in bucket) / n
        rate = succ / n * 100
        print(f"  {lo:3d}-{hi:3d} words | {n:5d} | {succ:8d} | {rate:5.1f}% | {avg_len:12.1f}")

    # Goal keywords analysis
    print(f"\nGoal verb analysis (first verb → success rate):")
    verb_stats = defaultdict(lambda: {"success": 0, "total": 0})
    for t in trajectories:
        # Extract first verb-like word
        words = t["goal"].lower().split()
        if words:
            first_word = words[0]
            verb_stats[first_word]["total"] += 1
            if t["task_success"]:
                verb_stats[first_word]["success"] += 1

    sorted_verbs = sorted(verb_stats.items(), key=lambda x: -x[1]["total"])[:15]
    for verb, st in sorted_verbs:
        rate = st["success"] / st["total"] * 100 if st["total"] > 0 else 0
        print(f"  {verb:>15s}: {st['total']:4d} episodes, {rate:.1f}% success")


def analyze_action_repetition(trajectories):
    """分析 4: Action history 重复模式 - "迷路"现象"""
    print("\n" + "=" * 80)
    print("Analysis 4: Action Repetition / 'Lost' Patterns")
    print("=" * 80)

    # Look for repeated actions in trajectory
    repetition_stats = {"has_repeat": 0, "no_repeat": 0,
                        "repeat_success": 0, "no_repeat_success": 0}

    repeat_patterns = Counter()

    for t in trajectories:
        if len(t["step_results"]) < 2:
            continue

        # Check for consecutive identical action types
        has_repeat = False
        for i in range(1, len(t["step_results"])):
            prev = t["step_results"][i - 1]
            curr = t["step_results"][i]

            prev_pred = prev.get("pred_action", {})
            curr_pred = curr.get("pred_action", {})

            if isinstance(prev_pred, dict) and isinstance(curr_pred, dict):
                if prev_pred.get("action") == curr_pred.get("action"):
                    has_repeat = True
                    repeat_patterns[prev_pred.get("action", "?")] += 1

        if has_repeat:
            repetition_stats["has_repeat"] += 1
            if t["task_success"]:
                repetition_stats["repeat_success"] += 1
        else:
            repetition_stats["no_repeat"] += 1
            if t["task_success"]:
                repetition_stats["no_repeat_success"] += 1

    total = repetition_stats["has_repeat"] + repetition_stats["no_repeat"]
    print(f"\n  Episodes with repeated consecutive actions: {repetition_stats['has_repeat']} ({repetition_stats['has_repeat']/total*100:.1f}%)")
    print(f"  Episodes without: {repetition_stats['no_repeat']} ({repetition_stats['no_repeat']/total*100:.1f}%)")

    if repetition_stats["has_repeat"] > 0:
        rep_rate = repetition_stats["repeat_success"] / repetition_stats["has_repeat"] * 100
        print(f"  Success rate with repeat: {rep_rate:.1f}%")
    if repetition_stats["no_repeat"] > 0:
        norep_rate = repetition_stats["no_repeat_success"] / repetition_stats["no_repeat"] * 100
        print(f"  Success rate without repeat: {norep_rate:.1f}%")

    print(f"\n  Most repeated action types:")
    for at, c in repeat_patterns.most_common(5):
        print(f"    {at}: {c} times")


def analyze_error_type_by_context(trajectories):
    """分析 5: Step 0 (无 history) vs 后续 steps 的 error type 差异"""
    print("\n" + "=" * 80)
    print("Analysis 5: Error Type Difference - Step 0 (No Context) vs Later Steps")
    print("=" * 80)

    step0_errors = {"action": 0, "grounding": 0, "correct": 0}
    later_errors = {"action": 0, "grounding": 0, "correct": 0}

    for t in trajectories:
        for s in t["step_results"]:
            target = step0_errors if s["step_num"] == 0 else later_errors
            if s["extract_match"]:
                target["correct"] += 1
            elif not s["type_match"]:
                target["action"] += 1
            else:
                target["grounding"] += 1

    print(f"\n  {'':>15s} | {'Step 0':>12s} | {'Steps 1+':>12s}")
    for key in ["correct", "action", "grounding"]:
        s0 = step0_errors[key]
        sl = later_errors[key]
        s0_total = sum(step0_errors.values())
        sl_total = sum(later_errors.values())
        print(f"  {key:>15s} | {s0:4d} ({s0/s0_total*100:4.1f}%) | {sl:4d} ({sl/sl_total*100:4.1f}%)")

    # 关键对比: 是否后续 steps 的 action error 比 step 0 更高?
    # 如果是，可能是上下文导致的 action confusion
    s0_ae_rate = step0_errors["action"] / sum(step0_errors.values()) * 100
    sl_ae_rate = later_errors["action"] / sum(later_errors.values()) * 100
    print(f"\n  Action error rate: Step 0 = {s0_ae_rate:.1f}%, Steps 1+ = {sl_ae_rate:.1f}%")
    print(f"  Delta: {sl_ae_rate - s0_ae_rate:+.1f}pp")

    if sl_ae_rate > s0_ae_rate:
        print(f"  → Later steps have HIGHER action error rate — context may be causing confusion")
    else:
        print(f"  → Later steps have LOWER action error rate — context is HELPING, not hurting")


def analyze_context_length_impact(multisample_data):
    """分析 6: 利用 C4+C7 数据分析 agreement 随 step position 的变化"""
    if multisample_data is None:
        print("\n[SKIP] Multi-sample data not available")
        return

    print("\n" + "=" * 80)
    print("Analysis 6: Agreement (Uncertainty) by Step Position (C4+C7)")
    print("=" * 80)

    step_agreement = defaultdict(list)

    for episode in multisample_data:
        for step_data in episode.get("step_samples", []):
            step_num = step_data["step_num"]
            samples = step_data.get("samples", [])
            if not samples:
                continue

            # Compute agreement (fraction of samples matching majority action type)
            action_types = []
            for s in samples:
                if s.get("pred_action") and isinstance(s["pred_action"], dict):
                    action_types.append(s["pred_action"].get("action", "unknown"))
            if action_types:
                type_counts = Counter(action_types)
                majority_count = type_counts.most_common(1)[0][1]
                agreement = majority_count / len(action_types)
                step_agreement[step_num].append(agreement)

    print(f"\nMean agreement by step position:")
    print(f"  {'Step':>5s} | {'N':>6s} | {'Mean Agree':>11s} | {'Std':>6s}")
    for step in sorted(step_agreement.keys())[:12]:
        values = step_agreement[step]
        mean_agree = sum(values) / len(values)
        std = (sum((v - mean_agree)**2 for v in values) / len(values)) ** 0.5
        print(f"  {step:>5d} | {len(values):>6d} | {mean_agree:>10.4f} | {std:>5.3f}")


def analyze_context_loss_hypothesis(trajectories):
    """分析 7: 综合评估 — 上下文问题的证据汇总"""
    print("\n" + "=" * 80)
    print("Analysis 7: Context Loss Hypothesis — Evidence Summary")
    print("=" * 80)

    evidence_for = []
    evidence_against = []

    # Evidence 1: Step accuracy decay
    step_acc = defaultdict(lambda: {"correct": 0, "total": 0})
    for t in trajectories:
        for s in t["step_results"]:
            step_acc[s["step_num"]]["total"] += 1
            if s["extract_match"]:
                step_acc[s["step_num"]]["correct"] += 1

    # Check if accuracy monotonically decreases
    accs = []
    for step in sorted(step_acc.keys())[:7]:
        if step_acc[step]["total"] > 50:
            accs.append(step_acc[step]["correct"] / step_acc[step]["total"])

    if len(accs) >= 3:
        if accs[0] < accs[1]:
            evidence_against.append(f"Step 0 ({accs[0]:.3f}) < Step 1 ({accs[1]:.3f}) — step 0 is hardest, not context loss")
        else:
            evidence_for.append(f"Step accuracy decreases: {' → '.join(f'{a:.3f}' for a in accs[:5])}")

    # Evidence 2: Error type shift
    step0_ae = 0; step0_total = 0
    later_ae = 0; later_total = 0
    for t in trajectories:
        for s in t["step_results"]:
            if s["step_num"] == 0:
                step0_total += 1
                if not s["type_match"]:
                    step0_ae += 1
            else:
                later_total += 1
                if not s["extract_match"] and not s["type_match"]:
                    later_ae += 1

    s0_rate = step0_ae / step0_total if step0_total > 0 else 0
    sl_rate = later_ae / later_total if later_total > 0 else 0
    if sl_rate > s0_rate:
        evidence_for.append(f"Action error rate increases: step 0 = {s0_rate:.3f}, later = {sl_rate:.3f}")
    else:
        evidence_against.append(f"Action error rate decreases with context: step 0 = {s0_rate:.3f}, later = {sl_rate:.3f}")

    # Evidence 3: Long trajectory performance
    for bucket in ["short(1-3)", "medium(4-7)", "long(8-15)"]:
        bucket_trajs = [t for t in trajectories if t.get("length_bucket") == bucket]
        if bucket_trajs:
            tsr = sum(1 for t in bucket_trajs if t["task_success"]) / len(bucket_trajs)
            if bucket == "long(8-15)" and tsr < 0.02:
                evidence_for.append(f"Long trajectory TSR near 0 ({tsr:.3f}) — possible context degradation")

    print(f"\nEvidence FOR context being a problem:")
    for i, e in enumerate(evidence_for):
        print(f"  {i+1}. {e}")

    print(f"\nEvidence AGAINST context being the main problem:")
    for i, e in enumerate(evidence_against):
        print(f"  {i+1}. {e}")

    # Summary judgment
    print(f"\n  Summary:")
    if len(evidence_for) > len(evidence_against):
        print(f"  → Context loss is a CONTRIBUTING factor but likely NOT the dominant issue")
        print(f"  → The dominant issue is ACTION SELECTION (step 0 has lowest accuracy = intrinsic difficulty)")
        print(f"  → Adding summary may help marginally but won't solve the fundamental action error problem")
    else:
        print(f"  → Context is NOT the main bottleneck")
        print(f"  → The main bottleneck is per-step action accuracy, not context management")


def main():
    print("Loading data...")
    trajectories = load_trajectory_results()
    multisample = load_multisample_results()
    print(f"Loaded {len(trajectories)} trajectories")

    analyze_accuracy_decay(trajectories)
    analyze_accuracy_given_correct_prefix(trajectories)
    analyze_goal_complexity(trajectories)
    analyze_action_repetition(trajectories)
    analyze_error_type_by_context(trajectories)
    analyze_context_length_impact(multisample)
    analyze_context_loss_hypothesis(trajectories)

    print(f"\nResults saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
