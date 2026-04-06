#!/usr/bin/env python3
"""
TODO 5: 对 failure case 进行 evaluation，思考 step 0, 1 对后续的影响

分析:
1. Step 0 的特殊性: 准确率、error type、对 TSR 的影响
2. Step 0+1 联合对 TSR 的影响
3. 条件概率: P(task_success | step 0 correct), P(task_success | step 0,1 correct)
4. Step 0 error 的 "致命性": 哪些 step 0 error 导致后续完全无法恢复
5. Oracle fix step 0 / step 0+1 的 TSR ceiling
6. Step 0 error 与 goal/app 类型的关系
7. 对比: 早期 error (step 0-1) vs 中期 error (step 2-4) 的影响差异
"""

import json
import os
from collections import Counter, defaultdict

EVAL_A_DIR = "outputs/eval_a_ac/Qwen2.5-VL-7B"
C4C7_DIR = "outputs/eval_c4c7_ac/Qwen2.5-VL-7B"
OUTPUT_DIR = "outputs/todo5_step01_impact"

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


def analyze_step0_specialness(trajectories):
    """分析 1: Step 0 的特殊性"""
    print("=" * 80)
    print("Analysis 1: Step 0 Specialness")
    print("=" * 80)

    step0_stats = {"correct": 0, "action_error": 0, "grounding_error": 0}
    step1_stats = {"correct": 0, "action_error": 0, "grounding_error": 0}
    other_stats = {"correct": 0, "action_error": 0, "grounding_error": 0}

    for t in trajectories:
        for s in t["step_results"]:
            if s["step_num"] == 0:
                target = step0_stats
            elif s["step_num"] == 1:
                target = step1_stats
            else:
                target = other_stats

            if s["extract_match"]:
                target["correct"] += 1
            elif not s["type_match"]:
                target["action_error"] += 1
            else:
                target["grounding_error"] += 1

    for name, stats in [("Step 0", step0_stats), ("Step 1", step1_stats), ("Steps 2+", other_stats)]:
        total = sum(stats.values())
        if total == 0:
            continue
        print(f"\n  {name} (N={total}):")
        for key in ["correct", "action_error", "grounding_error"]:
            print(f"    {key:>20s}: {stats[key]:4d} ({stats[key]/total*100:.1f}%)")


def analyze_step01_tsr_impact(trajectories):
    """分析 2: Step 0 和 Step 0+1 的 TSR 影响"""
    print("\n" + "=" * 80)
    print("Analysis 2: Step 0/1 Correctness → TSR Impact")
    print("=" * 80)

    # Conditional TSR
    s0_correct = [t for t in trajectories if t["step_results"] and t["step_results"][0]["extract_match"]]
    s0_wrong = [t for t in trajectories if t["step_results"] and not t["step_results"][0]["extract_match"]]

    s0_correct_tsr = sum(1 for t in s0_correct if t["task_success"]) / len(s0_correct) if s0_correct else 0
    s0_wrong_tsr = sum(1 for t in s0_wrong if t["task_success"]) / len(s0_wrong) if s0_wrong else 0

    print(f"\n  P(task_success | step 0 correct) = {s0_correct_tsr*100:.2f}% (N={len(s0_correct)})")
    print(f"  P(task_success | step 0 wrong)   = {s0_wrong_tsr*100:.2f}% (N={len(s0_wrong)})")
    print(f"  Ratio: {s0_correct_tsr/s0_wrong_tsr:.1f}x" if s0_wrong_tsr > 0 else "  Ratio: ∞")

    # Step 0 + Step 1 both correct
    s01_correct = [
        t for t in trajectories
        if len(t["step_results"]) >= 2
        and t["step_results"][0]["extract_match"]
        and t["step_results"][1]["extract_match"]
    ]
    s01_any_wrong = [
        t for t in trajectories
        if len(t["step_results"]) >= 2
        and (not t["step_results"][0]["extract_match"] or not t["step_results"][1]["extract_match"])
    ]

    if s01_correct:
        s01_tsr = sum(1 for t in s01_correct if t["task_success"]) / len(s01_correct)
        print(f"\n  P(task_success | step 0,1 both correct) = {s01_tsr*100:.2f}% (N={len(s01_correct)})")
    if s01_any_wrong:
        s01w_tsr = sum(1 for t in s01_any_wrong if t["task_success"]) / len(s01_any_wrong)
        print(f"  P(task_success | step 0 or 1 wrong)      = {s01w_tsr*100:.2f}% (N={len(s01_any_wrong)})")


def analyze_oracle_fix_early_steps(trajectories):
    """分析 3: Oracle fix step 0 / step 0+1 的 TSR ceiling"""
    print("\n" + "=" * 80)
    print("Analysis 3: Oracle Fix Early Steps → TSR Ceiling")
    print("=" * 80)

    def simulate_fix_steps(trajs, fix_steps_set):
        """模拟 oracle 修复指定 steps 的 TSR"""
        success = 0
        for t in trajs:
            all_correct = True
            for s in t["step_results"]:
                step_correct = s["extract_match"] or (s["step_num"] in fix_steps_set)
                if not step_correct:
                    all_correct = False
                    break

            if all_correct:
                # Check if we've gone through all steps
                if len(t["step_results"]) == t["num_steps"] or all_correct:
                    success += 1

        return success / len(trajs) if trajs else 0

    baseline_tsr = sum(1 for t in trajectories if t["task_success"]) / len(trajectories)
    fix0_tsr = simulate_fix_steps(trajectories, {0})
    fix01_tsr = simulate_fix_steps(trajectories, {0, 1})
    fix012_tsr = simulate_fix_steps(trajectories, {0, 1, 2})

    print(f"\n  {'Method':>30s} | {'TSR':>8s} | {'Delta':>8s}")
    print(f"  {'-'*30} | {'-'*8} | {'-'*8}")
    print(f"  {'Baseline':>30s} | {baseline_tsr*100:>7.2f}% | {'—':>8s}")
    print(f"  {'Oracle fix Step 0':>30s} | {fix0_tsr*100:>7.2f}% | +{(fix0_tsr-baseline_tsr)*100:.2f}pp")
    print(f"  {'Oracle fix Steps 0+1':>30s} | {fix01_tsr*100:>7.2f}% | +{(fix01_tsr-baseline_tsr)*100:.2f}pp")
    print(f"  {'Oracle fix Steps 0+1+2':>30s} | {fix012_tsr*100:>7.2f}% | +{(fix012_tsr-baseline_tsr)*100:.2f}pp")

    # By length bucket
    print(f"\n  Per length bucket (oracle fix step 0):")
    for bucket in ["short(1-3)", "medium(4-7)", "long(8-15)", "vlong(16+)"]:
        bucket_trajs = [t for t in trajectories if t.get("length_bucket") == bucket]
        if len(bucket_trajs) < 5:
            continue
        bl = sum(1 for t in bucket_trajs if t["task_success"]) / len(bucket_trajs)
        f0 = simulate_fix_steps(bucket_trajs, {0})
        f01 = simulate_fix_steps(bucket_trajs, {0, 1})
        print(f"    {bucket:15s}: BL={bl*100:.1f}% → fix0={f0*100:.1f}% (+{(f0-bl)*100:.1f}pp) → fix01={f01*100:.1f}% (+{(f01-bl)*100:.1f}pp)")


def analyze_step0_error_fatality(trajectories):
    """分析 4: Step 0 error 的"致命性"分析"""
    print("\n" + "=" * 80)
    print("Analysis 4: Step 0 Error Fatality Analysis")
    print("=" * 80)

    # In stop-on-error, step 0 error = immediate failure
    # But we can look at WHAT kind of step 0 errors are most common

    step0_errors = []
    for t in trajectories:
        if t["step_results"] and not t["step_results"][0]["extract_match"]:
            s = t["step_results"][0]
            step0_errors.append({
                "gt_type": s["gt_action_type"],
                "pred_type": s.get("pred_action", {}).get("action", "unknown") if isinstance(s.get("pred_action"), dict) else "unknown",
                "is_action_error": not s["type_match"],
                "num_steps": t["num_steps"],
                "bucket": t.get("length_bucket", "unknown"),
                "goal": t["goal"],
            })

    print(f"\nStep 0 errors: {len(step0_errors)} / {len(trajectories)} = {len(step0_errors)/len(trajectories)*100:.1f}%")

    # By GT action type
    gt_types = Counter(e["gt_type"] for e in step0_errors)
    print(f"\n  Step 0 errors by GT action type:")
    for at, c in gt_types.most_common():
        ae_count = sum(1 for e in step0_errors if e["gt_type"] == at and e["is_action_error"])
        ge_count = c - ae_count
        print(f"    {at:15s}: {c:4d} ({c/len(step0_errors)*100:.1f}%) — action_err: {ae_count}, grd_err: {ge_count}")

    # Common confusion patterns at step 0
    confusion = Counter((e["gt_type"], e["pred_type"]) for e in step0_errors)
    print(f"\n  Top 10 step 0 confusion patterns (GT → Pred):")
    for (gt, pred), c in confusion.most_common(10):
        print(f"    {gt:15s} → {pred:15s}: {c:4d} ({c/len(step0_errors)*100:.1f}%)")

    # Impact analysis: how many total steps are "wasted" by step 0 errors
    total_wasted = sum(e["num_steps"] - 1 for e in step0_errors)  # -1 because step 0 was "executed"
    print(f"\n  Total steps 'blocked' by step 0 errors: {total_wasted}")
    print(f"  Average remaining steps per step 0 failure: {total_wasted/len(step0_errors):.1f}")


def analyze_early_vs_late_error_impact(trajectories):
    """分析 5: 早期 error vs 中期 error 的影响差异"""
    print("\n" + "=" * 80)
    print("Analysis 5: Early Error (Step 0-1) vs Mid Error (Step 2-4) Impact")
    print("=" * 80)

    # For failed trajectories, categorize by where the first error occurs
    error_position_stats = defaultdict(lambda: {
        "count": 0,
        "avg_traj_len": 0,
        "total_steps_lost": 0,
        "by_error_type": Counter(),
    })

    for t in trajectories:
        if t["task_success"]:
            continue
        for s in t["step_results"]:
            if not s["extract_match"]:
                pos = s["step_num"]
                if pos <= 1:
                    group = "early(0-1)"
                elif pos <= 4:
                    group = "mid(2-4)"
                else:
                    group = "late(5+)"

                error_position_stats[group]["count"] += 1
                error_position_stats[group]["total_steps_lost"] += (t["num_steps"] - s["step_num"] - 1)
                error_position_stats[group]["avg_traj_len"] += t["num_steps"]
                etype = "action" if not s["type_match"] else "grounding"
                error_position_stats[group]["by_error_type"][etype] += 1
                break

    print(f"\n  {'Group':>12s} | {'Count':>6s} | {'Avg Traj Len':>13s} | {'Steps Lost':>11s} | {'Action%':>8s} | {'Grd%':>6s}")
    print(f"  {'-'*12} | {'-'*6} | {'-'*13} | {'-'*11} | {'-'*8} | {'-'*6}")
    for group in ["early(0-1)", "mid(2-4)", "late(5+)"]:
        st = error_position_stats[group]
        if st["count"] == 0:
            continue
        avg_len = st["avg_traj_len"] / st["count"]
        avg_lost = st["total_steps_lost"] / st["count"]
        ae_pct = st["by_error_type"]["action"] / st["count"] * 100
        ge_pct = st["by_error_type"]["grounding"] / st["count"] * 100
        print(f"  {group:>12s} | {st['count']:>6d} | {avg_len:>12.1f} | {avg_lost:>10.1f} | {ae_pct:>7.1f}% | {ge_pct:>5.1f}%")

    # The "wasted potential" metric
    total_failures = sum(st["count"] for st in error_position_stats.values())
    early_failures = error_position_stats["early(0-1)"]["count"]
    print(f"\n  {early_failures}/{total_failures} ({early_failures/total_failures*100:.1f}%) of failures happen at steps 0-1")
    print(f"  → Fixing early steps would eliminate the MAJORITY of failures")


def analyze_step0_multisample(multisample_data):
    """分析 6: C4+C7 多采样数据中 step 0 的 oracle headroom"""
    if multisample_data is None:
        print("\n[SKIP] Multi-sample data not available")
        return

    print("\n" + "=" * 80)
    print("Analysis 6: Step 0 Multi-Sample Oracle Headroom")
    print("=" * 80)

    step_oracle = defaultdict(lambda: {"greedy_correct": 0, "oracle_correct": 0, "total": 0})

    for episode in multisample_data:
        for step_data in episode.get("step_samples", []):
            step_num = step_data["step_num"]
            samples = step_data.get("samples", [])
            if not samples:
                continue

            greedy_correct = samples[0].get("extract_match", False)
            oracle_correct = any(s.get("extract_match", False) for s in samples)

            step_oracle[step_num]["total"] += 1
            if greedy_correct:
                step_oracle[step_num]["greedy_correct"] += 1
            if oracle_correct:
                step_oracle[step_num]["oracle_correct"] += 1

    print(f"\n  {'Step':>5s} | {'N':>6s} | {'Greedy':>8s} | {'Oracle':>8s} | {'Gap':>7s} | {'OG Rate':>8s}")
    for step in sorted(step_oracle.keys())[:8]:
        st = step_oracle[step]
        greedy_acc = st["greedy_correct"] / st["total"] * 100
        oracle_acc = st["oracle_correct"] / st["total"] * 100
        gap = oracle_acc - greedy_acc
        og_rate = (st["oracle_correct"] - st["greedy_correct"]) / (st["total"] - st["greedy_correct"]) * 100 if st["total"] > st["greedy_correct"] else 0
        print(f"  {step:>5d} | {st['total']:>6d} | {greedy_acc:>7.1f}% | {oracle_acc:>7.1f}% | {gap:>6.1f}pp | {og_rate:>7.1f}%")


def analyze_step0_step1_cascading(trajectories):
    """分析 7: Step 0 error type 如何影响 step 1 的 error type"""
    print("\n" + "=" * 80)
    print("Analysis 7: Step 0 → Step 1 Error Cascading")
    print("=" * 80)

    # Note: In AR stop-on-error, if step 0 fails, step 1 is never evaluated
    # So this analysis is about step 0 correctness → step 1 difficulty

    s0_correct_s1 = {"correct": 0, "action_error": 0, "grounding_error": 0}
    s0_wrong_but_continuing = {"correct": 0, "action_error": 0, "grounding_error": 0}

    for t in trajectories:
        if len(t["step_results"]) < 2:
            continue

        s0 = t["step_results"][0]
        s1 = t["step_results"][1]

        # In AR, s1 only exists if s0 was correct
        if s0["extract_match"]:
            if s1["extract_match"]:
                s0_correct_s1["correct"] += 1
            elif not s1["type_match"]:
                s0_correct_s1["action_error"] += 1
            else:
                s0_correct_s1["grounding_error"] += 1

    total = sum(s0_correct_s1.values())
    print(f"\n  Given step 0 correct (N={total}):")
    print(f"    Step 1 distribution:")
    for key in ["correct", "action_error", "grounding_error"]:
        c = s0_correct_s1[key]
        print(f"      {key:>20s}: {c:4d} ({c/total*100:.1f}%)")

    # Compare with overall step 1 accuracy (from C4+C7 independent eval, or just AR)
    print(f"\n  Key insight:")
    s1_acc = s0_correct_s1["correct"] / total * 100 if total > 0 else 0
    print(f"    P(step 1 correct | step 0 correct) = {s1_acc:.1f}%")
    print(f"    This is the CONDITIONAL accuracy that matters for TSR")


def main():
    print("Loading data...")
    trajectories = load_trajectory_results()
    multisample = load_multisample_results()
    print(f"Loaded {len(trajectories)} trajectories")

    analyze_step0_specialness(trajectories)
    analyze_step01_tsr_impact(trajectories)
    analyze_oracle_fix_early_steps(trajectories)
    analyze_step0_error_fatality(trajectories)
    analyze_early_vs_late_error_impact(trajectories)
    analyze_step0_multisample(multisample)
    analyze_step0_step1_cascading(trajectories)

    print(f"\nResults saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
