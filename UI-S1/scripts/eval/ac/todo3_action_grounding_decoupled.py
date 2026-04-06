#!/usr/bin/env python3
"""
TODO 3: 在 action+grounding agent system 下查看做的更好的和依旧没做好的

分析思路:
1. 将 error 严格分解为 action error vs grounding error
2. 模拟 oracle fix action / oracle fix grounding 的 TSR ceiling
3. 分析 action agent 做对但 grounding agent 做错的案例
4. 分析 grounding agent 做对但 action agent 做错的案例
5. 联合 agent (action+grounding) 的理论 ceiling
6. 利用 C4+C7 multi-sample 数据分析 action/grounding 的独立改善空间
"""

import json
import os
from collections import Counter, defaultdict

EVAL_A_DIR = "outputs/eval_a_ac/Qwen2.5-VL-7B"
C4C7_DIR = "outputs/eval_c4c7_ac/Qwen2.5-VL-7B"
OUTPUT_DIR = "outputs/todo3_action_grounding"

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


def simulate_oracle_tsr(trajectories, fix_action=False, fix_grounding=False):
    """模拟 oracle fix action/grounding 的 TSR"""
    success = 0
    for t in trajectories:
        all_correct = True
        for s in t["step_results"]:
            step_correct = s["extract_match"]
            if not step_correct:
                is_action_error = not s["type_match"]
                is_grounding_error = s["type_match"] and not s["extract_match"]

                if fix_action and is_action_error:
                    step_correct = True
                if fix_grounding and is_grounding_error:
                    step_correct = True

            if not step_correct:
                all_correct = False
                break

        if all_correct:
            # Need to also check if all GT steps would be executed
            if len(t["step_results"]) == t["num_steps"]:
                success += 1
            elif all_correct and fix_action and fix_grounding:
                success += 1  # If fixing both, assume all steps pass
            elif all_correct:
                # All evaluated steps correct, check if trajectory would continue
                success += 1

    return success / len(trajectories)


def analyze_action_grounding_decomposition(trajectories):
    """分析 1: Action vs Grounding error 严格分解"""
    print("=" * 80)
    print("Analysis 1: Action vs Grounding Error Decomposition")
    print("=" * 80)

    total_steps = 0
    correct = 0
    action_errors = 0
    grounding_errors = 0

    per_step = defaultdict(lambda: {"total": 0, "correct": 0, "action_err": 0, "grounding_err": 0})
    per_action = defaultdict(lambda: {"total": 0, "correct": 0, "action_err": 0, "grounding_err": 0})

    for t in trajectories:
        for s in t["step_results"]:
            total_steps += 1
            gt_type = s["gt_action_type"]
            step_num = s["step_num"]

            if s["extract_match"]:
                correct += 1
                per_step[step_num]["correct"] += 1
                per_action[gt_type]["correct"] += 1
            elif not s["type_match"]:
                action_errors += 1
                per_step[step_num]["action_err"] += 1
                per_action[gt_type]["action_err"] += 1
            else:
                grounding_errors += 1
                per_step[step_num]["grounding_err"] += 1
                per_action[gt_type]["grounding_err"] += 1

            per_step[step_num]["total"] += 1
            per_action[gt_type]["total"] += 1

    print(f"\nOverall (N={total_steps} steps):")
    print(f"  Correct:         {correct:5d} ({correct/total_steps*100:.1f}%)")
    print(f"  Action error:    {action_errors:5d} ({action_errors/total_steps*100:.1f}%)")
    print(f"  Grounding error: {grounding_errors:5d} ({grounding_errors/total_steps*100:.1f}%)")
    print(f"  Action/(Action+Grounding): {action_errors/(action_errors+grounding_errors)*100:.1f}%")

    print(f"\nPer step position:")
    print(f"  {'Step':>5s} | {'Total':>6s} | {'Correct':>10s} | {'Action Err':>12s} | {'Grd Err':>10s}")
    for step in sorted(per_step.keys())[:8]:
        st = per_step[step]
        print(f"  {step:>5d} | {st['total']:>6d} | {st['correct']:4d} ({st['correct']/st['total']*100:4.1f}%) | "
              f"{st['action_err']:4d} ({st['action_err']/st['total']*100:4.1f}%) | "
              f"{st['grounding_err']:4d} ({st['grounding_err']/st['total']*100:4.1f}%)")

    print(f"\nPer GT action type:")
    print(f"  {'Type':>15s} | {'Total':>6s} | {'Correct':>10s} | {'Action Err':>12s} | {'Grd Err':>10s}")
    for at in sorted(per_action.keys(), key=lambda x: -per_action[x]["total"]):
        st = per_action[at]
        print(f"  {at:>15s} | {st['total']:>6d} | {st['correct']:4d} ({st['correct']/st['total']*100:4.1f}%) | "
              f"{st['action_err']:4d} ({st['action_err']/st['total']*100:4.1f}%) | "
              f"{st['grounding_err']:4d} ({st['grounding_err']/st['total']*100:4.1f}%)")

    return per_step, per_action


def analyze_oracle_fix_ceilings(trajectories):
    """分析 2: Oracle fix action / grounding 的 TSR ceiling"""
    print("\n" + "=" * 80)
    print("Analysis 2: Oracle Fix Ceilings (Stop-on-Error TSR)")
    print("=" * 80)

    baseline_tsr = simulate_oracle_tsr(trajectories)
    fix_action_tsr = simulate_oracle_tsr(trajectories, fix_action=True)
    fix_grounding_tsr = simulate_oracle_tsr(trajectories, fix_grounding=True)
    fix_both_tsr = simulate_oracle_tsr(trajectories, fix_action=True, fix_grounding=True)

    print(f"\n  {'Method':>25s} | {'TSR':>7s} | {'Delta':>8s}")
    print(f"  {'-'*25} | {'-'*7} | {'-'*8}")
    print(f"  {'Baseline':>25s} | {baseline_tsr*100:>6.2f}% | {'—':>8s}")
    print(f"  {'Oracle fix Action':>25s} | {fix_action_tsr*100:>6.2f}% | +{(fix_action_tsr-baseline_tsr)*100:.2f}pp")
    print(f"  {'Oracle fix Grounding':>25s} | {fix_grounding_tsr*100:>6.2f}% | +{(fix_grounding_tsr-baseline_tsr)*100:.2f}pp")
    print(f"  {'Oracle fix Both':>25s} | {fix_both_tsr*100:>6.2f}% | +{(fix_both_tsr-baseline_tsr)*100:.2f}pp")

    if fix_grounding_tsr > baseline_tsr:
        ratio = (fix_action_tsr - baseline_tsr) / (fix_grounding_tsr - baseline_tsr)
        print(f"\n  Action/Grounding ceiling ratio: {ratio:.2f}x")

    # Per length bucket
    print(f"\n  Per length bucket:")
    for bucket in ["short(1-3)", "medium(4-7)", "long(8-15)", "vlong(16+)"]:
        bucket_trajs = [t for t in trajectories if t.get("length_bucket") == bucket]
        if len(bucket_trajs) < 5:
            continue
        bl = simulate_oracle_tsr(bucket_trajs)
        fa = simulate_oracle_tsr(bucket_trajs, fix_action=True)
        fg = simulate_oracle_tsr(bucket_trajs, fix_grounding=True)
        fb = simulate_oracle_tsr(bucket_trajs, fix_action=True, fix_grounding=True)
        print(f"    {bucket:15s} (N={len(bucket_trajs):4d}): BL={bl*100:.1f}% | +Action={fa*100:.1f}% (+{(fa-bl)*100:.1f}pp) | +Grd={fg*100:.1f}% (+{(fg-bl)*100:.1f}pp) | +Both={fb*100:.1f}% (+{(fb-bl)*100:.1f}pp)")

    return {
        "baseline": baseline_tsr,
        "fix_action": fix_action_tsr,
        "fix_grounding": fix_grounding_tsr,
        "fix_both": fix_both_tsr,
    }


def analyze_multisample_action_grounding(multisample_data):
    """分析 3: 利用 C4+C7 multi-sample 数据分析 action/grounding 独立改善空间"""
    if multisample_data is None:
        print("\n[SKIP] Multi-sample data not available")
        return None

    print("\n" + "=" * 80)
    print("Analysis 3: Multi-Sample Action+Grounding Decoupled Analysis")
    print("=" * 80)

    total_steps = 0
    # Categories:
    # - greedy_correct: greedy 就对了
    # - action_fixable: greedy action type 错, 但 K 中存在正确的 action type + grounding
    # - grounding_fixable: greedy action type 对, grounding 错, 但 K 中存在正确 grounding
    # - both_fixable: 需要同时修 action + grounding
    # - unfixable: K 个 sample 都没有正确答案

    categories = Counter()
    per_action_categories = defaultdict(Counter)

    for episode in multisample_data:
        for step_data in episode.get("step_samples", []):
            total_steps += 1
            gt_type = step_data["gt_action_type"]
            samples = step_data.get("samples", [])
            if not samples:
                continue

            greedy = samples[0]  # first sample is greedy
            greedy_correct = greedy.get("extract_match", False)
            greedy_type_match = greedy.get("type_match", False)

            if greedy_correct:
                categories["greedy_correct"] += 1
                per_action_categories[gt_type]["greedy_correct"] += 1
                continue

            # Check if any sample is fully correct
            any_correct = any(s.get("extract_match", False) for s in samples)

            # Check if any sample has correct action type
            any_type_correct = any(s.get("type_match", False) for s in samples)

            # Check if any sample with correct type also has correct grounding
            any_type_and_grounding = any(
                s.get("type_match", False) and s.get("extract_match", False)
                for s in samples
            )

            if greedy_type_match and not greedy_correct:
                # Action type correct, grounding wrong
                if any_correct:
                    categories["grounding_fixable"] += 1
                    per_action_categories[gt_type]["grounding_fixable"] += 1
                else:
                    categories["grounding_unfixable"] += 1
                    per_action_categories[gt_type]["grounding_unfixable"] += 1
            elif not greedy_type_match:
                # Action type wrong
                if any_type_and_grounding:
                    categories["action_fixable_with_grounding"] += 1
                    per_action_categories[gt_type]["action_fixable_with_grounding"] += 1
                elif any_type_correct:
                    categories["action_fixable_grounding_still_wrong"] += 1
                    per_action_categories[gt_type]["action_fixable_grounding_still_wrong"] += 1
                else:
                    categories["action_unfixable"] += 1
                    per_action_categories[gt_type]["action_unfixable"] += 1

    print(f"\nStep-level decomposition (N={total_steps}):")
    for cat in ["greedy_correct", "action_fixable_with_grounding", "grounding_fixable",
                "action_fixable_grounding_still_wrong", "grounding_unfixable", "action_unfixable"]:
        c = categories.get(cat, 0)
        print(f"  {cat:>45s}: {c:5d} ({c/total_steps*100:.1f}%)")

    # Summarize action agent potential vs grounding agent potential
    action_agent_gain = categories.get("action_fixable_with_grounding", 0)
    grounding_agent_gain = categories.get("grounding_fixable", 0)
    both_needed = categories.get("action_fixable_grounding_still_wrong", 0)
    unfixable = categories.get("action_unfixable", 0) + categories.get("grounding_unfixable", 0)

    print(f"\n  Action agent alone could fix:     {action_agent_gain:5d} steps ({action_agent_gain/total_steps*100:.1f}%)")
    print(f"  Grounding agent alone could fix:  {grounding_agent_gain:5d} steps ({grounding_agent_gain/total_steps*100:.1f}%)")
    print(f"  Need both agents:                 {both_needed:5d} steps ({both_needed/total_steps*100:.1f}%)")
    print(f"  Unfixable (all K wrong):          {unfixable:5d} steps ({unfixable/total_steps*100:.1f}%)")

    # Per action type
    print(f"\n  Per GT action type:")
    print(f"  {'Type':>15s} | {'Correct':>8s} | {'Act Fix':>8s} | {'Grd Fix':>8s} | {'Both':>8s} | {'Unfix':>8s}")
    for at in sorted(per_action_categories.keys(), key=lambda x: -sum(per_action_categories[x].values())):
        cats = per_action_categories[at]
        total = sum(cats.values())
        if total < 10:
            continue
        print(f"  {at:>15s} | {cats.get('greedy_correct',0):4d} ({cats.get('greedy_correct',0)/total*100:4.1f}%) | "
              f"{cats.get('action_fixable_with_grounding',0):4d} ({cats.get('action_fixable_with_grounding',0)/total*100:4.1f}%) | "
              f"{cats.get('grounding_fixable',0):4d} ({cats.get('grounding_fixable',0)/total*100:4.1f}%) | "
              f"{cats.get('action_fixable_grounding_still_wrong',0):4d} ({cats.get('action_fixable_grounding_still_wrong',0)/total*100:4.1f}%) | "
              f"{cats.get('action_unfixable',0)+cats.get('grounding_unfixable',0):4d}")

    return categories


def analyze_what_improves_what_doesnt(trajectories):
    """分析 4: 在 action+grounding 系统下，什么做的更好，什么依旧没做好"""
    print("\n" + "=" * 80)
    print("Analysis 4: What Improves vs What Doesn't Under Action+Grounding System")
    print("=" * 80)

    # Group failed episodes by their first error type
    first_error_episodes = {"action_error": [], "grounding_error": []}

    for t in trajectories:
        if t["task_success"]:
            continue
        for s in t["step_results"]:
            if not s["extract_match"]:
                if not s["type_match"]:
                    first_error_episodes["action_error"].append(t)
                else:
                    first_error_episodes["grounding_error"].append(t)
                break

    print(f"\nFailed episodes by first error type:")
    for etype, episodes in first_error_episodes.items():
        print(f"  {etype}: {len(episodes)} episodes")

        # Analyze what action types cause this error type
        action_types = Counter()
        for t in episodes:
            for s in t["step_results"]:
                if not s["extract_match"]:
                    action_types[s["gt_action_type"]] += 1
                    break

        print(f"    GT action types at first error:")
        for at, c in action_types.most_common():
            print(f"      {at}: {c} ({c/len(episodes)*100:.1f}%)")

    # What IMPROVES with action oracle
    print(f"\n  Episodes that would succeed with oracle action fix:")
    action_fixable = []
    for t in trajectories:
        if t["task_success"]:
            continue
        # Simulate: fix all action errors
        all_pass = True
        for s in t["step_results"]:
            if not s["extract_match"] and s["type_match"]:
                # grounding error remains
                all_pass = False
                break
            # action errors are fixed, so we continue
        if all_pass:
            action_fixable.append(t)

    action_fix_by_bucket = Counter(t.get("length_bucket") for t in action_fixable)
    for bucket, c in action_fix_by_bucket.most_common():
        total_bucket_fails = sum(1 for t in trajectories if not t["task_success"] and t.get("length_bucket") == bucket)
        print(f"    {bucket}: {c}/{total_bucket_fails} fails fixed ({c/total_bucket_fails*100:.1f}%)" if total_bucket_fails > 0 else "")

    # What STILL FAILS even with grounding oracle
    print(f"\n  Episodes that STILL FAIL even with oracle grounding fix:")
    still_fail_with_grd_fix = []
    for t in trajectories:
        if t["task_success"]:
            continue
        has_action_error = any(
            not s["extract_match"] and not s["type_match"]
            for s in t["step_results"]
        )
        if has_action_error:
            still_fail_with_grd_fix.append(t)

    print(f"    {len(still_fail_with_grd_fix)} episodes ({len(still_fail_with_grd_fix)/(len(trajectories)-sum(1 for t in trajectories if t['task_success']))*100:.1f}% of failures)")


def main():
    print("Loading data...")
    trajectories = load_trajectory_results()
    multisample = load_multisample_results()
    print(f"Loaded {len(trajectories)} trajectories")
    if multisample:
        print(f"Loaded {len(multisample)} multi-sample episodes")

    per_step, per_action = analyze_action_grounding_decomposition(trajectories)
    ceilings = analyze_oracle_fix_ceilings(trajectories)
    ms_categories = analyze_multisample_action_grounding(multisample)
    analyze_what_improves_what_doesnt(trajectories)

    results = {
        "oracle_ceilings": ceilings,
        "multisample_categories": dict(ms_categories) if ms_categories else None,
    }
    with open(os.path.join(OUTPUT_DIR, "action_grounding_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {OUTPUT_DIR}/action_grounding_results.json")


if __name__ == "__main__":
    main()
