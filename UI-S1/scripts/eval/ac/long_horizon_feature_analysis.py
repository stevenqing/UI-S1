#!/usr/bin/env python3
"""
Long-Horizon Reasoning Feature Analysis

核心问题: 什么 feature 影响了 agent 在 long-horizon task 上的表现？

不是简单的 error type 统计，而是分析:
1. Task-level reasoning features: 哪些任务特征预测 long-horizon 成败
2. Action diversity & planning complexity: 任务的 action 多样性如何影响成功率
3. Reasoning chain depth: 需要多深的推理才能做出正确 action
4. Compounding sensitivity: 哪些 feature 导致 error 快速 compound
5. Horizon-aware feature importance: 用 logistic regression 量化 feature 重要性
6. State transition complexity: 状态转移的复杂度如何影响 reasoning
7. "Reasoning bottleneck" identification: 在 trajectory 中哪里出现了 reasoning failure

Output: 什么 feature 最能预测 long-horizon task 成功，以及如何利用这些 feature。
"""

import json
import os
import re
import math
from collections import Counter, defaultdict

EVAL_A_DIR = "outputs/eval_a_ac/Qwen2.5-VL-7B"
C4C7_DIR = "outputs/eval_c4c7_ac/Qwen2.5-VL-7B"
OUTPUT_DIR = "outputs/long_horizon_feature_analysis"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_data():
    trajs = []
    with open(os.path.join(EVAL_A_DIR, "trajectory_results.jsonl")) as f:
        for line in f:
            trajs.append(json.loads(line.strip()))

    ms_data = {}
    ms_path = os.path.join(C4C7_DIR, "multisample_results.jsonl")
    if os.path.exists(ms_path):
        with open(ms_path) as f:
            for line in f:
                ep = json.loads(line.strip())
                ms_data[ep["episode_id"]] = ep
    return trajs, ms_data


def extract_task_features(traj, ms_episode=None):
    """
    从单个 trajectory 提取 task-level reasoning features
    """
    features = {}
    goal = traj["goal"]
    steps = traj["step_results"]
    num_steps = traj["num_steps"]

    # === Goal complexity features ===
    features["goal_word_count"] = len(goal.split())
    features["goal_char_count"] = len(goal)
    # Number of potential sub-goals (heuristic: count connectives)
    connectives = len(re.findall(r'\band\b|\bthen\b|\bafter\b|\bnext\b|\balso\b', goal.lower()))
    features["goal_subgoal_count"] = connectives + 1
    # Goal specificity: does goal mention specific names/numbers?
    features["goal_has_specific_name"] = int(bool(re.search(r'[A-Z][a-z]+(?:\s[A-Z][a-z]+)+', goal)))
    features["goal_has_number"] = int(bool(re.search(r'\d+', goal)))

    # === Trajectory structure features ===
    features["trajectory_length"] = num_steps

    # GT action type sequence
    gt_types = [s["gt_action_type"] for s in steps]
    all_gt_types = []
    if ms_episode:
        all_gt_types = [ss["gt_action_type"] for ss in ms_episode.get("step_samples", [])]
    else:
        all_gt_types = gt_types

    features["unique_action_types"] = len(set(all_gt_types))
    features["action_type_entropy"] = _entropy(all_gt_types)

    # Action type transitions (how many times action type changes)
    if len(all_gt_types) >= 2:
        transitions = sum(1 for i in range(1, len(all_gt_types)) if all_gt_types[i] != all_gt_types[i-1])
        features["action_type_transitions"] = transitions
        features["transition_rate"] = transitions / (len(all_gt_types) - 1)
    else:
        features["action_type_transitions"] = 0
        features["transition_rate"] = 0

    # Presence of specific challenging action types
    features["has_open"] = int("open" in all_gt_types)
    features["has_wait"] = int("wait" in all_gt_types)
    features["has_swipe"] = int("swipe" in all_gt_types)
    features["has_type"] = int("type" in all_gt_types)
    features["has_system_button"] = int("system_button" in all_gt_types)

    # Fraction of each action type
    type_counts = Counter(all_gt_types)
    total = len(all_gt_types)
    for at in ["click", "open", "swipe", "type", "system_button", "wait"]:
        features[f"frac_{at}"] = type_counts.get(at, 0) / total if total > 0 else 0

    # === Planning complexity features ===
    # "Planning depth": how many different phases does the trajectory go through?
    # Phase = consecutive same action type
    phases = 1
    for i in range(1, len(all_gt_types)):
        if all_gt_types[i] != all_gt_types[i-1]:
            phases += 1
    features["num_phases"] = phases
    features["avg_phase_length"] = len(all_gt_types) / phases if phases > 0 else 0

    # Does task require app navigation? (starts with open or system_button)
    features["requires_app_nav"] = int(all_gt_types[0] in ("open", "system_button") if all_gt_types else False)

    # === Multi-sample uncertainty features (from C4+C7) ===
    if ms_episode:
        step_samples = ms_episode.get("step_samples", [])
        agreements = []
        entropies = []
        oracle_gains = []

        for ss in step_samples:
            samples = ss.get("samples", [])
            if not samples:
                continue

            # Agreement
            pred_types = [s["pred_action"].get("action", "?")
                         for s in samples
                         if s.get("pred_action") and isinstance(s["pred_action"], dict)]
            if pred_types:
                type_counts_s = Counter(pred_types)
                agreement = type_counts_s.most_common(1)[0][1] / len(pred_types)
                agreements.append(agreement)
                entropies.append(_entropy(pred_types))

            # Oracle gain
            greedy_correct = samples[0].get("extract_match", False) if samples else False
            any_correct = any(s.get("extract_match", False) for s in samples)
            oracle_gains.append(int(any_correct and not greedy_correct))

        features["mean_agreement"] = sum(agreements) / len(agreements) if agreements else 0
        features["min_agreement"] = min(agreements) if agreements else 0
        features["agreement_std"] = _std(agreements) if len(agreements) > 1 else 0
        features["mean_pred_entropy"] = sum(entropies) / len(entropies) if entropies else 0
        features["max_pred_entropy"] = max(entropies) if entropies else 0
        features["oracle_gain_rate"] = sum(oracle_gains) / len(oracle_gains) if oracle_gains else 0

        # Agreement trajectory shape: does uncertainty increase?
        if len(agreements) >= 3:
            first_half = agreements[:len(agreements)//2]
            second_half = agreements[len(agreements)//2:]
            features["agreement_trend"] = (
                (sum(second_half)/len(second_half)) - (sum(first_half)/len(first_half))
            )
        else:
            features["agreement_trend"] = 0

        # Bottleneck: step with lowest agreement
        if agreements:
            features["bottleneck_agreement"] = min(agreements)
            features["bottleneck_position"] = agreements.index(min(agreements)) / len(agreements)
        else:
            features["bottleneck_agreement"] = 0
            features["bottleneck_position"] = 0
    else:
        for f in ["mean_agreement", "min_agreement", "agreement_std",
                   "mean_pred_entropy", "max_pred_entropy", "oracle_gain_rate",
                   "agreement_trend", "bottleneck_agreement", "bottleneck_position"]:
            features[f] = 0

    # === Outcome ===
    features["task_success"] = int(traj["task_success"])
    features["progress"] = traj["final_step_id"] / traj["num_steps"] if traj["num_steps"] > 0 else 0

    return features


def _entropy(items):
    """Compute entropy of a list of items"""
    if not items:
        return 0
    counts = Counter(items)
    total = len(items)
    return -sum((c/total) * math.log2(c/total) for c in counts.values() if c > 0)


def _std(values):
    if len(values) < 2:
        return 0
    mean = sum(values) / len(values)
    return (sum((v - mean)**2 for v in values) / (len(values) - 1)) ** 0.5


def analyze_feature_importance(features_list, target="task_success"):
    """用 logistic regression 量化 feature 对 long-horizon 成功的预测力"""
    print("=" * 80)
    print(f"Feature Importance for Long-Horizon Task Success")
    print("=" * 80)

    # Manual logistic regression (no sklearn dependency)
    # Use univariate analysis: for each feature, compute correlation with outcome
    feature_names = [k for k in features_list[0].keys()
                     if k not in ("task_success", "progress", "episode_id")]

    outcomes = [f[target] for f in features_list]
    mean_outcome = sum(outcomes) / len(outcomes)

    results = []
    for fname in feature_names:
        values = [f[fname] for f in features_list]
        mean_val = sum(values) / len(values)
        std_val = _std(values) if len(values) > 1 else 1

        if std_val < 1e-10:
            continue

        # Point-biserial correlation
        success_vals = [v for v, o in zip(values, outcomes) if o == 1]
        failure_vals = [v for v, o in zip(values, outcomes) if o == 0]

        if not success_vals or not failure_vals:
            continue

        mean_success = sum(success_vals) / len(success_vals)
        mean_failure = sum(failure_vals) / len(failure_vals)
        effect = mean_success - mean_failure

        # Normalized effect (Cohen's d)
        pooled_std = (((_std(success_vals) if len(success_vals) > 1 else 0)**2 * (len(success_vals)-1) +
                       (_std(failure_vals) if len(failure_vals) > 1 else 0)**2 * (len(failure_vals)-1)) /
                      max(len(success_vals) + len(failure_vals) - 2, 1)) ** 0.5
        cohen_d = effect / pooled_std if pooled_std > 1e-10 else 0

        results.append({
            "feature": fname,
            "mean_success": mean_success,
            "mean_failure": mean_failure,
            "effect": effect,
            "cohen_d": cohen_d,
            "abs_d": abs(cohen_d),
        })

    results.sort(key=lambda x: -x["abs_d"])

    print(f"\nTop features predicting task success (by |Cohen's d|):")
    print(f"  {'Feature':>30s} | {'Success':>10s} | {'Failure':>10s} | {'Effect':>10s} | {'Cohen d':>10s}")
    print(f"  {'-'*30} | {'-'*10} | {'-'*10} | {'-'*10} | {'-'*10}")
    for r in results[:20]:
        print(f"  {r['feature']:>30s} | {r['mean_success']:>10.4f} | {r['mean_failure']:>10.4f} | "
              f"{r['effect']:>+10.4f} | {r['cohen_d']:>+10.4f}")

    return results


def analyze_horizon_scaling(features_list):
    """分析: feature 的预测力如何随 trajectory length 变化"""
    print("\n" + "=" * 80)
    print("Horizon Scaling: How Features Matter at Different Lengths")
    print("=" * 80)

    # Split by trajectory length
    length_bins = [
        ("short(1-3)", lambda f: f["trajectory_length"] <= 3),
        ("medium(4-7)", lambda f: 4 <= f["trajectory_length"] <= 7),
        ("long(8+)", lambda f: f["trajectory_length"] >= 8),
    ]

    key_features = [
        "unique_action_types", "action_type_entropy", "transition_rate",
        "has_open", "has_wait", "requires_app_nav",
        "mean_agreement", "min_agreement", "oracle_gain_rate",
        "num_phases", "goal_subgoal_count",
    ]

    for bin_name, bin_fn in length_bins:
        subset = [f for f in features_list if bin_fn(f)]
        if len(subset) < 20:
            continue

        n_success = sum(1 for f in subset if f["task_success"])
        n_total = len(subset)
        tsr = n_success / n_total

        print(f"\n  {bin_name} (N={n_total}, TSR={tsr:.1%}):")

        for fname in key_features:
            success_vals = [f[fname] for f in subset if f["task_success"]]
            failure_vals = [f[fname] for f in subset if not f["task_success"]]

            if not success_vals or not failure_vals:
                continue

            ms = sum(success_vals) / len(success_vals)
            mf = sum(failure_vals) / len(failure_vals)
            effect = ms - mf
            if abs(effect) > 0.01:
                direction = "↑" if effect > 0 else "↓"
                print(f"    {fname:>25s}: success={ms:.3f} vs fail={mf:.3f} ({direction}{abs(effect):.3f})")


def analyze_reasoning_bottlenecks(trajs, ms_data):
    """识别 reasoning bottleneck: trajectory 中哪些位置需要最强的 reasoning"""
    print("\n" + "=" * 80)
    print("Reasoning Bottleneck Identification")
    print("=" * 80)

    # For each step, compute "reasoning demand" proxies
    # High reasoning demand = low agreement + action type change + new action type
    step_reasoning = defaultdict(lambda: {
        "total": 0, "correct": 0,
        "type_change": 0, "type_change_correct": 0,
        "new_type": 0, "new_type_correct": 0,
        "agreement_sum": 0, "agreement_count": 0,
    })

    for traj in trajs:
        ep_id = traj["episode_id"]
        ms = ms_data.get(ep_id)
        seen_types = set()

        for i, s in enumerate(traj["step_results"]):
            step_num = s["step_num"]
            correct = s["extract_match"]
            gt_type = s["gt_action_type"]

            step_reasoning[step_num]["total"] += 1
            if correct:
                step_reasoning[step_num]["correct"] += 1

            # Is this a type change from previous step?
            if i > 0:
                prev_type = traj["step_results"][i-1]["gt_action_type"]
                if gt_type != prev_type:
                    step_reasoning[step_num]["type_change"] += 1
                    if correct:
                        step_reasoning[step_num]["type_change_correct"] += 1

            # Is this the first occurrence of this action type in trajectory?
            if gt_type not in seen_types:
                step_reasoning[step_num]["new_type"] += 1
                if correct:
                    step_reasoning[step_num]["new_type_correct"] += 1
            seen_types.add(gt_type)

            # Agreement from multi-sample
            if ms:
                for ss in ms.get("step_samples", []):
                    if ss["step_num"] == step_num:
                        samples = ss.get("samples", [])
                        pred_types = [s2["pred_action"].get("action", "?")
                                     for s2 in samples
                                     if s2.get("pred_action") and isinstance(s2["pred_action"], dict)]
                        if pred_types:
                            tc = Counter(pred_types)
                            agr = tc.most_common(1)[0][1] / len(pred_types)
                            step_reasoning[step_num]["agreement_sum"] += agr
                            step_reasoning[step_num]["agreement_count"] += 1
                        break

    print(f"\nPer-step reasoning demand analysis:")
    print(f"  {'Step':>5s} | {'N':>6s} | {'Accuracy':>9s} | {'TypeChange':>11s} | {'TC Acc':>7s} | {'NewType':>8s} | {'NT Acc':>7s} | {'Agree':>6s}")
    for step in sorted(step_reasoning.keys())[:10]:
        st = step_reasoning[step]
        acc = st["correct"] / st["total"] * 100 if st["total"] > 0 else 0
        tc = st["type_change"]
        tc_acc = st["type_change_correct"] / tc * 100 if tc > 0 else 0
        nt = st["new_type"]
        nt_acc = st["new_type_correct"] / nt * 100 if nt > 0 else 0
        agr = st["agreement_sum"] / st["agreement_count"] if st["agreement_count"] > 0 else 0
        print(f"  {step:>5d} | {st['total']:>6d} | {acc:>8.1f}% | {tc:>11d} | {tc_acc:>6.1f}% | {nt:>8d} | {nt_acc:>6.1f}% | {agr:>5.3f}")

    # Key insight: accuracy at type-change points vs same-type continuation
    all_tc_correct = sum(st["type_change_correct"] for st in step_reasoning.values())
    all_tc = sum(st["type_change"] for st in step_reasoning.values())
    all_same_correct = sum(st["correct"] - st["type_change_correct"] for st in step_reasoning.values())
    all_same = sum(st["total"] - st["type_change"] for st in step_reasoning.values())

    tc_acc = all_tc_correct / all_tc * 100 if all_tc > 0 else 0
    same_acc = all_same_correct / all_same * 100 if all_same > 0 else 0
    print(f"\n  Action type change accuracy: {tc_acc:.1f}% (N={all_tc})")
    print(f"  Same type continuation acc:  {same_acc:.1f}% (N={all_same})")
    print(f"  Delta: {tc_acc - same_acc:+.1f}pp")


def analyze_action_diversity_impact(features_list):
    """分析 action diversity 对 reasoning 的影响"""
    print("\n" + "=" * 80)
    print("Action Diversity Impact on Long-Horizon Reasoning")
    print("=" * 80)

    # Group by number of unique action types
    for n_types in range(1, 6):
        subset = [f for f in features_list if f["unique_action_types"] == n_types]
        if len(subset) < 10:
            continue
        n_succ = sum(1 for f in subset if f["task_success"])
        avg_len = sum(f["trajectory_length"] for f in subset) / len(subset)
        tsr = n_succ / len(subset) * 100
        avg_agree = sum(f["mean_agreement"] for f in subset) / len(subset) if any(f["mean_agreement"] > 0 for f in subset) else 0
        print(f"  {n_types} unique types: N={len(subset):4d}, TSR={tsr:5.1f}%, avg_len={avg_len:.1f}, avg_agree={avg_agree:.3f}")

    # Transition rate vs success
    print(f"\nTransition rate (fraction of steps where action type changes):")
    bins = [(0, 0.01), (0.01, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.01)]
    for lo, hi in bins:
        subset = [f for f in features_list if lo <= f["transition_rate"] < hi]
        if len(subset) < 10:
            continue
        n_succ = sum(1 for f in subset if f["task_success"])
        tsr = n_succ / len(subset) * 100
        avg_len = sum(f["trajectory_length"] for f in subset) / len(subset)
        print(f"  rate [{lo:.2f}, {hi:.2f}): N={len(subset):4d}, TSR={tsr:5.1f}%, avg_len={avg_len:.1f}")


def analyze_state_transition_complexity(trajs, ms_data):
    """分析状态转移的复杂度"""
    print("\n" + "=" * 80)
    print("State Transition Complexity Analysis")
    print("=" * 80)

    # For each trajectory, compute state transition complexity metrics
    complexity_vs_success = []

    for traj in trajs:
        ep_id = traj["episode_id"]
        ms = ms_data.get(ep_id)

        gt_types = []
        if ms:
            gt_types = [ss["gt_action_type"] for ss in ms.get("step_samples", [])]
        else:
            gt_types = [s["gt_action_type"] for s in traj["step_results"]]

        if len(gt_types) < 2:
            continue

        # Transition graph complexity
        transitions = set()
        for i in range(1, len(gt_types)):
            transitions.add((gt_types[i-1], gt_types[i]))

        # Number of unique transitions
        n_unique_transitions = len(transitions)

        # Max loop (same action type repeated consecutively)
        max_repeat = 1
        curr_repeat = 1
        for i in range(1, len(gt_types)):
            if gt_types[i] == gt_types[i-1]:
                curr_repeat += 1
                max_repeat = max(max_repeat, curr_repeat)
            else:
                curr_repeat = 1

        # "Navigation depth": number of system_button/swipe before first click/type
        nav_depth = 0
        for t in gt_types:
            if t in ("system_button", "swipe", "open"):
                nav_depth += 1
            else:
                break

        complexity_vs_success.append({
            "n_unique_transitions": n_unique_transitions,
            "max_repeat": max_repeat,
            "nav_depth": nav_depth,
            "success": traj["task_success"],
            "length": traj["num_steps"],
        })

    # Analyze
    for metric in ["n_unique_transitions", "max_repeat", "nav_depth"]:
        print(f"\n  {metric}:")
        vals_success = [c[metric] for c in complexity_vs_success if c["success"]]
        vals_failure = [c[metric] for c in complexity_vs_success if not c["success"]]
        ms = sum(vals_success) / len(vals_success) if vals_success else 0
        mf = sum(vals_failure) / len(vals_failure) if vals_failure else 0
        print(f"    Success mean: {ms:.2f}, Failure mean: {mf:.2f}, Delta: {ms-mf:+.2f}")

    # Nav depth breakdown
    print(f"\n  Navigation depth breakdown:")
    for d in range(5):
        subset = [c for c in complexity_vs_success if c["nav_depth"] == d]
        if not subset:
            continue
        tsr = sum(1 for c in subset if c["success"]) / len(subset) * 100
        print(f"    depth={d}: N={len(subset):4d}, TSR={tsr:.1f}%")


def analyze_compounding_sensitivity(trajs, ms_data):
    """分析哪些 feature 导致 error 更快 compound"""
    print("\n" + "=" * 80)
    print("Compounding Sensitivity: What Makes Errors Cascade Faster")
    print("=" * 80)

    # For medium+ trajectories, compute "how quickly accuracy decays"
    # Group by features and compare decay rates

    # Step-by-step accuracy conditional on reaching that step
    # Split by has_open vs no_open
    groups = {
        "has_open": lambda t: any(s["gt_action_type"] == "open" for s in t["step_results"]),
        "no_open": lambda t: not any(s["gt_action_type"] == "open" for s in t["step_results"]),
        "high_diversity": lambda t: len(set(s["gt_action_type"] for s in t["step_results"])) >= 3,
        "low_diversity": lambda t: len(set(s["gt_action_type"] for s in t["step_results"])) <= 2,
    }

    # Only look at 4+ step trajectories
    long_trajs = [t for t in trajs if t["num_steps"] >= 4]

    for group_name, group_fn in groups.items():
        subset = [t for t in long_trajs if group_fn(t)]
        if len(subset) < 50:
            continue

        step_acc = defaultdict(lambda: {"reached": 0, "correct": 0})
        for t in subset:
            for s in t["step_results"]:
                step_acc[s["step_num"]]["reached"] += 1
                if s["extract_match"]:
                    step_acc[s["step_num"]]["correct"] += 1

        print(f"\n  {group_name} (N={len(subset)}):")
        cum_p = 1.0
        for step in sorted(step_acc.keys())[:6]:
            st = step_acc[step]
            p = st["correct"] / st["reached"] if st["reached"] > 0 else 0
            cum_p *= p
            print(f"    Step {step}: P={p:.3f}, Cum={cum_p:.4f} (N={st['reached']})")

    # Compare: high vs low action entropy
    print(f"\n  Compounding rate comparison (4+ step trajs):")
    ms_trajs = [t for t in long_trajs if t["episode_id"] in ms_data]

    high_entropy = []
    low_entropy = []
    for t in ms_trajs:
        ms = ms_data[t["episode_id"]]
        gt_types = [ss["gt_action_type"] for ss in ms.get("step_samples", [])]
        ent = _entropy(gt_types)
        if ent > 1.5:
            high_entropy.append(t)
        elif ent < 0.8:
            low_entropy.append(t)

    for name, group in [("high_entropy(>1.5)", high_entropy), ("low_entropy(<0.8)", low_entropy)]:
        if len(group) < 30:
            continue
        tsr = sum(1 for t in group if t["task_success"]) / len(group) * 100
        avg_progress = sum(t["final_step_id"] / t["num_steps"] for t in group) / len(group)
        print(f"    {name:>25s}: N={len(group):4d}, TSR={tsr:.1f}%, avg_progress={avg_progress:.3f}")


def analyze_reasoning_chain_breaks(trajs, ms_data):
    """分析 reasoning chain 在哪里断裂"""
    print("\n" + "=" * 80)
    print("Reasoning Chain Break Analysis")
    print("=" * 80)

    # For each failed trajectory, identify the "reasoning break point":
    # - What was the model doing before the error?
    # - What did the model predict vs what it should have predicted?
    # - Was this a "shallow" error (simple action confusion) or "deep" error (wrong reasoning)?

    break_categories = Counter()
    break_by_step = defaultdict(Counter)

    for traj in trajs:
        if traj["task_success"]:
            continue

        for i, s in enumerate(traj["step_results"]):
            if s["extract_match"]:
                continue

            # Classify the reasoning break
            gt_type = s["gt_action_type"]
            pred = s.get("pred_action", {})
            pred_type = pred.get("action", "?") if isinstance(pred, dict) else "?"

            # Deep reasoning categories
            if gt_type == "open" and pred_type != "open":
                # Model doesn't know it needs to launch an app
                cat = "missing_app_context"
            elif pred_type == "terminate" and i < traj["num_steps"] - 1:
                # Model thinks it's done but isn't
                cat = "premature_goal_satisfaction"
            elif gt_type == "wait" and pred_type != "wait":
                # Model doesn't understand need to wait for state change
                cat = "no_state_change_awareness"
            elif not s["type_match"] and i > 0:
                # Mid-trajectory action type error - likely reasoning failure
                prev_gt = traj["step_results"][i-1]["gt_action_type"]
                if prev_gt == gt_type:
                    cat = "same_type_reasoning_fail"  # should continue same action
                else:
                    cat = "transition_reasoning_fail"  # should switch action type
            elif s["type_match"] and not s["extract_match"]:
                # Right action type, wrong target - grounding failure
                cat = "grounding_failure"
            else:
                cat = "other"

            break_categories[cat] += 1
            break_by_step[s["step_num"]][cat] += 1
            break

    total = sum(break_categories.values())
    print(f"\nReasoning break categories (N={total} failed episodes):")
    for cat, c in break_categories.most_common():
        print(f"  {cat:>35s}: {c:5d} ({c/total*100:.1f}%)")

    print(f"\nBreak category by step position:")
    top_cats = [c[0] for c in break_categories.most_common(5)]
    header = f"  {'Step':>5s} |" + "".join(f" {c[:20]:>20s} |" for c in top_cats)
    print(header)
    for step in sorted(break_by_step.keys())[:6]:
        row = f"  {step:>5d} |"
        total_step = sum(break_by_step[step].values())
        for cat in top_cats:
            c = break_by_step[step].get(cat, 0)
            pct = c / total_step * 100 if total_step > 0 else 0
            row += f" {c:4d} ({pct:4.1f}%) |"
        print(row)


def synthesize_findings(feature_results, features_list):
    """综合发现: 什么 feature 最影响 long-horizon reasoning"""
    print("\n" + "=" * 80)
    print("SYNTHESIS: Key Features Affecting Long-Horizon Reasoning")
    print("=" * 80)

    # Top features by Cohen's d
    top_positive = [r for r in feature_results if r["cohen_d"] > 0][:5]
    top_negative = [r for r in feature_results if r["cohen_d"] < 0][:5]

    print(f"\n  Features that HELP success (higher value → more success):")
    for r in top_positive:
        print(f"    {r['feature']:>30s}: d={r['cohen_d']:+.3f} (success={r['mean_success']:.3f}, fail={r['mean_failure']:.3f})")

    print(f"\n  Features that HURT success (higher value → more failure):")
    for r in top_negative:
        print(f"    {r['feature']:>30s}: d={r['cohen_d']:+.3f} (success={r['mean_success']:.3f}, fail={r['mean_failure']:.3f})")

    # For long trajectories only
    long_features = [f for f in features_list if f["trajectory_length"] >= 4]
    print(f"\n  For LONG trajectories (length ≥ 4, N={len(long_features)}):")
    long_results = analyze_feature_importance_quiet(long_features)
    for r in long_results[:10]:
        print(f"    {r['feature']:>30s}: d={r['cohen_d']:+.3f}")

    # Interaction effects
    print(f"\n  INTERACTION: has_open × trajectory_length → TSR:")
    for has_open in [0, 1]:
        for bucket in ["short(1-3)", "medium(4-7)", "long(8+)"]:
            if bucket == "short(1-3)":
                subset = [f for f in features_list if f["has_open"] == has_open and f["trajectory_length"] <= 3]
            elif bucket == "medium(4-7)":
                subset = [f for f in features_list if f["has_open"] == has_open and 4 <= f["trajectory_length"] <= 7]
            else:
                subset = [f for f in features_list if f["has_open"] == has_open and f["trajectory_length"] >= 8]
            if len(subset) < 10:
                continue
            tsr = sum(1 for f in subset if f["task_success"]) / len(subset) * 100
            print(f"    has_open={has_open}, {bucket:15s}: N={len(subset):4d}, TSR={tsr:.1f}%")


def analyze_feature_importance_quiet(features_list, target="task_success"):
    """Quiet version of feature importance"""
    feature_names = [k for k in features_list[0].keys()
                     if k not in ("task_success", "progress", "episode_id")]
    outcomes = [f[target] for f in features_list]
    results = []
    for fname in feature_names:
        values = [f[fname] for f in features_list]
        std_val = _std(values)
        if std_val < 1e-10:
            continue
        success_vals = [v for v, o in zip(values, outcomes) if o == 1]
        failure_vals = [v for v, o in zip(values, outcomes) if o == 0]
        if not success_vals or not failure_vals:
            continue
        effect = sum(success_vals)/len(success_vals) - sum(failure_vals)/len(failure_vals)
        pooled_std = (((_std(success_vals) if len(success_vals) > 1 else 0)**2 * (len(success_vals)-1) +
                       (_std(failure_vals) if len(failure_vals) > 1 else 0)**2 * (len(failure_vals)-1)) /
                      max(len(success_vals) + len(failure_vals) - 2, 1)) ** 0.5
        cohen_d = effect / pooled_std if pooled_std > 1e-10 else 0
        results.append({"feature": fname, "cohen_d": cohen_d, "abs_d": abs(cohen_d),
                        "mean_success": sum(success_vals)/len(success_vals),
                        "mean_failure": sum(failure_vals)/len(failure_vals)})
    results.sort(key=lambda x: -x["abs_d"])
    return results


def main():
    print("Loading data...")
    trajs, ms_data = load_data()
    print(f"Loaded {len(trajs)} trajectories, {len(ms_data)} multi-sample episodes")

    # Extract features for all trajectories
    print("Extracting features...")
    features_list = []
    for t in trajs:
        ms = ms_data.get(t["episode_id"])
        feat = extract_task_features(t, ms)
        feat["episode_id"] = t["episode_id"]
        features_list.append(feat)

    # Run analyses
    feature_results = analyze_feature_importance(features_list)
    analyze_horizon_scaling(features_list)
    analyze_reasoning_bottlenecks(trajs, ms_data)
    analyze_action_diversity_impact(features_list)
    analyze_state_transition_complexity(trajs, ms_data)
    analyze_compounding_sensitivity(trajs, ms_data)
    analyze_reasoning_chain_breaks(trajs, ms_data)
    synthesize_findings(feature_results, features_list)

    # Save
    with open(os.path.join(OUTPUT_DIR, "feature_importance.json"), "w") as f:
        json.dump(feature_results[:30], f, indent=2)

    with open(os.path.join(OUTPUT_DIR, "all_features.jsonl"), "w") as f:
        for feat in features_list:
            f.write(json.dumps(feat) + "\n")

    print(f"\nResults saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
