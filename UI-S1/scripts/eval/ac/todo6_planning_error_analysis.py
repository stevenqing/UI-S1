#!/usr/bin/env python3
"""
TODO 6: Planning 的问题，利用 LLM 对是否是 planning 的 error 进行分析

分析思路 (离线, 不调用 LLM, 基于规则):
1. Planning error 的 heuristic 分类:
   - 错误的 action type → planning error (选错了高层动作)
   - 正确 action type 但错误 target → execution error (知道做什么，但执行错了)
2. Step 0 特殊分析: open action 失败 → 明确的 planning error (不知道要先打开 app)
3. Action sequence 合理性分析: 预测的 action sequence 是否符合常见 task pattern
4. Goal decomposition 分析: goal 中隐含的 subtask 数量 vs 实际 trajectory 长度
5. "Wrong direction" 分析: 某些 step 的 action 与 goal 方向相反
6. 总结: Planning error vs Execution error 的比例

注: 由于没有 LLM API access，使用基于规则的 heuristic 分类。
    如需 LLM-based 分类，需要在有 GPU/API 的环境中运行。
"""

import json
import os
import re
from collections import Counter, defaultdict

EVAL_A_DIR = "outputs/eval_a_ac/Qwen2.5-VL-7B"
OUTPUT_DIR = "outputs/todo6_planning_analysis"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_trajectory_results():
    results = []
    with open(os.path.join(EVAL_A_DIR, "trajectory_results.jsonl")) as f:
        for line in f:
            results.append(json.loads(line.strip()))
    return results


def classify_planning_vs_execution(step, trajectory):
    """
    基于规则的 planning vs execution error 分类

    Planning error (高层决策错误):
    - Wrong action type: 模型选择了错误的动作类型
    - 特别是: GT=open 但 pred=其他 (不知道要打开 app)
    - GT=click 但 pred=swipe/system_button (对 UI 交互方式理解错误)
    - GT=type 但 pred=click (不知道需要输入文字)

    Execution error (低层执行错误):
    - Action type 正确但 grounding 错误 (知道要 click 但点错了位置)
    - Coordinate 偏差 (near-miss)
    """
    if step["extract_match"]:
        return "correct"

    gt_type = step["gt_action_type"]
    pred_action = step.get("pred_action", {})
    pred_type = pred_action.get("action", "unknown") if isinstance(pred_action, dict) else "unknown"

    if not step["type_match"]:
        # Action type mismatch → primarily planning error
        # But subcategorize:
        if gt_type == "open" and pred_type != "open":
            return "planning_app_launch"  # 不知道要先打开 app
        elif gt_type == "type" and pred_type == "click":
            return "planning_input_needed"  # 不知道需要输入
        elif gt_type == "click" and pred_type in ("swipe", "system_button"):
            return "planning_wrong_interaction"  # 交互方式错误
        elif gt_type == "wait" and pred_type in ("click", "terminate"):
            return "planning_premature_action"  # 过早行动，应该等待
        elif pred_type == "terminate":
            return "planning_premature_terminate"  # 过早结束
        elif gt_type == "system_button":
            return "planning_system_nav"  # 系统导航理解错误
        else:
            return "planning_other"
    else:
        # Action type correct, grounding wrong → execution error
        if gt_type == "click":
            # Check coordinate distance
            gt_coord = step.get("gt_action", {}).get("coordinate", [])
            pred_coord = pred_action.get("coordinate", []) if isinstance(pred_action, dict) else []
            if gt_coord and pred_coord and len(gt_coord) == 2 and len(pred_coord) == 2:
                dist = ((gt_coord[0] - pred_coord[0])**2 + (gt_coord[1] - pred_coord[1])**2) ** 0.5
                if dist < 100:
                    return "execution_near_miss"  # 差一点点
                else:
                    return "execution_wrong_target"  # 点错了目标
            return "execution_grounding"
        elif gt_type == "type":
            return "execution_wrong_text"  # 输入了错误的文字
        elif gt_type == "swipe":
            return "execution_wrong_direction"  # swipe 方向错
        else:
            return "execution_other"


def analyze_planning_vs_execution(trajectories):
    """分析 1: Planning error vs Execution error 整体分布"""
    print("=" * 80)
    print("Analysis 1: Planning Error vs Execution Error Distribution")
    print("=" * 80)

    error_classes = Counter()
    per_step_classes = defaultdict(Counter)

    for t in trajectories:
        for s in t["step_results"]:
            cls = classify_planning_vs_execution(s, t)
            error_classes[cls] += 1
            per_step_classes[s["step_num"]][cls] += 1

    total = sum(error_classes.values())
    planning_total = sum(c for k, c in error_classes.items() if k.startswith("planning_"))
    execution_total = sum(c for k, c in error_classes.items() if k.startswith("execution_"))
    correct_total = error_classes.get("correct", 0)

    print(f"\nOverall distribution (N={total} steps):")
    print(f"  Correct:        {correct_total:5d} ({correct_total/total*100:.1f}%)")
    print(f"  Planning error: {planning_total:5d} ({planning_total/total*100:.1f}%)")
    print(f"  Execution error:{execution_total:5d} ({execution_total/total*100:.1f}%)")

    print(f"\n  Planning/Execution ratio: {planning_total/execution_total:.2f}x" if execution_total > 0 else "")

    print(f"\nDetailed breakdown:")
    for cls, c in sorted(error_classes.items(), key=lambda x: -x[1]):
        print(f"  {cls:>35s}: {c:5d} ({c/total*100:.1f}%)")

    # Planning vs Execution by step position
    print(f"\nPlanning vs Execution by step position:")
    print(f"  {'Step':>5s} | {'Plan Err':>10s} | {'Exec Err':>10s} | {'Plan%':>7s}")
    for step in sorted(per_step_classes.keys())[:8]:
        plan = sum(c for k, c in per_step_classes[step].items() if k.startswith("planning_"))
        exec_ = sum(c for k, c in per_step_classes[step].items() if k.startswith("execution_"))
        total_err = plan + exec_
        plan_pct = plan / total_err * 100 if total_err > 0 else 0
        print(f"  {step:>5d} | {plan:>10d} | {exec_:>10d} | {plan_pct:>6.1f}%")

    return error_classes


def analyze_planning_errors_detail(trajectories):
    """分析 2: Planning error 的详细特征"""
    print("\n" + "=" * 80)
    print("Analysis 2: Planning Error Details")
    print("=" * 80)

    # App launch failures (GT=open)
    app_launch_failures = []
    for t in trajectories:
        for s in t["step_results"]:
            if classify_planning_vs_execution(s, t) == "planning_app_launch":
                app_launch_failures.append({
                    "goal": t["goal"],
                    "gt": s.get("gt_action"),
                    "pred": s.get("pred_action"),
                    "step": s["step_num"],
                })
                break

    print(f"\n  App Launch Failures (GT=open, pred≠open): {len(app_launch_failures)}")
    # What does the model predict instead?
    pred_types = Counter()
    for f in app_launch_failures:
        if isinstance(f["pred"], dict):
            pred_types[f["pred"].get("action", "?")] += 1
    print(f"  What model predicts instead:")
    for pt, c in pred_types.most_common():
        print(f"    {pt}: {c} ({c/len(app_launch_failures)*100:.1f}%)")

    # Premature termination analysis
    premature_term = []
    for t in trajectories:
        for s in t["step_results"]:
            cls = classify_planning_vs_execution(s, t)
            if cls == "planning_premature_terminate":
                premature_term.append({
                    "goal": t["goal"],
                    "step": s["step_num"],
                    "num_steps": t["num_steps"],
                    "gt_type": s["gt_action_type"],
                })
                break

    print(f"\n  Premature Termination: {len(premature_term)}")
    if premature_term:
        avg_step = sum(p["step"] for p in premature_term) / len(premature_term)
        avg_remaining = sum(p["num_steps"] - p["step"] - 1 for p in premature_term) / len(premature_term)
        print(f"    Average step of termination: {avg_step:.1f}")
        print(f"    Average remaining steps: {avg_remaining:.1f}")
        gt_at_term = Counter(p["gt_type"] for p in premature_term)
        print(f"    GT action type when terminated:")
        for at, c in gt_at_term.most_common():
            print(f"      {at}: {c}")


def analyze_goal_complexity_vs_planning(trajectories):
    """分析 3: Goal 复杂度与 planning error 的关系"""
    print("\n" + "=" * 80)
    print("Analysis 3: Goal Complexity vs Planning Error Rate")
    print("=" * 80)

    # Count planning errors per episode
    episode_stats = []
    for t in trajectories:
        plan_errors = 0
        exec_errors = 0
        for s in t["step_results"]:
            cls = classify_planning_vs_execution(s, t)
            if cls.startswith("planning_"):
                plan_errors += 1
            elif cls.startswith("execution_"):
                exec_errors += 1

        episode_stats.append({
            "goal_words": len(t["goal"].split()),
            "num_steps": t["num_steps"],
            "plan_errors": plan_errors,
            "exec_errors": exec_errors,
            "success": t["task_success"],
            "bucket": t.get("length_bucket", "unknown"),
        })

    # By goal length
    bins = [(0, 10), (10, 15), (15, 20), (20, 30), (30, 100)]
    print(f"\n  {'Goal Words':>12s} | {'N':>5s} | {'Plan Err%':>10s} | {'Exec Err%':>10s} | {'Success%':>9s}")
    for lo, hi in bins:
        bucket = [e for e in episode_stats if lo <= e["goal_words"] < hi]
        if not bucket:
            continue
        n = len(bucket)
        has_plan = sum(1 for e in bucket if e["plan_errors"] > 0)
        has_exec = sum(1 for e in bucket if e["exec_errors"] > 0)
        succ = sum(1 for e in bucket if e["success"])
        print(f"  {lo:3d}-{hi:3d} words | {n:5d} | {has_plan/n*100:>9.1f}% | {has_exec/n*100:>9.1f}% | {succ/n*100:>8.1f}%")


def analyze_action_sequence_rationality(trajectories):
    """分析 4: Predicted action sequence 的合理性"""
    print("\n" + "=" * 80)
    print("Analysis 4: Action Sequence Rationality")
    print("=" * 80)

    # Common rational patterns in mobile UI tasks
    # Pattern 1: open → click → type → click (search pattern)
    # Pattern 2: click → click → click (navigation)
    # Pattern 3: open → click → swipe → click (browse & select)

    # Compare GT action sequences with predicted sequences
    gt_bigrams = Counter()
    pred_bigrams = Counter()

    for t in trajectories:
        if len(t["step_results"]) < 2:
            continue
        for i in range(1, len(t["step_results"])):
            gt_prev = t["step_results"][i - 1]["gt_action_type"]
            gt_curr = t["step_results"][i]["gt_action_type"]
            gt_bigrams[(gt_prev, gt_curr)] += 1

            pred_prev = t["step_results"][i - 1].get("pred_action", {})
            pred_curr = t["step_results"][i].get("pred_action", {})
            if isinstance(pred_prev, dict) and isinstance(pred_curr, dict):
                pred_bigrams[(pred_prev.get("action", "?"), pred_curr.get("action", "?"))] += 1

    print(f"\nGT action bigrams (top 10):")
    for (a, b), c in gt_bigrams.most_common(10):
        print(f"  {a:>15s} → {b:>15s}: {c:4d}")

    print(f"\nPredicted action bigrams (top 10):")
    for (a, b), c in pred_bigrams.most_common(10):
        print(f"  {a:>15s} → {b:>15s}: {c:4d}")

    # Divergence between GT and predicted patterns
    print(f"\nBigram divergence (GT has it but Pred doesn't, or vice versa):")
    all_bigrams = set(gt_bigrams.keys()) | set(pred_bigrams.keys())
    divergences = []
    for bg in all_bigrams:
        gt_freq = gt_bigrams.get(bg, 0)
        pred_freq = pred_bigrams.get(bg, 0)
        if gt_freq + pred_freq > 10:
            divergences.append((bg, gt_freq, pred_freq, pred_freq - gt_freq))

    divergences.sort(key=lambda x: -abs(x[3]))
    for bg, gt_f, pred_f, diff in divergences[:10]:
        direction = "↑OVER" if diff > 0 else "↓UNDER"
        print(f"  {bg[0]:>15s} → {bg[1]:>15s}: GT={gt_f:3d}, Pred={pred_f:3d}, {direction}-predicted by {abs(diff)}")


def analyze_planning_error_llm_prompt(trajectories):
    """分析 5: 生成 LLM-based planning error 分类的 prompt"""
    print("\n" + "=" * 80)
    print("Analysis 5: LLM-Based Planning Error Classification Template")
    print("=" * 80)

    # Generate sample prompts for LLM classification
    # This would be run in a separate step with GPU/API access

    sample_failures = [t for t in trajectories if not t["task_success"]][:5]

    print(f"\n  Template for LLM-based planning error classification:")
    print(f"  (Run this with Qwen or GPT-4 to classify errors)")
    print()

    for t in sample_failures[:3]:
        first_error = None
        for s in t["step_results"]:
            if not s["extract_match"]:
                first_error = s
                break

        if first_error is None:
            continue

        prompt = f"""Classify the following GUI agent error:

Goal: {t['goal']}
Total steps needed: {t['num_steps']}
Error at step: {first_error['step_num']}
Ground truth action: {json.dumps(first_error.get('gt_action', {}))}
Predicted action: {json.dumps(first_error.get('pred_action', {}))}

Is this error primarily:
A) PLANNING error - the agent chose the wrong high-level action (wrong action type, wrong strategy)
B) EXECUTION error - the agent chose the right action type but wrong target/parameter
C) CONTEXT error - the agent lost track of what has been done / current state
D) CAPABILITY error - the agent doesn't know this action type exists or how to use it

Classify and explain briefly."""

        print(f"  --- Example {t['episode_id']} ---")
        print(f"  {prompt}")
        print()


def analyze_planning_error_summary(trajectories):
    """分析 6: Planning error 总结"""
    print("\n" + "=" * 80)
    print("Analysis 6: Planning Error Summary & Recommendations")
    print("=" * 80)

    # Aggregate
    all_classes = Counter()
    first_error_classes = Counter()

    for t in trajectories:
        if t["task_success"]:
            continue
        first_found = False
        for s in t["step_results"]:
            cls = classify_planning_vs_execution(s, t)
            all_classes[cls] += 1
            if not first_found and cls != "correct":
                first_error_classes[cls] += 1
                first_found = True

    total_errors = sum(c for k, c in all_classes.items() if k != "correct")
    planning = sum(c for k, c in all_classes.items() if k.startswith("planning_"))
    execution = sum(c for k, c in all_classes.items() if k.startswith("execution_"))

    print(f"\n  SUMMARY:")
    print(f"  Total error steps: {total_errors}")
    print(f"  Planning errors:   {planning} ({planning/total_errors*100:.1f}%)")
    print(f"  Execution errors:  {execution} ({execution/total_errors*100:.1f}%)")

    first_total = sum(first_error_classes.values())
    first_plan = sum(c for k, c in first_error_classes.items() if k.startswith("planning_"))
    first_exec = sum(c for k, c in first_error_classes.items() if k.startswith("execution_"))

    print(f"\n  First error in trajectory:")
    print(f"  Planning errors:   {first_plan} ({first_plan/first_total*100:.1f}%)")
    print(f"  Execution errors:  {first_exec} ({first_exec/first_total*100:.1f}%)")

    print(f"\n  Top 5 first-error classes:")
    for cls, c in first_error_classes.most_common(5):
        print(f"    {cls:>35s}: {c:4d} ({c/first_total*100:.1f}%)")

    print(f"\n  RECOMMENDATIONS:")
    if planning > execution:
        print(f"  1. Planning errors dominate ({planning/total_errors*100:.0f}%)")
        print(f"     → Need better high-level action selection")
        print(f"     → A planning agent or action type predictor could help")
    else:
        print(f"  1. Execution errors dominate ({execution/total_errors*100:.0f}%)")
        print(f"     → Need better grounding/targeting")
        print(f"     → A grounding specialist agent could help")

    app_launch = all_classes.get("planning_app_launch", 0)
    if app_launch > 50:
        print(f"  2. App launch failures ({app_launch}) are a major planning error")
        print(f"     → M3 Router approach is the right direction")

    premature = all_classes.get("planning_premature_terminate", 0)
    if premature > 20:
        print(f"  3. Premature termination ({premature}) — agent thinks task is done early")
        print(f"     → Need better progress tracking / completion detection")


def main():
    print("Loading data...")
    trajectories = load_trajectory_results()
    print(f"Loaded {len(trajectories)} trajectories")

    error_classes = analyze_planning_vs_execution(trajectories)
    analyze_planning_errors_detail(trajectories)
    analyze_goal_complexity_vs_planning(trajectories)
    analyze_action_sequence_rationality(trajectories)
    analyze_planning_error_llm_prompt(trajectories)
    analyze_planning_error_summary(trajectories)

    # Save results
    results = {
        "error_classes": dict(error_classes),
    }
    with open(os.path.join(OUTPUT_DIR, "planning_analysis_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {OUTPUT_DIR}/planning_analysis_results.json")


if __name__ == "__main__":
    main()
