#!/usr/bin/env python3
"""
Sub-Task Decomposition Evaluation

核心问题: 如果把 long-horizon task 切成多个 sub-task 独立 evaluate，
agent 在 sub-task 上的表现是否显著更好？

分析思路:
1. Phase-based segmentation: 按 action type phase 切分 (连续相同 action type = 1 phase)
2. Transition-based segmentation: 在 action type 切换点切分
3. 对每个 sub-task 独立评估 accuracy
4. 比较: full-task TSR vs sub-task TSR product vs sub-task individual accuracy
5. 诊断: sub-task 间的 "handoff" 问题 — phase transition 处的 error rate
6. Oracle sub-task: 如果每个 sub-task 的第一步被 oracle fix，TSR 如何变化
7. 利用 C4+C7 独立步骤数据，避免 AR stop-on-error 的 survivorship bias

关键假说:
H1: Sub-task 内的 per-step accuracy 显著高于 overall per-step accuracy
H2: Phase transition 处 (sub-task 之间的衔接) 是主要瓶颈
H3: 如果能把 phase transition 做对，TSR 会大幅提升
"""

import json
import os
import math
from collections import Counter, defaultdict

EVAL_A_DIR = "outputs/eval_a_ac/Qwen2.5-VL-7B"
C4C7_DIR = "outputs/eval_c4c7_ac/Qwen2.5-VL-7B"
OUTPUT_DIR = "outputs/subtask_decomposition_eval"

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


def segment_into_phases(gt_action_types):
    """按连续相同 action type 切成 phases"""
    if not gt_action_types:
        return []
    phases = []
    curr_type = gt_action_types[0]
    curr_start = 0
    for i in range(1, len(gt_action_types)):
        if gt_action_types[i] != curr_type:
            phases.append({"type": curr_type, "start": curr_start, "end": i - 1, "length": i - curr_start})
            curr_type = gt_action_types[i]
            curr_start = i
    phases.append({"type": curr_type, "start": curr_start, "end": len(gt_action_types) - 1,
                   "length": len(gt_action_types) - curr_start})
    return phases


def segment_into_logical_subtasks(gt_action_types):
    """
    按逻辑 sub-task 切分:
    - Sub-task 1: App launch (open / 初始 system_button/click)
    - Sub-task 2+: 每次 action type 变化开始新的 sub-task
    但合并：连续的 navigation (system_button + swipe) 算一个 sub-task
    """
    if not gt_action_types:
        return []

    NAV_TYPES = {"system_button", "swipe", "open"}

    subtasks = []
    curr_start = 0
    curr_is_nav = gt_action_types[0] in NAV_TYPES

    for i in range(1, len(gt_action_types)):
        is_nav = gt_action_types[i] in NAV_TYPES
        # Start new subtask on nav ↔ non-nav transition
        if is_nav != curr_is_nav:
            subtasks.append({
                "start": curr_start, "end": i - 1,
                "length": i - curr_start,
                "types": list(set(gt_action_types[curr_start:i])),
                "is_nav": curr_is_nav,
            })
            curr_start = i
            curr_is_nav = is_nav

    subtasks.append({
        "start": curr_start, "end": len(gt_action_types) - 1,
        "length": len(gt_action_types) - curr_start,
        "types": list(set(gt_action_types[curr_start:])),
        "is_nav": curr_is_nav,
    })
    return subtasks


# ============================================================
# Analysis 1: Phase-level accuracy (C4+C7, no survivorship bias)
# ============================================================
def analyze_phase_accuracy_c4c7(ms_data):
    """用 C4+C7 独立评估数据分析每个 phase 的 accuracy"""
    print("=" * 80)
    print("Analysis 1: Phase-Level Accuracy (C4+C7 Independent Evaluation)")
    print("=" * 80)

    # For each episode, segment into phases, compute per-phase accuracy
    phase_stats = defaultdict(lambda: {
        "total_steps": 0, "correct_steps": 0,
        "total_phases": 0, "perfect_phases": 0,
        "lengths": [],
    })

    # Also track: accuracy at phase boundaries vs within phase
    boundary_stats = {"boundary_total": 0, "boundary_correct": 0,
                      "within_total": 0, "within_correct": 0}

    # Phase position stats
    phase_position_stats = defaultdict(lambda: {"total": 0, "correct": 0, "perfect": 0, "n_phases": 0})

    all_phase_records = []

    for ep_id, ep in ms_data.items():
        step_samples = ep.get("step_samples", [])
        if not step_samples:
            continue

        gt_types = [ss["gt_action_type"] for ss in step_samples]
        phases = segment_into_phases(gt_types)

        # Compute greedy accuracy for each step
        step_correct = []
        for ss in step_samples:
            samples = ss.get("samples", [])
            if samples:
                step_correct.append(samples[0].get("extract_match", False))
            else:
                step_correct.append(False)

        # Per-phase analysis
        for pi, phase in enumerate(phases):
            start, end = phase["start"], phase["end"]
            phase_steps = step_correct[start:end + 1]
            n_correct = sum(phase_steps)
            n_total = len(phase_steps)
            is_perfect = (n_correct == n_total)

            phase_stats[phase["type"]]["total_steps"] += n_total
            phase_stats[phase["type"]]["correct_steps"] += n_correct
            phase_stats[phase["type"]]["total_phases"] += 1
            phase_stats[phase["type"]]["perfect_phases"] += int(is_perfect)
            phase_stats[phase["type"]]["lengths"].append(n_total)

            # Phase position (first, middle, last)
            if pi == 0:
                pos = "first"
            elif pi == len(phases) - 1:
                pos = "last"
            else:
                pos = "middle"
            phase_position_stats[pos]["total"] += n_total
            phase_position_stats[pos]["correct"] += n_correct
            phase_position_stats[pos]["perfect"] += int(is_perfect)
            phase_position_stats[pos]["n_phases"] += 1

            all_phase_records.append({
                "ep_id": ep_id,
                "phase_idx": pi,
                "type": phase["type"],
                "length": n_total,
                "n_correct": n_correct,
                "is_perfect": is_perfect,
                "position": pos,
                "n_phases_in_traj": len(phases),
            })

            # Boundary analysis
            for j, sc in enumerate(phase_steps):
                if j == 0 and pi > 0:  # first step of non-first phase = boundary
                    boundary_stats["boundary_total"] += 1
                    if sc:
                        boundary_stats["boundary_correct"] += 1
                else:
                    boundary_stats["within_total"] += 1
                    if sc:
                        boundary_stats["within_correct"] += 1

    # Print results
    print(f"\nPer-action-type phase accuracy (greedy, C4+C7):")
    print(f"  {'Type':>15s} | {'#Phases':>8s} | {'#Steps':>7s} | {'Step Acc':>9s} | {'Phase Perfect%':>15s} | {'Avg Len':>8s}")
    print(f"  {'-'*15} | {'-'*8} | {'-'*7} | {'-'*9} | {'-'*15} | {'-'*8}")
    for at in sorted(phase_stats.keys(), key=lambda x: -phase_stats[x]["total_steps"]):
        st = phase_stats[at]
        step_acc = st["correct_steps"] / st["total_steps"] * 100 if st["total_steps"] > 0 else 0
        perfect_pct = st["perfect_phases"] / st["total_phases"] * 100 if st["total_phases"] > 0 else 0
        avg_len = sum(st["lengths"]) / len(st["lengths"]) if st["lengths"] else 0
        print(f"  {at:>15s} | {st['total_phases']:>8d} | {st['total_steps']:>7d} | {step_acc:>8.1f}% | {perfect_pct:>14.1f}% | {avg_len:>7.1f}")

    # Boundary vs Within
    b_acc = boundary_stats["boundary_correct"] / boundary_stats["boundary_total"] * 100 if boundary_stats["boundary_total"] > 0 else 0
    w_acc = boundary_stats["within_correct"] / boundary_stats["within_total"] * 100 if boundary_stats["within_total"] > 0 else 0
    print(f"\n  Phase boundary step accuracy: {b_acc:.1f}% ({boundary_stats['boundary_correct']}/{boundary_stats['boundary_total']})")
    print(f"  Within-phase step accuracy:   {w_acc:.1f}% ({boundary_stats['within_correct']}/{boundary_stats['within_total']})")
    print(f"  Boundary/Within ratio:        {b_acc/w_acc:.3f}x" if w_acc > 0 else "")
    print(f"  Delta:                        {b_acc - w_acc:+.1f}pp")

    # Phase position
    print(f"\n  Phase accuracy by position:")
    for pos in ["first", "middle", "last"]:
        st = phase_position_stats[pos]
        if st["total"] == 0:
            continue
        acc = st["correct"] / st["total"] * 100
        perf = st["perfect"] / st["n_phases"] * 100
        print(f"    {pos:>8s}: step_acc={acc:.1f}%, phase_perfect={perf:.1f}% (N={st['n_phases']})")

    return phase_stats, boundary_stats, all_phase_records


# ============================================================
# Analysis 2: Sub-task TSR simulation
# ============================================================
def analyze_subtask_tsr(ms_data):
    """
    模拟: 如果把每个 phase 当做独立 task, sub-task TSR 是多少？
    vs full-task TSR = product of all sub-task TSRs
    """
    print("\n" + "=" * 80)
    print("Analysis 2: Sub-Task TSR Simulation")
    print("=" * 80)

    # For each episode, compute:
    # 1. Full-task TSR (stop-on-error across all steps)
    # 2. Per-phase TSR (stop-on-error within each phase, reset between phases)
    # 3. Product TSR (product of per-phase TSRs = independence assumption)

    results_by_n_phases = defaultdict(lambda: {
        "full_success": 0, "product_success_sum": 0,
        "total": 0, "phase_successes": [],
    })

    per_phase_position = defaultdict(lambda: {"success": 0, "total": 0})

    for ep_id, ep in ms_data.items():
        step_samples = ep.get("step_samples", [])
        if not step_samples:
            continue

        gt_types = [ss["gt_action_type"] for ss in step_samples]
        phases = segment_into_phases(gt_types)

        # Greedy correctness
        step_correct = []
        for ss in step_samples:
            samples = ss.get("samples", [])
            if samples:
                step_correct.append(samples[0].get("extract_match", False))
            else:
                step_correct.append(False)

        n_phases = len(phases)

        # Full-task: all steps correct (stop-on-error)
        full_success = all(step_correct)

        # Per-phase: each phase independently all-correct
        phase_successes = []
        for pi, phase in enumerate(phases):
            start, end = phase["start"], phase["end"]
            phase_success = all(step_correct[start:end + 1])
            phase_successes.append(phase_success)

            per_phase_position[pi]["total"] += 1
            if phase_success:
                per_phase_position[pi]["success"] += 1

        # Product: product of phase success rates (= if phases were independent)
        product_success = all(phase_successes)  # for this episode, it's the same as full_success
        # But we want population-level product

        results_by_n_phases[n_phases]["total"] += 1
        if full_success:
            results_by_n_phases[n_phases]["full_success"] += 1
        results_by_n_phases[n_phases]["phase_successes"].append(phase_successes)

    # Compute population-level statistics
    print(f"\nFull-Task TSR vs Sub-Task Independent TSR:")
    print(f"  {'#Phases':>8s} | {'N':>6s} | {'Full TSR':>9s} | {'Sub-Task Product':>17s} | {'Gap':>8s} | {'Sub > Full?':>12s}")
    print(f"  {'-'*8} | {'-'*6} | {'-'*9} | {'-'*17} | {'-'*8} | {'-'*12}")

    for n_phases in sorted(results_by_n_phases.keys()):
        r = results_by_n_phases[n_phases]
        if r["total"] < 20:
            continue

        full_tsr = r["full_success"] / r["total"]

        # Compute per-phase success rate across population, then multiply
        phase_rates = []
        for pi in range(n_phases):
            successes = sum(1 for ps in r["phase_successes"] if len(ps) > pi and ps[pi])
            total = sum(1 for ps in r["phase_successes"] if len(ps) > pi)
            rate = successes / total if total > 0 else 0
            phase_rates.append(rate)

        product_tsr = 1.0
        for pr in phase_rates:
            product_tsr *= pr

        gap = product_tsr - full_tsr
        better = "YES" if product_tsr > full_tsr else "no"

        phase_str = " × ".join(f"{pr:.3f}" for pr in phase_rates[:5])
        if len(phase_rates) > 5:
            phase_str += f" × ... ({len(phase_rates)} total)"

        print(f"  {n_phases:>8d} | {r['total']:>6d} | {full_tsr:>8.1%} | {product_tsr:>16.1%} | {gap:>+7.1%} | {better:>12s}")
        print(f"           |        |           | = {phase_str}")

    # Per-phase-position success rate
    print(f"\n  Per-phase-position success rate (population level):")
    print(f"  {'Phase#':>7s} | {'N':>6s} | {'Success%':>9s}")
    for pi in sorted(per_phase_position.keys())[:8]:
        st = per_phase_position[pi]
        rate = st["success"] / st["total"] * 100 if st["total"] > 0 else 0
        print(f"  {pi:>7d} | {st['total']:>6d} | {rate:>8.1f}%")


# ============================================================
# Analysis 3: Phase transition handoff analysis
# ============================================================
def analyze_phase_transitions(ms_data):
    """分析 phase 之间的 handoff 问题"""
    print("\n" + "=" * 80)
    print("Analysis 3: Phase Transition (Handoff) Analysis")
    print("=" * 80)

    # For each transition A→B, compute accuracy of first step of B
    transition_acc = defaultdict(lambda: {"correct": 0, "total": 0})
    # Also: does the previous phase ending correctly help?
    transition_given_prev = defaultdict(lambda: {
        "prev_correct_correct": 0, "prev_correct_total": 0,
        "prev_wrong_correct": 0, "prev_wrong_total": 0,
    })

    for ep_id, ep in ms_data.items():
        step_samples = ep.get("step_samples", [])
        if not step_samples:
            continue

        gt_types = [ss["gt_action_type"] for ss in step_samples]
        phases = segment_into_phases(gt_types)

        step_correct = []
        for ss in step_samples:
            samples = ss.get("samples", [])
            if samples:
                step_correct.append(samples[0].get("extract_match", False))
            else:
                step_correct.append(False)

        for pi in range(1, len(phases)):
            prev_phase = phases[pi - 1]
            curr_phase = phases[pi]

            transition_key = f"{prev_phase['type']}→{curr_phase['type']}"
            first_step_correct = step_correct[curr_phase["start"]]
            last_prev_correct = step_correct[prev_phase["end"]]

            transition_acc[transition_key]["total"] += 1
            if first_step_correct:
                transition_acc[transition_key]["correct"] += 1

            if last_prev_correct:
                transition_given_prev[transition_key]["prev_correct_total"] += 1
                if first_step_correct:
                    transition_given_prev[transition_key]["prev_correct_correct"] += 1
            else:
                transition_given_prev[transition_key]["prev_wrong_total"] += 1
                if first_step_correct:
                    transition_given_prev[transition_key]["prev_wrong_correct"] += 1

    print(f"\nPhase transition accuracy (first step of new phase):")
    print(f"  {'Transition':>25s} | {'N':>6s} | {'Handoff Acc':>12s} | {'Prev OK→Acc':>12s} | {'Prev Fail→Acc':>14s}")
    print(f"  {'-'*25} | {'-'*6} | {'-'*12} | {'-'*12} | {'-'*14}")

    for tk in sorted(transition_acc.keys(), key=lambda x: -transition_acc[x]["total"]):
        ta = transition_acc[tk]
        if ta["total"] < 20:
            continue
        acc = ta["correct"] / ta["total"] * 100

        tgp = transition_given_prev[tk]
        prev_ok_acc = tgp["prev_correct_correct"] / tgp["prev_correct_total"] * 100 if tgp["prev_correct_total"] > 0 else 0
        prev_fail_acc = tgp["prev_wrong_correct"] / tgp["prev_wrong_total"] * 100 if tgp["prev_wrong_total"] > 0 else 0

        print(f"  {tk:>25s} | {ta['total']:>6d} | {acc:>11.1f}% | {prev_ok_acc:>11.1f}% | {prev_fail_acc:>13.1f}%")

    # Overall
    all_correct = sum(t["correct"] for t in transition_acc.values())
    all_total = sum(t["total"] for t in transition_acc.values())
    overall_acc = all_correct / all_total * 100 if all_total > 0 else 0
    print(f"\n  Overall transition handoff accuracy: {overall_acc:.1f}% (N={all_total})")


# ============================================================
# Analysis 4: Oracle sub-task boundary fix
# ============================================================
def analyze_oracle_subtask_fix(ms_data):
    """如果 oracle 修复每个 sub-task 的第一步, TSR 如何变化"""
    print("\n" + "=" * 80)
    print("Analysis 4: Oracle Sub-Task Boundary Fix Simulation")
    print("=" * 80)

    results = {
        "baseline": {"success": 0, "total": 0},
        "fix_phase_boundaries": {"success": 0, "total": 0},
        "fix_first_step_each_phase": {"success": 0, "total": 0},
        "fix_phase0_only": {"success": 0, "total": 0},
    }

    for ep_id, ep in ms_data.items():
        step_samples = ep.get("step_samples", [])
        if not step_samples:
            continue

        gt_types = [ss["gt_action_type"] for ss in step_samples]
        phases = segment_into_phases(gt_types)

        step_correct = []
        for ss in step_samples:
            samples = ss.get("samples", [])
            if samples:
                step_correct.append(samples[0].get("extract_match", False))
            else:
                step_correct.append(False)

        n_steps = len(step_correct)

        # Baseline: all correct
        baseline_success = all(step_correct)

        # Fix phase boundaries: make first step of each phase correct
        boundary_steps = set()
        for phase in phases:
            boundary_steps.add(phase["start"])
        fixed_boundary = [
            True if i in boundary_steps else step_correct[i]
            for i in range(n_steps)
        ]
        fix_boundary_success = all(fixed_boundary)

        # Fix first step of each phase (same as above)
        fix_first_each = fix_boundary_success

        # Fix only phase 0 (first phase entirely)
        phase0_steps = set(range(phases[0]["start"], phases[0]["end"] + 1))
        fixed_phase0 = [
            True if i in phase0_steps else step_correct[i]
            for i in range(n_steps)
        ]
        fix_phase0_success = all(fixed_phase0)

        for key, success in [
            ("baseline", baseline_success),
            ("fix_phase_boundaries", fix_boundary_success),
            ("fix_first_step_each_phase", fix_first_each),
            ("fix_phase0_only", fix_phase0_success),
        ]:
            results[key]["total"] += 1
            if success:
                results[key]["success"] += 1

    print(f"\n  {'Method':>30s} | {'TSR':>8s} | {'Delta':>8s}")
    print(f"  {'-'*30} | {'-'*8} | {'-'*8}")
    bl_tsr = results["baseline"]["success"] / results["baseline"]["total"]
    for key in ["baseline", "fix_phase0_only", "fix_phase_boundaries"]:
        tsr = results[key]["success"] / results[key]["total"]
        delta = tsr - bl_tsr
        print(f"  {key:>30s} | {tsr:>7.1%} | {delta:>+7.1%}")

    # By number of phases
    print(f"\n  By number of phases:")
    phase_group_results = defaultdict(lambda: {"bl_success": 0, "fix_success": 0, "total": 0})

    for ep_id, ep in ms_data.items():
        step_samples = ep.get("step_samples", [])
        if not step_samples:
            continue
        gt_types = [ss["gt_action_type"] for ss in step_samples]
        phases = segment_into_phases(gt_types)
        n_phases = min(len(phases), 6)

        step_correct = []
        for ss in step_samples:
            samples = ss.get("samples", [])
            if samples:
                step_correct.append(samples[0].get("extract_match", False))
            else:
                step_correct.append(False)

        boundary_steps = {phase["start"] for phase in phases}
        fixed = [True if i in boundary_steps else step_correct[i] for i in range(len(step_correct))]

        phase_group_results[n_phases]["total"] += 1
        if all(step_correct):
            phase_group_results[n_phases]["bl_success"] += 1
        if all(fixed):
            phase_group_results[n_phases]["fix_success"] += 1

    print(f"  {'#Phases':>8s} | {'N':>6s} | {'BL TSR':>8s} | {'Fix Boundaries':>15s} | {'Delta':>8s}")
    for np in sorted(phase_group_results.keys()):
        r = phase_group_results[np]
        if r["total"] < 10:
            continue
        bl = r["bl_success"] / r["total"]
        fx = r["fix_success"] / r["total"]
        label = f"{np}+" if np == 6 else str(np)
        print(f"  {label:>8s} | {r['total']:>6d} | {bl:>7.1%} | {fx:>14.1%} | {fx-bl:>+7.1%}")


# ============================================================
# Analysis 5: Logical sub-task decomposition
# ============================================================
def analyze_logical_subtasks(ms_data):
    """用 navigation vs interaction 的逻辑切分"""
    print("\n" + "=" * 80)
    print("Analysis 5: Logical Sub-Task (Nav vs Interact) Decomposition")
    print("=" * 80)

    nav_stats = {"total": 0, "correct": 0, "phases": 0, "perfect": 0}
    interact_stats = {"total": 0, "correct": 0, "phases": 0, "perfect": 0}

    for ep_id, ep in ms_data.items():
        step_samples = ep.get("step_samples", [])
        if not step_samples:
            continue

        gt_types = [ss["gt_action_type"] for ss in step_samples]
        subtasks = segment_into_logical_subtasks(gt_types)

        step_correct = []
        for ss in step_samples:
            samples = ss.get("samples", [])
            if samples:
                step_correct.append(samples[0].get("extract_match", False))
            else:
                step_correct.append(False)

        for st in subtasks:
            start, end = st["start"], st["end"]
            phase_correct = step_correct[start:end + 1]
            n_correct = sum(phase_correct)
            n_total = len(phase_correct)
            is_perfect = all(phase_correct)

            target = nav_stats if st["is_nav"] else interact_stats
            target["total"] += n_total
            target["correct"] += n_correct
            target["phases"] += 1
            target["perfect"] += int(is_perfect)

    print(f"\n  {'Sub-Task Type':>15s} | {'#Phases':>8s} | {'#Steps':>7s} | {'Step Acc':>9s} | {'Perfect%':>9s}")
    for name, st in [("Navigation", nav_stats), ("Interaction", interact_stats)]:
        acc = st["correct"] / st["total"] * 100 if st["total"] > 0 else 0
        perf = st["perfect"] / st["phases"] * 100 if st["phases"] > 0 else 0
        print(f"  {name:>15s} | {st['phases']:>8d} | {st['total']:>7d} | {acc:>8.1f}% | {perf:>8.1f}%")


# ============================================================
# Analysis 6: Key question — sub-task internal accuracy vs overall
# ============================================================
def analyze_subtask_vs_overall(ms_data):
    """核心对比: sub-task 内的 accuracy vs overall accuracy"""
    print("\n" + "=" * 80)
    print("Analysis 6: Sub-Task Internal Accuracy vs Overall Accuracy")
    print("=" * 80)

    # Overall step accuracy (all steps, independent eval)
    all_correct = 0
    all_total = 0

    # Within-phase accuracy (excluding phase boundaries)
    within_correct = 0
    within_total = 0

    # Phase boundary accuracy
    boundary_correct = 0
    boundary_total = 0

    # Single-type phase accuracy (homogeneous sub-task)
    single_type_correct = 0
    single_type_total = 0

    for ep_id, ep in ms_data.items():
        step_samples = ep.get("step_samples", [])
        if not step_samples:
            continue

        gt_types = [ss["gt_action_type"] for ss in step_samples]
        phases = segment_into_phases(gt_types)

        step_correct = []
        for ss in step_samples:
            samples = ss.get("samples", [])
            if samples:
                step_correct.append(samples[0].get("extract_match", False))
            else:
                step_correct.append(False)

        all_total += len(step_correct)
        all_correct += sum(step_correct)

        for pi, phase in enumerate(phases):
            for j in range(phase["start"], phase["end"] + 1):
                is_boundary = (j == phase["start"] and pi > 0)
                if is_boundary:
                    boundary_total += 1
                    if step_correct[j]:
                        boundary_correct += 1
                else:
                    within_total += 1
                    if step_correct[j]:
                        within_correct += 1

                # Single-type phase
                single_type_total += 1
                if step_correct[j]:
                    single_type_correct += 1

    overall_acc = all_correct / all_total * 100
    within_acc = within_correct / within_total * 100
    boundary_acc = boundary_correct / boundary_total * 100
    single_acc = single_type_correct / single_type_total * 100

    print(f"\n  KEY COMPARISON:")
    print(f"  {'Metric':>30s} | {'Accuracy':>9s} | {'N':>6s} | {'vs Overall':>11s}")
    print(f"  {'-'*30} | {'-'*9} | {'-'*6} | {'-'*11}")
    print(f"  {'Overall (all steps)':>30s} | {overall_acc:>8.1f}% | {all_total:>6d} | {'—':>11s}")
    print(f"  {'Within-phase (no boundary)':>30s} | {within_acc:>8.1f}% | {within_total:>6d} | {within_acc-overall_acc:>+10.1f}pp")
    print(f"  {'Phase boundary only':>30s} | {boundary_acc:>8.1f}% | {boundary_total:>6d} | {boundary_acc-overall_acc:>+10.1f}pp")

    print(f"\n  H1 test: Is within-phase accuracy > overall?")
    if within_acc > overall_acc:
        print(f"  ✅ YES: within-phase {within_acc:.1f}% > overall {overall_acc:.1f}% (+{within_acc-overall_acc:.1f}pp)")
        print(f"     → Sub-task 内确实更容易，phase boundary 是瓶颈")
    else:
        print(f"  ✗ NO: within-phase {within_acc:.1f}% ≤ overall {overall_acc:.1f}%")

    print(f"\n  H2 test: Is phase boundary the bottleneck?")
    if boundary_acc < within_acc:
        print(f"  ✅ YES: boundary {boundary_acc:.1f}% < within {within_acc:.1f}% ({boundary_acc-within_acc:.1f}pp)")
        print(f"     → Phase transition (handoff) 是显著瓶颈")
    else:
        print(f"  ✗ NO: boundary {boundary_acc:.1f}% ≥ within {within_acc:.1f}%")


# ============================================================
# Analysis 7: Predicted sub-task TSR vs actual full-task TSR
# ============================================================
def analyze_predicted_vs_actual_tsr(ms_data):
    """如果 sub-task 独立且给正确的 initial state, 预期 TSR 是多少？"""
    print("\n" + "=" * 80)
    print("Analysis 7: Predicted Sub-Task TSR vs Actual Full-Task TSR")
    print("=" * 80)

    # Group episodes by structure (number of phases)
    # For each group, compute:
    # 1. Actual full-task TSR
    # 2. Per-phase perfect rate
    # 3. Product of per-phase perfect rates (independence assumption)
    # 4. Product with oracle boundary fix

    by_structure = defaultdict(lambda: {"episodes": [], "full_success": 0})

    for ep_id, ep in ms_data.items():
        step_samples = ep.get("step_samples", [])
        if not step_samples:
            continue

        gt_types = [ss["gt_action_type"] for ss in step_samples]
        phases = segment_into_phases(gt_types)
        structure = tuple(p["type"] for p in phases)

        step_correct = []
        for ss in step_samples:
            samples = ss.get("samples", [])
            if samples:
                step_correct.append(samples[0].get("extract_match", False))
            else:
                step_correct.append(False)

        phase_perfects = []
        for phase in phases:
            start, end = phase["start"], phase["end"]
            phase_perfects.append(all(step_correct[start:end + 1]))

        by_structure[structure]["episodes"].append({
            "full_success": all(step_correct),
            "phase_perfects": phase_perfects,
        })
        if all(step_correct):
            by_structure[structure]["full_success"] += 1

    # Show top structures
    print(f"\nTop task structures (by frequency):")
    print(f"  {'Structure':>40s} | {'N':>5s} | {'Full TSR':>9s} | {'Phase Perfect Rates':>30s} | {'Product':>9s}")

    sorted_structures = sorted(by_structure.items(), key=lambda x: -len(x[1]["episodes"]))
    for structure, data in sorted_structures[:15]:
        n = len(data["episodes"])
        if n < 10:
            continue
        full_tsr = data["full_success"] / n

        # Per-phase perfect rate
        n_phases = len(structure)
        phase_rates = []
        for pi in range(n_phases):
            perfect_count = sum(1 for ep in data["episodes"] if ep["phase_perfects"][pi])
            rate = perfect_count / n
            phase_rates.append(rate)

        product = 1.0
        for pr in phase_rates:
            product *= pr

        struct_str = "→".join(structure)
        rates_str = " × ".join(f"{pr:.2f}" for pr in phase_rates)

        print(f"  {struct_str:>40s} | {n:>5d} | {full_tsr:>8.1%} | {rates_str:>30s} | {product:>8.1%}")

    # Summary: how much does decomposition help?
    total_episodes = sum(len(d["episodes"]) for d in by_structure.values())
    total_full_success = sum(d["full_success"] for d in by_structure.values())
    actual_tsr = total_full_success / total_episodes

    # Weighted product TSR
    product_tsr_sum = 0
    for structure, data in by_structure.items():
        n = len(data["episodes"])
        n_phases = len(structure)
        product = 1.0
        for pi in range(n_phases):
            rate = sum(1 for ep in data["episodes"] if ep["phase_perfects"][pi]) / n
            product *= rate
        product_tsr_sum += product * n

    predicted_product_tsr = product_tsr_sum / total_episodes

    print(f"\n  SUMMARY:")
    print(f"  Actual full-task TSR:          {actual_tsr:.1%}")
    print(f"  Predicted (product of phases): {predicted_product_tsr:.1%}")
    print(f"  Ratio:                         {predicted_product_tsr/actual_tsr:.3f}x" if actual_tsr > 0 else "")

    if abs(predicted_product_tsr - actual_tsr) / max(actual_tsr, 0.001) < 0.1:
        print(f"  → Phase errors are approximately INDEPENDENT (product ≈ actual)")
        print(f"  → Decomposing into sub-tasks gives accurate TSR prediction")
    elif predicted_product_tsr > actual_tsr:
        print(f"  → Phase errors are POSITIVELY CORRELATED (product > actual)")
        print(f"  → Some episodes are inherently harder (all phases fail together)")
    else:
        print(f"  → Phase errors are NEGATIVELY CORRELATED (product < actual)")


def main():
    print("Loading data...")
    trajs, ms_data = load_data()
    print(f"Loaded {len(trajs)} trajectories, {len(ms_data)} multi-sample episodes")

    phase_stats, boundary_stats, phase_records = analyze_phase_accuracy_c4c7(ms_data)
    analyze_subtask_tsr(ms_data)
    analyze_phase_transitions(ms_data)
    analyze_oracle_subtask_fix(ms_data)
    analyze_logical_subtasks(ms_data)
    analyze_subtask_vs_overall(ms_data)
    analyze_predicted_vs_actual_tsr(ms_data)

    # Save
    results = {
        "phase_stats": {k: {kk: vv for kk, vv in v.items() if kk != "lengths"}
                        for k, v in phase_stats.items()},
        "boundary_stats": boundary_stats,
    }
    with open(os.path.join(OUTPUT_DIR, "subtask_decomposition_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
