"""
Analyze Pattern B (genuine multi-subtask) trajectories from GUI-360.

Identifies trajectories with real sub-goal decomposition (not just "open app"
transitions), then compares AR baseline vs subtask_isolated metrics on this
subset to isolate the true cross-subtask error accumulation effect.
"""

import json
import os
import re
from collections import Counter, defaultdict

# === Paths ===
GUI360_DATA = "/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/datasets/GUI-360/test/data"

AR_BASE_RESULTS = "/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/scripts/exp2/results/gui360/nostop_20260319_170859/ar_evaluation_results_20260319_182012.json"
AR_SFTV2_RESULTS = "/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/scripts/exp2/results/sft_v2/gui360/nostop_20260320_053216/ar_evaluation_results_20260320_055609.json"
ISO_BASE_RESULTS = "/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/scripts/exp2/results/gui360/subtask_isolated_base_20260320_081233/ar_evaluation_results_20260320_084147.json"
ISO_SFTV2_RESULTS = "/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/scripts/exp2/results/gui360/subtask_isolated_sftv2_20260320_081259/ar_evaluation_results_20260320_083447.json"


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


def load_gui360_subtask_info():
    """Load trajectory subtask structure from GUI-360 test data."""
    traj_subtasks = {}  # traj_id -> list of (subtask_str, step_count)

    for domain in ["ppt", "excel", "word"]:
        for category in ["in_app", "online", "search"]:
            data_dir = os.path.join(GUI360_DATA, domain, category, "success")
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
                        # Skip drag steps (filtered out in evaluator)
                        if action_func == "drag":
                            continue
                        # Skip steps without rectangle (filtered out in evaluator)
                        if not data.get("step", {}).get("action", {}).get("rectangle", {}):
                            continue
                        steps.append(subtask)

                if not steps:
                    continue

                # Group consecutive steps by subtask
                segments = []
                current = steps[0]
                count = 1
                for i in range(1, len(steps)):
                    if steps[i] != current and steps[i]:
                        segments.append((current, count))
                        current = steps[i]
                        count = 1
                    else:
                        count += 1
                segments.append((current, count))

                traj_subtasks[traj_id] = segments

    return traj_subtasks


def classify_trajectory(segments):
    """Classify a multi-subtask trajectory into Pattern A/B/C."""
    if len(segments) <= 1:
        return "single"

    subtask_strs = [s[0] for s in segments]
    has_open_app = any(is_open_app_subtask(s) for s in subtask_strs)
    non_open_subtasks = [s for s in subtask_strs if not is_open_app_subtask(s)]
    unique_non_open = len(set(non_open_subtasks))

    if not has_open_app and unique_non_open >= 2:
        return "B"  # Genuine multi-subtask decomposition
    elif has_open_app and unique_non_open <= 1:
        return "A"  # task + open_app
    elif has_open_app and unique_non_open >= 2:
        return "C"  # task + open_app + more
    else:
        return "other"


def load_eval_results(filepath):
    """Load evaluation results and index by trajectory_id."""
    with open(filepath) as f:
        data = json.load(f)

    # The results are in data["trajectory_results"] or data["results"]
    results = data.get("trajectory_results", data.get("results", []))
    by_id = {}
    for r in results:
        tid = r["trajectory_id"]
        by_id[tid] = r
    return by_id


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


def main():
    print("=" * 70)
    print("Pattern B Analysis: Genuine Multi-Subtask Trajectories")
    print("=" * 70)

    # Step 1: Load subtask info and classify
    print("\n[1] Loading GUI-360 subtask structure...")
    traj_subtasks = load_gui360_subtask_info()
    print(f"    Loaded {len(traj_subtasks)} trajectories")

    # Classify
    patterns = defaultdict(list)
    for tid, segments in traj_subtasks.items():
        pattern = classify_trajectory(segments)
        patterns[pattern].append(tid)

    print(f"\n[2] Classification:")
    print(f"    Single subtask: {len(patterns['single'])}")
    print(f"    Pattern A (task + open_app): {len(patterns['A'])}")
    print(f"    Pattern B (genuine multi-subtask): {len(patterns['B'])}")
    print(f"    Pattern C (task + open_app + more): {len(patterns['C'])}")
    print(f"    Other: {len(patterns['other'])}")

    # Show Pattern B subtask statistics
    pattern_b_ids = set(patterns["B"])
    b_segment_counts = [len(traj_subtasks[tid]) for tid in pattern_b_ids]
    b_step_counts = [sum(s[1] for s in traj_subtasks[tid]) for tid in pattern_b_ids]
    b_steps_per_subtask = []
    for tid in pattern_b_ids:
        for _, count in traj_subtasks[tid]:
            b_steps_per_subtask.append(count)

    print(f"\n[3] Pattern B Statistics:")
    print(f"    N trajectories: {len(pattern_b_ids)}")
    print(f"    Avg subtasks/traj: {sum(b_segment_counts)/len(b_segment_counts):.2f}")
    print(f"    Avg steps/traj: {sum(b_step_counts)/len(b_step_counts):.2f}")
    print(f"    Avg steps/subtask: {sum(b_steps_per_subtask)/len(b_steps_per_subtask):.2f}")
    print(f"    Subtask count distribution: {Counter(b_segment_counts).most_common()}")

    # Show a few examples
    print(f"\n[4] Pattern B Examples:")
    shown = 0
    for tid in sorted(pattern_b_ids):
        if shown >= 5:
            break
        segments = traj_subtasks[tid]
        if len(segments) >= 3:
            print(f"\n    {tid} ({sum(s[1] for s in segments)} steps, {len(segments)} subtasks):")
            for i, (desc, cnt) in enumerate(segments):
                short_desc = desc[:100] + "..." if len(desc) > 100 else desc
                print(f"      Subtask {i+1} ({cnt} steps): {short_desc}")
            shown += 1

    # Step 2: Map to evaluation trajectory IDs
    # Evaluation uses trajectory_id format like "ppt_search_filename"
    # Our IDs are "domain_category_filename" — should match

    print(f"\n[5] Loading evaluation results...")

    # Need to figure out the trajectory_id mapping
    # Load one result file and check format
    ar_base = load_eval_results(AR_BASE_RESULTS)
    ar_sftv2 = load_eval_results(AR_SFTV2_RESULTS)
    iso_base = load_eval_results(ISO_BASE_RESULTS)
    iso_sftv2 = load_eval_results(ISO_SFTV2_RESULTS)

    print(f"    AR base: {len(ar_base)} trajectories")
    print(f"    AR SFT v2: {len(ar_sftv2)} trajectories")
    print(f"    ISO base: {len(iso_base)} trajectories")
    print(f"    ISO SFT v2: {len(iso_sftv2)} trajectories")

    # Check ID format
    sample_eval_ids = list(ar_base.keys())[:5]
    print(f"    Sample eval IDs: {sample_eval_ids}")
    sample_data_ids = list(pattern_b_ids)[:5]
    print(f"    Sample data IDs: {sample_data_ids}")

    # Try to match. Eval IDs might be like "ppt_search_filename" or just "filename"
    # Let's check overlap
    direct_match = pattern_b_ids & set(ar_base.keys())
    print(f"    Direct match: {len(direct_match)}/{len(pattern_b_ids)}")

    if len(direct_match) < len(pattern_b_ids) * 0.5:
        # Try different ID format mapping
        # Data IDs: "domain_category_filename" -> eval might use different format
        # Try stripping to just filename
        data_to_eval = {}
        eval_ids = set(ar_base.keys())

        for data_id in pattern_b_ids:
            # Try progressively shorter prefixes
            if data_id in eval_ids:
                data_to_eval[data_id] = data_id
                continue

            parts = data_id.split("_", 2)
            if len(parts) >= 3:
                # Try domain_category_rest format variations
                candidate = parts[2]  # just filename
                if candidate in eval_ids:
                    data_to_eval[data_id] = candidate
                    continue

            # Try matching by suffix
            for eid in eval_ids:
                if eid.endswith(data_id.split("_")[-1]):
                    data_to_eval[data_id] = eid
                    break

        print(f"    Mapped: {len(data_to_eval)}/{len(pattern_b_ids)}")
        matched_eval_ids = set(data_to_eval.values())
    else:
        matched_eval_ids = direct_match

    # Also compute for ALL multi-subtask trajectories
    all_multi_ids = set()
    for p in ["A", "B", "C", "other"]:
        all_multi_ids.update(patterns[p])
    all_multi_eval = all_multi_ids & set(ar_base.keys())

    # Compute metrics for each subset
    print(f"\n{'=' * 70}")
    print("RESULTS")
    print(f"{'=' * 70}")

    subsets = {
        "All trajectories (N=3233)": set(ar_base.keys()),
        "Single subtask (N~2500)": (set(patterns["single"]) & set(ar_base.keys())),
        "All multi-subtask": all_multi_eval,
        "Pattern B only": matched_eval_ids,
    }

    for subset_name, eval_ids_subset in subsets.items():
        if not eval_ids_subset:
            print(f"\n{subset_name}: NO MATCHING IDs FOUND")
            continue

        ar_base_sub = [ar_base[tid] for tid in eval_ids_subset if tid in ar_base]
        ar_sftv2_sub = [ar_sftv2[tid] for tid in eval_ids_subset if tid in ar_sftv2]
        iso_base_sub = [iso_base[tid] for tid in eval_ids_subset if tid in iso_base]
        iso_sftv2_sub = [iso_sftv2[tid] for tid in eval_ids_subset if tid in iso_sftv2]

        m_ar_base = compute_metrics(ar_base_sub)
        m_ar_sftv2 = compute_metrics(ar_sftv2_sub)
        m_iso_base = compute_metrics(iso_base_sub)
        m_iso_sftv2 = compute_metrics(iso_sftv2_sub)

        print(f"\n--- {subset_name} ---")
        print(f"  N (AR/ISO): {m_ar_base.get('n_trajectories', 0)}/{m_iso_base.get('n_trajectories', 0)}")

        if m_ar_base and m_iso_base:
            print(f"\n  BASE MODEL:")
            print(f"    {'Metric':<30} {'AR baseline':>12} {'Subtask ISO':>12} {'Delta':>12}")
            print(f"    {'-'*66}")
            for metric, label in [("tsr", "TSR"), ("avg_progress_rate", "Progress Rate"),
                                   ("avg_scattered_progress_rate", "Scattered Progress"), ("step_accuracy", "Step Accuracy")]:
                v_ar = m_ar_base.get(metric, 0)
                v_iso = m_iso_base.get(metric, 0)
                delta = v_iso - v_ar
                print(f"    {label:<30} {v_ar:>11.4f} {v_iso:>12.4f} {delta:>+12.4f}")

        if m_ar_sftv2 and m_iso_sftv2:
            print(f"\n  SFT v2:")
            print(f"    {'Metric':<30} {'AR baseline':>12} {'Subtask ISO':>12} {'Delta':>12}")
            print(f"    {'-'*66}")
            for metric, label in [("tsr", "TSR"), ("avg_progress_rate", "Progress Rate"),
                                   ("avg_scattered_progress_rate", "Scattered Progress"), ("step_accuracy", "Step Accuracy")]:
                v_ar = m_ar_sftv2.get(metric, 0)
                v_iso = m_iso_sftv2.get(metric, 0)
                delta = v_iso - v_ar
                print(f"    {label:<30} {v_ar:>11.4f} {v_iso:>12.4f} {delta:>+12.4f}")

    # Domain breakdown for Pattern B
    if matched_eval_ids:
        print(f"\n--- Pattern B by Domain (SFT v2) ---")
        domain_groups = defaultdict(list)
        for tid in matched_eval_ids:
            domain = tid.split("_")[0] if "_" in tid else "unknown"
            domain_groups[domain].append(tid)

        for domain in sorted(domain_groups.keys()):
            tids = domain_groups[domain]
            ar_sub = [ar_sftv2[t] for t in tids if t in ar_sftv2]
            iso_sub = [iso_sftv2[t] for t in tids if t in iso_sftv2]
            m_ar = compute_metrics(ar_sub)
            m_iso = compute_metrics(iso_sub)
            if m_ar and m_iso:
                print(f"  {domain} (N={m_ar['n_trajectories']}): "
                      f"AR TSR={m_ar['tsr']:.4f}, ISO TSR={m_iso['tsr']:.4f}, "
                      f"Delta={m_iso['tsr']-m_ar['tsr']:+.4f} | "
                      f"AR Scattered={m_ar['avg_scattered_progress_rate']:.4f}, "
                      f"ISO Scattered={m_iso['avg_scattered_progress_rate']:.4f}, "
                      f"Delta={m_iso['avg_scattered_progress_rate']-m_ar['avg_scattered_progress_rate']:+.4f}")


if __name__ == "__main__":
    main()
