#!/usr/bin/env python3
"""
Eval C: Hard Cases 视觉特征分析

Question: What characterizes the 18.2% hard cases (both V2 and V3 wrong)?
Are they visually ambiguous, concentrated in specific UI types, or step positions?

Data sources:
  - Exp 1.1 (V3 grounding): per-sample results with greedy_correct
  - Exp 1.2 (V2 grounding): per-sample results with greedy_correct
  - Raw GUI-360 data: UI tree, control labels, rectangles
  - Exp 1.6 summary: 3,326 both_wrong samples

Output: outputs/eval_c/
"""

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def extract_step_index(sample_id: str) -> int:
    parts = sample_id.rsplit("_", 1)
    try:
        return int(parts[-1])
    except (ValueError, IndexError):
        return -1


def extract_trajectory_id(sample_id: str) -> str:
    parts = sample_id.rsplit("_", 1)
    return parts[0] if len(parts) == 2 else sample_id


def extract_domain(sample_id: str) -> str:
    return sample_id.split("_")[0]


def load_exp1_results(path: str) -> dict:
    """Load per-sample results from exp1 JSONL. Returns {sample_id: record}."""
    results = {}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            results[r["sample_id"]] = r
    return results


def load_raw_data_index():
    """Load raw GUI-360 test data index for UI tree analysis."""
    dataset_root = PROJECT_ROOT / "datasets" / "GUI-360" / "test" / "data"
    index = {}

    for jsonl_file in dataset_root.rglob("*.jsonl"):
        # Parse path: datasets/GUI-360/test/data/{domain}/{category}/success/{execution_id}.jsonl
        rel = jsonl_file.relative_to(dataset_root)
        parts = list(rel.parts)
        if len(parts) < 4:
            continue
        domain = parts[0]
        category = parts[1]
        execution_id = jsonl_file.stem

        with open(jsonl_file) as f:
            for i, line in enumerate(f):
                if not line.strip():
                    continue
                step_data = json.loads(line)
                step_idx = i + 1
                sample_id = f"{domain}_{category}_{execution_id}_{step_idx}"
                index[sample_id] = step_data

    return index


def analyze_hard_cases():
    """Main analysis of hard cases."""
    print("=" * 70)
    print("  Eval C: Hard Cases Analysis (Both V2 and V3 Wrong)")
    print("=" * 70)

    # Load V3 (exp1_1 K=1) and V2 (exp1_2 K=1) grounding results
    v3_path = PROJECT_ROOT / "outputs" / "exp1_1" / "results_K1.jsonl"
    v2_path = PROJECT_ROOT / "outputs" / "exp1_2" / "results_K1.jsonl"

    if not v3_path.exists() or not v2_path.exists():
        # Try K=5 (which includes greedy_correct)
        v3_path = PROJECT_ROOT / "outputs" / "exp1_1" / "results_K5.jsonl"
        v2_path = PROJECT_ROOT / "outputs" / "exp1_2" / "results_K5.jsonl"

    if not v3_path.exists() or not v2_path.exists():
        print(f"  Missing result files!")
        print(f"  V3: {v3_path} exists={v3_path.exists()}")
        print(f"  V2: {v2_path} exists={v2_path.exists()}")
        return

    v3_results = load_exp1_results(str(v3_path))
    v2_results = load_exp1_results(str(v2_path))

    print(f"\n  V3 results: {len(v3_results)} samples")
    print(f"  V2 results: {len(v2_results)} samples")

    # Find common samples
    common = set(v3_results.keys()) & set(v2_results.keys())
    print(f"  Common samples: {len(common)}")

    # Categorize
    both_correct = []
    v3_only = []
    v2_only = []
    both_wrong = []

    for sid in common:
        v3_ok = v3_results[sid].get("greedy_correct", False)
        v2_ok = v2_results[sid].get("greedy_correct", False)
        if v3_ok and v2_ok:
            both_correct.append(sid)
        elif v3_ok and not v2_ok:
            v3_only.append(sid)
        elif not v3_ok and v2_ok:
            v2_only.append(sid)
        else:
            both_wrong.append(sid)

    n = len(common)
    print(f"\n  Error Diversity:")
    print(f"    Both correct: {len(both_correct)} ({len(both_correct)/n:.1%})")
    print(f"    V3 only:      {len(v3_only)} ({len(v3_only)/n:.1%})")
    print(f"    V2 only:      {len(v2_only)} ({len(v2_only)/n:.1%})")
    print(f"    Both wrong:   {len(both_wrong)} ({len(both_wrong)/n:.1%})")

    # --- Analysis 1: Step position distribution ---
    print("\n" + "-" * 50)
    print("  C.1: Step Position Distribution")
    print("-" * 50)

    hw_steps = Counter(extract_step_index(sid) for sid in both_wrong)
    all_steps = Counter(extract_step_index(sid) for sid in common)

    print(f"\n  Hard cases by step position:")
    print(f"  {'Step':>6s} {'Hard':>6s} {'Total':>6s} {'Rate':>8s} {'Fraction':>10s}")
    for step_idx in sorted(all_steps.keys()):
        if step_idx > 15 or step_idx < 1:
            continue
        hw_n = hw_steps.get(step_idx, 0)
        total_n = all_steps[step_idx]
        rate = hw_n / total_n if total_n > 0 else 0
        frac = hw_n / len(both_wrong) if both_wrong else 0
        print(f"  {step_idx:>6d} {hw_n:>6d} {total_n:>6d} {rate:>8.1%} {frac:>10.1%}")

    step1_hw = hw_steps.get(1, 0)
    print(f"\n  Step 1 hard cases: {step1_hw}/{len(both_wrong)} ({step1_hw/len(both_wrong):.1%})")

    # --- Analysis 2: Domain distribution ---
    print("\n" + "-" * 50)
    print("  C.2: Domain Distribution")
    print("-" * 50)

    hw_domains = Counter(extract_domain(sid) for sid in both_wrong)
    all_domains = Counter(extract_domain(sid) for sid in common)

    for domain in sorted(all_domains.keys()):
        hw_n = hw_domains.get(domain, 0)
        total_n = all_domains[domain]
        rate = hw_n / total_n if total_n > 0 else 0
        print(f"  {domain:>8s}: {hw_n:>5d}/{total_n:>5d} ({rate:.1%})")

    # --- Analysis 3: V3 coordinate distance for hard cases ---
    print("\n" + "-" * 50)
    print("  C.3: V3 Prediction Distance from GT (Hard Cases vs Others)")
    print("-" * 50)

    hw_distances = []
    other_wrong_distances = []  # V3 wrong but V2 correct
    correct_distances = []

    for sid in common:
        v3_rec = v3_results[sid]
        pred_coord = v3_rec.get("greedy_coordinate")
        gt_bbox = v3_rec.get("gt_bbox") or v3_rec.get("gt_rectangle")

        if pred_coord is None or not isinstance(pred_coord, list) or len(pred_coord) < 2:
            continue
        if pred_coord[0] is None or pred_coord[1] is None:
            continue

        # Compute distance to bbox center
        if gt_bbox and isinstance(gt_bbox, dict):
            cx = (gt_bbox.get("left", 0) + gt_bbox.get("right", 0)) / 2
            cy = (gt_bbox.get("top", 0) + gt_bbox.get("bottom", 0)) / 2
        elif gt_bbox and isinstance(gt_bbox, (list, tuple)) and len(gt_bbox) == 4:
            cx = (gt_bbox[0] + gt_bbox[2]) / 2
            cy = (gt_bbox[1] + gt_bbox[3]) / 2
        else:
            continue

        try:
            dist = ((float(pred_coord[0]) - cx) ** 2 + (float(pred_coord[1]) - cy) ** 2) ** 0.5
        except (TypeError, ValueError):
            continue

        if sid in both_wrong:
            hw_distances.append(dist)
        elif not v3_results[sid].get("greedy_correct", False):
            other_wrong_distances.append(dist)
        else:
            correct_distances.append(dist)

    if hw_distances:
        print(f"\n  V3 prediction distance to GT bbox center:")
        print(f"    Hard cases:  mean={np.mean(hw_distances):.1f}px, "
              f"median={np.median(hw_distances):.1f}px, n={len(hw_distances)}")
    if other_wrong_distances:
        print(f"    V3 wrong only: mean={np.mean(other_wrong_distances):.1f}px, "
              f"median={np.median(other_wrong_distances):.1f}px, n={len(other_wrong_distances)}")
    if correct_distances:
        print(f"    Correct:     mean={np.mean(correct_distances):.1f}px, "
              f"median={np.median(correct_distances):.1f}px, n={len(correct_distances)}")

    # Distance distribution for hard cases
    if hw_distances:
        print(f"\n  Hard case distance distribution:")
        for thresh in [25, 50, 100, 200, 400]:
            cnt = sum(1 for d in hw_distances if d <= thresh)
            print(f"    ≤{thresh}px: {cnt}/{len(hw_distances)} ({cnt/len(hw_distances):.1%})")

    # --- Analysis 4: Agreement rate for hard cases ---
    print("\n" + "-" * 50)
    print("  C.4: V3 Agreement Rate for Hard Cases (from K=5/10)")
    print("-" * 50)

    for K in [5, 10]:
        v3_k_path = PROJECT_ROOT / "outputs" / "exp1_1" / f"results_K{K}.jsonl"
        if not v3_k_path.exists():
            continue

        v3_k_results = load_exp1_results(str(v3_k_path))
        hw_agreements = []
        correct_agreements = []
        v3wrong_agreements = []

        for sid in common:
            if sid not in v3_k_results:
                continue
            agree = v3_k_results[sid].get("agreement_rate", 0)
            if sid in both_wrong:
                hw_agreements.append(agree)
            elif v3_k_results[sid].get("greedy_correct", False):
                correct_agreements.append(agree)
            else:
                v3wrong_agreements.append(agree)

        print(f"\n  K={K}:")
        if hw_agreements:
            print(f"    Hard cases: mean agreement={np.mean(hw_agreements):.3f}, "
                  f"≥0.9={sum(1 for a in hw_agreements if a >= 0.9)/len(hw_agreements):.1%}, "
                  f"<0.5={sum(1 for a in hw_agreements if a < 0.5)/len(hw_agreements):.1%}")
        if correct_agreements:
            print(f"    Correct:    mean agreement={np.mean(correct_agreements):.3f}, "
                  f"≥0.9={sum(1 for a in correct_agreements if a >= 0.9)/len(correct_agreements):.1%}")

        # "Confident wrong" samples: agreement ≥ 0.9 but both models wrong
        confident_wrong = sum(1 for a in hw_agreements if a >= 0.9)
        print(f"    Confident wrong (agreement ≥0.9): {confident_wrong}")

    # --- Analysis 5: Load raw data for UI control labels ---
    print("\n" + "-" * 50)
    print("  C.5: UI Control Analysis (loading raw data...)")
    print("-" * 50)

    raw_index = load_raw_data_index()
    print(f"  Raw data index: {len(raw_index)} samples")

    # Check how many hard cases we can map to raw data
    mapped = 0
    hw_control_labels = Counter()
    hw_control_types = Counter()
    bc_control_labels = Counter()

    for sid in both_wrong:
        if sid in raw_index:
            mapped += 1
            step_data = raw_index[sid]
            action = step_data.get("step", {}).get("action", {})
            label = action.get("control_label", "unknown")
            ctrl_type = action.get("action_type", "unknown")
            hw_control_labels[label] += 1
            hw_control_types[ctrl_type] += 1

    for sid in both_correct[:5000]:  # Sample for comparison
        if sid in raw_index:
            step_data = raw_index[sid]
            action = step_data.get("step", {}).get("action", {})
            label = action.get("control_label", "unknown")
            bc_control_labels[label] += 1

    print(f"  Mapped hard cases to raw data: {mapped}/{len(both_wrong)}")

    if hw_control_types:
        print(f"\n  Hard case action_type distribution:")
        for ct, cnt in hw_control_types.most_common(10):
            print(f"    {ct}: {cnt}")

    if hw_control_labels:
        print(f"\n  Top 15 control_labels in hard cases:")
        for label, cnt in hw_control_labels.most_common(15):
            bc_cnt = bc_control_labels.get(label, 0)
            over_rep = (cnt / len(both_wrong)) / max(bc_cnt / len(both_correct), 1e-6)
            print(f"    {label[:50]:>50s}: {cnt:>4d} (over-representation: {over_rep:.1f}x)")

    # Save detailed results
    output_dir = PROJECT_ROOT / "outputs" / "eval_c"
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "n_common": len(common),
        "n_both_correct": len(both_correct),
        "n_v3_only": len(v3_only),
        "n_v2_only": len(v2_only),
        "n_both_wrong": len(both_wrong),
        "hard_case_rate": len(both_wrong) / len(common),
        "step1_hard_case_fraction": step1_hw / max(len(both_wrong), 1),
        "hard_case_distance_mean": float(np.mean(hw_distances)) if hw_distances else None,
        "hard_case_distance_median": float(np.median(hw_distances)) if hw_distances else None,
        "domain_hard_rates": {d: hw_domains.get(d, 0) / all_domains[d] for d in all_domains},
        "hard_case_sample_ids": both_wrong[:100],  # First 100 for manual inspection
    }

    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n  Summary saved to {output_dir / 'summary.json'}")

    # Save full hard case list
    with open(output_dir / "hard_case_ids.json", "w") as f:
        json.dump(both_wrong, f)
    print(f"  Hard case IDs saved to {output_dir / 'hard_case_ids.json'}")


def main():
    analyze_hard_cases()

    print("\n" + "=" * 70)
    print("  Eval C: CONCLUSIONS")
    print("=" * 70)
    print("""
  Key questions answered:
  1. Are hard cases concentrated at step 0?
  2. Are they domain-specific (Excel)?
  3. Are they close to GT (ambiguous) or far (systematic blind spot)?
  4. Do hard cases have high or low agreement rate?
  5. What UI control types are over-represented?

  Decision matrix:
  - Concentrated in specific UI controls → targeted data augmentation
  - Close to GT + high agreement → visual ambiguity, accept as ceiling
  - Far from GT → systematic failure, RL can potentially fix
  - Concentrated at step 0 → step-0 specific RL amplification
    """)


if __name__ == "__main__":
    main()
