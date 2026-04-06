"""Part A: History-Screenshot Mismatch Analysis (Exp2e).

Analyzes existing C0 and F4 results to quantify how accuracy degrades
as the number of preceding wrong steps (within the same subtask) increases.
No new inference needed — pure analysis of existing data.
"""

import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np

# ---------------------------------------------------------------------------
# Add parent dir for imports
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXP2_DIR = os.path.dirname(SCRIPT_DIR)
if EXP2_DIR not in sys.path:
    sys.path.insert(0, EXP2_DIR)

from cognitive_interference_vllm import segment_by_subtask


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def load_c0_results(c0_path):
    """Load C0 results and return per-trajectory step results."""
    with open(c0_path) as f:
        data = json.load(f)
    return data["detailed_results"]


def load_f4_results(f4_path):
    """Load F4 JSONL results and group by trajectory."""
    by_traj = defaultdict(list)
    with open(f4_path) as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line.strip())
            by_traj[d["trajectory_id"]].append(d)
    # Sort each trajectory's steps by step_num
    for traj_id in by_traj:
        by_traj[traj_id].sort(key=lambda s: s["step_num"])
    return dict(by_traj)


def load_trajectories_raw(data_root, trajectory_ids):
    """Load raw trajectory data for subtask segmentation."""
    id_set = set(trajectory_ids)
    data_path = os.path.join(data_root, "data")
    trajectories = {}

    for domain in sorted(os.listdir(data_path)):
        domain_path = os.path.join(data_path, domain)
        if not os.path.isdir(domain_path):
            continue
        for category in sorted(os.listdir(domain_path)):
            success_path = os.path.join(domain_path, category, "success")
            if not os.path.isdir(success_path):
                continue
            for fname in sorted(os.listdir(success_path)):
                if not fname.endswith(".jsonl"):
                    continue
                file_stem = os.path.splitext(fname)[0]
                traj_id = f"{domain}_{category}_{file_stem}"
                if traj_id not in id_set:
                    continue

                fpath = os.path.join(success_path, fname)
                steps = []
                with open(fpath, "r") as f:
                    for line_num, line in enumerate(f, 1):
                        if not line.strip():
                            continue
                        try:
                            d = json.loads(line.strip())
                        except json.JSONDecodeError:
                            continue
                        action = d["step"]["action"]
                        if action.get("function", "") == "drag" or not action.get("rectangle", {}):
                            continue
                        sample_id = f"{traj_id}_{line_num}"
                        steps.append({
                            "sample_id": sample_id,
                            "subtask": d["step"].get("subtask", ""),
                        })

                trajectories[traj_id] = steps

    return trajectories


# ---------------------------------------------------------------------------
# Core analysis: compute preceding_wrong for each step
# ---------------------------------------------------------------------------

def annotate_preceding_wrong(traj_steps, raw_steps):
    """Annotate each step with preceding_wrong count within its subtask.

    traj_steps: list of step result dicts (sorted by step_num), with 'sample_id' and 'success'
    raw_steps: list of raw step dicts with 'sample_id' and 'subtask' fields
    Returns: list of annotated step dicts
    """
    # Build sample_id -> subtask mapping
    subtask_map = {}
    segments = segment_by_subtask(raw_steps)
    for seg_idx, (subtask_desc, seg_steps) in enumerate(segments):
        for s in seg_steps:
            subtask_map[s["sample_id"]] = seg_idx

    # Build sample_id -> success lookup from traj_steps
    success_map = {s["sample_id"]: s.get("success", False) for s in traj_steps}

    # Iterate through steps, counting preceding wrong within subtask
    annotated = []
    # Group by subtask index
    by_subtask = defaultdict(list)
    for s in traj_steps:
        seg_idx = subtask_map.get(s["sample_id"], -1)
        by_subtask[seg_idx].append(s)

    for seg_idx in sorted(by_subtask.keys()):
        seg_steps = by_subtask[seg_idx]
        preceding_wrong = 0
        for i, s in enumerate(seg_steps):
            step_annotated = {
                "sample_id": s["sample_id"],
                "success": s.get("success", False),
                "function_match": s.get("function_match", False),
                "args_match": s.get("args_match", False),
                "status_match": s.get("status_match", False),
                "preceding_wrong": preceding_wrong,
                "subtask_idx": seg_idx,
                "local_step": i,
                "subtask_length": len(seg_steps),
            }
            # Add domain/trajectory info if available
            for key in ["domain", "trajectory_id", "category", "step_num"]:
                if key in s:
                    step_annotated[key] = s[key]
            annotated.append(step_annotated)
            if not s.get("success", False):
                preceding_wrong += 1

    return annotated


# ---------------------------------------------------------------------------
# Metrics computation
# ---------------------------------------------------------------------------

def compute_bucket_metrics(annotated_steps, max_bucket=5):
    """Compute accuracy by preceding_wrong bucket.

    Buckets: 0, 1, 2, 3, 4, 5+ (where 5+ includes all >= 5)
    """
    buckets = defaultdict(list)
    for s in annotated_steps:
        pw = s["preceding_wrong"]
        bucket = str(pw) if pw < max_bucket else f"{max_bucket}+"
        buckets[bucket].append(s)

    results = {}
    for bucket in [str(i) for i in range(max_bucket)] + [f"{max_bucket}+"]:
        steps = buckets.get(bucket, [])
        n = len(steps)
        if n == 0:
            results[bucket] = {"n": 0, "step_accuracy": 0, "function_match": 0, "args_match": 0}
            continue
        results[bucket] = {
            "n": n,
            "step_accuracy": sum(1 for s in steps if s["success"]) / n,
            "function_match": sum(1 for s in steps if s["function_match"]) / n,
            "args_match": sum(1 for s in steps if s["args_match"]) / n,
        }

    return results


def compute_weighted_gradient(annotated_steps, max_x=5):
    """Compute weighted linear regression: accuracy vs preceding_wrong.

    Returns slope (gradient) and r-squared.
    """
    # Group by preceding_wrong (cap at max_x)
    by_pw = defaultdict(list)
    for s in annotated_steps:
        pw = min(s["preceding_wrong"], max_x)
        by_pw[pw].append(s)

    xs = []
    ys = []
    ws = []
    for pw in sorted(by_pw.keys()):
        steps = by_pw[pw]
        n = len(steps)
        acc = sum(1 for s in steps if s["success"]) / n
        xs.append(pw)
        ys.append(acc)
        ws.append(n)

    if len(xs) < 2:
        return 0.0, 0.0

    xs = np.array(xs, dtype=float)
    ys = np.array(ys, dtype=float)
    ws = np.array(ws, dtype=float)

    # Weighted least squares
    w_sum = ws.sum()
    x_mean = np.average(xs, weights=ws)
    y_mean = np.average(ys, weights=ws)
    num = np.sum(ws * (xs - x_mean) * (ys - y_mean))
    den = np.sum(ws * (xs - x_mean) ** 2)

    if den == 0:
        return 0.0, 0.0

    slope = num / den

    # R-squared
    ss_res = np.sum(ws * (ys - (y_mean + slope * (xs - x_mean))) ** 2)
    ss_tot = np.sum(ws * (ys - y_mean) ** 2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return float(slope), float(r_squared)


def compute_domain_breakdown(annotated_steps, max_bucket=5):
    """Compute accuracy gradient broken down by domain."""
    by_domain = defaultdict(list)
    for s in annotated_steps:
        by_domain[s.get("domain", "unknown")].append(s)

    results = {}
    for domain in sorted(by_domain.keys()):
        steps = by_domain[domain]
        gradient, r2 = compute_weighted_gradient(steps, max_bucket)
        buckets = compute_bucket_metrics(steps, max_bucket)
        results[domain] = {
            "n": len(steps),
            "gradient": gradient,
            "r_squared": r2,
            "buckets": buckets,
        }
    return results


def compute_subtask_length_breakdown(annotated_steps, max_bucket=5):
    """Compute accuracy gradient by subtask length category."""
    categories = {"short": [], "medium": [], "long": []}
    for s in annotated_steps:
        sl = s.get("subtask_length", 1)
        if sl <= 3:
            categories["short"].append(s)
        elif sl <= 7:
            categories["medium"].append(s)
        else:
            categories["long"].append(s)

    results = {}
    for cat, steps in categories.items():
        if not steps:
            results[cat] = {"n": 0, "gradient": 0, "r_squared": 0}
            continue
        gradient, r2 = compute_weighted_gradient(steps, max_bucket)
        results[cat] = {
            "n": len(steps),
            "gradient": gradient,
            "r_squared": r2,
        }
    return results


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def generate_report(c0_buckets, f4_buckets, c0_gradient, f4_gradient,
                    c0_r2, f4_r2, c0_domain, f4_domain,
                    c0_subtask_len, f4_subtask_len,
                    c0_first_step_acc, f4_first_step_acc,
                    c0_rest_acc, f4_rest_acc,
                    n_c0, n_f4):
    """Generate markdown report."""
    lines = []
    lines.append("# Exp2e Part A: History-Screenshot Mismatch Analysis\n")
    lines.append("## Accuracy by Number of Preceding Wrong Steps (within subtask)\n")
    lines.append("| preceding_wrong | n (C0) | C0 Acc | n (F4) | F4 Acc | Delta |")
    lines.append("|-----------------|--------|--------|--------|--------|-------|")

    all_buckets = sorted(set(list(c0_buckets.keys()) + list(f4_buckets.keys())),
                         key=lambda x: int(x.rstrip("+")) if x.rstrip("+").isdigit() else 99)

    for bucket in all_buckets:
        c0 = c0_buckets.get(bucket, {"n": 0, "step_accuracy": 0})
        f4 = f4_buckets.get(bucket, {"n": 0, "step_accuracy": 0})
        delta = f4["step_accuracy"] - c0["step_accuracy"] if c0["n"] > 0 and f4["n"] > 0 else 0
        delta_str = f"{delta:+.4f}" if c0["n"] > 0 and f4["n"] > 0 else "—"
        lines.append(
            f"| {bucket:>15} | {c0['n']:>6} | {c0['step_accuracy']:.4f} | "
            f"{f4['n']:>6} | {f4['step_accuracy']:.4f} | {delta_str} |"
        )

    lines.append(
        f"| {'Gradient':>15} | {'—':>6} | {c0_gradient:+.4f} | "
        f"{'—':>6} | {f4_gradient:+.4f} | {f4_gradient - c0_gradient:+.4f} |"
    )
    lines.append(
        f"| {'R-squared':>15} | {'—':>6} | {c0_r2:.4f} | "
        f"{'—':>6} | {f4_r2:.4f} | {'—':>5} |"
    )
    lines.append("")

    # First-step vs rest
    lines.append("## First-Step-in-Subtask vs Rest\n")
    lines.append("| Subset | n (C0) | C0 Acc | n (F4) | F4 Acc |")
    lines.append("|--------|--------|--------|--------|--------|")
    lines.append(
        f"| First step (pw=0) | {c0_buckets.get('0', {}).get('n', 0)} | "
        f"{c0_first_step_acc:.4f} | {f4_buckets.get('0', {}).get('n', 0)} | "
        f"{f4_first_step_acc:.4f} |"
    )
    lines.append(
        f"| Rest (pw>=1)      | {n_c0 - c0_buckets.get('0', {}).get('n', 0)} | "
        f"{c0_rest_acc:.4f} | {n_f4 - f4_buckets.get('0', {}).get('n', 0)} | "
        f"{f4_rest_acc:.4f} |"
    )
    lines.append("")

    # Domain breakdown
    lines.append("## Gradient by Domain\n")
    lines.append("| Domain | n (C0) | C0 Gradient | n (F4) | F4 Gradient |")
    lines.append("|--------|--------|-------------|--------|-------------|")
    for domain in ["excel", "ppt", "word"]:
        c0d = c0_domain.get(domain, {"n": 0, "gradient": 0})
        f4d = f4_domain.get(domain, {"n": 0, "gradient": 0})
        lines.append(
            f"| {domain} | {c0d['n']} | {c0d['gradient']:+.4f} | "
            f"{f4d['n']} | {f4d['gradient']:+.4f} |"
        )
    lines.append("")

    # Subtask length breakdown
    lines.append("## Gradient by Subtask Length\n")
    lines.append("| Length | n (C0) | C0 Gradient | n (F4) | F4 Gradient |")
    lines.append("|--------|--------|-------------|--------|-------------|")
    for cat in ["short", "medium", "long"]:
        label = {"short": "Short (<=3)", "medium": "Medium (4-7)", "long": "Long (8+)"}[cat]
        c0s = c0_subtask_len.get(cat, {"n": 0, "gradient": 0})
        f4s = f4_subtask_len.get(cat, {"n": 0, "gradient": 0})
        lines.append(
            f"| {label} | {c0s['n']} | {c0s['gradient']:+.4f} | "
            f"{f4s['n']} | {f4s['gradient']:+.4f} |"
        )
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Exp2e Part A: History-Screenshot Mismatch Analysis")
    parser.add_argument("--c0_results", type=str, required=True,
                        help="Path to C0 results JSON")
    parser.add_argument("--f4_results", type=str, required=True,
                        help="Path to F4 JSONL")
    parser.add_argument("--trajectory_ids", type=str, required=True,
                        help="Path to pattern_b_ids.json")
    parser.add_argument("--data_root", type=str, required=True,
                        help="GUI-360 test data root")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("Exp2e Part A: History-Screenshot Mismatch Analysis")
    print("=" * 60)

    # Load trajectory IDs
    with open(args.trajectory_ids) as f:
        traj_ids = json.load(f)
    print(f"Trajectories: {len(traj_ids)}")

    # Load raw trajectories for subtask segmentation
    print("Loading raw trajectories for subtask segmentation...")
    raw_trajs = load_trajectories_raw(args.data_root, traj_ids)
    print(f"  Loaded {len(raw_trajs)} trajectories")

    # Load C0 results
    print("Loading C0 results...")
    c0_trajs = load_c0_results(args.c0_results)
    print(f"  {len(c0_trajs)} trajectories, "
          f"{sum(len(t['step_results']) for t in c0_trajs)} steps")

    # Load F4 results
    print("Loading F4 results...")
    f4_by_traj = load_f4_results(args.f4_results)
    print(f"  {len(f4_by_traj)} trajectories, "
          f"{sum(len(s) for s in f4_by_traj.values())} steps")

    # Annotate C0 steps with preceding_wrong
    print("\nAnnotating C0 steps...")
    c0_annotated = []
    for traj in c0_trajs:
        traj_id = traj["trajectory_id"]
        raw = raw_trajs.get(traj_id)
        if not raw:
            continue
        # Convert step_results to have consistent keys
        steps = []
        for sr in traj["step_results"]:
            steps.append({
                "sample_id": sr["sample_id"],
                "success": sr.get("success", False),
                "function_match": sr.get("function_match", False),
                "args_match": sr.get("args_match", False),
                "status_match": sr.get("status_match", False),
                "domain": traj["domain"],
                "trajectory_id": traj_id,
                "category": traj["category"],
                "step_num": sr["step_num"],
            })
        annotated = annotate_preceding_wrong(steps, raw)
        c0_annotated.extend(annotated)
    print(f"  Annotated {len(c0_annotated)} C0 steps")

    # Annotate F4 steps with preceding_wrong
    print("Annotating F4 steps...")
    f4_annotated = []
    for traj_id, steps in f4_by_traj.items():
        raw = raw_trajs.get(traj_id)
        if not raw:
            continue
        annotated = annotate_preceding_wrong(steps, raw)
        f4_annotated.extend(annotated)
    print(f"  Annotated {len(f4_annotated)} F4 steps")

    # Compute bucket metrics
    print("\nComputing metrics...")
    c0_buckets = compute_bucket_metrics(c0_annotated)
    f4_buckets = compute_bucket_metrics(f4_annotated)

    c0_gradient, c0_r2 = compute_weighted_gradient(c0_annotated)
    f4_gradient, f4_r2 = compute_weighted_gradient(f4_annotated)

    c0_domain = compute_domain_breakdown(c0_annotated)
    f4_domain = compute_domain_breakdown(f4_annotated)

    c0_subtask_len = compute_subtask_length_breakdown(c0_annotated)
    f4_subtask_len = compute_subtask_length_breakdown(f4_annotated)

    # First-step vs rest accuracy
    c0_first = [s for s in c0_annotated if s["preceding_wrong"] == 0]
    c0_rest = [s for s in c0_annotated if s["preceding_wrong"] > 0]
    f4_first = [s for s in f4_annotated if s["preceding_wrong"] == 0]
    f4_rest = [s for s in f4_annotated if s["preceding_wrong"] > 0]

    c0_first_acc = sum(1 for s in c0_first if s["success"]) / len(c0_first) if c0_first else 0
    c0_rest_acc = sum(1 for s in c0_rest if s["success"]) / len(c0_rest) if c0_rest else 0
    f4_first_acc = sum(1 for s in f4_first if s["success"]) / len(f4_first) if f4_first else 0
    f4_rest_acc = sum(1 for s in f4_rest if s["success"]) / len(f4_rest) if f4_rest else 0

    # Print summary
    print(f"\n{'='*60}")
    print("Results Summary")
    print(f"{'='*60}")
    print(f"C0: gradient={c0_gradient:+.4f}, R2={c0_r2:.4f}")
    print(f"F4: gradient={f4_gradient:+.4f}, R2={f4_r2:.4f}")
    print(f"C0 first-step acc: {c0_first_acc:.4f} ({len(c0_first)} steps)")
    print(f"C0 rest acc: {c0_rest_acc:.4f} ({len(c0_rest)} steps)")
    print(f"F4 first-step acc: {f4_first_acc:.4f} ({len(f4_first)} steps)")
    print(f"F4 rest acc: {f4_rest_acc:.4f} ({len(f4_rest)} steps)")

    # Save JSON
    analysis = {
        "c0": {
            "n_steps": len(c0_annotated),
            "buckets": c0_buckets,
            "gradient": c0_gradient,
            "r_squared": c0_r2,
            "by_domain": c0_domain,
            "by_subtask_length": c0_subtask_len,
            "first_step_accuracy": c0_first_acc,
            "rest_accuracy": c0_rest_acc,
        },
        "f4": {
            "n_steps": len(f4_annotated),
            "buckets": f4_buckets,
            "gradient": f4_gradient,
            "r_squared": f4_r2,
            "by_domain": f4_domain,
            "by_subtask_length": f4_subtask_len,
            "first_step_accuracy": f4_first_acc,
            "rest_accuracy": f4_rest_acc,
        },
    }

    json_path = os.path.join(args.output_dir, "mismatch_analysis.json")
    with open(json_path, "w") as f:
        json.dump(analysis, f, indent=2, default=str)
    print(f"\nSaved analysis: {json_path}")

    # Generate and save report
    report = generate_report(
        c0_buckets, f4_buckets,
        c0_gradient, f4_gradient,
        c0_r2, f4_r2,
        c0_domain, f4_domain,
        c0_subtask_len, f4_subtask_len,
        c0_first_acc, f4_first_acc,
        c0_rest_acc, f4_rest_acc,
        len(c0_annotated), len(f4_annotated),
    )

    report_path = os.path.join(args.output_dir, "MISMATCH_ANALYSIS_REPORT.md")
    with open(report_path, "w") as f:
        f.write(report)
    print(f"Saved report: {report_path}")
    print(f"\n{report}")


if __name__ == "__main__":
    main()
