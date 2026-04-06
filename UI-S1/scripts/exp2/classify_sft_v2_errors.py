#!/usr/bin/env python3
"""
Classify SFT v2 (no-stop) evaluation errors and compare with base model distribution.

Error classification for failed steps:
  1. Stuck/repeating: same predicted coordinates as previous step (cascade artifact)
  2. Type mismatch: predicted action type != GT action type
  3. Near miss (<50px): type matches, coordinate distance < 50px
  4. Coord error (>=50px): type matches but coordinates significantly wrong

Also breaks down by domain, computes distance distributions, and samples
non-stuck coord errors with their thought text for qualitative analysis.
"""

import json
import math
import random
import sys
from collections import Counter, defaultdict

# ── Configuration ──────────────────────────────────────────────────────────
RESULTS_FILE = (
    "/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/scripts/exp2/results/"
    "sft_v2/gui360/nostop_20260320_053216/"
    "ar_evaluation_results_20260320_055609.json"
)
NEAR_MISS_THRESHOLD = 50.0  # pixels
RANDOM_SEED = 42
NUM_SAMPLES = 20

# ── Helpers ────────────────────────────────────────────────────────────────

def get_coord(args):
    """Extract (x, y) coordinate from predicted/gt args. Returns None if missing."""
    coord = args.get("coordinate")
    if coord and len(coord) == 2 and coord[0] is not None and coord[1] is not None:
        return (float(coord[0]), float(coord[1]))
    # For drag actions, try start_coordinate
    coord = args.get("start_coordinate")
    if coord and len(coord) == 2 and coord[0] is not None and coord[1] is not None:
        return (float(coord[0]), float(coord[1]))
    return None


def coord_distance(c1, c2):
    """Euclidean distance between two (x,y) tuples."""
    if c1 is None or c2 is None:
        return None
    return math.sqrt((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2)


def get_domain_from_sample_id(sample_id):
    """Extract domain (ppt/excel/word) from sample_id prefix."""
    if sample_id.startswith("ppt"):
        return "ppt"
    elif sample_id.startswith("excel"):
        return "excel"
    elif sample_id.startswith("word"):
        return "word"
    return "unknown"


def histogram_line(label, count, total, bar_width=40):
    pct = count / total * 100 if total > 0 else 0
    bar = "#" * int(pct / 100 * bar_width)
    return f"  {label:>10s}: {count:5d} ({pct:5.1f}%) |{bar}"


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    print("=" * 80)
    print("SFT v2 (no-stop) Error Classification Analysis")
    print("=" * 80)
    print(f"\nLoading results from:\n  {RESULTS_FILE}")

    with open(RESULTS_FILE) as f:
        data = json.load(f)

    trajectories = data["trajectory_results"]
    all_steps = data["detailed_results"]
    stats = data["statistics"]

    print(f"\n  Total trajectories : {len(trajectories)}")
    print(f"  Total steps        : {len(all_steps)}")
    print(f"  Step success rate  : {stats['step_success_rate']:.2f}%")

    # ── 1. Count failed steps ──────────────────────────────────────────────
    total_steps = len(all_steps)
    failed_steps = [s for s in all_steps if not s["success"]]
    success_steps = total_steps - len(failed_steps)

    print(f"\n{'─' * 80}")
    print("1. OVERALL STEP COUNTS")
    print(f"{'─' * 80}")
    print(f"  Successful steps : {success_steps:6d} ({success_steps/total_steps*100:.1f}%)")
    print(f"  Failed steps     : {len(failed_steps):6d} ({len(failed_steps)/total_steps*100:.1f}%)")
    print(f"  Total steps      : {total_steps:6d}")

    # ── 2. Classify failed steps ───────────────────────────────────────────
    # Build a map from trajectory_id -> ordered list of steps for stuck detection
    traj_steps = defaultdict(list)
    for t in trajectories:
        for s in t["step_results"]:
            traj_steps[t["trajectory_id"]].append(s)

    # For each trajectory, sort steps by step_num
    for tid in traj_steps:
        traj_steps[tid].sort(key=lambda s: s["step_num"])

    # Build trajectory-step lookup: (trajectory_id, step_num) -> previous step
    prev_step_map = {}
    traj_domain_map = {}
    for t in trajectories:
        traj_domain_map[t["trajectory_id"]] = t["domain"]
        sorted_steps = sorted(t["step_results"], key=lambda s: s["step_num"])
        for i in range(1, len(sorted_steps)):
            # Key by sample_id of current step
            prev_step_map[sorted_steps[i]["sample_id"]] = sorted_steps[i - 1]

    # Classify
    categories = {
        "stuck_repeating": [],
        "type_mismatch": [],
        "near_miss": [],
        "coord_error": [],
        "no_coord": [],   # steps where we can't extract coordinates
        "parse_error": [], # empty function / null coords
    }

    # Also track per-domain
    domain_categories = defaultdict(lambda: defaultdict(list))

    all_distances = []  # (distance, step_dict) for all failed, non-stuck, type-matched steps

    for step in failed_steps:
        sample_id = step["sample_id"]
        domain = get_domain_from_sample_id(sample_id)

        pred_func = step["predicted_function"]
        gt_func = step["ground_truth_function"]
        pred_coord = get_coord(step["predicted_args"])
        gt_coord = get_coord(step["ground_truth_args"])

        # Parse error: empty function or null coordinates
        if pred_func == "" or pred_coord is None:
            cat = "parse_error"
            categories[cat].append(step)
            domain_categories[domain][cat].append(step)
            continue

        # Check stuck/repeating: same predicted coordinate as previous step
        prev = prev_step_map.get(sample_id)
        is_stuck = False
        if prev is not None:
            prev_coord = get_coord(prev["predicted_args"])
            if pred_coord is not None and prev_coord is not None:
                # Consider "stuck" if coordinates are within 1px (floating point tolerance)
                if coord_distance(pred_coord, prev_coord) < 1.0:
                    is_stuck = True

        if is_stuck:
            cat = "stuck_repeating"
            categories[cat].append(step)
            domain_categories[domain][cat].append(step)
            continue

        # Type mismatch
        if pred_func != gt_func:
            cat = "type_mismatch"
            categories[cat].append(step)
            domain_categories[domain][cat].append(step)
            continue

        # From here: type matches, not stuck. Compute distance.
        if gt_coord is None:
            # GT has no coordinate (e.g., summary/type with text-only matching)
            cat = "no_coord"
            categories[cat].append(step)
            domain_categories[domain][cat].append(step)
            continue

        dist = coord_distance(pred_coord, gt_coord)

        if dist is not None and dist < NEAR_MISS_THRESHOLD:
            cat = "near_miss"
            categories[cat].append(step)
            domain_categories[domain][cat].append(step)
        else:
            cat = "coord_error"
            categories[cat].append(step)
            domain_categories[domain][cat].append(step)
            if dist is not None:
                all_distances.append((dist, step))

    # ── Print classification results ───────────────────────────────────────
    n_failed = len(failed_steps)

    print(f"\n{'─' * 80}")
    print("2. ERROR CLASSIFICATION (failed steps only)")
    print(f"{'─' * 80}")

    cat_labels = {
        "stuck_repeating": "Stuck/repeating (cascade artifact)",
        "type_mismatch": "Type mismatch (action type differs)",
        "near_miss": f"Near miss (<{NEAR_MISS_THRESHOLD:.0f}px, type matches)",
        "coord_error": f"Coordinate error (>={NEAR_MISS_THRESHOLD:.0f}px, type matches)",
        "no_coord": "No coordinate in GT (text-based mismatch)",
        "parse_error": "Parse error (empty func / null coords)",
    }

    print(f"\n  {'Category':<45s} {'Count':>6s} {'% of Failed':>11s} {'% of All':>9s}")
    print(f"  {'─' * 75}")
    for cat in ["stuck_repeating", "type_mismatch", "near_miss", "coord_error", "no_coord", "parse_error"]:
        n = len(categories[cat])
        pct_fail = n / n_failed * 100 if n_failed > 0 else 0
        pct_all = n / total_steps * 100
        print(f"  {cat_labels[cat]:<45s} {n:6d} {pct_fail:10.1f}% {pct_all:8.1f}%")

    total_classified = sum(len(v) for v in categories.values())
    print(f"  {'─' * 75}")
    print(f"  {'TOTAL':<45s} {total_classified:6d} {total_classified/n_failed*100:10.1f}%")
    assert total_classified == n_failed, f"Classification mismatch: {total_classified} != {n_failed}"

    # ── Compare with base model ────────────────────────────────────────────
    print(f"\n{'─' * 80}")
    print("2b. COMPARISON WITH BASE MODEL ERROR DISTRIBUTION")
    print(f"{'─' * 80}")

    # Map SFT categories to base model categories for comparison
    # Base: Grounding 76.5%, Vocabulary 14.6%, Planning 2.2%, Context 6.6%
    # Base grounding sub: 66.9% stuck, 17.7% wrong target, 6.3% hallucinated, 4.6% near miss

    sft_stuck_pct = len(categories["stuck_repeating"]) / n_failed * 100
    sft_type_pct = len(categories["type_mismatch"]) / n_failed * 100
    sft_near_pct = len(categories["near_miss"]) / n_failed * 100
    sft_coord_pct = len(categories["coord_error"]) / n_failed * 100
    sft_nocoord_pct = len(categories["no_coord"]) / n_failed * 100
    sft_parse_pct = len(categories["parse_error"]) / n_failed * 100

    # Grounding-like = stuck + coord_error + near_miss
    sft_grounding_like = sft_stuck_pct + sft_coord_pct + sft_near_pct

    print(f"\n  {'Metric':<45s} {'Base':>8s} {'SFT v2':>8s}")
    print(f"  {'─' * 65}")
    print(f"  {'Total failed steps':<45s} {'2608':>8s} {n_failed:>8d}")
    print(f"  {'Grounding-like (stuck+coord+near)':<45s} {'76.5%':>8s} {sft_grounding_like:>7.1f}%")
    print(f"    {'- Stuck/repeating':<43s} {'51.2%':>8s} {sft_stuck_pct:>7.1f}%")
    print(f"    {'- Coord error (wrong target / hallucinated)':<43s} {'18.4%':>8s} {sft_coord_pct:>7.1f}%")
    print(f"    {'- Near miss':<43s} {'3.5%':>8s} {sft_near_pct:>7.1f}%")
    print(f"  {'Type mismatch (~ Vocabulary)':<45s} {'14.6%':>8s} {sft_type_pct:>7.1f}%")
    print(f"  {'No-coord / text mismatch':<45s} {'---':>8s} {sft_nocoord_pct:>7.1f}%")
    print(f"  {'Parse error':<45s} {'---':>8s} {sft_parse_pct:>7.1f}%")

    # ── 3. Distance distribution for coord errors ──────────────────────────
    print(f"\n{'─' * 80}")
    print("3. DISTANCE DISTRIBUTION FOR COORDINATE ERRORS (non-stuck, type-matched)")
    print(f"{'─' * 80}")

    distances = [d for d, _ in all_distances]

    if distances:
        distances_sorted = sorted(distances)
        n_dist = len(distances_sorted)
        mean_dist = sum(distances_sorted) / n_dist
        median_dist = distances_sorted[n_dist // 2]
        p25 = distances_sorted[int(n_dist * 0.25)]
        p75 = distances_sorted[int(n_dist * 0.75)]
        p90 = distances_sorted[int(n_dist * 0.90)]
        p95 = distances_sorted[int(n_dist * 0.95)]

        print(f"\n  N = {n_dist}")
        print(f"  Mean distance   : {mean_dist:.1f} px")
        print(f"  Median distance : {median_dist:.1f} px")
        print(f"  25th percentile : {p25:.1f} px")
        print(f"  75th percentile : {p75:.1f} px")
        print(f"  90th percentile : {p90:.1f} px")
        print(f"  95th percentile : {p95:.1f} px")
        print(f"  Min             : {distances_sorted[0]:.1f} px")
        print(f"  Max             : {distances_sorted[-1]:.1f} px")

        # Histogram bins
        bins = [
            ("< 25 px", 0, 25),
            ("25-50 px", 25, 50),
            ("50-100 px", 50, 100),
            ("100-200 px", 100, 200),
            ("200-500 px", 200, 500),
            ("> 500 px", 500, float("inf")),
        ]

        print(f"\n  Distance Histogram:")
        print(f"  {'Bin':>12s} {'Count':>6s} {'%':>7s}  Bar")
        print(f"  {'─' * 60}")
        for label, lo, hi in bins:
            count = sum(1 for d in distances if lo <= d < hi)
            pct = count / n_dist * 100
            bar = "#" * int(pct / 2)
            print(f"  {label:>12s} {count:6d} {pct:6.1f}%  |{bar}")
    else:
        print("  No coordinate errors with computable distances found.")

    # Also show distances including near misses for the full picture
    print(f"\n  (Note: near misses have distance < {NEAR_MISS_THRESHOLD:.0f}px and are counted separately above)")
    near_dists = []
    for step in categories["near_miss"]:
        pc = get_coord(step["predicted_args"])
        gc = get_coord(step["ground_truth_args"])
        d = coord_distance(pc, gc)
        if d is not None:
            near_dists.append(d)
    if near_dists:
        print(f"  Near miss distances: N={len(near_dists)}, mean={sum(near_dists)/len(near_dists):.1f}px, "
              f"median={sorted(near_dists)[len(near_dists)//2]:.1f}px")

    # ── 4. Breakdown by domain ─────────────────────────────────────────────
    print(f"\n{'─' * 80}")
    print("4. BREAKDOWN BY DOMAIN")
    print(f"{'─' * 80}")

    # Count total and failed steps per domain
    domain_total = Counter()
    domain_failed = Counter()
    domain_success = Counter()
    for step in all_steps:
        d = get_domain_from_sample_id(step["sample_id"])
        domain_total[d] += 1
        if step["success"]:
            domain_success[d] += 1
        else:
            domain_failed[d] += 1

    for domain in ["ppt", "excel", "word"]:
        tot = domain_total[domain]
        fail = domain_failed[domain]
        succ = domain_success[domain]
        print(f"\n  === {domain.upper()} ===")
        print(f"  Total steps: {tot}, Successful: {succ} ({succ/tot*100:.1f}%), Failed: {fail} ({fail/tot*100:.1f}%)")

        if fail == 0:
            continue

        print(f"\n  {'Category':<45s} {'Count':>6s} {'% of Domain Failed':>18s}")
        print(f"  {'─' * 73}")
        for cat in ["stuck_repeating", "type_mismatch", "near_miss", "coord_error", "no_coord", "parse_error"]:
            n = len(domain_categories[domain][cat])
            pct = n / fail * 100
            print(f"  {cat_labels[cat]:<45s} {n:6d} {pct:17.1f}%")

        domain_total_classified = sum(len(domain_categories[domain][c]) for c in categories)
        print(f"  {'─' * 73}")
        print(f"  {'TOTAL':<45s} {domain_total_classified:6d} {domain_total_classified/fail*100:17.1f}%")

    # ── 4b. Domain comparison table ────────────────────────────────────────
    print(f"\n  --- Domain Comparison Summary (% of failed in each domain) ---")
    print(f"  {'Category':<30s} {'PPT':>8s} {'Excel':>8s} {'Word':>8s} {'All':>8s}")
    print(f"  {'─' * 60}")
    for cat in ["stuck_repeating", "type_mismatch", "near_miss", "coord_error", "no_coord", "parse_error"]:
        vals = []
        for domain in ["ppt", "excel", "word"]:
            n = len(domain_categories[domain][cat])
            f = domain_failed[domain]
            vals.append(n / f * 100 if f > 0 else 0)
        all_pct = len(categories[cat]) / n_failed * 100
        short_label = cat.replace("_", " ").title()
        print(f"  {short_label:<30s} {vals[0]:7.1f}% {vals[1]:7.1f}% {vals[2]:7.1f}% {all_pct:7.1f}%")

    # ── 5. Qualitative analysis: sample non-stuck coord errors ─────────────
    print(f"\n{'─' * 80}")
    print("5. QUALITATIVE ANALYSIS: SAMPLED NON-STUCK COORDINATE ERRORS")
    print(f"{'─' * 80}")
    print(f"   (Sampling {NUM_SAMPLES} random non-stuck, type-matched coord errors >= {NEAR_MISS_THRESHOLD:.0f}px)")

    random.seed(RANDOM_SEED)

    if len(all_distances) >= NUM_SAMPLES:
        sampled = random.sample(all_distances, NUM_SAMPLES)
    else:
        sampled = all_distances

    # Sort by distance for readability
    sampled.sort(key=lambda x: x[0], reverse=True)

    for i, (dist, step) in enumerate(sampled):
        pred_coord = get_coord(step["predicted_args"])
        gt_coord = get_coord(step["ground_truth_args"])
        gt_rect = step.get("ground_truth_rect", {})
        thought = step.get("thoughts", "") or ""
        domain = get_domain_from_sample_id(step["sample_id"])

        # Extract just the reasoning part if there's a thought chain
        # The thoughts field seems to contain the raw tool_call JSON
        # Check if there's actual reasoning before the tool call
        thought_display = thought.strip()
        if len(thought_display) > 500:
            thought_display = thought_display[:500] + "..."

        print(f"\n  ── Sample {i+1}/{len(sampled)} ──")
        print(f"  Sample ID  : {step['sample_id']}")
        print(f"  Domain     : {domain}")
        print(f"  Action     : {step['predicted_function']}")
        print(f"  Pred coord : ({pred_coord[0]:.1f}, {pred_coord[1]:.1f})")
        print(f"  GT coord   : ({gt_coord[0]:.1f}, {gt_coord[1]:.1f})")
        print(f"  GT rect    : left={gt_rect.get('left','?')}, top={gt_rect.get('top','?')}, "
              f"right={gt_rect.get('right','?')}, bottom={gt_rect.get('bottom','?')}")
        if gt_rect and all(k in gt_rect for k in ['left', 'top', 'right', 'bottom']):
            rect_w = gt_rect['right'] - gt_rect['left']
            rect_h = gt_rect['bottom'] - gt_rect['top']
            print(f"  GT rect sz : {rect_w}x{rect_h} px")
            # Is predicted coord inside GT rect?
            in_rect = (gt_rect['left'] <= pred_coord[0] <= gt_rect['right'] and
                       gt_rect['top'] <= pred_coord[1] <= gt_rect['bottom'])
            print(f"  Pred in GT rect? : {'YES' if in_rect else 'NO'}")
        print(f"  Distance   : {dist:.1f} px")

        # Check if pred_args has content/text mismatch too
        pred_text = step["predicted_args"].get("text", "")
        gt_text = step["ground_truth_args"].get("text", "")
        if pred_text or gt_text:
            print(f"  Pred text  : {str(pred_text)[:100]}")
            print(f"  GT text    : {str(gt_text)[:100]}")

        print(f"  Thought    :")
        for line in thought_display.split("\n"):
            print(f"    {line}")

    # ── 6. Additional: stuck pattern analysis ──────────────────────────────
    print(f"\n{'─' * 80}")
    print("6. STUCK/REPEATING PATTERN ANALYSIS")
    print(f"{'─' * 80}")

    # For stuck steps, how many consecutive stuck steps per trajectory?
    stuck_ids = set(s["sample_id"] for s in categories["stuck_repeating"])

    stuck_run_lengths = []
    for t in trajectories:
        sorted_steps = sorted(t["step_results"], key=lambda s: s["step_num"])
        current_run = 0
        for s in sorted_steps:
            if s["sample_id"] in stuck_ids:
                current_run += 1
            else:
                if current_run > 0:
                    stuck_run_lengths.append(current_run)
                current_run = 0
        if current_run > 0:
            stuck_run_lengths.append(current_run)

    if stuck_run_lengths:
        print(f"\n  Number of stuck runs: {len(stuck_run_lengths)}")
        print(f"  Mean run length     : {sum(stuck_run_lengths)/len(stuck_run_lengths):.1f}")
        print(f"  Median run length   : {sorted(stuck_run_lengths)[len(stuck_run_lengths)//2]}")
        print(f"  Max run length      : {max(stuck_run_lengths)}")

        run_hist = Counter()
        for r in stuck_run_lengths:
            if r == 1:
                run_hist["1"] += 1
            elif r <= 3:
                run_hist["2-3"] += 1
            elif r <= 5:
                run_hist["4-5"] += 1
            elif r <= 10:
                run_hist["6-10"] += 1
            else:
                run_hist[">10"] += 1

        print(f"\n  Run length distribution:")
        for label in ["1", "2-3", "4-5", "6-10", ">10"]:
            c = run_hist.get(label, 0)
            print(f"    {label:>5s}: {c:4d} runs")

    # How many trajectories have at least one stuck step?
    traj_with_stuck = set()
    for s in categories["stuck_repeating"]:
        # Extract trajectory ID from sample_id (remove _N suffix)
        parts = s["sample_id"].rsplit("_", 1)
        traj_with_stuck.add(parts[0])

    print(f"\n  Trajectories with stuck steps: {len(traj_with_stuck)} / {len(trajectories)} "
          f"({len(traj_with_stuck)/len(trajectories)*100:.1f}%)")

    # ── 7. Summary comparison ──────────────────────────────────────────────
    print(f"\n{'─' * 80}")
    print("7. SUMMARY: SFT v2 vs BASE MODEL")
    print(f"{'─' * 80}")

    base_failed = 2608
    base_total = 2608 + 1246  # approximate from 67.6% fail rate ~= base step fail

    print(f"""
  Metric                              Base Model    SFT v2
  ──────────────────────────────────────────────────────────
  Total steps                         ~{base_total:>6d}       {total_steps:>6d}
  Failed steps                         {base_failed:>6d}       {n_failed:>6d}
  Step fail rate                       ~67.6%       {n_failed/total_steps*100:>5.1f}%

  Error Distribution (% of failed):
  ──────────────────────────────────────────────────────────
  Stuck/repeating (cascade)            51.2%       {sft_stuck_pct:>5.1f}%
  Type mismatch / Vocabulary           14.6%       {sft_type_pct:>5.1f}%
  Coordinate error (>=50px)            ~18.4%       {sft_coord_pct:>5.1f}%
  Near miss (<50px)                     3.5%       {sft_near_pct:>5.1f}%
  No-coord / text mismatch              ---        {sft_nocoord_pct:>5.1f}%
  Parse error                            ---        {sft_parse_pct:>5.1f}%

  Key observations:
  - SFT v2 step success rate: {success_steps/total_steps*100:.1f}% (vs base ~32.4%)
  - Stuck/repeating as % of errors: {sft_stuck_pct:.1f}% (base: 51.2%)
  - Type mismatch as % of errors: {sft_type_pct:.1f}% (base: 14.6%)
  - Pure coordinate errors: {sft_coord_pct:.1f}% (base: ~18.4%)
  - Median coord error distance: {median_dist:.1f}px
""")

    print("=" * 80)
    print("Analysis complete.")
    print("=" * 80)


if __name__ == "__main__":
    main()
