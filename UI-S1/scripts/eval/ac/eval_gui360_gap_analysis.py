#!/usr/bin/env python3
"""
GUI-360 Gap 1 + Gap 2 + Coord Spread Analysis
===============================================
Full analysis of the GUI-360 multi-sample evaluation data (K=10).

Analyses:
  1. Gap 1: Temperature degradation (greedy vs temp per-step accuracy)
  2. Gap 2: State distribution shift (note: applies identically as AC)
  3. Coord spread analysis (Stage 2 grounding detection)
  4. Agreement-based Pareto curve (like AC E_NEW2)
  5. Oracle ceiling (per action type, step position, domain)
  6. Error type distribution comparison

Data source: multisample_results.jsonl (K=10 per-step predictions)
"""

import json
import math
import os
import sys
from collections import Counter, defaultdict

import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE = "/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1"
DATA_PATH = os.path.join(
    BASE, "outputs/eval_gui360_multisample/multisample_results.jsonl"
)
OUTPUT_DIR = os.path.join(BASE, "outputs/eval_gui360_gap_analysis")
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "gui360_gap_analysis.json")

K = 10  # samples per step

# Coordinate-based action types in GUI-360 (have x,y coordinates)
COORD_ACTION_TYPES = {"click", "type", "wheel_mouse_input", "set_focus",
                      "select_text", "select_paragraph", "select_table",
                      "select_table_range", "insert_excel_table", "insert_table",
                      "set_font", "set_background_color"}
# Actions unlikely to have coordinates
NON_COORD_ACTION_TYPES = {"run_shell", "save_as", "summary", "table2markdown",
                          "Spinner"}

# Agreement threshold for high-agreement filtering
AGREEMENT_THRESHOLD = 0.9

# AC reference values for comparison
AC_REF = {
    "greedy_step_acc": 71.7,    # approximate AC greedy step accuracy
    "greedy_args_acc": 57.3,    # approximate AC greedy args accuracy
    "coord_spread_auroc": 0.7355,
    "oracle_step_acc": 82.0,    # approximate AC oracle step accuracy
}


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def safe_mean(vals):
    """Mean of list, or NaN if empty."""
    if not vals:
        return float("nan")
    return float(np.mean(vals))


def safe_std(vals):
    """Std of list, or NaN if empty."""
    if not vals:
        return float("nan")
    return float(np.std(vals))


def percentile_stats(vals):
    """Return min, p25, median, p75, max, mean, std for a list of values."""
    if not vals:
        return {"min": None, "p25": None, "median": None, "p75": None,
                "max": None, "mean": None, "std": None, "n": 0}
    arr = np.array(vals)
    return {
        "min": float(np.min(arr)),
        "p25": float(np.percentile(arr, 25)),
        "median": float(np.median(arr)),
        "p75": float(np.percentile(arr, 75)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "n": len(vals),
    }


def auroc_manual(scores, labels):
    """Compute AUROC manually via trapezoidal rule.
    scores: higher = more likely positive.
    labels: 1 = positive, 0 = negative.
    Returns AUROC or NaN if degenerate."""
    pairs = list(zip(scores, labels))
    pairs = [(s, l) for s, l in pairs if not math.isnan(s)]
    if not pairs:
        return float("nan")
    positives = sum(1 for _, l in pairs if l == 1)
    negatives = len(pairs) - positives
    if positives == 0 or negatives == 0:
        return float("nan")

    pairs.sort(key=lambda x: -x[0])
    tp = 0
    fp = 0
    auc = 0.0
    prev_tp = 0
    prev_fp = 0
    prev_score = None
    for score, label in pairs:
        if prev_score is not None and score != prev_score:
            auc += (fp - prev_fp) * (tp + prev_tp) / 2.0
            prev_tp = tp
            prev_fp = fp
        if label == 1:
            tp += 1
        else:
            fp += 1
        prev_score = score
    auc += (fp - prev_fp) * (tp + prev_tp) / 2.0
    return auc / (positives * negatives)


def try_auroc_sklearn(scores, labels):
    """Try to compute AUROC with sklearn, fall back to manual."""
    try:
        from sklearn.metrics import roc_auc_score
        valid = [(s, l) for s, l in zip(scores, labels) if not math.isnan(s)]
        if not valid:
            return float("nan")
        s, l = zip(*valid)
        if len(set(l)) < 2:
            return float("nan")
        return float(roc_auc_score(l, s))
    except ImportError:
        return auroc_manual(scores, labels)


def classify_step(rec):
    """Classify a step into pass_through, action_error, or grounding_error."""
    if rec["greedy_function_match"] and rec["greedy_args_match"]:
        return "pass_through"
    elif not rec["greedy_function_match"]:
        return "action_error"
    else:
        return "grounding_error"


def step_position_bin(step_index, total_steps):
    """Bin step position into early/middle/late."""
    if total_steps <= 0:
        return "unknown"
    ratio = step_index / max(total_steps - 1, 1)
    if ratio <= 0.33:
        return "early"
    elif ratio <= 0.66:
        return "middle"
    else:
        return "late"


# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------

def load_data():
    print("=" * 80)
    print("GUI-360 Gap Analysis: Loading data")
    print("=" * 80)
    print(f"Data: {DATA_PATH}")

    records = []
    with open(DATA_PATH) as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                print(f"  [WARN] Skipping malformed JSON at line {line_num}")
                continue
            # Add derived fields
            rec["_error_type"] = classify_step(rec)
            rec["_position_bin"] = step_position_bin(
                rec.get("step_index", 0), rec.get("total_steps", 1)
            )
            # Extract coord spread from coord_stats
            cs = rec.get("coord_stats", {})
            rec["_coord_spread"] = cs.get("mean_std", float("nan")) if cs else float("nan")
            rec["_x_std"] = cs.get("x_std", float("nan")) if cs else float("nan")
            rec["_y_std"] = cs.get("y_std", float("nan")) if cs else float("nan")
            rec["_n_coord_samples"] = cs.get("n_coord_samples", 0) if cs else 0
            rec["_has_coords"] = bool(cs) and rec["_n_coord_samples"] >= 2

            records.append(rec)

    print(f"Loaded {len(records)} step records.\n")
    return records


# ---------------------------------------------------------------------------
# 2. Gap 1: Temperature Degradation
# ---------------------------------------------------------------------------

def gap1_analysis(records):
    print("=" * 80)
    print("ANALYSIS 1: Gap 1 - Temperature Degradation")
    print("=" * 80)

    results = {}

    # --- Overall greedy vs temperature ---
    greedy_func = [r["greedy_function_match"] for r in records]
    greedy_args = [r["greedy_args_match"] for r in records]
    temp_func = [r["temp_function_match_rate"] for r in records]
    temp_args = [r["temp_args_match_rate"] for r in records]

    greedy_func_acc = safe_mean(greedy_func) * 100
    greedy_args_acc = safe_mean(greedy_args) * 100
    temp_func_acc = safe_mean(temp_func) * 100
    temp_args_acc = safe_mean(temp_args) * 100

    delta_func = greedy_func_acc - temp_func_acc
    delta_args = greedy_args_acc - temp_args_acc

    results["overall"] = {
        "n_steps": len(records),
        "greedy_function_acc": round(greedy_func_acc, 2),
        "greedy_args_acc": round(greedy_args_acc, 2),
        "temp_function_acc": round(temp_func_acc, 2),
        "temp_args_acc": round(temp_args_acc, 2),
        "delta_function": round(delta_func, 2),
        "delta_args": round(delta_args, 2),
    }

    print(f"\n  Overall ({len(records)} steps):")
    print(f"    Greedy function acc: {greedy_func_acc:.2f}%")
    print(f"    Temp   function acc: {temp_func_acc:.2f}%")
    print(f"    Delta_func (greedy-temp): {delta_func:+.2f}pp")
    print(f"    Greedy args acc:    {greedy_args_acc:.2f}%")
    print(f"    Temp   args acc:    {temp_args_acc:.2f}%")
    print(f"    Delta_args (greedy-temp): {delta_args:+.2f}pp")

    # --- By action type ---
    by_action = defaultdict(list)
    for r in records:
        by_action[r["gt_function"]].append(r)

    action_results = {}
    print(f"\n  By action type:")
    print(f"    {'Action':<25s} {'N':>6s} {'Greedy%':>8s} {'Temp%':>8s} {'Delta':>8s}  {'GArgs%':>8s} {'TArgs%':>8s} {'DArgs':>8s}")
    print(f"    {'-'*25} {'-'*6} {'-'*8} {'-'*8} {'-'*8}  {'-'*8} {'-'*8} {'-'*8}")

    for act in sorted(by_action.keys(), key=lambda a: -len(by_action[a])):
        recs = by_action[act]
        n = len(recs)
        gf = safe_mean([r["greedy_function_match"] for r in recs]) * 100
        tf = safe_mean([r["temp_function_match_rate"] for r in recs]) * 100
        ga = safe_mean([r["greedy_args_match"] for r in recs]) * 100
        ta = safe_mean([r["temp_args_match_rate"] for r in recs]) * 100
        df = gf - tf
        da = ga - ta
        print(f"    {act:<25s} {n:>6d} {gf:>8.2f} {tf:>8.2f} {df:>+8.2f}  {ga:>8.2f} {ta:>8.2f} {da:>+8.2f}")
        action_results[act] = {
            "n": n,
            "greedy_func_acc": round(gf, 2),
            "temp_func_acc": round(tf, 2),
            "delta_func": round(df, 2),
            "greedy_args_acc": round(ga, 2),
            "temp_args_acc": round(ta, 2),
            "delta_args": round(da, 2),
        }
    results["by_action_type"] = action_results

    # --- By step position ---
    by_pos = defaultdict(list)
    for r in records:
        by_pos[r["_position_bin"]].append(r)

    pos_results = {}
    print(f"\n  By step position:")
    print(f"    {'Position':<10s} {'N':>6s} {'Greedy%':>8s} {'Temp%':>8s} {'Delta':>8s}  {'GArgs%':>8s} {'TArgs%':>8s} {'DArgs':>8s}")
    for pos in ["early", "middle", "late"]:
        recs = by_pos.get(pos, [])
        if not recs:
            continue
        n = len(recs)
        gf = safe_mean([r["greedy_function_match"] for r in recs]) * 100
        tf = safe_mean([r["temp_function_match_rate"] for r in recs]) * 100
        ga = safe_mean([r["greedy_args_match"] for r in recs]) * 100
        ta = safe_mean([r["temp_args_match_rate"] for r in recs]) * 100
        print(f"    {pos:<10s} {n:>6d} {gf:>8.2f} {tf:>8.2f} {gf-tf:>+8.2f}  {ga:>8.2f} {ta:>8.2f} {ga-ta:>+8.2f}")
        pos_results[pos] = {
            "n": n,
            "greedy_func_acc": round(gf, 2),
            "temp_func_acc": round(tf, 2),
            "delta_func": round(gf - tf, 2),
            "greedy_args_acc": round(ga, 2),
            "temp_args_acc": round(ta, 2),
            "delta_args": round(ga - ta, 2),
        }
    results["by_step_position"] = pos_results

    # --- By domain ---
    by_domain = defaultdict(list)
    for r in records:
        by_domain[r["domain"]].append(r)

    domain_results = {}
    print(f"\n  By domain:")
    print(f"    {'Domain':<10s} {'N':>6s} {'Greedy%':>8s} {'Temp%':>8s} {'Delta':>8s}  {'GArgs%':>8s} {'TArgs%':>8s} {'DArgs':>8s}")
    for dom in sorted(by_domain.keys()):
        recs = by_domain[dom]
        n = len(recs)
        gf = safe_mean([r["greedy_function_match"] for r in recs]) * 100
        tf = safe_mean([r["temp_function_match_rate"] for r in recs]) * 100
        ga = safe_mean([r["greedy_args_match"] for r in recs]) * 100
        ta = safe_mean([r["temp_args_match_rate"] for r in recs]) * 100
        print(f"    {dom:<10s} {n:>6d} {gf:>8.2f} {tf:>8.2f} {gf-tf:>+8.2f}  {ga:>8.2f} {ta:>8.2f} {ga-ta:>+8.2f}")
        domain_results[dom] = {
            "n": n,
            "greedy_func_acc": round(gf, 2),
            "temp_func_acc": round(tf, 2),
            "delta_func": round(gf - tf, 2),
            "greedy_args_acc": round(ga, 2),
            "temp_args_acc": round(ta, 2),
            "delta_args": round(ga - ta, 2),
        }
    results["by_domain"] = domain_results

    # --- Comparison with AC ---
    print(f"\n  Comparison with AC:")
    print(f"    AC greedy step acc (ref):  ~{AC_REF['greedy_step_acc']:.1f}%")
    print(f"    GUI-360 greedy func acc:    {greedy_func_acc:.2f}%")
    print(f"    AC greedy args acc (ref):  ~{AC_REF['greedy_args_acc']:.1f}%")
    print(f"    GUI-360 greedy args acc:    {greedy_args_acc:.2f}%")
    results["ac_comparison"] = {
        "ac_greedy_step_acc_ref": AC_REF["greedy_step_acc"],
        "gui360_greedy_func_acc": round(greedy_func_acc, 2),
        "ac_greedy_args_acc_ref": AC_REF["greedy_args_acc"],
        "gui360_greedy_args_acc": round(greedy_args_acc, 2),
    }

    print()
    return results


# ---------------------------------------------------------------------------
# 3. Gap 2: State Distribution Shift (note)
# ---------------------------------------------------------------------------

def gap2_note():
    print("=" * 80)
    print("ANALYSIS 2: Gap 2 - State Distribution Shift")
    print("=" * 80)
    note = (
        "Gap 2 (state distribution shift) is not directly testable from "
        "per-step static evaluation data. GUI-360 uses stop-on-error "
        "autoregressive evaluation, so Gap 2 applies identically as in AC: "
        "the model accumulates errors over the trajectory, and later steps "
        "see out-of-distribution states caused by earlier mistakes. "
        "This gap would require trajectory-level AR evaluation to quantify."
    )
    print(f"\n  NOTE: {note}\n")
    return {"note": note}


# ---------------------------------------------------------------------------
# 4. Coord Spread Analysis (Stage 2)
# ---------------------------------------------------------------------------

def coord_spread_analysis(records):
    print("=" * 80)
    print("ANALYSIS 3: Coordinate Spread (Stage 2 Grounding Detection)")
    print("=" * 80)

    results = {}

    # Overall error type distribution
    error_counts = Counter(r["_error_type"] for r in records)
    total = len(records)
    print(f"\n  Overall error type distribution ({total} steps):")
    for etype in ["pass_through", "action_error", "grounding_error"]:
        cnt = error_counts.get(etype, 0)
        print(f"    {etype:<20s}: {cnt:>6d}  ({cnt/total*100:.1f}%)")
    results["error_distribution"] = {
        etype: {"count": error_counts.get(etype, 0),
                "pct": round(error_counts.get(etype, 0) / total * 100, 2)}
        for etype in ["pass_through", "action_error", "grounding_error"]
    }

    # Filter to high-agreement steps with coordinate stats
    high_agree = [r for r in records if r["agreement_rate"] >= AGREEMENT_THRESHOLD]
    high_agree_with_coords = [r for r in high_agree if r["_has_coords"]]

    print(f"\n  High-agreement (>= {AGREEMENT_THRESHOLD}) steps: {len(high_agree)}")
    print(f"  High-agreement + has coords: {len(high_agree_with_coords)}")

    ha_error_counts = Counter(r["_error_type"] for r in high_agree_with_coords)
    print(f"\n  Error distribution in high-agreement + coords subset:")
    for etype in ["pass_through", "action_error", "grounding_error"]:
        cnt = ha_error_counts.get(etype, 0)
        n_ha = len(high_agree_with_coords) if high_agree_with_coords else 1
        print(f"    {etype:<20s}: {cnt:>6d}  ({cnt/n_ha*100:.1f}%)")

    results["high_agreement_coord_subset"] = {
        "n_high_agreement": len(high_agree),
        "n_high_agreement_with_coords": len(high_agree_with_coords),
        "error_distribution": {
            etype: ha_error_counts.get(etype, 0)
            for etype in ["pass_through", "action_error", "grounding_error"]
        },
    }

    # Coord spread comparison: pass_through vs grounding_error
    pt_spreads = [r["_coord_spread"] for r in high_agree_with_coords
                  if r["_error_type"] == "pass_through" and not math.isnan(r["_coord_spread"])]
    ge_spreads = [r["_coord_spread"] for r in high_agree_with_coords
                  if r["_error_type"] == "grounding_error" and not math.isnan(r["_coord_spread"])]

    print(f"\n  Coord spread (mean_std) comparison:")
    print(f"    pass_through   (n={len(pt_spreads):>5d}): mean={safe_mean(pt_spreads):.2f}, "
          f"median={float(np.median(pt_spreads)) if pt_spreads else float('nan'):.2f}")
    print(f"    grounding_error(n={len(ge_spreads):>5d}): mean={safe_mean(ge_spreads):.2f}, "
          f"median={float(np.median(ge_spreads)) if ge_spreads else float('nan'):.2f}")

    results["coord_spread_comparison"] = {
        "pass_through": percentile_stats(pt_spreads),
        "grounding_error": percentile_stats(ge_spreads),
    }

    # AUROC: coord_spread predicting grounding_error among high-agreement coord steps
    # Binary: grounding_error=1, pass_through=0 (exclude action_error for clean comparison)
    binary_subset = [r for r in high_agree_with_coords
                     if r["_error_type"] in ("pass_through", "grounding_error")
                     and not math.isnan(r["_coord_spread"])]

    if len(binary_subset) >= 10:
        scores = [r["_coord_spread"] for r in binary_subset]
        labels = [1 if r["_error_type"] == "grounding_error" else 0 for r in binary_subset]
        auroc = try_auroc_sklearn(scores, labels)
        auroc_manual_val = auroc_manual(scores, labels)

        n_pos = sum(labels)
        n_neg = len(labels) - n_pos
        print(f"\n  AUROC (coord_spread -> grounding detection):")
        print(f"    N={len(binary_subset)} (pos={n_pos}, neg={n_neg})")
        print(f"    AUROC = {auroc:.4f}")
        print(f"    AUROC (manual) = {auroc_manual_val:.4f}")
        print(f"    AC reference AUROC = {AC_REF['coord_spread_auroc']:.4f}")
        print(f"    Difference from AC: {auroc - AC_REF['coord_spread_auroc']:+.4f}")

        results["auroc"] = {
            "auroc": round(auroc, 4),
            "auroc_manual": round(auroc_manual_val, 4),
            "n_total": len(binary_subset),
            "n_positive": n_pos,
            "n_negative": n_neg,
            "ac_reference_auroc": AC_REF["coord_spread_auroc"],
            "delta_vs_ac": round(auroc - AC_REF["coord_spread_auroc"], 4),
        }
    else:
        print(f"\n  AUROC: Insufficient data (only {len(binary_subset)} binary samples)")
        results["auroc"] = {"auroc": None, "note": "insufficient data"}

    # Also compute AUROC by action type
    by_action = defaultdict(list)
    for r in binary_subset:
        by_action[r["gt_function"]].append(r)

    action_aurocs = {}
    print(f"\n  AUROC by action type (within high-agreement + coords):")
    print(f"    {'Action':<25s} {'N':>5s} {'Pos':>5s} {'Neg':>5s} {'AUROC':>8s}")
    for act in sorted(by_action.keys(), key=lambda a: -len(by_action[a])):
        recs = by_action[act]
        if len(recs) < 5:
            continue
        s = [r["_coord_spread"] for r in recs]
        l = [1 if r["_error_type"] == "grounding_error" else 0 for r in recs]
        n_p = sum(l)
        n_n = len(l) - n_p
        if n_p == 0 or n_n == 0:
            aval = float("nan")
        else:
            aval = try_auroc_sklearn(s, l)
        print(f"    {act:<25s} {len(recs):>5d} {n_p:>5d} {n_n:>5d} {aval:>8.4f}")
        action_aurocs[act] = {
            "n": len(recs), "n_pos": n_p, "n_neg": n_n,
            "auroc": round(aval, 4) if not math.isnan(aval) else None,
        }
    results["auroc_by_action"] = action_aurocs

    # Coord spread by domain
    by_domain = defaultdict(list)
    for r in binary_subset:
        by_domain[r["domain"]].append(r)

    domain_aurocs = {}
    print(f"\n  AUROC by domain:")
    print(f"    {'Domain':<10s} {'N':>5s} {'Pos':>5s} {'Neg':>5s} {'AUROC':>8s}")
    for dom in sorted(by_domain.keys()):
        recs = by_domain[dom]
        if len(recs) < 5:
            continue
        s = [r["_coord_spread"] for r in recs]
        l = [1 if r["_error_type"] == "grounding_error" else 0 for r in recs]
        n_p = sum(l)
        n_n = len(l) - n_p
        if n_p == 0 or n_n == 0:
            aval = float("nan")
        else:
            aval = try_auroc_sklearn(s, l)
        print(f"    {dom:<10s} {len(recs):>5d} {n_p:>5d} {n_n:>5d} {aval:>8.4f}")
        domain_aurocs[dom] = {
            "n": len(recs), "n_pos": n_p, "n_neg": n_n,
            "auroc": round(aval, 4) if not math.isnan(aval) else None,
        }
    results["auroc_by_domain"] = domain_aurocs

    print()
    return results


# ---------------------------------------------------------------------------
# 5. Agreement-based Pareto Curve (like AC E_NEW2)
# ---------------------------------------------------------------------------

def pareto_analysis(records):
    print("=" * 80)
    print("ANALYSIS 4: Agreement-based Pareto Curve")
    print("=" * 80)

    results = {}
    N = len(records)

    # Reference points
    greedy_func_acc = safe_mean([r["greedy_function_match"] for r in records]) * 100
    greedy_args_acc = safe_mean([r["greedy_args_match"] for r in records]) * 100
    oracle_func_acc = safe_mean([r["oracle_function_match"] for r in records]) * 100
    oracle_args_acc = safe_mean([r["oracle_args_match"] for r in records]) * 100

    print(f"\n  Reference points ({N} steps):")
    print(f"    Greedy function acc: {greedy_func_acc:.2f}%  (cost = 1.0)")
    print(f"    Oracle function acc: {oracle_func_acc:.2f}%  (cost = {K:.1f})")
    print(f"    Greedy args acc:     {greedy_args_acc:.2f}%  (cost = 1.0)")
    print(f"    Oracle args acc:     {oracle_args_acc:.2f}%  (cost = {K:.1f})")

    results["reference"] = {
        "n_steps": N,
        "greedy_func_acc": round(greedy_func_acc, 2),
        "greedy_args_acc": round(greedy_args_acc, 2),
        "oracle_func_acc": round(oracle_func_acc, 2),
        "oracle_args_acc": round(oracle_args_acc, 2),
    }

    # Agreement statistics
    agreements = [r["agreement_rate"] for r in records]
    print(f"\n  Agreement rate statistics:")
    astats = percentile_stats(agreements)
    for k in ["min", "p25", "median", "p75", "max", "mean"]:
        print(f"    {k:<8s}: {astats[k]:.3f}")
    results["agreement_stats"] = {k: round(v, 4) if v is not None else None
                                   for k, v in astats.items()}

    # Sweep thresholds
    thresholds = [round(t * 0.05, 2) for t in range(0, 21)]
    curve = []

    print(f"\n  Pareto sweep (step-level, no trajectory structure):")
    print(f"    {'tau':>6s} {'PT%':>8s} {'FuncAcc':>8s} {'ArgsAcc':>8s} {'AvgCost':>8s} {'Savings%':>9s}")
    print(f"    {'-'*6} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*9}")

    for tau in thresholds:
        n_pt = 0  # pass-through count
        n_sp = 0  # specialist count
        func_correct = 0
        args_correct = 0

        for r in records:
            if r["agreement_rate"] >= tau:
                # Pass-through: use greedy
                n_pt += 1
                if r["greedy_function_match"]:
                    func_correct += 1
                if r["greedy_args_match"]:
                    args_correct += 1
            else:
                # Specialist: use oracle best-of-K
                n_sp += 1
                if r["oracle_function_match"]:
                    func_correct += 1
                if r["oracle_args_match"]:
                    args_correct += 1

        func_acc = func_correct / N * 100
        args_acc = args_correct / N * 100
        pt_pct = n_pt / N * 100
        avg_cost = (n_pt * 1 + n_sp * K) / N
        savings = (1.0 - avg_cost / K) * 100

        print(f"    {tau:>6.2f} {pt_pct:>8.1f} {func_acc:>8.2f} {args_acc:>8.2f} "
              f"{avg_cost:>8.2f} {savings:>+9.1f}")

        curve.append({
            "tau": tau,
            "n_passthrough": n_pt,
            "n_specialist": n_sp,
            "passthrough_pct": round(pt_pct, 2),
            "func_acc": round(func_acc, 2),
            "args_acc": round(args_acc, 2),
            "avg_cost": round(avg_cost, 3),
            "compute_savings_pct": round(savings, 2),
        })

    results["pareto_curve"] = curve

    # Find the "sweet spot" - highest args_acc with savings > 30%
    sweet = None
    for pt in curve:
        if pt["compute_savings_pct"] > 30:
            if sweet is None or pt["args_acc"] > sweet["args_acc"]:
                sweet = pt
    if sweet:
        print(f"\n  Sweet spot (best args_acc with >30% savings):")
        print(f"    tau={sweet['tau']:.2f}: args_acc={sweet['args_acc']:.2f}%, "
              f"savings={sweet['compute_savings_pct']:.1f}%, cost={sweet['avg_cost']:.2f}")
        results["sweet_spot"] = sweet

    # Find breakeven: tau where args_acc >= greedy_args_acc
    breakeven = None
    for pt in sorted(curve, key=lambda x: -x["tau"]):
        if pt["args_acc"] >= greedy_args_acc:
            breakeven = pt
            break
    if breakeven:
        print(f"  Breakeven (highest tau with args_acc >= greedy):")
        print(f"    tau={breakeven['tau']:.2f}: args_acc={breakeven['args_acc']:.2f}%, "
              f"savings={breakeven['compute_savings_pct']:.1f}%, cost={breakeven['avg_cost']:.2f}")
        results["breakeven"] = breakeven

    print()
    return results


# ---------------------------------------------------------------------------
# 6. Oracle Ceiling
# ---------------------------------------------------------------------------

def oracle_analysis(records):
    print("=" * 80)
    print("ANALYSIS 5: Oracle Ceiling")
    print("=" * 80)

    results = {}

    # Overall
    oracle_func = safe_mean([r["oracle_function_match"] for r in records]) * 100
    oracle_args = safe_mean([r["oracle_args_match"] for r in records]) * 100
    greedy_func = safe_mean([r["greedy_function_match"] for r in records]) * 100
    greedy_args = safe_mean([r["greedy_args_match"] for r in records]) * 100

    print(f"\n  Overall ({len(records)} steps):")
    print(f"    Greedy function: {greedy_func:.2f}%    Oracle function: {oracle_func:.2f}%   Gap: {oracle_func-greedy_func:+.2f}pp")
    print(f"    Greedy args:     {greedy_args:.2f}%    Oracle args:     {oracle_args:.2f}%   Gap: {oracle_args-greedy_args:+.2f}pp")

    results["overall"] = {
        "greedy_func": round(greedy_func, 2),
        "greedy_args": round(greedy_args, 2),
        "oracle_func": round(oracle_func, 2),
        "oracle_args": round(oracle_args, 2),
        "gap_func": round(oracle_func - greedy_func, 2),
        "gap_args": round(oracle_args - greedy_args, 2),
    }

    # By action type
    by_action = defaultdict(list)
    for r in records:
        by_action[r["gt_function"]].append(r)

    action_results = {}
    print(f"\n  By action type:")
    print(f"    {'Action':<25s} {'N':>6s} {'GFunc%':>8s} {'OFunc%':>8s} {'FGap':>6s}  {'GArgs%':>8s} {'OArgs%':>8s} {'AGap':>6s}")
    print(f"    {'-'*25} {'-'*6} {'-'*8} {'-'*8} {'-'*6}  {'-'*8} {'-'*8} {'-'*6}")

    for act in sorted(by_action.keys(), key=lambda a: -len(by_action[a])):
        recs = by_action[act]
        n = len(recs)
        gf = safe_mean([r["greedy_function_match"] for r in recs]) * 100
        of_ = safe_mean([r["oracle_function_match"] for r in recs]) * 100
        ga = safe_mean([r["greedy_args_match"] for r in recs]) * 100
        oa = safe_mean([r["oracle_args_match"] for r in recs]) * 100
        print(f"    {act:<25s} {n:>6d} {gf:>8.2f} {of_:>8.2f} {of_-gf:>+6.1f}  {ga:>8.2f} {oa:>8.2f} {oa-ga:>+6.1f}")
        action_results[act] = {
            "n": n,
            "greedy_func": round(gf, 2), "oracle_func": round(of_, 2),
            "gap_func": round(of_ - gf, 2),
            "greedy_args": round(ga, 2), "oracle_args": round(oa, 2),
            "gap_args": round(oa - ga, 2),
        }
    results["by_action_type"] = action_results

    # By step position
    by_pos = defaultdict(list)
    for r in records:
        by_pos[r["_position_bin"]].append(r)

    pos_results = {}
    print(f"\n  By step position:")
    print(f"    {'Pos':<10s} {'N':>6s} {'GFunc%':>8s} {'OFunc%':>8s} {'FGap':>6s}  {'GArgs%':>8s} {'OArgs%':>8s} {'AGap':>6s}")
    for pos in ["early", "middle", "late"]:
        recs = by_pos.get(pos, [])
        if not recs:
            continue
        n = len(recs)
        gf = safe_mean([r["greedy_function_match"] for r in recs]) * 100
        of_ = safe_mean([r["oracle_function_match"] for r in recs]) * 100
        ga = safe_mean([r["greedy_args_match"] for r in recs]) * 100
        oa = safe_mean([r["oracle_args_match"] for r in recs]) * 100
        print(f"    {pos:<10s} {n:>6d} {gf:>8.2f} {of_:>8.2f} {of_-gf:>+6.1f}  {ga:>8.2f} {oa:>8.2f} {oa-ga:>+6.1f}")
        pos_results[pos] = {
            "n": n,
            "greedy_func": round(gf, 2), "oracle_func": round(of_, 2),
            "gap_func": round(of_ - gf, 2),
            "greedy_args": round(ga, 2), "oracle_args": round(oa, 2),
            "gap_args": round(oa - ga, 2),
        }
    results["by_step_position"] = pos_results

    # By domain
    by_domain = defaultdict(list)
    for r in records:
        by_domain[r["domain"]].append(r)

    domain_results = {}
    print(f"\n  By domain:")
    print(f"    {'Domain':<10s} {'N':>6s} {'GFunc%':>8s} {'OFunc%':>8s} {'FGap':>6s}  {'GArgs%':>8s} {'OArgs%':>8s} {'AGap':>6s}")
    for dom in sorted(by_domain.keys()):
        recs = by_domain[dom]
        n = len(recs)
        gf = safe_mean([r["greedy_function_match"] for r in recs]) * 100
        of_ = safe_mean([r["oracle_function_match"] for r in recs]) * 100
        ga = safe_mean([r["greedy_args_match"] for r in recs]) * 100
        oa = safe_mean([r["oracle_args_match"] for r in recs]) * 100
        print(f"    {dom:<10s} {n:>6d} {gf:>8.2f} {of_:>8.2f} {of_-gf:>+6.1f}  {ga:>8.2f} {oa:>8.2f} {oa-ga:>+6.1f}")
        domain_results[dom] = {
            "n": n,
            "greedy_func": round(gf, 2), "oracle_func": round(of_, 2),
            "gap_func": round(of_ - gf, 2),
            "greedy_args": round(ga, 2), "oracle_args": round(oa, 2),
            "gap_args": round(oa - ga, 2),
        }
    results["by_domain"] = domain_results

    print()
    return results


# ---------------------------------------------------------------------------
# 7. Error Type Distribution Comparison
# ---------------------------------------------------------------------------

def error_distribution_analysis(records):
    print("=" * 80)
    print("ANALYSIS 6: Error Type Distribution")
    print("=" * 80)

    results = {}
    N = len(records)

    # Overall
    error_counts = Counter(r["_error_type"] for r in records)
    print(f"\n  Overall error type distribution ({N} steps):")
    overall = {}
    for etype in ["pass_through", "action_error", "grounding_error"]:
        cnt = error_counts.get(etype, 0)
        pct = cnt / N * 100
        print(f"    {etype:<20s}: {cnt:>6d}  ({pct:.1f}%)")
        overall[etype] = {"count": cnt, "pct": round(pct, 2)}
    results["overall"] = overall

    # By action type
    by_action = defaultdict(lambda: Counter())
    action_totals = Counter()
    for r in records:
        by_action[r["gt_function"]][r["_error_type"]] += 1
        action_totals[r["gt_function"]] += 1

    action_results = {}
    print(f"\n  By action type:")
    print(f"    {'Action':<25s} {'N':>6s} {'PassT%':>8s} {'ActErr%':>8s} {'GrdErr%':>8s}")
    print(f"    {'-'*25} {'-'*6} {'-'*8} {'-'*8} {'-'*8}")

    for act in sorted(by_action.keys(), key=lambda a: -action_totals[a]):
        n = action_totals[act]
        ec = by_action[act]
        pt = ec.get("pass_through", 0) / n * 100
        ae = ec.get("action_error", 0) / n * 100
        ge = ec.get("grounding_error", 0) / n * 100
        print(f"    {act:<25s} {n:>6d} {pt:>8.1f} {ae:>8.1f} {ge:>8.1f}")
        action_results[act] = {
            "n": n,
            "pass_through_pct": round(pt, 2),
            "action_error_pct": round(ae, 2),
            "grounding_error_pct": round(ge, 2),
        }
    results["by_action_type"] = action_results

    # By domain
    by_domain = defaultdict(lambda: Counter())
    domain_totals = Counter()
    for r in records:
        by_domain[r["domain"]][r["_error_type"]] += 1
        domain_totals[r["domain"]] += 1

    domain_results = {}
    print(f"\n  By domain:")
    print(f"    {'Domain':<10s} {'N':>6s} {'PassT%':>8s} {'ActErr%':>8s} {'GrdErr%':>8s}")
    for dom in sorted(by_domain.keys()):
        n = domain_totals[dom]
        ec = by_domain[dom]
        pt = ec.get("pass_through", 0) / n * 100
        ae = ec.get("action_error", 0) / n * 100
        ge = ec.get("grounding_error", 0) / n * 100
        print(f"    {dom:<10s} {n:>6d} {pt:>8.1f} {ae:>8.1f} {ge:>8.1f}")
        domain_results[dom] = {
            "n": n,
            "pass_through_pct": round(pt, 2),
            "action_error_pct": round(ae, 2),
            "grounding_error_pct": round(ge, 2),
        }
    results["by_domain"] = domain_results

    # By category
    by_cat = defaultdict(lambda: Counter())
    cat_totals = Counter()
    for r in records:
        cat = r.get("category", "unknown")
        by_cat[cat][r["_error_type"]] += 1
        cat_totals[cat] += 1

    cat_results = {}
    print(f"\n  By category:")
    print(f"    {'Category':<12s} {'N':>6s} {'PassT%':>8s} {'ActErr%':>8s} {'GrdErr%':>8s}")
    for cat in sorted(by_cat.keys()):
        n = cat_totals[cat]
        ec = by_cat[cat]
        pt = ec.get("pass_through", 0) / n * 100
        ae = ec.get("action_error", 0) / n * 100
        ge = ec.get("grounding_error", 0) / n * 100
        print(f"    {cat:<12s} {n:>6d} {pt:>8.1f} {ae:>8.1f} {ge:>8.1f}")
        cat_results[cat] = {
            "n": n,
            "pass_through_pct": round(pt, 2),
            "action_error_pct": round(ae, 2),
            "grounding_error_pct": round(ge, 2),
        }
    results["by_category"] = cat_results

    print()
    return results


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def print_summary(all_results):
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)

    g1 = all_results["gap1"]
    g3 = all_results["coord_spread"]
    g4 = all_results["pareto"]
    g5 = all_results["oracle"]

    print(f"\n  Dataset: {g1['overall']['n_steps']} steps across "
          f"{len(g1.get('by_domain', {}))} domains")

    print(f"\n  === Gap 1: Temperature Degradation ===")
    o = g1["overall"]
    print(f"    Greedy func acc: {o['greedy_function_acc']:.2f}%")
    print(f"    Temp func acc:   {o['temp_function_acc']:.2f}%")
    print(f"    Delta_func:      {o['delta_function']:+.2f}pp  "
          f"({'significant' if abs(o['delta_function']) > 2 else 'minimal'})")
    print(f"    Greedy args acc: {o['greedy_args_acc']:.2f}%")
    print(f"    Temp args acc:   {o['temp_args_acc']:.2f}%")
    print(f"    Delta_args:      {o['delta_args']:+.2f}pp  "
          f"({'significant' if abs(o['delta_args']) > 2 else 'minimal'})")

    print(f"\n  === Gap 2: State Distribution Shift ===")
    print(f"    {all_results['gap2']['note'][:100]}...")

    print(f"\n  === Coord Spread (Stage 2 Grounding Detection) ===")
    if g3.get("auroc", {}).get("auroc") is not None:
        auroc = g3["auroc"]["auroc"]
        print(f"    AUROC: {auroc:.4f}  (AC ref: {AC_REF['coord_spread_auroc']:.4f})")
        print(f"    Delta vs AC: {g3['auroc']['delta_vs_ac']:+.4f}")
        if auroc > 0.6:
            print(f"    Stage 2 coord_spread IS a useful grounding signal (AUROC > 0.6)")
        else:
            print(f"    Stage 2 coord_spread is NOT a strong signal (AUROC <= 0.6)")
    else:
        print(f"    AUROC: Insufficient data")

    pt_n = g3.get("coord_spread_comparison", {}).get("pass_through", {}).get("n", 0)
    ge_n = g3.get("coord_spread_comparison", {}).get("grounding_error", {}).get("n", 0)
    pt_mean = g3.get("coord_spread_comparison", {}).get("pass_through", {}).get("mean", None)
    ge_mean = g3.get("coord_spread_comparison", {}).get("grounding_error", {}).get("mean", None)
    if pt_mean is not None and ge_mean is not None:
        print(f"    Mean spread: pass_through={pt_mean:.1f} (n={pt_n}), "
              f"grounding_error={ge_mean:.1f} (n={ge_n})")

    print(f"\n  === Pareto Curve (Agreement-based Routing) ===")
    ref = g4["reference"]
    print(f"    Greedy args acc (tau=0): {ref['greedy_args_acc']:.2f}%")
    print(f"    Oracle args acc (tau=1): {ref['oracle_args_acc']:.2f}%")
    if "sweet_spot" in g4:
        ss = g4["sweet_spot"]
        print(f"    Sweet spot: tau={ss['tau']:.2f}, args_acc={ss['args_acc']:.2f}%, "
              f"savings={ss['compute_savings_pct']:.1f}%")
    if "breakeven" in g4:
        be = g4["breakeven"]
        print(f"    Breakeven:  tau={be['tau']:.2f}, args_acc={be['args_acc']:.2f}%, "
              f"savings={be['compute_savings_pct']:.1f}%")

    print(f"\n  === Oracle Ceiling ===")
    oall = g5["overall"]
    print(f"    Function gap: {oall['greedy_func']:.2f}% -> {oall['oracle_func']:.2f}% "
          f"(+{oall['gap_func']:.2f}pp)")
    print(f"    Args gap:     {oall['greedy_args']:.2f}% -> {oall['oracle_args']:.2f}% "
          f"(+{oall['gap_args']:.2f}pp)")

    print(f"\n  === Error Distribution ===")
    ed = all_results["error_distribution"]["overall"]
    for etype in ["pass_through", "action_error", "grounding_error"]:
        d = ed[etype]
        print(f"    {etype:<20s}: {d['pct']:.1f}%  ({d['count']} steps)")

    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # Load
    records = load_data()

    # Run all analyses
    all_results = {}
    all_results["gap1"] = gap1_analysis(records)
    all_results["gap2"] = gap2_note()
    all_results["coord_spread"] = coord_spread_analysis(records)
    all_results["pareto"] = pareto_analysis(records)
    all_results["oracle"] = oracle_analysis(records)
    all_results["error_distribution"] = error_distribution_analysis(records)

    # Print summary
    print_summary(all_results)

    # Save
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"Results saved to: {OUTPUT_PATH}")
    print(f"Total records analyzed: {len(records)}")


if __name__ == "__main__":
    main()
