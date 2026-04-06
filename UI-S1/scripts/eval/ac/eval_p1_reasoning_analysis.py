#!/usr/bin/env python3
"""
P1: Reasoning Quality Proxy from Multi-Sample Diversity

Analyzes whether reasoning patterns differ structurally between error types
using existing C4+C7 multi-sample data (K=10) as a proxy for reasoning quality.

Key idea: If a model samples 10 predictions for the same step, the *diversity*
of those predictions reveals something about the model's internal uncertainty:
- High type consistency + correct => confident correct reasoning
- High type consistency + wrong type => confident wrong reasoning (action confusion)
- Low type consistency => genuine uncertainty (action exploration failure)
- High type consistency but low coord accuracy => grounding failure

Input:  outputs/eval_c4c7_ac/Qwen2.5-VL-7B/multisample_results.jsonl
Output: outputs/eval_p1/p1_reasoning_analysis.json
"""

import json
import math
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path("/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1")
DATA_PATH = BASE_DIR / "outputs/eval_c4c7_ac/Qwen2.5-VL-7B/multisample_results.jsonl"
OUT_DIR = BASE_DIR / "outputs/eval_p1"
OUT_PATH = OUT_DIR / "p1_reasoning_analysis.json"

# Action types that have coordinate fields (for coord_spread)
COORD_ACTION_TYPES = {"click", "long_press", "swipe", "scroll", "slide", "left_click",
                      "focus", "test", "#click", "window_scroll"}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def shannon_entropy(counts: dict) -> float:
    """Compute Shannon entropy (base-2) from a dict of counts."""
    total = sum(counts.values())
    if total == 0:
        return 0.0
    ent = 0.0
    for c in counts.values():
        if c > 0:
            p = c / total
            ent -= p * math.log2(p)
    return ent


def cohens_d(a: list, b: list) -> float:
    """Compute Cohen's d effect size between two groups."""
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    na, nb = len(a), len(b)
    va, vb = np.var(a, ddof=1), np.var(b, ddof=1)
    pooled_std = math.sqrt(((na - 1) * va + (nb - 1) * vb) / (na + nb - 2))
    if pooled_std == 0:
        return 0.0
    return float((np.mean(a) - np.mean(b)) / pooled_std)


def _is_valid_coord(x: float, y: float, max_val: float = 5000.0) -> bool:
    """Check if coordinates are within a plausible screen range.

    Mobile screens are typically <= ~2560 in either dimension.
    We use a generous max_val=5000 to avoid false positives while
    filtering obviously malformed values (e.g. 2.6e+16).
    """
    return (0 <= x <= max_val) and (0 <= y <= max_val)


def extract_coords(pred_action: dict) -> list:
    """Extract coordinate(s) from a predicted action, returns list of (x,y) tuples.

    Coordinates that fall outside a plausible range are silently dropped.
    """
    if pred_action is None:
        return []
    coords = []
    if "coordinate" in pred_action:
        c = pred_action["coordinate"]
        if isinstance(c, (list, tuple)) and len(c) == 2:
            x, y = float(c[0]), float(c[1])
            if _is_valid_coord(x, y):
                coords.append((x, y))
    if "coordinate2" in pred_action:
        c = pred_action["coordinate2"]
        if isinstance(c, (list, tuple)) and len(c) == 2:
            x, y = float(c[0]), float(c[1])
            if _is_valid_coord(x, y):
                coords.append((x, y))
    return coords


def get_pred_action_type(sample: dict) -> str:
    """Get predicted action type from a sample, handling errors."""
    if sample.get("pred_action") is None:
        return "__parse_error__"
    return sample["pred_action"].get("action", "__unknown__")


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # Load data
    print(f"Loading data from {DATA_PATH} ...")
    episodes = []
    with open(DATA_PATH) as f:
        for line in f:
            episodes.append(json.loads(line.strip()))
    print(f"  Loaded {len(episodes)} episodes")

    total_steps = sum(len(ep["step_samples"]) for ep in episodes)
    print(f"  Total steps: {total_steps}")

    # -----------------------------------------------------------------------
    # Per-step analysis
    # -----------------------------------------------------------------------
    # Collect per-step metrics grouped by error type
    error_type_metrics = defaultdict(lambda: defaultdict(list))
    # confusion_patterns[error_type] = Counter of (gt_type, pred_type) pairs
    confusion_patterns = defaultdict(lambda: Counter())
    # reasoning_layer counts
    reasoning_layers = Counter()
    # cross-tab: error_type x gt_action_type
    cross_tab = defaultdict(lambda: Counter())
    # all step records for detailed output
    all_step_records = []

    for ep in episodes:
        for step_data in ep["step_samples"]:
            step_num = step_data["step_num"]
            gt_action_type = step_data["gt_action_type"]
            samples = step_data["samples"]
            K = len(samples)

            if K == 0:
                continue

            # --- Error type classification (based on greedy = sample 0) ---
            greedy = samples[0]
            greedy_type_match = greedy.get("type_match", False)
            greedy_extract_match = greedy.get("extract_match", False)

            if greedy_type_match and greedy_extract_match:
                error_type = "correct"
            elif not greedy_type_match:
                error_type = "action_error"
            else:
                error_type = "grounding_error"

            # --- Compute per-sample metrics ---
            pred_types = []
            type_matches = []
            extract_matches = []
            all_coords_primary = []  # first coordinate for coord_spread

            for s in samples:
                pt = get_pred_action_type(s)
                pred_types.append(pt)
                type_matches.append(1 if s.get("type_match", False) else 0)
                extract_matches.append(1 if s.get("extract_match", False) else 0)

                # Extract coordinates for spread calculation
                if s.get("pred_action") is not None:
                    coords = extract_coords(s["pred_action"])
                    if coords:
                        all_coords_primary.append(coords[0])  # primary coord

            # --- (a) type_consistency: fraction matching majority type ---
            type_counter = Counter(pred_types)
            majority_type = type_counter.most_common(1)[0][0]
            majority_count = type_counter.most_common(1)[0][1]
            type_consistency = majority_count / K

            # --- (b) type_entropy ---
            type_entropy = shannon_entropy(type_counter)

            # --- (c) correct_rate: fraction with type_match=True ---
            correct_rate = sum(type_matches) / K

            # --- (d) full_correct_rate ---
            full_correct_rate = sum(extract_matches) / K

            # --- (e) confusion_pattern ---
            step_confusion = Counter()
            for pt, tm in zip(pred_types, type_matches):
                if not tm:
                    step_confusion[pt] += 1
            confusion_patterns[error_type] += step_confusion

            # --- (f) coord_spread ---
            coord_spread = float("nan")
            if gt_action_type in {"click", "type", "long_press"} and len(all_coords_primary) >= 2:
                coords_arr = np.array(all_coords_primary)
                # std of x and y, then average
                std_x = np.std(coords_arr[:, 0])
                std_y = np.std(coords_arr[:, 1])
                coord_spread = float((std_x + std_y) / 2.0)

            # --- (g) reasoning_confidence ---
            reasoning_confidence = 1.0 - type_entropy

            # --- (h) reasoning_correctness ---
            reasoning_correctness = correct_rate

            # --- Store metrics by error type ---
            metrics = {
                "type_consistency": type_consistency,
                "type_entropy": type_entropy,
                "correct_rate": correct_rate,
                "full_correct_rate": full_correct_rate,
                "coord_spread": coord_spread,
                "reasoning_confidence": reasoning_confidence,
                "reasoning_correctness": reasoning_correctness,
            }
            for k, v in metrics.items():
                if not (isinstance(v, float) and math.isnan(v)):
                    error_type_metrics[error_type][k].append(v)

            # --- Reasoning layer classification ---
            type_match_rate = sum(type_matches) / K
            extract_match_rate = sum(extract_matches) / K
            if type_match_rate >= 0.7 and extract_match_rate >= 0.7:
                reasoning_layer = "confident_correct"
            elif type_match_rate >= 0.7 and extract_match_rate < 0.7:
                reasoning_layer = "grounding_failure"
            elif type_match_rate < 0.3:
                reasoning_layer = "action_reasoning_failure"
            else:
                reasoning_layer = "action_exploration_failure"
            reasoning_layers[reasoning_layer] += 1

            # --- Cross-tabulation ---
            cross_tab[error_type][gt_action_type] += 1

            # Step record (for detailed inspection)
            all_step_records.append({
                "episode_id": ep["episode_id"],
                "step_num": step_num,
                "gt_action_type": gt_action_type,
                "error_type": error_type,
                "reasoning_layer": reasoning_layer,
                "type_consistency": round(type_consistency, 4),
                "type_entropy": round(type_entropy, 4),
                "correct_rate": round(correct_rate, 4),
                "full_correct_rate": round(full_correct_rate, 4),
                "coord_spread": round(coord_spread, 2) if not math.isnan(coord_spread) else None,
                "num_distinct_types": len(type_counter),
                "majority_type": majority_type,
                "pred_type_dist": dict(type_counter),
            })

    # -----------------------------------------------------------------------
    # Aggregate statistics per error type
    # -----------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("P1: REASONING QUALITY PROXY ANALYSIS")
    print("=" * 80)

    agg_stats = {}
    for etype in ["correct", "action_error", "grounding_error"]:
        metrics_dict = error_type_metrics[etype]
        n = len(metrics_dict.get("type_consistency", []))
        agg = {"n_steps": n}
        for mname in ["type_consistency", "type_entropy", "correct_rate",
                       "full_correct_rate", "coord_spread",
                       "reasoning_confidence", "reasoning_correctness"]:
            vals = metrics_dict.get(mname, [])
            if vals:
                agg[mname] = {
                    "mean": round(float(np.mean(vals)), 4),
                    "std": round(float(np.std(vals)), 4),
                    "median": round(float(np.median(vals)), 4),
                    "n": len(vals),
                }
            else:
                agg[mname] = {"mean": None, "std": None, "median": None, "n": 0}
        agg_stats[etype] = agg

    # Print summary table
    print(f"\n{'Metric':<25} {'correct':>15} {'action_error':>15} {'grounding_error':>18}")
    print("-" * 75)
    for mname in ["type_consistency", "type_entropy", "correct_rate",
                   "full_correct_rate", "coord_spread",
                   "reasoning_confidence", "reasoning_correctness"]:
        vals = []
        for etype in ["correct", "action_error", "grounding_error"]:
            m = agg_stats[etype][mname]["mean"]
            s = agg_stats[etype][mname]["std"]
            if m is not None:
                vals.append(f"{m:.3f} +/- {s:.3f}")
            else:
                vals.append("N/A")
        print(f"{mname:<25} {vals[0]:>15} {vals[1]:>15} {vals[2]:>18}")

    n_correct = agg_stats["correct"]["n_steps"]
    n_action = agg_stats["action_error"]["n_steps"]
    n_ground = agg_stats["grounding_error"]["n_steps"]
    print(f"\n{'N steps':<25} {n_correct:>15} {n_action:>15} {n_ground:>18}")
    print(f"{'Fraction':<25} {n_correct/total_steps:>15.3f} {n_action/total_steps:>15.3f} {n_ground/total_steps:>18.3f}")

    # -----------------------------------------------------------------------
    # Hypothesis testing: effect sizes (Cohen's d)
    # -----------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("HYPOTHESIS TESTING: Cohen's d Effect Sizes")
    print("=" * 80)

    comparisons = [
        ("correct", "action_error"),
        ("correct", "grounding_error"),
        ("action_error", "grounding_error"),
    ]
    effect_sizes = {}
    for (a, b) in comparisons:
        key = f"{a}_vs_{b}"
        effect_sizes[key] = {}
        for mname in ["type_consistency", "type_entropy", "correct_rate",
                       "full_correct_rate", "coord_spread",
                       "reasoning_confidence", "reasoning_correctness"]:
            va = error_type_metrics[a].get(mname, [])
            vb = error_type_metrics[b].get(mname, [])
            d = cohens_d(va, vb)
            effect_sizes[key][mname] = round(d, 4) if not math.isnan(d) else None

    for key, metrics in effect_sizes.items():
        print(f"\n  {key}:")
        for mname, d in metrics.items():
            if d is not None:
                magnitude = "negligible"
                if abs(d) >= 0.8:
                    magnitude = "LARGE"
                elif abs(d) >= 0.5:
                    magnitude = "MEDIUM"
                elif abs(d) >= 0.2:
                    magnitude = "small"
                print(f"    {mname:<30} d = {d:+.3f}  ({magnitude})")
            else:
                print(f"    {mname:<30} d = N/A")

    # -----------------------------------------------------------------------
    # Hypothesis verification summary
    # -----------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("HYPOTHESIS VERIFICATION")
    print("=" * 80)

    def get_mean(etype, metric):
        m = agg_stats[etype][metric]["mean"]
        return m if m is not None else float("nan")

    hypotheses = {}

    # H1: action_error => LOW type_consistency, HIGH type_entropy
    h1_consistency = get_mean("action_error", "type_consistency") < get_mean("correct", "type_consistency")
    h1_entropy = get_mean("action_error", "type_entropy") > get_mean("correct", "type_entropy")
    hypotheses["H1_action_error_low_consistency"] = h1_consistency
    hypotheses["H1_action_error_high_entropy"] = h1_entropy
    print(f"\n  H1a: action_error has LOWER type_consistency than correct: {h1_consistency}")
    print(f"       ({get_mean('action_error', 'type_consistency'):.4f} vs {get_mean('correct', 'type_consistency'):.4f})")
    print(f"  H1b: action_error has HIGHER type_entropy than correct: {h1_entropy}")
    print(f"       ({get_mean('action_error', 'type_entropy'):.4f} vs {get_mean('correct', 'type_entropy'):.4f})")

    # H2: grounding_error => HIGH type_consistency, LOW type_entropy, HIGH coord_spread
    h2_consistency = get_mean("grounding_error", "type_consistency") > get_mean("action_error", "type_consistency")
    h2_entropy = get_mean("grounding_error", "type_entropy") < get_mean("action_error", "type_entropy")
    h2_coord = get_mean("grounding_error", "coord_spread") > get_mean("correct", "coord_spread")
    hypotheses["H2_grounding_high_consistency"] = h2_consistency
    hypotheses["H2_grounding_low_entropy"] = h2_entropy
    hypotheses["H2_grounding_high_coord_spread"] = h2_coord
    print(f"\n  H2a: grounding_error has HIGHER type_consistency than action_error: {h2_consistency}")
    print(f"       ({get_mean('grounding_error', 'type_consistency'):.4f} vs {get_mean('action_error', 'type_consistency'):.4f})")
    print(f"  H2b: grounding_error has LOWER type_entropy than action_error: {h2_entropy}")
    print(f"       ({get_mean('grounding_error', 'type_entropy'):.4f} vs {get_mean('action_error', 'type_entropy'):.4f})")
    print(f"  H2c: grounding_error has HIGHER coord_spread than correct: {h2_coord}")
    print(f"       ({get_mean('grounding_error', 'coord_spread'):.4f} vs {get_mean('correct', 'coord_spread'):.4f})")

    # H3: correct => HIGH type_consistency, LOW type_entropy, LOW coord_spread
    h3_consistency = get_mean("correct", "type_consistency") > get_mean("action_error", "type_consistency")
    h3_entropy = get_mean("correct", "type_entropy") < get_mean("action_error", "type_entropy")
    hypotheses["H3_correct_high_consistency"] = h3_consistency
    hypotheses["H3_correct_low_entropy"] = h3_entropy
    print(f"\n  H3a: correct has HIGHER type_consistency than action_error: {h3_consistency}")
    print(f"       ({get_mean('correct', 'type_consistency'):.4f} vs {get_mean('action_error', 'type_consistency'):.4f})")
    print(f"  H3b: correct has LOWER type_entropy than action_error: {h3_entropy}")
    print(f"       ({get_mean('correct', 'type_entropy'):.4f} vs {get_mean('action_error', 'type_entropy'):.4f})")

    all_pass = all(v for v in hypotheses.values() if not (isinstance(v, float) and math.isnan(v)))
    print(f"\n  ALL HYPOTHESES CONFIRMED: {all_pass}")
    for hname, hval in hypotheses.items():
        status = "PASS" if hval else "FAIL"
        print(f"    {hname}: {status}")

    # -----------------------------------------------------------------------
    # Reasoning layer breakdown
    # -----------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("REASONING LAYER BREAKDOWN")
    print("=" * 80)

    layer_total = sum(reasoning_layers.values())
    reasoning_layer_results = {}
    for layer in ["confident_correct", "grounding_failure",
                   "action_reasoning_failure", "action_exploration_failure"]:
        count = reasoning_layers[layer]
        frac = count / layer_total if layer_total > 0 else 0
        reasoning_layer_results[layer] = {"count": count, "fraction": round(frac, 4)}
        print(f"  {layer:<35} {count:>6} steps  ({frac:.1%})")

    # -----------------------------------------------------------------------
    # Cross-tabulation: error_type x gt_action_type
    # -----------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("CROSS-TABULATION: error_type x gt_action_type")
    print("=" * 80)

    all_action_types = sorted(set(
        at for etype_counts in cross_tab.values() for at in etype_counts
    ))

    cross_tab_results = {}
    # Print header
    header = f"{'action_type':<15}"
    for etype in ["correct", "action_error", "grounding_error"]:
        header += f" {etype:>15}"
    header += f" {'total':>10}"
    print(f"\n{header}")
    print("-" * len(header))

    for at in all_action_types:
        row = f"{at:<15}"
        row_data = {}
        total_at = 0
        for etype in ["correct", "action_error", "grounding_error"]:
            c = cross_tab[etype].get(at, 0)
            row_data[etype] = c
            total_at += c
            row += f" {c:>15}"
        row += f" {total_at:>10}"
        row_data["total"] = total_at
        # Compute error rates for this action type
        if total_at > 0:
            row_data["action_error_rate"] = round(row_data["action_error"] / total_at, 4)
            row_data["grounding_error_rate"] = round(row_data["grounding_error"] / total_at, 4)
            row_data["correct_rate"] = round(row_data["correct"] / total_at, 4)
        cross_tab_results[at] = row_data
        print(row)

    # -----------------------------------------------------------------------
    # Confusion patterns: what does the model confuse with?
    # -----------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("CONFUSION PATTERNS (wrong predicted types across K samples)")
    print("=" * 80)

    confusion_results = {}
    for etype in ["correct", "action_error", "grounding_error"]:
        cp = confusion_patterns[etype]
        total_wrong = sum(cp.values())
        if total_wrong > 0:
            top10 = cp.most_common(10)
            confusion_results[etype] = {
                "total_wrong_predictions": total_wrong,
                "top_confused_types": [
                    {"type": t, "count": c, "fraction": round(c / total_wrong, 4)}
                    for t, c in top10
                ]
            }
            print(f"\n  {etype} (total wrong preds: {total_wrong}):")
            for t, c in top10:
                print(f"    {t:<25} {c:>6} ({c/total_wrong:.1%})")
        else:
            confusion_results[etype] = {"total_wrong_predictions": 0, "top_confused_types": []}
            print(f"\n  {etype}: no wrong predictions among samples")

    # -----------------------------------------------------------------------
    # Per-action-type reasoning profile
    # -----------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("PER-ACTION-TYPE REASONING PROFILE")
    print("=" * 80)

    # Collect metrics grouped by gt_action_type
    action_type_metrics = defaultdict(lambda: defaultdict(list))
    for rec in all_step_records:
        at = rec["gt_action_type"]
        for mname in ["type_consistency", "type_entropy", "correct_rate",
                       "full_correct_rate"]:
            action_type_metrics[at][mname].append(rec[mname])
        if rec["coord_spread"] is not None:
            action_type_metrics[at]["coord_spread"].append(rec["coord_spread"])

    action_type_profiles = {}
    print(f"\n{'action_type':<15} {'N':>5} {'consistency':>12} {'entropy':>10} {'correct_rate':>13} {'full_rate':>10} {'coord_spread':>13}")
    print("-" * 80)
    for at in sorted(action_type_metrics.keys()):
        metrics = action_type_metrics[at]
        n = len(metrics["type_consistency"])
        profile = {"n_steps": n}
        vals = []
        for mname in ["type_consistency", "type_entropy", "correct_rate", "full_correct_rate", "coord_spread"]:
            v = metrics.get(mname, [])
            if v:
                profile[mname] = {"mean": round(float(np.mean(v)), 4), "std": round(float(np.std(v)), 4)}
                vals.append(f"{np.mean(v):.3f}")
            else:
                profile[mname] = {"mean": None, "std": None}
                vals.append("N/A")
        action_type_profiles[at] = profile
        print(f"{at:<15} {n:>5} {vals[0]:>12} {vals[1]:>10} {vals[2]:>13} {vals[3]:>10} {vals[4]:>13}")

    # -----------------------------------------------------------------------
    # Reasoning layer x error type cross-tab
    # -----------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("REASONING LAYER x ERROR TYPE CROSS-TAB")
    print("=" * 80)

    layer_error_cross = defaultdict(lambda: Counter())
    for rec in all_step_records:
        layer_error_cross[rec["reasoning_layer"]][rec["error_type"]] += 1

    layer_error_results = {}
    layers_list = ["confident_correct", "grounding_failure",
                   "action_reasoning_failure", "action_exploration_failure"]
    header = f"{'reasoning_layer':<35}"
    for etype in ["correct", "action_error", "grounding_error"]:
        header += f" {etype:>15}"
    header += f" {'total':>8}"
    print(f"\n{header}")
    print("-" * len(header))

    for layer in layers_list:
        row = f"{layer:<35}"
        row_data = {}
        total_l = 0
        for etype in ["correct", "action_error", "grounding_error"]:
            c = layer_error_cross[layer].get(etype, 0)
            row_data[etype] = c
            total_l += c
            row += f" {c:>15}"
        row += f" {total_l:>8}"
        row_data["total"] = total_l
        layer_error_results[layer] = row_data
        print(row)

    # -----------------------------------------------------------------------
    # Build and save results JSON
    # -----------------------------------------------------------------------
    results = {
        "experiment": "P1: Reasoning Quality Proxy from Multi-Sample Diversity",
        "data_source": str(DATA_PATH),
        "n_episodes": len(episodes),
        "n_total_steps": total_steps,

        "error_type_distribution": {
            "correct": n_correct,
            "action_error": n_action,
            "grounding_error": n_ground,
            "correct_frac": round(n_correct / total_steps, 4),
            "action_error_frac": round(n_action / total_steps, 4),
            "grounding_error_frac": round(n_ground / total_steps, 4),
        },

        "aggregate_metrics_by_error_type": agg_stats,

        "effect_sizes_cohens_d": effect_sizes,

        "hypotheses": hypotheses,
        "all_hypotheses_confirmed": all_pass,

        "reasoning_layer_breakdown": reasoning_layer_results,
        "reasoning_layer_x_error_type": layer_error_results,

        "cross_tab_error_x_action": cross_tab_results,
        "confusion_patterns": confusion_results,
        "action_type_profiles": action_type_profiles,
    }

    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n\nResults saved to {OUT_PATH}")
    print(f"Total steps analyzed: {total_steps}")
    print("Done.")


if __name__ == "__main__":
    main()
