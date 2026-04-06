#!/usr/bin/env python3
"""
P5: Goal Reasoning Analysis of the action_exploration_failure Layer

The action_exploration_failure layer (~15% of steps from P1) contains steps
where the model shows mixed action-type agreement but K=10 contains some
correct answers. This script determines whether these represent:

  (a) noisy_action: the model knows WHAT to do (correct action type = majority
      type) but is uncertain about execution details (coordinates, args).
  (b) goal_confusion: the model is genuinely confused about the goal -- the
      correct action type differs from the majority vote among K=10 samples.

Layer classification recap (from P1):
  - For each step, sample[0] is greedy:
      greedy extract_match=true  -> "correct"
      greedy type_match=false    -> "action_error"
      greedy type_match=true, extract_match=false -> "grounding_error"
  - Subdivide using K=10 agreement:
      correct + type_consistency >= 0.8        -> confident_correct
      grounding_error + type_consistency >= 0.8 -> grounding_failure
      action_error + type_consistency < 0.5 OR always wrong -> action_reasoning_failure
      Everything else                          -> action_exploration_failure

For action_exploration_failure steps, type_consistency is defined as:
  fraction of K=10 samples with the same action type as the most common type.

Analysis:
  For each step in the layer, compare the oracle action type (the correct
  sample from K=10, or GT if none correct) against the K=10 majority type:
    oracle_type == majority_type  -> noisy_action
    oracle_type != majority_type  -> goal_confusion

Then compare these two subclasses across 10 dimensions with effect sizes
and statistical tests.

Input:  outputs/eval_c4c7_ac/Qwen2.5-VL-7B/multisample_results.jsonl
Output: outputs/eval_p5/p5_goal_reasoning.json
"""

import json
import math
import os
from collections import Counter, defaultdict

import numpy as np
from scipy import stats as scipy_stats

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = "/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1"
DATA_PATH = os.path.join(
    BASE_DIR, "outputs/eval_c4c7_ac/Qwen2.5-VL-7B/multisample_results.jsonl"
)
OUT_DIR = os.path.join(BASE_DIR, "outputs/eval_p5")
OUT_PATH = os.path.join(OUT_DIR, "p5_goal_reasoning.json")

os.makedirs(OUT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_jsonl(path):
    data = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


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


def cohens_d(a, b):
    """Compute Cohen's d effect size between two groups."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    na, nb = len(a), len(b)
    va, vb = np.var(a, ddof=1), np.var(b, ddof=1)
    pooled_std = math.sqrt(((na - 1) * va + (nb - 1) * vb) / (na + nb - 2))
    if pooled_std == 0:
        return 0.0
    return float((np.mean(a) - np.mean(b)) / pooled_std)


def _is_valid_coord(x, y, max_val=5000.0):
    return (0 <= x <= max_val) and (0 <= y <= max_val)


def extract_coords(pred_action):
    """Extract coordinate(s) from a predicted action, returns list of (x,y)."""
    if pred_action is None:
        return []
    coords = []
    if "coordinate" in pred_action:
        c = pred_action["coordinate"]
        if isinstance(c, (list, tuple)) and len(c) == 2:
            try:
                x, y = float(c[0]), float(c[1])
                if _is_valid_coord(x, y):
                    coords.append((x, y))
            except (TypeError, ValueError):
                pass
    if "coordinate2" in pred_action:
        c = pred_action["coordinate2"]
        if isinstance(c, (list, tuple)) and len(c) == 2:
            try:
                x, y = float(c[0]), float(c[1])
                if _is_valid_coord(x, y):
                    coords.append((x, y))
            except (TypeError, ValueError):
                pass
    return coords


def get_pred_action_type(sample):
    """Get predicted action type from a sample, handling errors/nulls."""
    if sample.get("pred_action") is None:
        return "__parse_error__"
    return sample["pred_action"].get("action", "__unknown__")


def safe_mean(vals):
    if not vals:
        return float("nan")
    return float(np.mean(vals))


def safe_std(vals):
    if len(vals) < 2:
        return float("nan")
    return float(np.std(vals, ddof=1))


def safe_median(vals):
    if not vals:
        return float("nan")
    return float(np.median(vals))


def mann_whitney(a, b):
    """Mann-Whitney U test. Returns (U, p_value)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) < 2 or len(b) < 2:
        return float("nan"), float("nan")
    try:
        u, p = scipy_stats.mannwhitneyu(a, b, alternative="two-sided")
        return float(u), float(p)
    except ValueError:
        return float("nan"), float("nan")


def welch_t(a, b):
    """Welch's t-test. Returns (t, p_value)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) < 2 or len(b) < 2:
        return float("nan"), float("nan")
    try:
        t, p = scipy_stats.ttest_ind(a, b, equal_var=False)
        return float(t), float(p)
    except Exception:
        return float("nan"), float("nan")


def magnitude_label(d_val):
    if math.isnan(d_val):
        return "N/A"
    ad = abs(d_val)
    if ad >= 0.8:
        return "LARGE"
    elif ad >= 0.5:
        return "MEDIUM"
    elif ad >= 0.2:
        return "small"
    return "negligible"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 80)
    print("P5: GOAL REASONING ANALYSIS OF action_exploration_failure LAYER")
    print("=" * 80)

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    print(f"\nLoading data from {DATA_PATH} ...")
    episodes = load_jsonl(DATA_PATH)
    total_steps = sum(len(ep["step_samples"]) for ep in episodes)
    print(f"  Loaded {len(episodes)} episodes, {total_steps} total steps")

    # ------------------------------------------------------------------
    # 2. Classify every step into reasoning layers (P1 logic)
    # ------------------------------------------------------------------
    layer_counts = Counter()
    exploration_steps = []  # the steps we care about

    for ep in episodes:
        num_steps = ep["num_steps"]
        for step_data in ep["step_samples"]:
            step_num = step_data["step_num"]
            gt_action_type = step_data["gt_action_type"]
            samples = step_data["samples"]
            K = len(samples)
            if K == 0:
                continue

            greedy = samples[0]
            greedy_type_match = greedy.get("type_match", False)
            greedy_extract_match = greedy.get("extract_match", False)

            # --- base error type ---
            if greedy_type_match and greedy_extract_match:
                error_type = "correct"
            elif not greedy_type_match:
                error_type = "action_error"
            else:
                error_type = "grounding_error"

            # --- K=10 metrics ---
            pred_types = [get_pred_action_type(s) for s in samples]
            type_matches = [s.get("type_match", False) for s in samples]
            extract_matches = [s.get("extract_match", False) for s in samples]

            type_counter = Counter(pred_types)
            majority_type = type_counter.most_common(1)[0][0]
            majority_count = type_counter.most_common(1)[0][1]
            type_consistency = majority_count / K

            type_match_rate = sum(type_matches) / K
            extract_match_rate = sum(extract_matches) / K

            # --- reasoning layer (P1 thresholds) ---
            if type_match_rate >= 0.7 and extract_match_rate >= 0.7:
                layer = "confident_correct"
            elif type_match_rate >= 0.7 and extract_match_rate < 0.7:
                layer = "grounding_failure"
            elif type_match_rate < 0.3:
                layer = "action_reasoning_failure"
            else:
                layer = "action_exploration_failure"

            layer_counts[layer] += 1

            if layer != "action_exploration_failure":
                continue

            # ---- Collect rich info for exploration steps ----

            # Find oracle action: first correct sample from K=10
            oracle_type = None
            oracle_idx = None
            for idx, s in enumerate(samples):
                if s.get("extract_match", False):
                    oracle_type = get_pred_action_type(s)
                    oracle_idx = idx
                    break
            # Fallback to GT
            if oracle_type is None:
                oracle_type = gt_action_type

            # Type entropy
            type_entropy = shannon_entropy(type_counter)

            # Number of unique action types
            n_unique_types = len(type_counter)

            # Full correct rate in K=10
            full_correct_rate = sum(extract_matches) / K

            # Coordinate spread (for coordinate-bearing action types)
            all_coords = []
            for s in samples:
                if s.get("pred_action") is not None:
                    coords = extract_coords(s["pred_action"])
                    if coords:
                        all_coords.append(coords[0])
            coord_spread = float("nan")
            if len(all_coords) >= 2:
                coords_arr = np.array(all_coords)
                std_x = np.std(coords_arr[:, 0])
                std_y = np.std(coords_arr[:, 1])
                coord_spread = float((std_x + std_y) / 2.0)

            # Relative step position
            rel_pos = step_num / max(num_steps - 1, 1)

            # P5 classification
            if oracle_type == majority_type:
                subclass = "noisy_action"
            else:
                subclass = "goal_confusion"

            # Check: is the greedy wrong but the majority correct?
            # (relevant for noisy_action fraction analysis)
            greedy_wrong = not greedy_extract_match
            majority_is_oracle = (majority_type == oracle_type)

            exploration_steps.append({
                "episode_id": ep["episode_id"],
                "step_num": step_num,
                "num_steps": num_steps,
                "rel_pos": rel_pos,
                "gt_action_type": gt_action_type,
                "oracle_type": oracle_type,
                "oracle_idx": oracle_idx,
                "majority_type": majority_type,
                "type_consistency": type_consistency,
                "type_entropy": type_entropy,
                "n_unique_types": n_unique_types,
                "full_correct_rate": full_correct_rate,
                "coord_spread": coord_spread,
                "subclass": subclass,
                "error_type": error_type,
                "greedy_wrong": greedy_wrong,
                "majority_is_oracle": majority_is_oracle,
                "type_counter": dict(type_counter),
            })

    # ------------------------------------------------------------------
    # 3. Print layer breakdown
    # ------------------------------------------------------------------
    print("\n--- Reasoning Layer Breakdown ---")
    for layer_name in ["confident_correct", "grounding_failure",
                       "action_reasoning_failure", "action_exploration_failure"]:
        c = layer_counts[layer_name]
        frac = c / total_steps if total_steps > 0 else 0
        print(f"  {layer_name:<35} {c:>6} steps  ({frac:.1%})")

    n_explore = len(exploration_steps)
    print(f"\n  action_exploration_failure steps to analyze: {n_explore}")

    if n_explore == 0:
        print("  No exploration steps found. Exiting.")
        results = {
            "experiment": "P5: Goal Reasoning Analysis",
            "error": "No action_exploration_failure steps found",
            "layer_counts": dict(layer_counts),
        }
        with open(OUT_PATH, "w") as f:
            json.dump(results, f, indent=2)
        return

    # ------------------------------------------------------------------
    # 4. Split into subclasses
    # ------------------------------------------------------------------
    noisy = [s for s in exploration_steps if s["subclass"] == "noisy_action"]
    confused = [s for s in exploration_steps if s["subclass"] == "goal_confusion"]

    print(f"\n--- P5 Subclass Breakdown ---")
    print(f"  noisy_action:  {len(noisy):>6} ({len(noisy)/n_explore:.1%} of exploration layer)")
    print(f"  goal_confusion: {len(confused):>6} ({len(confused)/n_explore:.1%} of exploration layer)")

    # ------------------------------------------------------------------
    # 5. Feature extraction per subclass
    # ------------------------------------------------------------------
    features = [
        "rel_pos",            # 1. Step position (relative)
        "num_steps",          # 2. Episode length
        "type_consistency",   # 3. Type consistency
        "type_entropy",       # 4. Type entropy
        "n_unique_types",     # 5. Number of unique action types in K=10
        "full_correct_rate",  # 6. Oracle correct rate in K=10
        "coord_spread",       # 7. Coordinate spread
    ]

    def gather_feature(steps, feat_name):
        """Collect non-NaN values for a feature."""
        vals = []
        for s in steps:
            v = s[feat_name]
            if isinstance(v, float) and math.isnan(v):
                continue
            vals.append(float(v))
        return vals

    noisy_feats = {f: gather_feature(noisy, f) for f in features}
    confused_feats = {f: gather_feature(confused, f) for f in features}

    # ------------------------------------------------------------------
    # 6. Descriptive stats
    # ------------------------------------------------------------------
    print("\n--- Descriptive Statistics ---")
    header = f"{'Feature':<22} {'noisy_action':>30} {'goal_confusion':>30}"
    print(header)
    print("-" * len(header))

    desc_stats = {}
    for f in features:
        nv = noisy_feats[f]
        cv = confused_feats[f]
        n_str = f"{safe_mean(nv):.4f} +/- {safe_std(nv):.4f} (n={len(nv)})"
        c_str = f"{safe_mean(cv):.4f} +/- {safe_std(cv):.4f} (n={len(cv)})"
        print(f"  {f:<20} {n_str:>30} {c_str:>30}")
        desc_stats[f] = {
            "noisy_action": {
                "mean": round(safe_mean(nv), 4),
                "std": round(safe_std(nv), 4),
                "median": round(safe_median(nv), 4),
                "n": len(nv),
            },
            "goal_confusion": {
                "mean": round(safe_mean(cv), 4),
                "std": round(safe_std(cv), 4),
                "median": round(safe_median(cv), 4),
                "n": len(cv),
            },
        }

    # ------------------------------------------------------------------
    # 7. GT action type distribution (#8)
    # ------------------------------------------------------------------
    print("\n--- GT Action Type Distribution ---")
    noisy_gt = Counter(s["gt_action_type"] for s in noisy)
    confused_gt = Counter(s["gt_action_type"] for s in confused)
    all_gt_types = sorted(set(list(noisy_gt.keys()) + list(confused_gt.keys())))

    print(f"  {'action_type':<18} {'noisy_action':>15} {'goal_confusion':>15}")
    print("  " + "-" * 50)
    gt_dist_results = {}
    for at in all_gt_types:
        nc = noisy_gt.get(at, 0)
        cc = confused_gt.get(at, 0)
        n_frac = nc / len(noisy) if noisy else 0
        c_frac = cc / len(confused) if confused else 0
        print(f"  {at:<18} {nc:>6} ({n_frac:.1%})   {cc:>6} ({c_frac:.1%})")
        gt_dist_results[at] = {
            "noisy_action": {"count": nc, "fraction": round(n_frac, 4)},
            "goal_confusion": {"count": cc, "fraction": round(c_frac, 4)},
        }

    # ------------------------------------------------------------------
    # 8. Cohen's d effect sizes (#9)
    # ------------------------------------------------------------------
    print("\n--- Cohen's d Effect Sizes (noisy_action vs goal_confusion) ---")
    effect_sizes = {}
    for f in features:
        d_val = cohens_d(noisy_feats[f], confused_feats[f])
        mag = magnitude_label(d_val)
        effect_sizes[f] = {
            "d": round(d_val, 4) if not math.isnan(d_val) else None,
            "magnitude": mag,
        }
        d_str = f"{d_val:+.4f}" if not math.isnan(d_val) else "N/A"
        print(f"  {f:<22} d = {d_str:>8}  ({mag})")

    # ------------------------------------------------------------------
    # 9. Statistical tests (#10) -- Welch t and Mann-Whitney U
    # ------------------------------------------------------------------
    print("\n--- Statistical Tests (noisy_action vs goal_confusion) ---")
    stat_tests = {}
    for f in features:
        nv = noisy_feats[f]
        cv = confused_feats[f]
        t_val, t_p = welch_t(nv, cv)
        u_val, u_p = mann_whitney(nv, cv)
        stat_tests[f] = {
            "welch_t": {
                "t": round(t_val, 4) if not math.isnan(t_val) else None,
                "p": round(t_p, 6) if not math.isnan(t_p) else None,
            },
            "mann_whitney_u": {
                "U": round(u_val, 1) if not math.isnan(u_val) else None,
                "p": round(u_p, 6) if not math.isnan(u_p) else None,
            },
        }
        t_str = f"t={t_val:+.3f}, p={t_p:.4f}" if not math.isnan(t_val) else "N/A"
        u_str = f"U={u_val:.0f}, p={u_p:.4f}" if not math.isnan(u_val) else "N/A"
        sig_t = " *" if (not math.isnan(t_p) and t_p < 0.05) else ""
        sig_u = " *" if (not math.isnan(u_p) and u_p < 0.05) else ""
        print(f"  {f:<22} Welch: {t_str}{sig_t}  |  MWU: {u_str}{sig_u}")

    # ------------------------------------------------------------------
    # 10. noisy_action subclass: fraction that are "correct but low agreement"
    # ------------------------------------------------------------------
    print("\n--- noisy_action: Correct-but-Low-Agreement Analysis ---")
    # "correct steps with low agreement" = greedy is correct
    # (extract_match=true for sample 0) but ended up in exploration layer
    # because type_consistency was moderate (0.3-0.7 range)
    noisy_greedy_correct = sum(1 for s in noisy if not s["greedy_wrong"])
    noisy_greedy_wrong_but_majority_correct = sum(
        1 for s in noisy if s["greedy_wrong"] and s["majority_is_oracle"]
    )

    print(f"  Total noisy_action steps: {len(noisy)}")
    print(f"  Greedy correct (correct step, low agreement): "
          f"{noisy_greedy_correct} ({noisy_greedy_correct / max(len(noisy), 1):.1%})")
    print(f"  Greedy wrong but majority matches oracle: "
          f"{noisy_greedy_wrong_but_majority_correct} "
          f"({noisy_greedy_wrong_but_majority_correct / max(len(noisy), 1):.1%})")

    # Also for goal_confusion
    confused_greedy_correct = sum(1 for s in confused if not s["greedy_wrong"])
    print(f"\n  goal_confusion greedy correct (for comparison): "
          f"{confused_greedy_correct} ({confused_greedy_correct / max(len(confused), 1):.1%})")

    # ------------------------------------------------------------------
    # 11. Supplementary: Majority type confusion matrix
    # ------------------------------------------------------------------
    print("\n--- Majority Type vs Oracle Type (goal_confusion only) ---")
    majority_oracle_pairs = Counter()
    for s in confused:
        majority_oracle_pairs[(s["majority_type"], s["oracle_type"])] += 1

    print(f"  {'majority_type':<20} {'oracle_type':<20} {'count':>6}")
    print("  " + "-" * 48)
    for (maj, orc), cnt in majority_oracle_pairs.most_common(20):
        print(f"  {maj:<20} {orc:<20} {cnt:>6}")

    majority_oracle_results = [
        {"majority_type": maj, "oracle_type": orc, "count": cnt}
        for (maj, orc), cnt in majority_oracle_pairs.most_common(30)
    ]

    # ------------------------------------------------------------------
    # 12. Step position histogram (deeper analysis for #1 and #2)
    # ------------------------------------------------------------------
    print("\n--- Step Position Analysis ---")

    # Relative position buckets
    def bucket_rel(val):
        if val <= 0.33:
            return "first_third"
        elif val <= 0.66:
            return "mid_third"
        else:
            return "last_third"

    noisy_buckets = Counter(bucket_rel(s["rel_pos"]) for s in noisy)
    confused_buckets = Counter(bucket_rel(s["rel_pos"]) for s in confused)

    pos_analysis = {}
    for b in ["first_third", "mid_third", "last_third"]:
        n_cnt = noisy_buckets.get(b, 0)
        c_cnt = confused_buckets.get(b, 0)
        n_frac = n_cnt / max(len(noisy), 1)
        c_frac = c_cnt / max(len(confused), 1)
        pos_analysis[b] = {
            "noisy_action": {"count": n_cnt, "fraction": round(n_frac, 4)},
            "goal_confusion": {"count": c_cnt, "fraction": round(c_frac, 4)},
        }
        print(f"  {b:<15} noisy: {n_cnt:>5} ({n_frac:.1%})   "
              f"confused: {c_cnt:>5} ({c_frac:.1%})")

    # Episode length buckets
    def bucket_len(n):
        if n <= 3:
            return "short(1-3)"
        elif n <= 7:
            return "mid(4-7)"
        elif n <= 15:
            return "long(8-15)"
        else:
            return "vlong(16+)"

    noisy_len_buckets = Counter(bucket_len(s["num_steps"]) for s in noisy)
    confused_len_buckets = Counter(bucket_len(s["num_steps"]) for s in confused)

    print("\n  Episode length distribution:")
    len_analysis = {}
    for b in ["short(1-3)", "mid(4-7)", "long(8-15)", "vlong(16+)"]:
        n_cnt = noisy_len_buckets.get(b, 0)
        c_cnt = confused_len_buckets.get(b, 0)
        n_frac = n_cnt / max(len(noisy), 1)
        c_frac = c_cnt / max(len(confused), 1)
        len_analysis[b] = {
            "noisy_action": {"count": n_cnt, "fraction": round(n_frac, 4)},
            "goal_confusion": {"count": c_cnt, "fraction": round(c_frac, 4)},
        }
        print(f"  {b:<15} noisy: {n_cnt:>5} ({n_frac:.1%})   "
              f"confused: {c_cnt:>5} ({c_frac:.1%})")

    # ------------------------------------------------------------------
    # 13. Summary and verdict
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("SUMMARY AND VERDICT")
    print("=" * 80)

    n_significant = sum(
        1 for f in features
        if stat_tests[f]["mann_whitney_u"]["p"] is not None
        and stat_tests[f]["mann_whitney_u"]["p"] < 0.05
    )
    n_large_effect = sum(
        1 for f in features
        if effect_sizes[f]["d"] is not None
        and abs(effect_sizes[f]["d"]) >= 0.5
    )

    goal_confusion_frac = len(confused) / n_explore if n_explore > 0 else 0

    print(f"\n  action_exploration_failure steps: {n_explore} "
          f"({n_explore / total_steps:.1%} of all steps)")
    print(f"  noisy_action:   {len(noisy):>5} ({len(noisy) / n_explore:.1%})")
    print(f"  goal_confusion: {len(confused):>5} ({len(confused) / n_explore:.1%})")
    print(f"\n  Features with significant MWU p < 0.05: {n_significant}/{len(features)}")
    print(f"  Features with |Cohen's d| >= 0.5:       {n_large_effect}/{len(features)}")

    # Verdict
    if goal_confusion_frac < 0.15:
        verdict = ("MOSTLY NOISE: goal_confusion is a small minority (<15%). "
                    "The exploration layer is dominated by noisy_action, "
                    "i.e., execution uncertainty, not goal confusion.")
    elif goal_confusion_frac > 0.50 and n_significant >= 3:
        verdict = ("REAL GOAL REASONING FAILURE: goal_confusion is the majority, "
                    "and the two subclasses differ significantly on multiple features. "
                    "This layer contains genuine goal-level confusion.")
    elif n_significant >= 3 and n_large_effect >= 2:
        verdict = ("MIXED BUT SEPARABLE: Both subclasses are present and "
                    "statistically distinguishable. The exploration layer "
                    "contains a real goal_confusion component worth targeting.")
    else:
        verdict = ("HARD TO SEPARATE: The two subclasses are not clearly "
                    "distinguishable. The exploration layer may be a gradient "
                    "rather than two discrete failure modes.")

    print(f"\n  VERDICT: {verdict}")

    # ------------------------------------------------------------------
    # 14. Build and save results JSON
    # ------------------------------------------------------------------
    results = {
        "experiment": "P5: Goal Reasoning Analysis of action_exploration_failure Layer",
        "data_source": DATA_PATH,
        "n_episodes": len(episodes),
        "n_total_steps": total_steps,

        "reasoning_layer_breakdown": {
            layer_name: {
                "count": layer_counts[layer_name],
                "fraction": round(layer_counts[layer_name] / total_steps, 4),
            }
            for layer_name in ["confident_correct", "grounding_failure",
                               "action_reasoning_failure", "action_exploration_failure"]
        },

        "n_exploration_steps": n_explore,
        "exploration_fraction_of_total": round(n_explore / total_steps, 4),

        "subclass_breakdown": {
            "noisy_action": {
                "count": len(noisy),
                "fraction_of_exploration": round(len(noisy) / max(n_explore, 1), 4),
            },
            "goal_confusion": {
                "count": len(confused),
                "fraction_of_exploration": round(len(confused) / max(n_explore, 1), 4),
            },
        },

        "descriptive_stats": desc_stats,
        "gt_action_type_distribution": gt_dist_results,
        "effect_sizes_cohens_d": effect_sizes,
        "statistical_tests": stat_tests,

        "step_position_analysis": pos_analysis,
        "episode_length_analysis": len_analysis,

        "noisy_action_analysis": {
            "total": len(noisy),
            "greedy_correct_low_agreement": noisy_greedy_correct,
            "greedy_correct_fraction": round(
                noisy_greedy_correct / max(len(noisy), 1), 4
            ),
            "greedy_wrong_but_majority_matches_oracle": noisy_greedy_wrong_but_majority_correct,
            "greedy_wrong_majority_oracle_fraction": round(
                noisy_greedy_wrong_but_majority_correct / max(len(noisy), 1), 4
            ),
        },

        "goal_confusion_analysis": {
            "total": len(confused),
            "greedy_correct": confused_greedy_correct,
            "greedy_correct_fraction": round(
                confused_greedy_correct / max(len(confused), 1), 4
            ),
        },

        "majority_vs_oracle_type_pairs": majority_oracle_results,

        "verdict": verdict,
        "n_significant_features": n_significant,
        "n_large_effect_features": n_large_effect,
    }

    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nResults saved to {OUT_PATH}")
    print("Done.")


if __name__ == "__main__":
    main()
