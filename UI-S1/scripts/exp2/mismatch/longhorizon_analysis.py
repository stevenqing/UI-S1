"""Long-Horizon Feature Analysis (Exp2e extended).

Analyzes how per-step accuracy varies as a function of long-horizon context features.
All analysis is on existing C0/F4/oracle data — no new inference needed.

Three analysis stages:
  Step 1: Univariate analysis — accuracy curve per feature
  Step 2: Multivariate logistic regression — feature importance
  Step 3: Feature interactions — key cross-feature patterns
"""

import argparse
import json
import math
import os
import sys
import warnings
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
from verifier_ar_inference import format_action_brief


# ---------------------------------------------------------------------------
# Feature computation
# ---------------------------------------------------------------------------

def compute_features(c0_trajs, raw_trajs):
    """Compute all long-horizon features for each step.

    Returns list of dicts, each with: sample_id, trajectory_id, domain, correct,
    and all feature values.
    """
    all_steps = []

    for traj in c0_trajs:
        traj_id = traj["trajectory_id"]
        domain = traj["domain"]
        step_results = traj["step_results"]
        num_steps = len(step_results)
        raw = raw_trajs.get(traj_id, [])

        # Build subtask segments from raw data
        segments = segment_by_subtask(raw)
        # Map sample_id -> (subtask_idx, position_in_subtask, subtask_length)
        subtask_map = {}
        for seg_idx, (subtask_desc, seg_steps) in enumerate(segments):
            for local_idx, seg_step in enumerate(seg_steps):
                subtask_map[seg_step["sample_id"]] = (seg_idx, local_idx, len(seg_steps))

        # Track running state for error accumulation features
        preceding_wrong = 0
        preceding_correct = 0
        consecutive_wrong = 0
        steps_since_last_correct = 0
        history_actions = []  # list of (fn, args) for history length computation

        # Track per-subtask state
        current_subtask_idx = -1
        subtask_preceding_wrong = 0
        subtask_step_count = 0

        for step_idx, sr in enumerate(step_results):
            sample_id = sr["sample_id"]
            correct = sr.get("success", False)

            # --- Position features ---
            step_index = step_idx  # 0-based
            relative_position = step_idx / max(num_steps - 1, 1)
            steps_remaining = num_steps - 1 - step_idx

            # --- Subtask context features ---
            seg_idx, pos_in_subtask, subtask_length = subtask_map.get(
                sample_id, (0, step_idx, num_steps))

            is_subtask_boundary = (pos_in_subtask == 0)

            # Reset subtask-local counters on boundary
            if seg_idx != current_subtask_idx:
                current_subtask_idx = seg_idx
                subtask_preceding_wrong = 0
                subtask_step_count = 0

            # --- Error accumulation features ---
            preceding_wrong_rate = preceding_wrong / max(step_idx, 1)

            # --- History length features ---
            # Compute approximate history token count
            # History is reset per subtask in subtask_isolated mode
            subtask_history_actions = []
            for prev_idx in range(step_idx):
                prev_sr = step_results[prev_idx]
                prev_seg = subtask_map.get(prev_sr["sample_id"], (None,))[0]
                if prev_seg == seg_idx:
                    fn = prev_sr.get("predicted_function", "unknown")
                    args = prev_sr.get("predicted_args", {})
                    subtask_history_actions.append((fn, args))

            history_action_count = len(subtask_history_actions)
            # Approximate token count: ~4 tokens per word in format_action_brief
            history_text_parts = []
            for i, (fn, args) in enumerate(subtask_history_actions):
                brief = format_action_brief(fn, args)
                history_text_parts.append(f"Step {i+1}: {brief}")
            history_text = "\n".join(history_text_parts) if history_text_parts else "None"
            # Rough token estimate: split by whitespace, each word ~1.3 tokens
            history_token_count = int(len(history_text.split()) * 1.3)

            # --- Mismatch magnitude ---
            # Count how many preceding actions in this subtask were wrong
            mismatch_count = 0
            for prev_idx in range(step_idx):
                prev_sr = step_results[prev_idx]
                prev_seg = subtask_map.get(prev_sr["sample_id"], (None,))[0]
                if prev_seg == seg_idx and not prev_sr.get("success", False):
                    mismatch_count += 1
            # Mismatch rate: fraction of history that's wrong
            mismatch_magnitude = mismatch_count / max(history_action_count, 1)

            step_data = {
                "sample_id": sample_id,
                "trajectory_id": traj_id,
                "domain": domain,
                "correct": correct,
                # Position features
                "step_index": step_index,
                "relative_position": relative_position,
                "steps_remaining": steps_remaining,
                "num_steps": num_steps,
                # Error accumulation features
                "preceding_wrong": subtask_preceding_wrong,
                "preceding_wrong_global": preceding_wrong,
                "preceding_wrong_rate": preceding_wrong_rate,
                "consecutive_wrong": consecutive_wrong,
                "steps_since_last_correct": steps_since_last_correct,
                # Subtask context features
                "subtask_index": seg_idx,
                "position_in_subtask": pos_in_subtask,
                "subtask_length": subtask_length,
                "is_subtask_boundary": is_subtask_boundary,
                # History length features
                "history_token_count": history_token_count,
                "history_action_count": history_action_count,
                "mismatch_magnitude": mismatch_magnitude,
                "mismatch_count": mismatch_count,
            }
            all_steps.append(step_data)

            # Update running state
            if correct:
                preceding_correct += 1
                consecutive_wrong = 0
                steps_since_last_correct = 0
            else:
                preceding_wrong += 1
                consecutive_wrong += 1
                steps_since_last_correct += 1
                subtask_preceding_wrong += 1

            subtask_step_count += 1

            # For steps_since_last_correct: increment for all steps,
            # but reset happens above when correct=True
            if correct:
                steps_since_last_correct = 0
            else:
                steps_since_last_correct += 1

    return all_steps


# ---------------------------------------------------------------------------
# Step 1: Univariate analysis
# ---------------------------------------------------------------------------

BUCKET_CONFIGS = {
    "step_index": {
        "buckets": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        "overflow": "10+",
        "key": "step_index",
    },
    "relative_position": {
        "ranges": [(0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.01)],
        "labels": ["0-20%", "20-40%", "40-60%", "60-80%", "80-100%"],
        "key": "relative_position",
    },
    "preceding_wrong": {
        "buckets": [0, 1, 2, 3, 4],
        "overflow": "5+",
        "key": "preceding_wrong",
    },
    "consecutive_wrong": {
        "buckets": [0, 1, 2, 3],
        "overflow": "4+",
        "key": "consecutive_wrong",
    },
    "steps_since_last_correct": {
        "buckets": [0, 1, 2, 3],
        "overflow": "4+",
        "key": "steps_since_last_correct",
    },
    "position_in_subtask": {
        "buckets": [0, 1, 2, 3],
        "overflow": "4+",
        "key": "position_in_subtask",
    },
    "is_subtask_boundary": {
        "bool_key": "is_subtask_boundary",
    },
    "subtask_index": {
        "buckets": [0, 1, 2],
        "overflow": "3+",
        "key": "subtask_index",
    },
    "history_action_count": {
        "buckets": [0, 1, 2, 3, 4, 5],
        "overflow": "6+",
        "key": "history_action_count",
    },
    "history_token_count": {
        "ranges": [(0, 10), (10, 30), (30, 60), (60, 100), (100, 9999)],
        "labels": ["0-10", "10-30", "30-60", "60-100", "100+"],
        "key": "history_token_count",
    },
    "mismatch_magnitude": {
        "ranges": [(0, 0.01), (0.01, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.01)],
        "labels": ["0%", "1-25%", "25-50%", "50-75%", "75-100%"],
        "key": "mismatch_magnitude",
    },
    "steps_remaining": {
        "buckets": [0, 1, 2, 3, 4, 5],
        "overflow": "6+",
        "key": "steps_remaining",
    },
}


def bucket_value(config, step):
    """Assign a step to a bucket for a given feature config."""
    if "bool_key" in config:
        return str(step[config["bool_key"]])

    val = step[config["key"]]

    if "ranges" in config:
        for (lo, hi), label in zip(config["ranges"], config["labels"]):
            if lo <= val < hi:
                return label
        return config["labels"][-1]

    if "buckets" in config:
        for b in config["buckets"]:
            if val == b:
                return str(b)
        return config.get("overflow", str(val))

    return str(val)


def univariate_analysis(steps, condition_label=""):
    """Run univariate analysis for all features."""
    results = {}

    for feat_name, config in BUCKET_CONFIGS.items():
        buckets = defaultdict(list)
        for s in steps:
            b = bucket_value(config, s)
            buckets[b].append(s)

        # Determine bucket order
        if "ranges" in config:
            order = config["labels"]
        elif "bool_key" in config:
            order = ["True", "False"]
        elif "buckets" in config:
            order = [str(b) for b in config["buckets"]]
            if config.get("overflow"):
                order.append(config["overflow"])
        else:
            order = sorted(buckets.keys())

        bucket_stats = {}
        xs = []
        ys = []
        ws = []
        for i, b in enumerate(order):
            b_steps = buckets.get(b, [])
            n = len(b_steps)
            if n == 0:
                bucket_stats[b] = {"n": 0, "accuracy": 0}
                continue
            acc = sum(1 for s in b_steps if s["correct"]) / n
            bucket_stats[b] = {"n": n, "accuracy": acc}
            xs.append(i)
            ys.append(acc)
            ws.append(n)

        # Weighted linear regression
        gradient, r2 = 0.0, 0.0
        if len(xs) >= 2:
            xs_a = np.array(xs, dtype=float)
            ys_a = np.array(ys, dtype=float)
            ws_a = np.array(ws, dtype=float)
            x_mean = np.average(xs_a, weights=ws_a)
            y_mean = np.average(ys_a, weights=ws_a)
            num = np.sum(ws_a * (xs_a - x_mean) * (ys_a - y_mean))
            den = np.sum(ws_a * (xs_a - x_mean) ** 2)
            if den > 0:
                gradient = float(num / den)
                y_pred = y_mean + gradient * (xs_a - x_mean)
                ss_res = np.sum(ws_a * (ys_a - y_pred) ** 2)
                ss_tot = np.sum(ws_a * (ys_a - y_mean) ** 2)
                r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0

        results[feat_name] = {
            "buckets": bucket_stats,
            "order": order,
            "gradient": gradient,
            "r_squared": r2,
        }

    return results


# ---------------------------------------------------------------------------
# Step 2: Multivariate logistic regression
# ---------------------------------------------------------------------------

def multivariate_regression(steps):
    """Fit logistic regression to predict step correctness from all features."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    feature_names = [
        "step_index", "relative_position", "steps_remaining",
        "preceding_wrong", "preceding_wrong_rate",
        "consecutive_wrong", "steps_since_last_correct",
        "subtask_index", "position_in_subtask", "subtask_length",
        "is_subtask_boundary",
        "history_token_count", "history_action_count",
        "mismatch_magnitude",
    ]

    X = []
    y = []
    for s in steps:
        row = []
        for f in feature_names:
            val = s[f]
            if isinstance(val, bool):
                val = int(val)
            row.append(float(val))
        X.append(row)
        y.append(int(s["correct"]))

    X = np.array(X)
    y = np.array(y)

    # Standardize features for comparable coefficients
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Fit logistic regression
    model = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X_scaled, y)

    accuracy = model.score(X_scaled, y)
    coefficients = {}
    for name, coef in zip(feature_names, model.coef_[0]):
        coefficients[name] = float(coef)

    # Sort by absolute coefficient magnitude
    sorted_features = sorted(coefficients.items(), key=lambda x: abs(x[1]), reverse=True)

    return {
        "accuracy": float(accuracy),
        "intercept": float(model.intercept_[0]),
        "coefficients": coefficients,
        "ranked_features": [(name, coef) for name, coef in sorted_features],
        "feature_names": feature_names,
        "n_samples": len(y),
        "n_positive": int(y.sum()),
        "base_rate": float(y.mean()),
    }


# ---------------------------------------------------------------------------
# Step 3: Feature interactions
# ---------------------------------------------------------------------------

def interaction_analysis(steps):
    """Analyze key feature interactions."""
    results = {}

    # Interaction 1: preceding_wrong x position_in_subtask
    results["preceding_wrong_x_position_in_subtask"] = _cross_table(
        steps,
        "preceding_wrong", [0, 1, 2, "3+"],
        "position_in_subtask", [0, 1, 2, "3+"],
    )

    # Interaction 2: preceding_wrong x subtask_index
    results["preceding_wrong_x_subtask_index"] = _cross_table(
        steps,
        "preceding_wrong", [0, 1, 2, "3+"],
        "subtask_index", [0, 1, "2+"],
    )

    # Interaction 3: consecutive_wrong x position_in_subtask
    results["consecutive_wrong_x_position_in_subtask"] = _cross_table(
        steps,
        "consecutive_wrong", [0, 1, 2, "3+"],
        "position_in_subtask", [0, 1, 2, "3+"],
    )

    # Interaction 4: step_index x preceding_wrong_rate (binned)
    # Bin step_index into early/mid/late
    for s in steps:
        rp = s["relative_position"]
        s["_position_bin"] = "early" if rp < 0.33 else ("mid" if rp < 0.67 else "late")
        pw_rate = s["preceding_wrong_rate"]
        s["_pw_rate_bin"] = ("0%" if pw_rate < 0.01 else
                             "1-33%" if pw_rate < 0.33 else
                             "33-67%" if pw_rate < 0.67 else "67-100%")

    results["position_x_error_rate"] = _cross_table_string(
        steps,
        "_position_bin", ["early", "mid", "late"],
        "_pw_rate_bin", ["0%", "1-33%", "33-67%", "67-100%"],
    )

    # Hypothesis test: consecutive vs scattered errors
    results["consecutive_vs_scattered"] = _consecutive_vs_scattered(steps)

    # Hypothesis test: subtask boundary reset effect
    results["subtask_boundary_reset"] = _boundary_reset_analysis(steps)

    return results


def _bucket_val(val, buckets_with_overflow):
    """Assign a numeric value to a bucket label."""
    for b in buckets_with_overflow:
        if isinstance(b, str) and b.endswith("+"):
            threshold = int(b.rstrip("+"))
            if val >= threshold:
                return b
        elif val == b:
            return str(b)
    # Fallback to overflow
    for b in buckets_with_overflow:
        if isinstance(b, str) and b.endswith("+"):
            return b
    return str(val)


def _cross_table(steps, feat1, buckets1, feat2, buckets2):
    """Compute accuracy cross-table for two features."""
    table = {}
    for b1 in buckets1:
        b1_label = str(b1)
        table[b1_label] = {}
        for b2 in buckets2:
            b2_label = str(b2)
            matching = [s for s in steps
                        if _bucket_val(s[feat1], buckets1) == b1_label
                        and _bucket_val(s[feat2], buckets2) == b2_label]
            n = len(matching)
            acc = sum(1 for s in matching if s["correct"]) / n if n > 0 else 0
            table[b1_label][b2_label] = {"n": n, "accuracy": acc}
    return table


def _cross_table_string(steps, feat1, labels1, feat2, labels2):
    """Cross-table for string-valued features."""
    table = {}
    for l1 in labels1:
        table[l1] = {}
        for l2 in labels2:
            matching = [s for s in steps if s[feat1] == l1 and s[feat2] == l2]
            n = len(matching)
            acc = sum(1 for s in matching if s["correct"]) / n if n > 0 else 0
            table[l1][l2] = {"n": n, "accuracy": acc}
    return table


def _consecutive_vs_scattered(steps):
    """Compare: for steps with preceding_wrong=3, is consecutive worse than scattered?"""
    consec_3plus = [s for s in steps if s["preceding_wrong"] >= 3 and s["consecutive_wrong"] >= 3]
    scattered_3plus = [s for s in steps if s["preceding_wrong"] >= 3 and s["consecutive_wrong"] < 3]

    n_consec = len(consec_3plus)
    n_scattered = len(scattered_3plus)
    acc_consec = sum(1 for s in consec_3plus if s["correct"]) / n_consec if n_consec > 0 else 0
    acc_scattered = sum(1 for s in scattered_3plus if s["correct"]) / n_scattered if n_scattered > 0 else 0

    return {
        "consecutive_3plus": {"n": n_consec, "accuracy": acc_consec},
        "scattered_3plus": {"n": n_scattered, "accuracy": acc_scattered},
        "delta": acc_scattered - acc_consec,
        "hypothesis": "consecutive_worse" if acc_consec < acc_scattered else "no_difference",
    }


def _boundary_reset_analysis(steps):
    """Compare accuracy at subtask boundary vs non-boundary steps."""
    boundary = [s for s in steps if s["is_subtask_boundary"]]
    non_boundary = [s for s in steps if not s["is_subtask_boundary"]]
    second_step = [s for s in steps if s["position_in_subtask"] == 1]

    n_b = len(boundary)
    n_nb = len(non_boundary)
    n_s2 = len(second_step)
    acc_b = sum(1 for s in boundary if s["correct"]) / n_b if n_b > 0 else 0
    acc_nb = sum(1 for s in non_boundary if s["correct"]) / n_nb if n_nb > 0 else 0
    acc_s2 = sum(1 for s in second_step if s["correct"]) / n_s2 if n_s2 > 0 else 0

    # Boundary steps with prior subtask errors
    boundary_after_errors = [s for s in boundary if s["preceding_wrong_global"] > 0]
    n_bae = len(boundary_after_errors)
    acc_bae = sum(1 for s in boundary_after_errors if s["correct"]) / n_bae if n_bae > 0 else 0

    return {
        "boundary": {"n": n_b, "accuracy": acc_b},
        "non_boundary": {"n": n_nb, "accuracy": acc_nb},
        "second_step": {"n": n_s2, "accuracy": acc_s2},
        "boundary_after_errors": {"n": n_bae, "accuracy": acc_bae},
        "reset_effect": acc_b - acc_s2,
    }


# ---------------------------------------------------------------------------
# Differential analysis: F4 vs C0 by feature
# ---------------------------------------------------------------------------

def differential_analysis(c0_steps, f4_steps):
    """Compare F4 - C0 accuracy delta across feature buckets."""
    # Index F4 steps by sample_id
    f4_by_id = {s["sample_id"]: s for s in f4_steps}

    results = {}
    for feat_name, config in BUCKET_CONFIGS.items():
        c0_buckets = defaultdict(list)
        f4_buckets = defaultdict(list)

        for s in c0_steps:
            b = bucket_value(config, s)
            c0_buckets[b].append(s)
            f4_s = f4_by_id.get(s["sample_id"])
            if f4_s:
                f4_buckets[b].append(f4_s)

        if "ranges" in config:
            order = config["labels"]
        elif "bool_key" in config:
            order = ["True", "False"]
        elif "buckets" in config:
            order = [str(b) for b in config["buckets"]]
            if config.get("overflow"):
                order.append(config["overflow"])
        else:
            order = sorted(c0_buckets.keys())

        bucket_deltas = {}
        for b in order:
            c0_list = c0_buckets.get(b, [])
            f4_list = f4_buckets.get(b, [])
            n_c0 = len(c0_list)
            n_f4 = len(f4_list)
            acc_c0 = sum(1 for s in c0_list if s["correct"]) / n_c0 if n_c0 > 0 else 0
            acc_f4 = sum(1 for s in f4_list if s["correct"]) / n_f4 if n_f4 > 0 else 0
            bucket_deltas[b] = {
                "n_c0": n_c0, "n_f4": n_f4,
                "c0_acc": acc_c0, "f4_acc": acc_f4,
                "delta": acc_f4 - acc_c0,
            }

        results[feat_name] = {"buckets": bucket_deltas, "order": order}

    return results


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def generate_report(c0_univariate, f4_univariate, c0_regression, f4_regression,
                    c0_interactions, differential, oracle_univariate=None):
    """Generate comprehensive markdown report."""
    lines = []
    lines.append("# Exp2e: Long-Horizon Feature Analysis\n")

    # --- Step 1: Univariate feature ranking ---
    lines.append("## Step 1: Univariate Feature Ranking\n")
    lines.append("Features ranked by R-squared (explaining step accuracy variance):\n")
    lines.append("| Rank | Feature | C0 Gradient | C0 R² | F4 Gradient | F4 R² |")
    lines.append("|------|---------|-------------|-------|-------------|-------|")

    # Sort by C0 R²
    feat_ranking = sorted(c0_univariate.keys(),
                          key=lambda f: c0_univariate[f]["r_squared"], reverse=True)
    for rank, feat in enumerate(feat_ranking, 1):
        c0 = c0_univariate[feat]
        f4 = f4_univariate.get(feat, {"gradient": 0, "r_squared": 0})
        lines.append(
            f"| {rank} | {feat} | {c0['gradient']:+.4f} | {c0['r_squared']:.4f} | "
            f"{f4['gradient']:+.4f} | {f4['r_squared']:.4f} |"
        )
    lines.append("")

    # --- Detailed univariate tables for top features ---
    lines.append("## Step 1 Detail: Accuracy Curves for Top Features\n")
    for feat in feat_ranking[:6]:
        c0 = c0_univariate[feat]
        f4 = f4_univariate.get(feat, {})
        lines.append(f"### {feat} (C0 R²={c0['r_squared']:.4f})\n")

        # Header
        header = f"| {feat} | n (C0) | C0 Acc |"
        sep = f"|{'---':>15}|--------|--------|"
        if f4:
            header += " n (F4) | F4 Acc | Delta |"
            sep += "--------|--------|-------|"
        lines.append(header)
        lines.append(sep)

        for b in c0.get("order", []):
            c0_b = c0["buckets"].get(b, {"n": 0, "accuracy": 0})
            row = f"| {b:>13} | {c0_b['n']:>6} | {c0_b['accuracy']:.4f} |"
            if f4:
                f4_b = f4.get("buckets", {}).get(b, {"n": 0, "accuracy": 0})
                delta = f4_b["accuracy"] - c0_b["accuracy"] if c0_b["n"] > 0 and f4_b["n"] > 0 else 0
                row += f" {f4_b['n']:>6} | {f4_b['accuracy']:.4f} | {delta:+.4f} |"
            lines.append(row)
        lines.append("")

    # --- Step 2: Multivariate regression ---
    lines.append("## Step 2: Multivariate Logistic Regression\n")
    lines.append(f"C0 model accuracy: {c0_regression['accuracy']:.4f} "
                 f"(base rate: {c0_regression['base_rate']:.4f})\n")
    lines.append("| Rank | Feature | C0 Coef | F4 Coef |")
    lines.append("|------|---------|---------|---------|")
    for rank, (name, coef) in enumerate(c0_regression["ranked_features"], 1):
        f4_coef = f4_regression["coefficients"].get(name, 0)
        lines.append(f"| {rank} | {name} | {coef:+.4f} | {f4_coef:+.4f} |")
    lines.append("")

    # --- Step 3: Interactions ---
    lines.append("## Step 3: Feature Interactions\n")

    # Consecutive vs scattered
    cvs = c0_interactions.get("consecutive_vs_scattered", {})
    lines.append("### Hypothesis 1: Consecutive vs Scattered Errors (pw>=3)\n")
    lines.append(f"- Consecutive 3+ wrong: n={cvs.get('consecutive_3plus', {}).get('n', 0)}, "
                 f"acc={cvs.get('consecutive_3plus', {}).get('accuracy', 0):.4f}")
    lines.append(f"- Scattered 3+ wrong:   n={cvs.get('scattered_3plus', {}).get('n', 0)}, "
                 f"acc={cvs.get('scattered_3plus', {}).get('accuracy', 0):.4f}")
    lines.append(f"- Delta (scattered - consecutive): {cvs.get('delta', 0):+.4f}")
    lines.append(f"- Verdict: **{cvs.get('hypothesis', 'unknown')}**\n")

    # Boundary reset
    bra = c0_interactions.get("subtask_boundary_reset", {})
    lines.append("### Hypothesis 2: Subtask Boundary Reset Effect\n")
    lines.append(f"- Boundary (pos=0):     n={bra.get('boundary', {}).get('n', 0)}, "
                 f"acc={bra.get('boundary', {}).get('accuracy', 0):.4f}")
    lines.append(f"- Second step (pos=1):  n={bra.get('second_step', {}).get('n', 0)}, "
                 f"acc={bra.get('second_step', {}).get('accuracy', 0):.4f}")
    lines.append(f"- Non-boundary:         n={bra.get('non_boundary', {}).get('n', 0)}, "
                 f"acc={bra.get('non_boundary', {}).get('accuracy', 0):.4f}")
    lines.append(f"- Boundary after global errors: n={bra.get('boundary_after_errors', {}).get('n', 0)}, "
                 f"acc={bra.get('boundary_after_errors', {}).get('accuracy', 0):.4f}")
    lines.append(f"- Reset effect (boundary - 2nd step): {bra.get('reset_effect', 0):+.4f}\n")

    # Cross-tables
    for interaction_name in ["preceding_wrong_x_position_in_subtask",
                             "preceding_wrong_x_subtask_index"]:
        table = c0_interactions.get(interaction_name, {})
        if not table:
            continue
        lines.append(f"### {interaction_name}\n")
        rows = sorted(table.keys(), key=lambda x: int(x.rstrip("+")) if x.rstrip("+").isdigit() else 99)
        if rows:
            cols = sorted(table[rows[0]].keys(),
                          key=lambda x: int(x.rstrip("+")) if x.rstrip("+").isdigit() else 99)
            header = f"| pw \\ {'pos_sub' if 'position' in interaction_name else 'sub_idx'} |"
            for c in cols:
                header += f" {c} |"
            lines.append(header)
            lines.append("|" + "---|" * (len(cols) + 1))
            for r in rows:
                row = f"| {r} |"
                for c in cols:
                    cell = table[r].get(c, {"n": 0, "accuracy": 0})
                    if cell["n"] > 0:
                        row += f" {cell['accuracy']:.3f} ({cell['n']}) |"
                    else:
                        row += " — |"
                lines.append(row)
        lines.append("")

    # --- Differential: F4 benefit by feature ---
    lines.append("## F4 vs C0: Where Does Multi-Agent Help Most?\n")
    # Show top features where delta varies most
    for feat in ["preceding_wrong", "consecutive_wrong", "position_in_subtask",
                 "is_subtask_boundary", "relative_position"]:
        diff = differential.get(feat, {})
        if not diff:
            continue
        lines.append(f"### F4 Delta by {feat}\n")
        lines.append(f"| {feat} | n | C0 Acc | F4 Acc | Delta |")
        lines.append("|---|---|--------|--------|-------|")
        for b in diff.get("order", []):
            bd = diff["buckets"].get(b, {})
            lines.append(
                f"| {b} | {bd.get('n_c0', 0)} | {bd.get('c0_acc', 0):.4f} | "
                f"{bd.get('f4_acc', 0):.4f} | {bd.get('delta', 0):+.4f} |"
            )
        lines.append("")

    # --- Oracle comparison if available ---
    if oracle_univariate:
        lines.append("## Oracle History: Feature-Level Comparison\n")
        lines.append("Accuracy with GT history, by preceding_wrong bucket:\n")
        feat = "preceding_wrong"
        c0 = c0_univariate.get(feat, {})
        orc = oracle_univariate.get(feat, {})
        lines.append(f"| {feat} | C0 Acc | Oracle Acc | Mismatch Tax |")
        lines.append("|---|--------|------------|-------------|")
        for b in c0.get("order", []):
            c0_b = c0["buckets"].get(b, {"accuracy": 0, "n": 0})
            orc_b = orc.get("buckets", {}).get(b, {"accuracy": 0, "n": 0})
            tax = orc_b["accuracy"] - c0_b["accuracy"] if c0_b["n"] > 0 and orc_b["n"] > 0 else 0
            lines.append(f"| {b} | {c0_b['accuracy']:.4f} | {orc_b['accuracy']:.4f} | {tax:+.4f} |")
        lines.append("")

    # --- Summary: answering the 3 core questions ---
    lines.append("## Core Questions\n")
    lines.append("### Q1: What drives per-step accuracy decline?\n")
    lines.append("Top 3 features by R² (C0):\n")
    for feat in feat_ranking[:3]:
        c0 = c0_univariate[feat]
        lines.append(f"  - {feat}: gradient={c0['gradient']:+.4f}, R²={c0['r_squared']:.4f}")
    lines.append("")
    lines.append("### Q2: Where does F4 help most?\n")
    pw_diff = differential.get("preceding_wrong", {}).get("buckets", {})
    if pw_diff:
        pw0 = pw_diff.get("0", {})
        pw3p = pw_diff.get("3+", pw_diff.get("3", {}))
        lines.append(f"  - pw=0: delta={pw0.get('delta', 0):+.4f} (clean state)")
        if pw3p:
            lines.append(f"  - pw=3+: delta={pw3p.get('delta', 0):+.4f} (error-laden state)")
    lines.append("")
    lines.append("### Q3: Most vulnerable step profile\n")
    lines.append("  Steps with highest failure rate: check interaction tables above.\n")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Data loading helpers (reused from mismatch_analysis.py)
# ---------------------------------------------------------------------------

def load_c0_trajs(c0_path):
    with open(c0_path) as f:
        return json.load(f)["detailed_results"]


def load_f4_steps(f4_path):
    """Load F4 JSONL and return list of step dicts with 'correct' key."""
    steps = []
    with open(f4_path) as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line.strip())
            d["correct"] = d.get("success", False)
            steps.append(d)
    return steps


def load_oracle_steps(oracle_path):
    """Load oracle JSONL and return list of step dicts with 'correct' key."""
    if not os.path.exists(oracle_path):
        return None
    steps = []
    with open(oracle_path) as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line.strip())
            d["correct"] = d.get("success", False)
            steps.append(d)
    return steps


def load_trajectories_raw(data_root, trajectory_ids):
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
                        steps.append({
                            "sample_id": f"{traj_id}_{line_num}",
                            "subtask": d["step"].get("subtask", ""),
                        })

                trajectories[traj_id] = steps

    return trajectories


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Exp2e: Long-Horizon Feature Analysis")
    parser.add_argument("--c0_results", type=str, required=True)
    parser.add_argument("--f4_results", type=str, required=True)
    parser.add_argument("--oracle_results", type=str, default=None,
                        help="Path to oracle.jsonl (optional)")
    parser.add_argument("--trajectory_ids", type=str, required=True)
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("Exp2e: Long-Horizon Feature Analysis")
    print("=" * 60)

    # Load data
    with open(args.trajectory_ids) as f:
        traj_ids = json.load(f)
    print(f"Trajectories: {len(traj_ids)}")

    print("Loading raw trajectories...")
    raw_trajs = load_trajectories_raw(args.data_root, traj_ids)

    print("Loading C0 results...")
    c0_trajs = load_c0_trajs(args.c0_results)

    print("Computing C0 features...")
    c0_steps = compute_features(c0_trajs, raw_trajs)
    print(f"  {len(c0_steps)} steps with features computed")

    # Load F4 — compute features using C0 trajectory structure
    # F4 shares the same trajectory structure but with different correctness
    print("Loading F4 results...")
    f4_raw_steps = load_f4_steps(args.f4_results)
    f4_by_id = {s["sample_id"]: s for s in f4_raw_steps}
    # Reuse C0 features but update correctness from F4
    f4_steps = []
    for s in c0_steps:
        f4_s = f4_by_id.get(s["sample_id"])
        if f4_s:
            s_copy = s.copy()
            s_copy["correct"] = f4_s.get("success", False)
            f4_steps.append(s_copy)
    print(f"  {len(f4_steps)} F4 steps matched")

    # Step 1: Univariate analysis
    print("\nStep 1: Univariate analysis...")
    c0_univariate = univariate_analysis(c0_steps, "C0")
    f4_univariate = univariate_analysis(f4_steps, "F4")

    # Step 2: Multivariate regression
    print("Step 2: Multivariate logistic regression...")
    c0_regression = multivariate_regression(c0_steps)
    f4_regression = multivariate_regression(f4_steps)
    print(f"  C0 model accuracy: {c0_regression['accuracy']:.4f}")
    print(f"  F4 model accuracy: {f4_regression['accuracy']:.4f}")

    # Step 3: Interactions
    print("Step 3: Feature interactions...")
    c0_interactions = interaction_analysis(c0_steps)

    # Differential analysis
    print("Differential analysis (F4 vs C0)...")
    differential = differential_analysis(c0_steps, f4_steps)

    # Optional: Oracle analysis
    oracle_univariate = None
    if args.oracle_results:
        print("Loading oracle results...")
        oracle_raw = load_oracle_steps(args.oracle_results)
        if oracle_raw:
            oracle_by_id = {s["sample_id"]: s for s in oracle_raw}
            oracle_steps = []
            for s in c0_steps:
                orc_s = oracle_by_id.get(s["sample_id"])
                if orc_s:
                    s_copy = s.copy()
                    s_copy["correct"] = orc_s.get("success", False)
                    oracle_steps.append(s_copy)
            if oracle_steps:
                oracle_univariate = univariate_analysis(oracle_steps, "Oracle")
                print(f"  {len(oracle_steps)} oracle steps analyzed")

    # Save JSON
    analysis = {
        "c0_univariate": c0_univariate,
        "f4_univariate": f4_univariate,
        "c0_regression": c0_regression,
        "f4_regression": f4_regression,
        "c0_interactions": c0_interactions,
        "differential": differential,
    }
    if oracle_univariate:
        analysis["oracle_univariate"] = oracle_univariate

    json_path = os.path.join(args.output_dir, "longhorizon_analysis.json")
    with open(json_path, "w") as f:
        json.dump(analysis, f, indent=2, default=str)
    print(f"\nSaved: {json_path}")

    # Generate report
    report = generate_report(
        c0_univariate, f4_univariate,
        c0_regression, f4_regression,
        c0_interactions, differential,
        oracle_univariate,
    )

    report_path = os.path.join(args.output_dir, "LONGHORIZON_ANALYSIS_REPORT.md")
    with open(report_path, "w") as f:
        f.write(report)
    print(f"Saved: {report_path}")
    print(f"\n{report}")


if __name__ == "__main__":
    main()
