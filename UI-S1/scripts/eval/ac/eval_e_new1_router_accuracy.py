#!/usr/bin/env python3
"""
E_NEW1 Analysis: 3-way Router Accuracy
=======================================
Test whether a 3-way classifier (Action / Grounding / Pass-through) can
accurately route steps based on observable features derived from K=10
multi-sample predictions.

Routing labels (derived from greedy = sample[0]):
  - pass_through: type_match=True AND extract_match=True
  - action:       type_match=False
  - grounding:    type_match=True AND extract_match=False

Features:
  - agreement_rate:   fraction of K samples sharing the mode action type
  - action_entropy:   Shannon entropy over predicted action type distribution
  - step_num:         step position in trajectory
  - coordinate_std:   std of coordinate predictions (x,y combined)
  - pred_action_type: one-hot of the mode predicted action type across K samples
  - gt_action_type:   one-hot of ground-truth action type (oracle feature)
"""

import json
import math
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_PATH = (
    "/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/outputs/eval_c4c7_ac/"
    "Qwen2.5-VL-7B/multisample_results.jsonl"
)
OUTPUT_DIR = "/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/outputs/eval_e_new1"
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "e_new1_router_accuracy.json")

# ---------------------------------------------------------------------------
# Helper: Shannon entropy
# ---------------------------------------------------------------------------
def shannon_entropy(counter: Counter) -> float:
    total = sum(counter.values())
    if total == 0:
        return 0.0
    ent = 0.0
    for c in counter.values():
        if c > 0:
            p = c / total
            ent -= p * math.log2(p)
    return ent


# ---------------------------------------------------------------------------
# 1. Load all steps
# ---------------------------------------------------------------------------
print("=" * 70)
print("E_NEW1: 3-Way Router Accuracy Analysis")
print("=" * 70)
print(f"\nLoading data from:\n  {DATA_PATH}\n")

steps = []
with open(DATA_PATH) as f:
    for line_num, line in enumerate(f, 1):
        line = line.strip()
        if not line:
            continue
        try:
            ep = json.loads(line)
        except json.JSONDecodeError:
            print(f"  [WARN] Skipping malformed JSON at line {line_num}")
            continue
        episode_id = ep.get("episode_id", f"ep_{line_num}")
        num_steps = ep.get("num_steps", 0)
        for step_data in ep.get("step_samples", []):
            step_data["_episode_id"] = episode_id
            step_data["_num_steps"] = num_steps
            steps.append(step_data)

print(f"Loaded {len(steps)} steps from {line_num} episodes.\n")

# ---------------------------------------------------------------------------
# 2. Determine GT routing label + extract features per step
# ---------------------------------------------------------------------------
# Canonical action types for one-hot encoding
CANONICAL_ACTION_TYPES = sorted(
    ["click", "type", "swipe", "wait", "open", "system_button", "long_press"]
)
at2idx = {at: i for i, at in enumerate(CANONICAL_ACTION_TYPES)}

records = []  # list of dicts with features + label
label_counter = Counter()

for step in steps:
    samples = step.get("samples", [])
    if not samples:
        continue

    gt_action_type = step.get("gt_action_type", "unknown")
    step_num = step.get("step_num", 0)
    num_steps = step.get("_num_steps", 1)

    # Greedy = first sample
    greedy = samples[0]
    type_match = greedy.get("type_match", False)
    extract_match = greedy.get("extract_match", False)

    # Routing label
    if type_match and extract_match:
        label = "pass_through"
    elif not type_match:
        label = "action"
    else:  # type_match True, extract_match False
        label = "grounding"
    label_counter[label] += 1

    # ---- Compute features from K samples ----
    pred_action_types = []
    coords_x = []
    coords_y = []
    for s in samples:
        pa = s.get("pred_action")
        if pa is None or not isinstance(pa, dict):
            pred_action_types.append("_none_")
        else:
            pred_action_types.append(pa.get("action", "_none_"))
            coord = pa.get("coordinate")
            if coord and isinstance(coord, (list, tuple)) and len(coord) >= 2:
                try:
                    cx, cy = float(coord[0]), float(coord[1])
                    # Clip to reasonable screen coordinate range
                    if abs(cx) < 1e6 and abs(cy) < 1e6:
                        coords_x.append(cx)
                        coords_y.append(cy)
                except (ValueError, TypeError):
                    pass

    type_counter = Counter(pred_action_types)
    K = len(samples)

    # agreement_rate: fraction sharing mode type
    mode_type, mode_count = type_counter.most_common(1)[0]
    agreement_rate = mode_count / K

    # action_entropy
    action_entropy = shannon_entropy(type_counter)

    # coordinate_std (combined x and y)
    if len(coords_x) >= 2:
        coord_std = (np.std(coords_x) + np.std(coords_y)) / 2.0
    else:
        coord_std = 0.0  # no variance computable

    # One-hot: gt_action_type
    gt_onehot = [0] * len(CANONICAL_ACTION_TYPES)
    if gt_action_type in at2idx:
        gt_onehot[at2idx[gt_action_type]] = 1

    # One-hot: predicted mode action type
    pred_onehot = [0] * len(CANONICAL_ACTION_TYPES)
    if mode_type in at2idx:
        pred_onehot[at2idx[mode_type]] = 1

    records.append(
        {
            "label": label,
            "agreement_rate": agreement_rate,
            "action_entropy": action_entropy,
            "step_num": step_num,
            "step_frac": step_num / max(num_steps, 1),
            "coordinate_std": coord_std,
            "gt_action_type": gt_action_type,
            "pred_mode_type": mode_type,
            "gt_onehot": gt_onehot,
            "pred_onehot": pred_onehot,
        }
    )

N = len(records)
print(f"Total usable steps: {N}")
print(f"Label distribution:")
for lbl in ["pass_through", "action", "grounding"]:
    cnt = label_counter[lbl]
    print(f"  {lbl:15s}: {cnt:5d}  ({100*cnt/N:.1f}%)")
print()

# ---------------------------------------------------------------------------
# 3. Build feature matrices
# ---------------------------------------------------------------------------
LABEL_ORDER = ["action", "grounding", "pass_through"]
label2int = {l: i for i, l in enumerate(LABEL_ORDER)}

y = np.array([label2int[r["label"]] for r in records])

# Full feature set (with GT action type)
X_full = np.array(
    [
        [
            r["agreement_rate"],
            r["action_entropy"],
            r["step_num"],
            r["step_frac"],
            r["coordinate_std"],
        ]
        + r["gt_onehot"]
        + r["pred_onehot"]
        for r in records
    ]
)

full_feature_names = (
    ["agreement_rate", "action_entropy", "step_num", "step_frac", "coordinate_std"]
    + [f"gt_{at}" for at in CANONICAL_ACTION_TYPES]
    + [f"pred_{at}" for at in CANONICAL_ACTION_TYPES]
)

# Observable feature set (no GT action type — realistic)
X_obs = np.array(
    [
        [
            r["agreement_rate"],
            r["action_entropy"],
            r["step_num"],
            r["step_frac"],
            r["coordinate_std"],
        ]
        + r["pred_onehot"]
        for r in records
    ]
)

obs_feature_names = (
    ["agreement_rate", "action_entropy", "step_num", "step_frac", "coordinate_std"]
    + [f"pred_{at}" for at in CANONICAL_ACTION_TYPES]
)

# Sanitise: replace inf/nan, clip extreme values
def sanitize(X):
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    X = np.clip(X, -1e8, 1e8)
    return X

X_full = sanitize(X_full)
X_obs = sanitize(X_obs)

# Ablation feature sets
X_agree_only = np.array([[r["agreement_rate"]] for r in records])
X_atype_only = np.array([r["pred_onehot"] for r in records])
X_agree_entropy = np.array(
    [[r["agreement_rate"], r["action_entropy"]] for r in records]
)

# ---------------------------------------------------------------------------
# 4. Cross-validated Logistic Regression
# ---------------------------------------------------------------------------
def run_cv(X, y, name, feature_names=None):
    """Run stratified 5-fold CV with Logistic Regression."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    clf = LogisticRegression(
        max_iter=2000,
        solver="lbfgs",
        C=1.0,
        random_state=42,
    )
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    y_pred = cross_val_predict(clf, X_scaled, y, cv=skf)

    acc = accuracy_score(y, y_pred)
    report = classification_report(
        y, y_pred, target_names=LABEL_ORDER, output_dict=True
    )
    cm = confusion_matrix(y, y_pred).tolist()
    macro_f1 = f1_score(y, y_pred, average="macro")

    # Fit on all data for feature importance
    clf.fit(X_scaled, y)
    coef_dict = {}
    if feature_names is not None:
        for cls_idx, cls_name in enumerate(LABEL_ORDER):
            coef_dict[cls_name] = {
                fn: round(float(clf.coef_[cls_idx, i]), 4)
                for i, fn in enumerate(feature_names)
            }

    return {
        "name": name,
        "accuracy": round(acc, 4),
        "macro_f1": round(macro_f1, 4),
        "per_class": {
            cls: {
                "precision": round(report[cls]["precision"], 4),
                "recall": round(report[cls]["recall"], 4),
                "f1": round(report[cls]["f1-score"], 4),
                "support": int(report[cls]["support"]),
            }
            for cls in LABEL_ORDER
        },
        "confusion_matrix": cm,
        "confusion_labels": LABEL_ORDER,
        "feature_importance": coef_dict,
    }


print("-" * 70)
print("Running Logistic Regression with Stratified 5-Fold CV ...")
print("-" * 70)

results = {}

# Full model (with GT action type — oracle upper bound)
res_full = run_cv(X_full, y, "full_with_gt_action_type", full_feature_names)
results["full_with_gt_action_type"] = res_full
print(f"\n[Full model w/ GT action type]  Accuracy: {res_full['accuracy']:.4f}  Macro-F1: {res_full['macro_f1']:.4f}")

# Observable model (no GT action type)
res_obs = run_cv(X_obs, y, "observable_features", obs_feature_names)
results["observable_features"] = res_obs
print(f"[Observable features]           Accuracy: {res_obs['accuracy']:.4f}  Macro-F1: {res_obs['macro_f1']:.4f}")

# Ablation: agreement_rate only
res_agree = run_cv(X_agree_only, y, "ablation_agreement_only", ["agreement_rate"])
results["ablation_agreement_only"] = res_agree
print(f"[Ablation: agreement only]      Accuracy: {res_agree['accuracy']:.4f}  Macro-F1: {res_agree['macro_f1']:.4f}")

# Ablation: action_type only (pred)
res_atype = run_cv(
    X_atype_only,
    y,
    "ablation_pred_action_type_only",
    [f"pred_{at}" for at in CANONICAL_ACTION_TYPES],
)
results["ablation_pred_action_type_only"] = res_atype
print(f"[Ablation: pred action type]    Accuracy: {res_atype['accuracy']:.4f}  Macro-F1: {res_atype['macro_f1']:.4f}")

# Ablation: agreement + entropy (no action type)
res_ae = run_cv(
    X_agree_entropy,
    y,
    "ablation_agreement_entropy",
    ["agreement_rate", "action_entropy"],
)
results["ablation_agreement_entropy"] = res_ae
print(f"[Ablation: agree + entropy]     Accuracy: {res_ae['accuracy']:.4f}  Macro-F1: {res_ae['macro_f1']:.4f}")

# ---------------------------------------------------------------------------
# 5. Print detailed results for the observable model
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("Detailed Results: Observable Features Model")
print("=" * 70)
print(f"\nAccuracy: {res_obs['accuracy']:.4f}")
print(f"Macro F1: {res_obs['macro_f1']:.4f}")
print(f"\nPer-class metrics:")
print(f"  {'Class':15s} {'Prec':>8s} {'Recall':>8s} {'F1':>8s} {'Support':>8s}")
print(f"  {'-'*15} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
for cls in LABEL_ORDER:
    m = res_obs["per_class"][cls]
    print(
        f"  {cls:15s} {m['precision']:8.4f} {m['recall']:8.4f} "
        f"{m['f1']:8.4f} {m['support']:8d}"
    )

print(f"\nConfusion Matrix (rows=true, cols=pred):")
print(f"  {'':15s} " + " ".join(f"{c:>12s}" for c in LABEL_ORDER))
cm = res_obs["confusion_matrix"]
for i, cls in enumerate(LABEL_ORDER):
    row_str = " ".join(f"{cm[i][j]:12d}" for j in range(3))
    print(f"  {cls:15s} {row_str}")

if res_obs["feature_importance"]:
    print(f"\nFeature Importance (LR coefficients, per class):")
    for cls in LABEL_ORDER:
        coeffs = res_obs["feature_importance"][cls]
        sorted_coeffs = sorted(coeffs.items(), key=lambda x: abs(x[1]), reverse=True)
        print(f"\n  [{cls}]")
        for fn, coef in sorted_coeffs[:8]:
            print(f"    {fn:25s}: {coef:+.4f}")

# ---------------------------------------------------------------------------
# 6. Rule-based router
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("Rule-Based Router")
print("=" * 70)

# Rule logic:
#   1. If agreement_rate == 1.0 AND pred_mode_type matches a "pass-friendly"
#      heuristic → pass_through
#   2. If pred_mode_type in {wait, swipe, open} and agreement < threshold → action
#   3. Otherwise, heuristics based on agreement
#
# Simplified rule set:
#   - If agreement_rate == 1.0 → pass_through
#   - If agreement_rate < 0.5 → action (high disagreement means type confusion)
#   - Else → grounding (moderate agreement, likely right type but wrong target)

rule_preds_simple = []
for r in records:
    if r["agreement_rate"] == 1.0:
        rule_preds_simple.append("pass_through")
    elif r["agreement_rate"] < 0.5:
        rule_preds_simple.append("action")
    else:
        rule_preds_simple.append("grounding")

y_labels = [r["label"] for r in records]
rule_acc_simple = accuracy_score(y_labels, rule_preds_simple)
rule_report_simple = classification_report(
    y_labels, rule_preds_simple, target_names=LABEL_ORDER, output_dict=True,
    labels=LABEL_ORDER,
)
rule_cm_simple = confusion_matrix(
    y_labels, rule_preds_simple, labels=LABEL_ORDER
).tolist()

print(f"\nRule v1 (agreement thresholds only):")
print(f"  Accuracy: {rule_acc_simple:.4f}")
print(f"  Per-class:")
for cls in LABEL_ORDER:
    m = rule_report_simple.get(cls, {})
    print(
        f"    {cls:15s}  P={m.get('precision',0):.4f}  "
        f"R={m.get('recall',0):.4f}  F1={m.get('f1-score',0):.4f}"
    )

# Rule v2: use predicted action type + agreement
rule_preds_v2 = []
# Types that the greedy model rarely gets wrong on type level
ACTION_ERROR_TYPES = {"wait", "open", "long_press"}
for r in records:
    pred_type = r["pred_mode_type"]
    agree = r["agreement_rate"]

    if agree == 1.0 and pred_type not in ("_none_",):
        rule_preds_v2.append("pass_through")
    elif agree < 0.5:
        rule_preds_v2.append("action")
    elif pred_type in ACTION_ERROR_TYPES:
        # These types have high type-error rates
        rule_preds_v2.append("action")
    else:
        rule_preds_v2.append("grounding")

rule_acc_v2 = accuracy_score(y_labels, rule_preds_v2)
rule_report_v2 = classification_report(
    y_labels, rule_preds_v2, target_names=LABEL_ORDER, output_dict=True,
    labels=LABEL_ORDER,
)
rule_cm_v2 = confusion_matrix(
    y_labels, rule_preds_v2, labels=LABEL_ORDER
).tolist()

print(f"\nRule v2 (agreement + pred action type heuristic):")
print(f"  Accuracy: {rule_acc_v2:.4f}")
print(f"  Per-class:")
for cls in LABEL_ORDER:
    m = rule_report_v2.get(cls, {})
    print(
        f"    {cls:15s}  P={m.get('precision',0):.4f}  "
        f"R={m.get('recall',0):.4f}  F1={m.get('f1-score',0):.4f}"
    )

# Rule v3: optimal agreement threshold (sweep)
print(f"\n  Sweeping agreement_rate thresholds for rule-based router ...")
best_rule_acc = 0.0
best_thresholds = (0.5, 1.0)
for lo in np.arange(0.3, 0.8, 0.05):
    for hi in np.arange(0.8, 1.01, 0.05):
        preds = []
        for r in records:
            if r["agreement_rate"] >= hi:
                preds.append("pass_through")
            elif r["agreement_rate"] < lo:
                preds.append("action")
            else:
                preds.append("grounding")
        acc = accuracy_score(y_labels, preds)
        if acc > best_rule_acc:
            best_rule_acc = acc
            best_thresholds = (round(float(lo), 2), round(float(hi), 2))

print(f"  Best thresholds: action < {best_thresholds[0]}, pass_through >= {best_thresholds[1]}")
print(f"  Best rule accuracy: {best_rule_acc:.4f}")

results["rule_based"] = {
    "rule_v1_agreement_only": {
        "accuracy": round(rule_acc_simple, 4),
        "per_class": {
            cls: {
                "precision": round(rule_report_simple.get(cls, {}).get("precision", 0), 4),
                "recall": round(rule_report_simple.get(cls, {}).get("recall", 0), 4),
                "f1": round(rule_report_simple.get(cls, {}).get("f1-score", 0), 4),
            }
            for cls in LABEL_ORDER
        },
        "confusion_matrix": rule_cm_simple,
    },
    "rule_v2_agreement_plus_type": {
        "accuracy": round(rule_acc_v2, 4),
        "per_class": {
            cls: {
                "precision": round(rule_report_v2.get(cls, {}).get("precision", 0), 4),
                "recall": round(rule_report_v2.get(cls, {}).get("recall", 0), 4),
                "f1": round(rule_report_v2.get(cls, {}).get("f1-score", 0), 4),
            }
            for cls in LABEL_ORDER
        },
        "confusion_matrix": rule_cm_v2,
    },
    "rule_v3_best_threshold": {
        "accuracy": round(best_rule_acc, 4),
        "lo_threshold": best_thresholds[0],
        "hi_threshold": best_thresholds[1],
    },
}

# ---------------------------------------------------------------------------
# 7. Summary comparison
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("Summary Comparison")
print("=" * 70)
print(f"\n  {'Model':45s} {'Accuracy':>10s} {'Macro-F1':>10s}")
print(f"  {'-'*45} {'-'*10} {'-'*10}")
for key in [
    "full_with_gt_action_type",
    "observable_features",
    "ablation_agreement_only",
    "ablation_pred_action_type_only",
    "ablation_agreement_entropy",
]:
    r = results[key]
    print(f"  {r['name']:45s} {r['accuracy']:10.4f} {r['macro_f1']:10.4f}")
print(f"  {'Rule v1 (agreement thresholds)':45s} {results['rule_based']['rule_v1_agreement_only']['accuracy']:10.4f} {'--':>10s}")
print(f"  {'Rule v2 (agreement + pred type)':45s} {results['rule_based']['rule_v2_agreement_plus_type']['accuracy']:10.4f} {'--':>10s}")
print(f"  {'Rule v3 (best threshold sweep)':45s} {results['rule_based']['rule_v3_best_threshold']['accuracy']:10.4f} {'--':>10s}")

# ---------------------------------------------------------------------------
# 8. Additional analysis: per-action-type routing accuracy (observable)
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("Per-Action-Type Routing Accuracy (Observable Model)")
print("=" * 70)

# Re-run to get per-step predictions
scaler = StandardScaler()
X_obs_scaled = scaler.fit_transform(X_obs)
clf_obs = LogisticRegression(
    max_iter=2000, solver="lbfgs", C=1.0, random_state=42
)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
y_pred_obs = cross_val_predict(clf_obs, X_obs_scaled, y, cv=skf)

per_at_results = {}
print(f"\n  {'Action Type':15s} {'N':>6s} {'Acc':>8s} {'Action%':>8s} {'Ground%':>8s} {'Pass%':>8s}")
print(f"  {'-'*15} {'-'*6} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
for at in CANONICAL_ACTION_TYPES:
    idxs = [i for i, r in enumerate(records) if r["gt_action_type"] == at]
    if not idxs:
        continue
    y_sub = y[idxs]
    yp_sub = y_pred_obs[idxs]
    acc = accuracy_score(y_sub, yp_sub)
    # Distribution of true labels
    true_dist = Counter(y_sub)
    n = len(idxs)
    per_at_results[at] = {
        "n": n,
        "accuracy": round(acc, 4),
        "true_label_dist": {
            LABEL_ORDER[k]: int(v) for k, v in true_dist.items()
        },
    }
    act_pct = 100 * true_dist.get(0, 0) / n
    gnd_pct = 100 * true_dist.get(1, 0) / n
    pas_pct = 100 * true_dist.get(2, 0) / n
    print(f"  {at:15s} {n:6d} {acc:8.4f} {act_pct:7.1f}% {gnd_pct:7.1f}% {pas_pct:7.1f}%")

results["per_action_type"] = per_at_results

# ---------------------------------------------------------------------------
# 9. Save results
# ---------------------------------------------------------------------------
results["metadata"] = {
    "data_path": DATA_PATH,
    "n_steps": N,
    "label_distribution": {k: v for k, v in label_counter.items()},
    "K_samples": 10,
    "canonical_action_types": CANONICAL_ACTION_TYPES,
    "label_order": LABEL_ORDER,
}

os.makedirs(OUTPUT_DIR, exist_ok=True)
with open(OUTPUT_PATH, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to:\n  {OUTPUT_PATH}")
print(f"\nDone.")
