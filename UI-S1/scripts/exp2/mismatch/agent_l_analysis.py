"""Exp2f: Agent_L Quality Verification.

Evaluates L15 hidden state probes as a lightweight replacement for Agent V.
Uses existing .npy probe data files (no new GPU inference needed).

Design: Option A — probe predictions → template descriptions
  - UI State probe: L15 image tokens → 5 classes
  - Target Element probe: L15 last token → 11 classes
  - Template: structured description combining both predictions

Comparison baseline: Agent V (base model visual description, thought-hit ~0.21)
"""

import argparse
import json
import os
import sys
import numpy as np
from collections import Counter, defaultdict

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXP2_DIR = os.path.dirname(SCRIPT_DIR)
if EXP2_DIR not in sys.path:
    sys.path.insert(0, EXP2_DIR)


# ──────────────────────────────────────────────────────────────────────
# Template descriptions from probe predictions
# ──────────────────────────────────────────────────────────────────────

UI_STATE_TEMPLATES = {
    "dialog": "A dialog box or popup window is currently open.",
    "main_view": "The main application workspace is displayed.",
    "menu_open": "A dropdown menu or context menu is expanded.",
    "ribbon_focus": "A ribbon tab or toolbar section is active.",
    "selection_active": "Content (text, cell, or object) is currently selected.",
}

TARGET_ELEMENT_TEMPLATES = {
    "ribbon_tab": "Navigate to a ribbon tab.",
    "dialog_button": "Interact with a dialog button (OK, Cancel, etc.).",
    "formatting": "Apply a formatting tool (font, alignment, etc.).",
    "content_area": "Edit content in the main area.",
    "cell_reference": "Select or edit a specific cell.",
    "navigation": "Navigate (scroll, page, switch sheet/slide).",
    "object_insertion": "Insert an object (shape, table, image, etc.).",
    "file_operation": "Perform a file operation (save, open, etc.).",
    "search_input": "Enter text in a search or address field.",
    "animation_property": "Set an animation property.",
    "other": "Interact with a UI element.",
}

# Map control_test strings to expected target_element categories
CATEGORY_KEYWORDS = {
    "ribbon_tab": [
        "home", "insert", "design", "view", "review", "layout", "data",
        "animations", "transitions", "slide show", "format", "references",
        "page layout", "mailings", "draw", "developer", "formulas",
        "recording", "table design", "table layout", "picture format",
    ],
    "dialog_button": [
        "ok", "cancel", "close", "yes", "no", "apply", "browse",
        "delete", "remove", "add", "replace", "find next",
    ],
    "formatting": [
        "font", "bold", "italic", "underline", "align", "center",
        "bullets", "numbering", "strikethrough", "heading",
    ],
    "navigation": [
        "page down", "page up", "scroll", "next", "previous", "back",
    ],
}


def generate_template(ui_state_pred, te_pred, ui_conf=None, te_conf=None):
    """Generate Agent_L template description from probe predictions."""
    ui_desc = UI_STATE_TEMPLATES.get(ui_state_pred, "Unknown UI state.")
    te_desc = TARGET_ELEMENT_TEMPLATES.get(te_pred, "Interact with a UI element.")

    parts = [f"Screen state: {ui_desc}"]
    parts.append(f"Expected action type: {te_desc}")

    if ui_conf is not None:
        parts[0] += f" ({ui_conf:.0%})"
    if te_conf is not None:
        parts[1] += f" ({te_conf:.0%})"

    return " ".join(parts)


def evaluate_category_match(pred_te_class, control_test):
    """Check if predicted target element category matches the GT control_test."""
    if not control_test:
        return False
    ct_lower = control_test.lower().strip()

    # Check if control_test maps to the predicted category
    for cat, keywords in CATEGORY_KEYWORDS.items():
        if cat == pred_te_class:
            if any(kw in ct_lower for kw in keywords):
                return True
            if ct_lower in [kw for kw in keywords]:
                return True
    return False


def main():
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.metrics import accuracy_score, f1_score, classification_report

    parser = argparse.ArgumentParser(description="Exp2f: Agent_L Quality Verification")
    parser.add_argument("--probe_data_dir", type=str, required=True,
                        help="Directory with .npy files and labels_v2.json")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for results")
    parser.add_argument("--agent_v_results", type=str, default=None,
                        help="Path to agent_v.jsonl (for comparison)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    PCA_DIM = 256
    TARGET_LAYER = 15

    print("=" * 60)
    print("Exp2f: Agent_L Quality Verification (L15 Probes)")
    print("=" * 60)

    # ── Load labels ──
    labels_path = os.path.join(args.probe_data_dir, "labels_v2.json")
    with open(labels_path) as f:
        labels = json.load(f)
    n = len(labels)

    is_pattern_b = np.array([l["is_pattern_b"] for l in labels])
    train_idx = np.where(~is_pattern_b)[0]
    test_idx = np.where(is_pattern_b)[0]
    print(f"Samples: {n} total ({len(train_idx)} train, {len(test_idx)} test)")

    # ── UI State probe: L15 image tokens ──
    print("\n--- Training UI State probe (L15, image tokens) ---")
    ui_labels = [l["ui_state_class"] for l in labels]
    le_ui = LabelEncoder()
    ui_encoded = le_ui.fit_transform(ui_labels)

    X_image = np.load(os.path.join(args.probe_data_dir, f"layer_{TARGET_LAYER}_image.npy"))
    scaler_ui = StandardScaler()
    X_image_scaled = scaler_ui.fit_transform(X_image)
    pca_ui = PCA(n_components=PCA_DIM, random_state=42)
    X_image_pca = pca_ui.fit_transform(X_image_scaled)

    clf_ui = LogisticRegression(max_iter=500, C=1.0, solver="lbfgs")
    clf_ui.fit(X_image_pca[train_idx], ui_encoded[train_idx])

    ui_train_acc = accuracy_score(ui_encoded[train_idx], clf_ui.predict(X_image_pca[train_idx]))
    ui_test_preds = clf_ui.predict(X_image_pca[test_idx])
    ui_test_probs = clf_ui.predict_proba(X_image_pca[test_idx])
    ui_test_acc = accuracy_score(ui_encoded[test_idx], ui_test_preds)
    ui_test_f1 = f1_score(ui_encoded[test_idx], ui_test_preds, average="macro")

    print(f"  Train acc: {ui_train_acc:.4f}")
    print(f"  Test acc:  {ui_test_acc:.4f}, F1: {ui_test_f1:.4f}")
    print(f"  Classes: {list(le_ui.classes_)}")
    print(classification_report(ui_encoded[test_idx], ui_test_preds,
                                target_names=le_ui.classes_))

    # ── Target Element probe: L15 last token ──
    print("\n--- Training Target Element probe (L15, last token) ---")
    te_labels = [l["target_element_class"] for l in labels]
    le_te = LabelEncoder()
    te_encoded = le_te.fit_transform(te_labels)

    X_last = np.load(os.path.join(args.probe_data_dir, f"layer_{TARGET_LAYER}_last.npy"))
    scaler_te = StandardScaler()
    X_last_scaled = scaler_te.fit_transform(X_last)
    pca_te = PCA(n_components=PCA_DIM, random_state=42)
    X_last_pca = pca_te.fit_transform(X_last_scaled)

    clf_te = LogisticRegression(max_iter=500, C=1.0, solver="lbfgs")
    clf_te.fit(X_last_pca[train_idx], te_encoded[train_idx])

    te_train_acc = accuracy_score(te_encoded[train_idx], clf_te.predict(X_last_pca[train_idx]))
    te_test_preds = clf_te.predict(X_last_pca[test_idx])
    te_test_probs = clf_te.predict_proba(X_last_pca[test_idx])
    te_test_acc = accuracy_score(te_encoded[test_idx], te_test_preds)
    te_test_f1 = f1_score(te_encoded[test_idx], te_test_preds, average="macro")

    print(f"  Train acc: {te_train_acc:.4f}")
    print(f"  Test acc:  {te_test_acc:.4f}, F1: {te_test_f1:.4f}")
    print(f"  Classes: {list(le_te.classes_)}")
    print(classification_report(te_encoded[test_idx], te_test_preds,
                                target_names=le_te.classes_))

    # ── Layer comparison: UI State across layers ──
    print("\n--- Layer comparison (UI State, image tokens) ---")
    layer_comparison = {}
    for layer_idx in [0, 5, 10, 13, 15, 17, 20, 25, 27]:
        fpath = os.path.join(args.probe_data_dir, f"layer_{layer_idx}_image.npy")
        if not os.path.exists(fpath):
            continue
        X = np.load(fpath)
        sc = StandardScaler()
        X_sc = sc.fit_transform(X)
        pc = PCA(n_components=PCA_DIM, random_state=42)
        X_pc = pc.fit_transform(X_sc)

        c = LogisticRegression(max_iter=500, C=1.0, solver="lbfgs")
        c.fit(X_pc[train_idx], ui_encoded[train_idx])
        acc = accuracy_score(ui_encoded[test_idx], c.predict(X_pc[test_idx]))
        f1 = f1_score(ui_encoded[test_idx], c.predict(X_pc[test_idx]), average="macro")
        layer_comparison[layer_idx] = {"acc": float(acc), "f1": float(f1)}
        marker = " <-- target" if layer_idx == TARGET_LAYER else ""
        print(f"  Layer {layer_idx:2d}: acc={acc:.4f}, F1={f1:.4f}{marker}")

    # ── Generate templates for test samples ──
    print("\n--- Generating Agent_L templates for Pattern B test set ---")
    templates = []
    category_correct = 0
    category_match_count = 0

    for i, test_i in enumerate(test_idx):
        label = labels[test_i]

        ui_pred_idx = ui_test_preds[i]
        te_pred_idx = te_test_preds[i]
        ui_pred_class = le_ui.classes_[ui_pred_idx]
        te_pred_class = le_te.classes_[te_pred_idx]

        ui_conf = float(ui_test_probs[i].max())
        te_conf = float(te_test_probs[i].max())

        template = generate_template(ui_pred_class, te_pred_class, ui_conf, te_conf)

        # Check exact class correctness
        gt_ui = label["ui_state_class"]
        gt_te = label["target_element_class"]
        cat_correct = (ui_pred_class == gt_ui) and (te_pred_class == gt_te)
        if cat_correct:
            category_correct += 1

        # Check semantic category match (softer metric)
        ct = label.get("control_test", "")
        cat_match = evaluate_category_match(te_pred_class, ct)
        if cat_match:
            category_match_count += 1

        templates.append({
            "sample_idx": int(test_i),
            "trajectory_id": label["trajectory_id"],
            "step_idx": label["step_idx"],
            "domain": label["domain"],
            "control_test": ct,
            "gt_ui_state": gt_ui,
            "gt_target_element": gt_te,
            "pred_ui_state": ui_pred_class,
            "pred_target_element": te_pred_class,
            "ui_confidence": ui_conf,
            "te_confidence": te_conf,
            "template": template,
            "category_correct": cat_correct,
            "category_match": cat_match,
        })

    n_test = len(test_idx)
    category_correct_rate = category_correct / n_test
    category_match_rate = category_match_count / n_test

    print(f"  Both probes correct: {category_correct_rate:.4f} ({category_correct}/{n_test})")
    print(f"  Category keyword match: {category_match_rate:.4f} ({category_match_count}/{n_test})")

    # ── Agent V comparison ──
    agent_v_hit_rate = None
    if args.agent_v_results and os.path.exists(args.agent_v_results):
        print("\n--- Agent V comparison ---")
        agent_v_hits = 0
        agent_v_total = 0
        with open(args.agent_v_results) as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line.strip())
                ct = r.get("gt_control_test", "")
                text = r.get("text_output", "")
                if ct and text:
                    agent_v_total += 1
                    if ct.lower() in text.lower():
                        agent_v_hits += 1
        if agent_v_total > 0:
            agent_v_hit_rate = agent_v_hits / agent_v_total
            print(f"  Agent V thought-hit: {agent_v_hit_rate:.4f} ({agent_v_hits}/{agent_v_total})")

    # ── Domain breakdown ──
    print("\n--- Domain breakdown ---")
    domain_stats = defaultdict(lambda: {"n": 0, "ui_correct": 0, "te_correct": 0, "both_correct": 0})
    for t in templates:
        d = t["domain"]
        domain_stats[d]["n"] += 1
        if t["pred_ui_state"] == t["gt_ui_state"]:
            domain_stats[d]["ui_correct"] += 1
        if t["pred_target_element"] == t["gt_target_element"]:
            domain_stats[d]["te_correct"] += 1
        if t["category_correct"]:
            domain_stats[d]["both_correct"] += 1

    for domain in sorted(domain_stats.keys()):
        s = domain_stats[domain]
        print(f"  {domain}: n={s['n']}, ui_acc={s['ui_correct']/s['n']:.4f}, "
              f"te_acc={s['te_correct']/s['n']:.4f}, "
              f"both_acc={s['both_correct']/s['n']:.4f}")

    # ── Confidence analysis ──
    print("\n--- Confidence calibration ---")
    for probe_name, confs, preds_correct in [
        ("UI State", [t["ui_confidence"] for t in templates],
         [t["pred_ui_state"] == t["gt_ui_state"] for t in templates]),
        ("Target Element", [t["te_confidence"] for t in templates],
         [t["pred_target_element"] == t["gt_target_element"] for t in templates]),
    ]:
        confs_arr = np.array(confs)
        correct_arr = np.array(preds_correct)
        for lo, hi in [(0, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.01)]:
            mask = (confs_arr >= lo) & (confs_arr < hi)
            if mask.sum() > 0:
                bin_acc = correct_arr[mask].mean()
                print(f"  {probe_name} conf [{lo:.1f},{hi:.1f}): "
                      f"n={mask.sum()}, acc={bin_acc:.4f}")

    # ── Save results ──
    output = {
        "config": {
            "target_layer": TARGET_LAYER,
            "pca_dim": PCA_DIM,
            "n_train": int(len(train_idx)),
            "n_test": int(len(test_idx)),
        },
        "ui_state_probe": {
            "train_acc": float(ui_train_acc),
            "test_acc": float(ui_test_acc),
            "test_f1": float(ui_test_f1),
            "classes": list(le_ui.classes_),
        },
        "target_element_probe": {
            "train_acc": float(te_train_acc),
            "test_acc": float(te_test_acc),
            "test_f1": float(te_test_f1),
            "classes": list(le_te.classes_),
        },
        "layer_comparison": {str(k): v for k, v in layer_comparison.items()},
        "template_metrics": {
            "category_correct_rate": float(category_correct_rate),
            "category_match_rate": float(category_match_rate),
            "agent_v_hit_rate": float(agent_v_hit_rate) if agent_v_hit_rate is not None else None,
        },
        "domain_breakdown": {k: dict(v) for k, v in domain_stats.items()},
        "templates": templates,
    }

    out_path = os.path.join(args.output_dir, "agent_l_analysis.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # ── Generate report ──
    generate_report(args.output_dir, output)


def generate_report(output_dir, results):
    """Generate markdown report."""
    cfg = results["config"]
    ui = results["ui_state_probe"]
    te = results["target_element_probe"]
    tm = results["template_metrics"]
    lc = results["layer_comparison"]
    db = results["domain_breakdown"]

    lines = [
        "# Exp2f: Agent_L Quality Verification Report",
        "",
        f"**Target Layer:** L{cfg['target_layer']} | **PCA dim:** {cfg['pca_dim']} | "
        f"**Train:** {cfg['n_train']} | **Test:** {cfg['n_test']}",
        "",
        "## Probe Quality",
        "",
        "### UI State Probe (L15, image tokens -> 5 classes)",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Train Accuracy | {ui['train_acc']:.4f} |",
        f"| Test Accuracy | {ui['test_acc']:.4f} |",
        f"| Test Macro F1 | {ui['test_f1']:.4f} |",
        f"| Classes | {', '.join(ui['classes'])} |",
        "",
        "### Target Element Probe (L15, last token -> 11 classes)",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Train Accuracy | {te['train_acc']:.4f} |",
        f"| Test Accuracy | {te['test_acc']:.4f} |",
        f"| Test Macro F1 | {te['test_f1']:.4f} |",
        f"| Classes | {', '.join(te['classes'])} |",
        "",
        "## Template Quality",
        "",
        "| Metric | Agent_L (L15) | Agent V |",
        "|--------|---------------|---------|",
        f"| Both classes correct | {tm['category_correct_rate']:.4f} | - |",
        f"| Category keyword match | {tm['category_match_rate']:.4f} | - |",
        f"| Thought-hit rate | - | {tm['agent_v_hit_rate']:.4f} |" if tm['agent_v_hit_rate'] else "| Thought-hit rate | - | N/A |",
        "",
        "## Layer Comparison (UI State probe, image tokens)",
        "",
        "| Layer | Test Accuracy | Test F1 |",
        "|-------|---------------|---------|",
    ]

    for layer_str in sorted(lc.keys(), key=lambda x: int(x)):
        v = lc[layer_str]
        marker = " *" if int(layer_str) == cfg['target_layer'] else ""
        lines.append(f"| L{layer_str}{marker} | {v['acc']:.4f} | {v['f1']:.4f} |")

    lines.extend([
        "",
        "## Domain Breakdown",
        "",
        "| Domain | n | UI Acc | TE Acc | Both Acc |",
        "|--------|---|--------|--------|----------|",
    ])

    for domain in sorted(db.keys()):
        s = db[domain]
        dn = s["n"]
        lines.append(
            f"| {domain} | {dn} | {s['ui_correct']/dn:.4f} | "
            f"{s['te_correct']/dn:.4f} | {s['both_correct']/dn:.4f} |"
        )

    lines.extend([
        "",
        "## Interpretation",
        "",
        "Agent_L uses lightweight L15 probes to generate structured descriptions "
        "of UI state and target element type. These template descriptions can be "
        "injected into the action prompt as a replacement for Agent V's free-text "
        "visual descriptions.",
        "",
        "The probe's category-level accuracy indicates how much information L15 "
        "hidden states retain about the visual context before it reaches the "
        "action-generating layers (L16-L27).",
        "",
        "Key comparison: Agent_L provides class-level descriptions (5+11 categories) "
        "while Agent V provides free-text element enumeration. Agent_L is much cheaper "
        "(no extra inference) but coarser. The downstream impact depends on whether "
        "class-level info is sufficient to guide the action model.",
    ])

    report_path = os.path.join(output_dir, "AGENT_L_REPORT.md")
    with open(report_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Report saved to {report_path}")


if __name__ == "__main__":
    main()
