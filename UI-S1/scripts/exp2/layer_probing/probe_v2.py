"""Redesigned layer-wise probing: Target Element / UI State / Correctness.

Reuses existing hidden state npy files (28 layers × 3 token types, 3000 samples).
Only creates new labels and trains new probes.

Probe 1 — Target Element Identification (last token → target element category)
  Does the hidden state encode WHICH UI element to interact with?
  Label: control_test grouped into functional categories

Probe 2 — UI State Understanding (image tokens → UI state class)
  Does the model understand the current UI context?
  Label: derived from observation + thought (dialog/menu/selection/main)

Probe 3 — Action Correctness Prediction (last token → correct/wrong, binary)
  Does the hidden state encode whether this step will be correct?
  Label: from AR evaluation results (clean samples only)
"""

import argparse
import json
import os
import re
import time
from collections import Counter, defaultdict

import numpy as np

NUM_TRANSFORMER_LAYERS = 28

# ──────────────────────────────────────────────────────────────────────
# Probe 1: Target Element — group control_test into functional categories
# ──────────────────────────────────────────────────────────────────────

RIBBON_TABS = {
    'Home', 'Insert', 'Design', 'View', 'Review', 'Layout', 'Data',
    'Animations', 'Transitions', 'Slide Show', 'Format', 'References',
    'Page Layout', 'Mailings', 'Draw', 'Developer', 'Table Design',
    'Formulas', 'Recording', 'Table Layout', 'Picture Format',
    'Slide Master', 'Chart Design',
}

DIALOG_BUTTONS = {
    'OK', 'Cancel', 'Close', 'Yes', 'No', 'Apply', 'Browse', 'Delete',
    'Remove', 'Add', 'Replace', 'Find Next', 'Replace All', 'Select All',
    'Don\'t Save', 'Stop', 'Close pane', 'Customize',
}

FORMATTING_TOOLS = {
    'Font', 'Font Size', 'B', 'I', 'U', 'Bold', 'Italic', 'Underline',
    'Font Color', 'Paragraph', 'Align Left', 'Center', 'Align Right',
    'Justify', 'Bullets', 'Numbering', 'Line Spacing',
    'Increase Font Size', 'Decrease Font Size', 'Text Highlight Color',
    'Strikethrough', 'Subscript', 'Superscript',
    'Heading 1', 'Heading 2', 'Heading 3', 'Style',
    'Conditional Formatting', 'Orientation', 'Vertical', 'Horizontal',
    'Tracking', 'Spell Check ',
}

FILE_OPERATIONS = {
    'File Tab', 'Save', 'Save As', 'Open', 'New', 'Print',
    'Export', 'Share', 'Info', 'Account', 'Options',
}

NAVIGATION = {
    'Page down', 'Page up', 'Scroll', 'More Options',
    'Add Sheet', 'Next', 'Previous', 'Back', 'Column right',
    'Line down', 'Line up', 'Column left', 'Pages', 'Slides',
    'Navigation Pane',
}

OBJECT_INSERTION = {
    'Shapes', 'New Slide', 'Picture', 'Chart', 'Table',
    'SmartArt', 'Icons', 'Text', 'Header & Footer',
    'Slide Number', 'Date & Time', 'WordArt', 'Equation',
    'Symbol', 'Online Pictures', 'Screenshot', 'Rectangle',
    'Slide Size', 'Animation Pane', 'Variants',
}

CONTENT_AREAS = {
    'Formula Bar', 'Edit Cell', 'Name Box',
    'Slide', 'Text Box', 'Draw Horizontal Text Box',
    'Content Pane Window', 'Header -Section 1-',
}

CELL_PATTERN = re.compile(r'^[A-Z]{1,3}\d{1,5}$')
SEARCH_PATTERN = re.compile(r'^(bing_search_query_|Type to search|Search |Address Bar)')
SHEET_PATTERN = re.compile(r'^Sheet\d+$')


def classify_target_element(control_test: str) -> str:
    """Classify control_test into functional category."""
    ct = control_test.strip()
    if not ct:
        return 'other'

    # Ribbon tabs
    if ct in RIBBON_TABS:
        return 'ribbon_tab'

    # Dialog buttons
    if ct in DIALOG_BUTTONS:
        return 'dialog_button'

    # File operations
    if ct in FILE_OPERATIONS:
        return 'file_operation'

    # Formatting tools
    if ct in FORMATTING_TOOLS:
        return 'formatting'

    # Content areas
    if ct in CONTENT_AREAS:
        return 'content_area'
    if 'Page' in ct and 'content' in ct.lower():
        return 'content_area'

    # Cell references (A1, B2, H4, etc.)
    if CELL_PATTERN.match(ct):
        return 'cell_reference'

    # Search/address bar
    if SEARCH_PATTERN.match(ct):
        return 'search_input'

    # Sheet tabs
    if SHEET_PATTERN.match(ct):
        return 'cell_reference'

    # Navigation
    if ct in NAVIGATION:
        return 'navigation'

    # Object insertion/manipulation
    if ct in OBJECT_INSERTION:
        return 'object_insertion'

    # Animation properties (PPT)
    if 'Animation' in ct or ct in ('Fade', 'Appear', 'Fly In', 'Wipe',
                                     'Direction From Center', 'Direction'):
        return 'animation_property'

    # Pane/panel controls
    if 'Pane' in ct or 'pane' in ct:
        return 'navigation'

    return 'other'


# ──────────────────────────────────────────────────────────────────────
# Probe 2: UI State — classify from observation + thought
# ──────────────────────────────────────────────────────────────────────

def classify_ui_state(observation: str, thought: str) -> str:
    """Classify current UI state from observation and thought text."""
    obs_lower = observation.lower()
    thought_lower = thought.lower()
    combined = obs_lower + ' ' + thought_lower

    # Dialog/popup present
    dialog_keywords = [
        'dialog', 'dialogue', 'pop-up', 'popup', 'message box',
        'window appears', 'dialog box', 'modal', 'prompt window',
        'confirmation', 'alert',
    ]
    if any(kw in combined for kw in dialog_keywords):
        return 'dialog'

    # Menu/dropdown open
    menu_keywords = [
        'menu', 'dropdown', 'submenu', 'context menu', 'expanded',
        'drop-down', 'flyout', 'gallery', 'list of options',
    ]
    if any(kw in combined for kw in menu_keywords):
        return 'menu_open'

    # Selection active
    selection_keywords = [
        'selected', 'highlighted', 'selection', 'cursor is',
        'text is selected', 'range is selected', 'cell is selected',
        'active cell',
    ]
    if any(kw in combined for kw in selection_keywords):
        return 'selection_active'

    # Ribbon/tab focus
    ribbon_keywords = [
        'ribbon', 'tab is active', 'tab is selected', 'toolbar',
    ]
    if any(kw in combined for kw in ribbon_keywords):
        return 'ribbon_focus'

    # Default: main view
    return 'main_view'


# ──────────────────────────────────────────────────────────────────────
# Label extraction: match probing samples to raw data + eval results
# ──────────────────────────────────────────────────────────────────────

def load_raw_step_data(data_root, trajectory_id, step_idx_filtered):
    """Load raw JSONL step for a given (trajectory_id, filtered step index).

    trajectory_id format: domain_category_filestem (e.g., excel_in_app_excel_1_98)
    step_idx_filtered: 0-indexed position after filtering out drag/no-rectangle steps
    """
    # Parse trajectory_id → file path
    parts = trajectory_id.split('_')
    domain = parts[0]

    # Category: could be 'in_app', 'search', 'online' — find it
    # The trajectory_id = domain_category_filestem, but category can have underscores
    # Actually the format is: domain + "_" + category + "_" + filestem
    # where filestem = original jsonl filename without extension
    # category is one of: in_app, search, online
    for cat_len in [1, 2]:  # category could be 1 or 2 parts
        cat_parts = parts[1:1+cat_len]
        candidate_cat = '_'.join(cat_parts)
        filestem = '_'.join(parts[1+cat_len:])
        candidate_path = os.path.join(
            data_root, 'data', domain, candidate_cat, 'success',
            filestem + '.jsonl'
        )
        if os.path.exists(candidate_path):
            break
    else:
        return None

    # Read the file and find the filtered step
    filtered_idx = 0
    with open(candidate_path) as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line.strip())
            action = d['step']['action']
            if action.get('function', '') == 'drag':
                continue
            if not action.get('rectangle', {}):
                continue

            if filtered_idx == step_idx_filtered:
                return d['step']
            filtered_idx += 1

    return None


def load_eval_results(eval_path):
    """Load AR evaluation results, return dict of (trajectory_id, step_num) → result."""
    with open(eval_path) as f:
        data = json.load(f)

    step_results = {}
    traj_first_error = {}

    for t in data['trajectory_results']:
        tid = t['trajectory_id']
        first_err = t.get('first_error_step')
        traj_first_error[tid] = first_err

        for sr in t['step_results']:
            step_num = sr['step_num']  # 1-indexed
            step_results[(tid, step_num)] = sr

    return step_results, traj_first_error


def extract_new_labels(args):
    """Extract new labels for existing 3000 probing samples."""
    print("Loading existing labels...")
    with open(os.path.join(args.output_dir, 'labels.json')) as f:
        labels = json.load(f)
    n = len(labels)
    print(f"  {n} samples")

    # Load eval results
    print("Loading eval results...")
    eval_path = os.path.join(
        args.eval_results_dir,
        'ar_evaluation_results_20260320_055609.json'
    )
    eval_steps, traj_first_error = load_eval_results(eval_path)
    print(f"  {len(eval_steps)} step results, {len(traj_first_error)} trajectories")

    # Process each sample
    new_labels = []
    stats = {
        'target_matched': 0,
        'eval_matched': 0,
        'eval_clean': 0,
        'eval_clean_correct': 0,
        'eval_clean_wrong': 0,
    }

    for i, lbl in enumerate(labels):
        tid = lbl['trajectory_id']
        step_idx = lbl['step_idx']

        entry = {
            'trajectory_id': tid,
            'step_idx': step_idx,
            'domain': lbl['domain'],
            'is_pattern_b': lbl['is_pattern_b'],
        }

        # --- Probe 1 & 2: Load raw step data ---
        raw_step = load_raw_step_data(args.data_root, tid, step_idx)
        if raw_step:
            stats['target_matched'] += 1
            ct = raw_step['action'].get('control_test', '')
            entry['control_test'] = ct
            entry['target_element_class'] = classify_target_element(ct)

            obs = raw_step.get('observation', '')
            thought = raw_step.get('thought', '')
            entry['ui_state_class'] = classify_ui_state(obs, thought)
            entry['observation_snippet'] = obs[:200]
        else:
            entry['control_test'] = ''
            entry['target_element_class'] = 'unknown'
            entry['ui_state_class'] = 'unknown'
            entry['observation_snippet'] = ''

        # --- Probe 3: Correctness from eval ---
        # Match: eval step_num is 1-indexed, our step_idx is 0-indexed (filtered)
        # The eval also filters drag/no-rect, so step_num = step_idx + 1
        eval_step_num = step_idx + 1
        eval_key = (tid, eval_step_num)

        if eval_key in eval_steps:
            stats['eval_matched'] += 1
            sr = eval_steps[eval_key]
            first_err = traj_first_error.get(tid)

            # Clean label: step <= first_error_step has unambiguous context
            if first_err is None:
                # All steps correct in this trajectory
                entry['step_correct'] = True
                entry['correctness_clean'] = True
                stats['eval_clean'] += 1
                stats['eval_clean_correct'] += 1
            elif eval_step_num < first_err:
                entry['step_correct'] = True
                entry['correctness_clean'] = True
                stats['eval_clean'] += 1
                stats['eval_clean_correct'] += 1
            elif eval_step_num == first_err:
                entry['step_correct'] = False
                entry['correctness_clean'] = True
                stats['eval_clean'] += 1
                stats['eval_clean_wrong'] += 1
            else:
                # After first error: model saw corrupted context
                entry['step_correct'] = bool(sr['success'])
                entry['correctness_clean'] = False
        else:
            entry['step_correct'] = None
            entry['correctness_clean'] = False

        new_labels.append(entry)

        if (i + 1) % 500 == 0:
            print(f"  Processed {i+1}/{n}")

    # Save
    out_path = os.path.join(args.output_dir, 'labels_v2.json')
    with open(out_path, 'w') as f:
        json.dump(new_labels, f, indent=2)

    print(f"\nLabel extraction complete:")
    print(f"  Target element matched: {stats['target_matched']}/{n}")
    print(f"  Eval matched: {stats['eval_matched']}/{n}")
    print(f"  Eval clean: {stats['eval_clean']} "
          f"(correct: {stats['eval_clean_correct']}, wrong: {stats['eval_clean_wrong']})")

    # Print distributions
    te_counter = Counter(e['target_element_class'] for e in new_labels)
    ui_counter = Counter(e['ui_state_class'] for e in new_labels)
    print(f"\n  Target element classes: {dict(te_counter.most_common())}")
    print(f"  UI state classes: {dict(ui_counter.most_common())}")

    print(f"\nSaved to {out_path}")
    return new_labels


# ──────────────────────────────────────────────────────────────────────
# Probe training
# ──────────────────────────────────────────────────────────────────────

def train_probes_v2(args):
    """Train 3 redesigned probes on existing hidden states."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import accuracy_score, f1_score

    PCA_DIM = 256

    # Load labels
    labels_path = os.path.join(args.output_dir, 'labels_v2.json')
    print(f"Loading labels from {labels_path}...")
    with open(labels_path) as f:
        labels = json.load(f)
    n = len(labels)
    print(f"  {n} samples")

    is_pattern_b = np.array([l['is_pattern_b'] for l in labels])
    train_idx = np.where(~is_pattern_b)[0]
    test_idx = np.where(is_pattern_b)[0]
    print(f"  Train: {len(train_idx)}, Test: {len(test_idx)}")

    # ── Probe 1: Target Element ──
    # Filter out 'unknown' and 'other' for cleaner signal, or keep 'other'
    te_labels_raw = [l['target_element_class'] for l in labels]
    le_te = LabelEncoder()
    te_encoded = le_te.fit_transform(te_labels_raw)
    te_classes = le_te.classes_
    te_majority = max(np.bincount(te_encoded)) / n
    print(f"\n  Probe 1 — Target Element: {len(te_classes)} classes, "
          f"majority={te_majority:.3f}")
    print(f"    Classes: {dict(Counter(te_labels_raw).most_common())}")

    # ── Probe 2: UI State ──
    ui_labels_raw = [l['ui_state_class'] for l in labels]
    le_ui = LabelEncoder()
    ui_encoded = le_ui.fit_transform(ui_labels_raw)
    ui_classes = le_ui.classes_
    ui_majority = max(np.bincount(ui_encoded)) / n
    print(f"\n  Probe 2 — UI State: {len(ui_classes)} classes, "
          f"majority={ui_majority:.3f}")
    print(f"    Classes: {dict(Counter(ui_labels_raw).most_common())}")

    # ── Probe 3: Correctness ──
    # Only use clean samples; use 5-fold CV (Pattern B test set too small = 62)
    clean_mask = np.array([l['correctness_clean'] for l in labels])
    correct_labels = np.array([
        int(l['step_correct']) if l['step_correct'] is not None else -1
        for l in labels
    ])

    clean_idx = np.where(clean_mask)[0]
    n_clean = len(clean_idx)
    n_clean_correct = (correct_labels[clean_mask] == 1).sum()
    n_clean_wrong = (correct_labels[clean_mask] == 0).sum()
    corr_majority = max(n_clean_correct, n_clean_wrong) / n_clean if n_clean > 0 else 0
    print(f"\n  Probe 3 — Correctness: {n_clean} clean samples "
          f"(correct: {n_clean_correct}, wrong: {n_clean_wrong}), "
          f"majority={corr_majority:.3f}")
    print(f"    Using 5-fold CV on all clean samples (Pattern B test too small)")

    # ── Probe configs ──
    # Probes 1 & 2: standard Pattern B train/test split
    # Probe 3: 5-fold CV only (clean_idx subset, no separate test set)
    split_probe_configs = [
        {
            'name': 'target_element',
            'token_type': 'last',
            'y_all': te_encoded,
            'train_idx': train_idx,
            'test_idx': test_idx,
            'classes': te_classes,
            'majority': te_majority,
            'description': 'Target Element (last token → element category)',
        },
        {
            'name': 'ui_state',
            'token_type': 'image',
            'y_all': ui_encoded,
            'train_idx': train_idx,
            'test_idx': test_idx,
            'classes': ui_classes,
            'majority': ui_majority,
            'description': 'UI State (image tokens → state class)',
        },
    ]

    correctness_config = {
        'name': 'correctness',
        'token_type': 'last',
        'y_all': correct_labels,
        'subset_idx': clean_idx,  # only use clean samples
        'classes': ['wrong', 'correct'],
        'majority': corr_majority,
        'description': 'Correctness (last token → correct/wrong, 5-fold CV)',
    }

    results = {}

    for layer_idx in range(NUM_TRANSFORMER_LAYERS):
        t0 = time.time()
        layer_results = {}

        # --- Probes 1 & 2: train/test split ---
        for pc in split_probe_configs:
            fpath = os.path.join(
                args.output_dir,
                f'layer_{layer_idx}_{pc["token_type"]}.npy'
            )
            if not os.path.exists(fpath):
                continue

            X = np.load(fpath)
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            pca = PCA(n_components=PCA_DIM, random_state=42)
            X_reduced = pca.fit_transform(X_scaled)

            tr_idx = pc['train_idx']
            te_idx = pc['test_idx']

            X_train = X_reduced[tr_idx]
            X_test = X_reduced[te_idx]
            y_train = pc['y_all'][tr_idx]
            y_test = pc['y_all'][te_idx]

            if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
                continue

            clf = LogisticRegression(max_iter=500, C=1.0, solver='lbfgs')
            clf.fit(X_train, y_train)

            train_acc = accuracy_score(y_train, clf.predict(X_train))
            test_acc = accuracy_score(y_test, clf.predict(X_test))
            test_f1 = f1_score(y_test, clf.predict(X_test), average='macro')

            cv_scores = cross_val_score(
                LogisticRegression(max_iter=500, C=1.0, solver='lbfgs'),
                X_reduced, pc['y_all'], cv=5, scoring='accuracy'
            )

            layer_results[pc['name']] = {
                'train_acc': float(train_acc),
                'test_acc': float(test_acc),
                'test_f1': float(test_f1),
                'cv_mean': float(cv_scores.mean()),
                'cv_std': float(cv_scores.std()),
                'pca_var_ratio': float(pca.explained_variance_ratio_.sum()),
            }

        # --- Probe 3: Correctness (5-fold CV only) ---
        cc = correctness_config
        fpath = os.path.join(
            args.output_dir, f'layer_{layer_idx}_{cc["token_type"]}.npy'
        )
        if os.path.exists(fpath) and len(cc['subset_idx']) >= 10:
            X = np.load(fpath)
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            pca = PCA(n_components=PCA_DIM, random_state=42)
            X_reduced = pca.fit_transform(X_scaled)

            X_sub = X_reduced[cc['subset_idx']]
            y_sub = cc['y_all'][cc['subset_idx']]

            if len(np.unique(y_sub)) >= 2:
                cv_acc = cross_val_score(
                    LogisticRegression(max_iter=500, C=1.0, solver='lbfgs'),
                    X_sub, y_sub, cv=5, scoring='accuracy'
                )
                cv_f1 = cross_val_score(
                    LogisticRegression(max_iter=500, C=1.0, solver='lbfgs'),
                    X_sub, y_sub, cv=5, scoring='f1_macro'
                )

                layer_results[cc['name']] = {
                    'cv_acc_mean': float(cv_acc.mean()),
                    'cv_acc_std': float(cv_acc.std()),
                    'cv_f1_mean': float(cv_f1.mean()),
                    'cv_f1_std': float(cv_f1.std()),
                    'pca_var_ratio': float(pca.explained_variance_ratio_.sum()),
                }

        if layer_results:
            results[f'layer_{layer_idx}'] = layer_results

        elapsed = time.time() - t0
        if (layer_idx + 1) % 4 == 0 or layer_idx == 0:
            summary_parts = []
            for pc in split_probe_configs:
                if pc['name'] in layer_results:
                    r = layer_results[pc['name']]
                    summary_parts.append(
                        f"{pc['name']}: {r['test_acc']:.3f} (F1={r['test_f1']:.3f})"
                    )
            if cc['name'] in layer_results:
                r = layer_results[cc['name']]
                summary_parts.append(
                    f"correctness: CV={r['cv_acc_mean']:.3f}±{r['cv_acc_std']:.3f}"
                )
            print(f"  Layer {layer_idx:2d} ({elapsed:.1f}s): " +
                  " | ".join(summary_parts))

    # ── Cross-probes (all token types × all probe labels) ──
    print("\n--- Cross-probes (sampled layers) ---")
    cross_results = {}
    all_probe_configs = split_probe_configs + [correctness_config]

    for layer_idx in [0, 6, 13, 20, 27]:
        layer_cross = {}
        for token_type in ['image', 'history', 'last']:
            fpath = os.path.join(
                args.output_dir, f'layer_{layer_idx}_{token_type}.npy'
            )
            if not os.path.exists(fpath):
                continue

            X = np.load(fpath)
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            pca = PCA(n_components=PCA_DIM, random_state=42)
            X_reduced = pca.fit_transform(X_scaled)

            for pc in all_probe_configs:
                pname = pc['name']

                if pname == 'correctness':
                    # CV-only for correctness
                    sub_idx = pc['subset_idx']
                    X_sub = X_reduced[sub_idx]
                    y_sub = pc['y_all'][sub_idx]
                    if len(np.unique(y_sub)) < 2:
                        continue
                    cv = cross_val_score(
                        LogisticRegression(max_iter=500, C=1.0, solver='lbfgs'),
                        X_sub, y_sub, cv=5, scoring='accuracy'
                    )
                    layer_cross[f'{token_type}→{pname}'] = {
                        'cv_acc': float(cv.mean()),
                    }
                else:
                    tr_idx = pc['train_idx']
                    te_idx = pc['test_idx']
                    y_all = pc['y_all']
                    X_train = X_reduced[tr_idx]
                    X_test = X_reduced[te_idx]
                    y_train = y_all[tr_idx]
                    y_test = y_all[te_idx]

                    if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
                        continue

                    clf = LogisticRegression(max_iter=500, C=1.0, solver='lbfgs')
                    clf.fit(X_train, y_train)
                    acc = accuracy_score(y_test, clf.predict(X_test))
                    layer_cross[f'{token_type}→{pname}'] = {
                        'acc': float(acc),
                    }

        cross_results[f'layer_{layer_idx}'] = layer_cross
        parts = []
        for k, v in sorted(layer_cross.items()):
            val = v.get('acc', v.get('cv_acc', 0))
            parts.append(f"{k}: {val:.3f}")
        print(f"  Layer {layer_idx:2d}: " + " | ".join(parts))

    # Save
    all_configs = split_probe_configs + [correctness_config]
    output = {
        'primary_probes': results,
        'cross_probes': cross_results,
        'metadata': {
            'n_samples': n,
            'n_train': len(train_idx),
            'n_test': len(test_idx),
            'probe_configs': {
                pc['name']: {
                    'token_type': pc['token_type'],
                    'description': pc['description'],
                    'classes': pc['classes'].tolist() if hasattr(pc['classes'], 'tolist')
                               else list(pc['classes']),
                    'majority': pc['majority'],
                }
                for pc in all_configs
            },
            'n_clean_correctness': int(n_clean),
        },
    }

    out_path = os.path.join(args.output_dir, 'probe_v2_results.json')
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")


# ──────────────────────────────────────────────────────────────────────
# Plotting
# ──────────────────────────────────────────────────────────────────────

def plot_v2(args):
    """Generate layer-wise probing curves for v2 probes."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    with open(os.path.join(args.output_dir, 'probe_v2_results.json')) as f:
        data = json.load(f)

    results = data['primary_probes']
    metadata = data['metadata']
    pconfigs = metadata['probe_configs']

    layers = list(range(NUM_TRANSFORMER_LAYERS))

    probe_info = [
        ('target_element', 'Target Element', 'tab:blue', 'o'),
        ('ui_state', 'UI State', 'tab:green', 's'),
        ('correctness', 'Correctness', 'tab:red', '^'),
    ]

    # ── Main plot: accuracy (test_acc for P1/P2, cv_acc_mean for P3) ──
    fig, ax = plt.subplots(1, 1, figsize=(14, 6))

    for probe_name, display_name, color, marker in probe_info:
        accs = []
        for li in layers:
            key = f'layer_{li}'
            r = results.get(key, {}).get(probe_name, {})
            # P3 uses cv_acc_mean, P1/P2 use test_acc
            acc = r.get('test_acc', r.get('cv_acc_mean', None))
            accs.append(acc)

        majority = pconfigs[probe_name]['majority']
        valid_accs = [(l, a) for l, a in zip(layers, accs) if a is not None]
        if valid_accs:
            ls, acs = zip(*valid_accs)
            suffix = ' (5-fold CV)' if probe_name == 'correctness' else ''
            ax.plot(ls, acs, f'-{marker}', color=color, markersize=4,
                    linewidth=2,
                    label=f'{display_name}{suffix} [majority={majority:.2f}]')
            ax.axhline(y=majority, color=color, linestyle='--', alpha=0.3)

    ax.set_xlabel('Transformer Layer', fontsize=13)
    ax.set_ylabel('Probe Accuracy', fontsize=13)
    ax.set_title('Layer-wise Probing v2: Target Element / UI State / Correctness',
                 fontsize=14)
    ax.legend(fontsize=11)
    ax.set_xticks(layers)
    ax.set_xlim(-0.5, NUM_TRANSFORMER_LAYERS - 0.5)
    ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(args.output_dir, 'probing_v2_curves.png'), dpi=150)
    print(f"Plot saved: probing_v2_curves.png")

    # ── F1 plot ──
    fig2, ax2 = plt.subplots(1, 1, figsize=(14, 6))
    for probe_name, display_name, color, marker in probe_info:
        f1s = []
        for li in layers:
            key = f'layer_{li}'
            r = results.get(key, {}).get(probe_name, {})
            f1 = r.get('test_f1', r.get('cv_f1_mean', None))
            f1s.append(f1)

        valid = [(l, f) for l, f in zip(layers, f1s) if f is not None]
        if valid:
            ls, fs = zip(*valid)
            suffix = ' (5-fold CV)' if probe_name == 'correctness' else ''
            ax2.plot(ls, fs, f'-{marker}', color=color, markersize=4,
                     linewidth=2, label=f'{display_name}{suffix} (macro F1)')

    ax2.set_xlabel('Transformer Layer', fontsize=13)
    ax2.set_ylabel('Macro F1', fontsize=13)
    ax2.set_title('Layer-wise Probing v2: Macro F1 Scores', fontsize=14)
    ax2.legend(fontsize=11)
    ax2.set_xticks(layers)
    ax2.set_xlim(-0.5, NUM_TRANSFORMER_LAYERS - 0.5)
    ax2.set_ylim(0, 1.0)
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    fig2.savefig(os.path.join(args.output_dir, 'probing_v2_f1.png'), dpi=150)
    print(f"Plot saved: probing_v2_f1.png")

    # ── Cross-probe heatmap ──
    cross = data.get('cross_probes', {})
    if cross:
        sampled_layers = [0, 6, 13, 20, 27]
        token_types = ['image', 'history', 'last']
        probe_names = ['target_element', 'ui_state', 'correctness']

        heatmap = np.zeros((len(sampled_layers) * len(token_types),
                            len(probe_names)))
        row_labels = []
        for li_idx, layer_idx in enumerate(sampled_layers):
            key = f'layer_{layer_idx}'
            for ti, tt in enumerate(token_types):
                row_idx = li_idx * len(token_types) + ti
                row_labels.append(f'L{layer_idx} {tt}')
                for pi, pn in enumerate(probe_names):
                    probe_key = f'{tt}→{pn}'
                    entry = cross.get(key, {}).get(probe_key, {})
                    val = entry.get('acc', entry.get('cv_acc', 0))
                    heatmap[row_idx, pi] = val

        fig3, ax3 = plt.subplots(1, 1, figsize=(8, 12))
        im = ax3.imshow(heatmap, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
        ax3.set_xticks(range(len(probe_names)))
        ax3.set_xticklabels(['Target\nElement', 'UI State', 'Correctness'],
                            fontsize=11)
        ax3.set_yticks(range(len(row_labels)))
        ax3.set_yticklabels(row_labels, fontsize=9)
        for i in range(heatmap.shape[0]):
            for j in range(heatmap.shape[1]):
                ax3.text(j, i, f'{heatmap[i,j]:.2f}',
                         ha='center', va='center', fontsize=8)
        ax3.set_title('Cross-probe v2: Token Type × Probe (sampled layers)',
                      fontsize=13)
        fig3.colorbar(im, ax=ax3, shrink=0.6)
        plt.tight_layout()
        fig3.savefig(os.path.join(args.output_dir, 'cross_probe_v2_heatmap.png'),
                     dpi=150)
        print(f"Heatmap saved: cross_probe_v2_heatmap.png")

    # ── Summary table ──
    print("\n" + "=" * 95)
    print("LAYER-WISE PROBING v2 SUMMARY")
    print("  Target Element & UI State: Pattern B test accuracy")
    print("  Correctness: 5-fold CV accuracy (on clean samples)")
    print("=" * 95)
    print(f"{'Layer':>5} {'TargetElem':>12} {'UIState':>10} {'Correct(CV)':>12}  "
          f"{'TE-F1':>8} {'UI-F1':>8} {'C-F1(CV)':>10}")
    print("-" * 95)
    for li in layers:
        key = f'layer_{li}'
        te = results.get(key, {}).get('target_element', {})
        ui = results.get(key, {}).get('ui_state', {})
        co = results.get(key, {}).get('correctness', {})
        print(f"{li:>5} "
              f"{te.get('test_acc', 0):>11.3f} "
              f"{ui.get('test_acc', 0):>9.3f} "
              f"{co.get('cv_acc_mean', 0):>11.3f}  "
              f"{te.get('test_f1', 0):>7.3f} "
              f"{ui.get('test_f1', 0):>7.3f} "
              f"{co.get('cv_f1_mean', 0):>9.3f}")

    for pn, dn, _, _ in probe_info:
        print(f"\n{dn}: majority baseline = {pconfigs[pn]['majority']:.3f}, "
              f"classes = {pconfigs[pn]['classes']}")


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Layer-wise probing v2")
    parser.add_argument('--mode', type=str, required=True,
                        choices=['label', 'probe', 'plot', 'all'],
                        help="'label' to extract new labels, 'probe' to train, "
                             "'plot' to generate curves, 'all' for everything")
    parser.add_argument('--output_dir', type=str, required=True,
                        help="Directory with existing npy files and labels.json")
    parser.add_argument('--data_root', type=str, default=None,
                        help="GUI-360 test data root (for label extraction)")
    parser.add_argument('--eval_results_dir', type=str, default=None,
                        help="Directory with AR eval results (for correctness labels)")

    args = parser.parse_args()

    if args.mode in ('label', 'all'):
        if not args.data_root:
            parser.error("--data_root required for label extraction")
        if not args.eval_results_dir:
            parser.error("--eval_results_dir required for label extraction")
        extract_new_labels(args)

    if args.mode in ('probe', 'all'):
        train_probes_v2(args)

    if args.mode in ('plot', 'all'):
        plot_v2(args)
