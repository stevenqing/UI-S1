"""Layer-wise probing experiment for Qwen2.5-VL-7B on GUI-360.

Extracts hidden states from all 28 transformer layers at three token positions
(image tokens, history tokens, last token), then trains linear probes to test
each layer's encoding of scene, progress, and action information.

Three modes:
  extract  — Forward pass, extract hidden states for all layers + token types
  probe    — Train linear probes (scene/progress/action) on extracted states
  plot     — Generate layer-wise probing accuracy curves

Probe tasks:
  Scene   (image token mean)   → domain classification (ppt/excel/word)
  Progress (history token mean) → step position bucket (5-class)
  Action   (last token)         → action type classification (click/type/scroll/...)
"""

import argparse
import json
import os
import sys
import time
import traceback
from collections import defaultdict

import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
EVAL_ROOT = os.path.join(PROJECT_ROOT, 'train_GUI_360', 'GUI-360-eval')
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, EVAL_ROOT)

from prompts.prompt_action_prediction import (
    ACTION_PREDICTION_SYS_PROMPT_GPT,
    ACTION_PREDICTION_USER_PROMPT_QWEN,
    SUPPORTED_ACTIONS_WORD,
    SUPPORTED_ACTIONS_EXCEL,
    SUPPORTED_ACTIONS_PPT,
)

DOMAIN_ACTIONS = {
    'word': SUPPORTED_ACTIONS_WORD,
    'excel': SUPPORTED_ACTIONS_EXCEL,
    'ppt': SUPPORTED_ACTIONS_PPT,
}

NUM_TRANSFORMER_LAYERS = 28


# ─── Data Loading ───────────────────────────────────────────────────────

def load_gui360_trajectories(data_root, pattern_b_ids=None, max_trajectories=None):
    """Load GUI-360 test trajectories (matching evaluator format).

    Args:
        data_root: Path to GUI-360 test directory (contains data/ and image/)
        pattern_b_ids: If set, only load these trajectory IDs
        max_trajectories: Max trajectories to load

    Returns:
        List of trajectory dicts with keys:
          trajectory_id, request, domain, category, steps: [...]
    """
    data_path = os.path.join(data_root, "data")
    trajectories = []
    id_set = set(pattern_b_ids) if pattern_b_ids else None

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
                trajectory_id = f"{domain}_{category}_{file_stem}"

                if id_set and trajectory_id not in id_set:
                    continue

                file_path = os.path.join(success_path, fname)
                steps = []

                with open(file_path, "r") as f:
                    for line in f:
                        if not line.strip():
                            continue
                        data = json.loads(line.strip())

                        # Filter drag and no-rectangle steps (same as evaluator)
                        action = data["step"]["action"]
                        if action.get("function", "") == "drag":
                            continue
                        if not action.get("rectangle", {}):
                            continue

                        # Build image path
                        clean_img = os.path.join(
                            data_root, "image", domain, category,
                            data["step"]["screenshot_clean"],
                        )
                        if not os.path.exists(clean_img):
                            continue

                        steps.append({
                            "request": data["request"],
                            "screenshot_clean": clean_img,
                            "thought": data["step"].get("thought", ""),
                            "subtask": data["step"].get("subtask", ""),
                            "observation": data["step"].get("observation", ""),
                            "action": action,
                            "status": data["step"].get("status", "CONTINUE"),
                            "domain": domain,
                            "category": category,
                            "step_index": len(steps),
                        })

                if steps:
                    trajectories.append({
                        "trajectory_id": trajectory_id,
                        "request": steps[0]["request"],
                        "domain": domain,
                        "category": category,
                        "steps": steps,
                    })

                if max_trajectories and len(trajectories) >= max_trajectories:
                    return trajectories

    return trajectories


def format_action_brief(action):
    """Create a brief text summary of a GT action."""
    func = action.get("function", "unknown")
    args = action.get("args", {})
    rect = action.get("rectangle", {})

    if func == "click":
        cx = (rect.get("left", 0) + rect.get("right", 0)) / 2
        cy = (rect.get("top", 0) + rect.get("bottom", 0)) / 2
        button = args.get("button", "left")
        return f"click(coordinate=[{cx:.0f}, {cy:.0f}], button='{button}')"
    elif func == "type":
        cx = (rect.get("left", 0) + rect.get("right", 0)) / 2
        cy = (rect.get("top", 0) + rect.get("bottom", 0)) / 2
        keys = args.get("keys", args.get("text", ""))
        if len(keys) > 30:
            keys = keys[:30] + "..."
        return f"type(coordinate=[{cx:.0f}, {cy:.0f}], keys='{keys}')"
    elif func == "scroll":
        cx = (rect.get("left", 0) + rect.get("right", 0)) / 2
        cy = (rect.get("top", 0) + rect.get("bottom", 0)) / 2
        direction = args.get("direction", "down")
        return f"scroll(coordinate=[{cx:.0f}, {cy:.0f}], direction='{direction}')"
    elif func == "hotkey":
        keys = args.get("keys", "")
        return f"hotkey(keys='{keys}')"
    else:
        return f"{func}(...)"


# ─── Token Position Finding ────────────────────────────────────────────

def find_subsequence(seq, subseq):
    """Find first occurrence of subseq in seq. Returns start index or -1."""
    n, m = len(seq), len(subseq)
    for i in range(n - m + 1):
        if seq[i:i + m] == subseq:
            return i
    return -1


def find_token_positions(input_ids, processor):
    """Find image, history, and last token positions in the input sequence.

    Returns:
        image_positions: list of int (vision token positions)
        history_positions: list of int (history text token positions)
        last_position: int (last token position)
    """
    ids = input_ids.tolist()

    # --- Image token positions ---
    # Qwen2.5-VL uses <|vision_start|> ... <|vision_end|> to bracket image tokens
    vision_start_id = processor.tokenizer.convert_tokens_to_ids('<|vision_start|>')
    vision_end_id = processor.tokenizer.convert_tokens_to_ids('<|vision_end|>')

    image_positions = []
    in_vision = False
    for i, tid in enumerate(ids):
        if tid == vision_start_id:
            in_vision = True
            continue
        elif tid == vision_end_id:
            in_vision = False
            continue
        if in_vision:
            image_positions.append(i)

    # --- History token positions ---
    # Find text between "The history of actions are:\n" and "The actions supported are"
    # Note: markers must be encoded WITHOUT \n prefix/suffix because BPE merges
    # newlines with adjacent tokens differently in context vs isolation.
    marker_start_text = "The history of actions are:\n"
    marker_end_text = "The actions supported are"

    marker_start_ids = processor.tokenizer.encode(
        marker_start_text, add_special_tokens=False
    )
    marker_end_ids = processor.tokenizer.encode(
        marker_end_text, add_special_tokens=False
    )

    history_start = find_subsequence(ids, marker_start_ids)
    # Search for end marker AFTER the start marker
    history_end = -1
    if history_start >= 0:
        offset = history_start + len(marker_start_ids)
        rel = find_subsequence(ids[offset:], marker_end_ids)
        if rel >= 0:
            history_end = offset + rel

    history_positions = []
    if history_start >= 0 and history_end >= 0:
        h_start = history_start + len(marker_start_ids)
        history_positions = list(range(h_start, history_end))

    # --- Last token ---
    last_position = len(ids) - 1

    return image_positions, history_positions, last_position


# ─── Hidden State Extraction ───────────────────────────────────────────

def extract_hidden_states(args):
    """Extract hidden states from all 28 layers at 3 token positions."""
    import torch
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    from qwen_vl_utils import process_vision_info

    print(f"Loading model from {args.model_path}...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="flash_attention_2",
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(args.model_path)

    num_layers = model.config.num_hidden_layers
    print(f"Model has {num_layers} transformer layers")
    assert num_layers == NUM_TRANSFORMER_LAYERS

    # Load data
    pattern_b_ids = None
    if args.pattern_b_file:
        with open(args.pattern_b_file) as f:
            pattern_b_ids_list = json.load(f)
        pattern_b_ids_set = set(pattern_b_ids_list)
        print(f"Loaded {len(pattern_b_ids_set)} Pattern B IDs")
    else:
        pattern_b_ids_set = set()

    trajectories = load_gui360_trajectories(
        args.data_root,
        max_trajectories=args.max_trajectories,
    )
    # Shuffle to get balanced domain coverage (default order is alphabetical = all excel first)
    import random
    random.seed(42)
    random.shuffle(trajectories)
    domain_counts = defaultdict(int)
    for t in trajectories:
        domain_counts[t["domain"]] += 1
    print(f"Loaded {len(trajectories)} trajectories ({dict(domain_counts)})")

    # Prepare storage: per-layer, per-token-type lists
    all_labels = []
    # We'll accumulate hidden states in lists, then save as numpy arrays
    layer_image_states = defaultdict(list)    # layer_idx -> list of vectors
    layer_history_states = defaultdict(list)
    layer_last_states = defaultdict(list)

    sample_count = 0
    skip_count = 0
    t_start = time.time()

    for traj_idx, traj in enumerate(trajectories):
        domain = traj["domain"]
        actions_str = DOMAIN_ACTIONS.get(domain, SUPPORTED_ACTIONS_WORD)
        action_history = []  # cumulative GT action briefs

        for step_idx, step in enumerate(traj["steps"]):
            if sample_count >= args.max_samples:
                break

            # Build history string
            if action_history:
                history_str = "\n".join(
                    f"Step {i+1}: {a}" for i, a in enumerate(action_history)
                )
            else:
                history_str = "No previous actions."

            # Build prompt using standard AR template
            user_text = ACTION_PREDICTION_USER_PROMPT_QWEN.format(
                instruction=step["request"],
                history=history_str,
                actions=actions_str,
            )

            messages = [
                {
                    'role': 'system',
                    'content': [{'type': 'text', 'text': ACTION_PREDICTION_SYS_PROMPT_GPT.strip()}],
                },
                {
                    'role': 'user',
                    'content': [
                        {'type': 'text', 'text': user_text},
                        {'type': 'image', 'image': step["screenshot_clean"]},
                    ],
                },
            ]

            try:
                text = processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                image_inputs, video_inputs = process_vision_info(messages)
                inputs = processor(
                    text=[text],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                ).to(model.device)

                with torch.no_grad():
                    outputs = model(
                        **inputs,
                        output_hidden_states=True,
                        return_dict=True,
                    )

                # Find token positions
                input_ids = inputs['input_ids'][0]
                image_pos, history_pos, last_pos = find_token_positions(
                    input_ids, processor
                )

                if not image_pos:
                    skip_count += 1
                    action_history.append(format_action_brief(step["action"]))
                    continue

                # outputs.hidden_states: tuple of (num_layers + 1) tensors
                # [0] = embedding output, [1..28] = transformer layer outputs
                for layer_idx in range(num_layers):
                    hs = outputs.hidden_states[layer_idx + 1][0]  # [seq_len, hidden_dim]

                    # Image tokens: mean pool
                    img_hs = hs[image_pos].mean(dim=0).cpu().float().numpy()
                    layer_image_states[layer_idx].append(img_hs)

                    # History tokens: mean pool (fall back to last token if empty)
                    if history_pos:
                        hist_hs = hs[history_pos].mean(dim=0).cpu().float().numpy()
                    else:
                        hist_hs = hs[last_pos].cpu().float().numpy()
                    layer_history_states[layer_idx].append(hist_hs)

                    # Last token
                    last_hs = hs[last_pos].cpu().float().numpy()
                    layer_last_states[layer_idx].append(last_hs)

                # Labels
                gt_action_type = step["action"].get("function", "unknown")
                subtask = step.get("subtask", "")
                num_steps = len(traj["steps"])
                position_norm = step_idx / max(num_steps - 1, 1)

                # Subtask index within trajectory
                subtask_list = []
                current_st = traj["steps"][0].get("subtask", "")
                subtask_list.append(current_st)
                for s in traj["steps"][1:step_idx + 1]:
                    st = s.get("subtask", "")
                    if st != current_st:
                        subtask_list.append(st)
                        current_st = st
                subtask_idx = len(set(subtask_list))  # number of unique subtasks seen so far

                all_labels.append({
                    'trajectory_id': traj["trajectory_id"],
                    'step_idx': step_idx,
                    'num_steps': num_steps,
                    'domain': domain,
                    'category': traj["category"],
                    'action_type': gt_action_type,
                    'subtask': subtask,
                    'subtask_idx': subtask_idx,
                    'position_norm': position_norm,
                    'position_bucket': min(int(position_norm * 5), 4),  # 0-4
                    'is_pattern_b': traj["trajectory_id"] in pattern_b_ids_set,
                    'n_image_tokens': len(image_pos),
                    'n_history_tokens': len(history_pos),
                })
                sample_count += 1

                if sample_count % 50 == 0:
                    elapsed = time.time() - t_start
                    rate = sample_count / elapsed
                    print(f"  [{sample_count}/{args.max_samples}] "
                          f"traj {traj_idx+1}/{len(trajectories)} "
                          f"({rate:.1f} samples/s, "
                          f"img_tok={len(image_pos)}, hist_tok={len(history_pos)})")

            except Exception as e:
                print(f"  Error at traj {traj_idx} step {step_idx}: {e}")
                traceback.print_exc()
                skip_count += 1

            # Update action history with GT
            action_history.append(format_action_brief(step["action"]))

        if sample_count >= args.max_samples:
            print(f"Reached max_samples limit ({args.max_samples})")
            break

    elapsed = time.time() - t_start
    print(f"\nExtraction complete: {sample_count} samples, {skip_count} skipped, "
          f"{elapsed:.1f}s ({sample_count/max(elapsed,1):.1f} samples/s)")

    # Save
    os.makedirs(args.output_dir, exist_ok=True)

    # Labels
    with open(os.path.join(args.output_dir, 'labels.json'), 'w') as f:
        json.dump(all_labels, f, indent=2)
    print(f"Saved labels: {len(all_labels)} samples")

    # Hidden states per layer per token type
    for layer_idx in range(num_layers):
        if layer_image_states[layer_idx]:
            arr = np.stack(layer_image_states[layer_idx])
            np.save(os.path.join(args.output_dir, f'layer_{layer_idx}_image.npy'), arr)

        if layer_history_states[layer_idx]:
            arr = np.stack(layer_history_states[layer_idx])
            np.save(os.path.join(args.output_dir, f'layer_{layer_idx}_history.npy'), arr)

        if layer_last_states[layer_idx]:
            arr = np.stack(layer_last_states[layer_idx])
            np.save(os.path.join(args.output_dir, f'layer_{layer_idx}_last.npy'), arr)

    # Print sizes
    total_bytes = 0
    for f in os.listdir(args.output_dir):
        fp = os.path.join(args.output_dir, f)
        total_bytes += os.path.getsize(fp)
    print(f"Total saved: {total_bytes / 1e9:.2f} GB in {args.output_dir}")


# ─── Probe Training ────────────────────────────────────────────────────

def train_probes(args):
    """Train linear probes on extracted hidden states."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import accuracy_score

    print(f"Loading labels from {args.output_dir}...")
    with open(os.path.join(args.output_dir, 'labels.json')) as f:
        labels = json.load(f)

    n = len(labels)
    print(f"Total samples: {n}")

    # Prepare label arrays
    domains = [l['domain'] for l in labels]
    action_types = [l['action_type'] for l in labels]
    position_buckets = np.array([l['position_bucket'] for l in labels])
    is_pattern_b = np.array([l['is_pattern_b'] for l in labels])

    # Encode labels
    le_domain = LabelEncoder()
    domain_encoded = le_domain.fit_transform(domains)
    le_action = LabelEncoder()
    action_encoded = le_action.fit_transform(action_types)

    print(f"Domains: {le_domain.classes_} (majority: {max(np.bincount(domain_encoded))/n:.3f})")
    print(f"Action types: {le_action.classes_} (majority: {max(np.bincount(action_encoded))/n:.3f})")
    print(f"Position buckets: {np.bincount(position_buckets)} "
          f"(majority: {max(np.bincount(position_buckets))/n:.3f})")
    print(f"Pattern B samples: {is_pattern_b.sum()} / {n}")

    # Split indices
    train_mask = ~is_pattern_b
    test_mask = is_pattern_b
    train_idx = np.where(train_mask)[0]
    test_idx = np.where(test_mask)[0]
    print(f"Train (non-Pattern-B): {len(train_idx)}, Test (Pattern B): {len(test_idx)}")

    # Probe configurations: (name, token_type, label_array, label_encoder_or_None)
    probe_configs = [
        ('scene_domain', 'image', domain_encoded, le_domain),
        ('progress_position', 'history', position_buckets, None),
        ('action_type', 'last', action_encoded, le_action),
    ]

    # PCA for speed: reduce 3584-dim to 256-dim before probing
    from sklearn.decomposition import PCA
    PCA_DIM = 256
    print(f"Using PCA({PCA_DIM}) for dimensionality reduction")

    results = {}
    import time as _time

    for layer_idx in range(NUM_TRANSFORMER_LAYERS):
        t0 = _time.time()
        layer_results = {}

        for probe_name, token_type, y_all, le in probe_configs:
            fpath = os.path.join(args.output_dir, f'layer_{layer_idx}_{token_type}.npy')
            if not os.path.exists(fpath):
                continue

            X = np.load(fpath)

            # PCA + Scale
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            pca = PCA(n_components=PCA_DIM, random_state=42)
            X_reduced = pca.fit_transform(X_scaled)

            X_train = X_reduced[train_idx]
            X_test = X_reduced[test_idx]
            y_train = y_all[train_idx]
            y_test = y_all[test_idx]

            # Train
            clf = LogisticRegression(max_iter=500, C=1.0, solver='lbfgs')
            clf.fit(X_train, y_train)

            # Evaluate
            train_acc = accuracy_score(y_train, clf.predict(X_train))
            test_acc = accuracy_score(y_test, clf.predict(X_test))

            # 5-fold CV on all data (using already-reduced features)
            cv_scores = cross_val_score(
                LogisticRegression(max_iter=500, C=1.0, solver='lbfgs'),
                X_reduced, y_all, cv=5, scoring='accuracy'
            )

            layer_results[probe_name] = {
                'train_acc': float(train_acc),
                'test_acc': float(test_acc),
                'cv_mean': float(cv_scores.mean()),
                'cv_std': float(cv_scores.std()),
                'pca_variance_ratio': float(pca.explained_variance_ratio_.sum()),
            }

        if layer_results:
            results[f'layer_{layer_idx}'] = layer_results

        elapsed = _time.time() - t0
        if (layer_idx + 1) % 4 == 0 or layer_idx == 0:
            print(f"Layer {layer_idx:2d} ({elapsed:.1f}s): " + " | ".join(
                f"{k}: {v['test_acc']:.3f}" for k, v in layer_results.items()
            ))

    # Add cross-probes: test each token type against each label
    print("\n--- Cross-probes (all token types × all labels) ---")
    cross_results = {}
    for layer_idx in [0, 6, 13, 20, 27]:  # Sample 5 layers
        layer_cross = {}
        for token_type in ['image', 'history', 'last']:
            fpath = os.path.join(args.output_dir, f'layer_{layer_idx}_{token_type}.npy')
            if not os.path.exists(fpath):
                continue
            X = np.load(fpath)
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            pca = PCA(n_components=PCA_DIM, random_state=42)
            X_reduced = pca.fit_transform(X_scaled)
            X_train = X_reduced[train_idx]
            X_test = X_reduced[test_idx]

            for label_name, y_all_label in [('domain', domain_encoded),
                                             ('position', position_buckets),
                                             ('action', action_encoded)]:
                y_train = y_all_label[train_idx]
                y_test = y_all_label[test_idx]
                clf = LogisticRegression(max_iter=1000, C=1.0, solver='lbfgs')
                clf.fit(X_train, y_train)
                acc = accuracy_score(y_test, clf.predict(X_test))
                layer_cross[f'{token_type}_→_{label_name}'] = float(acc)

        cross_results[f'layer_{layer_idx}'] = layer_cross
        print(f"Layer {layer_idx:2d}: " + " | ".join(
            f"{k}: {v:.3f}" for k, v in sorted(layer_cross.items())
        ))

    # Save results
    output = {
        'primary_probes': results,
        'cross_probes': cross_results,
        'metadata': {
            'n_samples': n,
            'n_train': len(train_idx),
            'n_test': len(test_idx),
            'domain_classes': le_domain.classes_.tolist(),
            'action_classes': le_action.classes_.tolist(),
            'domain_majority': float(max(np.bincount(domain_encoded)) / n),
            'action_majority': float(max(np.bincount(action_encoded)) / n),
            'position_majority': float(max(np.bincount(position_buckets)) / n),
        },
    }

    with open(os.path.join(args.output_dir, 'probe_results.json'), 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nProbe results saved to {os.path.join(args.output_dir, 'probe_results.json')}")


# ─── Plotting ──────────────────────────────────────────────────────────

def plot_curves(args):
    """Generate layer-wise probing accuracy curves."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    with open(os.path.join(args.output_dir, 'probe_results.json')) as f:
        data = json.load(f)

    results = data['primary_probes']
    metadata = data['metadata']

    layers = list(range(NUM_TRANSFORMER_LAYERS))
    scene_acc = []
    progress_acc = []
    action_acc = []

    for layer_idx in layers:
        key = f'layer_{layer_idx}'
        if key in results:
            scene_acc.append(results[key].get('scene_domain', {}).get('test_acc', 0))
            progress_acc.append(results[key].get('progress_position', {}).get('test_acc', 0))
            action_acc.append(results[key].get('action_type', {}).get('test_acc', 0))
        else:
            scene_acc.append(0)
            progress_acc.append(0)
            action_acc.append(0)

    fig, ax = plt.subplots(1, 1, figsize=(14, 6))

    ax.plot(layers, scene_acc, 'b-o', markersize=4, linewidth=2,
            label=f'Scene (domain) [majority={metadata["domain_majority"]:.2f}]')
    ax.plot(layers, progress_acc, 'g-s', markersize=4, linewidth=2,
            label=f'Progress (position bucket) [majority={metadata["position_majority"]:.2f}]')
    ax.plot(layers, action_acc, 'r-^', markersize=4, linewidth=2,
            label=f'Action (type) [majority={metadata["action_majority"]:.2f}]')

    # Majority baselines
    ax.axhline(y=metadata['domain_majority'], color='b', linestyle='--', alpha=0.3)
    ax.axhline(y=metadata['position_majority'], color='g', linestyle='--', alpha=0.3)
    ax.axhline(y=metadata['action_majority'], color='r', linestyle='--', alpha=0.3)

    ax.set_xlabel('Transformer Layer', fontsize=13)
    ax.set_ylabel('Probe Accuracy (Pattern B test)', fontsize=13)
    ax.set_title('Layer-wise Linear Probing: Qwen2.5-VL-7B (SFT v2) on GUI-360', fontsize=14)
    ax.legend(fontsize=11)
    ax.set_xticks(layers)
    ax.set_xlim(-0.5, NUM_TRANSFORMER_LAYERS - 0.5)
    ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(args.output_dir, 'probing_curves.png')
    fig.savefig(plot_path, dpi=150)
    print(f"Plot saved to {plot_path}")

    # Also plot CV results
    scene_cv = []
    progress_cv = []
    action_cv = []
    for layer_idx in layers:
        key = f'layer_{layer_idx}'
        if key in results:
            scene_cv.append(results[key].get('scene_domain', {}).get('cv_mean', 0))
            progress_cv.append(results[key].get('progress_position', {}).get('cv_mean', 0))
            action_cv.append(results[key].get('action_type', {}).get('cv_mean', 0))
        else:
            scene_cv.append(0)
            progress_cv.append(0)
            action_cv.append(0)

    fig2, ax2 = plt.subplots(1, 1, figsize=(14, 6))
    ax2.plot(layers, scene_cv, 'b-o', markersize=4, linewidth=2, label='Scene (5-fold CV)')
    ax2.plot(layers, progress_cv, 'g-s', markersize=4, linewidth=2, label='Progress (5-fold CV)')
    ax2.plot(layers, action_cv, 'r-^', markersize=4, linewidth=2, label='Action (5-fold CV)')
    ax2.axhline(y=metadata['domain_majority'], color='b', linestyle='--', alpha=0.3)
    ax2.axhline(y=metadata['position_majority'], color='g', linestyle='--', alpha=0.3)
    ax2.axhline(y=metadata['action_majority'], color='r', linestyle='--', alpha=0.3)
    ax2.set_xlabel('Transformer Layer', fontsize=13)
    ax2.set_ylabel('Probe Accuracy (5-fold CV)', fontsize=13)
    ax2.set_title('Layer-wise Linear Probing (5-fold CV): Qwen2.5-VL-7B (SFT v2)', fontsize=14)
    ax2.legend(fontsize=11)
    ax2.set_xticks(layers)
    ax2.set_xlim(-0.5, NUM_TRANSFORMER_LAYERS - 0.5)
    ax2.set_ylim(0, 1.0)
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    fig2.savefig(os.path.join(args.output_dir, 'probing_curves_cv.png'), dpi=150)
    print(f"CV plot saved to {os.path.join(args.output_dir, 'probing_curves_cv.png')}")

    # Cross-probe heatmap (5 sampled layers × 9 combinations)
    cross = data.get('cross_probes', {})
    if cross:
        sampled_layers = [0, 6, 13, 20, 27]
        token_types = ['image', 'history', 'last']
        label_types = ['domain', 'position', 'action']

        heatmap = np.zeros((len(sampled_layers) * len(token_types), len(label_types)))
        row_labels = []
        for li, layer_idx in enumerate(sampled_layers):
            key = f'layer_{layer_idx}'
            for ti, tt in enumerate(token_types):
                row_idx = li * len(token_types) + ti
                row_labels.append(f'L{layer_idx} {tt}')
                for ci, lt in enumerate(label_types):
                    probe_key = f'{tt}_→_{lt}'
                    val = cross.get(key, {}).get(probe_key, 0)
                    heatmap[row_idx, ci] = val

        fig3, ax3 = plt.subplots(1, 1, figsize=(8, 12))
        im = ax3.imshow(heatmap, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
        ax3.set_xticks(range(len(label_types)))
        ax3.set_xticklabels(label_types, fontsize=11)
        ax3.set_yticks(range(len(row_labels)))
        ax3.set_yticklabels(row_labels, fontsize=9)
        for i in range(heatmap.shape[0]):
            for j in range(heatmap.shape[1]):
                ax3.text(j, i, f'{heatmap[i,j]:.2f}', ha='center', va='center', fontsize=8)
        ax3.set_title('Cross-probe: Token Type × Label (sampled layers)', fontsize=13)
        fig3.colorbar(im, ax=ax3, shrink=0.6)
        plt.tight_layout()
        fig3.savefig(os.path.join(args.output_dir, 'cross_probe_heatmap.png'), dpi=150)
        print(f"Cross-probe heatmap saved")

    # Print summary table
    print("\n" + "=" * 80)
    print("LAYER-WISE PROBING SUMMARY (Pattern B test accuracy)")
    print("=" * 80)
    print(f"{'Layer':>5} {'Scene':>10} {'Progress':>10} {'Action':>10}")
    print(f"{'-----':>5} {'----------':>10} {'----------':>10} {'----------':>10}")
    for layer_idx in layers:
        print(f"{layer_idx:>5} {scene_acc[layer_idx]:>9.3f} {progress_acc[layer_idx]:>9.3f} "
              f"{action_acc[layer_idx]:>9.3f}")
    print(f"\nMajority baselines: Scene={metadata['domain_majority']:.3f}, "
          f"Progress={metadata['position_majority']:.3f}, "
          f"Action={metadata['action_majority']:.3f}")


# ─── Main ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Layer-wise probing for Qwen2.5-VL-7B on GUI-360")
    parser.add_argument("--mode", type=str, required=True,
                        choices=['extract', 'probe', 'plot'],
                        help="'extract' to extract hidden states, 'probe' to train probes, "
                             "'plot' to generate curves")
    parser.add_argument("--model_path", type=str, default=None,
                        help="Model path (for extract mode)")
    parser.add_argument("--data_root", type=str, default=None,
                        help="GUI-360 test data root (contains data/ and image/)")
    parser.add_argument("--pattern_b_file", type=str, default=None,
                        help="Path to pattern_b_ids.json (for train/test split in probe mode)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for hidden states and results")
    parser.add_argument("--max_trajectories", type=int, default=None,
                        help="Max trajectories to process")
    parser.add_argument("--max_samples", type=int, default=3000,
                        help="Max total samples to extract")
    args = parser.parse_args()

    if args.mode == 'extract':
        if not args.model_path:
            parser.error("--model_path required for extract mode")
        if not args.data_root:
            parser.error("--data_root required for extract mode")
        extract_hidden_states(args)
    elif args.mode == 'probe':
        train_probes(args)
    elif args.mode == 'plot':
        plot_curves(args)
