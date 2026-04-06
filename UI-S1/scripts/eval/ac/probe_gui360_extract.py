"""Q1/Q6: Hidden State Probing for GUI-360

Adapted from probe_hidden_states.py and probe_baseline_extract.py for GUI-360.
Uses GUI-360 prompt templates (ACTION_PREDICTION_SYS_PROMPT_GPT + ACTION_PREDICTION_USER_PROMPT_QWEN).

Four modes:
  extract_oracle   — Forward pass with step 'thought' as instruction, GT action history
  extract_baseline — Forward pass with episode 'goal' as instruction, GT action history
  probe            — Train Ridge/LogisticRegression probes (5-fold CV)
  cross_apply      — Load oracle probes, apply to baseline hidden states

Domain detection: from execution_id prefix (excel_*, word_*, ppt_*)
Action history format: compressed action descriptions matching GUI-360 eval pipeline
"""

import argparse
import json
import os
import sys
import torch
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'train_GUI_360', 'GUI-360-eval'))

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


def detect_domain(execution_id):
    """Detect domain from execution_id prefix."""
    eid_lower = execution_id.lower()
    if eid_lower.startswith('excel'):
        return 'excel'
    elif eid_lower.startswith('word'):
        return 'word'
    elif eid_lower.startswith('ppt') or eid_lower.startswith('powerpoint'):
        return 'ppt'
    return 'word'  # default fallback


def format_action_brief(action_content):
    """Create a brief text summary of a GT action (matching GUI-360 eval pipeline style)."""
    atype = action_content.get('action', 'unknown')
    if not atype:
        return "unknown action"

    if atype == 'click':
        coord = action_content.get('coordinate', [])
        button = action_content.get('button', 'left')
        return f"click(coordinate={coord}, button='{button}')"
    elif atype == 'type':
        keys = action_content.get('text', action_content.get('keys', ''))
        coord = action_content.get('coordinate', [])
        return f"type(coordinate={coord}, keys='{keys}')"
    elif atype == 'drag':
        start = action_content.get('coordinate', action_content.get('start_coordinate', []))
        end = action_content.get('coordinate2', action_content.get('end_coordinate', []))
        return f"drag(start_coordinate={start}, end_coordinate={end})"
    elif atype == 'wheel_mouse_input':
        coord = action_content.get('coordinate', [])
        dist = action_content.get('wheel_dist', action_content.get('time', 0))
        return f"wheel_mouse_input(coordinate={coord}, wheel_dist={dist})"
    elif atype == 'select_text':
        text = action_content.get('text', '')
        return f"select_text(text='{text}')"
    elif atype == 'select_table_range':
        return f"select_table_range(...)"
    elif atype == 'select_paragraph':
        return f"select_paragraph(...)"
    elif atype == 'insert_table':
        return f"insert_table(...)"
    elif atype == 'insert_excel_table':
        return f"insert_excel_table(...)"
    elif atype == 'set_cell_value':
        return f"set_cell_value(...)"
    elif atype == 'table2markdown':
        return f"table2markdown(...)"
    elif atype == 'auto_fill':
        return f"auto_fill(...)"
    elif atype == 'reorder_columns':
        return f"reorder_columns(...)"
    elif atype == 'save_as':
        return f"save_as(...)"
    elif atype == 'set_font':
        return f"set_font(...)"
    elif atype == 'set_background_color':
        return f"set_background_color(...)"
    else:
        return f"{atype}(...)"


def build_gui360_messages(episode, step_idx, action_history, domain, use_oracle=False):
    """Build GUI-360 format messages for probing.

    Args:
        episode: dict with 'goal', 'steps', 'execution_id'
        step_idx: current step index (0-based)
        action_history: list of action brief strings
        domain: 'excel'/'word'/'ppt'
        use_oracle: if True, use step['thought'] as instruction; else use episode['goal']
    """
    step = episode['steps'][step_idx]

    # Instruction
    if use_oracle:
        instruction = step.get('thought', episode['goal'])
        if not instruction or not instruction.strip():
            instruction = episode['goal']
    else:
        instruction = episode['goal']

    # History
    if action_history:
        history_str = "\n".join(
            f"Step {i+1}: {a}" for i, a in enumerate(action_history)
        )
    else:
        history_str = "No previous actions."

    # Supported actions by domain
    actions_str = DOMAIN_ACTIONS.get(domain, SUPPORTED_ACTIONS_WORD)

    # System message
    system_content = ACTION_PREDICTION_SYS_PROMPT_GPT.strip()

    # User message (using Qwen template)
    user_text = ACTION_PREDICTION_USER_PROMPT_QWEN.format(
        instruction=instruction,
        history=history_str,
        actions=actions_str,
    )

    # Build Qwen-format messages
    messages = [
        {
            'role': 'system',
            'content': [{'type': 'text', 'text': system_content}],
        },
        {
            'role': 'user',
            'content': [
                {'type': 'text', 'text': user_text},
                {'type': 'image', 'image': step['screenshot']},
            ],
        },
    ]

    return messages


def load_gui360_trajectories(jsonl_path, max_episodes=200):
    """Load GUI-360 test trajectories."""
    episodes = []
    with open(jsonl_path) as f:
        for i, line in enumerate(f):
            if i >= max_episodes:
                break
            ep = json.loads(line.strip())
            episodes.append(ep)
    return episodes


def extract_hidden_states(args, use_oracle=True):
    """Extract hidden states from GUI-360 episodes."""
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    from qwen_vl_utils import process_vision_info

    mode_name = 'oracle' if use_oracle else 'baseline'
    print(f"=== Extract Mode: {mode_name} ===")
    print(f"Loading model from {args.model_path}...")

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path,
        dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="flash_attention_2",
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(args.model_path)

    data = load_gui360_trajectories(args.jsonl_file, args.max_episodes)
    print(f"Loaded {len(data)} episodes.")

    probe_layers = [0, 7, 14, 21, 27]
    num_layers = model.config.num_hidden_layers
    print(f"Model has {num_layers} layers. Probing layers: {probe_layers}")

    all_samples = []
    sample_count = 0

    for ep_idx, episode in enumerate(data):
        eid = episode.get('execution_id', str(ep_idx))
        num_steps = len(episode['steps'])
        domain = detect_domain(eid)
        action_history = []

        for step_idx in range(min(num_steps, args.max_steps_per_episode)):
            step = episode['steps'][step_idx]
            gt_action = step['action_content']
            gt_type = gt_action.get('action', '')

            if not gt_type:
                action_history.append(format_action_brief(gt_action))
                continue

            # Boundary detection
            is_boundary = False
            if step_idx > 0:
                prev_type = episode['steps'][step_idx - 1]['action_content'].get('action', '')
                if prev_type:
                    is_boundary = (prev_type != gt_type)

            # Build messages
            messages = build_gui360_messages(
                episode, step_idx, action_history, domain, use_oracle=use_oracle
            )

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

                # Extract last token hidden state from probe layers
                hidden_states = {}
                for layer_idx in probe_layers:
                    if layer_idx < len(outputs.hidden_states):
                        hs = outputs.hidden_states[layer_idx][0, -1, :].cpu().float().numpy()
                        hidden_states[f'layer_{layer_idx}'] = hs

                sample = {
                    'episode_id': eid,
                    'step_num': step_idx,
                    'num_steps': num_steps,
                    'position': step_idx / max(num_steps - 1, 1),
                    'action_type': gt_type,
                    'is_boundary': is_boundary,
                    'completion': step_idx / num_steps,
                    'domain': domain,
                    'hidden_states': hidden_states,
                }
                all_samples.append(sample)
                sample_count += 1

                if sample_count % 50 == 0:
                    print(f"  Extracted {sample_count} samples from {ep_idx+1}/{len(data)} episodes")

            except Exception as e:
                print(f"  Error at episode {ep_idx} step {step_idx}: {e}")
                continue

            # Update action history with GT
            action_history.append(format_action_brief(gt_action))

        if sample_count >= args.max_samples:
            print(f"Reached max_samples limit ({args.max_samples})")
            break

    print(f"\nTotal samples extracted: {sample_count}")

    # Save
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # Labels JSON
    labels_filename = 'probe_labels.json' if use_oracle else 'baseline_labels.json'
    labels = []
    for s in all_samples:
        labels.append({k: v for k, v in s.items() if k != 'hidden_states'})
    with open(os.path.join(output_dir, labels_filename), 'w') as f:
        json.dump(labels, f)

    # Hidden states as per-layer .npy
    for layer_key in [f'layer_{i}' for i in probe_layers]:
        states = []
        for s in all_samples:
            if layer_key in s['hidden_states']:
                states.append(s['hidden_states'][layer_key])
        if states:
            arr = np.stack(states)
            np.save(os.path.join(output_dir, f'{layer_key}.npy'), arr)
            print(f"Saved {layer_key}: shape {arr.shape}")

    print(f"Labels and hidden states saved to {output_dir}")
    return model, processor  # return for reuse if extracting both modes


def train_probes(args):
    """Train linear probes on extracted hidden states (5-fold CV)."""
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.model_selection import cross_val_score
    from sklearn.preprocessing import LabelEncoder, StandardScaler

    print(f"Loading data from {args.output_dir}...")

    # Try oracle labels first, then baseline
    labels_path = os.path.join(args.output_dir, 'probe_labels.json')
    if not os.path.exists(labels_path):
        labels_path = os.path.join(args.output_dir, 'baseline_labels.json')
    with open(labels_path) as f:
        labels = json.load(f)

    positions = np.array([l['position'] for l in labels])
    completions = np.array([l['completion'] for l in labels])
    action_types = [l['action_type'] for l in labels]
    boundaries = np.array([int(l['is_boundary']) for l in labels])

    le = LabelEncoder()
    action_type_encoded = le.fit_transform(action_types)
    print(f"Action types: {le.classes_}")
    print(f"Samples: {len(labels)}")
    print(f"Boundary ratio: {boundaries.mean():.3f}")

    action_majority = max(np.bincount(action_type_encoded)) / len(action_type_encoded)
    boundary_majority = max(boundaries.mean(), 1 - boundaries.mean())
    print(f"  Action type majority class: {action_majority:.3f}")
    print(f"  Boundary majority class: {boundary_majority:.3f}")

    probe_layers = [0, 7, 14, 21, 27]
    results = {}

    for layer_idx in probe_layers:
        layer_key = f'layer_{layer_idx}'
        fpath = os.path.join(args.output_dir, f'{layer_key}.npy')
        if not os.path.exists(fpath):
            print(f"Skipping {layer_key} (not found)")
            continue

        X = np.load(fpath)
        print(f"\n{'='*50}")
        print(f"Layer {layer_idx} (shape: {X.shape})")
        print(f"{'='*50}")

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        layer_results = {}

        # Probe 1: Position (regression)
        ridge = Ridge(alpha=1.0)
        scores = cross_val_score(ridge, X_scaled, positions, cv=5, scoring='r2')
        print(f"  Position R²: {scores.mean():.4f} ± {scores.std():.4f}")
        layer_results['position_r2'] = float(scores.mean())

        # Probe 2: Action Type (classification)
        lr = LogisticRegression(max_iter=1000, C=1.0)
        scores = cross_val_score(lr, X_scaled, action_type_encoded, cv=5, scoring='accuracy')
        print(f"  Action Type Acc: {scores.mean():.4f} ± {scores.std():.4f} (majority: {action_majority:.3f})")
        layer_results['action_type_acc'] = float(scores.mean())
        layer_results['action_type_majority'] = float(action_majority)

        # Probe 3: Boundary (classification)
        lr_b = LogisticRegression(max_iter=1000, C=1.0)
        scores = cross_val_score(lr_b, X_scaled, boundaries, cv=5, scoring='accuracy')
        print(f"  Boundary Acc: {scores.mean():.4f} ± {scores.std():.4f} (majority: {boundary_majority:.3f})")
        layer_results['boundary_acc'] = float(scores.mean())
        layer_results['boundary_majority'] = float(boundary_majority)

        # Probe 4: Completion (regression)
        ridge_c = Ridge(alpha=1.0)
        scores = cross_val_score(ridge_c, X_scaled, completions, cv=5, scoring='r2')
        print(f"  Completion R²: {scores.mean():.4f} ± {scores.std():.4f}")
        layer_results['completion_r2'] = float(scores.mean())

        results[layer_key] = layer_results

    with open(os.path.join(args.output_dir, 'probe_results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nProbe results saved to {os.path.join(args.output_dir, 'probe_results.json')}")


def cross_apply_probes(args):
    """Load oracle-trained probes, apply to baseline hidden states."""
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.metrics import accuracy_score, r2_score

    print("=== Cross-Apply: Oracle probes → Baseline data ===")

    oracle_dir = args.oracle_dir
    baseline_dir = args.output_dir

    with open(os.path.join(oracle_dir, 'probe_labels.json')) as f:
        oracle_labels = json.load(f)
    with open(os.path.join(baseline_dir, 'baseline_labels.json')) as f:
        baseline_labels = json.load(f)

    print(f"Oracle samples: {len(oracle_labels)}, Baseline samples: {len(baseline_labels)}")

    # Prepare label arrays
    oracle_action_types = [l['action_type'] for l in oracle_labels]
    oracle_boundaries = np.array([int(l['is_boundary']) for l in oracle_labels])
    oracle_positions = np.array([l['position'] for l in oracle_labels])
    oracle_completions = np.array([l['completion'] for l in oracle_labels])

    baseline_action_types = [l['action_type'] for l in baseline_labels]
    baseline_boundaries = np.array([int(l['is_boundary']) for l in baseline_labels])
    baseline_positions = np.array([l['position'] for l in baseline_labels])
    baseline_completions = np.array([l['completion'] for l in baseline_labels])

    # Fit label encoder on union
    all_types = list(set(oracle_action_types + baseline_action_types))
    le = LabelEncoder()
    le.fit(all_types)
    oracle_at_enc = le.transform(oracle_action_types)
    baseline_at_enc = le.transform(baseline_action_types)

    probe_layers = [0, 7, 14, 21, 27]
    results = {}

    for layer_idx in probe_layers:
        layer_key = f'layer_{layer_idx}'
        oracle_path = os.path.join(oracle_dir, f'{layer_key}.npy')
        baseline_path = os.path.join(baseline_dir, f'{layer_key}.npy')

        if not os.path.exists(oracle_path) or not os.path.exists(baseline_path):
            print(f"Skipping {layer_key} (missing data)")
            continue

        X_oracle = np.load(oracle_path)
        X_baseline = np.load(baseline_path)

        print(f"\n{'='*50}")
        print(f"Layer {layer_idx}: oracle {X_oracle.shape} → baseline {X_baseline.shape}")
        print(f"{'='*50}")

        scaler = StandardScaler()
        X_oracle_scaled = scaler.fit_transform(X_oracle)
        X_baseline_scaled = scaler.transform(X_baseline)

        layer_results = {}

        # Action Type: train on oracle, test on baseline
        lr = LogisticRegression(max_iter=1000, C=1.0)
        lr.fit(X_oracle_scaled, oracle_at_enc)
        baseline_pred = lr.predict(X_baseline_scaled)
        cross_acc = accuracy_score(baseline_at_enc, baseline_pred)
        oracle_self_pred = lr.predict(X_oracle_scaled)
        oracle_self_acc = accuracy_score(oracle_at_enc, oracle_self_pred)
        print(f"  Action Type: oracle_self={oracle_self_acc:.4f}, cross_apply={cross_acc:.4f}")
        layer_results['action_type_oracle_self'] = float(oracle_self_acc)
        layer_results['action_type_cross_apply'] = float(cross_acc)

        # Boundary: train on oracle, test on baseline
        lr_b = LogisticRegression(max_iter=1000, C=1.0)
        lr_b.fit(X_oracle_scaled, oracle_boundaries)
        baseline_pred_b = lr_b.predict(X_baseline_scaled)
        cross_acc_b = accuracy_score(baseline_boundaries, baseline_pred_b)
        oracle_self_b = accuracy_score(oracle_boundaries, lr_b.predict(X_oracle_scaled))
        print(f"  Boundary: oracle_self={oracle_self_b:.4f}, cross_apply={cross_acc_b:.4f}")
        layer_results['boundary_oracle_self'] = float(oracle_self_b)
        layer_results['boundary_cross_apply'] = float(cross_acc_b)

        # Position: train on oracle, test on baseline
        ridge = Ridge(alpha=1.0)
        ridge.fit(X_oracle_scaled, oracle_positions)
        baseline_pred_p = ridge.predict(X_baseline_scaled)
        cross_r2_p = r2_score(baseline_positions, baseline_pred_p)
        oracle_self_r2_p = r2_score(oracle_positions, ridge.predict(X_oracle_scaled))
        print(f"  Position R²: oracle_self={oracle_self_r2_p:.4f}, cross_apply={cross_r2_p:.4f}")
        layer_results['position_oracle_self_r2'] = float(oracle_self_r2_p)
        layer_results['position_cross_apply_r2'] = float(cross_r2_p)

        # Completion: train on oracle, test on baseline
        ridge_c = Ridge(alpha=1.0)
        ridge_c.fit(X_oracle_scaled, oracle_completions)
        baseline_pred_c = ridge_c.predict(X_baseline_scaled)
        cross_r2_c = r2_score(baseline_completions, baseline_pred_c)
        oracle_self_r2_c = r2_score(oracle_completions, ridge_c.predict(X_oracle_scaled))
        print(f"  Completion R²: oracle_self={oracle_self_r2_c:.4f}, cross_apply={cross_r2_c:.4f}")
        layer_results['completion_oracle_self_r2'] = float(oracle_self_r2_c)
        layer_results['completion_cross_apply_r2'] = float(cross_r2_c)

        results[layer_key] = layer_results

    with open(os.path.join(baseline_dir, 'cross_apply_results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nCross-apply results saved to {os.path.join(baseline_dir, 'cross_apply_results.json')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GUI-360 Hidden State Probing (Q1/Q6)")
    parser.add_argument("--mode", type=str, required=True,
                        choices=['extract_oracle', 'extract_baseline', 'probe', 'cross_apply'],
                        help="Extraction or analysis mode")
    parser.add_argument("--model_path", type=str, default=None, help="Model path (for extract modes)")
    parser.add_argument("--jsonl_file", type=str, default=None,
                        help="Path to gui360 trajectory JSONL (train or test)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for hidden states and probe results")
    parser.add_argument("--oracle_dir", type=str, default=None,
                        help="Oracle probe output directory (for cross_apply mode)")
    parser.add_argument("--max_episodes", type=int, default=200, help="Max episodes to process")
    parser.add_argument("--max_steps_per_episode", type=int, default=20, help="Max steps per episode")
    parser.add_argument("--max_samples", type=int, default=2000, help="Max total samples")
    args = parser.parse_args()

    if args.mode == 'extract_oracle':
        extract_hidden_states(args, use_oracle=True)
    elif args.mode == 'extract_baseline':
        extract_hidden_states(args, use_oracle=False)
    elif args.mode == 'probe':
        train_probes(args)
    elif args.mode == 'cross_apply':
        cross_apply_probes(args)
