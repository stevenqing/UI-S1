"""Q6: Baseline Probing — Does the model encode action type WITHOUT oracle step_instruction?

Forked from probe_hidden_states.py. Key difference: build_baseline_messages() omits
step_instruction from the user prompt. Everything else (model loading, layer extraction,
probe training) is identical.

Three modes:
  extract     — Forward pass with GT action history but NO step_instruction
  probe       — Train Ridge/LogisticRegression probes on baseline hidden states (5-fold CV)
  cross_apply — Load oracle-trained probes, apply to baseline hidden states
"""

import argparse
import copy
import json
import os
import sys
import torch
import numpy as np
from pathlib import Path

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.dirname(__file__))

from ac_utils import load_ac_trajectories, fix_line, init_format
from x.data.agent.json import MOBILE_USE, OUTPUT_FORMAT, generate_prompt
from x.qwen.image import make_qwen_image_item


def build_baseline_messages(fixed, step_id, action_history, fm_obj):
    """Build baseline messages — same as oracle but WITHOUT step_instruction."""
    line_can_thought = fm_obj.can_thought(fixed)
    _format = 'thought_action' if line_can_thought else 'only_action'
    system_prompt = MOBILE_USE.format(OUTPUT_FORMAT[_format], generate_prompt(fm_obj.space))

    messages = [{'role': 'system', 'content': [{'text': system_prompt}]}]

    # NOTE: No step_instruction lookup — this is the key difference from oracle
    text_parts = [f"Overall Task: {fixed['goal']}"]

    if action_history:
        text_parts.append(f"\nCompleted actions ({len(action_history)} step(s)):")
        for i, action_text in enumerate(action_history):
            text_parts.append(f"  Step {i+1}: {action_text}")
        text_parts.append(f"\nPlease perform step {len(action_history)+1}.")
    else:
        text_parts.append(f"\nThis is the first step. Please begin the task.")

    format_instruct = f"Output Format: {OUTPUT_FORMAT[_format]}"
    text_parts.append(f"\n{format_instruct}")

    user_content = [{'text': '\n'.join(text_parts)}]

    step = fixed['steps'][step_id]
    if step_id == 0:
        user_content.append({
            'text': "If the query asks a question, please answer the question through the answer action before terminating the process.\n"
        })

    image_ele = make_qwen_image_item(
        step['screenshot'],
        image=step.get('screenshot_pil', None)
    )
    user_content.append(image_ele)

    messages.append({'role': 'user', 'content': user_content})
    return messages


def format_action_text(action):
    """Brief text for action history."""
    if action is None:
        return "unknown"
    atype = action.get('action', 'unknown')
    if atype == 'click':
        return f"click at {action.get('coordinate', [])}"
    elif atype == 'type':
        return f"type \"{action.get('text', '')}\""
    elif atype == 'open':
        return f"open \"{action.get('text', action.get('app', ''))}\""
    elif atype == 'swipe':
        return f"swipe {action.get('direction', '')} at {action.get('coordinate', [])}"
    elif atype == 'system_button':
        return f"press {action.get('button', '')}"
    elif atype == 'wait':
        return "wait"
    else:
        return f"{atype}"


def extract_hidden_states(args):
    """Extract hidden states from the model under baseline condition (no step_instruction)."""
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    from qwen_vl_utils import process_vision_info

    print(f"Loading model from {args.model_path}...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path,
        dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="flash_attention_2",
    )
    model.eval()

    processor = AutoProcessor.from_pretrained(args.model_path)

    fm = init_format()

    data = load_ac_trajectories(
        jsonl_path=args.jsonl_file,
        max_episodes=args.max_episodes
    )
    print(f"Loaded {len(data)} episodes.")

    probe_layers = [0, 7, 14, 21, 27]
    num_layers = model.config.num_hidden_layers
    print(f"Model has {num_layers} layers. Probing layers: {probe_layers}")

    all_samples = []
    sample_count = 0

    for ep_idx, episode in enumerate(data):
        fixed = fix_line(copy.deepcopy(episode))
        num_steps = len(fixed['steps'])
        action_history = []

        for step_id in range(min(num_steps, args.max_steps_per_episode)):
            step = fixed['steps'][step_id]
            gt_action = step['action_content']
            gt_type = gt_action['action']

            # Check subgoal boundary
            is_boundary = False
            if step_id > 0:
                prev_type = fixed['steps'][step_id - 1]['action_content']['action']
                is_boundary = (prev_type != gt_type)

            # Build BASELINE messages (no step_instruction)
            messages = build_baseline_messages(fixed, step_id, action_history, fm)

            # Convert to Qwen format for processor
            qwen_messages = []
            for msg in messages:
                role = msg['role']
                content_parts = []
                for item in msg['content']:
                    if 'text' in item:
                        content_parts.append({'type': 'text', 'text': item['text']})
                    elif 'image' in item:
                        content_parts.append({'type': 'image', 'image': item['image']})
                    elif 'image_url' in item:
                        content_parts.append({'type': 'image', 'image': item['image_url']['url']})
                qwen_messages.append({'role': role, 'content': content_parts})

            try:
                text = processor.apply_chat_template(
                    qwen_messages, tokenize=False, add_generation_prompt=True
                )
                image_inputs, video_inputs = process_vision_info(qwen_messages)
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
                    'episode_id': episode.get('episode_id', ep_idx),
                    'step_num': step_id,
                    'num_steps': num_steps,
                    'position': step_id / max(num_steps - 1, 1),
                    'action_type': gt_type,
                    'is_boundary': is_boundary,
                    'completion': step_id / num_steps,
                    'hidden_states': hidden_states,
                }
                all_samples.append(sample)
                sample_count += 1

                if sample_count % 50 == 0:
                    print(f"  Extracted {sample_count} samples from {ep_idx+1}/{len(data)} episodes")

            except Exception as e:
                print(f"  Error at episode {ep_idx} step {step_id}: {e}")
                continue

            # Update action history with GT (same as oracle — only prompt differs)
            action_history.append(format_action_text(gt_action))

        if sample_count >= args.max_samples:
            print(f"Reached max_samples limit ({args.max_samples})")
            break

    print(f"\nTotal samples extracted: {sample_count}")

    # Save
    os.makedirs(args.output_dir, exist_ok=True)

    labels = []
    for s in all_samples:
        labels.append({k: v for k, v in s.items() if k != 'hidden_states'})
    with open(os.path.join(args.output_dir, 'baseline_labels.json'), 'w') as f:
        json.dump(labels, f)

    for layer_key in [f'layer_{i}' for i in probe_layers]:
        states = []
        for s in all_samples:
            if layer_key in s['hidden_states']:
                states.append(s['hidden_states'][layer_key])
        if states:
            arr = np.stack(states)
            np.save(os.path.join(args.output_dir, f'{layer_key}.npy'), arr)
            print(f"Saved {layer_key}: shape {arr.shape}")

    print(f"Labels and hidden states saved to {args.output_dir}")


def train_probes(args):
    """Train linear probes on baseline hidden states (5-fold CV)."""
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.model_selection import cross_val_score
    from sklearn.preprocessing import LabelEncoder, StandardScaler

    print(f"Loading data from {args.output_dir}...")

    with open(os.path.join(args.output_dir, 'baseline_labels.json')) as f:
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
    """Load oracle-trained probes, apply to baseline hidden states.

    Trains probes on oracle data, evaluates on baseline data.
    Reports cross-condition transfer accuracy per layer.
    """
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.metrics import accuracy_score, r2_score
    import pickle

    print("=== Cross-Apply: Oracle probes → Baseline data ===")

    oracle_dir = args.oracle_dir
    baseline_dir = args.output_dir

    # Load oracle labels and baseline labels
    with open(os.path.join(oracle_dir, 'probe_labels.json')) as f:
        oracle_labels = json.load(f)
    with open(os.path.join(baseline_dir, 'baseline_labels.json')) as f:
        baseline_labels = json.load(f)

    print(f"Oracle samples: {len(oracle_labels)}, Baseline samples: {len(baseline_labels)}")

    # Prepare oracle label arrays
    oracle_action_types = [l['action_type'] for l in oracle_labels]
    oracle_boundaries = np.array([int(l['is_boundary']) for l in oracle_labels])
    oracle_positions = np.array([l['position'] for l in oracle_labels])
    oracle_completions = np.array([l['completion'] for l in oracle_labels])

    # Prepare baseline label arrays
    baseline_action_types = [l['action_type'] for l in baseline_labels]
    baseline_boundaries = np.array([int(l['is_boundary']) for l in baseline_labels])
    baseline_positions = np.array([l['position'] for l in baseline_labels])
    baseline_completions = np.array([l['completion'] for l in baseline_labels])

    # Fit label encoder on union of action types
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

        # Fit scaler on oracle, transform both
        scaler = StandardScaler()
        X_oracle_scaled = scaler.fit_transform(X_oracle)
        X_baseline_scaled = scaler.transform(X_baseline)

        layer_results = {}

        # Action Type: train on oracle, test on baseline
        lr = LogisticRegression(max_iter=1000, C=1.0)
        lr.fit(X_oracle_scaled, oracle_at_enc)
        baseline_pred = lr.predict(X_baseline_scaled)
        cross_acc = accuracy_score(baseline_at_enc, baseline_pred)
        # Also get oracle self-accuracy for reference
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
    parser = argparse.ArgumentParser(description="Q6: Baseline Hidden State Probing (no step_instruction)")
    parser.add_argument("--mode", type=str, required=True,
                        choices=['extract', 'probe', 'cross_apply'],
                        help="'extract' hidden states, 'probe' train probes, 'cross_apply' oracle→baseline")
    parser.add_argument("--model_path", type=str, default=None, help="Model path (for extract mode)")
    parser.add_argument("--jsonl_file", type=str, default=None, help="Path to evaluation JSONL")
    parser.add_argument("--output_dir", type=str, default="outputs/probe_hidden_states_baseline",
                        help="Output directory for baseline hidden states and probe results")
    parser.add_argument("--oracle_dir", type=str, default="outputs/probe_hidden_states",
                        help="Oracle probe output directory (for cross_apply mode)")
    parser.add_argument("--max_episodes", type=int, default=200, help="Max episodes to process")
    parser.add_argument("--max_steps_per_episode", type=int, default=20, help="Max steps per episode")
    parser.add_argument("--max_samples", type=int, default=2000, help="Max total samples")
    args = parser.parse_args()

    if args.mode == 'extract':
        extract_hidden_states(args)
    elif args.mode == 'probe':
        train_probes(args)
    elif args.mode == 'cross_apply':
        cross_apply_probes(args)
