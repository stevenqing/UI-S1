"""Q1: Probing Experiment — Does the model's hidden state encode task progress state?

Extracts hidden states from Qwen2.5-VL-7B under oracle subtask condition,
then trains linear probes to predict:
  1. Trajectory position (step_num / total_steps)
  2. Action type of current step
  3. Whether current step is a subgoal boundary
  4. Task completion proportion (steps_correct_so_far / total_steps)

If probes achieve high accuracy → information exists but isn't decoded (Hypothesis B)
If probes are near-random → information not represented (Hypothesis A)
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


def build_oracle_messages(fixed, step_id, action_history, fm_obj):
    """Build oracle subtask messages (same as eval_context_subtask.py)."""
    line_can_thought = fm_obj.can_thought(fixed)
    _format = 'thought_action' if line_can_thought else 'only_action'
    system_prompt = MOBILE_USE.format(OUTPUT_FORMAT[_format], generate_prompt(fm_obj.space))

    messages = [{'role': 'system', 'content': [{'text': system_prompt}]}]

    step = fixed['steps'][step_id]
    step_instruction = step.get('step_instruction', '')

    text_parts = [f"Overall Task: {fixed['goal']}"]
    if step_instruction:
        text_parts.append(f"\nCurrent Step Instruction: {step_instruction}")

    if action_history:
        text_parts.append(f"\nCompleted actions ({len(action_history)} step(s)):")
        for i, action_text in enumerate(action_history):
            text_parts.append(f"  Step {i+1}: {action_text}")
        text_parts.append(f"\nPlease perform step {len(action_history)+1} as instructed above.")
    else:
        text_parts.append(f"\nThis is the first step. Please begin the task.")

    format_instruct = f"Output Format: {OUTPUT_FORMAT[_format]}"
    text_parts.append(f"\n{format_instruct}")

    user_content = [{'text': '\n'.join(text_parts)}]

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
    """Extract hidden states from the model for each step in oracle condition."""
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

    # Extract layers to probe
    probe_layers = [0, 7, 14, 21, 27]  # 28 layers total for 7B, sample 5
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

            # Build oracle messages
            messages = build_oracle_messages(fixed, step_id, action_history, fm)

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

                # Labels
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

            # Update action history with GT (oracle condition)
            action_history.append(format_action_text(gt_action))

        if sample_count >= args.max_samples:
            print(f"Reached max_samples limit ({args.max_samples})")
            break

    print(f"\nTotal samples extracted: {sample_count}")

    # Save
    os.makedirs(args.output_dir, exist_ok=True)

    # Save labels separately (small JSON)
    labels = []
    for s in all_samples:
        labels.append({k: v for k, v in s.items() if k != 'hidden_states'})
    with open(os.path.join(args.output_dir, 'probe_labels.json'), 'w') as f:
        json.dump(labels, f)

    # Save hidden states as numpy arrays (per layer)
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
    """Train linear probes on extracted hidden states."""
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.model_selection import cross_val_score
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.metrics import accuracy_score, r2_score

    print(f"Loading data from {args.output_dir}...")

    with open(os.path.join(args.output_dir, 'probe_labels.json')) as f:
        labels = json.load(f)

    # Prepare label arrays
    positions = np.array([l['position'] for l in labels])
    completions = np.array([l['completion'] for l in labels])
    action_types = [l['action_type'] for l in labels]
    boundaries = np.array([int(l['is_boundary']) for l in labels])

    le = LabelEncoder()
    action_type_encoded = le.fit_transform(action_types)
    print(f"Action types: {le.classes_}")
    print(f"Samples: {len(labels)}")
    print(f"Boundary ratio: {boundaries.mean():.3f}")

    # Random baselines
    print(f"\n--- Random Baselines ---")
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

    # Save results
    with open(os.path.join(args.output_dir, 'probe_results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nProbe results saved to {os.path.join(args.output_dir, 'probe_results.json')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Q1: Hidden State Probing for Task Progress State")
    parser.add_argument("--mode", type=str, required=True, choices=['extract', 'probe'],
                        help="'extract' to extract hidden states, 'probe' to train probes")
    parser.add_argument("--model_path", type=str, default=None, help="Model path (for extract mode)")
    parser.add_argument("--jsonl_file", type=str, default=None, help="Path to evaluation JSONL")
    parser.add_argument("--output_dir", type=str, default="outputs/probe_hidden_states",
                        help="Output directory for hidden states and probe results")
    parser.add_argument("--max_episodes", type=int, default=200, help="Max episodes to process")
    parser.add_argument("--max_steps_per_episode", type=int, default=20, help="Max steps per episode")
    parser.add_argument("--max_samples", type=int, default=2000, help="Max total samples")
    args = parser.parse_args()

    if args.mode == 'extract':
        extract_hidden_states(args)
    elif args.mode == 'probe':
        train_probes(args)
