"""HF-based AR Trajectory Evaluation for GUI-Odyssey with Cooperative LoRA.

vLLM is incompatible with the cooperative wrapper (forward_pre_hook routing
breaks under vLLM optimizations), so this script uses raw HF generate() via
``cooperative_trajectory_common.cooperative_generate``.

Mirrors the flow of:
  - gui_odyssey_eval/eval_ar_trajectory.py     (JsonFormat / RAW_SPACE / slim_messages)
  - evaluation/eval_cooperative_batch.py        (HF wrapper loading)
  - evaluation/eval_cooperative_ac_trajectory.py (sister AC script)

Per-shard execution (1 GPU per shard, sequential within shard). Use the
companion slurm script to launch N shards across N GPUs.

Key Odyssey-specific differences vs AC:
  - GT coordinates are in [0,1000] normalized space.
  - Predicted coordinates are in resized pixel space; ``evaluate_odyssey_action``
    converts internally via ``pred_coord_to_1k``.
  - Per-category and per-device breakdowns are added to the per-shard summary.

Usage:
    python evaluation/eval_cooperative_odyssey_trajectory.py \
        --base_model checkpoints/Qwen2.5-VL-7B-Instruct \
        --coop_checkpoint train_GUI_360/llamafactory/output/cooperative_v6_5_odyssey/epoch-4 \
        --jsonl_file datasets/GUI-Odyssey/gui_odyssey_random_split_test.jsonl \
        --output_dir outputs/eval_v6_5_odyssey/epoch4 \
        --gpu_id 0 --shard_id 0 --num_shards 4 \
        --no_stop
"""

import argparse
import copy
import json
import os
import sys
import time
import traceback

import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
# gui_odyssey_eval is needed for odyssey_action_matching imports
GUI_ODYSSEY_DIR = os.path.join(PROJECT_ROOT, 'gui_odyssey_eval')
if GUI_ODYSSEY_DIR not in sys.path:
    sys.path.insert(0, GUI_ODYSSEY_DIR)

from x.data.agent.json import JsonFormat
from x.data.agent.space.std_space import RAW_SPACE
from x.qwen.data_format import slim_messages

from evaluation.qwenvl_utils import find_last_image_ele
from evaluation.cooperative_trajectory_common import (
    _json_default,
    aggregate_action_stats,
    compute_trajectory_metrics,
    cooperative_generate,
    length_bucket,
    load_cooperative_model,
    safe_parse_response,
    shard_episodes,
)
from odyssey_action_matching import evaluate_odyssey_action, pred_coord_to_1k


# ── Data loading ──────────────────────────────────────────────────────

def load_odyssey_trajectories(jsonl_path, max_episodes=None):
    """Load GUI-Odyssey episodes (output of convert_to_eval_format.py)."""
    data = []
    with open(jsonl_path) as f:
        for line in f:
            episode = json.loads(line.strip())
            data.append(episode)
            if max_episodes and len(data) >= max_episodes:
                break
    return data


# ── Episode loop ──────────────────────────────────────────────────────

def process_episode(episode, fm, model, processor, device, args):
    """Run one Odyssey episode autoregressively against the cooperative wrapper."""
    ep = copy.deepcopy(episode)
    num_steps = len(ep['steps'])
    state = None
    model_response = None
    step_results = []

    try:
        for step_id in range(num_steps):
            current_check = ep['steps'][step_id]['check_options']
            gt_action = ep['steps'][step_id]['action_content']

            state = fm.gen_next_round(ep, state, previous_model_response=model_response)
            if state is None:
                break

            messages = slim_messages(
                messages=state['messages'],
                num_image_limit=args.n_history_image_limit,
            )

            _, width, height, resized_width, resized_height = find_last_image_ele(messages)

            try:
                model_response = cooperative_generate(
                    model=model,
                    processor=processor,
                    messages=messages,
                    device=device,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                )
            except Exception as gen_e:
                print(f"  [shard {args.shard_id}] generate error ep={ep.get('episode_id','?')} "
                      f"step={step_id}: {gen_e}")
                model_response = ''

            try:
                pred = safe_parse_response(fm, model_response)
                pred_action = pred['action_content']
            except Exception as parse_e:
                print(f"  [shard {args.shard_id}] parse error ep={ep.get('episode_id','?')} "
                      f"step={step_id}: {parse_e}")
                pred_action = {'action': 'wait', 'time': 0.1}

            type_match, extract_match = evaluate_odyssey_action(
                pred_action, current_check,
                resized_width, resized_height,
            )
            type_match = bool(type_match)
            extract_match = bool(extract_match)

            # Capture pred coordinate in [0,1000] for downstream analysis
            p_coord_1k = None
            pred_coord_raw = pred_action.get('coordinate') if isinstance(pred_action, dict) else None
            if (pred_coord_raw and isinstance(pred_coord_raw, (list, tuple))
                    and len(pred_coord_raw) >= 2):
                try:
                    p_coord_1k = pred_coord_to_1k(
                        [float(pred_coord_raw[0]), float(pred_coord_raw[1])],
                        resized_width, resized_height,
                    )
                except (ValueError, TypeError):
                    pass
            gt_coord_1k = current_check.get('coordinate')  # already in [0,1000]

            step_results.append({
                'step_num': step_id,
                'type_match': type_match,
                'extract_match': extract_match,
                'pred_action': pred_action,
                'gt_action': gt_action,
                'gt_action_type': gt_action.get('action', 'unknown'),
                'resized_width': resized_width,
                'resized_height': resized_height,
                'pred_coord_1k': p_coord_1k,
                'gt_coord_1k': gt_coord_1k,
                'model_response': model_response,
            })

            if not extract_match and not args.no_stop:
                break

    except Exception as e:
        print(f"[shard {args.shard_id}] Error episode {episode.get('episode_id', '?')}: {e}")
        traceback.print_exc()

    correct_steps = sum(1 for s in step_results if s['extract_match'])
    if args.no_stop:
        final_step_id = correct_steps
    else:
        final_step_id = 0
        for s in step_results:
            if s['extract_match']:
                final_step_id += 1
            else:
                break
    task_success = (correct_steps == num_steps and len(step_results) == num_steps)

    return {
        'episode_id': episode.get('episode_id', None),
        'goal': episode['goal'],
        'category': episode.get('category', ''),
        'device_name': episode.get('device_name', ''),
        'num_steps': num_steps,
        'task_success': task_success,
        'final_step_id': final_step_id,
        'correct_steps': correct_steps,
        'evaluated_steps': len(step_results),
        'step_results': step_results,
        'length_bucket': length_bucket(num_steps),
    }


# ── Main ──────────────────────────────────────────────────────────────

def main(args):
    device = f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu'
    print(f"[shard {args.shard_id}] device={device}")

    model, processor, model_device = load_cooperative_model(
        base_model_path=args.base_model,
        coop_checkpoint_path=args.coop_checkpoint,
        device=device,
    )
    fm = JsonFormat(RAW_SPACE, add_thought=True, force_add_thought=True)

    os.makedirs(args.output_dir, exist_ok=True)
    out_traj = os.path.join(args.output_dir, f'trajectory_results_shard{args.shard_id}.jsonl')
    out_summary = os.path.join(args.output_dir, f'summary_shard{args.shard_id}.json')
    if os.path.exists(out_traj):
        os.remove(out_traj)

    episodes = load_odyssey_trajectories(
        jsonl_path=args.jsonl_file,
        max_episodes=args.max_episodes,
    )
    episodes = shard_episodes(episodes, args.shard_id, args.num_shards)
    print(f"[shard {args.shard_id}] {len(episodes)} episodes "
          f"(mode={'no_stop' if args.no_stop else 'stop_on_error'})")

    results = []
    t0 = time.time()
    for i, ep in enumerate(episodes):
        result = process_episode(ep, fm, model, processor, model_device, args)
        results.append(result)

        with open(out_traj, 'a') as f:
            f.write(json.dumps(result, ensure_ascii=False, default=_json_default) + '\n')

        if (i + 1) % 10 == 0 or (i + 1) == len(episodes):
            metrics = compute_trajectory_metrics(results)
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed if elapsed > 0 else 0.0
            print(f"[shard {args.shard_id}] {i+1}/{len(episodes)} "
                  f"TSR={metrics['tsr']:.4f} AvgProg={metrics['avg_progress']:.4f} "
                  f"Scattered={metrics['scattered_progress']:.4f} "
                  f"({rate:.2f} ep/s)")

    metrics = compute_trajectory_metrics(results)
    action_stats = aggregate_action_stats(results)

    length_stats = {}
    for r in results:
        length_stats.setdefault(r['length_bucket'], []).append(r)
    length_metrics = {b: compute_trajectory_metrics(v) for b, v in length_stats.items()}

    category_stats = {}
    for r in results:
        category_stats.setdefault(r.get('category', 'unknown'), []).append(r)
    category_metrics = {c: compute_trajectory_metrics(v) for c, v in category_stats.items()}

    device_stats = {}
    for r in results:
        device_stats.setdefault(r.get('device_name', 'unknown'), []).append(r)
    device_metrics = {d: compute_trajectory_metrics(v) for d, v in device_stats.items()}

    summary = {
        'shard_id': args.shard_id,
        'num_shards': args.num_shards,
        'base_model': args.base_model,
        'coop_checkpoint': args.coop_checkpoint,
        'mode': 'no_stop' if args.no_stop else 'stop_on_error',
        'split_name': args.split_name,
        'n_history_image_limit': args.n_history_image_limit,
        'total_episodes': len(results),
        **metrics,
        'action_type_stats': action_stats,
        'length_bucket_stats': length_metrics,
        'category_stats': category_metrics,
        'device_stats': device_metrics,
    }
    with open(out_summary, 'w') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=_json_default)

    print(f"\n[shard {args.shard_id}] DONE.")
    print(f"  TSR        = {metrics['tsr']:.4f} ({metrics['success_count']}/{metrics['n']})")
    print(f"  AvgProg    = {metrics['avg_progress']:.4f}")
    print(f"  Scattered  = {metrics['scattered_progress']:.4f}")
    print(f"  results -> {out_traj}")
    print(f"  summary -> {out_summary}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Cooperative LoRA AR trajectory evaluation on GUI-Odyssey (HF generate)'
    )
    parser.add_argument('--base_model', type=str,
                        default='checkpoints/Qwen2.5-VL-7B-Instruct')
    parser.add_argument('--coop_checkpoint', type=str, required=True,
                        help='Path to cooperative LoRA checkpoint directory')
    parser.add_argument('--jsonl_file', type=str, required=True,
                        help='Path to GUI-Odyssey eval JSONL '
                             '(from convert_to_eval_format.py)')
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--shard_id', type=int, default=0)
    parser.add_argument('--num_shards', type=int, default=1)
    parser.add_argument('--n_history_image_limit', type=int, default=2)
    parser.add_argument('--max_new_tokens', type=int, default=512)
    parser.add_argument('--max_episodes', type=int, default=None,
                        help='Limit episodes for testing (applied before sharding)')
    parser.add_argument('--no_stop', action='store_true',
                        help='Continue evaluating after errors (no-stop mode)')
    parser.add_argument('--split_name', type=str, default='random_split',
                        help='Split name for summary metadata')
    args = parser.parse_args()
    main(args)
