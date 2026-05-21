"""HF-based AR Trajectory Evaluation for AndroidControl with Cooperative LoRA.

vLLM is incompatible with the cooperative wrapper (forward_pre_hook routing
breaks under vLLM optimizations), so this script uses raw HF generate() via
``cooperative_trajectory_common.cooperative_generate``.

Mirrors the flow of:
  - scripts/eval/ac/eval_a_ar_trajectory.py     (JsonFormat / RAW_SPACE / slim_messages)
  - evaluation/eval_cooperative_batch.py        (HF wrapper loading)
  - gui_odyssey_eval/eval_ar_trajectory.py      (safe parser, episode loop)

Per-shard execution (1 GPU per shard, sequential within shard). Use the
companion slurm script to launch N shards across N GPUs.

Usage:
    python evaluation/eval_cooperative_ac_trajectory.py \
        --base_model checkpoints/Qwen2.5-VL-7B-Instruct \
        --coop_checkpoint train_GUI_360/llamafactory/output/cooperative_v6_5_ac/epoch-4 \
        --jsonl_file datasets/android_control_evaluation_std.jsonl \
        --output_dir outputs/eval_v6_5_ac/epoch4 \
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
from PIL import Image

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from x.data.agent.json import JsonFormat
from x.data.agent.space.std_space import RAW_SPACE
from x.qwen.data_format import slim_messages

from evaluation.qwenvl_utils import (
    evaluate_android_control_action,
    find_last_image_ele,
)
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
from evaluation.corrected_cooperative_model import (
    corrected_generate,
    load_corrected_model,
)


# ── Data loading (matches scripts/eval/ac/ac_utils.py:load_ac_trajectories) ─

def load_ac_trajectories(jsonl_path, image_root, max_episodes=None):
    """Load AndroidControl episodes, fix screenshot paths and check_options."""
    data = []
    with open(jsonl_path, 'r') as f:
        for line in f:
            episode = json.loads(line.strip())
            for step in episode['steps']:
                if step['screenshot'].startswith('/datasets/'):
                    step['screenshot'] = step['screenshot'].replace(
                        '/datasets/', image_root + '/', 1
                    )
                if 'check_options' not in step:
                    check_options = copy.deepcopy(step['action_content'])
                    if 'candidate_bbox' not in check_options and 'bbox' in step:
                        check_options['candidate_bbox'] = step['bbox']
                    elif 'candidate_bbox' not in check_options:
                        check_options['candidate_bbox'] = []
                    step['check_options'] = check_options
            data.append(episode)
            if max_episodes and len(data) >= max_episodes:
                break
    return data


# ── Episode loop ──────────────────────────────────────────────────────

def process_episode(episode, fm, model, processor, device, args, generate_fn=None):
    """Run one AC episode autoregressively against the cooperative wrapper."""
    if generate_fn is None:
        generate_fn = cooperative_generate
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
                model_response = generate_fn(
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

            type_match, extract_match = evaluate_android_control_action(
                pred_action, current_check,
                width, height, resized_width, resized_height,
                ignore_actions=[],
            )
            type_match = bool(type_match)
            extract_match = bool(extract_match)

            step_results.append({
                'step_num': step_id,
                'type_match': type_match,
                'extract_match': extract_match,
                'pred_action': pred_action,
                'gt_action': gt_action,
                'gt_action_type': gt_action.get('action', 'unknown'),
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

    # ── Load model + format ────────────────────────────────────────────
    if args.merged_dir:
        # Lossless corrected model (merge_cooperative_lossless.py output)
        print(f"[shard {args.shard_id}] Loading corrected model from {args.merged_dir}")
        model, processor = load_corrected_model(args.merged_dir, device=device)
        model_device = device
        generate_fn = corrected_generate
    else:
        # Standard cooperative wrapper (lora_v + lora_a + routing)
        model, processor, model_device = load_cooperative_model(
            base_model_path=args.base_model,
            coop_checkpoint_path=args.coop_checkpoint,
            device=device,
        )
        generate_fn = cooperative_generate
    fm = JsonFormat(RAW_SPACE, add_thought=True, force_add_thought=True)

    # ── Output paths (per-shard) ───────────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)
    out_traj = os.path.join(args.output_dir, f'trajectory_results_shard{args.shard_id}.jsonl')
    out_summary = os.path.join(args.output_dir, f'summary_shard{args.shard_id}.json')
    if os.path.exists(out_traj):
        os.remove(out_traj)

    # ── Load and shard data ────────────────────────────────────────────
    episodes = load_ac_trajectories(
        jsonl_path=args.jsonl_file,
        image_root=os.path.join(PROJECT_ROOT, 'datasets'),
        max_episodes=args.max_episodes,
    )
    episodes = shard_episodes(episodes, args.shard_id, args.num_shards)
    print(f"[shard {args.shard_id}] {len(episodes)} episodes "
          f"(of total before sharding, mode={'no_stop' if args.no_stop else 'stop_on_error'})")

    # ── Episode loop ───────────────────────────────────────────────────
    results = []
    t0 = time.time()
    for i, ep in enumerate(episodes):
        result = process_episode(ep, fm, model, processor, model_device, args,
                                 generate_fn=generate_fn)
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

    # ── Per-shard summary ──────────────────────────────────────────────
    metrics = compute_trajectory_metrics(results)
    action_stats = aggregate_action_stats(results)

    length_stats = {}
    for r in results:
        b = r['length_bucket']
        length_stats.setdefault(b, []).append(r)
    length_metrics = {b: compute_trajectory_metrics(v) for b, v in length_stats.items()}

    summary = {
        'shard_id': args.shard_id,
        'num_shards': args.num_shards,
        'base_model': args.base_model,
        'coop_checkpoint': args.coop_checkpoint,
        'mode': 'no_stop' if args.no_stop else 'stop_on_error',
        'n_history_image_limit': args.n_history_image_limit,
        'total_episodes': len(results),
        **metrics,
        'action_type_stats': action_stats,
        'length_bucket_stats': length_metrics,
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
        description='Cooperative LoRA AR trajectory evaluation on AndroidControl (HF generate)'
    )
    parser.add_argument('--base_model', type=str,
                        default='checkpoints/Qwen2.5-VL-7B-Instruct')
    parser.add_argument('--coop_checkpoint', type=str, default=None,
                        help='Path to cooperative LoRA checkpoint directory')
    parser.add_argument('--merged_dir', type=str, default=None,
                        help='Path to lossless merged model (from merge_cooperative_lossless.py). '
                             'If set, --base_model and --coop_checkpoint are ignored.')
    parser.add_argument('--jsonl_file', type=str,
                        default=os.path.join(PROJECT_ROOT,
                                             'datasets',
                                             'android_control_evaluation_std.jsonl'))
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
    args = parser.parse_args()
    main(args)
