"""V12 Soft Cooperative LoRA — AR Trajectory Evaluation on AndroidControl.

Reuses the existing eval infrastructure (cooperative_trajectory_common) but
loads model via SoftCooperativeVLMWrapper instead of CooperativeVLMWrapper.

Usage:
    python v12_general/eval_ac_trajectory.py \
        --base_model checkpoints/Qwen2.5-VL-7B-Instruct \
        --coop_checkpoint checkpoints/v12_sft/cooperative_final \
        --output_dir outputs/v12_sft_ep4 \
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

from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from v12_general.soft_cooperative_wrapper import SoftCooperativeVLMWrapper

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
    safe_parse_response,
    shard_episodes,
)


# ── V12 model loading ────────────────────────────────────────────────

def load_v12_model(base_model_path, coop_checkpoint_path, device):
    """Load Qwen2.5-VL + SoftCooperativeVLMWrapper from a V12 checkpoint."""
    print(f"[{device}] Loading base model from {base_model_path}...")
    base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map=device,
    )

    config_path = os.path.join(coop_checkpoint_path, "cooperative_config.json")
    with open(config_path) as f:
        coop_config = json.load(f)

    r = coop_config.get("lora_r", 256)
    alpha = coop_config.get("lora_alpha", r * 2)
    target_modules = coop_config.get("target_modules",
                                     ["q_proj", "k_proj", "v_proj", "o_proj"])

    print(f"[{device}] Wrapping SoftCooperativeVLMWrapper "
          f"(r={r}, alpha={alpha}, targets={target_modules})")

    model = SoftCooperativeVLMWrapper(
        base_model=base_model,
        lora_r=r,
        lora_alpha=alpha,
        lora_dropout=0.0,
        target_modules=target_modules,
        balance_weight=0.0,  # no balance loss during eval
    )
    model.load_cooperative(coop_checkpoint_path, device=device)
    model.eval()

    processor = AutoProcessor.from_pretrained(base_model_path, trust_remote_code=True)
    processor.tokenizer.padding_side = "left"
    return model, processor, base_model.device


# ── Data loading ─────────────────────────────────────────────────────

def load_ac_trajectories(jsonl_path, image_root, max_episodes=None):
    """Load AndroidControl episodes."""
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


# ── Episode loop ─────────────────────────────────────────────────────

def process_episode(episode, fm, model, processor, device, args):
    """Run one AC episode autoregressively."""
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
            except Exception:
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


# ── Main ─────────────────────────────────────────────────────────────

def main(args):
    device = f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu'
    print(f"[shard {args.shard_id}] device={device}")

    model, processor, model_device = load_v12_model(
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

    episodes = load_ac_trajectories(
        jsonl_path=args.jsonl_file,
        image_root=os.path.join(PROJECT_ROOT, 'datasets'),
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='V12 Soft Cooperative LoRA — AR trajectory eval on AndroidControl')
    parser.add_argument('--base_model', type=str,
                        default='checkpoints/Qwen2.5-VL-7B-Instruct')
    parser.add_argument('--coop_checkpoint', type=str, required=True,
                        help='Path to V12 cooperative checkpoint directory')
    parser.add_argument('--jsonl_file', type=str,
                        default=os.path.join(PROJECT_ROOT, 'datasets',
                                             'android_control_evaluation_std.jsonl'))
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--shard_id', type=int, default=0)
    parser.add_argument('--num_shards', type=int, default=1)
    parser.add_argument('--n_history_image_limit', type=int, default=2)
    parser.add_argument('--max_new_tokens', type=int, default=512)
    parser.add_argument('--max_episodes', type=int, default=None)
    parser.add_argument('--no_stop', action='store_true',
                        help='Continue evaluating after errors (no-stop mode)')
    args = parser.parse_args()
    main(args)
