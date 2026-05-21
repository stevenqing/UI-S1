"""
Base Model Type Distribution — check if base Qwen2.5-VL already has 100% click bias
or if our RL training collapsed diversity.

Runs the BASE model (no cooperative weights) on all 968 test episodes.
"""

import json
import os
import re
import sys
import argparse
from collections import defaultdict

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
import numpy as np


SYSTEM_PROMPT = (
    "You are a GUI agent. You are given a screenshot and a task instruction. "
    "Perform the next action to complete the task.\n\n"
    "Action space:\n"
    '  click: {"action": "click", "coordinate": [x, y]}\n'
    '  type: {"action": "type", "text": "content"}\n'
    '  drag: {"action": "drag", "coordinate": [x1, y1], "endCoordinate": [x2, y2]}\n'
    '  terminate: {"action": "terminate", "status": "success|failure"}\n\n'
    "Output format: <action>{JSON action}</action>"
)


def extract_action(text):
    atype = 'unknown'
    coord = None
    match = re.search(r'<action>\s*\{.*?"action"\s*:\s*"([^"]+)"', text, re.DOTALL)
    if match:
        a = match.group(1).lower()
        if 'click' in a:
            atype = 'click'
        elif 'type' in a:
            atype = 'type'
        elif 'swipe' in a or 'drag' in a:
            atype = 'swipe'
        else:
            atype = a
    coord_match = re.search(r'"coordinate"\s*:\s*\[(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\]', text)
    if coord_match:
        coord = [float(coord_match.group(1)), float(coord_match.group(2))]
    return atype, coord


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", required=True)
    parser.add_argument("--test_data", required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--shard_id", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    device = f"cuda:{args.gpu}"

    with open(args.test_data) as f:
        all_episodes = [json.loads(l) for l in f]
    n_per_shard = len(all_episodes) // args.num_shards
    start = args.shard_id * n_per_shard
    end = start + n_per_shard if args.shard_id < args.num_shards - 1 else len(all_episodes)
    episodes = all_episodes[start:end]
    print(f"Shard {args.shard_id}/{args.num_shards}: {len(episodes)} episodes [{start}:{end}]")

    print("Loading BASE model (no cooperative weights)...")
    processor = AutoProcessor.from_pretrained(args.base_model, trust_remote_code=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16,
        trust_remote_code=True, device_map={"": args.gpu},
    )
    model.eval()
    print("Base model loaded")

    results = []
    for ep_idx, ep in enumerate(episodes):
        goal = ep['goal']
        step = ep['steps'][0]
        screenshot = step['screenshot']

        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        content = []
        if os.path.exists(screenshot):
            content.append({"type": "image", "image": f"file://{screenshot}"})
        content.append({"type": "text", "text": f"Task: {goal}"})
        messages.append({"role": "user", "content": content})

        try:
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            if os.path.exists(screenshot):
                img = Image.open(screenshot).convert("RGB")
                inputs = processor(text=[text], images=[img], return_tensors="pt", padding=True).to(device)
            else:
                inputs = processor(text=[text], return_tensors="pt", padding=True).to(device)

            prompt_len = inputs["input_ids"].shape[1]

            with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
                output_ids = model.generate(**{k: v for k, v in inputs.items()},
                                            max_new_tokens=256, do_sample=False)
                gen_text = processor.tokenizer.decode(output_ids[0, prompt_len:], skip_special_tokens=True)

            atype, coord = extract_action(gen_text)
            results.append({
                'episode_id': ep.get('episode_id', ep_idx),
                'gt_type': step['action']['action'].replace('left_click', 'click'),
                'pred_type': atype,
                'coord': coord,
                'text': gen_text[:200],
            })

        except Exception as e:
            print(f"  Error ep{ep_idx}: {e}")
            results.append({
                'episode_id': ep.get('episode_id', ep_idx),
                'gt_type': step['action']['action'].replace('left_click', 'click'),
                'pred_type': 'error',
                'coord': None,
                'text': str(e)[:100],
            })

        if (ep_idx + 1) % 10 == 0:
            print(f"  [{ep_idx+1}/{len(episodes)}] done")

    # Save
    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, f"shard_{args.shard_id}.json")
    with open(out_path, 'w') as f:
        json.dump(results, f)
    print(f"Saved to {out_path}")


def merge(output_dir):
    """Merge and analyze."""
    import glob as g
    shards = sorted(g.glob(os.path.join(output_dir, "shard_*.json")))
    print(f"Merging {len(shards)} shards")

    all_results = []
    for sf in shards:
        with open(sf) as f:
            all_results.extend(json.load(f))

    print(f"Total: {len(all_results)} episodes")

    from collections import Counter
    pred_types = Counter(r['pred_type'] for r in all_results)
    gt_types = Counter(r['gt_type'] for r in all_results)

    print(f"\nGT distribution:   {dict(gt_types.most_common())}")
    print(f"Pred distribution: {dict(pred_types.most_common())}")

    # Confusion matrix
    print(f"\n{'GT':<8} | {'N':>4} | {'pred_click':>10} | {'pred_type':>10} | {'pred_swipe':>10} | {'pred_other':>10}")
    print("-" * 70)
    for gt in ['click', 'type', 'swipe']:
        eps = [r for r in all_results if r['gt_type'] == gt]
        n = len(eps)
        if n == 0:
            continue
        pc = sum(1 for r in eps if r['pred_type'] == 'click') / n * 100
        pt = sum(1 for r in eps if r['pred_type'] == 'type') / n * 100
        ps = sum(1 for r in eps if r['pred_type'] == 'swipe') / n * 100
        po = 100 - pc - pt - ps
        print(f"  {gt:<6} | {n:>4} | {pc:>9.1f}% | {pt:>9.1f}% | {ps:>9.1f}% | {po:>9.1f}%")

    # Coordinate accuracy for click predictions
    gt_data_path = "/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/v13_gui_360/data/gui360_test_968.jsonl"
    with open(gt_data_path) as f:
        gt_episodes = [json.loads(l) for l in f]
    gt_by_id = {ep['episode_id']: ep['steps'][0]['action'] for ep in gt_episodes}

    dists = []
    for r in all_results:
        if r['pred_type'] == 'click' and r['coord']:
            gt = gt_by_id.get(r['episode_id'], {})
            gt_coord = gt.get('coordinate')
            if gt_coord:
                d = ((r['coord'][0] - gt_coord[0])**2 + (r['coord'][1] - gt_coord[1])**2)**0.5
                dists.append(d)

    if dists:
        print(f"\nCoordinate accuracy (base model, click predictions):")
        print(f"  Mean dist: {np.mean(dists):.1f}px, Median: {np.median(dists):.1f}px")
        print(f"  <50px: {sum(1 for d in dists if d < 50)/len(dists)*100:.1f}%")
        print(f"  <100px: {sum(1 for d in dists if d < 100)/len(dists)*100:.1f}%")

    # Save summary
    summary = {
        'pred_distribution': dict(pred_types),
        'gt_distribution': dict(gt_types),
        'coord_mean_dist': float(np.mean(dists)) if dists else None,
        'coord_lt50': sum(1 for d in dists if d < 50) / len(dists) * 100 if dists else None,
    }
    with open(os.path.join(output_dir, "summary.json"), 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved summary to {output_dir}/summary.json")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == '--merge':
        merge(sys.argv[2])
    else:
        main()
