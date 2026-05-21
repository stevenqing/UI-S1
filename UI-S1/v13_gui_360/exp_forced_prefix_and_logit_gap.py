"""
Combined experiment: Forced-Prefix Decode + Logit Gap Analysis.

Part 1 — Forced Prefix:
  For GT=type/swipe episodes, force the action type prefix and let model complete.
  Tests whether model "can't" or "won't" predict type/swipe.

Part 2 — Logit Gap:
  At the action_type decision point, measure logit difference between
  "left_click" vs "type" vs "swipe" tokens.

Both run on 1 GPU since they're fast (forward-only or partial generate).
"""

import json
import os
import re
import sys
import argparse
from collections import defaultdict

import torch
import torch.nn.functional as F
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


def extract_coordinate(text):
    match = re.search(r'"coordinate"\s*:\s*\[(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\]', text)
    if match:
        return [float(match.group(1)), float(match.group(2))]
    return None


def extract_text_content(text):
    """Extract text content from type action."""
    match = re.search(r'"text"\s*:\s*"([^"]*)"', text)
    if match:
        return match.group(1)
    return None


def extract_direction(text):
    """Extract swipe direction."""
    match = re.search(r'"coordinate"\s*:\s*\[([^\]]+)\].*?"endCoordinate"\s*:\s*\[([^\]]+)\]', text, re.DOTALL)
    if match:
        return match.group(0)
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", required=True)
    parser.add_argument("--coop_checkpoint", required=True)
    parser.add_argument("--test_data", required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--shard_id", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    device = f"cuda:{args.gpu}"

    # Load episodes
    with open(args.test_data) as f:
        all_episodes = [json.loads(l) for l in f]
    n_per_shard = len(all_episodes) // args.num_shards
    start = args.shard_id * n_per_shard
    end = start + n_per_shard if args.shard_id < args.num_shards - 1 else len(all_episodes)
    episodes = all_episodes[start:end]
    print(f"Shard {args.shard_id}/{args.num_shards}: {len(episodes)} episodes [{start}:{end}]")

    # Load model
    print("Loading model...")
    processor = AutoProcessor.from_pretrained(args.base_model, trust_remote_code=True)
    base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16,
        trust_remote_code=True, device_map={"": args.gpu},
    )
    from v13_gui_360.iterative_cooperative_wrapper import IterativeCooperativeVLMWrapper
    config_path = os.path.join(args.coop_checkpoint, "cooperative_config.json")
    with open(config_path) as f:
        coop_config = json.load(f)
    model = IterativeCooperativeVLMWrapper(
        base_model,
        lora_r=coop_config.get("lora_r", 128),
        lora_alpha=coop_config.get("lora_alpha", 256),
        target_modules=coop_config.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"]),
        num_comm_rounds=coop_config.get("num_comm_rounds", 2),
    )
    model.load_cooperative(args.coop_checkpoint)
    model.eval()
    print("Model loaded")

    tokenizer = processor.tokenizer

    # ═══════════════════════════════════════════
    # Part 1: Forced-Prefix Decode
    # ═══════════════════════════════════════════
    print("\n" + "=" * 60)
    print("PART 1: FORCED-PREFIX DECODE")
    print("=" * 60)

    forced_results = []
    for ep_idx, ep in enumerate(episodes):
        goal = ep['goal']
        step = ep['steps'][0]
        screenshot = step['screenshot']
        gt_action = step['action']
        gt_type = gt_action['action'].replace('left_click', 'click')

        # Only test on type/swipe episodes (and a sample of click for reference)
        if gt_type == 'click' and ep_idx % 10 != 0:
            continue  # Sample 10% of clicks for reference

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
            result_entry = {
                'episode_id': ep.get('episode_id', ep_idx),
                'gt_type': gt_type,
                'gt_action': gt_action,
            }

            # (a) Normal generation (greedy)
            with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
                output_ids = model.generate(**{k: v for k, v in inputs.items()},
                                            max_new_tokens=256, do_sample=False)
                normal_text = tokenizer.decode(output_ids[0, prompt_len:], skip_special_tokens=True)
            result_entry['normal_output'] = normal_text[:300]

            # (b) Forced prefix: force the correct action type
            if gt_type == 'type':
                forced_prefix = 'Action: I will type the required text.\n<action>{"action": "type", "text": "'
            elif gt_type == 'swipe':
                forced_prefix = 'Action: I will perform a swipe gesture.\n<action>{"action": "drag", "coordinate": ['
            else:
                forced_prefix = 'Action: I will click on the target element.\n<action>{"action": "left_click", "coordinate": ['

            # Append prefix to prompt and generate completion
            forced_text = text + forced_prefix
            if os.path.exists(screenshot):
                forced_inputs = processor(text=[forced_text], images=[img], return_tensors="pt", padding=True).to(device)
            else:
                forced_inputs = processor(text=[forced_text], return_tensors="pt", padding=True).to(device)

            forced_prompt_len = forced_inputs["input_ids"].shape[1]
            with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
                forced_output_ids = model.generate(**{k: v for k, v in forced_inputs.items()},
                                                    max_new_tokens=128, do_sample=False)
                forced_completion = tokenizer.decode(forced_output_ids[0, forced_prompt_len:], skip_special_tokens=True)

            result_entry['forced_prefix'] = forced_prefix
            result_entry['forced_completion'] = forced_completion[:300]
            result_entry['forced_full'] = forced_prefix + forced_completion

            # Evaluate forced output
            if gt_type == 'type':
                pred_text = extract_text_content(forced_prefix + forced_completion)
                gt_text = gt_action.get('text', '')
                if pred_text and gt_text:
                    # Simple overlap score
                    from difflib import SequenceMatcher
                    score = SequenceMatcher(None, pred_text.lower(), gt_text.lower()).ratio()
                    result_entry['forced_text_score'] = score
                    result_entry['pred_text'] = pred_text
                    result_entry['gt_text'] = gt_text
            elif gt_type == 'click':
                coord = extract_coordinate(forced_prefix + forced_completion)
                gt_coord = gt_action.get('coordinate')
                if coord and gt_coord:
                    dist = ((coord[0] - gt_coord[0])**2 + (coord[1] - gt_coord[1])**2)**0.5
                    result_entry['forced_coord_dist'] = dist

            forced_results.append(result_entry)

        except Exception as e:
            print(f"  Error ep{ep_idx}: {e}")
            continue

        if (ep_idx + 1) % 20 == 0:
            n_done = len(forced_results)
            print(f"  [{ep_idx+1}/{len(episodes)}] forced_prefix: {n_done} episodes done")

    # ═══════════════════════════════════════════
    # Part 2: Logit Gap Analysis
    # ═══════════════════════════════════════════
    print("\n" + "=" * 60)
    print("PART 2: LOGIT GAP ANALYSIS")
    print("=" * 60)

    # Find token IDs for action types
    # The model generates: <action>{"action": "left_click" or "type" or "drag"
    # We look at the logits right after: "action": "
    target_tokens = {
        'left': tokenizer.encode('left', add_special_tokens=False),  # "left" from "left_click"
        'type': tokenizer.encode('type', add_special_tokens=False),
        'drag': tokenizer.encode('drag', add_special_tokens=False),
        'sw': tokenizer.encode('sw', add_special_tokens=False),  # "sw" from "swipe"
    }
    print(f"Target token IDs: { {k: v for k, v in target_tokens.items()} }")

    # The key token ID is the first token of each action type word
    left_id = target_tokens['left'][0]
    type_id = target_tokens['type'][0]
    drag_id = target_tokens['drag'][0]

    logit_results = []
    for ep_idx, ep in enumerate(episodes):
        goal = ep['goal']
        step = ep['steps'][0]
        screenshot = step['screenshot']
        gt_type = step['action']['action'].replace('left_click', 'click')

        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        content = []
        if os.path.exists(screenshot):
            content.append({"type": "image", "image": f"file://{screenshot}"})
        content.append({"type": "text", "text": f"Task: {goal}"})
        messages.append({"role": "user", "content": content})

        try:
            # Build prompt up to the action type decision point
            # Append: 'Action: ...\n<action>{"action": "'
            decision_prefix = 'Action: I will perform the next step.\n<action>{"action": "'
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            full_text = text + decision_prefix

            if os.path.exists(screenshot):
                img = Image.open(screenshot).convert("RGB")
                inputs = processor(text=[full_text], images=[img], return_tensors="pt", padding=True).to(device)
            else:
                inputs = processor(text=[full_text], return_tensors="pt", padding=True).to(device)

            # Single forward pass to get logits at the decision point
            with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
                outputs = model(**{k: v for k, v in inputs.items()})
                # Last position logits = next token prediction after '"action": "'
                last_logits = outputs.logits[0, -1, :]  # [vocab_size]

            logit_left = last_logits[left_id].item()
            logit_type = last_logits[type_id].item()
            logit_drag = last_logits[drag_id].item()

            # Softmax probs
            probs = torch.softmax(last_logits.float(), dim=0)
            prob_left = probs[left_id].item()
            prob_type = probs[type_id].item()
            prob_drag = probs[drag_id].item()

            logit_results.append({
                'episode_id': ep.get('episode_id', ep_idx),
                'gt_type': gt_type,
                'logit_left': logit_left,
                'logit_type': logit_type,
                'logit_drag': logit_drag,
                'prob_left': prob_left,
                'prob_type': prob_type,
                'prob_drag': prob_drag,
                'gap_left_type': logit_left - logit_type,
                'gap_left_drag': logit_left - logit_drag,
            })

        except Exception as e:
            print(f"  Error ep{ep_idx} logit: {e}")
            continue

        if (ep_idx + 1) % 50 == 0:
            print(f"  [{ep_idx+1}/{len(episodes)}] logit_gap: done")

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, f"shard_{args.shard_id}.json")
    with open(out_path, 'w') as f:
        json.dump({'forced_prefix': forced_results, 'logit_gap': logit_results}, f)
    print(f"Saved to {out_path}")


def merge(output_dir):
    """Merge shards and analyze."""
    import glob as g
    shards = sorted(g.glob(os.path.join(output_dir, "shard_*.json")))
    print(f"Merging {len(shards)} shards")

    all_forced = []
    all_logit = []
    for sf in shards:
        with open(sf) as f:
            data = json.load(f)
        all_forced.extend(data['forced_prefix'])
        all_logit.extend(data['logit_gap'])

    # ═══ Forced Prefix Analysis ═══
    print("\n" + "=" * 80)
    print("FORCED-PREFIX DECODE RESULTS")
    print("=" * 80)

    by_gt_type = defaultdict(list)
    for r in all_forced:
        by_gt_type[r['gt_type']].append(r)

    print(f"\nGT=type ({len(by_gt_type['type'])} episodes):")
    type_scores = [r['forced_text_score'] for r in by_gt_type['type'] if 'forced_text_score' in r]
    if type_scores:
        print(f"  Forced-prefix text similarity: mean={np.mean(type_scores):.3f}, median={np.median(type_scores):.3f}")
        print(f"  Score > 0.5: {sum(1 for s in type_scores if s > 0.5)}/{len(type_scores)} ({sum(1 for s in type_scores if s > 0.5)/len(type_scores)*100:.1f}%)")
        print(f"  Score > 0.8: {sum(1 for s in type_scores if s > 0.8)}/{len(type_scores)} ({sum(1 for s in type_scores if s > 0.8)/len(type_scores)*100:.1f}%)")
        # Show examples
        print("  Examples (top 3):")
        sorted_by_score = sorted([r for r in by_gt_type['type'] if 'forced_text_score' in r], key=lambda x: -x['forced_text_score'])
        for r in sorted_by_score[:3]:
            print(f"    score={r['forced_text_score']:.2f}: pred='{r.get('pred_text','')[:50]}' gt='{r.get('gt_text','')[:50]}'")

    print(f"\nGT=click (sample, {len(by_gt_type['click'])} episodes):")
    click_dists = [r['forced_coord_dist'] for r in by_gt_type['click'] if 'forced_coord_dist' in r]
    if click_dists:
        print(f"  Forced-prefix coord dist: mean={np.mean(click_dists):.1f}px, median={np.median(click_dists):.1f}px")
        print(f"  <50px: {sum(1 for d in click_dists if d < 50)}/{len(click_dists)} ({sum(1 for d in click_dists if d < 50)/len(click_dists)*100:.1f}%)")

    print(f"\nGT=swipe ({len(by_gt_type['swipe'])} episodes):")
    for r in by_gt_type['swipe'][:5]:
        print(f"  ep{r['episode_id']}: forced_output='{r.get('forced_full','')[:100]}'")

    # ═══ Logit Gap Analysis ═══
    print("\n" + "=" * 80)
    print("LOGIT GAP ANALYSIS")
    print("=" * 80)

    by_gt = defaultdict(list)
    for r in all_logit:
        by_gt[r['gt_type']].append(r)

    print(f"\n{'GT Type':<8} | {'N':>4} | {'P(left)':>8} | {'P(type)':>8} | {'P(drag)':>8} | {'Gap L-T':>8} | {'Gap L-D':>8}")
    print("-" * 75)
    for gt_type in ['click', 'type', 'swipe']:
        entries = by_gt[gt_type]
        if not entries:
            continue
        n = len(entries)
        p_left = np.mean([e['prob_left'] for e in entries])
        p_type = np.mean([e['prob_type'] for e in entries])
        p_drag = np.mean([e['prob_drag'] for e in entries])
        gap_lt = np.mean([e['gap_left_type'] for e in entries])
        gap_ld = np.mean([e['gap_left_drag'] for e in entries])
        print(f"  {gt_type:<6} | {n:>4} | {p_left:>7.4f} | {p_type:>7.4f} | {p_drag:>7.4f} | {gap_lt:>7.2f} | {gap_ld:>7.2f}")

    # Is the gap different for GT=type vs GT=click?
    if by_gt['type'] and by_gt['click']:
        click_gaps = [e['gap_left_type'] for e in by_gt['click']]
        type_gaps = [e['gap_left_type'] for e in by_gt['type']]
        print(f"\n  Gap(left-type) when GT=click: {np.mean(click_gaps):.2f} ± {np.std(click_gaps):.2f}")
        print(f"  Gap(left-type) when GT=type:  {np.mean(type_gaps):.2f} ± {np.std(type_gaps):.2f}")
        if np.mean(type_gaps) < np.mean(click_gaps):
            print(f"  → Model is SLIGHTLY less confident about click when GT=type (gap {np.mean(click_gaps)-np.mean(type_gaps):.2f} smaller)")
        else:
            print(f"  → Model is equally confident about click regardless of GT type")

    # Save summary
    summary_path = os.path.join(output_dir, "summary.json")
    summary = {
        'forced_prefix': {
            'type_scores': type_scores if type_scores else [],
            'click_dists': click_dists if click_dists else [],
            'n_type': len(by_gt_type['type']),
            'n_swipe': len(by_gt_type['swipe']),
        },
        'logit_gap': {
            gt: {
                'n': len(entries),
                'mean_prob_left': float(np.mean([e['prob_left'] for e in entries])),
                'mean_prob_type': float(np.mean([e['prob_type'] for e in entries])),
                'mean_prob_drag': float(np.mean([e['prob_drag'] for e in entries])),
                'mean_gap_left_type': float(np.mean([e['gap_left_type'] for e in entries])),
                'mean_gap_left_drag': float(np.mean([e['gap_left_drag'] for e in entries])),
            }
            for gt, entries in by_gt.items() if entries
        }
    }
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved summary to {summary_path}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == '--merge':
        merge(sys.argv[2])
    else:
        main()
