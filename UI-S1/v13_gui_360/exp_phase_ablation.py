"""
Phase-Conditional Communication Ablation.

Run model with communication enabled only during specific generation phases:
  1. full: normal communication (baseline)
  2. no_comm: gates forced to 0 everywhere
  3. planning_only: comm only during planning phase (before <action>)
  4. coord_only: comm only during coordinate phase
  5. type_only: comm only during action_type phase

Measures how each phase's communication causally affects coordinate accuracy.
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
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, LogitsProcessor


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


class CommState:
    """Global communication state read by patched modules."""
    enabled = True


class PhaseTracker(LogitsProcessor):
    """Track generation phase and update CommState accordingly."""

    def __init__(self, tokenizer, prompt_len, mode):
        self.tokenizer = tokenizer
        self.prompt_len = prompt_len
        self.mode = mode  # 'full', 'no_comm', 'planning_only', 'coord_only', 'type_only'

    def __call__(self, input_ids, scores):
        # Determine phase from generated tokens
        gen_ids = input_ids[0, self.prompt_len:]
        if len(gen_ids) == 0:
            phase = 'planning'
        else:
            text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
            phase = self._get_phase(text)

        # Set communication state based on mode
        if self.mode == 'full':
            CommState.enabled = True
        elif self.mode == 'no_comm':
            CommState.enabled = False
        elif self.mode == 'planning_only':
            CommState.enabled = (phase == 'planning')
        elif self.mode == 'coord_only':
            CommState.enabled = (phase == 'coordinate')
        elif self.mode == 'type_only':
            CommState.enabled = (phase == 'action_type')

        return scores  # Don't modify logits

    def _get_phase(self, text):
        if '<action>' not in text:
            return 'planning'
        after_action = text.split('<action>')[-1]
        if '"coordinate"' in after_action and '[' in after_action.split('"coordinate"')[-1]:
            return 'coordinate'
        elif '"action"' in after_action:
            return 'action_type'
        else:
            return 'action_start'


def patch_model_with_comm_control(model):
    """Patch LoRA modules to respect CommState.enabled."""
    from v13_gui_360.iterative_cooperative_lora import IterativeCooperativeLoRALinear

    for name, module in model.named_modules():
        if isinstance(module, IterativeCooperativeLoRALinear):
            _patch_module(module)


def _patch_module(module):
    if module._comm_params is None:
        return

    orig_comm_params = module._comm_params

    def controlled_forward(x):
        base_out = module.base_linear(x)
        if module._route_weight is None:
            return base_out

        x_drop = module.lora_dropout(x)
        dtype = x_drop.dtype
        w = module._route_weight.to(dtype)
        logits = F.linear(x_drop, w.unsqueeze(0))
        r = torch.sigmoid(logits)
        module._last_routing_weights = r.detach()

        h_1 = F.linear(x_drop, module.lora_A_1.to(dtype))
        h_2 = F.linear(x_drop, module.lora_A_2.to(dtype))

        # Communication controlled by global state
        if CommState.enabled:
            T = orig_comm_params['T']
            for t in range(T):
                g_12 = torch.sigmoid(
                    F.linear(h_1, orig_comm_params['gate_12'][t].to(dtype).unsqueeze(0)))
                g_21 = torch.sigmoid(
                    F.linear(h_2, orig_comm_params['gate_21'][t].to(dtype).unsqueeze(0)))
                h_1 = h_1 + g_12 * F.linear(h_2, orig_comm_params['W_12'][t].to(dtype))
                h_2 = h_2 + g_21 * F.linear(h_1, orig_comm_params['W_21'][t].to(dtype))

        h_blend = r * h_1 + (1 - r) * h_2
        delta_out = F.linear(h_blend, module.lora_B.to(dtype)) * module.scaling
        return base_out + delta_out

    module.forward = controlled_forward


def extract_action(text):
    """Extract action type and coordinate from generated text."""
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
    patch_model_with_comm_control(model)
    print("Model patched with comm control")

    modes = ['full', 'no_comm', 'planning_only', 'coord_only', 'type_only']
    results = {mode: [] for mode in modes}

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

            for mode in modes:
                tracker = PhaseTracker(processor.tokenizer, prompt_len, mode)
                CommState.enabled = (mode == 'full')

                with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    output_ids = model.generate(
                        **{k: v for k, v in inputs.items()},
                        max_new_tokens=256, do_sample=False,
                        logits_processor=[tracker],
                    )
                    gen_text = processor.tokenizer.decode(output_ids[0, prompt_len:], skip_special_tokens=True)

                atype, coord = extract_action(gen_text)
                results[mode].append({
                    'episode_id': ep.get('episode_id', ep_idx),
                    'type': atype,
                    'coord': coord,
                    'text': gen_text[:200],
                })

        except Exception as e:
            print(f"Error ep{ep_idx}: {e}")
            for mode in modes:
                results[mode].append({'episode_id': ep.get('episode_id', ep_idx), 'type': 'error', 'coord': None, 'text': ''})

        if (ep_idx + 1) % 10 == 0:
            print(f"  [{ep_idx+1}/{len(episodes)}] done")

    # Save
    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, f"shard_{args.shard_id}.json")
    with open(out_path, 'w') as f:
        json.dump(results, f)
    print(f"Saved to {out_path}")


def merge(output_dir):
    """Merge shards and analyze."""
    import glob as g
    shards = sorted(g.glob(os.path.join(output_dir, "shard_*.json")))
    print(f"Merging {len(shards)} shards")

    merged = defaultdict(list)
    for sf in shards:
        with open(sf) as f:
            data = json.load(f)
        for mode, eps in data.items():
            merged[mode].extend(eps)

    # Load GT
    test_data_path = os.path.join(os.path.dirname(output_dir), "data", "gui360_test_968.jsonl")
    if not os.path.exists(test_data_path):
        test_data_path = "/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/v13_gui_360/data/gui360_test_968.jsonl"
    with open(test_data_path) as f:
        gt_episodes = [json.loads(l) for l in f]

    # Build GT lookup
    gt_by_id = {}
    for ep in gt_episodes:
        eid = ep['episode_id']
        step0 = ep['steps'][0]
        gt_by_id[eid] = {
            'type': step0['action']['action'].replace('left_click', 'click'),
            'coord': step0['action'].get('coordinate'),
        }

    print(f"\nTotal episodes: {len(merged['full'])}")
    print("\n" + "=" * 80)
    print("PHASE-CONDITIONAL ABLATION RESULTS")
    print("=" * 80)

    modes = ['full', 'no_comm', 'planning_only', 'coord_only', 'type_only']
    print(f"\n{'Mode':<18} | {'Type Acc':>8} | {'Click%':>7} | {'Mean dist':>9} | {'<50px':>6} | {'<100px':>7}")
    print("-" * 75)

    for mode in modes:
        eps = merged[mode]
        n = len(eps)
        type_correct = 0
        click_count = 0
        dists = []
        for ep in eps:
            eid = ep['episode_id']
            gt = gt_by_id.get(eid, {})
            if ep['type'] == gt.get('type'):
                type_correct += 1
            if ep['type'] == 'click':
                click_count += 1
            if ep['coord'] and gt.get('coord'):
                d = ((ep['coord'][0] - gt['coord'][0])**2 + (ep['coord'][1] - gt['coord'][1])**2)**0.5
                dists.append(d)

        type_acc = type_correct / n * 100 if n else 0
        click_pct = click_count / n * 100 if n else 0
        mean_d = sum(dists) / len(dists) if dists else 0
        lt50 = sum(1 for d in dists if d < 50) / len(dists) * 100 if dists else 0
        lt100 = sum(1 for d in dists if d < 100) / len(dists) * 100 if dists else 0
        print(f"  {mode:<16} | {type_acc:>7.1f}% | {click_pct:>6.1f}% | {mean_d:>8.1f}px | {lt50:>5.1f}% | {lt100:>6.1f}%")

    # Detailed: full vs no_comm coordinate comparison
    print("\n" + "=" * 80)
    print("FULL vs NO_COMM — Per-episode coordinate comparison")
    print("=" * 80)
    full_eps = merged['full']
    nocomm_eps = merged['no_comm']
    full_better = 0
    nocomm_better = 0
    same = 0
    for fe, ne in zip(full_eps, nocomm_eps):
        gt = gt_by_id.get(fe['episode_id'], {})
        if not gt.get('coord') or not fe['coord'] or not ne['coord']:
            continue
        d_full = ((fe['coord'][0] - gt['coord'][0])**2 + (fe['coord'][1] - gt['coord'][1])**2)**0.5
        d_nocomm = ((ne['coord'][0] - gt['coord'][0])**2 + (ne['coord'][1] - gt['coord'][1])**2)**0.5
        if d_full < d_nocomm - 1:
            full_better += 1
        elif d_nocomm < d_full - 1:
            nocomm_better += 1
        else:
            same += 1
    total = full_better + nocomm_better + same
    print(f"  Full comm better: {full_better}/{total} ({full_better/total*100:.1f}%)")
    print(f"  No comm better:   {nocomm_better}/{total} ({nocomm_better/total*100:.1f}%)")
    print(f"  Same (±1px):      {same}/{total} ({same/total*100:.1f}%)")

    # Save merged summary
    summary_path = os.path.join(output_dir, "summary.json")
    summary = {}
    for mode in modes:
        eps = merged[mode]
        dists = []
        for ep in eps:
            gt = gt_by_id.get(ep['episode_id'], {})
            if ep['coord'] and gt.get('coord'):
                d = ((ep['coord'][0] - gt['coord'][0])**2 + (ep['coord'][1] - gt['coord'][1])**2)**0.5
                dists.append(d)
        summary[mode] = {
            'n': len(eps),
            'mean_dist': sum(dists) / len(dists) if dists else 0,
            'lt50': sum(1 for d in dists if d < 50) / len(dists) * 100 if dists else 0,
            'lt100': sum(1 for d in dists if d < 100) / len(dists) * 100 if dists else 0,
        }
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved summary to {summary_path}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == '--merge':
        merge(sys.argv[2])
    else:
        main()
