"""
Experiment: Analyze V13 communication gate values.

Hypothesis: Gate values (expert deliberation intensity) correlate with
step difficulty — higher gates on type/swipe steps (uncertain) vs click (routine).

Method:
1. Load V13 ep3 model
2. For each test step, run a single forward pass with the prompt + screenshot
3. Hook into IterativeCooperativeLoRALinear to capture gate values
4. Aggregate and correlate with GT action type, step success, trajectory position

Usage:
    python v13_gui_360/exp_gate_analysis.py \
        --base_model checkpoints/Qwen2.5-VL-7B-Instruct \
        --coop_checkpoint checkpoints/v13_gui360_rl/epoch-3/cooperative \
        --test_data v13_gui_360/data/gui360_test_968.jsonl \
        --max_episodes 200 \
        --output v13_gui_360/gate_analysis_results.json
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from typing import Dict, List

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

# ── Gate Hook ──

class GateCollector:
    """Hooks into IterativeCooperativeLoRALinear modules to capture gate values."""

    def __init__(self):
        self.reset()
        self._hooks = []

    def reset(self):
        """Clear collected gate values for a new forward pass."""
        self.gate_values = []  # list of (layer_idx, module_name, round_t, g_12_mean, g_21_mean)

    def hook_model(self, wrapper):
        """Install hooks on all cooperative LoRA modules."""
        from v13_gui_360.iterative_cooperative_lora import IterativeCooperativeLoRALinear

        for name, module in wrapper.named_modules():
            if isinstance(module, IterativeCooperativeLoRALinear):
                # Monkey-patch the forward to capture gates
                self._patch_forward(module, name)

    def _patch_forward(self, module, name):
        """Patch forward to store gate values without changing output."""
        original_forward = module.forward
        collector = self

        def hooked_forward(x):
            base_out = module.base_linear(x)
            if module._route_weight is None:
                module._last_routing_weights = None
                return base_out

            x_drop = module.lora_dropout(x)
            dtype = x_drop.dtype

            w = module._route_weight.to(dtype)
            logits = F.linear(x_drop, w.unsqueeze(0))
            r = torch.sigmoid(logits)
            module._last_routing_weights = r.detach()

            h_1 = F.linear(x_drop, module.lora_A_1.to(dtype))
            h_2 = F.linear(x_drop, module.lora_A_2.to(dtype))

            if module._comm_params is not None:
                T = module._comm_params['T']
                for t in range(T):
                    g_12 = torch.sigmoid(
                        F.linear(h_1, module._comm_params['gate_12'][t].to(dtype).unsqueeze(0))
                    )
                    h_1 = h_1 + g_12 * F.linear(h_2, module._comm_params['W_12'][t].to(dtype))

                    g_21 = torch.sigmoid(
                        F.linear(h_2, module._comm_params['gate_21'][t].to(dtype).unsqueeze(0))
                    )
                    h_2 = h_2 + g_21 * F.linear(h_1, module._comm_params['W_21'][t].to(dtype))

                    # Record gate values (mean over all tokens in sequence)
                    collector.gate_values.append({
                        'module': name,
                        'round': t,
                        'g_12_mean': g_12.mean().item(),
                        'g_21_mean': g_21.mean().item(),
                        'g_12_std': g_12.std().item(),
                        'g_21_std': g_21.std().item(),
                        'routing_mean': r.mean().item(),
                    })

            h_blend = r * h_1 + (1 - r) * h_2
            delta = F.linear(h_blend, module.lora_B.to(dtype)) * module.scaling
            return base_out + delta

        module.forward = hooked_forward


# ── Prompt builder ──

SYSTEM_PROMPT = """You are a GUI operation assistant. You will receive a screenshot of a current GUI and a user's instruction. Your task is to determine the next action to perform.

Output format: First describe what you will do, then output the action in this format:
<action>{"action": "ACTION_TYPE", "coordinate": [x, y]}</action>
or for typing:
<action>{"action": "type", "text": "TEXT_TO_TYPE"}</action>
or for swiping:
<action>{"action": "swipe", "coordinate": [x1, y1], "direction": "DIRECTION"}</action>"""


def build_messages(goal, screenshot_path, history_text=""):
    """Build chat messages for a single step."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    content = []
    if os.path.exists(screenshot_path):
        content.append({"type": "image", "image": f"file://{screenshot_path}"})
    content.append({"type": "text", "text": f"Goal: {goal}\n{history_text}\nWhat is the next action?"})

    messages.append({"role": "user", "content": content})
    return messages


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", required=True)
    parser.add_argument("--coop_checkpoint", required=True)
    parser.add_argument("--test_data", required=True)
    parser.add_argument("--max_episodes", type=int, default=200)
    parser.add_argument("--output", default="v13_gui_360/gate_analysis_results.json")
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    device = f"cuda:{args.gpu}"

    # Load test data
    with open(args.test_data) as f:
        episodes = [json.loads(l) for l in f]
    episodes = episodes[:args.max_episodes]
    print(f"Loaded {len(episodes)} episodes")

    # Build metadata
    meta = {}
    for ep in episodes:
        eid = ep['episode_id']
        path = ep['steps'][0]['screenshot']
        parts = path.split('/')
        app = parts[parts.index('image') + 1]
        meta[eid] = {'app': app, 'num_steps': ep['num_steps']}

    # Load eval results for success labels
    import glob
    eval_files = sorted(glob.glob(
        f"{os.path.dirname(args.test_data)}/../outputs/epoch-3/eval_results_*.json"
    ))
    if not eval_files:
        eval_files = sorted(glob.glob(
            "/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/v13_gui_360/outputs/epoch-3/eval_results_*.json"
        ))
    eval_results = {}
    if eval_files:
        with open(eval_files[-1]) as f:
            data = json.load(f)
        eval_results = {ep['episode_id']: ep for ep in data['episodes']}
        print(f"Loaded eval results: {len(eval_results)} episodes")

    # Load model
    print("Loading processor...")
    processor = AutoProcessor.from_pretrained(args.base_model, trust_remote_code=True)

    print("Loading base model...")
    base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map={"": args.gpu},
    )

    print("Creating V13 wrapper...")
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

    print("Loading cooperative weights...")
    model.load_cooperative(args.coop_checkpoint)
    model.eval()

    # Install gate hooks
    collector = GateCollector()
    collector.hook_model(model)
    print("Gate hooks installed")

    # ── Run analysis ──
    all_step_records = []
    total_steps = 0

    for ep_idx, ep in enumerate(episodes):
        eid = ep['episode_id']
        goal = ep['goal']
        num_steps = ep['num_steps']

        # Get eval results for this episode
        eval_ep = eval_results.get(eid, {})
        step_results = eval_ep.get('step_results', [])

        for step_idx, step in enumerate(ep['steps']):
            screenshot = step['screenshot']
            gt_action = step['action']
            gt_type = gt_action.get('action', 'unknown')
            # Normalize type
            if gt_type in ('click', 'left_click', 'right_click'):
                gt_type_norm = 'click'
            elif gt_type == 'type':
                gt_type_norm = 'type'
            elif gt_type == 'swipe':
                gt_type_norm = 'swipe'
            else:
                gt_type_norm = gt_type

            # Get success label from eval results
            step_success = None
            if step_idx < len(step_results):
                step_success = step_results[step_idx].get('success', None)

            # Build prompt (just step 1 view — simplified, no history)
            messages = build_messages(goal, screenshot)

            try:
                text = processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )

                # Load image
                if os.path.exists(screenshot):
                    img = Image.open(screenshot).convert("RGB")
                    inputs = processor(
                        text=[text], images=[img], return_tensors="pt",
                        padding=True
                    ).to(device)
                else:
                    inputs = processor(
                        text=[text], return_tensors="pt",
                        padding=True
                    ).to(device)

                # Forward pass to capture gates (no generation needed)
                collector.reset()
                with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    outputs = model(**inputs)

                # Aggregate gate values across layers
                if collector.gate_values:
                    # Group by round
                    by_round = defaultdict(list)
                    for gv in collector.gate_values:
                        by_round[gv['round']].append(gv)

                    # Average across all modules for each round
                    round_summaries = {}
                    for t, gvs in by_round.items():
                        round_summaries[t] = {
                            'g_12_mean': sum(g['g_12_mean'] for g in gvs) / len(gvs),
                            'g_21_mean': sum(g['g_21_mean'] for g in gvs) / len(gvs),
                            'g_12_std': sum(g['g_12_std'] for g in gvs) / len(gvs),
                            'g_21_std': sum(g['g_21_std'] for g in gvs) / len(gvs),
                            'routing_mean': sum(g['routing_mean'] for g in gvs) / len(gvs),
                        }

                    # Overall gate magnitude
                    all_g12 = [g['g_12_mean'] for g in collector.gate_values]
                    all_g21 = [g['g_21_mean'] for g in collector.gate_values]
                    gate_avg = (sum(all_g12) + sum(all_g21)) / (len(all_g12) + len(all_g21))

                    record = {
                        'episode_id': eid,
                        'step_idx': step_idx,
                        'num_steps': num_steps,
                        'gt_type': gt_type_norm,
                        'step_success': step_success,
                        'app': meta[eid]['app'],
                        'step_fraction': (step_idx + 1) / num_steps,
                        'gate_avg': gate_avg,
                        'round_summaries': round_summaries,
                        'num_gate_readings': len(collector.gate_values),
                        # Per-layer detail (optional — for deeper analysis)
                        'per_module': collector.gate_values,
                    }
                    all_step_records.append(record)

            except Exception as e:
                print(f"  Error on ep {eid} step {step_idx}: {e}")
                continue

            total_steps += 1

        if (ep_idx + 1) % 20 == 0:
            print(f"Processed {ep_idx+1}/{len(episodes)} episodes, {total_steps} steps")

    print(f"\nTotal: {len(all_step_records)} step records")

    # ── Analysis ──
    print("\n" + "=" * 80)
    print("GATE ANALYSIS RESULTS")
    print("=" * 80)

    # 1. Gate values by action type
    by_type = defaultdict(list)
    for rec in all_step_records:
        by_type[rec['gt_type']].append(rec['gate_avg'])

    print("\n1. Gate avg by GT action type:")
    for t in ['click', 'type', 'swipe']:
        vals = by_type.get(t, [])
        if vals:
            import statistics
            print(f"  {t:>6}: mean={statistics.mean(vals):.4f}, std={statistics.stdev(vals) if len(vals)>1 else 0:.4f}, n={len(vals)}")

    # 2. Gate values by success/failure
    by_success = defaultdict(list)
    for rec in all_step_records:
        if rec['step_success'] is not None:
            by_success[rec['step_success']].append(rec['gate_avg'])

    print("\n2. Gate avg by step success:")
    for s in [True, False]:
        vals = by_success.get(s, [])
        if vals:
            print(f"  {'success' if s else 'failure':>8}: mean={statistics.mean(vals):.4f}, std={statistics.stdev(vals) if len(vals)>1 else 0:.4f}, n={len(vals)}")

    # 3. Gate values by app
    by_app = defaultdict(list)
    for rec in all_step_records:
        by_app[rec['app']].append(rec['gate_avg'])

    print("\n3. Gate avg by app:")
    for app in ['word', 'excel', 'ppt']:
        vals = by_app.get(app, [])
        if vals:
            print(f"  {app:>6}: mean={statistics.mean(vals):.4f}, n={len(vals)}")

    # 4. Gate values by step position
    print("\n4. Gate avg by step position in trajectory:")
    by_pos = defaultdict(list)
    for rec in all_step_records:
        frac = rec['step_fraction']
        if frac <= 0.25: bucket = '0-25%'
        elif frac <= 0.5: bucket = '25-50%'
        elif frac <= 0.75: bucket = '50-75%'
        else: bucket = '75-100%'
        by_pos[bucket].append(rec['gate_avg'])

    for b in ['0-25%', '25-50%', '50-75%', '75-100%']:
        vals = by_pos.get(b, [])
        if vals:
            print(f"  {b:>8}: mean={statistics.mean(vals):.4f}, n={len(vals)}")

    # 5. Cross: action type × success
    print("\n5. Gate avg by action type × success:")
    cross = defaultdict(list)
    for rec in all_step_records:
        if rec['step_success'] is not None:
            key = (rec['gt_type'], rec['step_success'])
            cross[key].append(rec['gate_avg'])

    for t in ['click', 'type', 'swipe']:
        for s in [True, False]:
            vals = cross.get((t, s), [])
            if vals:
                print(f"  {t:>6} {'ok' if s else 'fail':>4}: mean={statistics.mean(vals):.4f}, n={len(vals)}")

    # 6. Per-round breakdown
    print("\n6. Gate values per communication round:")
    round_by_type = defaultdict(lambda: defaultdict(list))
    for rec in all_step_records:
        for t_str, rs in rec['round_summaries'].items():
            t = int(t_str)
            round_by_type[t][rec['gt_type']].append((rs['g_12_mean'], rs['g_21_mean']))

    for t in sorted(round_by_type.keys()):
        print(f"  Round {t}:")
        for action_type in ['click', 'type', 'swipe']:
            vals = round_by_type[t].get(action_type, [])
            if vals:
                g12_mean = sum(v[0] for v in vals) / len(vals)
                g21_mean = sum(v[1] for v in vals) / len(vals)
                print(f"    {action_type:>6}: g_12={g12_mean:.4f}, g_21={g21_mean:.4f}, n={len(vals)}")

    # Save results (without per_module to keep file small)
    save_records = []
    for rec in all_step_records:
        r = {k: v for k, v in rec.items() if k != 'per_module'}
        save_records.append(r)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump({'records': save_records}, f)
    print(f"\nSaved {len(save_records)} records to {args.output}")


if __name__ == "__main__":
    main()
