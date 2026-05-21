"""
Gate analysis v2: Per-layer gate values, focusing on high-norm layers.
Only analyzes first 100 episodes but captures per-layer detail.
"""

import argparse
import json
import os
import sys
from collections import defaultdict
import statistics

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


class PerLayerGateCollector:
    """Collect gate values per layer per module."""

    def __init__(self):
        self.reset()

    def reset(self):
        # layer_module -> list of (round, g_12_mean, g_21_mean, g_12_last_tok, g_21_last_tok)
        self.layer_gates = defaultdict(list)

    def hook_model(self, wrapper):
        from v13_gui_360.iterative_cooperative_lora import IterativeCooperativeLoRALinear

        for name, module in wrapper.named_modules():
            if isinstance(module, IterativeCooperativeLoRALinear):
                # Extract layer index from name
                # e.g. "base_model.model.layers.17.self_attn.q_proj"
                parts = name.split('.')
                layer_idx = None
                for i, p in enumerate(parts):
                    if p == 'layers' and i + 1 < len(parts):
                        try:
                            layer_idx = int(parts[i + 1])
                        except ValueError:
                            pass
                proj = parts[-1] if parts else name  # q_proj, k_proj, etc.

                self._patch(module, layer_idx, proj)

    def _patch(self, module, layer_idx, proj_name):
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

                    # Record per-layer detail
                    key = f"L{layer_idx:02d}_{proj_name}"
                    collector.layer_gates[key].append({
                        'round': t,
                        'g_12_mean': g_12.mean().item(),
                        'g_21_mean': g_21.mean().item(),
                        # Last token (the one being generated) gate value
                        'g_12_last': g_12[0, -1, 0].item(),
                        'g_21_last': g_21[0, -1, 0].item(),
                        # Routing weight
                        'r_mean': r.mean().item(),
                        'r_last': r[0, -1, 0].item(),
                        # h norms for context
                        'h1_norm': h_1.norm(dim=-1).mean().item(),
                        'h2_norm': h_2.norm(dim=-1).mean().item(),
                    })

            h_blend = r * h_1 + (1 - r) * h_2
            delta = F.linear(h_blend, module.lora_B.to(dtype)) * module.scaling
            return base_out + delta

        module.forward = hooked_forward


SYSTEM_PROMPT = """You are a GUI operation assistant. You will receive a screenshot of a current GUI and a user's instruction. Your task is to determine the next action to perform.

Output format: First describe what you will do, then output the action in this format:
<action>{"action": "ACTION_TYPE", "coordinate": [x, y]}</action>
or for typing:
<action>{"action": "type", "text": "TEXT_TO_TYPE"}</action>
or for swiping:
<action>{"action": "swipe", "coordinate": [x1, y1], "direction": "DIRECTION"}</action>"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", required=True)
    parser.add_argument("--coop_checkpoint", required=True)
    parser.add_argument("--test_data", required=True)
    parser.add_argument("--max_episodes", type=int, default=100)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    device = f"cuda:{args.gpu}"

    with open(args.test_data) as f:
        episodes = [json.loads(l) for l in f]
    episodes = episodes[:args.max_episodes]

    # Metadata
    meta = {}
    for ep in episodes:
        eid = ep['episode_id']
        path = ep['steps'][0]['screenshot']
        parts = path.split('/')
        app = parts[parts.index('image') + 1]
        meta[eid] = {'app': app, 'num_steps': ep['num_steps']}

    # Load eval results
    import glob
    eval_files = sorted(glob.glob(
        "/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/v13_gui_360/outputs/epoch-3/eval_results_*.json"
    ))
    eval_results = {}
    if eval_files:
        with open(eval_files[-1]) as f:
            data = json.load(f)
        eval_results = {ep['episode_id']: ep for ep in data['episodes']}

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

    collector = PerLayerGateCollector()
    collector.hook_model(model)
    print("Hooks installed")

    # ── Run ──
    # Collect per-step, per-layer gate values
    step_records = []

    for ep_idx, ep in enumerate(episodes):
        eid = ep['episode_id']
        goal = ep['goal']

        eval_ep = eval_results.get(eid, {})
        step_results_list = eval_ep.get('step_results', [])

        for step_idx, step in enumerate(ep['steps']):
            screenshot = step['screenshot']
            gt_action = step['action']
            gt_type = gt_action.get('action', 'unknown')
            if gt_type in ('click', 'left_click', 'right_click'):
                gt_type = 'click'

            step_success = None
            if step_idx < len(step_results_list):
                step_success = step_results_list[step_idx].get('success', None)

            messages = [{"role": "system", "content": SYSTEM_PROMPT}]
            content = []
            if os.path.exists(screenshot):
                content.append({"type": "image", "image": f"file://{screenshot}"})
            content.append({"type": "text", "text": f"Goal: {goal}\nWhat is the next action?"})
            messages.append({"role": "user", "content": content})

            try:
                text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                if os.path.exists(screenshot):
                    img = Image.open(screenshot).convert("RGB")
                    inputs = processor(text=[text], images=[img], return_tensors="pt", padding=True).to(device)
                else:
                    inputs = processor(text=[text], return_tensors="pt", padding=True).to(device)

                collector.reset()
                with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    _ = model(**inputs)

                # Extract per-layer summary for this step
                layer_summary = {}
                for key, entries in collector.layer_gates.items():
                    for e in entries:
                        lkey = f"{key}_r{e['round']}"
                        layer_summary[lkey] = {
                            'g12': e['g_12_mean'],
                            'g21': e['g_21_mean'],
                            'g12_last': e['g_12_last'],
                            'g21_last': e['g_21_last'],
                            'r': e['r_mean'],
                            'r_last': e['r_last'],
                            'h1_norm': e['h1_norm'],
                            'h2_norm': e['h2_norm'],
                        }

                step_records.append({
                    'eid': eid, 'step': step_idx, 'gt_type': gt_type,
                    'success': step_success, 'app': meta[eid]['app'],
                    'layers': layer_summary,
                })

            except Exception as e:
                print(f"Error ep{eid} step{step_idx}: {e}")
                continue

        if (ep_idx + 1) % 25 == 0:
            print(f"  {ep_idx+1}/{len(episodes)} episodes done")

    print(f"\nCollected {len(step_records)} step records")

    # ── Analysis: focus on high-gate-norm layers ──
    high_gate_layers = ['L10', 'L18', 'L19', 'L26', 'L27']  # from gate norm analysis
    low_gate_layers = ['L08', 'L13', 'L23']

    print("\n" + "=" * 80)
    print("PER-LAYER GATE ANALYSIS: HIGH-NORM vs LOW-NORM LAYERS")
    print("=" * 80)

    for layer_group, layer_list, label in [
        (high_gate_layers, high_gate_layers, "HIGH gate norm"),
        (low_gate_layers, low_gate_layers, "LOW gate norm"),
    ]:
        print(f"\n--- {label} layers: {layer_list} ---")

        for proj in ['q_proj']:  # focus on q_proj for clarity
            for rnd in [1]:  # round 1 has larger gates
                by_type = defaultdict(list)
                by_success = defaultdict(list)

                for rec in step_records:
                    for layer in layer_list:
                        key = f"{layer}_{proj}_r{rnd}"
                        if key in rec['layers']:
                            ldata = rec['layers'][key]
                            g_avg = (ldata['g12'] + ldata['g21']) / 2
                            g_last = (ldata['g12_last'] + ldata['g21_last']) / 2
                            by_type[rec['gt_type']].append(g_avg)
                            if rec['success'] is not None:
                                by_success[rec['success']].append(g_avg)

                print(f"\n  {proj} round {rnd}:")
                print(f"  Gate avg by action type:")
                for t in ['click', 'type', 'swipe']:
                    vals = by_type.get(t, [])
                    if len(vals) > 1:
                        print(f"    {t:>6}: mean={statistics.mean(vals):.4f}, std={statistics.stdev(vals):.4f}, n={len(vals)}")
                    elif vals:
                        print(f"    {t:>6}: mean={vals[0]:.4f}, n=1")

                print(f"  Gate avg by success:")
                for s in [True, False]:
                    vals = by_success.get(s, [])
                    if len(vals) > 1:
                        print(f"    {'ok' if s else 'fail':>6}: mean={statistics.mean(vals):.4f}, std={statistics.stdev(vals):.4f}, n={len(vals)}")

    # ── Per-layer routing weights ──
    print("\n" + "=" * 80)
    print("ROUTING WEIGHTS (r) BY LAYER — expert 1 vs expert 2 preference")
    print("=" * 80)

    for layer in ['L00', 'L07', 'L10', 'L14', 'L18', 'L21', 'L27']:
        by_type_r = defaultdict(list)
        for rec in step_records:
            key = f"{layer}_q_proj_r0"
            if key in rec['layers']:
                by_type_r[rec['gt_type']].append(rec['layers'][key]['r'])

        parts = []
        for t in ['click', 'type', 'swipe']:
            vals = by_type_r.get(t, [])
            if vals:
                parts.append(f"{t}={statistics.mean(vals):.4f}")
        if parts:
            print(f"  {layer}: {', '.join(parts)}")

    # ── h1 vs h2 norm ratio by action type ──
    print("\n" + "=" * 80)
    print("h1/h2 NORM RATIO BY ACTION TYPE (expert activity)")
    print("=" * 80)

    for layer in ['L10', 'L18', 'L27']:
        by_type_h = defaultdict(list)
        for rec in step_records:
            key = f"{layer}_q_proj_r1"
            if key in rec['layers']:
                h1 = rec['layers'][key]['h1_norm']
                h2 = rec['layers'][key]['h2_norm']
                ratio = h1 / (h2 + 1e-8)
                by_type_h[rec['gt_type']].append(ratio)

        parts = []
        for t in ['click', 'type', 'swipe']:
            vals = by_type_h.get(t, [])
            if len(vals) > 1:
                parts.append(f"{t}: h1/h2={statistics.mean(vals):.3f}±{statistics.stdev(vals):.3f}")
        if parts:
            print(f"  {layer}: {', '.join(parts)}")


if __name__ == "__main__":
    main()
