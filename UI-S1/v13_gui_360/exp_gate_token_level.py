"""
Gate analysis v3: Token-level gate values.
Check if gates differ between image tokens vs text tokens.
Only runs 20 episodes, captures full per-token gate distributions.
"""

import json
import os
import sys
from collections import defaultdict
import statistics

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
import argparse


class TokenLevelGateCollector:
    def __init__(self):
        self.reset()

    def reset(self):
        # key -> list of full gate tensors [S, 1]
        self.data = {}

    def hook_model(self, wrapper):
        from v13_gui_360.iterative_cooperative_lora import IterativeCooperativeLoRALinear
        for name, module in wrapper.named_modules():
            if isinstance(module, IterativeCooperativeLoRALinear):
                parts = name.split('.')
                layer_idx = None
                for i, p in enumerate(parts):
                    if p == 'layers' and i + 1 < len(parts):
                        try: layer_idx = int(parts[i + 1])
                        except: pass
                proj = parts[-1]
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

                    key = f"L{layer_idx:02d}_{proj_name}_r{t}"
                    collector.data[key] = {
                        'g_12': g_12[0, :, 0].detach().cpu(),  # [S]
                        'g_21': g_21[0, :, 0].detach().cpu(),
                        'r': r[0, :, 0].detach().cpu(),
                    }

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
    parser.add_argument("--max_episodes", type=int, default=20)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    device = f"cuda:{args.gpu}"

    with open(args.test_data) as f:
        episodes = [json.loads(l) for l in f]
    episodes = episodes[:args.max_episodes]

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

    collector = TokenLevelGateCollector()
    collector.hook_model(model)
    print("Hooks installed")

    # We need to identify which tokens are image vs text
    # Qwen2.5-VL uses special image tokens: <|vision_start|> ... <|vision_end|>
    vision_start_id = processor.tokenizer.convert_tokens_to_ids("<|vision_start|>")
    vision_end_id = processor.tokenizer.convert_tokens_to_ids("<|vision_end|>")
    image_pad_id = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")

    print(f"Vision tokens: start={vision_start_id}, end={vision_end_id}, pad={image_pad_id}")

    # Focus on a few key layers
    focus_layers = ['L10', 'L18', 'L27']

    # Collect results
    all_results = []

    for ep_idx, ep in enumerate(episodes):
        eid = ep['episode_id']
        goal = ep['goal']

        # Only first step
        step = ep['steps'][0]
        screenshot = step['screenshot']
        gt_type = step['action'].get('action', 'unknown')
        if gt_type in ('click', 'left_click', 'right_click'):
            gt_type = 'click'

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

            input_ids = inputs["input_ids"][0].cpu()
            seq_len = len(input_ids)

            # Find image vs text token ranges
            is_image = torch.zeros(seq_len, dtype=torch.bool)
            in_vision = False
            for i, tid in enumerate(input_ids.tolist()):
                if tid == vision_start_id:
                    in_vision = True
                elif tid == vision_end_id:
                    in_vision = False
                elif in_vision or tid == image_pad_id:
                    is_image[i] = True

            n_img = is_image.sum().item()
            n_txt = seq_len - n_img

            collector.reset()
            with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
                _ = model(**inputs)

            # Analyze per-token gates for focus layers
            result = {'eid': eid, 'gt_type': gt_type, 'seq_len': seq_len,
                      'n_img': n_img, 'n_txt': n_txt, 'layers': {}}

            for layer in focus_layers:
                key = f"{layer}_q_proj_r1"
                if key in collector.data:
                    d = collector.data[key]
                    g12 = d['g_12'][:seq_len]
                    g21 = d['g_21'][:seq_len]
                    r = d['r'][:seq_len]

                    # Split by image vs text
                    img_mask = is_image[:len(g12)]
                    txt_mask = ~img_mask

                    g_avg = (g12 + g21) / 2

                    result['layers'][layer] = {
                        'img_gate_mean': g_avg[img_mask].mean().item() if img_mask.any() else None,
                        'img_gate_std': g_avg[img_mask].std().item() if img_mask.sum() > 1 else None,
                        'txt_gate_mean': g_avg[txt_mask].mean().item() if txt_mask.any() else None,
                        'txt_gate_std': g_avg[txt_mask].std().item() if txt_mask.sum() > 1 else None,
                        'img_g12_mean': g12[img_mask].mean().item() if img_mask.any() else None,
                        'img_g21_mean': g21[img_mask].mean().item() if img_mask.any() else None,
                        'txt_g12_mean': g12[txt_mask].mean().item() if txt_mask.any() else None,
                        'txt_g21_mean': g21[txt_mask].mean().item() if txt_mask.any() else None,
                        'img_r_mean': r[img_mask].mean().item() if img_mask.any() else None,
                        'txt_r_mean': r[txt_mask].mean().item() if txt_mask.any() else None,
                        # Distribution stats
                        'gate_min': g_avg.min().item(),
                        'gate_max': g_avg.max().item(),
                        'gate_p10': g_avg.float().quantile(0.1).item(),
                        'gate_p90': g_avg.float().quantile(0.9).item(),
                    }

            all_results.append(result)

        except Exception as e:
            print(f"Error ep{eid}: {e}")
            continue

        if (ep_idx + 1) % 5 == 0:
            print(f"  {ep_idx+1}/{len(episodes)} done")

    # ── Analysis ──
    print(f"\nCollected {len(all_results)} step records")
    print(f"Avg seq_len: {statistics.mean(r['seq_len'] for r in all_results):.0f}")
    print(f"Avg n_img: {statistics.mean(r['n_img'] for r in all_results):.0f}")
    print(f"Avg n_txt: {statistics.mean(r['n_txt'] for r in all_results):.0f}")

    print("\n" + "=" * 80)
    print("IMAGE TOKENS vs TEXT TOKENS: GATE VALUES")
    print("=" * 80)

    for layer in focus_layers:
        img_gates = [r['layers'][layer]['img_gate_mean'] for r in all_results
                     if layer in r['layers'] and r['layers'][layer]['img_gate_mean'] is not None]
        txt_gates = [r['layers'][layer]['txt_gate_mean'] for r in all_results
                     if layer in r['layers'] and r['layers'][layer]['txt_gate_mean'] is not None]

        if img_gates and txt_gates:
            img_m = statistics.mean(img_gates)
            txt_m = statistics.mean(txt_gates)
            diff = img_m - txt_m
            print(f"\n  {layer} q_proj round 1:")
            print(f"    Image tokens: gate={img_m:.4f} ± {statistics.stdev(img_gates):.4f}")
            print(f"    Text tokens:  gate={txt_m:.4f} ± {statistics.stdev(txt_gates):.4f}")
            print(f"    Diff (img-txt): {diff:+.4f}")

        # Also g_12 vs g_21 separately
        img_g12 = [r['layers'][layer]['img_g12_mean'] for r in all_results
                   if layer in r['layers'] and r['layers'][layer]['img_g12_mean'] is not None]
        img_g21 = [r['layers'][layer]['img_g21_mean'] for r in all_results
                   if layer in r['layers'] and r['layers'][layer]['img_g21_mean'] is not None]
        txt_g12 = [r['layers'][layer]['txt_g12_mean'] for r in all_results
                   if layer in r['layers'] and r['layers'][layer]['txt_g12_mean'] is not None]
        txt_g21 = [r['layers'][layer]['txt_g21_mean'] for r in all_results
                   if layer in r['layers'] and r['layers'][layer]['txt_g21_mean'] is not None]

        if img_g12:
            print(f"    Breakdown: img g_12={statistics.mean(img_g12):.4f}, img g_21={statistics.mean(img_g21):.4f}")
            print(f"               txt g_12={statistics.mean(txt_g12):.4f}, txt g_21={statistics.mean(txt_g21):.4f}")

    print("\n" + "=" * 80)
    print("IMAGE TOKENS vs TEXT TOKENS: ROUTING WEIGHTS")
    print("=" * 80)

    for layer in focus_layers:
        img_r = [r['layers'][layer]['img_r_mean'] for r in all_results
                 if layer in r['layers'] and r['layers'][layer]['img_r_mean'] is not None]
        txt_r = [r['layers'][layer]['txt_r_mean'] for r in all_results
                 if layer in r['layers'] and r['layers'][layer]['txt_r_mean'] is not None]

        if img_r and txt_r:
            print(f"  {layer}: img_r={statistics.mean(img_r):.4f}, txt_r={statistics.mean(txt_r):.4f}, diff={statistics.mean(img_r)-statistics.mean(txt_r):+.4f}")

    print("\n" + "=" * 80)
    print("GATE VALUE RANGE (per sample)")
    print("=" * 80)

    for layer in focus_layers:
        mins = [r['layers'][layer]['gate_min'] for r in all_results if layer in r['layers']]
        maxs = [r['layers'][layer]['gate_max'] for r in all_results if layer in r['layers']]
        p10s = [r['layers'][layer]['gate_p10'] for r in all_results if layer in r['layers']]
        p90s = [r['layers'][layer]['gate_p90'] for r in all_results if layer in r['layers']]
        if mins:
            print(f"  {layer}: min={statistics.mean(mins):.4f}, p10={statistics.mean(p10s):.4f}, p90={statistics.mean(p90s):.4f}, max={statistics.mean(maxs):.4f}")

    # By action type
    print("\n" + "=" * 80)
    print("IMAGE vs TEXT GATE BY ACTION TYPE")
    print("=" * 80)

    for layer in focus_layers:
        for gt_type in ['click', 'type', 'swipe']:
            img_g = [r['layers'][layer]['img_gate_mean'] for r in all_results
                     if r['gt_type'] == gt_type and layer in r['layers']
                     and r['layers'][layer]['img_gate_mean'] is not None]
            txt_g = [r['layers'][layer]['txt_gate_mean'] for r in all_results
                     if r['gt_type'] == gt_type and layer in r['layers']
                     and r['layers'][layer]['txt_gate_mean'] is not None]
            if img_g:
                print(f"  {layer} {gt_type:>6}: img={statistics.mean(img_g):.4f}, txt={statistics.mean(txt_g):.4f}, diff={statistics.mean(img_g)-statistics.mean(txt_g):+.4f}, n={len(img_g)}")


if __name__ == "__main__":
    main()
