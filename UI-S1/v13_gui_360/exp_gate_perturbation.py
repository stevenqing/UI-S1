"""
Gate Perturbation Experiment — Verify if gate bias can change action type predictions.

For each episode, generate with:
  1. Normal gates (baseline)
  2. Gates + delta (more communication)
  3. Gates - delta (less communication)

Check: does perturbation change action type (click→type/swipe)?
"""

import json
import os
import re
import sys
from collections import defaultdict

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
import argparse


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


class GatePerturbation:
    """Apply additive bias to all gate values during forward."""

    def __init__(self):
        self.delta = 0.0  # additive bias to gate logits (pre-sigmoid)
        self.layer_filter = None  # None = all layers, or set of layer indices

    def set_delta(self, delta, layer_filter=None):
        self.delta = delta
        self.layer_filter = layer_filter


def patch_model_with_perturbation(model, perturbation):
    """Patch cooperative LoRA modules to apply gate perturbation."""
    from v13_gui_360.iterative_cooperative_lora import IterativeCooperativeLoRALinear

    for name, module in model.named_modules():
        if isinstance(module, IterativeCooperativeLoRALinear):
            # Extract layer index
            parts = name.split('.')
            layer_idx = None
            for i, p in enumerate(parts):
                if p == 'layers' and i + 1 < len(parts):
                    try:
                        layer_idx = int(parts[i + 1])
                    except:
                        pass

            _patch_module(module, layer_idx, perturbation)


def _patch_module(module, layer_idx, perturbation):
    orig_comm_params = module._comm_params
    if orig_comm_params is None:
        return

    # Save original forward
    _original_forward = module.forward

    def perturbed_forward(x):
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

        T = orig_comm_params['T']
        for t in range(T):
            # Apply perturbation to gate logits
            gate_12_logit = F.linear(h_1, orig_comm_params['gate_12'][t].to(dtype).unsqueeze(0))
            gate_21_logit = F.linear(h_2, orig_comm_params['gate_21'][t].to(dtype).unsqueeze(0))

            # Add delta if this layer is in filter (or filter is None)
            delta = perturbation.delta
            if perturbation.layer_filter is not None and layer_idx not in perturbation.layer_filter:
                delta = 0.0

            g_12 = torch.sigmoid(gate_12_logit + delta)
            g_21 = torch.sigmoid(gate_21_logit + delta)

            h_1 = h_1 + g_12 * F.linear(h_2, orig_comm_params['W_12'][t].to(dtype))
            h_2 = h_2 + g_21 * F.linear(h_1, orig_comm_params['W_21'][t].to(dtype))

        h_blend = r * h_1 + (1 - r) * h_2
        delta_out = F.linear(h_blend, module.lora_B.to(dtype)) * module.scaling
        return base_out + delta_out

    module.forward = perturbed_forward


def extract_action_type(text):
    """Extract action type from generated text."""
    match = re.search(r'<action>\s*\{.*?"action"\s*:\s*"([^"]+)"', text, re.DOTALL)
    if match:
        atype = match.group(1).lower()
        if 'click' in atype:
            return 'click'
        elif 'type' in atype:
            return 'type'
        elif 'swipe' in atype or 'drag' in atype:
            return 'swipe'
        else:
            return atype
    return 'unknown'


def extract_coordinate(text):
    """Extract first coordinate from generated text."""
    match = re.search(r'"coordinate"\s*:\s*\[(\d+)\s*,\s*(\d+)\]', text)
    if match:
        return (int(match.group(1)), int(match.group(2)))
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", required=True)
    parser.add_argument("--coop_checkpoint", required=True)
    parser.add_argument("--test_data", required=True)
    parser.add_argument("--max_episodes", type=int, default=100)
    parser.add_argument("--deltas", type=str, default="-0.5,-0.2,-0.1,0,0.1,0.2,0.5",
                        help="Comma-separated gate logit deltas")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--shard_id", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    device = f"cuda:{args.gpu}"
    deltas = [float(d) for d in args.deltas.split(',')]

    with open(args.test_data) as f:
        all_episodes = [json.loads(l) for l in f]
    all_episodes = all_episodes[:args.max_episodes]
    episodes = all_episodes[args.shard_id::args.num_shards]
    print(f"Shard {args.shard_id}/{args.num_shards}: {len(episodes)} episodes (total {len(all_episodes)})")

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

    perturbation = GatePerturbation()
    patch_model_with_perturbation(model, perturbation)
    print("Model patched with perturbation hooks")

    # Also test layer-specific perturbation
    layer_configs = [
        ("all", None),
        ("L10_only", {10}),
        ("L18_only", {18}),
        ("L27_only", {27}),
    ]

    # Results: delta -> list of (action_type, coordinate, text)
    results = defaultdict(lambda: defaultdict(list))

    for ep_idx, ep in enumerate(episodes):
        eid = ep['episode_id']
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

            # Test each delta (all layers)
            for delta in deltas:
                perturbation.set_delta(delta, layer_filter=None)
                with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    output_ids = model.generate(**{k: v for k, v in inputs.items()},
                                                max_new_tokens=256, do_sample=False)
                    prompt_len = inputs["input_ids"].shape[1]
                    gen_text = processor.tokenizer.decode(output_ids[0, prompt_len:], skip_special_tokens=True)

                atype = extract_action_type(gen_text)
                coord = extract_coordinate(gen_text)
                results[f"all_d{delta}"]["types"].append(atype)
                results[f"all_d{delta}"]["coords"].append(coord)
                results[f"all_d{delta}"]["texts"].append(gen_text[:150])

            # Test large delta on specific layers
            for layer_name, layer_set in layer_configs[1:]:  # skip "all"
                for delta in [-0.5, 0.5]:
                    perturbation.set_delta(delta, layer_filter=layer_set)
                    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
                        output_ids = model.generate(**{k: v for k, v in inputs.items()},
                                                    max_new_tokens=256, do_sample=False)
                        prompt_len = inputs["input_ids"].shape[1]
                        gen_text = processor.tokenizer.decode(output_ids[0, prompt_len:], skip_special_tokens=True)

                    atype = extract_action_type(gen_text)
                    coord = extract_coordinate(gen_text)
                    results[f"{layer_name}_d{delta}"]["types"].append(atype)
                    results[f"{layer_name}_d{delta}"]["coords"].append(coord)

        except Exception as e:
            print(f"Error ep{eid}: {e}")
            continue

        if (ep_idx + 1) % 10 == 0:
            print(f"  [{ep_idx+1}/{len(episodes)}] done")

    # Reset perturbation
    perturbation.set_delta(0.0)

    # Save shard results
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        shard_file = os.path.join(args.output_dir, f"shard_{args.shard_id}.json")
        # Convert results to serializable format
        save_data = {}
        for key, val in results.items():
            save_data[key] = {"types": val["types"], "coords": val["coords"], "texts": val["texts"]}
        with open(shard_file, 'w') as f:
            json.dump(save_data, f)
        print(f"Saved shard {args.shard_id} to {shard_file}")
        if args.num_shards > 1:
            print(f"Shard {args.shard_id} done.")
            return

    # ── Analysis ──
    n_eps = len(results.get("all_d0", {}).get("types", []))
    print(f"\nCompleted {n_eps} episodes")

    # 1. Action type distribution by delta
    print("\n" + "=" * 80)
    print("ACTION TYPE DISTRIBUTION BY GATE DELTA (all layers)")
    print("=" * 80)

    for delta in deltas:
        key = f"all_d{delta}"
        types = results[key]["types"]
        if not types:
            continue
        type_counts = defaultdict(int)
        for t in types:
            type_counts[t] += 1
        total = len(types)
        parts = [f"{k}={v}({100*v/total:.1f}%)" for k, v in sorted(type_counts.items())]
        print(f"  delta={delta:+.1f}: {', '.join(parts)}")

    # 2. How many episodes changed action type?
    print("\n" + "=" * 80)
    print("ACTION TYPE CHANGES (vs baseline delta=0)")
    print("=" * 80)

    baseline_types = results["all_d0"]["types"]
    for delta in deltas:
        if delta == 0:
            continue
        key = f"all_d{delta}"
        perturbed_types = results[key]["types"]
        n = min(len(baseline_types), len(perturbed_types))
        changed = sum(1 for i in range(n) if baseline_types[i] != perturbed_types[i])
        # What changed to what?
        transitions = defaultdict(int)
        for i in range(n):
            if baseline_types[i] != perturbed_types[i]:
                transitions[f"{baseline_types[i]}→{perturbed_types[i]}"] += 1
        trans_str = ", ".join(f"{k}:{v}" for k, v in sorted(transitions.items(), key=lambda x: -x[1]))
        print(f"  delta={delta:+.1f}: {changed}/{n} changed ({100*changed/n:.1f}%)  [{trans_str}]")

    # 3. Coordinate changes
    print("\n" + "=" * 80)
    print("COORDINATE CHANGES (vs baseline delta=0)")
    print("=" * 80)

    baseline_coords = results["all_d0"]["coords"]
    for delta in deltas:
        if delta == 0:
            continue
        key = f"all_d{delta}"
        perturbed_coords = results[key]["coords"]
        n = min(len(baseline_coords), len(perturbed_coords))
        dists = []
        changed = 0
        for i in range(n):
            c1 = baseline_coords[i]
            c2 = perturbed_coords[i]
            if c1 and c2 and c1 != c2:
                changed += 1
                dist = ((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2)**0.5
                dists.append(dist)
        avg_dist = sum(dists) / len(dists) if dists else 0
        print(f"  delta={delta:+.1f}: {changed}/{n} coords changed, avg_dist={avg_dist:.1f}px")

    # 4. Layer-specific perturbation
    print("\n" + "=" * 80)
    print("LAYER-SPECIFIC PERTURBATION (delta=±0.5)")
    print("=" * 80)

    for layer_name, _ in layer_configs[1:]:
        for delta in [-0.5, 0.5]:
            key = f"{layer_name}_d{delta}"
            types = results[key]["types"]
            if not types:
                continue
            n = min(len(baseline_types), len(types))
            changed = sum(1 for i in range(n) if baseline_types[i] != types[i])
            type_counts = defaultdict(int)
            for t in types:
                type_counts[t] += 1
            total = len(types)
            parts = [f"{k}={v}" for k, v in sorted(type_counts.items())]
            print(f"  {layer_name} d={delta:+.1f}: changed={changed}/{n}, types: {', '.join(parts)}")

    # 5. Show examples where type changed
    print("\n" + "=" * 80)
    print("EXAMPLES: ACTION TYPE CHANGED BY PERTURBATION")
    print("=" * 80)

    shown = 0
    for delta in [0.5, -0.5]:
        key = f"all_d{delta}"
        perturbed_types = results[key]["types"]
        perturbed_texts = results[key]["texts"]
        baseline_texts = results["all_d0"]["texts"]
        n = min(len(baseline_types), len(perturbed_types))
        for i in range(n):
            if baseline_types[i] != perturbed_types[i] and shown < 10:
                print(f"\n  Episode idx={i}: {baseline_types[i]} → {perturbed_types[i]} (delta={delta})")
                print(f"    Baseline: {baseline_texts[i][:120]}")
                print(f"    Perturbed: {perturbed_texts[i][:120]}")
                shown += 1


def merge_and_analyze(output_dir):
    """Load all shard files and run analysis."""
    import glob as globmod
    shard_files = sorted(globmod.glob(os.path.join(output_dir, "shard_*.json")))
    print(f"Found {len(shard_files)} shard files")

    # Merge results
    results = defaultdict(lambda: defaultdict(list))
    for sf in shard_files:
        with open(sf) as f:
            data = json.load(f)
        for key, val in data.items():
            results[key]["types"].extend(val["types"])
            results[key]["coords"].extend(val["coords"])
            results[key]["texts"].extend(val["texts"])

    # Parse deltas from keys
    deltas = sorted(set(float(k.split('_d')[1]) for k in results if k.startswith('all_d')))

    n_eps = len(results.get("all_d0.0", {}).get("types", []))
    if n_eps == 0:
        n_eps = len(results.get("all_d0", {}).get("types", []))
    print(f"Total: {n_eps} episodes")

    # Find the baseline key
    baseline_key = "all_d0.0" if "all_d0.0" in results else "all_d0"
    baseline_types = results[baseline_key]["types"]
    baseline_coords = results[baseline_key]["coords"]

    # 1. Action type distribution
    print("\n" + "=" * 80)
    print("ACTION TYPE DISTRIBUTION BY GATE DELTA (all layers)")
    print("=" * 80)
    for delta in deltas:
        key = f"all_d{delta}"
        types = results[key]["types"]
        if not types:
            continue
        type_counts = defaultdict(int)
        for t in types:
            type_counts[t] += 1
        total = len(types)
        parts = [f"{k}={v}({100*v/total:.1f}%)" for k, v in sorted(type_counts.items())]
        print(f"  delta={delta:+.1f}: {', '.join(parts)}")

    # 2. Changes vs baseline
    print("\n" + "=" * 80)
    print("ACTION TYPE CHANGES (vs baseline delta=0)")
    print("=" * 80)
    for delta in deltas:
        if delta == 0:
            continue
        key = f"all_d{delta}"
        perturbed_types = results[key]["types"]
        n = min(len(baseline_types), len(perturbed_types))
        changed = sum(1 for i in range(n) if baseline_types[i] != perturbed_types[i])
        transitions = defaultdict(int)
        for i in range(n):
            if baseline_types[i] != perturbed_types[i]:
                transitions[f"{baseline_types[i]}→{perturbed_types[i]}"] += 1
        trans_str = ", ".join(f"{k}:{v}" for k, v in sorted(transitions.items(), key=lambda x: -x[1]))
        print(f"  delta={delta:+.1f}: {changed}/{n} changed ({100*changed/n:.1f}%)  [{trans_str}]")

    # 3. Coordinate changes
    print("\n" + "=" * 80)
    print("COORDINATE CHANGES (vs baseline delta=0)")
    print("=" * 80)
    for delta in deltas:
        if delta == 0:
            continue
        key = f"all_d{delta}"
        perturbed_coords = results[key]["coords"]
        n = min(len(baseline_coords), len(perturbed_coords))
        dists = []
        changed = 0
        for i in range(n):
            c1 = baseline_coords[i]
            c2 = perturbed_coords[i]
            if c1 and c2 and c1 != c2:
                changed += 1
                dist = ((c1[0]-c2[0])**2 + (c1[1]-c2[1])**2)**0.5
                dists.append(dist)
        avg_dist = sum(dists)/len(dists) if dists else 0
        print(f"  delta={delta:+.1f}: {changed}/{n} coords changed, avg_dist={avg_dist:.1f}px")

    # 4. Layer-specific
    print("\n" + "=" * 80)
    print("LAYER-SPECIFIC PERTURBATION (delta=±0.5)")
    print("=" * 80)
    for layer_name in ["L10_only", "L18_only", "L27_only"]:
        for delta in [-0.5, 0.5]:
            key = f"{layer_name}_d{delta}"
            types = results[key]["types"]
            if not types:
                continue
            n = min(len(baseline_types), len(types))
            changed = sum(1 for i in range(n) if baseline_types[i] != types[i])
            type_counts = defaultdict(int)
            for t in types:
                type_counts[t] += 1
            total = len(types)
            parts = [f"{k}={v}" for k, v in sorted(type_counts.items())]
            print(f"  {layer_name} d={delta:+.1f}: changed={changed}/{n}, types: {', '.join(parts)}")

    # 5. Examples
    print("\n" + "=" * 80)
    print("EXAMPLES: ACTION TYPE CHANGED BY PERTURBATION")
    print("=" * 80)
    shown = 0
    for delta in [0.5, -0.5]:
        key = f"all_d{delta}"
        ptypes = results[key]["types"]
        ptexts = results[key]["texts"]
        btexts = results[baseline_key]["texts"]
        n = min(len(baseline_types), len(ptypes))
        for i in range(n):
            if baseline_types[i] != ptypes[i] and shown < 10:
                print(f"\n  idx={i}: {baseline_types[i]}→{ptypes[i]} (delta={delta})")
                print(f"    Baseline:  {btexts[i][:120]}")
                print(f"    Perturbed: {ptexts[i][:120]}")
                shown += 1


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == '--merge':
        merge_and_analyze(sys.argv[2])
    else:
        main()
