#!/usr/bin/env python3
"""Convert PEFT standard LoRA checkpoint to cooperative LoRA (V13) format.

PEFT standard LoRA has:
  adapter_model.safetensors with keys like:
    base_model.model.model.layers.{L}.self_attn.{mod}.lora_A.weight  [r, in_f]
    base_model.model.model.layers.{L}.self_attn.{mod}.lora_B.weight  [out_f, r]

Cooperative LoRA (V13) has:
  lora_weights.pt with keys like:
    base_model.model.language_model.layers.{L}.self_attn.{mod}.lora_A_1  [r, in_f]
    base_model.model.language_model.layers.{L}.self_attn.{mod}.lora_A_2  [r, in_f]
    base_model.model.language_model.layers.{L}.self_attn.{mod}.lora_B    [out_f, r]
  route_weights.pt: zeros → sigmoid(0) = 0.5
  comm_weights.pt: Kaiming W, zero gates
  cooperative_config.json

Conversion:
  B unchanged
  A_1 = A + δ,  A_2 = A - δ   (δ = noise for expert diversity)
  At init (route=0.5): delta = B @ (0.5*A_1 + 0.5*A_2) @ x * scaling = B @ A @ x * scaling

Usage:
    python v15_gui_360/convert_peft_to_cooperative.py \
        --peft_checkpoint train_GUI_360/moe_rl/extracted_lora_fullsft_r128 \
        --output checkpoints/v15_peft_to_cooperative_r128 \
        --num_comm_rounds 2 --noise_scale 0.1
"""

import argparse
import json
import math
import os
import re

import torch
from safetensors.torch import load_file


def log(msg=""):
    print(msg, flush=True)


NUM_LAYERS = 28
HIDDEN_SIZE = 3584


def convert(
    peft_dir: str,
    output_dir: str,
    num_comm_rounds: int = 2,
    noise_scale: float = 0.1,
):
    log(f"{'=' * 60}")
    log(f"  Convert PEFT Standard LoRA → Cooperative LoRA (V13)")
    log(f"{'=' * 60}")
    log(f"  PEFT checkpoint: {peft_dir}")
    log(f"  Output:          {output_dir}")
    log(f"  Noise scale:     {noise_scale}")
    log(f"  Comm rounds:     {num_comm_rounds}")
    log()

    # Load PEFT config
    config_path = os.path.join(peft_dir, "adapter_config.json")
    with open(config_path) as f:
        peft_config = json.load(f)

    rank = peft_config["r"]
    alpha = peft_config["lora_alpha"]
    target_modules = peft_config["target_modules"]
    log(f"  Rank: {rank}, Alpha: {alpha}")
    log(f"  Target modules: {target_modules}")
    log()

    # Load PEFT weights
    safetensors_path = os.path.join(peft_dir, "adapter_model.safetensors")
    peft_state = load_file(safetensors_path)
    log(f"  Loaded {len(peft_state)} PEFT tensors")

    # Parse PEFT keys and convert
    # PEFT key: base_model.model.model.layers.{L}.{submod}.{mod}.lora_{A|B}.weight
    # Coop key: base_model.model.language_model.layers.{L}.{submod}.{mod}.lora_{A_1|A_2|B}
    peft_pattern = re.compile(
        r"base_model\.model\.model\.layers\.(\d+)\.(self_attn|mlp)\.(\w+)\.lora_(A|B)\.weight"
    )

    # Group by module
    modules = {}
    for key in peft_state:
        m = peft_pattern.match(key)
        if m:
            layer, submod, mod, ab = m.groups()
            mod_key = (int(layer), submod, mod)
            modules.setdefault(mod_key, {})[ab] = key
        else:
            log(f"  WARNING: unrecognized key: {key}")

    log(f"  Found {len(modules)} LoRA modules")

    # Convert
    coop_state = {}
    for (layer, submod, mod), keys in sorted(modules.items()):
        if "A" not in keys or "B" not in keys:
            log(f"  WARNING: incomplete module L{layer}.{submod}.{mod}, skipping")
            continue

        A = peft_state[keys["A"]]  # [r, in_f]
        B = peft_state[keys["B"]]  # [out_f, r]

        # Cooperative key prefix uses language_model (for Qwen2.5-VL)
        coop_prefix = f"base_model.model.language_model.layers.{layer}.{submod}.{mod}"

        # B unchanged
        coop_state[f"{coop_prefix}.lora_B"] = B

        # A → A_1, A_2 with perturbation
        if noise_scale > 0:
            gen = torch.Generator()
            gen.manual_seed(42 + layer * len(target_modules))
            A_float = A.float()
            noise = torch.randn(A_float.shape, generator=gen, dtype=torch.float32)
            r, in_f = A_float.shape
            a_norm = torch.norm(A_float)
            delta = noise_scale * noise * (a_norm / math.sqrt(r * in_f))
            A_1 = (A_float + delta).to(A.dtype)
            A_2 = (A_float - delta).to(A.dtype)
        else:
            A_1 = A.clone()
            A_2 = A.clone()

        coop_state[f"{coop_prefix}.lora_A_1"] = A_1
        coop_state[f"{coop_prefix}.lora_A_2"] = A_2

    log(f"  Converted {len(coop_state)} cooperative tensors "
        f"({len(coop_state) // 3} modules × 3)")

    # Route weights: zeros → sigmoid(0) = 0.5
    dtype = torch.bfloat16
    route_state = {}
    for layer_idx in range(NUM_LAYERS):
        route_state[f"route_weights.{layer_idx}"] = torch.zeros(HIDDEN_SIZE, dtype=dtype)

    # Comm weights: Kaiming W, zero gates
    T = num_comm_rounds
    comm_state = {}
    for idx in range(NUM_LAYERS * T):
        w_12 = torch.zeros(rank, rank)
        w_21 = torch.zeros(rank, rank)
        torch.nn.init.kaiming_uniform_(w_12, a=math.sqrt(5))
        torch.nn.init.kaiming_uniform_(w_21, a=math.sqrt(5))
        comm_state[f"comm_W_12.{idx}"] = w_12.to(dtype)
        comm_state[f"comm_W_21.{idx}"] = w_21.to(dtype)
        comm_state[f"comm_gate_12.{idx}"] = torch.zeros(rank, dtype=dtype)
        comm_state[f"comm_gate_21.{idx}"] = torch.zeros(rank, dtype=dtype)

    # Save
    os.makedirs(output_dir, exist_ok=True)
    torch.save(coop_state, os.path.join(output_dir, "lora_weights.pt"))
    torch.save(route_state, os.path.join(output_dir, "route_weights.pt"))
    torch.save(comm_state, os.path.join(output_dir, "comm_weights.pt"))

    config = {
        "lora_r": rank,
        "lora_alpha": alpha,
        "target_modules": target_modules,
        "balance_weight": 0.01,
        "num_comm_rounds": num_comm_rounds,
        "type": "iterative_cooperative_v13",
        "conversion_info": {
            "source": "peft_standard_lora",
            "peft_checkpoint": os.path.abspath(peft_dir),
            "noise_scale": noise_scale,
        },
    }
    with open(os.path.join(output_dir, "cooperative_config.json"), "w") as f:
        json.dump(config, f, indent=2)

    # Stats
    lora_params = sum(t.numel() for t in coop_state.values())
    route_params = sum(t.numel() for t in route_state.values())
    comm_params = sum(t.numel() for t in comm_state.values())

    log()
    log(f"  Saved to: {output_dir}")
    log(f"    lora_weights.pt   ({len(coop_state)} tensors, {lora_params/1e6:.1f}M params)")
    log(f"    route_weights.pt  ({len(route_state)} tensors)")
    log(f"    comm_weights.pt   ({len(comm_state)} tensors)")
    log(f"    cooperative_config.json")
    log(f"  Total parameters: {(lora_params + route_params + comm_params)/1e6:.1f}M")
    log()
    log(f"  Key sample: {next(iter(coop_state))}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert PEFT standard LoRA to cooperative LoRA (V13)")
    parser.add_argument("--peft_checkpoint", required=True,
                        help="PEFT checkpoint dir (with adapter_model.safetensors)")
    parser.add_argument("--output", required=True,
                        help="Output dir for cooperative checkpoint")
    parser.add_argument("--num_comm_rounds", type=int, default=2)
    parser.add_argument("--noise_scale", type=float, default=0.1)
    args = parser.parse_args()

    convert(args.peft_checkpoint, args.output,
            num_comm_rounds=args.num_comm_rounds,
            noise_scale=args.noise_scale)
