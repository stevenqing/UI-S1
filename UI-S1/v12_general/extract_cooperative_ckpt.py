#!/usr/bin/env python3
"""Extract V12 cooperative weights from HF Trainer checkpoint.

HF Trainer saves all parameters (frozen base + trainable lora/route) in a
single model.safetensors. This script extracts lora + route weights into
the cooperative_final format used by merge_to_hf.py.

Usage:
    python v12_general/extract_cooperative_ckpt.py \
        --hf_checkpoint checkpoints/v12_sft/checkpoint-150 \
        --output_dir checkpoints/v12_sft/cooperative_ep2
"""

import argparse
import json
import os
import sys

import torch

sys.stdout.reconfigure(line_buffering=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf_checkpoint", required=True,
                        help="HF Trainer checkpoint dir with model.safetensors")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--lora_r", type=int, default=256)
    parser.add_argument("--lora_alpha", type=int, default=512)
    args = parser.parse_args()

    from safetensors.torch import load_file

    ckpt_path = os.path.join(args.hf_checkpoint, "model.safetensors")
    print(f"Loading {ckpt_path}...")
    sd = load_file(ckpt_path)
    print(f"  {len(sd)} tensors")

    lora_weights = {}
    route_weights = {}
    target_modules = set()

    for key, val in sd.items():
        if key.startswith("route_weights."):
            route_weights[key] = val
        elif any(s in key for s in ("lora_A_1", "lora_A_2", "lora_B")):
            lora_weights[key] = val
            # Extract target module name (q_proj, k_proj, etc.)
            for mod in ("q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"):
                if mod in key:
                    target_modules.add(mod)

    print(f"  LoRA tensors: {len(lora_weights)}")
    print(f"  Route tensors: {len(route_weights)}")
    print(f"  Target modules: {sorted(target_modules)}")

    os.makedirs(args.output_dir, exist_ok=True)

    torch.save(lora_weights, os.path.join(args.output_dir, "lora_weights.pt"))
    torch.save(route_weights, os.path.join(args.output_dir, "route_weights.pt"))

    config = {
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "target_modules": sorted(target_modules),
        "balance_weight": 0.01,
        "type": "soft_cooperative_v12",
    }
    with open(os.path.join(args.output_dir, "cooperative_config.json"), "w") as f:
        json.dump(config, f, indent=2)

    lora_mb = sum(v.numel() * v.element_size() for v in lora_weights.values()) / 1e6
    route_mb = sum(v.numel() * v.element_size() for v in route_weights.values()) / 1e6
    print(f"\nSaved to {args.output_dir}")
    print(f"  lora_weights.pt: {lora_mb:.1f} MB")
    print(f"  route_weights.pt: {route_mb:.3f} MB")
    print(f"  cooperative_config.json")


if __name__ == "__main__":
    main()
