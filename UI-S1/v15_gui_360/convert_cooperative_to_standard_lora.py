#!/usr/bin/env python3
"""Convert cooperative LoRA checkpoint to standard LoRA by averaging experts.

Cooperative LoRA has dual A matrices (A_1, A_2) + shared B.
Standard LoRA has single A + B.

Conversion: A = 0.5 * (A_1 + A_2), B unchanged.
At init (route_weight=0 → sigmoid=0.5), this is exact.

Usage:
    python v15_gui_360/convert_cooperative_to_standard_lora.py \
        --coop_checkpoint checkpoints/v15_svd_extracted_cooperative \
        --output checkpoints/v15_svd_extracted_standard_lora
"""

import argparse
import json
import os
import re

import torch


def convert(coop_dir: str, output_dir: str):
    print(f"Loading cooperative checkpoint: {coop_dir}")

    # Load cooperative config
    config_path = os.path.join(coop_dir, "cooperative_config.json")
    with open(config_path) as f:
        coop_config = json.load(f)

    # Load cooperative LoRA weights
    lora_state = torch.load(
        os.path.join(coop_dir, "lora_weights.pt"),
        map_location="cpu", weights_only=True,
    )
    print(f"  Loaded {len(lora_state)} cooperative LoRA tensors")

    # Group by module: find all (A_1, A_2, B) triples
    # Key format: base_model.model.language_model.layers.{L}.self_attn.{mod}.lora_{A_1|A_2|B}
    modules = {}
    for key in lora_state:
        if key.endswith(".lora_A_1"):
            prefix = key[:-len(".lora_A_1")]
            modules.setdefault(prefix, {})["A_1"] = key
        elif key.endswith(".lora_A_2"):
            prefix = key[:-len(".lora_A_2")]
            modules.setdefault(prefix, {})["A_2"] = key
        elif key.endswith(".lora_B"):
            prefix = key[:-len(".lora_B")]
            modules.setdefault(prefix, {})["B"] = key

    # Convert: average A_1, A_2 → A
    std_state = {}
    for prefix, keys in sorted(modules.items()):
        if "A_1" not in keys or "A_2" not in keys or "B" not in keys:
            print(f"  WARNING: incomplete module {prefix}, skipping")
            continue

        A_1 = lora_state[keys["A_1"]]
        A_2 = lora_state[keys["A_2"]]
        B = lora_state[keys["B"]]

        # Average the two experts
        A = (A_1.float() + A_2.float()) / 2.0
        A = A.to(A_1.dtype)

        std_state[f"{prefix}.lora_A"] = A
        std_state[f"{prefix}.lora_B"] = B

    print(f"  Converted {len(std_state)} standard LoRA tensors "
          f"({len(std_state) // 2} modules)")

    # Save
    os.makedirs(output_dir, exist_ok=True)

    torch.save(std_state, os.path.join(output_dir, "lora_weights.pt"))

    std_config = {
        "lora_r": coop_config["lora_r"],
        "lora_alpha": coop_config["lora_alpha"],
        "target_modules": coop_config["target_modules"],
        "type": "standard_lora_baseline",
    }
    with open(os.path.join(output_dir, "lora_config.json"), "w") as f:
        json.dump(std_config, f, indent=2)

    total_params = sum(t.numel() for t in std_state.values())
    print(f"  Total parameters: {total_params:,} ({total_params / 1e6:.1f}M)")
    print(f"  Saved to: {output_dir}")
    print(f"    - lora_weights.pt ({len(std_state)} tensors)")
    print(f"    - lora_config.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert cooperative LoRA to standard LoRA (average experts)")
    parser.add_argument("--coop_checkpoint", required=True,
                        help="Cooperative checkpoint dir (with lora_weights.pt)")
    parser.add_argument("--output", required=True,
                        help="Output dir for standard LoRA checkpoint")
    args = parser.parse_args()

    convert(args.coop_checkpoint, args.output)
