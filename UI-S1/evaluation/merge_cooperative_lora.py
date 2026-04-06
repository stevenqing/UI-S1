#!/usr/bin/env python3
"""
Merge cooperative LoRA weights (LoRA_V + LoRA_A) into base model.

For evaluation via vLLM: since vLLM can't do token-level routing,
we merge both adapters into the base weights:
  W_merged = W_base + scaling * (B_v @ A_v + B_a @ A_a)

This means all tokens get both adaptations (approximate but fast).

Usage:
  python evaluation/merge_cooperative_lora.py \
      --base_model checkpoints/Qwen2.5-VL-7B-Instruct \
      --coop_checkpoint train_GUI_360/llamafactory/output/cooperative_thought_v1/final \
      --output_dir checkpoints/cooperative_thought_v1_merged
"""

import argparse
import json
import os
import sys
import re

import torch

sys.stdout.reconfigure(line_buffering=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", required=True)
    parser.add_argument("--coop_checkpoint", required=True,
                        help="Directory with lora_v.pt, lora_a.pt, cooperative_config.json")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--merge_mode", choices=["both", "a_only"], default="both",
                        help="both: merge LoRA_V+LoRA_A; a_only: merge only LoRA_A (better approx)")
    args = parser.parse_args()

    # Load cooperative config
    config_path = os.path.join(args.coop_checkpoint, "cooperative_config.json")
    with open(config_path) as f:
        coop_config = json.load(f)
    print(f"Cooperative config: {coop_config}")

    # Load LoRA weights
    print("Loading LoRA_V weights...")
    lora_v = torch.load(
        os.path.join(args.coop_checkpoint, "lora_v.pt"),
        map_location="cpu", weights_only=True)
    print(f"  LoRA_V keys: {len(lora_v)}")

    print("Loading LoRA_A weights...")
    lora_a = torch.load(
        os.path.join(args.coop_checkpoint, "lora_a.pt"),
        map_location="cpu", weights_only=True)
    print(f"  LoRA_A keys: {len(lora_a)}")

    # Load base model state dict
    from safetensors.torch import load_file, save_file
    from glob import glob

    shard_files = sorted(glob(os.path.join(args.base_model, "model-*.safetensors")))
    print(f"Loading base model from {len(shard_files)} shards...")

    # Build mapping: LoRA key -> base model key + delta
    # LoRA keys look like:
    #   base_model.model.language_model.layers.0.self_attn.q_proj.lora_A_v
    #   base_model.model.language_model.layers.0.self_attn.q_proj.lora_B_v
    # Base model keys look like:
    #   model.language_model.layers.0.self_attn.q_proj.weight

    # Parse LoRA weights into per-module deltas
    # Group by module path (everything before .lora_A_v/.lora_B_v etc.)
    deltas = {}  # base_key -> delta tensor

    def parse_lora_weights(lora_state, suffix):
        """Parse lora_v or lora_a state dict into per-module A and B."""
        modules = {}  # module_path -> {"A": tensor, "B": tensor}
        for key, val in lora_state.items():
            # key: base_model.model.language_model.layers.X.self_attn.Y_proj.lora_A_v
            # Extract module path and A/B
            if f"lora_A_{suffix}" in key:
                module_path = key.replace(f".lora_A_{suffix}", "")
                modules.setdefault(module_path, {})["A"] = val
            elif f"lora_B_{suffix}" in key:
                module_path = key.replace(f".lora_B_{suffix}", "")
                modules.setdefault(module_path, {})["B"] = val
        return modules

    a_modules = parse_lora_weights(lora_a, "a")
    print(f"  LoRA_A modules: {len(a_modules)}")

    if args.merge_mode == "both":
        v_modules = parse_lora_weights(lora_v, "v")
        print(f"  LoRA_V modules: {len(v_modules)}")
        all_module_paths = set(v_modules.keys()) | set(a_modules.keys())
    else:
        v_modules = {}
        all_module_paths = set(a_modules.keys())
        print(f"  merge_mode=a_only: skipping LoRA_V")

    for module_path in sorted(all_module_paths):
        base_key = module_path.replace("base_model.", "") + ".weight"

        r = None
        if module_path in a_modules:
            r = a_modules[module_path]["A"].shape[0]
        elif module_path in v_modules:
            r = v_modules[module_path]["A"].shape[0]

        alpha = 32  # default
        scaling = alpha / r if r else 1.0

        delta = torch.zeros(1)

        if module_path in v_modules:
            A_v = v_modules[module_path]["A"].float()
            B_v = v_modules[module_path]["B"].float()
            delta = B_v @ A_v

        if module_path in a_modules:
            A_a = a_modules[module_path]["A"].float()
            B_a = a_modules[module_path]["B"].float()
            if delta.dim() == 1:
                delta = B_a @ A_a
            else:
                delta = delta + B_a @ A_a

        delta = delta * scaling
        deltas[base_key] = delta

    print(f"Computed {len(deltas)} weight deltas")

    # Apply deltas to base model and save
    os.makedirs(args.output_dir, exist_ok=True)

    # Copy non-weight files from base model
    import shutil
    for fname in os.listdir(args.base_model):
        if fname.endswith(".safetensors"):
            continue
        src = os.path.join(args.base_model, fname)
        dst = os.path.join(args.output_dir, fname)
        if os.path.isfile(src) and not os.path.exists(dst):
            shutil.copy2(src, dst)

    # Load each shard, apply deltas, save
    # First, load index to know which keys are in which shard
    index_path = os.path.join(args.base_model, "model.safetensors.index.json")
    if os.path.exists(index_path):
        with open(index_path) as f:
            index = json.load(f)
        weight_map = index["weight_map"]
    else:
        # Single file
        weight_map = None

    applied = set()
    for shard_file in shard_files:
        shard_name = os.path.basename(shard_file)
        print(f"Processing {shard_name}...")
        shard = load_file(shard_file)

        modified = False
        for key in list(shard.keys()):
            if key in deltas:
                original = shard[key].float()
                shard[key] = (original + deltas[key]).to(shard[key].dtype)
                applied.add(key)
                modified = True

        out_path = os.path.join(args.output_dir, shard_name)
        save_file(shard, out_path)

    # Copy index file
    if os.path.exists(index_path):
        shutil.copy2(index_path, os.path.join(args.output_dir, "model.safetensors.index.json"))

    print(f"\nApplied deltas to {len(applied)}/{len(deltas)} modules")
    unapplied = set(deltas.keys()) - applied
    if unapplied:
        print(f"WARNING: {len(unapplied)} deltas not applied:")
        for k in sorted(unapplied):
            print(f"  {k}")

    print(f"\nMerged model saved to {args.output_dir}")


if __name__ == "__main__":
    main()
