#!/usr/bin/env python3
"""Merge V12 Soft Cooperative LoRA into base model + save correction adapters.

Two-step lossless merge for vLLM serving:
  1. Merge r=0.5 approximation into base weights:
       W_merged = W_base + scaling * B @ (0.5 * A_1 + 0.5 * A_2)
  2. Save correction adapters for exact reconstruction via hooks:
       correction(x) = scaling * B @ ((sigmoid(x @ w_route) - 0.5) * A_diff @ x)
       where A_diff = A_1 - A_2

Math proof:
  actual_delta = scaling * B @ (r * A_1 @ x + (1-r) * A_2 @ x)
               = scaling * B @ (0.5*(A_1+A_2) @ x + (r-0.5)*(A_1-A_2) @ x)
               = merged_delta + correction

Usage:
    python v12_general/merge_to_hf.py \
        --base_model checkpoints/Qwen2.5-VL-7B-Instruct \
        --coop_checkpoint checkpoints/v12_sft/cooperative_final \
        --output_dir /tmp/v12_merged_lossless
"""

import argparse
import json
import os
import shutil
import sys

import torch

sys.stdout.reconfigure(line_buffering=True)


def lora_path_to_base_key(module_path: str) -> str:
    """Convert LoRA module path to base model weight key."""
    p = module_path
    if p.startswith("base_model."):
        p = p[len("base_model."):]
    p = p.replace("model.language_model.", "model.")
    return p + ".weight"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", required=True)
    parser.add_argument("--coop_checkpoint", required=True,
                        help="V12 cooperative checkpoint dir")
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    # Load cooperative config
    config_path = os.path.join(args.coop_checkpoint, "cooperative_config.json")
    with open(config_path) as f:
        coop_config = json.load(f)

    lora_r = coop_config.get("lora_r", 256)
    alpha = coop_config.get("lora_alpha", lora_r * 2)
    scaling = alpha / lora_r
    target_modules = coop_config.get("target_modules",
                                      ["q_proj", "k_proj", "v_proj", "o_proj"])
    print(f"LoRA r={lora_r}, alpha={alpha}, scaling={scaling}")
    print(f"Target modules: {target_modules}")

    # Load LoRA weights
    print("Loading lora_weights.pt...")
    lora = torch.load(
        os.path.join(args.coop_checkpoint, "lora_weights.pt"),
        map_location="cpu", weights_only=True)
    print(f"  {len(lora)} tensors")

    # Load route weights
    print("Loading route_weights.pt...")
    route_raw = torch.load(
        os.path.join(args.coop_checkpoint, "route_weights.pt"),
        map_location="cpu", weights_only=True)
    print(f"  {len(route_raw)} tensors")

    # Group LoRA by module: module_path -> {lora_A_1, lora_A_2, lora_B}
    modules = {}
    for key, val in lora.items():
        for suffix in ("lora_A_1", "lora_A_2", "lora_B"):
            if key.endswith("." + suffix):
                module_path = key[: -(len(suffix) + 1)]
                modules.setdefault(module_path, {})[suffix] = val
                break
    print(f"  {len(modules)} LoRA modules")

    # ── Step 1: Compute r=0.5 merge deltas ──────────────────────────────
    deltas = {}
    for module_path, tensors in sorted(modules.items()):
        A_1 = tensors["lora_A_1"].float()
        A_2 = tensors["lora_A_2"].float()
        B = tensors["lora_B"].float()
        A_merged = 0.5 * A_1 + 0.5 * A_2
        delta = (B @ A_merged) * scaling
        base_key = lora_path_to_base_key(module_path)
        deltas[base_key] = delta
    print(f"Computed {len(deltas)} merge deltas")

    # ── Step 2: Save correction adapters ────────────────────────────────
    # Per module: A_diff = A_1 - A_2, B (for the correction)
    # Per layer: w_route (routing vector)
    corrections = {}
    for module_path, tensors in sorted(modules.items()):
        base_key = lora_path_to_base_key(module_path)
        A_diff = tensors["lora_A_1"] - tensors["lora_A_2"]  # [lora_r, in_f]
        B = tensors["lora_B"]  # [out_f, lora_r]
        corrections[base_key + ".A_diff"] = A_diff.to(torch.bfloat16)
        corrections[base_key + ".B"] = B.to(torch.bfloat16)

    # Route weights: route_weights.{layer_idx} -> [hidden_size]
    for key, val in route_raw.items():
        corrections[key] = val.to(torch.bfloat16)

    print(f"Correction adapters: {len(corrections)} tensors")
    corr_size_mb = sum(v.numel() * v.element_size() for v in corrections.values()) / 1e6
    print(f"Correction size: {corr_size_mb:.1f} MB")

    # ── Step 3: Apply merge deltas to base model ────────────────────────
    from safetensors.torch import load_file, save_file
    from glob import glob

    os.makedirs(args.output_dir, exist_ok=True)

    # Copy non-weight files
    for fname in os.listdir(args.base_model):
        if fname.endswith(".safetensors"):
            continue
        src = os.path.join(args.base_model, fname)
        dst = os.path.join(args.output_dir, fname)
        if os.path.isfile(src) and not os.path.exists(dst):
            shutil.copy2(src, dst)

    # Copy index
    index_path = os.path.join(args.base_model, "model.safetensors.index.json")
    if os.path.exists(index_path):
        shutil.copy2(index_path,
                      os.path.join(args.output_dir, "model.safetensors.index.json"))

    shard_files = sorted(glob(os.path.join(args.base_model, "model-*.safetensors")))
    print(f"Processing {len(shard_files)} base model shards...")

    applied = set()
    for shard_file in shard_files:
        shard_name = os.path.basename(shard_file)
        shard = load_file(shard_file)
        n_applied = 0
        for key in list(shard.keys()):
            if key in deltas:
                original = shard[key].float()
                shard[key] = (original + deltas[key]).to(shard[key].dtype)
                applied.add(key)
                n_applied += 1
        save_file(shard, os.path.join(args.output_dir, shard_name))
        print(f"  {shard_name}: {n_applied} deltas applied")

    unapplied = set(deltas.keys()) - applied
    if unapplied:
        print(f"WARNING: {len(unapplied)} deltas not applied:")
        for k in sorted(unapplied):
            print(f"  {k}")

    # ── Step 4: Save correction adapters and config ─────────────────────
    torch.save(corrections,
               os.path.join(args.output_dir, "v12_correction_adapters.pt"))

    corr_config = {
        "type": "v12_soft_cooperative",
        "lora_r": lora_r,
        "lora_alpha": alpha,
        "scaling": scaling,
        "target_modules": target_modules,
        "merge_routing_value": 0.5,
        "num_layers": len(route_raw),
        "num_modules": len(modules),
    }
    with open(os.path.join(args.output_dir, "v12_correction_config.json"), "w") as f:
        json.dump(corr_config, f, indent=2)

    print(f"\nMerged model + corrections saved to {args.output_dir}")
    print(f"  Base weights: r=0.5 merge applied to {len(applied)} modules")
    print(f"  Corrections:  v12_correction_adapters.pt ({corr_size_mb:.1f} MB)")
    print(f"  Config:       v12_correction_config.json")


if __name__ == "__main__":
    main()
