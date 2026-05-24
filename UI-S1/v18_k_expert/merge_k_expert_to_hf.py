#!/usr/bin/env python3
"""Merge K-expert cooperative LoRA into base model for vLLM serving.

For uniform routing (pre-RL or route_weights=0), the output is:
  delta = B @ mean(A_0, ..., A_{K-1}) @ x * (alpha/r) = B @ A_avg @ x * s

This is identical to standard weight merging: W_merged = W_base + B @ A_avg * s

For post-RL checkpoints with learned routing, this is an approximation
(exact when routing is uniform, good when routing is near-uniform).

Usage:
    python v18_k_expert/merge_k_expert_to_hf.py \
        --base_model checkpoints/Qwen2.5-VL-7B-Instruct \
        --coop_checkpoint checkpoints/v18_svd_k4_extracted \
        --output_dir checkpoints/v18_k4_merged
"""

import argparse
import json
import math
import os
import sys
import time

import torch
from safetensors.torch import load_file
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor


def log(msg=""):
    print(msg, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", required=True)
    parser.add_argument("--coop_checkpoint", required=True,
                        help="Path to cooperative checkpoint dir")
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    t0 = time.time()

    # Load cooperative config
    config_path = os.path.join(args.coop_checkpoint, "cooperative_config.json")
    with open(config_path) as f:
        coop_config = json.load(f)

    K = coop_config.get("num_experts", 2)
    r = coop_config["lora_r"]
    alpha = coop_config["lora_alpha"]
    scaling = alpha / r
    target_modules = coop_config.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"])
    coop_type = coop_config.get("type", "unknown")

    log(f"{'=' * 60}")
    log(f"  Merge K-Expert Cooperative LoRA → HuggingFace Model")
    log(f"{'=' * 60}")
    log(f"  Base model:     {args.base_model}")
    log(f"  Coop checkpoint: {args.coop_checkpoint}")
    log(f"  Output:         {args.output_dir}")
    log(f"  Type: {coop_type}, K={K}, r={r}, alpha={alpha}, scaling={scaling}")
    log(f"  Target modules: {target_modules}")
    log()

    # Load base model on CPU
    log("Loading base model on CPU...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16, device_map="cpu",
        trust_remote_code=True,
    )
    log(f"  Loaded ({time.time() - t0:.1f}s)")

    # Load LoRA weights
    log("Loading cooperative LoRA weights...")
    lora_path = os.path.join(args.coop_checkpoint, "lora_weights.pt")
    lora_state = torch.load(lora_path, map_location="cpu", weights_only=True)
    log(f"  Loaded {len(lora_state)} tensors")

    # Detect key format and parse
    sample_key = next(iter(lora_state))
    log(f"  Key format sample: {sample_key}")

    # Build per-module delta and merge
    # Key patterns:
    #   V15: ...layers.{L}.self_attn.{mod}.lora_A_1 / lora_A_2 / lora_B
    #   V18: ...layers.{L}.self_attn.{mod}.lora_A.{k} / lora_B

    # Find all B keys to enumerate modules
    b_keys = [k for k in lora_state if k.endswith(".lora_B")]
    log(f"  Found {len(b_keys)} modules with lora_B")

    # Locate transformer layers
    vlm = model.model
    if hasattr(vlm, "language_model"):
        layers = vlm.language_model.layers
        prefix_in_model = "model.language_model"
    elif hasattr(vlm, "layers"):
        layers = vlm.layers
        prefix_in_model = "model"
    else:
        raise AttributeError(f"Cannot find transformer layers")

    merged_count = 0
    for b_key in sorted(b_keys):
        # Parse layer index and module name from key
        # e.g. "base_model.model.language_model.layers.0.self_attn.q_proj.lora_B"
        module_path = b_key.rsplit(".lora_B", 1)[0]

        # Find corresponding A matrices
        if coop_type == "k_expert_cooperative_v18":
            # V18: lora_A.0, lora_A.1, ..., lora_A.{K-1}
            a_keys = [f"{module_path}.lora_A.{k}" for k in range(K)]
        elif coop_type == "iterative_cooperative_v13":
            # V15: lora_A_1, lora_A_2
            a_keys = [f"{module_path}.lora_A_1", f"{module_path}.lora_A_2"]
        else:
            # Auto-detect
            a_keys = sorted([k for k in lora_state if k.startswith(module_path + ".lora_A")])

        # Verify all A keys exist
        missing = [k for k in a_keys if k not in lora_state]
        if missing:
            log(f"  WARNING: Missing A keys for {module_path}: {missing}")
            continue

        B = lora_state[b_key]  # [out_f, r]
        As = [lora_state[k] for k in a_keys]  # K × [r, in_f]
        A_avg = sum(As) / len(As)  # [r, in_f]

        # delta = B @ A_avg * scaling
        dtype = B.dtype
        delta = (B.float() @ A_avg.float()) * scaling
        delta = delta.to(dtype)

        # Find the actual weight in the model
        # Parse "layers.{L}.self_attn.{mod}" from the key
        # Key: ...layers.{L}.self_attn.{mod}.lora_B
        parts = module_path.split(".")
        # Find "layers" and get index
        layer_idx = None
        mod_name = None
        for i, p in enumerate(parts):
            if p == "layers" and i + 1 < len(parts):
                layer_idx = int(parts[i + 1])
            if p in ("self_attn", "mlp") and i + 1 < len(parts):
                parent_type = p
                mod_name = parts[i + 1]

        if layer_idx is None or mod_name is None:
            log(f"  WARNING: Cannot parse {module_path}")
            continue

        # Get the weight tensor
        layer = layers[layer_idx]
        if parent_type == "self_attn":
            weight = getattr(layer.self_attn, mod_name).weight
        elif parent_type == "mlp":
            weight = getattr(layer.mlp, mod_name).weight
        else:
            log(f"  WARNING: Unknown parent {parent_type}")
            continue

        # Merge
        weight.data += delta
        merged_count += 1

    log(f"\n  Merged {merged_count} modules")

    # Save merged model
    log(f"\nSaving merged model to {args.output_dir}...")
    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir)

    # Copy processor/tokenizer
    log("Copying processor/tokenizer...")
    processor = AutoProcessor.from_pretrained(args.base_model, trust_remote_code=True)
    processor.save_pretrained(args.output_dir)

    # Save merge info
    merge_info = {
        "base_model": os.path.abspath(args.base_model),
        "coop_checkpoint": os.path.abspath(args.coop_checkpoint),
        "coop_type": coop_type,
        "num_experts": K,
        "lora_r": r,
        "lora_alpha": alpha,
        "merge_method": "uniform_routing_mean_A",
        "merged_modules": merged_count,
    }
    with open(os.path.join(args.output_dir, "merge_info.json"), "w") as f:
        json.dump(merge_info, f, indent=2)

    log(f"\nDone in {time.time() - t0:.1f}s")
    log(f"Output: {args.output_dir}")


if __name__ == "__main__":
    main()
