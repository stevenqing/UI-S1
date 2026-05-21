#!/usr/bin/env python3
"""Merge PEFT-format LoRA (adapter_model.safetensors) into base model.

Applies delta = B @ A * (alpha/r) to each target weight and saves
a clean full model with processor/tokenizer.

Usage:
    python v15_gui_360/merge_peft_lora.py \
        --base_model checkpoints/Qwen2.5-VL-7B-Instruct \
        --peft_checkpoint train_GUI_360/moe_rl/extracted_lora_fullsft_r128 \
        --output_dir checkpoints/merged_model
"""

import argparse
import json
import os
import re

import torch
from safetensors.torch import load_file
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", required=True)
    parser.add_argument("--peft_checkpoint", required=True,
                        help="PEFT dir with adapter_model.safetensors + adapter_config.json")
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    # Load PEFT config
    with open(os.path.join(args.peft_checkpoint, "adapter_config.json")) as f:
        cfg = json.load(f)
    rank = cfg["r"]
    alpha = cfg["lora_alpha"]
    scaling = alpha / rank
    target_modules = cfg["target_modules"]
    print(f"PEFT config: r={rank}, alpha={alpha}, scaling={scaling}")
    print(f"Target modules: {target_modules}")

    # Load PEFT weights
    peft_state = load_file(os.path.join(args.peft_checkpoint, "adapter_model.safetensors"))
    print(f"Loaded {len(peft_state)} PEFT tensors")

    # Load base model
    print(f"Loading base model: {args.base_model}")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16, device_map="cpu"
    )

    # Parse PEFT keys and apply deltas
    # PEFT key: base_model.model.model.layers.{L}.{submod}.{mod}.lora_{A|B}.weight
    # HF key:   model.layers.{L}.{submod}.{mod}.weight
    peft_pattern = re.compile(
        r"base_model\.model\.model\.layers\.(\d+)\.(self_attn|mlp)\.(\w+)\.lora_(A|B)\.weight"
    )

    # Group A/B pairs
    modules = {}
    for key in peft_state:
        m = peft_pattern.match(key)
        if m:
            layer, submod, mod, ab = m.groups()
            mod_key = f"model.layers.{layer}.{submod}.{mod}.weight"
            modules.setdefault(mod_key, {})[ab] = key

    print(f"Found {len(modules)} LoRA module pairs")

    # Apply delta = B @ A * scaling
    merged = 0
    state_dict = model.state_dict()
    for hf_key, ab_keys in sorted(modules.items()):
        if "A" not in ab_keys or "B" not in ab_keys:
            print(f"  WARNING: incomplete pair for {hf_key}, skipping")
            continue

        A = peft_state[ab_keys["A"]].float()  # [r, in_f]
        B = peft_state[ab_keys["B"]].float()  # [out_f, r]
        delta = (B @ A) * scaling

        # Find the weight in model state dict
        # For VL models, the actual key might have language_model prefix
        actual_key = None
        for candidate in [hf_key, hf_key.replace("model.layers", "model.language_model.layers")]:
            if candidate in state_dict:
                actual_key = candidate
                break

        if actual_key is None:
            print(f"  WARNING: {hf_key} not found in model, skipping")
            continue

        param = state_dict[actual_key]
        param.data += delta.to(param.dtype)
        merged += 1

    print(f"Merged {merged} LoRA modules into base model")

    # Save
    print(f"Saving to {args.output_dir}")
    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir)

    processor = AutoProcessor.from_pretrained(args.base_model)
    processor.save_pretrained(args.output_dir)
    print("Done!")


if __name__ == "__main__":
    main()
