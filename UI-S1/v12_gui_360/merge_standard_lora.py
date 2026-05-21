#!/usr/bin/env python3
"""Merge standard LoRA weights into base model and save as full model."""

import argparse
import json
import os
import sys

import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

from v12_gui_360.standard_lora import StandardLoRALinear
from v12_gui_360.standard_lora_wrapper import StandardLoRAWrapper


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", required=True)
    parser.add_argument("--lora_checkpoint", required=True,
                        help="Path to lora dir containing lora_weights.pt")
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    print(f"Loading base model: {args.base_model}")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16, device_map="cpu"
    )

    # Load LoRA config
    config_path = os.path.join(args.lora_checkpoint, "lora_config.json")
    with open(config_path) as f:
        lora_config = json.load(f)

    print(f"Wrapping with LoRA: r={lora_config['lora_r']}, "
          f"target_modules={lora_config['target_modules']}")
    wrapper = StandardLoRAWrapper(
        model,
        lora_r=lora_config["lora_r"],
        lora_alpha=lora_config["lora_alpha"],
        target_modules=lora_config["target_modules"],
    )
    wrapper.load_lora(args.lora_checkpoint, device="cpu")

    # Merge LoRA into base weights and restore original linear modules
    print("Merging LoRA into base model...")
    merged = 0
    for mod in wrapper.lora_modules:
        dtype = mod.base_linear.weight.dtype
        # delta = B @ A * scaling
        delta = (mod.lora_B.data.to(dtype) @ mod.lora_A.data.to(dtype)) * mod.scaling
        mod.base_linear.weight.data += delta
        merged += 1
    print(f"  Merged {merged} LoRA modules")

    # Replace StandardLoRALinear back with original nn.Linear
    # so save_pretrained produces clean keys (no .base_linear.)
    print("Restoring original linear modules...")
    for name, module in model.named_modules():
        for attr_name, child in list(module.named_children()):
            if isinstance(child, StandardLoRALinear):
                setattr(module, attr_name, child.base_linear)

    # Save merged model
    print(f"Saving to {args.output_dir}")
    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir)

    # Copy processor/tokenizer
    processor = AutoProcessor.from_pretrained(args.base_model)
    processor.save_pretrained(args.output_dir)

    print("Done!")


if __name__ == "__main__":
    main()
