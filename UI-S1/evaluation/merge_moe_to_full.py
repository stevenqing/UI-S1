"""
Merge MoE LoRA checkpoint into base model weights for vLLM inference.

For each module, computes:
  ΔW = Σ_i w_i * B_i @ A_i * (alpha / r)
and merges into base: W_new = W_base + ΔW

Since routing weights are near-uniform (~0.5/0.5), uses fixed uniform weights
by default. Can also use per-sample mean routing weights from eval results.

Usage:
    python merge_moe_to_full.py \
        --base_model /path/to/Qwen2.5-VL-7B-Instruct \
        --checkpoint_dir /path/to/moe_checkpoint \
        --output_dir /path/to/merged_model
"""

import argparse
import json
import os
import shutil

import torch
from safetensors.torch import save_file


def merge_moe_checkpoint(base_model_path: str, checkpoint_dir: str, output_dir: str,
                         routing_weights: list = None):
    """Merge MoE LoRA weights into base model and save as full model."""

    # Load MoE config
    config_path = os.path.join(checkpoint_dir, 'moe_config.json')
    with open(config_path) as f:
        moe_config = json.load(f)

    target_modules = moe_config['target_modules']
    moe_modules = set(moe_config.get('moe_modules') or target_modules)
    num_experts = moe_config['num_experts']
    moe_r = moe_config['expert_lora_r']
    moe_alpha = moe_config['expert_lora_alpha']
    std_r = moe_config.get('standard_lora_r', 32)
    std_alpha = moe_config.get('standard_lora_alpha', 64)

    moe_scaling = moe_alpha / moe_r
    std_scaling = std_alpha / std_r

    print(f"MoE config: {num_experts} experts, moe_r={moe_r}, moe_alpha={moe_alpha}")
    print(f"Standard LoRA: r={std_r}, alpha={std_alpha}")
    print(f"MoE modules: {sorted(moe_modules)}")
    print(f"Standard modules: {sorted(set(target_modules) - moe_modules)}")

    # Default routing weights: uniform
    if routing_weights is None:
        routing_weights = [1.0 / num_experts] * num_experts
    print(f"Routing weights: {routing_weights}")

    # Load LoRA weights
    lora_path = os.path.join(checkpoint_dir, 'lora_weights.pt')
    lora_state_dict = torch.load(lora_path, map_location='cpu')
    print(f"Loaded {len(lora_state_dict)} LoRA weight tensors")

    # Load base model weights (safetensors)
    from safetensors import safe_open
    index_path = os.path.join(base_model_path, 'model.safetensors.index.json')
    with open(index_path) as f:
        index = json.load(f)
    weight_map = index['weight_map']

    # Build mapping: base_weight_key -> delta_W
    # For Qwen2.5-VL: model.layers.{i}.self_attn.{q,k,v,o}_proj.weight
    #                  model.layers.{i}.mlp.{gate,up,down}_proj.weight
    deltas = {}

    # Get num_layers from config
    model_config_path = os.path.join(base_model_path, 'config.json')
    with open(model_config_path) as f:
        model_config = json.load(f)
    num_layers = model_config.get('num_hidden_layers', 28)
    print(f"Model layers: {num_layers}")

    merged_count = 0
    for layer_idx in range(num_layers):
        for module_name in target_modules:
            is_moe = module_name in moe_modules

            if module_name in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
                base_key = f"model.layers.{layer_idx}.self_attn.{module_name}.weight"
            elif module_name in ['gate_proj', 'up_proj', 'down_proj']:
                base_key = f"model.layers.{layer_idx}.mlp.{module_name}.weight"
            else:
                continue

            if is_moe:
                # Compute weighted sum of expert deltas
                delta_W = None
                for expert_idx in range(num_experts):
                    prefix = f"moe.layer_{layer_idx}.{module_name}.expert_{expert_idx}"
                    a_key = f"{prefix}.lora_A"
                    b_key = f"{prefix}.lora_B"

                    if a_key not in lora_state_dict or b_key not in lora_state_dict:
                        continue

                    A = lora_state_dict[a_key].float()  # [r, in]
                    B = lora_state_dict[b_key].float()  # [out, r]

                    expert_delta = (B @ A) * moe_scaling * routing_weights[expert_idx]
                    if delta_W is None:
                        delta_W = expert_delta
                    else:
                        delta_W += expert_delta
            else:
                # Standard LoRA: single expert
                prefix = f"std.layer_{layer_idx}.{module_name}"
                a_key = f"{prefix}.lora_A"
                b_key = f"{prefix}.lora_B"

                if a_key not in lora_state_dict or b_key not in lora_state_dict:
                    continue

                A = lora_state_dict[a_key].float()
                B = lora_state_dict[b_key].float()
                delta_W = (B @ A) * std_scaling

            if delta_W is not None:
                deltas[base_key] = delta_W
                merged_count += 1

    print(f"Computed {merged_count} weight deltas to merge")

    # Copy base model files to output
    os.makedirs(output_dir, exist_ok=True)

    # Copy all non-weight files
    for fname in os.listdir(base_model_path):
        src = os.path.join(base_model_path, fname)
        dst = os.path.join(output_dir, fname)
        if fname.endswith('.safetensors'):
            continue  # Will rewrite these
        if os.path.isfile(src):
            shutil.copy2(src, dst)

    # Load, merge, and save each safetensors shard
    shard_files = sorted(set(weight_map.values()))
    new_weight_map = {}

    for shard_file in shard_files:
        print(f"Processing {shard_file}...")
        shard_path = os.path.join(base_model_path, shard_file)
        shard_tensors = {}

        with safe_open(shard_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                tensor = f.get_tensor(key)
                if key in deltas:
                    original_dtype = tensor.dtype
                    tensor = tensor.float() + deltas[key]
                    tensor = tensor.to(original_dtype)
                    print(f"  Merged: {key}")
                shard_tensors[key] = tensor
                new_weight_map[key] = shard_file

        # Save merged shard
        out_shard_path = os.path.join(output_dir, shard_file)
        save_file(shard_tensors, out_shard_path)

    # Update index
    new_index = {
        "metadata": index.get("metadata", {}),
        "weight_map": new_weight_map
    }
    with open(os.path.join(output_dir, 'model.safetensors.index.json'), 'w') as f:
        json.dump(new_index, f, indent=2)

    # Save merge info
    merge_info = {
        "base_model": base_model_path,
        "checkpoint_dir": checkpoint_dir,
        "moe_config": moe_config,
        "routing_weights": routing_weights,
        "merged_modules": merged_count,
    }
    with open(os.path.join(output_dir, 'merge_info.json'), 'w') as f:
        json.dump(merge_info, f, indent=2)

    print(f"\nMerged model saved to {output_dir}")
    print(f"Total merged modules: {merged_count}")


def main():
    parser = argparse.ArgumentParser(description="Merge MoE LoRA into base model")
    parser.add_argument("--base_model", type=str, required=True,
                        help="Path to base model (Qwen2.5-VL-7B-Instruct)")
    parser.add_argument("--checkpoint_dir", type=str, required=True,
                        help="Path to MoE checkpoint directory")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for merged model")
    parser.add_argument("--routing_weights", type=float, nargs='+', default=None,
                        help="Routing weights (default: uniform)")
    args = parser.parse_args()

    merge_moe_checkpoint(
        base_model_path=args.base_model,
        checkpoint_dir=args.checkpoint_dir,
        output_dir=args.output_dir,
        routing_weights=args.routing_weights,
    )


if __name__ == "__main__":
    main()
