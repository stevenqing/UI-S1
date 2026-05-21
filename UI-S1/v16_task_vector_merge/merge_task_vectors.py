#!/usr/bin/env python3
"""Merge two full-param fine-tuned models using task vector methods.

Implements four methods from the LoRA Soups paper (COLING 2025):
  1. Simple average
  2. Task arithmetic (weighted sum)
  3. TIES-Merging (trim, elect sign, merge)
  4. DARE (drop and rescale)

Memory-efficient: processes one base shard at a time (~7GB × 3 peak).

Usage:
    python v16_task_vector_merge/merge_task_vectors.py \
        --base_model checkpoints/Qwen2.5-VL-7B-Instruct \
        --model_a checkpoints/gui_grounding \
        --model_b checkpoints/gui_action \
        --output_dir checkpoints/v16_merged_avg_05_05 \
        --method avg
"""

import argparse
import json
import os
import shutil
from collections import defaultdict

import torch
from safetensors.torch import load_file, save_file


def load_shard_index(model_dir):
    """Load model.safetensors.index.json and return key->shard mapping."""
    index_path = os.path.join(model_dir, "model.safetensors.index.json")
    with open(index_path) as f:
        index = json.load(f)
    return index["weight_map"]


def group_keys_by_base_shard(base_map, map_a, map_b):
    """Group all keys by their base model shard for sequential processing."""
    shard_groups = defaultdict(list)
    for key in base_map:
        shard_groups[base_map[key]].append(key)
    return shard_groups


def merge_avg(delta_a, delta_b, alpha=0.5, beta=0.5):
    """Simple weighted average of task vectors."""
    return alpha * delta_a + beta * delta_b


def merge_task_arith(delta_a, delta_b, alpha, beta):
    """Task arithmetic: weighted sum of task vectors."""
    return alpha * delta_a + beta * delta_b


def merge_ties(delta_a, delta_b, k, lam):
    """TIES-Merging: trim, elect sign, merge.

    1. Trim: zero out small-magnitude elements (keep top-k% by |value|)
    2. Elect sign: pick sign with largest total magnitude across tasks
    3. Merge: average values that agree with elected sign, zero out rest
    """
    deltas = [delta_a.clone(), delta_b.clone()]

    # Step 1: Trim - keep top-k% by magnitude per task vector
    for i in range(2):
        flat = deltas[i].flatten()
        n_keep = max(1, int(len(flat) * k / 100.0))
        if n_keep < len(flat):
            threshold = flat.abs().kthvalue(len(flat) - n_keep + 1).values
            mask = flat.abs() >= threshold
            flat[~mask] = 0.0
            deltas[i] = flat.view_as(delta_a)

    # Step 2: Elect sign - for each element, pick sign with largest total magnitude
    stack = torch.stack(deltas)  # [2, ...]
    pos_mass = (stack.clamp(min=0)).sum(dim=0)   # total positive magnitude
    neg_mass = (stack.clamp(max=0)).abs().sum(dim=0)  # total negative magnitude
    elected_sign = torch.where(pos_mass >= neg_mass, torch.ones_like(pos_mass), -torch.ones_like(pos_mass))

    # Step 3: Merge - average values that agree with elected sign
    merged = torch.zeros_like(delta_a)
    count = torch.zeros_like(delta_a)
    for d in deltas:
        agree = (d.sign() == elected_sign) & (d != 0)
        merged[agree] += d[agree]
        count[agree] += 1

    count = count.clamp(min=1)
    merged = merged / count

    return lam * merged


def merge_dare(delta_a, delta_b, p, lam, rng):
    """DARE: drop and rescale, then sum.

    1. Drop: randomly zero out (1-p) fraction of each task vector
    2. Rescale: multiply remaining by 1/p
    3. Sum the rescaled task vectors
    """
    rescale = 1.0 / p

    # Process each task vector
    mask_a = torch.bernoulli(torch.full_like(delta_a, p), generator=rng).bool()
    dropped_a = torch.where(mask_a, delta_a * rescale, torch.zeros_like(delta_a))

    mask_b = torch.bernoulli(torch.full_like(delta_b, p), generator=rng).bool()
    dropped_b = torch.where(mask_b, delta_b * rescale, torch.zeros_like(delta_b))

    return lam * (dropped_a + dropped_b)


def main():
    parser = argparse.ArgumentParser(description="Merge two fine-tuned models using task vector methods")
    parser.add_argument("--base_model", required=True, help="Base model directory")
    parser.add_argument("--model_a", required=True, help="First fine-tuned model (gui_grounding)")
    parser.add_argument("--model_b", required=True, help="Second fine-tuned model (gui_action)")
    parser.add_argument("--output_dir", required=True, help="Output merged model directory")
    parser.add_argument("--method", required=True, choices=["avg", "task_arith", "ties", "dare"])
    parser.add_argument("--alpha", type=float, default=0.5, help="Weight for model_a task vector")
    parser.add_argument("--beta", type=float, default=0.5, help="Weight for model_b task vector")
    parser.add_argument("--ties_k", type=float, default=20, help="TIES top-k%% to keep")
    parser.add_argument("--ties_lambda", type=float, default=1.0, help="TIES scaling factor")
    parser.add_argument("--dare_p", type=float, default=0.5, help="DARE keep probability")
    parser.add_argument("--dare_lambda", type=float, default=1.0, help="DARE scaling factor")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for DARE")
    args = parser.parse_args()

    print("=" * 60)
    print(f"Task Vector Merge: {args.method}")
    print(f"Base:    {args.base_model}")
    print(f"Model A: {args.model_a}")
    print(f"Model B: {args.model_b}")
    print(f"Output:  {args.output_dir}")
    if args.method == "avg":
        print(f"Params:  alpha={args.alpha}, beta={args.beta}")
    elif args.method == "task_arith":
        print(f"Params:  alpha={args.alpha}, beta={args.beta}")
    elif args.method == "ties":
        print(f"Params:  k={args.ties_k}, lambda={args.ties_lambda}")
    elif args.method == "dare":
        print(f"Params:  p={args.dare_p}, lambda={args.dare_lambda}, seed={args.seed}")
    print("=" * 60)

    # Load shard indices
    base_map = load_shard_index(args.base_model)
    map_a = load_shard_index(args.model_a)
    map_b = load_shard_index(args.model_b)

    # Verify key sets match
    base_keys = set(base_map.keys())
    keys_a = set(map_a.keys())
    keys_b = set(map_b.keys())
    assert base_keys == keys_a, f"Key mismatch: base has {len(base_keys)} keys, model_a has {len(keys_a)}"
    assert base_keys == keys_b, f"Key mismatch: base has {len(base_keys)} keys, model_b has {len(keys_b)}"
    print(f"All 3 models have {len(base_keys)} keys")

    # Group keys by base shard
    shard_groups = group_keys_by_base_shard(base_map, map_a, map_b)
    base_shards = sorted(shard_groups.keys())
    print(f"Base model has {len(base_shards)} shards")

    # Prepare output
    os.makedirs(args.output_dir, exist_ok=True)

    # Setup DARE RNG if needed
    rng = None
    if args.method == "dare":
        rng = torch.Generator()
        rng.manual_seed(args.seed)

    # Track output shard info
    output_weight_map = {}
    total_size = 0
    output_shard_idx = 0

    # Process shard by shard
    for shard_name in base_shards:
        keys = shard_groups[shard_name]
        print(f"\nProcessing {shard_name} ({len(keys)} keys)...")

        # Load base shard
        base_shard = load_file(os.path.join(args.base_model, shard_name))

        # Find which model_a and model_b shards we need
        needed_a = set(map_a[k] for k in keys)
        needed_b = set(map_b[k] for k in keys)

        # Load needed shards from model_a and model_b
        shards_a = {}
        for s in needed_a:
            shards_a.update(load_file(os.path.join(args.model_a, s)))

        shards_b = {}
        for s in needed_b:
            shards_b.update(load_file(os.path.join(args.model_b, s)))

        # Compute merged weights for each key
        merged_shard = {}
        for key in keys:
            w_base = base_shard[key].float()
            w_a = shards_a[key].float()
            w_b = shards_b[key].float()

            delta_a = w_a - w_base
            delta_b = w_b - w_base

            if args.method == "avg":
                delta_merged = merge_avg(delta_a, delta_b, args.alpha, args.beta)
            elif args.method == "task_arith":
                delta_merged = merge_task_arith(delta_a, delta_b, args.alpha, args.beta)
            elif args.method == "ties":
                delta_merged = merge_ties(delta_a, delta_b, args.ties_k, args.ties_lambda)
            elif args.method == "dare":
                delta_merged = merge_dare(delta_a, delta_b, args.dare_p, args.dare_lambda, rng)

            merged_shard[key] = (w_base + delta_merged).to(torch.bfloat16)
            total_size += merged_shard[key].nelement() * merged_shard[key].element_size()

        # Save output shard (keep same shard name as base)
        output_shard_idx += 1
        save_file(merged_shard, os.path.join(args.output_dir, shard_name))
        for key in keys:
            output_weight_map[key] = shard_name
        print(f"  Saved {shard_name} ({len(keys)} keys)")

        # Free memory
        del base_shard, shards_a, shards_b, merged_shard

    # Write index file
    index = {
        "metadata": {"total_size": total_size},
        "weight_map": output_weight_map,
    }
    with open(os.path.join(args.output_dir, "model.safetensors.index.json"), "w") as f:
        json.dump(index, f, indent=2)

    # Copy config and tokenizer files from base model
    copy_files = [
        "config.json",
        "generation_config.json",
        "preprocessor_config.json",
        "tokenizer_config.json",
        "tokenizer.json",
        "vocab.json",
        "merges.txt",
        "special_tokens_map.json",
        "chat_template.json",
    ]
    for fname in copy_files:
        src = os.path.join(args.base_model, fname)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(args.output_dir, fname))
            print(f"Copied {fname}")

    # Verify
    final_keys = set(output_weight_map.keys())
    print(f"\nMerge complete: {len(final_keys)} keys in {output_shard_idx} shards")
    assert len(final_keys) == len(base_keys), f"Key count mismatch: {len(final_keys)} vs {len(base_keys)}"
    print(f"Output: {args.output_dir}")
    print("Done!")


if __name__ == "__main__":
    main()
