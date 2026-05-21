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
    parser.add_argument("--alpha", type=int, default=None,
                        help="LoRA alpha. If None, auto-detect from experiment_config.json (fallback 32).")
    args = parser.parse_args()

    # Load cooperative config
    config_path = os.path.join(args.coop_checkpoint, "cooperative_config.json")
    with open(config_path) as f:
        coop_config = json.load(f)
    # Print config without gate_values (too verbose)
    print("Cooperative config:")
    for k, v in coop_config.items():
        if k != "gate_values":
            print(f"  {k}: {v}")

    # Try to auto-detect alpha from experiment_config.json (one dir up)
    if args.alpha is None:
        exp_config_path = os.path.join(os.path.dirname(args.coop_checkpoint.rstrip("/")), "experiment_config.json")
        if os.path.exists(exp_config_path):
            with open(exp_config_path) as f:
                exp_config = json.load(f)
            args.alpha = exp_config.get("lora_alpha", 32)
            print(f"Auto-detected lora_alpha={args.alpha} from {exp_config_path}")
        else:
            args.alpha = 32
            print(f"WARNING: no experiment_config.json found, using default lora_alpha=32")

    cooperative_comm = coop_config.get("cooperative_comm", False)

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

    # v6: Load cooperative communication weights (W_av, W_va, gate_av, gate_va)
    lora_comm = {}
    comm_params = {}  # module_path -> {"W_av", "W_va", "g_av", "g_va"}
    if cooperative_comm:
        comm_path = os.path.join(args.coop_checkpoint, "lora_comm.pt")
        if os.path.exists(comm_path):
            print("Loading LoRA communication weights (v6)...")
            lora_comm = torch.load(comm_path, map_location="cpu", weights_only=True)
            print(f"  LoRA comm keys: {len(lora_comm)}")
            # Parse into per-module dict
            for key, val in lora_comm.items():
                for suffix in ("W_av", "W_va", "gate_av", "gate_va"):
                    if key.endswith("." + suffix):
                        mp = key[: -(len(suffix) + 1)]
                        comm_params.setdefault(mp, {})[suffix] = val
                        break
            print(f"  Communication modules: {len(comm_params)}")
        else:
            print(f"WARNING: cooperative_comm=True but {comm_path} not found")

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

    # Build key mapping from LoRA module_path -> base model weight key.
    # LoRA keys look like: base_model.model.language_model.layers.0.self_attn.q_proj
    # Qwen2.5-VL base keys look like: model.layers.0.self_attn.q_proj.weight
    # (no "language_model." prefix on the language model side)
    def lora_path_to_base_key(module_path: str) -> str:
        p = module_path
        if p.startswith("base_model."):
            p = p[len("base_model."):]
        # Strip "language_model." that cooperative wrapper inserts
        p = p.replace("model.language_model.", "model.")
        return p + ".weight"

    for module_path in sorted(all_module_paths):
        base_key = lora_path_to_base_key(module_path)

        r = None
        if module_path in a_modules:
            r = a_modules[module_path]["A"].shape[0]
        elif module_path in v_modules:
            r = v_modules[module_path]["A"].shape[0]

        scaling = args.alpha / r if r else 1.0

        # Get cooperative communication for this module (if v6)
        cp = comm_params.get(module_path, {})
        has_comm = cooperative_comm and all(k in cp for k in ("W_av", "W_va", "gate_av", "gate_va"))

        delta = None

        # LoRA_V side: delta_v_weight = scaling * B_v @ (A_v + g_av * W_av @ A_a)
        if module_path in v_modules and args.merge_mode == "both":
            A_v = v_modules[module_path]["A"].float()
            B_v = v_modules[module_path]["B"].float()
            A_v_eff = A_v
            if has_comm and module_path in a_modules:
                A_a_raw = a_modules[module_path]["A"].float()
                g_av = torch.sigmoid(cp["gate_av"].float())
                W_av = cp["W_av"].float()
                A_v_eff = A_v + g_av * (W_av @ A_a_raw)
            delta = B_v @ A_v_eff

        # LoRA_A side: delta_a_weight = scaling * B_a @ (A_a + g_va * W_va @ (A_v + g_av * W_av @ A_a))
        if module_path in a_modules:
            A_a = a_modules[module_path]["A"].float()
            B_a = a_modules[module_path]["B"].float()
            A_a_eff = A_a
            if has_comm and module_path in v_modules:
                A_v_raw = v_modules[module_path]["A"].float()
                g_av = torch.sigmoid(cp["gate_av"].float())
                g_va = torch.sigmoid(cp["gate_va"].float())
                W_av = cp["W_av"].float()
                W_va = cp["W_va"].float()
                # h_v_new uses new V (after av communication)
                A_v_new = A_v_raw + g_av * (W_av @ A_a)
                A_a_eff = A_a + g_va * (W_va @ A_v_new)
            contrib = B_a @ A_a_eff
            delta = contrib if delta is None else delta + contrib

        if delta is None:
            continue
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
