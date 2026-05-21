#!/usr/bin/env python3
"""Lossless Cooperative LoRA Merge: merge LoRA_A into base + save correction adapters.

Mathematics:
  Text tokens:  y = W_merged @ x                              (exact)
  Image tokens: y = W_merged @ x + scaling * C_out @ C_in @ x (exact)

Where:
  W_merged    = W_base + scaling * B_a @ A_a_eff          (LoRA_A baked into base)
  correction  = scaling * (B_v @ A_v_eff - B_a @ A_a_eff) (delta_v - delta_a)
  C_out       = [B_v | -B_a]        (d_out × 2r)
  C_in        = [A_v_eff ; A_a_eff] (2r × d_in)

During autoregressive decode ALL generated tokens are text → W_merged is exact,
no correction applied, same speed as a standard merged model.
Corrections are only needed during prefill where image tokens exist.

Usage:
  python evaluation/merge_cooperative_lossless.py \
      --base_model checkpoints/Qwen2.5-VL-7B-Instruct \
      --coop_checkpoint train_GUI_360/llamafactory/output/cooperative_v6_5_ac/epoch-4 \
      --output_dir /tmp/merged_lossless_ac_ep4
"""

import argparse
import json
import os
import shutil
import sys

import torch

sys.stdout.reconfigure(line_buffering=True)


def main():
    parser = argparse.ArgumentParser(
        description="Lossless cooperative LoRA merge: LoRA_A into base + correction adapters"
    )
    parser.add_argument("--base_model", required=True)
    parser.add_argument("--coop_checkpoint", required=True,
                        help="Directory with lora_v.pt, lora_a.pt, cooperative_config.json")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--alpha", type=int, default=None,
                        help="LoRA alpha. Auto-detected from experiment_config.json if omitted.")
    args = parser.parse_args()

    # ── Load cooperative config ───────────────────────────────────────
    config_path = os.path.join(args.coop_checkpoint, "cooperative_config.json")
    with open(config_path) as f:
        coop_config = json.load(f)
    print("Cooperative config:")
    for k, v in coop_config.items():
        if k != "gate_values":
            print(f"  {k}: {v}")

    # Auto-detect alpha
    if args.alpha is None:
        exp_config_path = os.path.join(
            os.path.dirname(args.coop_checkpoint.rstrip("/")),
            "experiment_config.json",
        )
        if os.path.exists(exp_config_path):
            with open(exp_config_path) as f:
                exp_config = json.load(f)
            args.alpha = exp_config.get("lora_alpha", 32)
            print(f"Auto-detected lora_alpha={args.alpha}")
        else:
            args.alpha = 32
            print(f"WARNING: no experiment_config.json, using default lora_alpha=32")

    cooperative_comm = coop_config.get("cooperative_comm", False)
    gate_type = coop_config.get("gate_type", "sigmoid")
    gate_fn = torch.tanh if gate_type == "tanh" else torch.sigmoid

    # ── Load LoRA weights ─────────────────────────────────────────────
    print("Loading LoRA weights...")
    lora_v = torch.load(os.path.join(args.coop_checkpoint, "lora_v.pt"),
                        map_location="cpu", weights_only=True)
    lora_a = torch.load(os.path.join(args.coop_checkpoint, "lora_a.pt"),
                        map_location="cpu", weights_only=True)
    print(f"  LoRA_V keys: {len(lora_v)}, LoRA_A keys: {len(lora_a)}")

    # Load communication weights (v6)
    comm_params = {}
    if cooperative_comm:
        comm_path = os.path.join(args.coop_checkpoint, "lora_comm.pt")
        if os.path.exists(comm_path):
            lora_comm = torch.load(comm_path, map_location="cpu", weights_only=True)
            for key, val in lora_comm.items():
                for suffix in ("W_av", "W_va", "gate_av", "gate_va"):
                    if key.endswith("." + suffix):
                        mp = key[:-(len(suffix) + 1)]
                        comm_params.setdefault(mp, {})[suffix] = val
                        break
            print(f"  Communication modules: {len(comm_params)}")
        else:
            print(f"WARNING: cooperative_comm=True but {comm_path} not found")

    # ── Parse LoRA into per-module A/B ────────────────────────────────
    def parse_lora(state, suffix):
        modules = {}
        for key, val in state.items():
            if f"lora_A_{suffix}" in key:
                mp = key.replace(f".lora_A_{suffix}", "")
                modules.setdefault(mp, {})["A"] = val
            elif f"lora_B_{suffix}" in key:
                mp = key.replace(f".lora_B_{suffix}", "")
                modules.setdefault(mp, {})["B"] = val
        return modules

    v_modules = parse_lora(lora_v, "v")
    a_modules = parse_lora(lora_a, "a")
    all_paths = sorted(set(v_modules.keys()) | set(a_modules.keys()))
    print(f"  LoRA_V modules: {len(v_modules)}, LoRA_A modules: {len(a_modules)}")

    # Map cooperative LoRA path → base model weight key
    def lora_path_to_base_key(mp):
        p = mp
        if p.startswith("base_model."):
            p = p[len("base_model."):]
        p = p.replace("model.language_model.", "model.")
        return p + ".weight"

    # ── Pre-fold communication & compute deltas + corrections ─────────
    deltas_a = {}       # base_key → delta_a tensor
    corrections = {}    # base_key → {"C_in": [2r, d_in], "C_out": [d_out, 2r]}

    for mp in all_paths:
        base_key = lora_path_to_base_key(mp)

        has_v = mp in v_modules
        has_a = mp in a_modules
        if not has_a:
            continue

        A_a = a_modules[mp]["A"].float()
        B_a = a_modules[mp]["B"].float()
        r = A_a.shape[0]
        scaling = args.alpha / r

        A_v = v_modules[mp]["A"].float() if has_v else None
        B_v = v_modules[mp]["B"].float() if has_v else None

        # Pre-fold communication
        cp = comm_params.get(mp, {})
        has_comm = cooperative_comm and all(
            k in cp for k in ("W_av", "W_va", "gate_av", "gate_va")
        )

        if has_comm and has_v:
            g_av = gate_fn(cp["gate_av"].float())
            g_va = gate_fn(cp["gate_va"].float())
            W_av = cp["W_av"].float()
            W_va = cp["W_va"].float()
            A_v_eff = A_v + g_av * (W_av @ A_a)
            A_a_eff = A_a + g_va * (W_va @ A_v_eff)
        else:
            A_v_eff = A_v
            A_a_eff = A_a

        # delta_a: merge into base weights
        delta_a = scaling * (B_a @ A_a_eff)
        deltas_a[base_key] = delta_a

        # Correction adapter for image tokens: delta_v - delta_a
        if has_v:
            C_out = torch.cat([B_v, -B_a], dim=1)       # [d_out, 2r]
            C_in = torch.cat([A_v_eff, A_a_eff], dim=0)  # [2r, d_in]
            corrections[base_key] = {
                "C_in": C_in.to(torch.bfloat16),
                "C_out": C_out.to(torch.bfloat16),
            }

    print(f"Computed {len(deltas_a)} delta_a, {len(corrections)} correction adapters")

    # ── Apply delta_a to base model shards & save ─────────────────────
    from safetensors.torch import load_file, save_file
    from glob import glob

    os.makedirs(args.output_dir, exist_ok=True)

    # Copy non-weight files from base model
    for fname in os.listdir(args.base_model):
        if fname.endswith(".safetensors"):
            continue
        src = os.path.join(args.base_model, fname)
        dst = os.path.join(args.output_dir, fname)
        if os.path.isfile(src) and not os.path.exists(dst):
            shutil.copy2(src, dst)

    shard_files = sorted(glob(os.path.join(args.base_model, "model-*.safetensors")))
    index_path = os.path.join(args.base_model, "model.safetensors.index.json")

    applied = set()
    for sf in shard_files:
        shard_name = os.path.basename(sf)
        print(f"Processing {shard_name}...")
        shard = load_file(sf)
        for key in list(shard.keys()):
            if key in deltas_a:
                original = shard[key].float()
                shard[key] = (original + deltas_a[key]).to(shard[key].dtype)
                applied.add(key)
        save_file(shard, os.path.join(args.output_dir, shard_name))

    if os.path.exists(index_path):
        shutil.copy2(
            index_path,
            os.path.join(args.output_dir, "model.safetensors.index.json"),
        )

    print(f"Applied delta_a to {len(applied)}/{len(deltas_a)} modules")
    unapplied = set(deltas_a.keys()) - applied
    if unapplied:
        print(f"WARNING: {len(unapplied)} deltas not applied:")
        for k in sorted(unapplied):
            print(f"  {k}")

    # ── Save correction adapters ──────────────────────────────────────
    corr_state = {}
    for base_key, cd in corrections.items():
        corr_state[base_key + ".C_in"] = cd["C_in"]
        corr_state[base_key + ".C_out"] = cd["C_out"]

    corr_path = os.path.join(args.output_dir, "correction_adapters.pt")
    torch.save(corr_state, corr_path)

    corr_params = sum(v.numel() for v in corr_state.values())
    corr_bytes = sum(v.numel() * v.element_size() for v in corr_state.values())
    print(f"Saved {len(corrections)} correction adapters to {corr_path}")
    print(f"  Correction params: {corr_params:,} ({corr_bytes / 1e6:.1f} MB)")

    # ── Save correction config ────────────────────────────────────────
    first_key = next(iter(corrections.keys()))
    corr_rank = corrections[first_key]["C_in"].shape[0]

    corr_config = {
        "scaling": scaling,
        "correction_rank": int(corr_rank),
        "original_lora_rank": int(r),
        "original_lora_alpha": int(args.alpha),
        "target_modules": coop_config["target_modules"],
        "image_pad_id": 151655,
        "merge_mode": "lossless",
        "gate_type": gate_type,
        "cooperative_comm": cooperative_comm,
    }
    with open(os.path.join(args.output_dir, "correction_config.json"), "w") as f:
        json.dump(corr_config, f, indent=2)

    print(f"\nLossless merged model saved to {args.output_dir}")
    print(f"  Text tokens  → merged weights (exact, no correction)")
    print(f"  Image tokens → merged weights + correction adapters (exact)")
    print(f"  Decode phase → merged weights only (zero overhead)")


if __name__ == "__main__":
    main()
