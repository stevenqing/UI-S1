#!/usr/bin/env python3
"""Merge two cooperative LoRA checkpoints (grounder + actor) into one.

Takes two V13-format cooperative checkpoints trained on different tasks:
  - Grounder: trained on gui_grounding (predict coordinates)
  - Actor:    trained on gui_action (predict tool_call actions)

Merge strategy:
  lora_A_1 <- grounder's lora_A_1   (Expert 1 = grounding specialist)
  lora_A_2 <- actor's lora_A_2      (Expert 2 = action specialist)
  lora_B   <- 0.5 * (grounder_B + actor_B)   (shared projection, averaged)
  route_weights <- zeros (sigmoid=0.5, equal expert weighting)
  comm_weights  <- fresh Kaiming init

This leverages the fact that Expert 1 was trained as the grounding expert in the
grounder checkpoint, and Expert 2 was trained as the action expert in the actor
checkpoint. By cross-selecting A matrices, each expert retains its specialization.

Input:
  --grounder_checkpoint: e.g. checkpoints/v16_peft_grounding_cooperative_r128
  --actor_checkpoint:    e.g. checkpoints/v15_peft_balanced_cooperative_r128

Output:
  checkpoints/v16_merged_grounder_actor_cooperative/
    lora_weights.pt, route_weights.pt, comm_weights.pt, cooperative_config.json

Usage:
    python v16_task_vector_merge/merge_cooperative_experts.py \
        --grounder_checkpoint checkpoints/v16_peft_grounding_cooperative_r128 \
        --actor_checkpoint checkpoints/v15_peft_balanced_cooperative_r128 \
        --output_dir checkpoints/v16_merged_grounder_actor_cooperative \
        --num_comm_rounds 2
"""

import argparse
import json
import math
import os

import torch


NUM_LAYERS = 28
HIDDEN_SIZE = 3584


def log(msg=""):
    print(msg, flush=True)


def load_cooperative_checkpoint(ckpt_dir: str) -> dict:
    """Load a V13 cooperative checkpoint (lora, route, comm, config)."""
    lora = torch.load(os.path.join(ckpt_dir, "lora_weights.pt"),
                      map_location="cpu", weights_only=True)
    route = torch.load(os.path.join(ckpt_dir, "route_weights.pt"),
                       map_location="cpu", weights_only=True)
    comm = torch.load(os.path.join(ckpt_dir, "comm_weights.pt"),
                      map_location="cpu", weights_only=True)
    with open(os.path.join(ckpt_dir, "cooperative_config.json")) as f:
        config = json.load(f)
    return {"lora": lora, "route": route, "comm": comm, "config": config}


def merge(
    grounder_dir: str,
    actor_dir: str,
    output_dir: str,
    num_comm_rounds: int = 2,
):
    log(f"{'=' * 70}")
    log(f"  Merge Cooperative Experts: Grounder + Actor")
    log(f"{'=' * 70}")
    log(f"  Grounder: {grounder_dir}")
    log(f"  Actor:    {actor_dir}")
    log(f"  Output:   {output_dir}")
    log(f"  Comm rounds: {num_comm_rounds}")
    log()

    # Load checkpoints
    log("Loading grounder checkpoint...")
    grounder = load_cooperative_checkpoint(grounder_dir)
    log(f"  {len(grounder['lora'])} lora tensors, config: r={grounder['config']['lora_r']}")

    log("Loading actor checkpoint...")
    actor = load_cooperative_checkpoint(actor_dir)
    log(f"  {len(actor['lora'])} lora tensors, config: r={actor['config']['lora_r']}")

    # Validate compatibility
    g_cfg = grounder["config"]
    a_cfg = actor["config"]
    rank = g_cfg["lora_r"]
    alpha = g_cfg["lora_alpha"]
    assert rank == a_cfg["lora_r"], (
        f"Rank mismatch: grounder={rank}, actor={a_cfg['lora_r']}")
    assert alpha == a_cfg["lora_alpha"], (
        f"Alpha mismatch: grounder={alpha}, actor={a_cfg['lora_alpha']}")

    log(f"  Rank: {rank}, Alpha: {alpha}")

    # Discover modules from grounder checkpoint
    # Keys: prefix.lora_A_1, prefix.lora_A_2, prefix.lora_B
    module_prefixes = set()
    for key in grounder["lora"]:
        if key.endswith(".lora_A_1"):
            prefix = key.rsplit(".lora_A_1", 1)[0]
            module_prefixes.add(prefix)

    log(f"  Modules: {len(module_prefixes)}")
    log()

    # Merge lora weights
    merged_lora = {}
    mismatches = 0

    for prefix in sorted(module_prefixes):
        a1_key = f"{prefix}.lora_A_1"
        a2_key = f"{prefix}.lora_A_2"
        b_key = f"{prefix}.lora_B"

        # Validate both checkpoints have this module
        for key in (a1_key, a2_key, b_key):
            if key not in grounder["lora"]:
                log(f"  WARNING: {key} missing from grounder, skipping module")
                mismatches += 1
                continue
            if key not in actor["lora"]:
                log(f"  WARNING: {key} missing from actor, skipping module")
                mismatches += 1
                continue

        # A_1 from grounder (Expert 1 = grounding specialist)
        merged_lora[a1_key] = grounder["lora"][a1_key]

        # A_2 from actor (Expert 2 = action specialist)
        merged_lora[a2_key] = actor["lora"][a2_key]

        # B averaged from both
        g_B = grounder["lora"][b_key].float()
        a_B = actor["lora"][b_key].float()
        merged_B = (0.5 * g_B + 0.5 * a_B).to(grounder["lora"][b_key].dtype)
        merged_lora[b_key] = merged_B

    log(f"  Merged {len(merged_lora)} tensors ({len(merged_lora) // 3} modules)")
    if mismatches > 0:
        log(f"  WARNING: {mismatches} key mismatches")

    # Fresh route weights: zeros -> sigmoid(0) = 0.5
    dtype = torch.bfloat16
    route_state = {}
    for layer_idx in range(NUM_LAYERS):
        route_state[f"route_weights.{layer_idx}"] = torch.zeros(
            HIDDEN_SIZE, dtype=dtype)

    # Fresh comm weights: Kaiming W, zero gates
    T = num_comm_rounds
    comm_state = {}
    for idx in range(NUM_LAYERS * T):
        w_12 = torch.zeros(rank, rank)
        w_21 = torch.zeros(rank, rank)
        torch.nn.init.kaiming_uniform_(w_12, a=math.sqrt(5))
        torch.nn.init.kaiming_uniform_(w_21, a=math.sqrt(5))
        comm_state[f"comm_W_12.{idx}"] = w_12.to(dtype)
        comm_state[f"comm_W_21.{idx}"] = w_21.to(dtype)
        comm_state[f"comm_gate_12.{idx}"] = torch.zeros(rank, dtype=dtype)
        comm_state[f"comm_gate_21.{idx}"] = torch.zeros(rank, dtype=dtype)

    # Determine target modules from both configs
    target_modules = list(set(
        g_cfg.get("target_modules", []) + a_cfg.get("target_modules", [])
    ))

    # Save
    os.makedirs(output_dir, exist_ok=True)
    torch.save(merged_lora, os.path.join(output_dir, "lora_weights.pt"))
    torch.save(route_state, os.path.join(output_dir, "route_weights.pt"))
    torch.save(comm_state, os.path.join(output_dir, "comm_weights.pt"))

    config = {
        "lora_r": rank,
        "lora_alpha": alpha,
        "target_modules": sorted(target_modules),
        "balance_weight": 0.01,
        "num_comm_rounds": num_comm_rounds,
        "type": "iterative_cooperative_v13",
        "conversion_info": {
            "source": "merge_cooperative_experts",
            "grounder_checkpoint": os.path.abspath(grounder_dir),
            "actor_checkpoint": os.path.abspath(actor_dir),
            "merge_strategy": {
                "A_1": "grounder (Expert 1 = grounding specialist)",
                "A_2": "actor (Expert 2 = action specialist)",
                "B": "0.5 * grounder_B + 0.5 * actor_B",
                "route_weights": "zeros (sigmoid=0.5)",
                "comm_weights": "fresh Kaiming init",
            },
        },
    }
    with open(os.path.join(output_dir, "cooperative_config.json"), "w") as f:
        json.dump(config, f, indent=2)

    # Statistics
    lora_params = sum(t.numel() for t in merged_lora.values())
    route_params = sum(t.numel() for t in route_state.values())
    comm_params = sum(t.numel() for t in comm_state.values())

    log()
    log(f"{'=' * 70}")
    log(f"  Merge Statistics")
    log(f"{'=' * 70}")
    log(f"  Modules merged: {len(merged_lora) // 3}")
    log(f"  Target modules: {sorted(target_modules)}")
    log(f"  Total parameters: {(lora_params + route_params + comm_params) / 1e6:.1f}M")
    log(f"    LoRA:   {lora_params / 1e6:.1f}M ({len(merged_lora)} tensors)")
    log(f"    Route:  {route_params / 1e6:.2f}M ({len(route_state)} tensors)")
    log(f"    Comm:   {comm_params / 1e6:.2f}M ({len(comm_state)} tensors)")
    log()

    # Verify shapes match between grounder A_1 and actor A_2
    sample_prefix = sorted(module_prefixes)[0]
    a1 = merged_lora[f"{sample_prefix}.lora_A_1"]
    a2 = merged_lora[f"{sample_prefix}.lora_A_2"]
    b = merged_lora[f"{sample_prefix}.lora_B"]
    log(f"  Sample module: {sample_prefix}")
    log(f"    A_1 (grounder): {list(a1.shape)}, dtype={a1.dtype}")
    log(f"    A_2 (actor):    {list(a2.shape)}, dtype={a2.dtype}")
    log(f"    B (averaged):   {list(b.shape)}, dtype={b.dtype}")

    # Check A_1 != A_2 (they should differ since they come from different checkpoints)
    a1_f = a1.float()
    a2_f = a2.float()
    cos_sim = torch.nn.functional.cosine_similarity(
        a1_f.flatten().unsqueeze(0),
        a2_f.flatten().unsqueeze(0),
    ).item()
    log(f"    A_1 vs A_2 cosine sim: {cos_sim:.4f} (should be < 1.0)")

    log()
    log(f"  Saved to: {output_dir}")
    log(f"    lora_weights.pt")
    log(f"    route_weights.pt")
    log(f"    comm_weights.pt")
    log(f"    cooperative_config.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Merge grounder + actor cooperative checkpoints")
    parser.add_argument("--grounder_checkpoint", required=True,
                        help="Path to grounder cooperative checkpoint dir")
    parser.add_argument("--actor_checkpoint", required=True,
                        help="Path to actor cooperative checkpoint dir")
    parser.add_argument("--output_dir", required=True,
                        help="Output directory for merged checkpoint")
    parser.add_argument("--num_comm_rounds", type=int, default=2,
                        help="Number of communication rounds (default: 2)")
    args = parser.parse_args()

    merge(
        grounder_dir=args.grounder_checkpoint,
        actor_dir=args.actor_checkpoint,
        output_dir=args.output_dir,
        num_comm_rounds=args.num_comm_rounds,
    )
