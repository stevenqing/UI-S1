#!/usr/bin/env python3
"""Convert two full-param fine-tuned models into a cooperative LoRA (V13) checkpoint.

Given a base model and two full-param SFT models (e.g. gui_grounding, gui_action),
computes task vectors (delta = W_sft - W_base) and performs joint SVD to extract
a shared B matrix with two expert A matrices.

Joint SVD:
  M = [delta_g | delta_a]           # [out_f, 2*in_f]
  U_r, S_r, Vt_r = SVD(M, rank=r)  # truncated
  sqrt_s = sqrt(S_r / scaling)

  B   = U_r * sqrt_s               # [out_f, r]  — shared
  A_1 = Vt_r[:, :in_f] * sqrt_s    # [r, in_f]   — expert 1 (model_a)
  A_2 = Vt_r[:, in_f:] * sqrt_s    # [r, in_f]   — expert 2 (model_b)

This is optimal: B @ A_1 * scaling ≈ delta_g, B @ A_2 * scaling ≈ delta_a.

Output format matches V13 cooperative LoRA (iterative_cooperative_wrapper.py):
  lora_weights.pt, route_weights.pt, comm_weights.pt, cooperative_config.json

Usage:
    python v16_task_vector_merge/convert_fullsft_to_cooperative.py \
        --base_model checkpoints/Qwen2.5-VL-7B-Instruct \
        --model_a checkpoints/gui_grounding \
        --model_b checkpoints/gui_action \
        --output_dir checkpoints/v16_coop_r128_attn \
        --rank 128 --alpha 256 --num_comm_rounds 2
"""

import argparse
import json
import math
import os
import time
from collections import defaultdict

import torch
from safetensors.torch import load_file


def log(msg=""):
    print(msg, flush=True)


NUM_LAYERS = 28
HIDDEN_SIZE = 3584


def get_hf_key(layer_idx: int, module_name: str) -> str:
    """Get the HuggingFace weight key for a target module."""
    if module_name in ("q_proj", "k_proj", "v_proj", "o_proj"):
        return f"model.layers.{layer_idx}.self_attn.{module_name}.weight"
    elif module_name in ("gate_proj", "up_proj", "down_proj"):
        return f"model.layers.{layer_idx}.mlp.{module_name}.weight"
    else:
        raise ValueError(f"Unknown module: {module_name}")


def get_coop_prefix(layer_idx: int, module_name: str) -> str:
    """Get the cooperative LoRA key prefix for a target module."""
    if module_name in ("q_proj", "k_proj", "v_proj", "o_proj"):
        return f"base_model.model.language_model.layers.{layer_idx}.self_attn.{module_name}"
    elif module_name in ("gate_proj", "up_proj", "down_proj"):
        return f"base_model.model.language_model.layers.{layer_idx}.mlp.{module_name}"
    else:
        raise ValueError(f"Unknown module: {module_name}")


def load_shard_index(model_dir: str) -> dict:
    """Load model.safetensors.index.json and return key->shard mapping."""
    index_path = os.path.join(model_dir, "model.safetensors.index.json")
    with open(index_path) as f:
        index = json.load(f)
    return index["weight_map"]


def joint_svd_extract(
    delta_a: torch.Tensor,
    delta_b: torch.Tensor,
    rank: int,
    alpha: int,
    device: torch.device = None,
) -> tuple:
    """Joint SVD of two task vectors to extract shared B and two expert A matrices.

    Concatenates [delta_a | delta_b] horizontally and performs truncated SVD.
    Distributes singular values symmetrically (sqrt pattern).

    Args:
        delta_a: [out_f, in_f] task vector for model_a (expert 1)
        delta_b: [out_f, in_f] task vector for model_b (expert 2)
        rank: LoRA rank
        alpha: LoRA alpha
        device: GPU device for SVD computation

    Returns:
        lora_B:  [out_f, rank]  shared B matrix (bfloat16)
        lora_A_1: [rank, in_f]  expert 1 A matrix (bfloat16)
        lora_A_2: [rank, in_f]  expert 2 A matrix (bfloat16)
        error_a: relative reconstruction error for expert 1
        error_b: relative reconstruction error for expert 2
        energy: fraction of total energy captured
    """
    scaling = alpha / rank
    in_f = delta_a.shape[1]

    # Concatenate horizontally: [out_f, 2*in_f]
    M = torch.cat([delta_a.float(), delta_b.float()], dim=1)
    if device is not None:
        M = M.to(device)

    # Full SVD (we only need rank singular values)
    U, S, Vt = torch.linalg.svd(M, full_matrices=False)

    # Truncate to rank
    U_r = U[:, :rank]       # [out_f, rank]
    S_r = S[:rank]           # [rank]
    Vt_r = Vt[:rank, :]     # [rank, 2*in_f]

    # Energy captured
    total_energy = (S ** 2).sum()
    captured_energy = ((S_r ** 2).sum() / total_energy).item() if total_energy > 0 else 1.0

    # Distribute singular values and account for LoRA scaling
    sqrt_s = torch.sqrt(S_r / scaling)

    lora_B = U_r * sqrt_s.unsqueeze(0)             # [out_f, rank]
    Vt_scaled = Vt_r * sqrt_s.unsqueeze(1)          # [rank, 2*in_f]
    lora_A_1 = Vt_scaled[:, :in_f]                  # [rank, in_f]  expert 1
    lora_A_2 = Vt_scaled[:, in_f:]                   # [rank, in_f]  expert 2

    # Reconstruction errors
    # B @ A_1 * scaling should ≈ delta_a
    delta_a_gpu = delta_a.float().to(M.device)
    delta_b_gpu = delta_b.float().to(M.device)
    recon_a = (lora_B @ lora_A_1) * scaling
    recon_b = (lora_B @ lora_A_2) * scaling

    norm_a = torch.norm(delta_a_gpu)
    norm_b = torch.norm(delta_b_gpu)
    error_a = (torch.norm(delta_a_gpu - recon_a) / norm_a).item() if norm_a > 0 else 0.0
    error_b = (torch.norm(delta_b_gpu - recon_b) / norm_b).item() if norm_b > 0 else 0.0

    # Move to CPU, convert to bfloat16, ensure contiguous
    return (
        lora_B.cpu().to(torch.bfloat16).contiguous(),
        lora_A_1.cpu().to(torch.bfloat16).contiguous(),
        lora_A_2.cpu().to(torch.bfloat16).contiguous(),
        error_a,
        error_b,
        captured_energy,
    )


def convert(
    base_model_dir: str,
    model_a_dir: str,
    model_b_dir: str,
    output_dir: str,
    rank: int = 128,
    alpha: int = 256,
    num_comm_rounds: int = 2,
    target_modules: list = None,
):
    if target_modules is None:
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]

    scaling = alpha / rank

    log(f"{'=' * 70}")
    log(f"  Joint SVD: Full-Param SFT × 2 → Cooperative LoRA (V13)")
    log(f"{'=' * 70}")
    log(f"  Base model:  {base_model_dir}")
    log(f"  Model A:     {model_a_dir}  (→ expert 1)")
    log(f"  Model B:     {model_b_dir}  (→ expert 2)")
    log(f"  Output:      {output_dir}")
    log(f"  Rank: {rank}, Alpha: {alpha}, Scaling: {scaling:.1f}")
    log(f"  Target modules: {target_modules}")
    log(f"  Comm rounds: {num_comm_rounds}")
    log(f"  Layers: {NUM_LAYERS}")
    log()

    # Setup GPU
    device = None
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        log(f"  Device: {torch.cuda.get_device_name(0)} (GPU-accelerated)")
    else:
        log(f"  Device: CPU (this will be slow!)")
    log()

    t0 = time.time()

    # 1. Load shard indices from all 3 models
    log("Loading shard indices...")
    base_map = load_shard_index(base_model_dir)
    map_a = load_shard_index(model_a_dir)
    map_b = load_shard_index(model_b_dir)

    # 2. Determine which HF keys we need (target modules only)
    target_hf_keys = {}  # hf_key -> (layer_idx, module_name)
    for layer_idx in range(NUM_LAYERS):
        for mod in target_modules:
            hf_key = get_hf_key(layer_idx, mod)
            target_hf_keys[hf_key] = (layer_idx, mod)

    log(f"  Target keys: {len(target_hf_keys)} modules across {NUM_LAYERS} layers")

    # 3. Group target keys by base shard for memory-efficient processing
    shard_groups = defaultdict(list)
    for hf_key in target_hf_keys:
        if hf_key in base_map:
            shard_groups[base_map[hf_key]].append(hf_key)
        else:
            log(f"  WARNING: {hf_key} not found in base model, skipping")

    base_shards = sorted(shard_groups.keys())
    log(f"  Grouped into {len(base_shards)} base shards")
    log()

    # 4. Process shard by shard
    coop_state = {}
    errors_a = []
    errors_b = []
    energies = []
    total_modules = len(target_hf_keys)
    done = 0

    log(f"{'#':<5} {'Layer':<7} {'Module':<10} {'Shape':<22} "
        f"{'ErrA':>7} {'ErrB':>7} {'Energy':>8} {'Time':>6}")
    log("-" * 80)

    t_svd_start = time.time()

    for shard_name in base_shards:
        keys_in_shard = shard_groups[shard_name]
        log(f"\n  Loading shard: {shard_name} ({len(keys_in_shard)} target keys)")

        # Load base shard
        base_shard = load_file(os.path.join(base_model_dir, shard_name))

        # Find which model_a and model_b shards we need for these keys
        needed_a_shards = set(map_a[k] for k in keys_in_shard if k in map_a)
        needed_b_shards = set(map_b[k] for k in keys_in_shard if k in map_b)

        shard_a = {}
        for s in needed_a_shards:
            shard_a.update(load_file(os.path.join(model_a_dir, s)))

        shard_b = {}
        for s in needed_b_shards:
            shard_b.update(load_file(os.path.join(model_b_dir, s)))

        # Process each target module in this shard
        for hf_key in sorted(keys_in_shard):
            layer_idx, mod_name = target_hf_keys[hf_key]
            done += 1
            t_mod = time.time()

            w_base = base_shard[hf_key]
            w_a = shard_a[hf_key]
            w_b = shard_b[hf_key]

            delta_a = w_a - w_base  # [out_f, in_f]
            delta_b = w_b - w_base

            # Joint SVD
            lora_B, lora_A_1, lora_A_2, err_a, err_b, energy = joint_svd_extract(
                delta_a, delta_b, rank, alpha, device=device,
            )

            # Store in cooperative key format
            prefix = get_coop_prefix(layer_idx, mod_name)
            coop_state[f"{prefix}.lora_B"] = lora_B
            coop_state[f"{prefix}.lora_A_1"] = lora_A_1
            coop_state[f"{prefix}.lora_A_2"] = lora_A_2

            errors_a.append(err_a)
            errors_b.append(err_b)
            energies.append(energy)

            elapsed_mod = time.time() - t_mod
            elapsed_total = time.time() - t_svd_start
            eta = elapsed_total / done * (total_modules - done)
            shape_str = f"{list(delta_a.shape)}"
            log(f"  {done:>3}/{total_modules} L{layer_idx:<4} {mod_name:<10} {shape_str:<22} "
                f"{err_a:>6.4f} {err_b:>6.4f} {energy:>7.4f} {elapsed_mod:>5.1f}s "
                f"[ETA {eta:.0f}s]")

        # Free shard memory
        del base_shard, shard_a, shard_b

    log("-" * 80)
    svd_time = time.time() - t_svd_start
    log(f"  Joint SVD completed in {svd_time:.1f}s ({svd_time/max(done,1):.2f}s/module)")
    log()

    # 5. Route weights: zeros → sigmoid(0) = 0.5 (equal blend)
    dtype = torch.bfloat16
    route_state = {}
    for layer_idx in range(NUM_LAYERS):
        route_state[f"route_weights.{layer_idx}"] = torch.zeros(HIDDEN_SIZE, dtype=dtype)

    # 6. Comm weights: Kaiming W, zero gates
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

    # 7. Save
    os.makedirs(output_dir, exist_ok=True)
    torch.save(coop_state, os.path.join(output_dir, "lora_weights.pt"))
    torch.save(route_state, os.path.join(output_dir, "route_weights.pt"))
    torch.save(comm_state, os.path.join(output_dir, "comm_weights.pt"))

    config = {
        "lora_r": rank,
        "lora_alpha": alpha,
        "target_modules": target_modules,
        "balance_weight": 0.01,
        "num_comm_rounds": num_comm_rounds,
        "type": "iterative_cooperative_v13",
        "conversion_info": {
            "source": "joint_svd_fullsft",
            "base_model": os.path.abspath(base_model_dir),
            "model_a": os.path.abspath(model_a_dir),
            "model_b": os.path.abspath(model_b_dir),
            "rank": rank,
            "alpha": alpha,
        },
    }
    with open(os.path.join(output_dir, "cooperative_config.json"), "w") as f:
        json.dump(config, f, indent=2)

    # 8. Statistics
    lora_params = sum(t.numel() for t in coop_state.values())
    route_params = sum(t.numel() for t in route_state.values())
    comm_params = sum(t.numel() for t in comm_state.values())

    log(f"{'=' * 70}")
    log(f"  Conversion Statistics (rank={rank})")
    log(f"{'=' * 70}")
    log(f"  Modules converted: {len(coop_state) // 3}")
    log(f"  Total parameters:  {(lora_params + route_params + comm_params) / 1e6:.1f}M")
    log(f"    LoRA:   {lora_params / 1e6:.1f}M ({len(coop_state)} tensors)")
    log(f"    Route:  {route_params / 1e6:.2f}M ({len(route_state)} tensors)")
    log(f"    Comm:   {comm_params / 1e6:.2f}M ({len(comm_state)} tensors)")
    log()
    log(f"  Reconstruction error (||delta - B@A*s|| / ||delta||):")
    log(f"    Expert 1 (model_a): mean={sum(errors_a)/len(errors_a):.6f}, "
        f"max={max(errors_a):.6f}, min={min(errors_a):.6f}")
    log(f"    Expert 2 (model_b): mean={sum(errors_b)/len(errors_b):.6f}, "
        f"max={max(errors_b):.6f}, min={min(errors_b):.6f}")
    log()
    log(f"  Energy captured (joint SVD top-{rank}):")
    log(f"    Mean: {sum(energies)/len(energies)*100:.2f}%")
    log(f"    Min:  {min(energies)*100:.2f}%")
    log()
    log(f"  Saved to: {output_dir}")
    log(f"    lora_weights.pt")
    log(f"    route_weights.pt")
    log(f"    comm_weights.pt")
    log(f"    cooperative_config.json")
    log()
    log(f"  Key sample: {next(iter(coop_state))}")
    log(f"  Total time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert two full-param SFT models to cooperative LoRA (V13) via joint SVD"
    )
    parser.add_argument("--base_model", required=True,
                        help="Base model directory (e.g. Qwen2.5-VL-7B-Instruct)")
    parser.add_argument("--model_a", required=True,
                        help="First fine-tuned model directory (→ expert 1)")
    parser.add_argument("--model_b", required=True,
                        help="Second fine-tuned model directory (→ expert 2)")
    parser.add_argument("--output_dir", required=True,
                        help="Output cooperative checkpoint directory")
    parser.add_argument("--rank", type=int, default=128,
                        help="LoRA rank (default: 128)")
    parser.add_argument("--alpha", type=int, default=256,
                        help="LoRA alpha (default: 256)")
    parser.add_argument("--num_comm_rounds", type=int, default=2,
                        help="Number of communication rounds (default: 2)")
    parser.add_argument("--target_modules", type=str,
                        default="q_proj,k_proj,v_proj,o_proj",
                        help="Comma-separated target modules (default: q_proj,k_proj,v_proj,o_proj)")
    args = parser.parse_args()

    target_modules = [m.strip() for m in args.target_modules.split(",")]

    convert(
        base_model_dir=args.base_model,
        model_a_dir=args.model_a,
        model_b_dir=args.model_b,
        output_dir=args.output_dir,
        rank=args.rank,
        alpha=args.alpha,
        num_comm_rounds=args.num_comm_rounds,
        target_modules=target_modules,
    )
