#!/usr/bin/env python3
"""Extract cooperative LoRA weights from Full-Parameter SFT via SVD decomposition.

Takes a full-param SFT model and the original base model, computes the weight
difference for each target module, performs truncated SVD, and outputs a
cooperative LoRA checkpoint (V13 format: shared B, dual A_1/A_2).

Math:
  Forward pass at init (route_weight=0 → sigmoid=0.5):
    delta = B @ (0.5 * A_1 + 0.5 * A_2) @ x * (α/r)

  We want: B @ A_avg * (α/r) ≈ ΔW  where A_avg = 0.5*(A_1 + A_2)

  SVD: ΔW = U Σ V^T, truncate to rank r
    B     = U[:,:r] × diag(sqrt(Σ[:r] / scaling))     [out_f, r]
    A_avg = diag(sqrt(Σ[:r] / scaling)) × V[:r,:]      [r, in_f]
    A_1   = A_avg + δ                                   [r, in_f]
    A_2   = A_avg - δ                                   [r, in_f]
    δ     = noise_scale × randn × (||A_avg||_F / sqrt(r × in_f))

  With noise_scale=0: both experts identical, perfectly reconstructs rank-r ΔW.

Output format: V13 cooperative (IterativeCooperativeVLMWrapper)
  lora_weights.pt    — A_1, A_2, B per module
  route_weights.pt   — zeros (sigmoid(0) = 0.5 → equal blend)
  comm_weights.pt    — Kaiming-init W, zero gates
  cooperative_config.json

Usage:
    python v15_gui_360/extract_fullsft_to_cooperative.py \
        --sft_model checkpoints/gui360_balanced_full_sft/checkpoint-250 \
        --base_model checkpoints/Qwen2.5-VL-7B-Instruct \
        --output checkpoints/v15_svd_extracted_cooperative \
        --rank 128 --alpha 256 --noise_scale 0.1
"""

import argparse
import json
import math
import os
import sys
import time

import torch
from safetensors.torch import load_file


# Unbuffered output for SLURM
def log(msg=""):
    print(msg, flush=True)


NUM_LAYERS = 28
HIDDEN_SIZE = 3584  # Qwen2.5-VL-7B hidden_size


def load_model_weights(model_dir: str) -> dict:
    """Load all safetensors from a model directory into a single state dict."""
    index_file = os.path.join(model_dir, "model.safetensors.index.json")

    if os.path.exists(index_file):
        with open(index_file) as f:
            index = json.load(f)
        weight_map = index["weight_map"]

        state_dict = {}
        loaded_files = set()
        for key, filename in weight_map.items():
            if filename not in loaded_files:
                filepath = os.path.join(model_dir, filename)
                log(f"  Loading {filename}...")
                shard = load_file(filepath)
                state_dict.update(shard)
                loaded_files.add(filename)
        return state_dict
    else:
        filepath = os.path.join(model_dir, "model.safetensors")
        return load_file(filepath)


def get_hf_key(layer_idx: int, module_name: str) -> str:
    """Get the HuggingFace weight key for a target module."""
    if module_name in ("q_proj", "k_proj", "v_proj", "o_proj"):
        return f"model.layers.{layer_idx}.self_attn.{module_name}.weight"
    elif module_name in ("gate_proj", "up_proj", "down_proj"):
        return f"model.layers.{layer_idx}.mlp.{module_name}.weight"
    else:
        raise ValueError(f"Unknown module: {module_name}")


def detect_cooperative_prefix(base_weights: dict) -> str:
    """Detect the cooperative key prefix for V13 wrapper's named_parameters().

    The V13 wrapper's _get_transformer_layers checks:
      1. base_model.model.language_model.layers (Qwen2-VL, Qwen2.5-VL)
      2. base_model.model.layers (fallback)

    NOTE: HF safetensor keys omit 'language_model' even when the model has it
    (Qwen2.5-VL keys are 'model.layers.{L}...' but the actual nn.Module path
    includes 'language_model'). We check both HF keys and known architectures.
    """
    all_keys = base_weights.keys()

    # Explicit language_model in HF keys → use it directly
    if any("language_model" in k for k in all_keys):
        return "base_model.model.language_model"

    # Qwen2.5-VL: HF keys are model.layers.* but the ForConditionalGeneration
    # model wraps them as model.language_model.layers.* at the nn.Module level.
    # Detect VL models by the presence of visual encoder keys.
    has_visual = any("visual" in k for k in all_keys)
    has_model_layers = any(k.startswith("model.layers.") for k in all_keys)
    if has_visual and has_model_layers:
        return "base_model.model.language_model"

    return "base_model.model"


def get_coop_lora_key(prefix: str, layer_idx: int, module_name: str, lora_type: str) -> str:
    """Get the cooperative key matching V13 wrapper's named_parameters().

    Args:
        prefix: e.g. "base_model.model" or "base_model.model.language_model"
        lora_type: "lora_A_1", "lora_A_2", or "lora_B"
    """
    if module_name in ("q_proj", "k_proj", "v_proj", "o_proj"):
        return f"{prefix}.layers.{layer_idx}.self_attn.{module_name}.{lora_type}"
    elif module_name in ("gate_proj", "up_proj", "down_proj"):
        return f"{prefix}.layers.{layer_idx}.mlp.{module_name}.{lora_type}"
    else:
        raise ValueError(f"Unknown module: {module_name}")


def extract_cooperative_svd(
    delta_w: torch.Tensor,
    rank: int,
    alpha: int,
    noise_scale: float,
    device: torch.device = None,
    seed: int = 42,
) -> tuple:
    """Extract cooperative LoRA B, A_1, A_2 from weight delta via SVD.

    Returns:
        lora_B:  [out_features, rank] (CPU, original dtype)
        lora_A_1: [rank, in_features]
        lora_A_2: [rank, in_features]
        error: relative reconstruction error
        captured_energy: fraction of singular value energy captured
    """
    scaling = alpha / rank
    orig_dtype = delta_w.dtype

    delta_float = delta_w.float()
    if device is not None:
        delta_float = delta_float.to(device)

    U, S, Vt = torch.linalg.svd(delta_float, full_matrices=False)

    # Truncate to rank
    U_r = U[:, :rank]      # [out_features, rank]
    S_r = S[:rank]          # [rank]
    Vt_r = Vt[:rank, :]    # [rank, in_features]

    # Reconstruction error
    delta_approx = U_r @ torch.diag(S_r) @ Vt_r
    error = (torch.norm(delta_float - delta_approx) / torch.norm(delta_float)).item()

    # Energy captured
    total_energy = (S ** 2).sum()
    captured_energy = ((S_r ** 2).sum() / total_energy).item()

    # Distribute singular values and account for LoRA scaling
    # B @ A_avg * scaling ≈ ΔW → B @ A_avg ≈ ΔW / scaling
    # U_r @ diag(S_r) @ Vt_r ≈ ΔW
    # So: B @ A_avg = U_r @ diag(S_r / scaling) @ Vt_r
    sqrt_s = torch.sqrt(S_r / scaling)
    lora_B = U_r * sqrt_s.unsqueeze(0)       # [out_features, rank]
    A_avg = Vt_r * sqrt_s.unsqueeze(1)       # [rank, in_features]

    # Split A_avg into A_1, A_2 with perturbation for expert diversity
    if noise_scale > 0:
        gen = torch.Generator(device=delta_float.device)
        gen.manual_seed(seed)
        noise = torch.randn(A_avg.shape, generator=gen, device=A_avg.device, dtype=A_avg.dtype)
        r, in_f = A_avg.shape
        a_norm = torch.norm(A_avg)
        delta_pert = noise_scale * noise * (a_norm / math.sqrt(r * in_f))
    else:
        delta_pert = torch.zeros_like(A_avg)

    lora_A_1 = A_avg + delta_pert
    lora_A_2 = A_avg - delta_pert

    # Move to CPU and original dtype
    return (
        lora_B.cpu().to(orig_dtype).contiguous(),
        lora_A_1.cpu().to(orig_dtype).contiguous(),
        lora_A_2.cpu().to(orig_dtype).contiguous(),
        error,
        captured_energy,
    )


def extract(
    sft_model_dir: str,
    base_model_dir: str,
    output_dir: str,
    rank: int = 128,
    alpha: int = 256,
    num_comm_rounds: int = 2,
    noise_scale: float = 0.1,
    target_modules: list = None,
    use_gpu: bool = True,
):
    """Extract cooperative LoRA from full-param SFT via SVD."""
    if target_modules is None:
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]

    scaling = alpha / rank

    log(f"{'=' * 60}")
    log(f"  SVD Extraction: Full-Param SFT → Cooperative LoRA (V13)")
    log(f"{'=' * 60}")
    log(f"  SFT model:     {sft_model_dir}")
    log(f"  Base model:    {base_model_dir}")
    log(f"  Output:        {output_dir}")
    log(f"  Rank: {rank}, Alpha: {alpha}, Scaling: {scaling:.1f}")
    log(f"  Noise scale:   {noise_scale}")
    log(f"  Comm rounds:   {num_comm_rounds}")
    log(f"  Target modules: {target_modules}")
    log(f"  Layers: {NUM_LAYERS}")
    log(f"  Total SVDs: {NUM_LAYERS * len(target_modules)}")

    device = None
    if use_gpu and torch.cuda.is_available():
        device = torch.device("cuda:0")
        log(f"  Device: {torch.cuda.get_device_name(0)} (GPU-accelerated)")
    else:
        log(f"  Device: CPU (this will be slow!)")
    log()

    t0 = time.time()

    # 1. Load both models
    log("Loading SFT model weights...")
    sft_weights = load_model_weights(sft_model_dir)
    log(f"  Loaded {len(sft_weights)} weight tensors ({time.time() - t0:.1f}s)")

    log("Loading base model weights...")
    base_weights = load_model_weights(base_model_dir)
    log(f"  Loaded {len(base_weights)} weight tensors ({time.time() - t0:.1f}s)")
    log()

    # Detect cooperative key prefix from model structure
    coop_prefix = detect_cooperative_prefix(base_weights)
    log(f"Cooperative key prefix: {coop_prefix}")
    log()

    # 2. SVD extraction for each target module
    lora_state = {}
    errors = []
    energies = []
    delta_norms = []
    total_modules = NUM_LAYERS * len(target_modules)
    done = 0

    log(f"{'#':<5} {'Layer':<7} {'Module':<10} {'Shape':<20} "
        f"{'ΔW norm':>9} {'Error':>8} {'Energy':>8} {'Time':>7}")
    log("-" * 78)

    t_svd_start = time.time()

    for layer_idx in range(NUM_LAYERS):
        for module_name in target_modules:
            hf_key = get_hf_key(layer_idx, module_name)
            done += 1

            if hf_key not in sft_weights or hf_key not in base_weights:
                log(f"  WARNING: {hf_key} not found, skipping")
                continue

            w_sft = sft_weights[hf_key]
            w_base = base_weights[hf_key]
            delta_w = w_sft - w_base

            delta_norm = torch.norm(delta_w.float()).item()
            delta_norms.append(delta_norm)

            if delta_norm < 1e-8:
                log(f"  Layer {layer_idx} {module_name}: delta ~0, skipping")
                continue

            t_mod = time.time()

            # SVD extraction → B, A_1, A_2
            lora_B, lora_A_1, lora_A_2, error, energy = extract_cooperative_svd(
                delta_w, rank, alpha, noise_scale,
                device=device,
                seed=42 + layer_idx * len(target_modules) + target_modules.index(module_name),
            )

            elapsed_mod = time.time() - t_mod

            # Store with cooperative key format
            b_key = get_coop_lora_key(coop_prefix, layer_idx, module_name, "lora_B")
            a1_key = get_coop_lora_key(coop_prefix, layer_idx, module_name, "lora_A_1")
            a2_key = get_coop_lora_key(coop_prefix, layer_idx, module_name, "lora_A_2")
            lora_state[b_key] = lora_B
            lora_state[a1_key] = lora_A_1
            lora_state[a2_key] = lora_A_2

            errors.append(error)
            energies.append(energy)

            shape_str = f"{list(delta_w.shape)}"
            elapsed_total = time.time() - t_svd_start
            eta = elapsed_total / done * (total_modules - done)
            log(f"  {done:>3}/{total_modules} L{layer_idx:<4} {module_name:<10} {shape_str:<20} "
                f"{delta_norm:>8.3f} {error:>7.4f} {energy:>7.4f} {elapsed_mod:>5.1f}s "
                f"[ETA {eta:.0f}s]")

    log("-" * 78)
    svd_time = time.time() - t_svd_start
    log(f"  SVD completed in {svd_time:.1f}s ({svd_time/total_modules:.2f}s/module)")

    # 3. Initialize route_weights — zeros → sigmoid(0) = 0.5 (equal blend)
    route_state = {}
    dtype = torch.bfloat16
    for layer_idx in range(NUM_LAYERS):
        key = f"route_weights.{layer_idx}"
        route_state[key] = torch.zeros(HIDDEN_SIZE, dtype=dtype)

    # 4. Initialize comm_weights — Kaiming W, zero gates
    T = num_comm_rounds
    comm_state = {}
    for idx in range(NUM_LAYERS * T):
        # W matrices: Kaiming uniform [r, r]
        w_12 = torch.zeros(rank, rank, dtype=dtype)
        w_21 = torch.zeros(rank, rank, dtype=dtype)
        # Use float32 for init then convert
        w_12_f = torch.zeros(rank, rank)
        w_21_f = torch.zeros(rank, rank)
        torch.nn.init.kaiming_uniform_(w_12_f, a=math.sqrt(5))
        torch.nn.init.kaiming_uniform_(w_21_f, a=math.sqrt(5))
        comm_state[f"comm_W_12.{idx}"] = w_12_f.to(dtype)
        comm_state[f"comm_W_21.{idx}"] = w_21_f.to(dtype)

        # Gate vectors: zeros [r]
        comm_state[f"comm_gate_12.{idx}"] = torch.zeros(rank, dtype=dtype)
        comm_state[f"comm_gate_21.{idx}"] = torch.zeros(rank, dtype=dtype)

    # 5. Save
    os.makedirs(output_dir, exist_ok=True)

    torch.save(lora_state, os.path.join(output_dir, "lora_weights.pt"))
    torch.save(route_state, os.path.join(output_dir, "route_weights.pt"))
    torch.save(comm_state, os.path.join(output_dir, "comm_weights.pt"))

    config = {
        "lora_r": rank,
        "lora_alpha": alpha,
        "target_modules": target_modules,
        "balance_weight": 0.01,
        "num_comm_rounds": num_comm_rounds,
        "type": "iterative_cooperative_v13",
        "extraction_info": {
            "method": "svd",
            "sft_model": os.path.abspath(sft_model_dir),
            "base_model": os.path.abspath(base_model_dir),
            "noise_scale": noise_scale,
            "mean_error": sum(errors) / len(errors) if errors else 0,
            "mean_energy": sum(energies) / len(energies) if energies else 0,
        },
    }
    with open(os.path.join(output_dir, "cooperative_config.json"), "w") as f:
        json.dump(config, f, indent=2)

    # 6. Statistics
    log()
    log(f"{'=' * 60}")
    log(f"  Extraction Statistics (rank={rank})")
    log(f"{'=' * 60}")
    log(f"  LoRA tensors saved: {len(lora_state)} "
        f"({NUM_LAYERS} layers × {len(target_modules)} modules × 3)")
    total_params = sum(t.numel() for t in lora_state.values())
    route_params = sum(t.numel() for t in route_state.values())
    comm_params = sum(t.numel() for t in comm_state.values())
    log(f"  LoRA parameters:    {total_params:,} ({total_params / 1e6:.1f}M)")
    log(f"  Route parameters:   {route_params:,} ({route_params / 1e6:.2f}M)")
    log(f"  Comm parameters:    {comm_params:,} ({comm_params / 1e6:.2f}M)")
    log(f"  Total parameters:   {total_params + route_params + comm_params:,} "
        f"({(total_params + route_params + comm_params) / 1e6:.1f}M)")
    log()
    log(f"  Reconstruction error (||ΔW - ΔW_approx|| / ||ΔW||):")
    log(f"    Mean: {sum(errors) / len(errors):.6f}")
    log(f"    Max:  {max(errors):.6f}")
    log(f"    Min:  {min(errors):.6f}")
    log()
    log(f"  Energy captured (top-{rank} singular values / total):")
    log(f"    Mean: {sum(energies) / len(energies):.6f} "
        f"({sum(energies) / len(energies) * 100:.2f}%)")
    log(f"    Min:  {min(energies):.6f} ({min(energies) * 100:.2f}%)")
    log()
    log(f"  Weight delta norms (||W_sft - W_base||):")
    log(f"    Mean: {sum(delta_norms) / len(delta_norms):.4f}")
    log(f"    Max:  {max(delta_norms):.4f}")
    log(f"    Min:  {min(delta_norms):.4f}")

    # Rank analysis on representative layer
    log()
    log(f"  Rank analysis (Layer 14 q_proj):")
    sample_key = get_hf_key(14, "q_proj")
    if sample_key in sft_weights and sample_key in base_weights:
        delta_sample = (sft_weights[sample_key] - base_weights[sample_key]).float()
        if device is not None:
            delta_sample = delta_sample.to(device)
        _, S_full, _ = torch.linalg.svd(delta_sample, full_matrices=False)
        S_full = S_full.cpu()
        total_e = (S_full ** 2).sum()
        for r in [16, 32, 64, 128, 256]:
            e_r = (S_full[:r] ** 2).sum() / total_e
            marker = " <<<" if r == rank else ""
            log(f"    rank={r:3d}: energy={e_r:.4f} ({e_r * 100:.1f}%){marker}")

    log()
    log(f"  Total time: {time.time() - t0:.1f}s")
    log(f"  Output saved to: {output_dir}")
    log(f"    - lora_weights.pt   ({len(lora_state)} tensors)")
    log(f"    - route_weights.pt  ({len(route_state)} tensors)")
    log(f"    - comm_weights.pt   ({len(comm_state)} tensors)")
    log(f"    - cooperative_config.json")
    log()
    log(f"  Key format sample:")
    sample = next(iter(lora_state))
    log(f"    {sample}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract Cooperative LoRA from Full-Param SFT via SVD"
    )
    parser.add_argument("--sft_model", type=str, required=True,
                        help="Path to full-param SFT model directory")
    parser.add_argument("--base_model", type=str, required=True,
                        help="Path to original base model directory")
    parser.add_argument("--output", type=str, required=True,
                        help="Output directory for cooperative checkpoint")
    parser.add_argument("--rank", type=int, default=128,
                        help="LoRA rank (default: 128)")
    parser.add_argument("--alpha", type=int, default=256,
                        help="LoRA alpha (default: 256)")
    parser.add_argument("--num_comm_rounds", type=int, default=2,
                        help="Number of communication rounds (default: 2)")
    parser.add_argument("--noise_scale", type=float, default=0.1,
                        help="Expert diversity perturbation scale (default: 0.1)")
    parser.add_argument("--target_modules", nargs="+",
                        default=["q_proj", "k_proj", "v_proj", "o_proj"],
                        help="Which modules to extract LoRA for")
    args = parser.parse_args()

    extract(
        args.sft_model,
        args.base_model,
        args.output,
        rank=args.rank,
        alpha=args.alpha,
        num_comm_rounds=args.num_comm_rounds,
        noise_scale=args.noise_scale,
        target_modules=args.target_modules,
    )
