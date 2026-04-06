#!/usr/bin/env python3
"""
Extract LoRA weights from Full-Parameter SFT via SVD decomposition.

Takes a full-param SFT model and the original base model, computes the
weight difference for each target module, and extracts low-rank LoRA
approximations via truncated SVD.

Usage:
    python extract_fullsft_to_lora.py \
        --sft_model train_GUI_360/llamafactory/output/gui360_full_sft_v2 \
        --base_model checkpoints/Qwen2.5-VL-7B-Instruct \
        --output train_GUI_360/moe_rl/extracted_lora_from_fullsft \
        --rank 32 \
        --alpha 64
"""

import argparse
import json
import os
import sys
import time

import torch
from safetensors.torch import load_file, save_file


# Unbuffered output for SLURM
def log(msg=""):
    print(msg, flush=True)


# Qwen2.5-VL-7B target modules (matching LoRA v4 config)
TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

# Number of transformer layers in Qwen2.5-VL-7B
NUM_LAYERS = 28


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
                print(f"  Loading {filename}...")
                shard = load_file(filepath)
                state_dict.update(shard)
                loaded_files.add(filename)
        return state_dict
    else:
        filepath = os.path.join(model_dir, "model.safetensors")
        return load_file(filepath)


def get_hf_key(layer_idx: int, module_name: str) -> str:
    """Get the HuggingFace weight key for a target module."""
    if module_name in ["q_proj", "k_proj", "v_proj", "o_proj"]:
        return f"model.layers.{layer_idx}.self_attn.{module_name}.weight"
    elif module_name in ["gate_proj", "up_proj", "down_proj"]:
        return f"model.layers.{layer_idx}.mlp.{module_name}.weight"
    else:
        raise ValueError(f"Unknown module: {module_name}")


def get_peft_key(layer_idx: int, module_name: str, lora_type: str) -> str:
    """Get the PEFT-format key for LoRA weights."""
    if module_name in ["q_proj", "k_proj", "v_proj", "o_proj"]:
        prefix = f"base_model.model.model.layers.{layer_idx}.self_attn.{module_name}"
    elif module_name in ["gate_proj", "up_proj", "down_proj"]:
        prefix = f"base_model.model.model.layers.{layer_idx}.mlp.{module_name}"
    else:
        raise ValueError(f"Unknown module: {module_name}")
    return f"{prefix}.lora_{lora_type}.weight"


def extract_lora_svd(
    delta_w: torch.Tensor,
    rank: int,
    alpha: int,
    device: torch.device = None,
) -> tuple:
    """
    Extract LoRA A and B matrices from weight delta using SVD.

    LoRA forward: output = base(x) + (alpha/r) * B @ A @ x
    So we need: (alpha/r) * B @ A ≈ ΔW
    Therefore:  B @ A ≈ (r/alpha) * ΔW

    Args:
        delta_w: [out_features, in_features] weight difference
        rank: LoRA rank
        alpha: LoRA alpha (scaling)
        device: GPU device for SVD computation (None = CPU)

    Returns:
        lora_A: [rank, in_features] (CPU, original dtype)
        lora_B: [out_features, rank] (CPU, original dtype)
        error: relative reconstruction error
        captured_energy: fraction of energy captured
    """
    scaling = alpha / rank
    orig_dtype = delta_w.dtype

    # Move to GPU for fast SVD, use float32 for stability
    delta_float = delta_w.float()
    if device is not None:
        delta_float = delta_float.to(device)

    U, S, Vt = torch.linalg.svd(delta_float, full_matrices=False)

    # Truncate to rank
    U_r = U[:, :rank]       # [out_features, rank]
    S_r = S[:rank]           # [rank]
    Vt_r = Vt[:rank, :]     # [rank, in_features]

    # Reconstruction error (computed on GPU)
    delta_approx = U_r @ torch.diag(S_r) @ Vt_r
    error = (torch.norm(delta_float - delta_approx) / torch.norm(delta_float)).item()

    # Energy captured
    total_energy = (S ** 2).sum()
    captured_energy = ((S_r ** 2).sum() / total_energy).item()

    # Distribute singular values and account for LoRA scaling
    sqrt_s = torch.sqrt(S_r / scaling)
    lora_B = U_r * sqrt_s.unsqueeze(0)       # [out_features, rank]
    lora_A = Vt_r * sqrt_s.unsqueeze(1)      # [rank, in_features]

    # Move back to CPU and original dtype, ensure contiguous for safetensors
    return (
        lora_A.cpu().to(orig_dtype).contiguous(),
        lora_B.cpu().to(orig_dtype).contiguous(),
        error,
        captured_energy,
    )


def extract(
    sft_model_dir: str,
    base_model_dir: str,
    output_dir: str,
    rank: int = 32,
    alpha: int = 64,
    use_gpu: bool = True,
):
    """Extract LoRA from full-param SFT via SVD (GPU-accelerated)."""
    log(f"{'=' * 60}")
    log(f"  SVD Extraction: Full-Param SFT → LoRA")
    log(f"{'=' * 60}")
    log(f"  SFT model:  {sft_model_dir}")
    log(f"  Base model: {base_model_dir}")
    log(f"  Output:     {output_dir}")
    log(f"  Rank: {rank}, Alpha: {alpha}, Scaling: {alpha/rank:.1f}")
    log(f"  Target modules: {TARGET_MODULES}")
    log(f"  Layers: {NUM_LAYERS}")
    log(f"  Total SVDs: {NUM_LAYERS * len(TARGET_MODULES)}")

    # Setup device
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

    # 2. Extract LoRA for each target module
    peft_state_dict = {}
    errors = []
    energies = []
    delta_norms = []
    total_modules = NUM_LAYERS * len(TARGET_MODULES)
    done = 0

    log(f"{'#':<5} {'Layer':<7} {'Module':<10} {'Shape':<20} {'ΔW norm':>9} {'Error':>8} {'Energy':>8} {'Time':>7}")
    log("-" * 78)

    t_svd_start = time.time()

    for layer_idx in range(NUM_LAYERS):
        for module_name in TARGET_MODULES:
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

            # SVD extraction (on GPU if available)
            lora_A, lora_B, error, energy = extract_lora_svd(
                delta_w, rank, alpha, device=device
            )

            elapsed_mod = time.time() - t_mod

            # Store in PEFT format
            a_key = get_peft_key(layer_idx, module_name, "A")
            b_key = get_peft_key(layer_idx, module_name, "B")
            peft_state_dict[a_key] = lora_A
            peft_state_dict[b_key] = lora_B

            errors.append(error)
            energies.append(energy)

            # Progress: print every module
            shape_str = f"{list(delta_w.shape)}"
            elapsed_total = time.time() - t_svd_start
            eta = elapsed_total / done * (total_modules - done)
            log(f"  {done:>3}/{total_modules} L{layer_idx:<4} {module_name:<10} {shape_str:<20} "
                f"{delta_norm:>8.3f} {error:>7.4f} {energy:>7.4f} {elapsed_mod:>5.1f}s "
                f"[ETA {eta:.0f}s]")

    log("-" * 78)
    svd_time = time.time() - t_svd_start
    log(f"  SVD completed in {svd_time:.1f}s ({svd_time/total_modules:.2f}s/module)")

    # 3. Save in PEFT format
    os.makedirs(output_dir, exist_ok=True)

    save_file(peft_state_dict, os.path.join(output_dir, "adapter_model.safetensors"))

    adapter_config = {
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "r": rank,
        "lora_alpha": alpha,
        "target_modules": TARGET_MODULES,
        "lora_dropout": 0.05,
        "bias": "none",
        "base_model_name_or_path": os.path.abspath(base_model_dir),
    }
    with open(os.path.join(output_dir, "adapter_config.json"), "w") as f:
        json.dump(adapter_config, f, indent=2)

    # 4. Statistics
    log()
    log(f"{'=' * 60}")
    log(f"  Extraction Statistics (rank={rank})")
    log(f"{'=' * 60}")
    log(f"  LoRA pairs extracted: {len(peft_state_dict) // 2}")
    total_params = sum(t.numel() for t in peft_state_dict.values())
    log(f"  Total LoRA parameters: {total_params:,} ({total_params / 1e6:.1f}M)")
    log()
    log(f"  Reconstruction error (||ΔW - ΔW_approx|| / ||ΔW||):")
    log(f"    Mean: {sum(errors) / len(errors):.6f}")
    log(f"    Max:  {max(errors):.6f}")
    log(f"    Min:  {min(errors):.6f}")
    log()
    log(f"  Energy captured (top-{rank} singular values / total):")
    log(f"    Mean: {sum(energies) / len(energies):.6f} ({sum(energies) / len(energies) * 100:.2f}%)")
    log(f"    Min:  {min(energies):.6f} ({min(energies) * 100:.2f}%)")
    log()
    log(f"  Weight delta norms (||W_sft - W_base||):")
    log(f"    Mean: {sum(delta_norms) / len(delta_norms):.4f}")
    log(f"    Max:  {max(delta_norms):.4f}")
    log(f"    Min:  {min(delta_norms):.4f}")

    # 5. Rank analysis on representative layers
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
    log()
    log(f"  Next steps:")
    log(f"    1. Merge & evaluate:  llamafactory-cli export + vLLM eval")
    log(f"    2. Convert to MoE:    python convert_sft_lora_to_moe.py --checkpoint {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract LoRA from Full-Param SFT via SVD"
    )
    parser.add_argument("--sft_model", type=str, required=True,
                        help="Path to full-param SFT model directory")
    parser.add_argument("--base_model", type=str, required=True,
                        help="Path to original base model directory")
    parser.add_argument("--output", type=str, required=True,
                        help="Output directory for extracted LoRA (PEFT format)")
    parser.add_argument("--rank", type=int, default=32,
                        help="LoRA rank (default: 32)")
    parser.add_argument("--alpha", type=int, default=64,
                        help="LoRA alpha (default: 64)")
    args = parser.parse_args()

    extract(args.sft_model, args.base_model, args.output, args.rank, args.alpha)
