#!/usr/bin/env python3
"""Extract K-expert cooperative LoRA weights from Full-Parameter SFT via SVD.

Takes a full-param SFT model and the original base model, computes the weight
difference for each target module, performs truncated SVD, and outputs a
K-expert cooperative LoRA checkpoint (V18 format: shared B, K A matrices).

Math:
  Forward pass at init (route_weight=0 → softmax=uniform 1/K):
    delta = B @ (1/K * Σ_k A_k) @ x * (α/r)

  We want: B @ A_avg * (α/r) ≈ ΔW  where A_avg = mean(A_k)

  SVD: ΔW = U Σ V^T, truncate to rank r
    B     = U[:,:r] × diag(sqrt(Σ[:r] / scaling))     [out_f, r]
    A_avg = diag(sqrt(Σ[:r] / scaling)) × V[:r,:]      [r, in_f]

  K-way Gram-Schmidt perturbation:
    1. Generate K random noise vectors in R^{r × in_f}
    2. Gram-Schmidt orthogonalize → K orthogonal directions
    3. Scale by noise_scale * ||A_avg||_F / sqrt(r * in_f)
    4. Center: δ_i = scaled_i - mean(scaled) → sum(δ_i) = 0
    5. A_i = A_avg + δ_i

  Property: mean(A_i) = A_avg → at uniform routing, output ≈ ΔW · x

Output format: V18 K-expert cooperative (KExpertCooperativeVLMWrapper)
  lora_weights.pt    — K A matrices + B per module
  route_weights.pt   — zeros [hidden_size, K] → softmax = uniform 1/K
  comm_weights.pt    — Kaiming-init W, zero gates (topology-dependent)
  cooperative_config.json

Usage:
    python v18_k_expert/extract_fullsft_to_k_expert.py \
        --sft_model checkpoints/gui360_full_sft_step250 \
        --base_model checkpoints/Qwen2.5-VL-7B-Instruct \
        --output checkpoints/v18_svd_k4_extracted \
        --rank 128 --alpha 256 --noise_scale 0.1 \
        --num_experts 4 --comm_topology top2
"""

import argparse
import json
import math
import os
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
    """Detect the cooperative key prefix for V18 wrapper's named_parameters()."""
    all_keys = base_weights.keys()

    if any("language_model" in k for k in all_keys):
        return "base_model.model.language_model"

    has_visual = any("visual" in k for k in all_keys)
    has_model_layers = any(k.startswith("model.layers.") for k in all_keys)
    if has_visual and has_model_layers:
        return "base_model.model.language_model"

    return "base_model.model"


def get_coop_lora_key(prefix: str, layer_idx: int, module_name: str, lora_type: str) -> str:
    """Get the cooperative key matching V18 wrapper's named_parameters().

    Args:
        prefix: e.g. "base_model.model" or "base_model.model.language_model"
        lora_type: "lora_A.{i}" (ParameterList index) or "lora_B"
    """
    if module_name in ("q_proj", "k_proj", "v_proj", "o_proj"):
        return f"{prefix}.layers.{layer_idx}.self_attn.{module_name}.{lora_type}"
    elif module_name in ("gate_proj", "up_proj", "down_proj"):
        return f"{prefix}.layers.{layer_idx}.mlp.{module_name}.{lora_type}"
    else:
        raise ValueError(f"Unknown module: {module_name}")


def gram_schmidt_orthogonalize(vectors: list) -> list:
    """Gram-Schmidt orthogonalization for a list of vectors (flattened tensors).

    Args:
        vectors: list of K tensors, each [r, in_f]

    Returns:
        list of K orthogonalized tensors, each [r, in_f]
    """
    result = []
    for v in vectors:
        v_flat = v.reshape(-1).clone()
        for u in result:
            u_flat = u.reshape(-1)
            proj = (v_flat @ u_flat) / (u_flat @ u_flat + 1e-10)
            v_flat = v_flat - proj * u_flat
        norm = v_flat.norm()
        if norm > 1e-10:
            v_flat = v_flat / norm
        result.append(v_flat.reshape(v.shape))
    return result


def extract_k_expert_svd(
    delta_w: torch.Tensor,
    rank: int,
    alpha: int,
    num_experts: int,
    noise_scale: float,
    device: torch.device = None,
    seed: int = 42,
) -> tuple:
    """Extract K-expert cooperative LoRA B, [A_0,...,A_{K-1}] from weight delta via SVD.

    Uses K-way Gram-Schmidt perturbation for maximum initial diversity.

    Returns:
        lora_B:  [out_features, rank]
        lora_As: list of K [rank, in_features] tensors
        error: relative reconstruction error
        captured_energy: fraction of singular value energy captured
    """
    scaling = alpha / rank
    orig_dtype = delta_w.dtype
    K = num_experts

    delta_float = delta_w.float()
    if device is not None:
        delta_float = delta_float.to(device)

    U, S, Vt = torch.linalg.svd(delta_float, full_matrices=False)

    # Truncate to rank
    U_r = U[:, :rank]
    S_r = S[:rank]
    Vt_r = Vt[:rank, :]

    # Reconstruction error
    delta_approx = U_r @ torch.diag(S_r) @ Vt_r
    error = (torch.norm(delta_float - delta_approx) / torch.norm(delta_float)).item()

    # Energy captured
    total_energy = (S ** 2).sum()
    captured_energy = ((S_r ** 2).sum() / total_energy).item()

    # Distribute singular values and account for LoRA scaling
    sqrt_s = torch.sqrt(S_r / scaling)
    lora_B = U_r * sqrt_s.unsqueeze(0)       # [out_features, rank]
    A_avg = Vt_r * sqrt_s.unsqueeze(1)        # [rank, in_features]

    # K-way Gram-Schmidt perturbation
    if noise_scale > 0 and K > 1:
        gen = torch.Generator(device=delta_float.device)
        gen.manual_seed(seed)

        r, in_f = A_avg.shape
        a_norm = torch.norm(A_avg)
        scale_factor = noise_scale * a_norm / math.sqrt(r * in_f)

        # 1. Generate K random noise vectors
        noise_vecs = [
            torch.randn(r, in_f, generator=gen, device=A_avg.device, dtype=A_avg.dtype)
            for _ in range(K)
        ]

        # 2. Gram-Schmidt orthogonalize
        ortho_vecs = gram_schmidt_orthogonalize(noise_vecs)

        # 3. Scale
        scaled_vecs = [v * scale_factor for v in ortho_vecs]

        # 4. Center: δ_i = scaled_i - mean(scaled) → sum(δ_i) = 0
        mean_vec = sum(scaled_vecs) / K
        perturbations = [v - mean_vec for v in scaled_vecs]

        # 5. A_i = A_avg + δ_i
        lora_As = [A_avg + pert for pert in perturbations]
    else:
        # No perturbation: all experts identical
        lora_As = [A_avg.clone() for _ in range(K)]

    # Verify centroid preservation
    A_reconstructed = sum(lora_As) / K
    centroid_error = torch.norm(A_reconstructed - A_avg) / (torch.norm(A_avg) + 1e-10)
    if centroid_error > 1e-5:
        log(f"  WARNING: centroid preservation error = {centroid_error.item():.8f}")

    # Move to CPU and original dtype
    lora_B_out = lora_B.cpu().to(orig_dtype).contiguous()
    lora_As_out = [a.cpu().to(orig_dtype).contiguous() for a in lora_As]

    return lora_B_out, lora_As_out, error, captured_energy


def extract(
    sft_model_dir: str,
    base_model_dir: str,
    output_dir: str,
    rank: int = 128,
    alpha: int = 256,
    num_comm_rounds: int = 2,
    noise_scale: float = 0.1,
    num_experts: int = 4,
    comm_topology: str = "top2",
    target_modules: list = None,
    use_gpu: bool = True,
):
    """Extract K-expert cooperative LoRA from full-param SFT via SVD."""
    if target_modules is None:
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]

    K = num_experts
    scaling = alpha / rank

    log(f"{'=' * 60}")
    log(f"  SVD Extraction: Full-Param SFT → K-Expert Cooperative LoRA (V18)")
    log(f"{'=' * 60}")
    log(f"  SFT model:     {sft_model_dir}")
    log(f"  Base model:    {base_model_dir}")
    log(f"  Output:        {output_dir}")
    log(f"  Rank: {rank}, Alpha: {alpha}, Scaling: {scaling:.1f}")
    log(f"  Num experts:   {K}")
    log(f"  Comm topology: {comm_topology}")
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

            # SVD extraction → B, [A_0, ..., A_{K-1}]
            lora_B, lora_As, error, energy = extract_k_expert_svd(
                delta_w, rank, alpha, num_experts, noise_scale,
                device=device,
                seed=42 + layer_idx * len(target_modules) + target_modules.index(module_name),
            )

            elapsed_mod = time.time() - t_mod

            # Store with cooperative key format
            b_key = get_coop_lora_key(coop_prefix, layer_idx, module_name, "lora_B")
            lora_state[b_key] = lora_B

            for i in range(K):
                a_key = get_coop_lora_key(coop_prefix, layer_idx, module_name, f"lora_A.{i}")
                lora_state[a_key] = lora_As[i]

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

    # 3. Initialize route_weights — zeros [hidden_size, K] → softmax = uniform 1/K
    route_state = {}
    dtype = torch.bfloat16
    for layer_idx in range(NUM_LAYERS):
        key = f"route_weights.{layer_idx}"
        route_state[key] = torch.zeros(HIDDEN_SIZE, K, dtype=dtype)

    # 4. Initialize comm_weights — topology-dependent
    T = num_comm_rounds
    comm_state = {}

    if comm_topology == "none":
        pass  # No comm params
    elif comm_topology in ("top2", "shared"):
        for idx in range(NUM_LAYERS * T):
            w_f = torch.zeros(rank, rank)
            torch.nn.init.kaiming_uniform_(w_f, a=math.sqrt(5))
            comm_state[f"comm_W.{idx}"] = w_f.to(dtype)
            comm_state[f"comm_gate.{idx}"] = torch.zeros(rank, dtype=dtype)
    elif comm_topology == "full":
        num_pairs = K * (K - 1)
        for idx in range(NUM_LAYERS * T * num_pairs):
            w_f = torch.zeros(rank, rank)
            torch.nn.init.kaiming_uniform_(w_f, a=math.sqrt(5))
            comm_state[f"comm_W.{idx}"] = w_f.to(dtype)
            comm_state[f"comm_gate.{idx}"] = torch.zeros(rank, dtype=dtype)

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
        "diversity_weight": 0.001,
        "num_comm_rounds": num_comm_rounds,
        "num_experts": num_experts,
        "comm_topology": comm_topology,
        "type": "k_expert_cooperative_v18",
        "extraction_info": {
            "method": "svd_gram_schmidt",
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
    log(f"  Extraction Statistics (rank={rank}, K={K})")
    log(f"{'=' * 60}")
    log(f"  LoRA tensors saved: {len(lora_state)} "
        f"({NUM_LAYERS} layers × {len(target_modules)} modules × (1 B + {K} A))")
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
        description="Extract K-Expert Cooperative LoRA from Full-Param SFT via SVD"
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
    parser.add_argument("--num_experts", type=int, default=4,
                        help="Number of experts K (default: 4)")
    parser.add_argument("--comm_topology", type=str, default="top2",
                        choices=["none", "top2", "shared", "full"],
                        help="Communication topology (default: top2)")
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
        num_experts=args.num_experts,
        comm_topology=args.comm_topology,
        target_modules=args.target_modules,
    )
