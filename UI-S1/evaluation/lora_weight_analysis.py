#!/usr/bin/env python3
"""
LoRA Weight Analysis (Experiment 5)

Analyzes a trained LoRA checkpoint to see if uniform LoRA implicitly
forms stage-wise specialization patterns.

Metrics per layer:
  - Effective rank (entropy of singular values)
  - Frobenius norm (magnitude of change to base model)
  - Inter-layer weight similarity (do layers cluster by stage?)

Usage:
  python lora_weight_analysis.py \
      --model_path checkpoints/Qwen2.5-VL-7B-Instruct \
      --lora_path results/stagewise_lora_A/final \
      --output_dir results/lora_analysis_A
"""

import argparse
import json
import os
import re
import sys
import numpy as np
import torch

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

LAYER_GROUPS = {
    "encode": list(range(0, 10)),
    "bind":   list(range(10, 19)),
    "ground": list(range(19, 28)),
}

LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"]


def load_lora_weights(lora_path):
    """Load LoRA adapter weights from checkpoint."""
    from safetensors.torch import load_file

    adapter_path = os.path.join(lora_path, "adapter_model.safetensors")
    if os.path.exists(adapter_path):
        state_dict = load_file(adapter_path)
    else:
        # Try .bin format
        bin_path = os.path.join(lora_path, "adapter_model.bin")
        state_dict = torch.load(bin_path, map_location="cpu")

    return state_dict


def extract_lora_pairs(state_dict):
    """Extract (lora_A, lora_B) pairs per layer per module."""
    pairs = {}  # {(layer_id, module_name): (A, B)}

    for key, tensor in state_dict.items():
        # Match patterns like: base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight
        m = re.search(r'layers\.(\d+)\.[^.]+\.(\w+_proj)\.lora_([AB])', key)
        if m is None:
            continue
        layer_id = int(m.group(1))
        module = m.group(2)
        ab = m.group(3)

        k = (layer_id, module)
        if k not in pairs:
            pairs[k] = {}
        pairs[k][ab] = tensor.float()

    return pairs


def compute_effective_rank(singular_values):
    """Entropy-based effective rank."""
    sv = singular_values[singular_values > 1e-10]
    if len(sv) == 0:
        return 0.0
    p = sv / sv.sum()
    entropy = -(p * torch.log(p)).sum().item()
    return np.exp(entropy)


def analyze(lora_path, output_dir):
    print(f"Loading LoRA from {lora_path}")
    state_dict = load_lora_weights(lora_path)
    pairs = extract_lora_pairs(state_dict)

    if not pairs:
        print("ERROR: No LoRA weight pairs found.")
        return

    print(f"Found {len(pairs)} LoRA (A, B) pairs across {len(set(k[0] for k in pairs))} layers")

    # ── Per-layer metrics ──
    layer_metrics = {}
    layer_flat_weights = {}  # for inter-layer similarity

    for (layer_id, module), ab_dict in sorted(pairs.items()):
        if "A" not in ab_dict or "B" not in ab_dict:
            continue

        A = ab_dict["A"]  # (rank, in_dim)
        B = ab_dict["B"]  # (out_dim, rank)
        rank = A.shape[0]

        # Efficient SVD: instead of SVD on full (d_out, d_in) delta_W,
        # use QR(B) to reduce to (r, d_in) SVD
        Q, R = torch.linalg.qr(B)  # Q: (d_out, r), R: (r, r)
        P = R @ A                    # (r, d_in) — much smaller
        S = torch.linalg.svdvals(P)  # singular values of delta_W
        eff_rank = compute_effective_rank(S)
        frob_norm = float(S.norm())  # ||delta_W||_F = ||S||_2
        spectral_norm = float(S[0]) if len(S) > 0 else 0.0

        k = f"L{layer_id}_{module}"
        layer_metrics[k] = {
            "layer_id": layer_id,
            "module": module,
            "rank": rank,
            "effective_rank": round(eff_rank, 3),
            "frobenius_norm": round(frob_norm, 6),
            "spectral_norm": round(spectral_norm, 6),
            "sv_ratio": round(float(S[0] / S.sum()) if S.sum() > 0 else 0, 4),
        }

        # Store flattened P (=R@A, compact repr) for inter-layer similarity
        if layer_id not in layer_flat_weights:
            layer_flat_weights[layer_id] = []
        layer_flat_weights[layer_id].append(P.flatten())

    # ── Aggregate per layer ──
    layer_agg = {}
    for layer_id in range(28):
        mets = [v for k, v in layer_metrics.items() if v["layer_id"] == layer_id]
        if not mets:
            continue
        layer_agg[layer_id] = {
            "mean_eff_rank": round(np.mean([m["effective_rank"] for m in mets]), 3),
            "mean_frob_norm": round(np.mean([m["frobenius_norm"] for m in mets]), 6),
            "mean_spectral_norm": round(np.mean([m["spectral_norm"] for m in mets]), 6),
            "mean_sv_ratio": round(np.mean([m["sv_ratio"] for m in mets]), 4),
        }

    # ── Inter-layer similarity matrix ──
    print("\nComputing inter-layer similarity...")
    layer_ids = sorted(layer_flat_weights.keys())
    # Concatenate all modules per layer into one vector
    layer_vectors = {}
    for lid in layer_ids:
        layer_vectors[lid] = torch.cat(layer_flat_weights[lid])

    sim_matrix = np.zeros((len(layer_ids), len(layer_ids)))
    for i, l1 in enumerate(layer_ids):
        for j, l2 in enumerate(layer_ids):
            cos = float(torch.nn.functional.cosine_similarity(
                layer_vectors[l1].unsqueeze(0),
                layer_vectors[l2].unsqueeze(0)))
            sim_matrix[i, j] = cos

    # ── Cluster analysis ──
    # Check if layers within the same group are more similar than across groups
    within_group_sims = {g: [] for g in LAYER_GROUPS}
    across_group_sims = []

    for i, l1 in enumerate(layer_ids):
        for j, l2 in enumerate(layer_ids):
            if l1 == l2:
                continue
            g1 = None
            g2 = None
            for g, ids in LAYER_GROUPS.items():
                if l1 in ids:
                    g1 = g
                if l2 in ids:
                    g2 = g
            if g1 and g2:
                if g1 == g2:
                    within_group_sims[g1].append(sim_matrix[i, j])
                else:
                    across_group_sims.append(sim_matrix[i, j])

    # ── Report ──
    lines = []
    lines.append("=" * 80)
    lines.append("LORA WEIGHT ANALYSIS (Experiment 5)")
    lines.append("=" * 80)
    lines.append(f"LoRA path: {lora_path}")
    lines.append(f"Layers analyzed: {len(layer_ids)}")
    lines.append("")

    # Per-layer aggregate
    lines.append("── Per-Layer Aggregate Metrics ──")
    lines.append(f"{'Layer':>6} | {'Group':>7} | {'Eff Rank':>9} | {'Frob Norm':>10} | {'SV Ratio':>9}")
    lines.append("-" * 55)
    for lid in layer_ids:
        agg = layer_agg.get(lid, {})
        group = "?"
        for g, ids in LAYER_GROUPS.items():
            if lid in ids:
                group = g
                break
        lines.append(f"{lid:>6} | {group:>7} | {agg.get('mean_eff_rank', 0):>9.3f} | "
                     f"{agg.get('mean_frob_norm', 0):>10.6f} | {agg.get('mean_sv_ratio', 0):>9.4f}")
    lines.append("")

    # Group aggregate
    lines.append("── Group Aggregate ──")
    for g, ids in LAYER_GROUPS.items():
        norms = [layer_agg[lid]["mean_frob_norm"] for lid in ids if lid in layer_agg]
        ranks = [layer_agg[lid]["mean_eff_rank"] for lid in ids if lid in layer_agg]
        if norms:
            lines.append(f"  {g:>7}: mean_norm={np.mean(norms):.6f}, "
                         f"std_norm={np.std(norms):.6f}, "
                         f"mean_eff_rank={np.mean(ranks):.3f}")
    lines.append("")

    # Clustering
    lines.append("── Inter-Layer Similarity (Stage Clustering) ──")
    for g, sims in within_group_sims.items():
        if sims:
            lines.append(f"  Within {g:>7}: mean_sim={np.mean(sims):+.4f}, std={np.std(sims):.4f}")
    if across_group_sims:
        lines.append(f"  Across groups:  mean_sim={np.mean(across_group_sims):+.4f}, "
                     f"std={np.std(across_group_sims):.4f}")

    within_all = []
    for sims in within_group_sims.values():
        within_all.extend(sims)
    if within_all and across_group_sims:
        within_mean = np.mean(within_all)
        across_mean = np.mean(across_group_sims)
        gap = within_mean - across_mean
        lines.append(f"\n  Within - Across gap = {gap:+.4f}")
        if gap > 0.05:
            lines.append(f"  → POSITIVE gap: layers DO cluster by stage")
            lines.append(f"  → Uniform LoRA implicitly tries stage-wise specialization")
        elif gap < -0.05:
            lines.append(f"  → NEGATIVE gap: layers are more similar across stages")
        else:
            lines.append(f"  → No clear clustering pattern")

    # Similarity matrix (compact heatmap)
    lines.append("\n── Similarity Matrix (rows/cols = layer IDs) ──")
    header = "     " + "".join(f"{l:>5}" for l in layer_ids)
    lines.append(header)
    for i, l1 in enumerate(layer_ids):
        row = f"{l1:>4} "
        for j, l2 in enumerate(layer_ids):
            val = sim_matrix[i, j]
            row += f"{val:>5.2f}"
        lines.append(row)

    lines.append("\n" + "=" * 80)

    report = "\n".join(lines)
    print(report)

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "lora_weight_analysis.txt"), "w") as f:
        f.write(report)

    # Save raw data
    with open(os.path.join(output_dir, "lora_weight_metrics.json"), "w") as f:
        json.dump({
            "per_module": layer_metrics,
            "per_layer_agg": {str(k): v for k, v in layer_agg.items()},
            "sim_matrix": sim_matrix.tolist(),
            "layer_ids": layer_ids,
        }, f, indent=2)

    print(f"\nSaved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="LoRA Weight Analysis (Exp 5)")
    parser.add_argument("--lora_path", required=True, help="Path to LoRA adapter checkpoint")
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    analyze(args.lora_path, args.output_dir)


if __name__ == "__main__":
    main()
