"""SVD subspace analysis for cooperative LoRA checkpoints.

For each cooperative LoRA module in a checkpoint, this script computes:

  1. Frobenius cosine between effective deltas  cos(flat(B_v A_v), flat(B_a A_a))
  2. Stable rank of each effective delta  ||·||_F^2 / ||·||_2^2
  3. Top-k singular values of each effective delta
  4. Principal angles between column spaces (top-k left singular vectors of B)
  5. Principal angles between row spaces (top-k right singular vectors of A)

All "effective delta" math uses the trick:

    ΔW = B A           B is [out, r],  A is [r, in]
    SVD of ΔW          (without materializing the [out, in] matrix)
    => decompose B = U_B Σ_B V_B^T,  A = U_A Σ_A V_A^T
    => ΔW = U_B (Σ_B V_B^T U_A Σ_A) V_A^T
    => singular values of ΔW = singular values of the central [r, r] matrix

Frobenius inner product also uses the trace trick:

    <B1 A1, B2 A2>_F = trace(A1^T B1^T B2 A2) = trace((B1^T B2)(A2 A1^T))
                    = sum_elementwise( (B1^T B2) * (A2 A1^T)^T )

This way the largest tensor we ever materialize per module is [256, 256].

Output: JSON file with per-module stats + a per-layer / per-module-type
aggregate summary printed to stdout.

Usage:
    python evaluation/analyze_lora_subspace.py \
        --checkpoint train_GUI_360/llamafactory/output/cooperative_v6_4_comm_thought/epoch-4 \
        --output evaluation/analysis_results_v3/v6_4_ep4_lora_subspace.json \
        --top_k 64
"""
import argparse
import json
import math
import os
from collections import defaultdict

import torch


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True,
                   help="Cooperative checkpoint dir containing lora_v.pt and lora_a.pt")
    p.add_argument("--output", required=True,
                   help="Output JSON path")
    p.add_argument("--top_k", type=int, default=64,
                   help="Number of top singular vectors to use as subspace basis")
    p.add_argument("--keep_top_sigma", type=int, default=10,
                   help="Number of singular values to record per module")
    return p.parse_args()


def group_modules(v_state, a_state):
    """Group lora_A_*/lora_B_* tensors by module path."""
    modules = defaultdict(dict)
    for k, t in v_state.items():
        if k.endswith(".lora_A_v"):
            modules[k[: -len(".lora_A_v")]]["A_v"] = t
        elif k.endswith(".lora_B_v"):
            modules[k[: -len(".lora_B_v")]]["B_v"] = t
    for k, t in a_state.items():
        if k.endswith(".lora_A_a"):
            modules[k[: -len(".lora_A_a")]]["A_a"] = t
        elif k.endswith(".lora_B_a"):
            modules[k[: -len(".lora_B_a")]]["B_a"] = t
    # Drop incomplete modules
    return {k: v for k, v in modules.items() if len(v) == 4}


def parse_module_path(path):
    """Extract layer index and module type from a fully-qualified module path.

    Examples:
        base_model.model.language_model.layers.5.self_attn.q_proj
            -> (5, 'q_proj')
        base_model.model.language_model.layers.27.mlp.down_proj
            -> (27, 'down_proj')
    """
    parts = path.split(".")
    layer_idx = None
    mod_type = parts[-1]
    for i, p in enumerate(parts):
        if p == "layers" and i + 1 < len(parts):
            try:
                layer_idx = int(parts[i + 1])
            except ValueError:
                pass
            break
    return layer_idx, mod_type


def effective_singular_values(B, A):
    """Singular values of ΔW = B @ A without materializing ΔW.

    B: [out, r],  A: [r, in].
    Returns 1-D tensor of length r containing the (up to r) singular values.
    """
    # SVD of B and A in their reduced forms
    Ub, Sb, Vb = torch.linalg.svd(B, full_matrices=False)  # Ub: [out,r], Sb: [r], Vb: [r,r]
    Ua, Sa, Va = torch.linalg.svd(A, full_matrices=False)  # Ua: [r,r], Sa: [r], Va: [r,in]
    # ΔW = Ub diag(Sb) Vb @ Ua diag(Sa) Va = Ub (diag(Sb) Vb Ua diag(Sa)) Va
    # Central r×r matrix M has same singular values as ΔW
    M = (Sb.unsqueeze(1) * Vb) @ (Ua * Sa.unsqueeze(0))
    sigma = torch.linalg.svdvals(M)
    return sigma, Ub, Va


def frob_inner(B1, A1, B2, A2):
    """<B1 A1, B2 A2>_F via the trace trick (no [out, in] materialization).

    = trace(A1^T B1^T B2 A2)
    = trace((B1^T B2) (A2 A1^T))
    = sum_elementwise( (B1^T B2) * (A2 A1^T)^T )
    """
    M1 = B1.T @ B2          # [r, r]
    M2 = A2 @ A1.T          # [r, r]
    return (M1 * M2.T).sum()


def principal_angles(Q1, Q2):
    """Principal angles (radians) between subspaces spanned by columns of Q1 and Q2.

    Q1, Q2 must be orthonormal: shape [n, k]. Returns k angles in [0, pi/2].
    """
    M = Q1.T @ Q2                          # [k, k]
    sigmas = torch.linalg.svdvals(M)       # cosines of principal angles
    sigmas = sigmas.clamp(-1.0, 1.0)
    return torch.acos(sigmas)


def stable_rank(sigma):
    """Stable rank = ||M||_F^2 / ||M||_2^2 = sum(sigma^2) / sigma_max^2."""
    smax = sigma[0].clamp_min(1e-12)
    return float(((sigma ** 2).sum() / (smax ** 2)).item())


def analyze_module(name, t, top_k, keep_top_sigma):
    A_v = t["A_v"].float()
    B_v = t["B_v"].float()
    A_a = t["A_a"].float()
    B_a = t["B_a"].float()

    # Effective singular values + bases (Ub for column space, Va for row space)
    sigma_v, Ub_v, Va_v = effective_singular_values(B_v, A_v)
    sigma_a, Ub_a, Va_a = effective_singular_values(B_a, A_a)

    # Frobenius norms (squared)
    norm_v_sq = float((sigma_v ** 2).sum().item())
    norm_a_sq = float((sigma_a ** 2).sum().item())

    # Frobenius inner product via trace trick
    inner = float(frob_inner(B_v, A_v, B_a, A_a).item())

    frob_cos = inner / (math.sqrt(norm_v_sq * norm_a_sq) + 1e-12)

    # Subspace overlap: principal angles
    # Use top-k of effective decomposition. Note: effective_singular_values
    # returned Ub from SVD of B (columns) and Va from SVD of A (rows).
    # These ARE the bases of the column/row spaces of ΔW = B A (assuming
    # full rank A and B, the column space of BA equals column space of B,
    # and the row space of BA equals row space of A).
    r = Ub_v.shape[1]
    k = min(top_k, r)
    angles_col = principal_angles(Ub_v[:, :k], Ub_a[:, :k])  # [k]
    angles_row = principal_angles(Va_v[:k, :].T, Va_a[:k, :].T)  # [k]

    return {
        "shape_in": A_v.shape[1],
        "shape_out": B_v.shape[0],
        "rank": r,
        "frob_cos": frob_cos,
        "frob_norm_v": math.sqrt(norm_v_sq),
        "frob_norm_a": math.sqrt(norm_a_sq),
        "stable_rank_v": stable_rank(sigma_v),
        "stable_rank_a": stable_rank(sigma_a),
        "sigma_v_top": sigma_v[:keep_top_sigma].tolist(),
        "sigma_a_top": sigma_a[:keep_top_sigma].tolist(),
        "principal_angles_col_deg": (angles_col * 180 / math.pi).tolist(),
        "principal_angles_row_deg": (angles_row * 180 / math.pi).tolist(),
        "mean_angle_col_deg": float((angles_col * 180 / math.pi).mean().item()),
        "mean_angle_row_deg": float((angles_row * 180 / math.pi).mean().item()),
        "min_angle_col_deg": float((angles_col * 180 / math.pi).min().item()),
        "max_angle_col_deg": float((angles_col * 180 / math.pi).max().item()),
    }


def aggregate(results, top_k):
    """Print per-layer and per-module-type summaries."""
    by_layer = defaultdict(list)
    by_type = defaultdict(list)
    for path, r in results.items():
        layer_idx, mod_type = parse_module_path(path)
        by_layer[layer_idx].append(r)
        by_type[mod_type].append(r)

    def _summary(group, label):
        if not group:
            return
        n = len(group)
        cos = sum(r["frob_cos"] for r in group) / n
        sr_v = sum(r["stable_rank_v"] for r in group) / n
        sr_a = sum(r["stable_rank_a"] for r in group) / n
        ang_col = sum(r["mean_angle_col_deg"] for r in group) / n
        ang_row = sum(r["mean_angle_row_deg"] for r in group) / n
        print(f"  {label:<20s} n={n:3d}  frob_cos={cos:+.4f}  "
              f"stable_rank V={sr_v:6.2f} A={sr_a:6.2f}  "
              f"mean_angle col={ang_col:5.1f}° row={ang_row:5.1f}°")

    print("\n=== Per-module-type aggregate (top-k principal angles k="
          f"{top_k}) ===")
    for mt in sorted(by_type.keys()):
        _summary(by_type[mt], mt)

    print("\n=== Per-layer aggregate (selected layers) ===")
    layer_keys = sorted(k for k in by_layer.keys() if k is not None)
    # Print every 4th layer + first/last
    show = set([layer_keys[0], layer_keys[-1]] + layer_keys[::4])
    for li in layer_keys:
        if li in show:
            _summary(by_layer[li], f"layer{li:02d}")

    print("\n=== Overall ===")
    _summary(list(results.values()), "ALL")

    # Top-5 most-aligned and most-orthogonal modules
    sorted_by_cos = sorted(results.items(), key=lambda kv: kv[1]["frob_cos"])
    print("\n=== 5 most ANTI-correlated modules (lowest frob_cos) ===")
    for p, r in sorted_by_cos[:5]:
        short = ".".join(p.split(".")[-3:])
        print(f"  {short:<35s} cos={r['frob_cos']:+.4f}  "
              f"angle_col={r['mean_angle_col_deg']:5.1f}°")
    print("\n=== 5 most CORRELATED modules (highest frob_cos) ===")
    for p, r in sorted_by_cos[-5:]:
        short = ".".join(p.split(".")[-3:])
        print(f"  {short:<35s} cos={r['frob_cos']:+.4f}  "
              f"angle_col={r['mean_angle_col_deg']:5.1f}°")


def main():
    args = parse_args()

    v_path = os.path.join(args.checkpoint, "lora_v.pt")
    a_path = os.path.join(args.checkpoint, "lora_a.pt")
    print(f"Loading {v_path}")
    v_state = torch.load(v_path, map_location="cpu", weights_only=True)
    print(f"Loading {a_path}")
    a_state = torch.load(a_path, map_location="cpu", weights_only=True)

    modules = group_modules(v_state, a_state)
    print(f"Grouped {len(modules)} modules")

    results = {}
    for i, (path, t) in enumerate(modules.items()):
        if (i + 1) % 20 == 0:
            print(f"  processed {i+1}/{len(modules)} modules")
        results[path] = analyze_module(path, t, args.top_k, args.keep_top_sigma)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({
            "checkpoint": args.checkpoint,
            "top_k": args.top_k,
            "n_modules": len(results),
            "per_module": results,
        }, f, indent=2)
    print(f"\nWrote per-module stats -> {args.output}")

    aggregate(results, args.top_k)


if __name__ == "__main__":
    main()
