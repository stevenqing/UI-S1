#!/usr/bin/env python3
"""
v7 Emergent Specialization — Differentiation analysis.

Measures how much two cooperative LoRA adapters (LoRA_V, LoRA_A) have
self-differentiated under merge-mode + diversity-loss training.

Primary metric:
  cos_sim( flat(lora_A_v_m), flat(lora_A_a_m) )
for every target module m, aggregated by:
  - overall mean / std / min / max
  - per-layer (average across module types within a layer)
  - per-module-type (q/k/v/o/gate/up/down)

Baseline comparisons (run this script on):
  - v6.4 final (hard routing, no diversity loss) — structural baseline
  - v7.0 final (merge + diversity) — emergent baseline

Usage:
  python evaluation/analysis_v7_differentiation.py \
      --checkpoints train_GUI_360/llamafactory/output/cooperative_v6_4_comm_thought/epoch-4 \
                    train_GUI_360/llamafactory/output/cooperative_v7_0_merge_diversity/epoch-4
"""

import argparse
import json
import os
import re
import sys
from collections import defaultdict

import torch
import torch.nn.functional as F

# Parse something like "base_model.model.language_model.layers.27.self_attn.q_proj.lora_A_v"
# (accept a few historical variants; the layer index and module name are the
# important parts for aggregation.)
MODULE_RE = re.compile(
    r"layers\.(?P<layer>\d+)\."
    r"(?:self_attn|mlp)\."
    r"(?P<module>q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)\."
    r"lora_[AB]_[va]"
)


def load_lora_dict(path):
    return torch.load(path, map_location="cpu", weights_only=True)


def _cos(x, y, eps=1e-8):
    x = x.float().flatten()
    y = y.float().flatten()
    if x.norm() < eps or y.norm() < eps:
        return float("nan")
    return F.cosine_similarity(x.unsqueeze(0), y.unsqueeze(0)).item()


def analyze_checkpoint(ckpt_dir):
    """Return dict of differentiation stats for a cooperative checkpoint.

    Measures three per-module metrics:
      - cos(lora_A_v, lora_A_a)  — input projection similarity
      - cos(lora_B_v, lora_B_a)  — output projection similarity (primary)
      - cos(delta_W_v, delta_W_a) — full learned delta-W  (computed as B@A)
    """
    lora_v_path = os.path.join(ckpt_dir, "lora_v.pt")
    lora_a_path = os.path.join(ckpt_dir, "lora_a.pt")
    if not (os.path.exists(lora_v_path) and os.path.exists(lora_a_path)):
        return None

    v_state = load_lora_dict(lora_v_path)
    a_state = load_lora_dict(lora_a_path)

    # Build (layer, module) -> {"A_v": t, "B_v": t, "A_a": t, "B_a": t}
    by_mod = defaultdict(dict)
    for name, tensor in v_state.items():
        m = MODULE_RE.search(name)
        if not m:
            continue
        key = (int(m.group("layer")), m.group("module"))
        if "lora_A_v" in name:
            by_mod[key]["A_v"] = tensor
        elif "lora_B_v" in name:
            by_mod[key]["B_v"] = tensor
    for name, tensor in a_state.items():
        m = MODULE_RE.search(name)
        if not m:
            continue
        key = (int(m.group("layer")), m.group("module"))
        if "lora_A_a" in name:
            by_mod[key]["A_a"] = tensor
        elif "lora_B_a" in name:
            by_mod[key]["B_a"] = tensor

    # Compute per-module cos similarities
    per_module = []
    for (layer, mod), parts in sorted(by_mod.items()):
        if not all(k in parts for k in ("A_v", "B_v", "A_a", "B_a")):
            continue
        A_v, B_v = parts["A_v"], parts["B_v"]
        A_a, B_a = parts["A_a"], parts["B_a"]

        cos_A = _cos(A_v, A_a)
        cos_B = _cos(B_v, B_a)
        # Full delta-W = B @ A, shape [out_f, in_f]
        dW_v = (B_v.float() @ A_v.float())
        dW_a = (B_a.float() @ A_a.float())
        cos_W = _cos(dW_v, dW_a)

        per_module.append({
            "layer": layer,
            "module": mod,
            "cos_A": cos_A,
            "cos_B": cos_B,
            "cos_W": cos_W,
            "A_v_norm": A_v.float().norm().item(),
            "A_a_norm": A_a.float().norm().item(),
            "B_v_norm": B_v.float().norm().item(),
            "B_a_norm": B_a.float().norm().item(),
            "dW_v_norm": dW_v.norm().item(),
            "dW_a_norm": dW_a.norm().item(),
        })

    if not per_module:
        return None

    def _stats(vals):
        vals = [v for v in vals if v == v]  # drop NaN
        if not vals:
            return {"n": 0, "mean": float("nan"), "std": float("nan"),
                    "min": float("nan"), "max": float("nan"), "median": float("nan")}
        mean = sum(vals) / len(vals)
        return {
            "n": len(vals),
            "mean": mean,
            "std": (sum((v - mean) ** 2 for v in vals) / len(vals)) ** 0.5,
            "min": min(vals),
            "max": max(vals),
            "median": sorted(vals)[len(vals) // 2],
        }

    overall = {
        "A": _stats([r["cos_A"] for r in per_module]),
        "B": _stats([r["cos_B"] for r in per_module]),
        "W": _stats([r["cos_W"] for r in per_module]),
    }

    # Per-layer (mean across module types) — B is the primary metric
    layer_map_B = defaultdict(list)
    layer_map_W = defaultdict(list)
    for r in per_module:
        layer_map_B[r["layer"]].append(r["cos_B"])
        layer_map_W[r["layer"]].append(r["cos_W"])
    per_layer = {
        layer: {
            "cos_B": sum(v for v in layer_map_B[layer] if v == v) / max(
                1, sum(1 for v in layer_map_B[layer] if v == v)),
            "cos_W": sum(v for v in layer_map_W[layer] if v == v) / max(
                1, sum(1 for v in layer_map_W[layer] if v == v)),
        }
        for layer in sorted(layer_map_B.keys())
    }

    # Per-module-type (mean across layers)
    mod_map_B = defaultdict(list)
    mod_map_W = defaultdict(list)
    for r in per_module:
        mod_map_B[r["module"]].append(r["cos_B"])
        mod_map_W[r["module"]].append(r["cos_W"])
    per_module_type = {
        mod: {
            "cos_B": _stats(mod_map_B[mod]),
            "cos_W": _stats(mod_map_W[mod]),
        }
        for mod in sorted(mod_map_B.keys())
    }

    return {
        "checkpoint": ckpt_dir,
        "overall": overall,
        "per_layer": per_layer,
        "per_module_type": per_module_type,
        "per_module": per_module,
    }


def print_report(result):
    print("=" * 72)
    print(f"Checkpoint: {result['checkpoint']}")
    print("=" * 72)
    for label, key in (("cos(A_v, A_a)", "A"),
                       ("cos(B_v, B_a)", "B"),
                       ("cos(dW_v, dW_a)", "W")):
        o = result["overall"][key]
        print(f"Overall {label:<18s}  n={o['n']}  "
              f"mean={o['mean']:+.4f}  std={o['std']:.4f}  "
              f"min={o['min']:+.4f}  max={o['max']:+.4f}  "
              f"median={o['median']:+.4f}")

    print("\nPer-module-type (mean across layers):")
    print(f"  {'module':<10s} {'cos_B':>10s} {'cos_W':>10s}  (n per module type)")
    for mod, stats in result["per_module_type"].items():
        print(f"  {mod:<10s} {stats['cos_B']['mean']:+10.4f} "
              f"{stats['cos_W']['mean']:+10.4f}  n={stats['cos_B']['n']}")

    print("\nPer-layer cos_B (primary differentiation metric):")
    for layer, stats in sorted(result["per_layer"].items()):
        cB = stats["cos_B"]
        bar_len = int(max(0, (cB + 1) * 20))  # [-1,1] -> [0,40]
        print(f"  L{layer:02d}: cos_B={cB:+.4f}  cos_W={stats['cos_W']:+.4f}  "
              f"{'#' * bar_len}")

    # Top-5 most differentiated / most similar modules by cos_B
    sorted_mods = sorted(
        [r for r in result["per_module"] if r["cos_B"] == r["cos_B"]],  # drop NaN
        key=lambda r: r["cos_B"])
    print("\nTop-5 MOST DIFFERENTIATED (lowest cos_B):")
    for r in sorted_mods[:5]:
        print(f"  L{r['layer']:02d} {r['module']:<10s} cos_B={r['cos_B']:+.4f}  "
              f"|B_v|={r['B_v_norm']:7.3f}  |B_a|={r['B_a_norm']:7.3f}  "
              f"cos_W={r['cos_W']:+.4f}")
    print("\nTop-5 MOST SIMILAR (highest cos_B):")
    for r in sorted_mods[-5:]:
        print(f"  L{r['layer']:02d} {r['module']:<10s} cos_B={r['cos_B']:+.4f}  "
              f"|B_v|={r['B_v_norm']:7.3f}  |B_a|={r['B_a_norm']:7.3f}  "
              f"cos_W={r['cos_W']:+.4f}")
    print()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoints", nargs="+", required=True,
                   help="Cooperative checkpoint dirs (must contain lora_v.pt + lora_a.pt).")
    p.add_argument("--output_json", type=str, default=None,
                   help="Optional: write combined results to this JSON file.")
    args = p.parse_args()

    all_results = {}
    for ckpt in args.checkpoints:
        result = analyze_checkpoint(ckpt)
        if result is None:
            print(f"[skip] {ckpt}: missing lora_v.pt / lora_a.pt", file=sys.stderr)
            continue
        print_report(result)
        # Drop the detailed per-module list from JSON dump to keep it small
        all_results[ckpt] = {
            "overall": result["overall"],
            "per_layer": result["per_layer"],
            "per_module_type": result["per_module_type"],
        }

    if args.output_json:
        with open(args.output_json, "w") as fp:
            json.dump(all_results, fp, indent=2)
        print(f"Wrote aggregate results to {args.output_json}")


if __name__ == "__main__":
    main()
