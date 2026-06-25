#!/usr/bin/env python3
"""Analyze full-SFT deltas across visual encoder and language layers.

Compares two Hugging Face safetensors checkpoints with matching keys and writes
CSV/JSON/Markdown summaries of W_sft - W_base. Designed for Qwen2.5-VL style
keys but keeps the grouping logic simple enough to inspect other checkpoints.
"""

import argparse
import csv
import json
import math
import os
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import torch
from safetensors.torch import load_file


def log(message: str = "") -> None:
    print(message, flush=True)


def load_weight_map(model_dir: Path) -> Dict[str, str]:
    index_path = model_dir / "model.safetensors.index.json"
    if index_path.exists():
        with index_path.open() as f:
            return json.load(f)["weight_map"]
    single = model_dir / "model.safetensors"
    if single.exists():
        return {key: single.name for key in load_file(str(single)).keys()}
    raise FileNotFoundError(f"No safetensors index or model.safetensors in {model_dir}")


def classify_key(key: str) -> Tuple[str, str, Optional[int], str]:
    """Return domain, component, layer/block index, module label."""
    if key.startswith("visual."):
        block_match = re.match(r"visual\.blocks\.(\d+)\.(.*)", key)
        if block_match:
            block = int(block_match.group(1))
            rest = block_match.group(2)
            if rest.startswith("attn."):
                component = "visual_attn"
            elif rest.startswith("mlp."):
                component = "visual_mlp"
            elif rest.startswith("norm"):
                component = "visual_norm"
            else:
                component = "visual_block_other"
            return "visual", component, block, rest
        if key.startswith("visual.patch_embed"):
            return "visual", "visual_patch_embed", None, key.removeprefix("visual.")
        if key.startswith("visual.merger"):
            return "visual", "visual_merger", None, key.removeprefix("visual.")
        return "visual", "visual_other", None, key.removeprefix("visual.")

    if key.startswith("model.layers."):
        layer_match = re.match(r"model\.layers\.(\d+)\.(.*)", key)
        layer = int(layer_match.group(1)) if layer_match else None
        rest = layer_match.group(2) if layer_match else key
        if ".self_attn." in key:
            component = "language_attn"
        elif ".mlp." in key:
            component = "language_mlp"
        elif "layernorm" in key:
            component = "language_norm"
        else:
            component = "language_layer_other"
        return "language", component, layer, rest

    if key.startswith("model.embed_tokens") or key.startswith("lm_head"):
        return "language_io", "embedding_or_head", None, key
    if key.startswith("model."):
        return "language", "language_other", None, key.removeprefix("model.")
    return "other", "other", None, key


def should_compute_svd(
    key: str,
    shape: Tuple[int, ...],
    numel: int,
    max_svd_numel: int,
) -> bool:
    if len(shape) != 2:
        return False
    if numel > max_svd_numel:
        return False
    if key.startswith("model.embed_tokens") or key.startswith("lm_head"):
        return False
    if key.endswith("bias"):
        return False
    return True


def tensor_norm(tensor: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(tensor.float()).item())


def compute_stats(key: str, base: torch.Tensor, sft: torch.Tensor) -> Dict[str, object]:
    delta = sft.float() - base.float()
    base_float = base.float()
    sft_float = sft.float()

    delta_norm = float(torch.linalg.vector_norm(delta).item())
    base_norm = float(torch.linalg.vector_norm(base_float).item())
    sft_norm = float(torch.linalg.vector_norm(sft_float).item())
    numel = delta.numel()
    domain, component, layer, module = classify_key(key)
    mean_abs = float(delta.abs().mean().item()) if numel else 0.0
    max_abs = float(delta.abs().max().item()) if numel else 0.0
    rms_delta = delta_norm / math.sqrt(numel) if numel else 0.0

    return {
        "key": key,
        "domain": domain,
        "component": component,
        "layer": "" if layer is None else layer,
        "module": module,
        "shape": "x".join(str(x) for x in delta.shape),
        "numel": numel,
        "delta_norm": delta_norm,
        "base_norm": base_norm,
        "sft_norm": sft_norm,
        "rel_delta": delta_norm / base_norm if base_norm > 0 else 0.0,
        "rms_delta": rms_delta,
        "mean_abs_delta": mean_abs,
        "max_abs_delta": max_abs,
    }


def compute_svd_energy(
    key: str,
    base: torch.Tensor,
    sft: torch.Tensor,
    ranks: Iterable[int],
    device: Optional[torch.device],
) -> Dict[str, object]:
    delta = sft.float() - base.float()
    if device is not None:
        delta = delta.to(device)
    singular_values = torch.linalg.svdvals(delta)
    energy = singular_values.square()
    total = float(energy.sum().item())
    domain, component, layer, module = classify_key(key)

    result: Dict[str, object] = {
        "key": key,
        "domain": domain,
        "component": component,
        "layer": "" if layer is None else layer,
        "module": module,
        "shape": "x".join(str(x) for x in delta.shape),
        "rank_full": int(singular_values.numel()),
        "sv_energy_total": total,
    }
    for rank in ranks:
        usable = min(rank, singular_values.numel())
        captured = float(energy[:usable].sum().item() / total) if total > 0 else 1.0
        result[f"energy_r{rank}"] = captured
    return result


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def aggregate(rows: List[Dict[str, object]], group_keys: List[str]) -> List[Dict[str, object]]:
    grouped: Dict[Tuple[object, ...], Dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for row in rows:
        group = tuple(row[key] for key in group_keys)
        item = grouped[group]
        item["tensors"] += 1
        item["params"] += float(row["numel"])
        item["delta_sq"] += float(row["delta_norm"]) ** 2
        item["base_sq"] += float(row["base_norm"]) ** 2
        item["weighted_rms_delta_num"] += float(row["rms_delta"]) * float(row["numel"])

    out = []
    for group, item in grouped.items():
        delta_norm = math.sqrt(item["delta_sq"])
        base_norm = math.sqrt(item["base_sq"])
        params = item["params"]
        rec = {key: value for key, value in zip(group_keys, group)}
        rec.update({
            "tensors": int(item["tensors"]),
            "params": int(params),
            "delta_norm": delta_norm,
            "base_norm": base_norm,
            "rel_delta": delta_norm / base_norm if base_norm > 0 else 0.0,
            "weighted_rms_delta": item["weighted_rms_delta_num"] / params if params else 0.0,
        })
        out.append(rec)
    out.sort(key=lambda row: float(row["delta_norm"]), reverse=True)
    return out


def format_pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def make_report(
    path: Path,
    args: argparse.Namespace,
    domain_rows: List[Dict[str, object]],
    component_rows: List[Dict[str, object]],
    language_layer_rows: List[Dict[str, object]],
    visual_block_rows: List[Dict[str, object]],
    tensor_rows: List[Dict[str, object]],
    svd_rows: List[Dict[str, object]],
) -> None:
    total_delta_sq = sum(float(row["delta_norm"]) ** 2 for row in domain_rows)
    lines = [
        "# Visual/Language Delta Analysis",
        "",
        f"Base model: `{args.base_model}`",
        f"SFT model: `{args.sft_model}`",
        f"Ranks analyzed: `{args.svd_ranks}`",
        "",
        "## Domain Summary",
        "",
        "| Domain | Tensors | Params | Delta Norm | Rel Delta | Delta Energy Share |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in domain_rows:
        share = (float(row["delta_norm"]) ** 2 / total_delta_sq) if total_delta_sq else 0.0
        lines.append(
            f"| {row['domain']} | {row['tensors']} | {row['params']:,} | "
            f"{float(row['delta_norm']):.4f} | {format_pct(float(row['rel_delta']))} | {format_pct(share)} |"
        )

    lines += ["", "## Top Components by Delta Norm", "", "| Component | Tensors | Params | Delta Norm | Rel Delta |", "|---|---:|---:|---:|---:|"]
    for row in component_rows[:20]:
        label = f"{row['domain']} / {row['component']}"
        lines.append(
            f"| {label} | {row['tensors']} | {row['params']:,} | "
            f"{float(row['delta_norm']):.4f} | {format_pct(float(row['rel_delta']))} |"
        )

    lines += ["", "## Top Language Layers", "", "| Layer | Delta Norm | Rel Delta |", "|---:|---:|---:|"]
    for row in language_layer_rows[:12]:
        lines.append(f"| {row['layer']} | {float(row['delta_norm']):.4f} | {format_pct(float(row['rel_delta']))} |")

    lines += ["", "## Top Visual Blocks", "", "| Block | Delta Norm | Rel Delta |", "|---:|---:|---:|"]
    for row in visual_block_rows[:12]:
        lines.append(f"| {row['layer']} | {float(row['delta_norm']):.4f} | {format_pct(float(row['rel_delta']))} |")

    lines += ["", "## Top Individual Tensors by Delta Norm", "", "| Key | Domain | Shape | Delta Norm | Rel Delta |", "|---|---|---:|---:|---:|"]
    for row in sorted(tensor_rows, key=lambda item: float(item["delta_norm"]), reverse=True)[:20]:
        lines.append(
            f"| `{row['key']}` | {row['domain']} | {row['shape']} | "
            f"{float(row['delta_norm']):.4f} | {format_pct(float(row['rel_delta']))} |"
        )

    if svd_rows:
        lines += ["", "## SVD Energy Summary", "", "| Domain/Component | Count | Mean r128 Energy | Mean r256 Energy |", "|---|---:|---:|---:|"]
        by_component: Dict[Tuple[str, str], List[Dict[str, object]]] = defaultdict(list)
        for row in svd_rows:
            by_component[(str(row["domain"]), str(row["component"]))].append(row)
        for (domain, component), rows in sorted(by_component.items()):
            r128 = sum(float(row.get("energy_r128", 0.0)) for row in rows) / len(rows)
            r256 = sum(float(row.get("energy_r256", 0.0)) for row in rows) / len(rows)
            lines.append(f"| {domain} / {component} | {len(rows)} | {format_pct(r128)} | {format_pct(r256)} |")

    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze visual/language deltas between base and SFT checkpoints")
    parser.add_argument("--base_model", required=True)
    parser.add_argument("--sft_model", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--svd_ranks", nargs="+", type=int, default=[16, 32, 64, 128, 256])
    parser.add_argument("--max_svd_numel", type=int, default=80_000_000)
    parser.add_argument("--no_svd", action="store_true")
    parser.add_argument("--cpu_svd", action="store_true")
    args = parser.parse_args()

    base_dir = Path(args.base_model)
    sft_dir = Path(args.sft_model)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    base_map = load_weight_map(base_dir)
    sft_map = load_weight_map(sft_dir)
    common_keys = sorted(set(base_map) & set(sft_map))
    missing_base = sorted(set(sft_map) - set(base_map))
    missing_sft = sorted(set(base_map) - set(sft_map))

    log(f"Base keys: {len(base_map)}  SFT keys: {len(sft_map)}  Common: {len(common_keys)}")
    log(f"Missing in base: {len(missing_base)}  Missing in SFT: {len(missing_sft)}")

    groups: Dict[Tuple[str, str], List[str]] = defaultdict(list)
    for key in common_keys:
        groups[(base_map[key], sft_map[key])].append(key)
    log(f"Shard pairs: {len(groups)}")

    svd_device = None
    if not args.no_svd and not args.cpu_svd and torch.cuda.is_available():
        svd_device = torch.device("cuda:0")
        log(f"SVD device: {torch.cuda.get_device_name(0)}")
    elif not args.no_svd:
        log("SVD device: CPU")

    tensor_rows: List[Dict[str, object]] = []
    svd_rows: List[Dict[str, object]] = []
    t0 = time.time()

    for pair_idx, ((base_file, sft_file), keys) in enumerate(sorted(groups.items()), 1):
        log(f"[{pair_idx}/{len(groups)}] Loading base={base_file} sft={sft_file} keys={len(keys)}")
        base_shard = load_file(str(base_dir / base_file))
        sft_shard = load_file(str(sft_dir / sft_file))
        for key in keys:
            base_tensor = base_shard[key]
            sft_tensor = sft_shard[key]
            row = compute_stats(key, base_tensor, sft_tensor)
            tensor_rows.append(row)

            if not args.no_svd and should_compute_svd(key, tuple(base_tensor.shape), int(row["numel"]), args.max_svd_numel):
                try:
                    svd_rows.append(compute_svd_energy(key, base_tensor, sft_tensor, args.svd_ranks, svd_device))
                except RuntimeError as exc:
                    log(f"  WARNING: SVD failed for {key}: {exc}")
        del base_shard, sft_shard

    domain_rows = aggregate(tensor_rows, ["domain"])
    component_rows = aggregate(tensor_rows, ["domain", "component"])
    layer_rows = aggregate([row for row in tensor_rows if row["layer"] != ""], ["domain", "layer"])
    language_layer_rows = [row for row in layer_rows if row["domain"] == "language"]
    visual_block_rows = [row for row in layer_rows if row["domain"] == "visual"]

    write_csv(output_dir / "tensor_delta_stats.csv", tensor_rows)
    write_csv(output_dir / "domain_summary.csv", domain_rows)
    write_csv(output_dir / "component_summary.csv", component_rows)
    write_csv(output_dir / "layer_block_summary.csv", layer_rows)
    write_csv(output_dir / "svd_energy.csv", svd_rows)

    summary = {
        "base_model": str(base_dir),
        "sft_model": str(sft_dir),
        "num_common_keys": len(common_keys),
        "num_missing_base": len(missing_base),
        "num_missing_sft": len(missing_sft),
        "domain_summary": domain_rows,
        "num_svd_tensors": len(svd_rows),
        "svd_ranks": args.svd_ranks,
        "elapsed_seconds": time.time() - t0,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    make_report(
        output_dir / "delta_report.md",
        args,
        domain_rows,
        component_rows,
        language_layer_rows,
        visual_block_rows,
        tensor_rows,
        svd_rows,
    )

    log(f"Done in {time.time() - t0:.1f}s")
    log(f"Wrote: {output_dir}")


if __name__ == "__main__":
    main()