#!/usr/bin/env python3
"""Recompute f-values from saved VLM checkpoints (fix bf16→numpy bug)."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from verl.models.moe.vlm_eigenfunction import (
    VLMEigenfunctionConfig,
    VLMEigenfunctionModel,
    VLMCollator,
)
from verl.models.moe.graph_analysis import (
    identify_bottlenecks,
    map_bottlenecks_to_descriptions,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    force=True,
)
sys.stderr.reconfigure(line_buffering=True)
logger = logging.getLogger(__name__)

BASE_DIR = PROJECT_ROOT
OUTPUT_DIR = BASE_DIR / "outputs" / "vlm_fnet"
DATA_DIR = OUTPUT_DIR / "data"
REGISTRY_PATH = BASE_DIR / "outputs" / "transitions" / "gui360_full" / "state_registry.json"
MLP_FNET_DIR = BASE_DIR / "outputs" / "fnet" / "gui360"
APPS = ["excel", "word", "ppt"]


@torch.no_grad()
def compute_f_values(model, all_hashes, collator, device, batch_size=16):
    """Compute f-values with bf16→float fix."""
    model.eval()
    f_values = []
    for i in range(0, len(all_hashes), batch_size):
        batch_hashes = all_hashes[i:i + batch_size]
        batch = collator(batch_hashes)
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        f_batch = model(**batch)
        f_values.extend(f_batch.float().cpu().numpy().flatten().tolist())
        if i % (batch_size * 10) == 0:
            logger.info(f"  Computed {i + len(batch_hashes)}/{len(all_hashes)} f-values")
    return f_values


def recompute_app(app: str, device: str):
    """Recompute f-values for one app from saved checkpoint."""
    app_dir = OUTPUT_DIR / app
    data_dir = DATA_DIR / app

    # Load config
    with open(app_dir / "config.json") as f:
        cfg_dict = json.load(f)

    config = VLMEigenfunctionConfig(
        model_path=cfg_dict["model_path"],
        lora_rank=cfg_dict["lora_rank"],
        lora_alpha=cfg_dict["lora_alpha"],
        lora_dropout=cfg_dict.get("lora_dropout", 0.05),
        input_mode=cfg_dict["input_mode"],
        image_size=cfg_dict["image_size"],
        eta=cfg_dict["eta"],
        batch_size=cfg_dict["batch_size"],
        lr=cfg_dict["lr"],
        num_epochs=cfg_dict["num_epochs"],
        bf16=cfg_dict["bf16"],
        percentile_k=cfg_dict["percentile_k"],
    )

    # Load manifest
    with open(data_dir / "state_manifest.json") as f:
        manifest_list = json.load(f)
    manifest_lookup = {e["hash"]: e for e in manifest_list}
    all_hashes = list(manifest_lookup.keys())
    logger.info(f"{app}: {len(all_hashes)} states")

    # Load model from saved adapter
    logger.info(f"{app}: Loading model from {app_dir}")
    model = VLMEigenfunctionModel.load_adapter(app_dir, config, device=device)

    # Load processor
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(
        config.model_path,
        min_pixels=config.image_size * config.image_size,
        max_pixels=config.image_size * config.image_size,
    )

    collator = VLMCollator(
        manifest=manifest_lookup,
        processor=processor,
        input_mode=config.input_mode,
        image_size=config.image_size,
    )

    # Compute f-values
    logger.info(f"{app}: Computing f-values...")
    f_values_list = compute_f_values(model, all_hashes, collator, device, batch_size=16)

    state_hashes_arr = np.array(all_hashes)
    f_values_arr = np.array(f_values_list)

    logger.info(f"{app}: f-values: mean={f_values_arr.mean():.4f}, std={f_values_arr.std():.4f}, "
                f"min={f_values_arr.min():.4f}, max={f_values_arr.max():.4f}")

    # Save f-values
    np.savez(app_dir / "f_values.npz", hashes=state_hashes_arr, f_values=f_values_arr)

    # Identify bottlenecks
    bottleneck_info = identify_bottlenecks(f_values_arr, state_hashes_arr, config.percentile_k)
    with open(app_dir / "bottlenecks.json", "w") as f:
        json.dump(bottleneck_info, f, indent=2)

    logger.info(f"{app}: {bottleneck_info['num_bottleneck_states']} bottleneck states "
                f"(threshold={bottleneck_info['threshold']:.4f})")

    # Bottleneck descriptions
    if REGISTRY_PATH.exists():
        described = map_bottlenecks_to_descriptions(bottleneck_info, REGISTRY_PATH)
        with open(app_dir / "bottlenecks_described.json", "w") as f:
            json.dump(described, f, indent=2)
    else:
        described = []

    # MLP comparison
    mlp_path = MLP_FNET_DIR / app / "f_values.npz"
    if mlp_path.exists():
        mlp_data = np.load(mlp_path, allow_pickle=True)
        mlp_hashes = set(str(h) for h in mlp_data["hashes"])
        vlm_hash_set = set(str(h) for h in state_hashes_arr)
        common = vlm_hash_set & mlp_hashes

        vlm_threshold = float(np.percentile(f_values_arr, 30))
        vlm_bottleneck_set = set(str(h) for h, fv in zip(state_hashes_arr, f_values_arr) if fv < vlm_threshold)

        mlp_f_arr = mlp_data["f_values"]
        mlp_threshold = float(np.percentile(mlp_f_arr, 30))
        mlp_bottleneck_set = set(str(h) for h, fv in zip(mlp_data["hashes"], mlp_f_arr) if fv < mlp_threshold)

        both = vlm_bottleneck_set & mlp_bottleneck_set & common
        vlm_only = (vlm_bottleneck_set & common) - mlp_bottleneck_set
        mlp_only = (mlp_bottleneck_set & common) - vlm_bottleneck_set

        comparison = {
            "common_states": len(common),
            "vlm_bottlenecks": len(vlm_bottleneck_set & common),
            "mlp_bottlenecks": len(mlp_bottleneck_set & common),
            "overlap": len(both),
            "vlm_only": len(vlm_only),
            "mlp_only": len(mlp_only),
            "jaccard_index": len(both) / max(1, len((vlm_bottleneck_set | mlp_bottleneck_set) & common)),
        }
        with open(app_dir / "mlp_comparison.json", "w") as f:
            json.dump(comparison, f, indent=2)
        logger.info(f"{app}: MLP comparison — overlap={len(both)}, Jaccard={comparison['jaccard_index']:.3f}")

    # Update results.json
    with open(app_dir / "results.json") as f:
        results = json.load(f)
    results["f_value_stats"] = {
        "mean": float(f_values_arr.mean()),
        "std": float(f_values_arr.std()),
        "min": float(f_values_arr.min()),
        "max": float(f_values_arr.max()),
    }
    results["bottleneck_threshold"] = bottleneck_info["threshold"]
    results["num_bottleneck_states"] = bottleneck_info["num_bottleneck_states"]
    with open(app_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Print top bottlenecks
    print(f"\n{'='*60}")
    print(f"  {app.upper()} — VLM Eigenfunction Results (recomputed)")
    print(f"{'='*60}")
    print(f"  f-values: mean={f_values_arr.mean():.4f}, std={f_values_arr.std():.4f}, "
          f"range=[{f_values_arr.min():.4f}, {f_values_arr.max():.4f}]")
    print(f"  Bottlenecks: {bottleneck_info['num_bottleneck_states']} "
          f"(threshold={bottleneck_info['threshold']:.4f})")
    if described:
        print(f"  Top-10 bottleneck states:")
        for i, entry in enumerate(described[:10]):
            if isinstance(entry, dict):
                fv = entry.get("f_value", 0.0)
                dialog = entry.get("dialog_state", "none")
                tabs = entry.get("active_tab_signature", "")
                if isinstance(tabs, str):
                    tabs_short = tabs[:50] + "..." if len(tabs) > 50 else tabs
                else:
                    tabs_short = str(tabs)[:50]
                print(f"    {i+1:3d}. f={fv:+.4f}  dialog={dialog:30s}  tabs=[{tabs_short}]")
    print(f"{'='*60}\n")

    # Free GPU memory
    del model
    torch.cuda.empty_cache()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--app", type=str, choices=APPS + ["all"], default="all")
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    logger.info(f"Using device: {device}")

    apps = APPS if args.app == "all" else [args.app]
    for app in apps:
        logger.info(f"\n{'='*60}")
        logger.info(f"Recomputing f-values for: {app}")
        logger.info(f"{'='*60}")
        recompute_app(app, device)

    logger.info("All done!")


if __name__ == "__main__":
    main()
