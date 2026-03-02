#!/usr/bin/env python3
"""
Task 3.1: Train VLM eigenfunction (Qwen2.5-VL + LoRA + regression head).

Trains the VLM to approximate the second eigenvector of the graph Laplacian
directly from screenshots (+optional a11y text).

Usage:
    # Train on Excel with screenshot+a11y mode
    python scripts/train_vlm_eigenfunction.py --app excel --input-mode screenshot_a11y

    # Train on all apps
    python scripts/train_vlm_eigenfunction.py --app all

    # CPU smoke test (tiny)
    python scripts/train_vlm_eigenfunction.py --app excel --device cpu --batch-size 1 --epochs 1 --max-states 10

    # Custom LoRA rank
    python scripts/train_vlm_eigenfunction.py --app word --lora-rank 8

Inputs (from prepare_vlm_eigenfunction_data.py):
    outputs/vlm_fnet/data/{app}/state_manifest.json
    outputs/vlm_fnet/data/{app}/transition_pairs.json

Outputs:
    outputs/vlm_fnet/{app}/
        lora_adapter/          — LoRA adapter weights
        regression_head.pt     — Regression head weights
        f_values.npz           — f-values for all states
        bottlenecks.json       — Identified bottleneck states
        results.json           — Training results
        training_history.json  — Per-epoch metrics
        config.json            — Training configuration
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from verl.models.moe.vlm_eigenfunction import (
    VLMEigenfunctionConfig,
    VLMEigenfunctionModel,
    VLMTransitionDataset,
    VLMStateDataset,
    VLMCollator,
    vlm_eigenfunction_loss,
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
# Force unbuffered stderr for SLURM
import sys
sys.stderr.reconfigure(line_buffering=True)
logger = logging.getLogger(__name__)

DEFAULT_DATA_DIR = PROJECT_ROOT / "outputs" / "vlm_fnet" / "data"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "vlm_fnet"
DEFAULT_REGISTRY_PATH = PROJECT_ROOT / "outputs" / "transitions" / "gui360_full" / "state_registry.json"
DEFAULT_MLP_FNET_DIR = PROJECT_ROOT / "outputs" / "fnet" / "gui360"

APPS = ["excel", "word", "ppt"]


def build_manifest_lookup(manifest: list[dict]) -> dict[str, dict]:
    """Build hash → state_info lookup from manifest list."""
    return {entry["hash"]: entry for entry in manifest}


def train_one_app(
    app: str,
    config: VLMEigenfunctionConfig,
    data_dir: Path,
    output_dir: Path,
    device: str,
    max_states: int | None = None,
    registry_path: Path | None = None,
    mlp_fnet_dir: Path | None = None,
):
    """Train VLM eigenfunction for a single app."""
    app_data_dir = data_dir / app
    app_output_dir = output_dir / app
    app_output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    manifest_path = app_data_dir / "state_manifest.json"
    transitions_path = app_data_dir / "transition_pairs.json"

    if not manifest_path.exists():
        logger.error(f"Manifest not found: {manifest_path}. Run prepare_vlm_eigenfunction_data.py first.")
        return

    with open(manifest_path) as f:
        manifest_list = json.load(f)

    if max_states and len(manifest_list) > max_states:
        manifest_list = manifest_list[:max_states]
        logger.info(f"Limited to {max_states} states for testing")

    manifest_lookup = build_manifest_lookup(manifest_list)
    logger.info(f"Loaded manifest: {len(manifest_lookup)} states")

    # Load transitions, filtering to states in manifest
    transition_dataset = VLMTransitionDataset(transitions_path)

    # Filter transitions to states in manifest
    valid_transitions = []
    for t in transition_dataset.transitions:
        if t["src_hash"] in manifest_lookup and t["dst_hash"] in manifest_lookup:
            valid_transitions.append(t)
    transition_dataset.transitions = valid_transitions
    logger.info(f"Valid transitions after filtering: {len(valid_transitions)}")

    if len(valid_transitions) == 0:
        logger.error("No valid transitions. Check data preparation.")
        return

    # State dataset for random sampling
    state_dataset = VLMStateDataset(manifest_path)
    if max_states:
        state_dataset.hashes = state_dataset.hashes[:max_states]

    # Build model
    logger.info("Building VLM model...")
    model = VLMEigenfunctionModel(config)
    model.to(device)

    # Load processor
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(
        config.model_path,
        min_pixels=config.image_size * config.image_size,
        max_pixels=config.image_size * config.image_size,
    )

    # Collator
    collator = VLMCollator(
        manifest=manifest_lookup,
        processor=processor,
        input_mode=config.input_mode,
        image_size=config.image_size,
    )

    # Transition DataLoader
    transition_loader = DataLoader(
        transition_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
    )

    # Optimizer (only trainable params)
    trainable_params = model.get_trainable_params()
    optimizer = torch.optim.AdamW(
        trainable_params, lr=config.lr, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.num_epochs
    )

    # Save config
    with open(app_output_dir / "config.json", "w") as f:
        json.dump(config.to_dict(), f, indent=2)

    # Training loop
    logger.info(f"Starting training for {config.num_epochs} epochs, "
                f"{len(transition_loader)} batches/epoch")
    history = []
    t0 = time.time()

    for epoch in range(1, config.num_epochs + 1):
        model.train()
        epoch_metrics = _train_one_epoch(
            model=model,
            optimizer=optimizer,
            transition_loader=transition_loader,
            state_dataset=state_dataset,
            collator=collator,
            config=config,
            device=device,
            epoch=epoch,
        )
        scheduler.step()

        epoch_metrics["epoch"] = epoch
        epoch_metrics["lr"] = scheduler.get_last_lr()[0]
        history.append(epoch_metrics)

        elapsed = time.time() - t0
        logger.info(
            f"Epoch {epoch}/{config.num_epochs}  "
            f"loss={epoch_metrics['loss']:.4f}  "
            f"smooth={epoch_metrics['smoothness']:.4f}  "
            f"repul={epoch_metrics['repulsive']:.4f}  "
            f"[{elapsed:.0f}s]"
        )

        # Save checkpoint every epoch (epochs are expensive with VLM)
        if epoch % config.save_every == 0:
            ckpt_dir = app_output_dir / f"checkpoint_epoch{epoch}"
            model.save_adapter(ckpt_dir)

    total_time = time.time() - t0
    logger.info(f"Training complete in {total_time:.1f}s ({total_time/60:.1f}min)")

    # Save final model
    model.save_adapter(app_output_dir)

    # Compute f-values for all states
    logger.info("Computing f-values for all states...")
    model.eval()
    all_hashes = list(manifest_lookup.keys())
    all_f_values = _compute_all_f_values(
        model=model,
        all_hashes=all_hashes,
        collator=collator,
        device=device,
        batch_size=config.batch_size,
    )

    # Identify bottlenecks
    state_hashes_arr = np.array(all_hashes)
    f_values_arr = np.array(all_f_values)

    bottleneck_info = identify_bottlenecks(
        f_values_arr, state_hashes_arr, config.percentile_k
    )

    # Save results
    results = {
        "app": app,
        "config": config.to_dict(),
        "training_time_seconds": round(total_time, 1),
        "num_transitions": len(valid_transitions),
        "num_states": len(manifest_lookup),
        "device": device,
        "final_loss": history[-1]["loss"],
        "final_smoothness": history[-1]["smoothness"],
        "final_repulsive": history[-1]["repulsive"],
        "f_value_stats": {
            "mean": float(f_values_arr.mean()),
            "std": float(f_values_arr.std()),
            "min": float(f_values_arr.min()),
            "max": float(f_values_arr.max()),
        },
        "bottleneck_threshold": bottleneck_info["threshold"],
        "num_bottleneck_states": bottleneck_info["num_bottleneck_states"],
    }

    with open(app_output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    np.savez(
        app_output_dir / "f_values.npz",
        hashes=state_hashes_arr,
        f_values=f_values_arr,
    )

    with open(app_output_dir / "bottlenecks.json", "w") as f:
        json.dump(bottleneck_info, f, indent=2)

    with open(app_output_dir / "training_history.json", "w") as f:
        json.dump(history, f)

    # Map bottlenecks to descriptions if registry available
    if registry_path and registry_path.exists():
        described = map_bottlenecks_to_descriptions(bottleneck_info, registry_path)
        with open(app_output_dir / "bottlenecks_described.json", "w") as f:
            json.dump(described, f, indent=2)
    else:
        described = bottleneck_info.get("bottleneck_states", [])

    # Compare with MLP f-values if available
    if mlp_fnet_dir:
        _compare_with_mlp(app, f_values_arr, state_hashes_arr, mlp_fnet_dir, app_output_dir)

    # Print summary
    _print_summary(app, described, results)

    logger.info(f"All results saved to {app_output_dir}")


def _train_one_epoch(
    model: VLMEigenfunctionModel,
    optimizer: torch.optim.Optimizer,
    transition_loader: DataLoader,
    state_dataset: VLMStateDataset,
    collator: VLMCollator,
    config: VLMEigenfunctionConfig,
    device: str,
    epoch: int,
) -> dict[str, float]:
    """Train one epoch, returning averaged metrics."""
    total_metrics: dict[str, float] = {}
    n_batches = 0

    for batch_idx, (src_hashes, dst_hashes) in enumerate(transition_loader):
        B = len(src_hashes)

        # Sample random states for repulsive term
        rand_hashes = state_dataset.sample(B)

        # Stack all 3 groups for single VLM forward pass
        all_hashes = list(src_hashes) + list(dst_hashes) + rand_hashes

        if batch_idx == 0:
            logger.info(f"  First batch: {len(all_hashes)} hashes, collating...")
            import sys; sys.stderr.flush()

        # Build VLM batch
        try:
            batch = collator(all_hashes)
        except Exception as e:
            logger.warning(f"Batch {batch_idx} collation failed: {e}, skipping")
            continue

        if batch_idx == 0:
            logger.info(f"  Collation done. Keys: {list(batch.keys())}")
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    logger.info(f"    {k}: shape={v.shape}, dtype={v.dtype}")
            sys.stderr.flush()

        # Move to device
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        if batch_idx == 0:
            logger.info("  Starting forward pass...")
            sys.stderr.flush()

        # Forward pass (single call for all 3 groups)
        f_all = model(**batch)  # (3B, 1)

        if batch_idx == 0:
            logger.info(f"  Forward done. f_all shape={f_all.shape}, dtype={f_all.dtype}")
            sys.stderr.flush()

        f_src, f_dst, f_rand = f_all.chunk(3, dim=0)

        # Eigenfunction loss
        loss, metrics = vlm_eigenfunction_loss(f_src, f_dst, f_rand, config.eta)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        if batch_idx == 0:
            logger.info(f"  Backward done. loss={loss.item():.4f}")
            sys.stderr.flush()

        torch.nn.utils.clip_grad_norm_(model.get_trainable_params(), max_norm=1.0)
        optimizer.step()

        for k, v in metrics.items():
            total_metrics[k] = total_metrics.get(k, 0.0) + v
        n_batches += 1

        if batch_idx % config.log_every == 0:
            logger.info(
                f"  Epoch {epoch} batch {batch_idx}/{len(transition_loader)}  "
                f"loss={metrics['loss']:.4f}  smooth={metrics['smoothness']:.4f}"
            )

    if n_batches == 0:
        return {"loss": 0.0, "smoothness": 0.0, "repulsive": 0.0}

    return {k: v / n_batches for k, v in total_metrics.items()}


@torch.no_grad()
def _compute_all_f_values(
    model: VLMEigenfunctionModel,
    all_hashes: list[str],
    collator: VLMCollator,
    device: str,
    batch_size: int = 4,
) -> list[float]:
    """Compute f-values for all states."""
    model.eval()
    f_values = []

    for i in range(0, len(all_hashes), batch_size):
        batch_hashes = all_hashes[i:i + batch_size]
        try:
            batch = collator(batch_hashes)
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            f_batch = model(**batch)
            f_values.extend(f_batch.float().cpu().numpy().flatten().tolist())
        except Exception as e:
            logger.warning(f"Failed to compute f-values for batch starting at {i}: {e}")
            f_values.extend([0.0] * len(batch_hashes))

    return f_values


def _compare_with_mlp(
    app: str,
    vlm_f_values: np.ndarray,
    vlm_hashes: np.ndarray,
    mlp_fnet_dir: Path,
    output_dir: Path,
):
    """Compare VLM bottlenecks with MLP bottlenecks."""
    mlp_path = mlp_fnet_dir / app / "f_values.npz"
    if not mlp_path.exists():
        logger.info(f"MLP f-values not found at {mlp_path}, skipping comparison")
        return

    mlp_data = np.load(mlp_path, allow_pickle=True)
    mlp_hashes = set(str(h) for h in mlp_data["hashes"])
    mlp_f_map = {str(h): float(fv) for h, fv in zip(mlp_data["hashes"], mlp_data["f_values"])}

    # Find common hashes
    vlm_hash_set = set(str(h) for h in vlm_hashes)
    common = vlm_hash_set & mlp_hashes
    logger.info(f"Comparison: {len(common)} common hashes between VLM and MLP")

    # VLM bottlenecks (below 30th percentile)
    vlm_threshold = float(np.percentile(vlm_f_values, 30))
    vlm_bottleneck_set = set(
        str(h) for h, fv in zip(vlm_hashes, vlm_f_values) if fv < vlm_threshold
    )

    # MLP bottlenecks
    mlp_f_arr = mlp_data["f_values"]
    mlp_threshold = float(np.percentile(mlp_f_arr, 30))
    mlp_bottleneck_set = set(
        str(h) for h, fv in zip(mlp_data["hashes"], mlp_f_arr) if fv < mlp_threshold
    )

    # Overlap
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

    with open(output_dir / "mlp_comparison.json", "w") as f:
        json.dump(comparison, f, indent=2)

    logger.info(
        f"Bottleneck comparison: overlap={len(both)}, "
        f"VLM-only={len(vlm_only)}, MLP-only={len(mlp_only)}, "
        f"Jaccard={comparison['jaccard_index']:.3f}"
    )


def _print_summary(app: str, described: list[dict], results: dict):
    """Print a human-readable summary."""
    print(f"\n{'='*60}")
    print(f"  {app.upper()} — VLM Eigenfunction Summary")
    print(f"{'='*60}")
    print(f"  Training: {results['training_time_seconds']:.0f}s, "
          f"final loss={results['final_loss']:.4f}")
    print(f"  f-values: mean={results['f_value_stats']['mean']:.3f}, "
          f"std={results['f_value_stats']['std']:.3f}, "
          f"range=[{results['f_value_stats']['min']:.3f}, {results['f_value_stats']['max']:.3f}]")
    print(f"  Bottleneck states: {results['num_bottleneck_states']} "
          f"(threshold={results['bottleneck_threshold']:.4f})")
    print()
    print(f"  Top-20 bottleneck states (lowest f-values):")
    for i, entry in enumerate(described[:20]):
        if isinstance(entry, dict):
            f_val = entry.get("f_value", 0.0)
            dialog = entry.get("dialog_state", "none")
            tabs = entry.get("active_tab_signature", "")
            if isinstance(tabs, str):
                tabs_short = tabs[:40] + "..." if len(tabs) > 40 else tabs
            else:
                tabs_short = str(tabs)[:40]
            print(f"    {i+1:3d}. f={f_val:+.4f}  "
                  f"dialog={dialog:30s}  tabs=[{tabs_short}]")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Train VLM eigenfunction (Task 3.1)",
    )
    parser.add_argument(
        "--app", type=str, choices=APPS + ["all"], default="all",
        help="Train on a single app or all apps (default: all)",
    )
    parser.add_argument(
        "--data-dir", type=Path, default=DEFAULT_DATA_DIR,
        help="Directory containing prepared data (state_manifest.json, transition_pairs.json)",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
        help="Output directory for trained models and results",
    )
    parser.add_argument(
        "--model-path", type=str,
        default="checkpoints/Qwen2.5-VL-7B-Instruct",
        help="Path to Qwen2.5-VL model",
    )
    parser.add_argument("--input-mode", type=str, default="screenshot_a11y",
                        choices=["screenshot", "screenshot_a11y"])
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--eta", type=float, default=1.0)
    parser.add_argument("--image-size", type=int, default=448)
    parser.add_argument("--percentile-k", type=float, default=30.0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--max-states", type=int, default=None,
                        help="Limit number of states (for CPU smoke test)")
    parser.add_argument(
        "--registry-path", type=Path, default=DEFAULT_REGISTRY_PATH,
        help="Path to state_registry.json (for bottleneck descriptions)",
    )
    parser.add_argument(
        "--mlp-fnet-dir", type=Path, default=DEFAULT_MLP_FNET_DIR,
        help="MLP f_net output dir for comparison",
    )
    args = parser.parse_args()

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    logger.info(f"Using device: {device}")

    config = VLMEigenfunctionConfig(
        model_path=args.model_path,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        input_mode=args.input_mode,
        image_size=args.image_size,
        eta=args.eta,
        batch_size=args.batch_size,
        lr=args.lr,
        num_epochs=args.epochs,
        percentile_k=args.percentile_k,
        bf16=(device != "cpu"),
    )

    apps_to_train = APPS if args.app == "all" else [args.app]

    for app in apps_to_train:
        logger.info(f"\n{'='*60}")
        logger.info(f"Training VLM eigenfunction for app: {app}")
        logger.info(f"{'='*60}")

        train_one_app(
            app=app,
            config=config,
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            device=device,
            max_states=args.max_states,
            registry_path=args.registry_path,
            mlp_fnet_dir=args.mlp_fnet_dir,
        )

    logger.info("\nAll training complete!")


if __name__ == "__main__":
    main()
