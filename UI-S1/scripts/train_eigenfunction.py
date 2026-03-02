#!/usr/bin/env python3
"""
Task 3: Train the Laplacian eigenfunction approximator (f_net).

Trains a neural network to approximate the second eigenvector of the graph
Laplacian on GUI-360 state transitions. After training, identifies bottleneck
states (hardest-to-reach UI states) via f-value percentile thresholding.

Usage:
    # Train on all apps (recommended: per-app mode)
    python scripts/train_eigenfunction.py

    # Train on a single app
    python scripts/train_eigenfunction.py --app excel

    # Quick test (fewer epochs)
    python scripts/train_eigenfunction.py --epochs 20 --app word

    # Custom output
    python scripts/train_eigenfunction.py --output-dir outputs/fnet/exp1

Inputs (from Task 2):
    outputs/transitions/gui360_full/transition_pairs.npz
    outputs/transitions/gui360_full/all_state_embeddings.npz
    outputs/transitions/gui360_full/state_registry.json

Outputs (in --output-dir):
    Per-app subdirectories (e.g., excel/, word/, ppt/) each containing:
        f_net_final.pt           — Trained model checkpoint
        f_values.npz             — f-values for all states (hashes + values)
        bottlenecks.json         — Identified bottleneck states
        results.json             — Training results and statistics
        training_history.json    — Per-epoch loss history
        config.json              — Training configuration
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from verl.models.moe.graph_analysis import (
    EigenfunctionConfig,
    train_eigenfunction,
    map_bottlenecks_to_descriptions,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Default paths
DEFAULT_TRANSITIONS_DIR = PROJECT_ROOT / "outputs" / "transitions" / "gui360_full"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "fnet" / "gui360"

APPS = ["excel", "word", "ppt"]


def main():
    parser = argparse.ArgumentParser(
        description="Train Laplacian eigenfunction approximator (f_net)",
    )
    parser.add_argument(
        "--transitions-dir", type=Path, default=DEFAULT_TRANSITIONS_DIR,
        help="Directory containing transition_pairs.npz and all_state_embeddings.npz",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
        help="Output directory for trained models and results",
    )
    parser.add_argument(
        "--app", type=str, choices=APPS + ["all"], default="all",
        help="Train on a single app or all apps separately (default: all)",
    )
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--eta", type=float, default=1.0)
    parser.add_argument("--percentile-k", type=float, default=30.0)
    parser.add_argument("--hidden-dims", type=int, nargs="+", default=[256, 256])
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    trans_dir = args.transitions_dir
    pairs_path = trans_dir / "transition_pairs.npz"
    states_path = trans_dir / "all_state_embeddings.npz"
    registry_path = trans_dir / "state_registry.json"

    if not pairs_path.exists():
        logger.error(f"Transition pairs not found: {pairs_path}")
        sys.exit(1)

    apps_to_train = APPS if args.app == "all" else [args.app]

    for app in apps_to_train:
        logger.info(f"\n{'='*60}")
        logger.info(f"Training f_net for app: {app}")
        logger.info(f"{'='*60}")

        config = EigenfunctionConfig(
            hidden_dims=args.hidden_dims,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            eta=args.eta,
            percentile_k=args.percentile_k,
            per_app=True,
            app_filter=app,
        )

        app_output_dir = args.output_dir / app
        f_net, results = train_eigenfunction(
            transition_pairs_path=pairs_path,
            state_embeddings_path=states_path,
            config=config,
            output_dir=app_output_dir,
            device=args.device,
            state_registry_path=registry_path,
        )

        # Map bottlenecks to human-readable descriptions (for Task 4)
        import json
        with open(app_output_dir / "bottlenecks.json") as f:
            bottleneck_info = json.load(f)

        # Always use the same registry as the transitions dir —
        # coarse hashes differ from fine hashes, so cross-lookup fails.
        described = map_bottlenecks_to_descriptions(bottleneck_info, registry_path)
        with open(app_output_dir / "bottlenecks_described.json", "w") as f:
            json.dump(described, f, indent=2)

        # Print summary
        _print_bottleneck_summary(app, described, results)

    logger.info("\nAll training complete!")


def _print_bottleneck_summary(app: str, described: list[dict], results: dict):
    """Print a human-readable bottleneck summary."""
    print(f"\n{'='*60}")
    print(f"  {app.upper()} — Bottleneck Summary")
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
        dialog = entry.get("dialog_state", "none")
        tabs = entry.get("active_tab_signature", "")
        tabs_short = tabs[:40] + "..." if len(tabs) > 40 else tabs
        print(f"    {i+1:3d}. f={entry['f_value']:+.4f}  "
              f"dialog={dialog:30s}  tabs=[{tabs_short}]")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
