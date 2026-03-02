#!/usr/bin/env python3
"""
Task 2: Collect state transitions from GUI-360 trajectories and build a transition graph.

This script reads all GUI-360 trajectory JSONL files, extracts state IDs and
embeddings using the state representation module (Task 1), builds a state
transition graph, and saves all outputs for downstream spectral analysis (Task 3).

Usage:
    # Full run (all 13,750 trajectories, ~6 min):
    python scripts/collect_transitions.py

    # Quick test run:
    python scripts/collect_transitions.py --max-files 200

    # Custom output directory:
    python scripts/collect_transitions.py --output-dir outputs/transitions/exp1

    # Include failed trajectories:
    python scripts/collect_transitions.py --include-failed

Outputs (in --output-dir):
    transitions.jsonl        — All (state_hash, next_state_hash, action) records
    state_registry.json      — StateID metadata for each unique state hash
    state_embeddings.npz     — Dense embeddings (N_states x 43) for f_net training
    adjacency.json           — Adjacency dict: {src_hash: {dst_hash: count}}
    statistics.json          — Graph statistics summary
    per_app_statistics.json  — Statistics broken down by app domain
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from verl.models.moe.state_representation import (
    GUI360StepData,
    GUI360TrajectoryProcessor,
    TransitionRecord,
    extract_state_id,
    save_transitions,
    get_embedding_dim,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Default paths
DEFAULT_DATA_DIR = PROJECT_ROOT / "datasets" / "GUI-360" / "train" / "data"
DEFAULT_FAIL_DIR = PROJECT_ROOT / "datasets" / "GUI-360" / "fail" / "data"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "transitions" / "gui360"


def collect_transitions(
    data_dir: Path,
    output_dir: Path,
    max_files: int | None = None,
    compute_embeddings: bool = True,
    include_failed: bool = False,
    fail_dir: Path | None = None,
    granularity: str = "fine",
) -> dict:
    """Main collection pipeline.

    Args:
        data_dir: Root of GUI-360 train/data directory.
        output_dir: Where to save all outputs.
        max_files: Limit on number of files (for testing).
        compute_embeddings: Whether to compute dense state embeddings.
        granularity: "fine" or "coarse" state abstraction level.
        include_failed: Whether to also process failed trajectories.
        fail_dir: Root of GUI-360 fail/data directory.

    Returns:
        Statistics dict.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    processor = GUI360TrajectoryProcessor(granularity=granularity)
    logger.info(f"Granularity: {granularity}")

    # --- Phase 1: Process successful trajectories ---
    logger.info(f"Processing successful trajectories from {data_dir}")
    t0 = time.time()
    transitions = processor.process_directory(
        data_dir, compute_embeddings=compute_embeddings, max_files=max_files,
    )
    t_success = time.time() - t0
    logger.info(f"Successful trajectories: {len(transitions)} transitions in {t_success:.1f}s")

    # --- Phase 2: Optionally process failed trajectories ---
    if include_failed and fail_dir and fail_dir.exists():
        logger.info(f"Processing failed trajectories from {fail_dir}")
        t1 = time.time()
        fail_transitions = processor.process_directory(
            fail_dir, compute_embeddings=compute_embeddings, max_files=max_files,
        )
        transitions.extend(fail_transitions)
        t_fail = time.time() - t1
        logger.info(f"Failed trajectories: {len(fail_transitions)} transitions in {t_fail:.1f}s")

    # --- Phase 3: Build adjacency and compute statistics ---
    adjacency = processor.build_adjacency_dict(transitions)
    stats = processor.get_statistics(transitions)

    # Per-app statistics
    per_app_stats = _compute_per_app_statistics(transitions, processor)

    # Action type distribution
    action_dist = _compute_action_distribution(transitions)
    stats["action_distribution"] = action_dist

    # --- Phase 4: Save outputs ---
    logger.info(f"Saving outputs to {output_dir}")

    # 1. Transitions JSONL
    save_transitions(transitions, output_dir / "transitions.jsonl")

    # 2. State registry + embeddings (via processor.save)
    processor.save(output_dir)

    # 3. Adjacency dict
    with open(output_dir / "adjacency.json", "w") as f:
        json.dump(adjacency, f)
    logger.info(f"Saved adjacency dict ({len(adjacency)} source states)")

    # 4. Statistics
    stats["embedding_dim"] = get_embedding_dim()
    stats["data_dir"] = str(data_dir)
    stats["include_failed"] = include_failed
    stats["max_files"] = max_files
    stats["processing_time_seconds"] = round(time.time() - t0, 1)

    with open(output_dir / "statistics.json", "w") as f:
        json.dump(stats, f, indent=2, default=_json_default)
    logger.info(f"Saved statistics")

    # 5. Per-app statistics
    with open(output_dir / "per_app_statistics.json", "w") as f:
        json.dump(per_app_stats, f, indent=2, default=_json_default)

    # 6. Save transition pairs as numpy arrays (for efficient f_net training)
    if compute_embeddings:
        _save_training_data(transitions, processor, output_dir)

    # --- Phase 5: Print summary ---
    _print_summary(stats, per_app_stats, output_dir)

    return stats


def _compute_per_app_statistics(
    transitions: list[TransitionRecord],
    processor: GUI360TrajectoryProcessor,
) -> dict:
    """Break down statistics by app domain."""
    # Group states by app domain
    state_apps: dict[str, str] = {}
    for sid in processor.get_all_state_ids():
        state_apps[sid.hash_value] = sid.app_domain

    # Group transitions by source state's app
    app_transitions: dict[str, list[TransitionRecord]] = defaultdict(list)
    for t in transitions:
        app = state_apps.get(t.state_id.hash_value, "unknown")
        app_transitions[app].append(t)

    per_app = {}
    for app in ("excel", "word", "ppt"):
        app_trans = app_transitions.get(app, [])
        app_states = {h for h, a in state_apps.items() if a == app}

        # Unique edges for this app
        edge_set = set()
        for t in app_trans:
            edge_set.add((t.state_id.hash_value, t.next_state_id.hash_value))

        per_app[app] = {
            "num_states": len(app_states),
            "num_transitions": len(app_trans),
            "num_unique_edges": len(edge_set),
        }

    return per_app


def _compute_action_distribution(transitions: list[TransitionRecord]) -> dict:
    """Count action types and functions across all transitions."""
    type_counts = Counter()
    func_counts = Counter()
    for t in transitions:
        type_counts[t.action_type] += 1
        func_counts[t.action_function] += 1

    return {
        "by_action_type": dict(type_counts.most_common()),
        "by_function": dict(func_counts.most_common(20)),
    }


def _save_training_data(
    transitions: list[TransitionRecord],
    processor: GUI360TrajectoryProcessor,
    output_dir: Path,
) -> None:
    """Save transition pairs as numpy arrays for efficient f_net training.

    Saves:
    - transition_pairs.npz: per-step src/dst embeddings (actual, not hash-looked-up)
    - all_state_embeddings.npz: all unique state embeddings + hashes

    IMPORTANT: transition_pairs uses per-step embeddings stored directly on each
    TransitionRecord. This ensures f_net training data is independent of hashing
    granularity (fine/coarse). The hash-based all_state_embeddings.npz is a
    separate output used for graph-level analysis only.
    """
    # Build transition pairs from per-step embeddings (granularity-independent)
    src_list = []
    dst_list = []
    skipped = 0
    for t in transitions:
        if t.src_embedding is not None and t.dst_embedding is not None:
            src_list.append(t.src_embedding)
            dst_list.append(t.dst_embedding)
        else:
            skipped += 1

    if skipped > 0:
        logger.warning(f"Skipped {skipped} transitions without embeddings")

    if not src_list:
        logger.warning("No transition pairs with embeddings, skipping training data save")
        return

    src_array = np.stack(src_list)
    dst_array = np.stack(dst_list)

    np.savez(
        output_dir / "transition_pairs.npz",
        src_embeddings=src_array,
        dst_embeddings=dst_array,
    )
    logger.info(
        f"Saved transition pairs: {src_array.shape[0]} pairs, "
        f"embedding dim = {src_array.shape[1]}"
    )

    # All unique state embeddings (hash-keyed, for graph-level analysis)
    state_embeddings = processor.get_state_embeddings()
    if not state_embeddings:
        return
    hashes = sorted(state_embeddings.keys())
    all_emb = np.stack([state_embeddings[h] for h in hashes])
    np.savez(
        output_dir / "all_state_embeddings.npz",
        hashes=np.array(hashes),
        embeddings=all_emb,
    )
    logger.info(f"Saved {len(hashes)} unique state embeddings")


def _print_summary(stats: dict, per_app: dict, output_dir: Path) -> None:
    """Print a human-readable summary."""
    print("\n" + "=" * 60)
    print("  GUI-360 Transition Collection Summary")
    print("=" * 60)
    print(f"  Unique states:     {stats['num_states']:,}")
    print(f"  Total transitions: {stats['num_transitions']:,}")
    print(f"  Unique edges:      {stats['num_unique_edges']:,}")
    print(f"  Graph density:     {stats['graph_density']:.6f}")
    print(f"  Avg out-degree:    {stats['avg_out_degree']:.2f}")
    print(f"  Max out-degree:    {stats['max_out_degree']}")
    print(f"  Embedding dim:     {stats['embedding_dim']}")
    print(f"  Processing time:   {stats['processing_time_seconds']:.1f}s")
    print()
    print("  Per-app breakdown:")
    for app in ("excel", "word", "ppt"):
        a = per_app.get(app, {})
        print(f"    {app:12s}  states={a.get('num_states', 0):>5,}  "
              f"transitions={a.get('num_transitions', 0):>6,}  "
              f"edges={a.get('num_unique_edges', 0):>5,}")
    print()

    # Action distribution
    action_dist = stats.get("action_distribution", {})
    if action_dist.get("by_function"):
        print("  Top action functions:")
        for func, cnt in list(action_dist["by_function"].items())[:10]:
            print(f"    {func:25s}  {cnt:>7,}")
    print()
    print(f"  Output directory: {output_dir}")
    print("=" * 60 + "\n")


def _json_default(obj):
    """JSON serializer for numpy types."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def main():
    parser = argparse.ArgumentParser(
        description="Collect state transitions from GUI-360 trajectories",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--data-dir", type=Path, default=DEFAULT_DATA_DIR,
        help=f"Path to GUI-360 train/data directory (default: {DEFAULT_DATA_DIR})",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--max-files", type=int, default=None,
        help="Limit number of trajectory files to process (for testing)",
    )
    parser.add_argument(
        "--no-embeddings", action="store_true",
        help="Skip computing dense state embeddings (faster, but no .npz output)",
    )
    parser.add_argument(
        "--include-failed", action="store_true",
        help="Also process failed trajectories from fail/data",
    )
    parser.add_argument(
        "--fail-dir", type=Path, default=DEFAULT_FAIL_DIR,
        help=f"Path to GUI-360 fail/data directory (default: {DEFAULT_FAIL_DIR})",
    )
    parser.add_argument(
        "--granularity", choices=["fine", "coarse"], default="fine",
        help="State abstraction level: 'fine' (~6K states) or 'coarse' (~600 states)",
    )
    args = parser.parse_args()

    if not args.data_dir.exists():
        logger.error(f"Data directory not found: {args.data_dir}")
        sys.exit(1)

    collect_transitions(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        max_files=args.max_files,
        compute_embeddings=not args.no_embeddings,
        include_failed=args.include_failed,
        fail_dir=args.fail_dir,
        granularity=args.granularity,
    )


if __name__ == "__main__":
    main()
