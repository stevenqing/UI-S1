#!/usr/bin/env python3
"""
Precompute f_pseudo reward shaping values from trained f_net models.

f_pseudo(t) = f(s_t) - f(s_{t+1})

When an agent moves from a high-f region toward a low-f region (i.e., toward
a connectivity bottleneck), f_pseudo > 0, providing a positive reward signal.

Inputs:
  - outputs/fnet/gui360/{app}/f_values.npz      -> {hash: f_value} for known states
  - outputs/fnet/gui360/{app}/f_net_final.pt     -> for inference on unknown states
  - outputs/transitions/gui360_full/transitions.jsonl -> state transitions
  - GUI-360 raw trajectory data (datasets/GUI-360/train/data/)

Outputs:
  - outputs/f_pseudo/f_pseudo_map.json
    {execution_id: {step_id_str: f_pseudo_value, ...}, ...}
  - outputs/f_pseudo/statistics.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from verl.models.moe.graph_analysis import EigenfunctionNet, EigenfunctionConfig, load_f_net
from verl.models.moe.state_representation import (
    GUI360StepData,
    extract_state_id,
    extract_state_embedding,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def load_f_value_maps(fnet_dir: Path) -> dict[str, float]:
    """Load precomputed f-values from all apps into a single hash->value map."""
    f_value_map: dict[str, float] = {}
    apps = ["excel", "word", "ppt"]
    for app in apps:
        npz_path = fnet_dir / app / "f_values.npz"
        if not npz_path.exists():
            logger.warning(f"f_values.npz not found for {app} at {npz_path}")
            continue
        data = np.load(npz_path)
        hashes = data["hashes"]
        f_values = data["f_values"]
        for h, fv in zip(hashes, f_values):
            f_value_map[str(h)] = float(fv)
        logger.info(f"Loaded {len(hashes)} f-values for {app} (range [{f_values.min():.3f}, {f_values.max():.3f}])")

    logger.info(f"Total f-value map: {len(f_value_map)} states")
    return f_value_map


def load_f_nets(fnet_dir: Path, device: str = "cpu") -> dict[str, EigenfunctionNet]:
    """Load trained f_net models for fallback inference on unknown states."""
    f_nets: dict[str, EigenfunctionNet] = {}
    apps = ["excel", "word", "ppt"]
    for app in apps:
        ckpt_path = fnet_dir / app / "f_net_final.pt"
        if not ckpt_path.exists():
            logger.warning(f"f_net_final.pt not found for {app} at {ckpt_path}")
            continue
        f_nets[app] = load_f_net(ckpt_path, device=device)
        logger.info(f"Loaded f_net for {app}")
    return f_nets


def compute_f_value(
    state_hash: str,
    f_value_map: dict[str, float],
) -> float | None:
    """Look up f-value for a state hash. Returns None if not found."""
    return f_value_map.get(state_hash)


def infer_f_value(
    embedding: np.ndarray,
    app_domain: str,
    f_nets: dict[str, EigenfunctionNet],
    device: str = "cpu",
) -> float | None:
    """Infer f-value using the trained f_net for unknown states."""
    if app_domain not in f_nets:
        return None
    f_net = f_nets[app_domain]
    with torch.no_grad():
        x = torch.from_numpy(embedding).float().unsqueeze(0).to(device)
        fv = f_net(x).item()
    return fv


def process_transitions(
    transitions_path: Path,
    f_value_map: dict[str, float],
) -> dict[str, dict[str, float]]:
    """Process transitions.jsonl to compute f_pseudo per (execution_id, step_id).

    Uses the precomputed transitions which already contain state_hash and
    next_state_hash for each step.

    Returns:
        {execution_id: {step_id_str: f_pseudo_value}}
    """
    f_pseudo_map: dict[str, dict[str, float]] = defaultdict(dict)
    stats = {"total": 0, "both_found": 0, "src_missing": 0, "dst_missing": 0}

    with open(transitions_path) as f:
        for line in f:
            record = json.loads(line)
            execution_id = record["execution_id"]
            step_id = record["step_id"]
            src_hash = record["state_hash"]
            dst_hash = record["next_state_hash"]

            stats["total"] += 1

            src_f = f_value_map.get(src_hash)
            dst_f = f_value_map.get(dst_hash)

            if src_f is not None and dst_f is not None:
                f_pseudo = src_f - dst_f
                f_pseudo_map[execution_id][str(step_id)] = round(f_pseudo, 6)
                stats["both_found"] += 1
            elif src_f is None:
                stats["src_missing"] += 1
            else:
                stats["dst_missing"] += 1

    return dict(f_pseudo_map), stats


def process_raw_trajectories(
    data_dir: Path,
    f_value_map: dict[str, float],
    f_nets: dict[str, EigenfunctionNet],
    device: str = "cpu",
) -> tuple[dict[str, dict[str, float]], dict]:
    """Process raw GUI-360 trajectory files to compute f_pseudo.

    This handles cases where transitions.jsonl may not cover all training data
    by extracting state hashes directly from the raw trajectory files.

    Returns:
        (f_pseudo_map, stats)
    """
    f_pseudo_map: dict[str, dict[str, float]] = defaultdict(dict)
    stats = {
        "total_trajectories": 0,
        "total_steps": 0,
        "total_transitions": 0,
        "both_found": 0,
        "inferred": 0,
        "missing": 0,
    }

    traj_files = sorted(data_dir.rglob("*.jsonl"))
    logger.info(f"Found {len(traj_files)} trajectory files in {data_dir}")

    for fpath in traj_files:
        with open(fpath) as f:
            lines = f.readlines()

        if len(lines) < 2:
            continue

        stats["total_trajectories"] += 1
        stats["total_steps"] += len(lines)

        # Parse all steps in this trajectory
        steps_data = []
        for line in lines:
            raw = json.loads(line)
            step_data = GUI360StepData.from_raw(raw)
            state_id = extract_state_id(step_data)
            embedding = extract_state_embedding(step_data)
            steps_data.append((raw, step_data, state_id, embedding))

        execution_id = steps_data[0][0].get("execution_id", fpath.stem)

        # Compute f_pseudo for consecutive step pairs
        for i in range(len(steps_data) - 1):
            raw_curr, step_curr, sid_curr, emb_curr = steps_data[i]
            raw_next, step_next, sid_next, emb_next = steps_data[i + 1]

            step_id = raw_curr.get("step_id", i + 1)
            stats["total_transitions"] += 1

            # Look up or infer f-values
            f_curr = compute_f_value(sid_curr.hash_value, f_value_map)
            if f_curr is None:
                f_curr = infer_f_value(emb_curr, step_curr.app_domain, f_nets, device)
                if f_curr is not None:
                    stats["inferred"] += 1

            f_next = compute_f_value(sid_next.hash_value, f_value_map)
            if f_next is None:
                f_next = infer_f_value(emb_next, step_next.app_domain, f_nets, device)
                if f_next is not None:
                    stats["inferred"] += 1

            if f_curr is not None and f_next is not None:
                f_pseudo = f_curr - f_next
                f_pseudo_map[execution_id][str(step_id)] = round(f_pseudo, 6)
                stats["both_found"] += 1
            else:
                stats["missing"] += 1

    return dict(f_pseudo_map), stats


def compute_statistics(f_pseudo_map: dict[str, dict[str, float]]) -> dict:
    """Compute summary statistics of the f_pseudo distribution."""
    all_values = []
    for exec_id, steps in f_pseudo_map.items():
        all_values.extend(steps.values())

    if not all_values:
        return {"num_trajectories": 0, "num_steps": 0}

    arr = np.array(all_values)
    return {
        "num_trajectories": len(f_pseudo_map),
        "num_steps": len(all_values),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "p5": float(np.percentile(arr, 5)),
        "p25": float(np.percentile(arr, 25)),
        "p50": float(np.percentile(arr, 50)),
        "p75": float(np.percentile(arr, 75)),
        "p95": float(np.percentile(arr, 95)),
        "num_positive": int((arr > 0.01).sum()),
        "num_negative": int((arr < -0.01).sum()),
        "num_near_zero": int((np.abs(arr) <= 0.01).sum()),
        "fraction_positive": float((arr > 0.01).mean()),
        "fraction_negative": float((arr < -0.01).mean()),
    }


def main():
    parser = argparse.ArgumentParser(description="Precompute f_pseudo reward shaping values")
    parser.add_argument(
        "--fnet_dir",
        type=str,
        default="outputs/fnet/gui360",
        help="Directory containing per-app f_net outputs (f_values.npz, f_net_final.pt)",
    )
    parser.add_argument(
        "--transitions_path",
        type=str,
        default="outputs/transitions/gui360_full/transitions.jsonl",
        help="Path to transitions.jsonl from Task 2",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="datasets/GUI-360/train/data",
        help="Raw GUI-360 trajectory data directory (fallback if transitions not available)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/f_pseudo",
        help="Output directory for f_pseudo_map.json and statistics.json",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device for f_net inference (cpu or cuda)",
    )
    args = parser.parse_args()

    # Resolve paths relative to project root
    fnet_dir = PROJECT_ROOT / args.fnet_dir
    transitions_path = PROJECT_ROOT / args.transitions_path
    data_dir = PROJECT_ROOT / args.data_dir
    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Load f-value maps from all apps
    logger.info("Loading f-value maps...")
    f_value_map = load_f_value_maps(fnet_dir)

    if not f_value_map:
        logger.error("No f-values loaded. Ensure f_net training (Task 3) has been completed.")
        sys.exit(1)

    # Step 2: Compute f_pseudo from transitions
    f_pseudo_map = {}
    all_stats = {}

    if transitions_path.exists():
        logger.info(f"Processing transitions from {transitions_path}...")
        f_pseudo_map, trans_stats = process_transitions(transitions_path, f_value_map)
        all_stats["transitions"] = trans_stats
        logger.info(
            f"From transitions.jsonl: {trans_stats['both_found']}/{trans_stats['total']} "
            f"steps covered ({trans_stats['both_found']/max(trans_stats['total'],1)*100:.1f}%)"
        )

    # Step 3: If coverage is low or transitions not available, process raw trajectories
    if data_dir.exists():
        logger.info(f"Processing raw trajectories from {data_dir}...")
        f_nets = load_f_nets(fnet_dir, device=args.device)
        raw_map, raw_stats = process_raw_trajectories(data_dir, f_value_map, f_nets, args.device)
        all_stats["raw_trajectories"] = raw_stats

        # Merge: raw trajectories fill gaps not covered by transitions
        for exec_id, steps in raw_map.items():
            if exec_id not in f_pseudo_map:
                f_pseudo_map[exec_id] = steps
            else:
                for step_id, val in steps.items():
                    if step_id not in f_pseudo_map[exec_id]:
                        f_pseudo_map[exec_id][step_id] = val

        logger.info(
            f"From raw trajectories: {raw_stats['both_found']}/{raw_stats['total_transitions']} "
            f"transitions covered, {raw_stats['inferred']} inferred via f_net"
        )

    # Step 4: Compute and save statistics
    distribution_stats = compute_statistics(f_pseudo_map)
    all_stats["distribution"] = distribution_stats

    logger.info(f"Final f_pseudo_map: {distribution_stats['num_trajectories']} trajectories, "
                f"{distribution_stats['num_steps']} steps")
    if distribution_stats["num_steps"] > 0:
        logger.info(
            f"f_pseudo distribution: mean={distribution_stats['mean']:.4f}, "
            f"std={distribution_stats['std']:.4f}, "
            f"range=[{distribution_stats['min']:.4f}, {distribution_stats['max']:.4f}]"
        )
        logger.info(
            f"Positive (bottleneck crossing): {distribution_stats['num_positive']} "
            f"({distribution_stats['fraction_positive']*100:.1f}%), "
            f"Negative: {distribution_stats['num_negative']} "
            f"({distribution_stats['fraction_negative']*100:.1f}%)"
        )

    # Step 5: Save outputs
    f_pseudo_path = output_dir / "f_pseudo_map.json"
    with open(f_pseudo_path, "w") as f:
        json.dump(f_pseudo_map, f)
    logger.info(f"Saved f_pseudo_map to {f_pseudo_path}")

    stats_path = output_dir / "statistics.json"
    with open(stats_path, "w") as f:
        json.dump(all_stats, f, indent=2)
    logger.info(f"Saved statistics to {stats_path}")


if __name__ == "__main__":
    main()
