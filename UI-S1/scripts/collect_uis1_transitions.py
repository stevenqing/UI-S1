#!/usr/bin/env python3
"""
Collect state transitions from UI-S1 (AndroidControl) trajectories and
analyze graph connectivity to determine if f_pseudo is feasible.

Unlike GUI-360 which has accessibility trees (43-dim embeddings), UI-S1
only has screenshots + action coordinates. Each screenshot is essentially
a unique state, so the transition graph is expected to be very sparse.

This script answers the key question:
  "Is the UI-S1 state transition graph connected enough for
   eigenfunction-based f_pseudo to work?"

Usage:
    # Full analysis:
    python scripts/collect_uis1_transitions.py

    # Quick test:
    python scripts/collect_uis1_transitions.py --max-episodes 100

    # Custom output:
    python scripts/collect_uis1_transitions.py --output-dir outputs/transitions/uis1_test

Outputs (in --output-dir):
    transition_pairs.json     — [{src_hash, dst_hash, execution_id, step_id}, ...]
    state_manifest.json       — {hash: {screenshot_path, goal, step_id, action_type}}
    adjacency.json            — {src_hash: [dst_hash, ...]}
    statistics.json           — Graph connectivity analysis
    connectivity_report.txt   — Human-readable feasibility report
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from collections import Counter, defaultdict
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_DATA_PATH = PROJECT_ROOT / "datasets" / "ui_s1_dataset" / "ui_s1_train.jsonl"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "transitions" / "uis1"


def hash_screenshot(path: str) -> str:
    """Use screenshot path as unique state identifier."""
    return hashlib.md5(path.encode()).hexdigest()[:16]


def collect_transitions(
    data_path: Path,
    output_dir: Path,
    max_episodes: int | None = None,
) -> dict:
    """Parse UI-S1 JSONL and extract state transitions."""
    output_dir.mkdir(parents=True, exist_ok=True)

    transition_pairs = []
    state_manifest = {}
    adjacency = defaultdict(set)

    # Per-episode stats
    episode_lengths = []
    action_counts = Counter()
    app_counts = Counter()  # approximate app from goal/open actions

    with open(data_path) as f:
        for line_idx, line in enumerate(f):
            if max_episodes and line_idx >= max_episodes:
                break

            episode = json.loads(line)
            goal = episode.get("goal", "")
            steps = episode.get("steps", [])
            execution_id = str(line_idx)

            episode_lengths.append(len(steps))

            # Track app from "open" actions
            for step in steps:
                ac = step.get("action_content", {})
                action_type = ac.get("action", "")
                action_counts[action_type] += 1
                if action_type == "open" and ac.get("text"):
                    app_counts[ac["text"]] += 1

            # Build transitions from consecutive steps
            for i in range(len(steps) - 1):
                step_cur = steps[i]
                step_nxt = steps[i + 1]

                screenshot_cur = step_cur.get("screenshot", "")
                screenshot_nxt = step_nxt.get("screenshot", "")

                if not screenshot_cur or not screenshot_nxt:
                    continue

                src_hash = hash_screenshot(screenshot_cur)
                dst_hash = hash_screenshot(screenshot_nxt)

                # Record transition
                transition_pairs.append({
                    "src_hash": src_hash,
                    "dst_hash": dst_hash,
                    "execution_id": execution_id,
                    "step_id": i,
                })

                adjacency[src_hash].add(dst_hash)

                # Record state info
                ac_cur = step_cur.get("action_content", {})
                if src_hash not in state_manifest:
                    state_manifest[src_hash] = {
                        "screenshot_path": screenshot_cur,
                        "goal": goal,
                        "step_id": i,
                        "action_type": ac_cur.get("action", ""),
                        "execution_id": execution_id,
                    }
                if dst_hash not in state_manifest:
                    ac_nxt = step_nxt.get("action_content", {})
                    state_manifest[dst_hash] = {
                        "screenshot_path": screenshot_nxt,
                        "goal": goal,
                        "step_id": i + 1,
                        "action_type": ac_nxt.get("action", ""),
                        "execution_id": execution_id,
                    }

    # ================================================================
    # Connectivity analysis
    # ================================================================
    num_states = len(state_manifest)
    num_transitions = len(transition_pairs)
    num_edges = sum(len(dsts) for dsts in adjacency.values())

    # Build undirected graph for connected components
    undirected = defaultdict(set)
    for src, dsts in adjacency.items():
        for dst in dsts:
            undirected[src].add(dst)
            undirected[dst].add(src)

    # Add isolated states (states that only appear as dst, never as src)
    all_states = set(state_manifest.keys())
    for s in all_states:
        if s not in undirected:
            undirected[s] = set()

    # BFS to find connected components
    visited = set()
    components = []
    for node in all_states:
        if node in visited:
            continue
        component = set()
        queue = [node]
        while queue:
            n = queue.pop()
            if n in visited:
                continue
            visited.add(n)
            component.add(n)
            for neighbor in undirected.get(n, set()):
                if neighbor not in visited:
                    queue.append(neighbor)
        components.append(component)

    components.sort(key=len, reverse=True)
    component_sizes = [len(c) for c in components]

    # Degree distribution
    degrees = [len(undirected.get(s, set())) for s in all_states]
    import numpy as np
    degrees_arr = np.array(degrees) if degrees else np.array([0])

    # ================================================================
    # Statistics
    # ================================================================
    stats = {
        "data_path": str(data_path),
        "num_episodes": len(episode_lengths),
        "num_states": num_states,
        "num_transitions": num_transitions,
        "num_unique_edges": num_edges,
        "graph_density": num_edges / max(num_states * (num_states - 1), 1),

        # Connectivity
        "num_connected_components": len(components),
        "largest_component_size": component_sizes[0] if component_sizes else 0,
        "largest_component_fraction": component_sizes[0] / max(num_states, 1) if component_sizes else 0,
        "top_5_component_sizes": component_sizes[:5],
        "num_isolated_states": sum(1 for d in degrees if d == 0),
        "num_singleton_components": sum(1 for s in component_sizes if s == 1),
        "num_pair_components": sum(1 for s in component_sizes if s == 2),

        # Degree distribution
        "avg_degree": float(degrees_arr.mean()),
        "median_degree": float(np.median(degrees_arr)),
        "max_degree": int(degrees_arr.max()),
        "min_degree": int(degrees_arr.min()),
        "degree_percentiles": {
            "p10": float(np.percentile(degrees_arr, 10)),
            "p25": float(np.percentile(degrees_arr, 25)),
            "p50": float(np.percentile(degrees_arr, 50)),
            "p75": float(np.percentile(degrees_arr, 75)),
            "p90": float(np.percentile(degrees_arr, 90)),
        },

        # Episode stats
        "avg_episode_length": float(np.mean(episode_lengths)),
        "median_episode_length": float(np.median(episode_lengths)),
        "max_episode_length": int(np.max(episode_lengths)),
        "min_episode_length": int(np.min(episode_lengths)),

        # Action distribution
        "action_distribution": dict(action_counts.most_common()),

        # App distribution (from "open" actions)
        "top_apps": dict(app_counts.most_common(20)),
        "num_unique_apps": len(app_counts),
    }

    # ================================================================
    # Feasibility assessment
    # ================================================================
    feasibility_lines = []
    feasibility_lines.append("=" * 60)
    feasibility_lines.append("  UI-S1 f_pseudo Feasibility Report")
    feasibility_lines.append("=" * 60)
    feasibility_lines.append("")
    feasibility_lines.append(f"  Episodes:           {stats['num_episodes']:,}")
    feasibility_lines.append(f"  Unique states:      {stats['num_states']:,}")
    feasibility_lines.append(f"  Transitions:        {stats['num_transitions']:,}")
    feasibility_lines.append(f"  Unique edges:       {stats['num_unique_edges']:,}")
    feasibility_lines.append(f"  Graph density:      {stats['graph_density']:.8f}")
    feasibility_lines.append(f"  Avg degree:         {stats['avg_degree']:.2f}")
    feasibility_lines.append("")
    feasibility_lines.append("  Connectivity:")
    feasibility_lines.append(f"    Connected components:     {stats['num_connected_components']:,}")
    feasibility_lines.append(f"    Largest component:        {stats['largest_component_size']:,} states "
                             f"({stats['largest_component_fraction']*100:.1f}%)")
    feasibility_lines.append(f"    Singleton components:     {stats['num_singleton_components']:,}")
    feasibility_lines.append(f"    Pair components:          {stats['num_pair_components']:,}")
    feasibility_lines.append(f"    Top 5 component sizes:    {stats['top_5_component_sizes']}")
    feasibility_lines.append("")

    # Comparison with GUI-360
    feasibility_lines.append("  Comparison with GUI-360:")
    feasibility_lines.append(f"    {'':20s} {'GUI-360':>12s}  {'UI-S1':>12s}")
    feasibility_lines.append(f"    {'Unique states':20s} {'6,505':>12s}  {stats['num_states']:>12,}")
    feasibility_lines.append(f"    {'Transitions':20s} {'91,618':>12s}  {stats['num_transitions']:>12,}")
    feasibility_lines.append(f"    {'Avg degree':20s} {'~28':>12s}  {stats['avg_degree']:>12.2f}")
    feasibility_lines.append(f"    {'Components':20s} {'3 (per-app)':>12s}  {stats['num_connected_components']:>12,}")
    feasibility_lines.append("")

    # Verdict
    feasibility_lines.append("  VERDICT:")
    if stats['num_connected_components'] <= 5 and stats['largest_component_fraction'] > 0.5:
        feasibility_lines.append("    FEASIBLE - Graph is well-connected, eigenfunction should work.")
        feasibility_lines.append("    Proceed with VLM f_net training.")
        stats["feasibility"] = "FEASIBLE"
    elif stats['largest_component_fraction'] > 0.3:
        feasibility_lines.append("    MARGINAL - Largest component has >30% states.")
        feasibility_lines.append("    Could work on largest component only, or try coarse-grained states.")
        stats["feasibility"] = "MARGINAL"
    else:
        feasibility_lines.append("    NOT FEASIBLE - Graph is too fragmented.")
        feasibility_lines.append("    Each episode is essentially isolated (no shared states).")
        feasibility_lines.append("    Eigenfunction will degenerate to trivial/constant solution.")
        feasibility_lines.append("")
        feasibility_lines.append("    Alternatives:")
        feasibility_lines.append("      1. Use per-app grouping to create denser sub-graphs")
        feasibility_lines.append("      2. Use VLM embedding similarity to merge 'similar' states")
        feasibility_lines.append("      3. Use action-type-based heuristic reward instead of spectral f_pseudo")
        feasibility_lines.append("      4. Skip f_pseudo, rely on naive reward + MoE exploration")
        stats["feasibility"] = "NOT_FEASIBLE"

    feasibility_lines.append("")
    feasibility_lines.append("=" * 60)

    report = "\n".join(feasibility_lines)

    # ================================================================
    # Save outputs
    # ================================================================

    # 1. Transition pairs
    with open(output_dir / "transition_pairs.json", "w") as f:
        json.dump(transition_pairs, f)
    logger.info(f"Saved {len(transition_pairs)} transition pairs")

    # 2. State manifest
    with open(output_dir / "state_manifest.json", "w") as f:
        json.dump(state_manifest, f, indent=2)
    logger.info(f"Saved state manifest ({len(state_manifest)} states)")

    # 3. Adjacency (convert sets to lists for JSON)
    adj_json = {k: list(v) for k, v in adjacency.items()}
    with open(output_dir / "adjacency.json", "w") as f:
        json.dump(adj_json, f)

    # 4. Statistics
    with open(output_dir / "statistics.json", "w") as f:
        json.dump(stats, f, indent=2, default=lambda o: int(o) if hasattr(o, 'item') else o)

    # 5. Connectivity report
    with open(output_dir / "connectivity_report.txt", "w") as f:
        f.write(report)

    # Print report
    print(report)

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Collect UI-S1 transitions and analyze graph connectivity for f_pseudo feasibility",
    )
    parser.add_argument(
        "--data-path", type=Path, default=DEFAULT_DATA_PATH,
        help=f"Path to UI-S1 training JSONL (default: {DEFAULT_DATA_PATH})",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--max-episodes", type=int, default=None,
        help="Limit number of episodes to process (for testing)",
    )
    args = parser.parse_args()

    if not args.data_path.exists():
        logger.error(f"Data file not found: {args.data_path}")
        sys.exit(1)

    collect_transitions(
        data_path=args.data_path,
        output_dir=args.output_dir,
        max_episodes=args.max_episodes,
    )


if __name__ == "__main__":
    main()
