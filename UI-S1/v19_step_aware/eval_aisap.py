#!/usr/bin/env python3
"""AISAP: Adaptive Inference-Time Scaling for Sequential Action Prediction.

Addresses the exponential TSR decay in long-horizon GUI navigation through:
1. Diverse Ensemble Sampling — multiple prompts × temperatures per step
2. Consensus Aggregation — cluster actions, vote, use centroid + cleaned history
3. Trajectory Repair — identify weak steps (low consensus), re-sample, cascade

Theoretical grounding:
- Self-consistency (Wang et al., 2023): sampling + voting > single pass
- Adaptive computation: spend more compute on hard steps (low consensus)
- Error containment: consensus history + repair breaks error cascading chain

Usage:
    # E1: Voting baseline (N=5 same prompt)
    python v19_step_aware/eval_aisap.py --mode voting --n_samples 5

    # E2: Diverse prompts
    python v19_step_aware/eval_aisap.py --mode voting --n_samples 5 --diverse_prompts

    # E3: E2 + consensus history (default when not gt_history)
    python v19_step_aware/eval_aisap.py --mode voting --n_samples 5 --diverse_prompts

    # E4: E3 + trajectory repair
    python v19_step_aware/eval_aisap.py --mode repair --n_samples 5 --diverse_prompts

    # E5: E4 + adaptive N
    python v19_step_aware/eval_aisap.py --mode adaptive --diverse_prompts
"""

import argparse
import json
import math
import os
import re
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from v13_gui_360.eval_gui360_template import (
    parse_tool_call, _format_action_for_history, SUPPORTED_ACTIONS,
    _encode_screenshot, USER_PROMPT_TEMPLATE,
)
from v13_gui_360.reward import compute_step_reward
from v19_step_aware.eval_step_aware import TYPE_FOCUSED_PROMPT


# ═══════════════════════════════════════════════════════════════════════
# Sampling: generate N diverse action candidates per step
# ═══════════════════════════════════════════════════════════════════════

# Agent configurations for diverse ensemble
AGENT_CONFIGS_DIVERSE = [
    {"prompt": "type_focused", "temperature": 0.7},
    {"prompt": "type_focused", "temperature": 0.7},
    {"prompt": "standard",     "temperature": 0.7},
    {"prompt": "standard",     "temperature": 0.7},
    {"prompt": "type_focused", "temperature": 0.0},  # greedy anchor
]

AGENT_CONFIGS_SINGLE = [
    {"prompt": "type_focused", "temperature": 0.7},
    {"prompt": "type_focused", "temperature": 0.7},
    {"prompt": "type_focused", "temperature": 0.7},
    {"prompt": "type_focused", "temperature": 0.7},
    {"prompt": "type_focused", "temperature": 0.0},  # greedy anchor
]

PROMPT_TEMPLATES = {
    "type_focused": TYPE_FOCUSED_PROMPT,
    "standard": USER_PROMPT_TEMPLATE,
}


def _build_prompt_text(prompt_name: str, goal: str, history_text: str) -> str:
    """Build prompt text from template name."""
    return PROMPT_TEMPLATES[prompt_name].format(
        instruction=goal,
        history=history_text,
        actions=SUPPORTED_ACTIONS,
    )


def sample_actions(
    client: OpenAI,
    model_name: str,
    b64_image: str,
    goal: str,
    history_text: str,
    agent_configs: List[Dict],
) -> List[Dict[str, Any]]:
    """Sample N actions from diverse agents. Returns list of parsed results."""
    samples = []
    for cfg in agent_configs:
        prompt_text = _build_prompt_text(cfg["prompt"], goal, history_text)
        messages = [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_image}"}},
            {"type": "text", "text": prompt_text},
        ]}]

        pred_text = ""
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                max_tokens=1024,
                temperature=cfg["temperature"],
            )
            pred_text = response.choices[0].message.content or ""
        except Exception:
            pass

        # Parse action
        pred_action = parse_tool_call(pred_text)
        if pred_action is None:
            m = re.search(r'<action>\s*(\{.*?\})\s*</action>', pred_text, re.DOTALL)
            if m:
                try:
                    pred_action = json.loads(m.group(1))
                except json.JSONDecodeError:
                    pass

        samples.append({
            "action": pred_action,
            "text": pred_text[:500],
            "prompt": cfg["prompt"],
            "temperature": cfg["temperature"],
        })

    return samples


# ═══════════════════════════════════════════════════════════════════════
# Consensus Aggregation: cluster actions, vote, compute centroid
# ═══════════════════════════════════════════════════════════════════════

def _action_type(action: Optional[Dict]) -> Optional[str]:
    """Extract normalized action type."""
    if action is None:
        return None
    return action.get("action", "unknown")


def _get_coord(action: Optional[Dict]) -> Optional[Tuple[float, float]]:
    """Extract (x, y) coordinate from action, or None."""
    if action is None:
        return None
    coord = action.get("coordinate")
    if coord and len(coord) >= 2 and coord[0] is not None and coord[1] is not None:
        try:
            return (float(coord[0]), float(coord[1]))
        except (TypeError, ValueError):
            return None
    return None


def _coord_distance(c1: Tuple[float, float], c2: Tuple[float, float]) -> float:
    """Euclidean distance between two coordinates."""
    return math.sqrt((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2)


def _cluster_actions(actions: List[Dict], merge_radius: float = 80.0) -> List[List[Dict]]:
    """Cluster actions by type, then merge nearby coordinates.

    Uses single-linkage clustering: two actions are in the same cluster if
    they have the same type and their coordinates are within merge_radius px.
    For 'type' actions, also requires text similarity.

    Args:
        actions: List of parsed action dicts.
        merge_radius: Max pixel distance to merge click/swipe coordinates.

    Returns:
        List of clusters (each cluster is a list of actions).
    """
    if not actions:
        return []

    # Group by action type first
    type_groups = defaultdict(list)
    for a in actions:
        atype = _action_type(a)
        type_groups[atype].append(a)

    all_clusters = []
    for atype, group in type_groups.items():
        if atype == "type":
            # For type actions: cluster by text similarity + coordinate proximity
            text_groups = defaultdict(list)
            for a in group:
                text_key = str(a.get("text", ""))[:30].lower().strip()
                text_groups[text_key].append(a)
            for tg in text_groups.values():
                all_clusters.append(tg)
        else:
            # For click/swipe: cluster by coordinate proximity
            remaining = list(group)
            while remaining:
                seed = remaining.pop(0)
                cluster = [seed]
                seed_coord = _get_coord(seed)
                still_remaining = []
                for a in remaining:
                    a_coord = _get_coord(a)
                    if seed_coord and a_coord and _coord_distance(seed_coord, a_coord) <= merge_radius:
                        cluster.append(a)
                    elif seed_coord is None and a_coord is None:
                        cluster.append(a)
                    else:
                        still_remaining.append(a)
                remaining = still_remaining
                all_clusters.append(cluster)

    return all_clusters


def aggregate_consensus(samples: List[Dict[str, Any]], merge_radius: float = 80.0) -> Tuple[Optional[Dict], float, Dict]:
    """Aggregate N samples into a consensus action.

    Uses distance-based clustering (merge_radius px) instead of fixed-grid
    bucketing to avoid boundary artifacts.

    Returns:
        (consensus_action, consensus_score, details)
        consensus_score = size of largest cluster / N
    """
    N = len(samples)
    if N == 0:
        return None, 0.0, {}

    # Filter to valid predictions
    valid_actions = [s["action"] for s in samples if s["action"] is not None]
    if not valid_actions:
        return None, 0.0, {"n_valid": 0, "n_total": N}

    # Cluster by type + coordinate proximity
    clusters = _cluster_actions(valid_actions, merge_radius)

    if not clusters:
        return None, 0.0, {"n_valid": len(valid_actions), "n_total": N, "n_clusters": 0}

    # Find largest cluster
    clusters.sort(key=len, reverse=True)
    best_cluster = clusters[0]
    consensus_score = len(best_cluster) / N

    # Compute centroid of winning cluster
    consensus_action = _compute_centroid(best_cluster)

    details = {
        "n_valid": len(valid_actions),
        "n_total": N,
        "n_clusters": len(clusters),
        "best_cluster_size": len(best_cluster),
        "cluster_sizes": [len(c) for c in clusters],
    }

    return consensus_action, consensus_score, details


def _compute_centroid(actions: List[Dict]) -> Dict:
    """Compute centroid action from a cluster of similar actions."""
    if not actions:
        return {}

    # Action type: majority vote
    type_counts = Counter(_action_type(a) for a in actions)
    best_type = type_counts.most_common(1)[0][0]

    result = {"action": best_type}

    if best_type in ("click", "long_press"):
        # Average coordinates
        coords = [a.get("coordinate") for a in actions if a.get("coordinate")]
        coords = [c for c in coords if c and len(c) >= 2 and c[0] is not None and c[1] is not None]
        if coords:
            avg_x = sum(float(c[0]) for c in coords) / len(coords)
            avg_y = sum(float(c[1]) for c in coords) / len(coords)
            result["coordinate"] = [int(round(avg_x)), int(round(avg_y))]

    elif best_type == "type":
        # Coordinate: average
        coords = [a.get("coordinate") for a in actions if a.get("coordinate")]
        coords = [c for c in coords if c and len(c) >= 2 and c[0] is not None and c[1] is not None]
        if coords:
            avg_x = sum(float(c[0]) for c in coords) / len(coords)
            avg_y = sum(float(c[1]) for c in coords) / len(coords)
            result["coordinate"] = [int(round(avg_x)), int(round(avg_y))]
        # Text: most common
        texts = [str(a.get("text", "")) for a in actions if a.get("text")]
        if texts:
            result["text"] = Counter(texts).most_common(1)[0][0]

    elif best_type in ("swipe", "drag"):
        # Average start/end coordinates
        starts = [a.get("coordinate") for a in actions if a.get("coordinate")]
        starts = [c for c in starts if c and len(c) >= 2 and c[0] is not None]
        ends = [a.get("endCoordinate") for a in actions if a.get("endCoordinate")]
        ends = [c for c in ends if c and len(c) >= 2 and c[0] is not None]
        if starts:
            result["coordinate"] = [
                int(round(sum(float(c[0]) for c in starts) / len(starts))),
                int(round(sum(float(c[1]) for c in starts) / len(starts))),
            ]
        if ends:
            result["endCoordinate"] = [
                int(round(sum(float(c[0]) for c in ends) / len(ends))),
                int(round(sum(float(c[1]) for c in ends) / len(ends))),
            ]

    return result


# ═══════════════════════════════════════════════════════════════════════
# Trajectory Repair: identify weak steps, re-sample, cascade
# ═══════════════════════════════════════════════════════════════════════

def repair_trajectory(
    client: OpenAI,
    model_name: str,
    episode: Dict,
    initial_results: List[Dict],
    consensus_threshold: float = 0.4,
    repair_n_samples: int = 10,
    max_repair_rounds: int = 2,
    diverse_prompts: bool = True,
    gt_history: bool = False,
    image_max_pixels: Optional[int] = None,
) -> List[Dict]:
    """Repair weak steps in a trajectory and cascade re-prediction.

    Finds steps with consensus_score < threshold, re-samples with more
    diversity, and re-predicts all subsequent steps with updated history.

    Returns:
        Updated step results list.
    """
    steps_data = episode["steps"]
    goal = episode["goal"]
    results = list(initial_results)  # copy

    for repair_round in range(max_repair_rounds):
        # Find weak steps (low consensus, not already repaired to high confidence)
        weak_indices = []
        for i, r in enumerate(results):
            if r.get("consensus_score", 1.0) < consensus_threshold:
                weak_indices.append(i)

        if not weak_indices:
            break  # No weak steps, trajectory is stable

        # Repair from the earliest weak step (to minimize cascading)
        repair_from = min(weak_indices)

        # Build history up to repair_from
        history = []
        for j in range(repair_from):
            if gt_history:
                history.append(_format_action_for_history(steps_data[j]["action"], j + 1))
            else:
                history.append(_format_action_for_history(results[j].get("consensus_action"), j + 1))

        # Build repair agent configs (more samples, more diversity)
        repair_configs = _build_agent_configs(repair_n_samples, diverse_prompts)

        # Re-predict from repair_from onwards
        for i in range(repair_from, len(steps_data)):
            step = steps_data[i]
            screenshot = step["screenshot"]
            gt_action = step["action"]
            image_w = step.get("image_w", 1040)
            image_h = step.get("image_h", 736)

            history_text = "\n".join(history) if history else "None"
            b64 = _encode_screenshot(screenshot, image_max_pixels)

            # Use more samples for the actual weak step, normal for cascade
            if i in weak_indices:
                configs = repair_configs
            else:
                configs = _build_agent_configs(5, diverse_prompts)

            samples = sample_actions(client, model_name, b64, goal, history_text, configs)
            consensus_action, consensus_score, agg_details = aggregate_consensus(samples)

            # Score against GT
            if consensus_action:
                fake_text = f"<action>{json.dumps(consensus_action)}</action>"
            else:
                fake_text = ""
            reward, info = compute_step_reward(fake_text, gt_action, image_w, image_h)

            results[i] = {
                "step_idx": i,
                "success": reward >= 0.5,
                "reward": reward,
                "consensus_action": consensus_action,
                "consensus_score": consensus_score,
                "n_samples": len(configs),
                "agg_details": agg_details,
                "pred_type": info.get("pred_type"),
                "gt_type": info.get("gt_type"),
                "type_reward": info.get("type_reward", 0),
                "content_reward": info.get("content_reward", 0),
                "repaired": True,
                "repair_round": repair_round,
            }

            # Update history
            if gt_history:
                history.append(_format_action_for_history(gt_action, i + 1))
            else:
                history.append(_format_action_for_history(consensus_action, i + 1))

    return results


# ═══════════════════════════════════════════════════════════════════════
# Adaptive N: more samples for harder steps
# ═══════════════════════════════════════════════════════════════════════

def adaptive_n_for_step(step_idx: int, num_steps: int, prev_consensus: Optional[float]) -> int:
    """Decide how many samples to use for this step.

    Heuristic:
    - Step 0: N=5 (always, no history signal yet)
    - If previous consensus was high (>0.6): N=3 (easy context)
    - If previous consensus was low (<0.4): N=7 (hard context, need more diversity)
    - Default: N=5
    - Late steps (>60% through task): +2 (error cascading risk)
    """
    base_n = 5

    if step_idx == 0:
        return base_n

    # Adjust based on previous step consensus
    if prev_consensus is not None:
        if prev_consensus >= 0.6:
            base_n = 3
        elif prev_consensus < 0.4:
            base_n = 7

    # Late-step bonus
    if num_steps > 3 and step_idx / num_steps > 0.6:
        base_n += 2

    return base_n


# ═══════════════════════════════════════════════════════════════════════
# Helper: build agent configs for N samples
# ═══════════════════════════════════════════════════════════════════════

def _build_agent_configs(n_samples: int, diverse_prompts: bool) -> List[Dict]:
    """Build agent configurations for N samples."""
    if diverse_prompts:
        # Pattern: alternate TF/std, last one is greedy anchor
        configs = []
        for i in range(n_samples - 1):
            prompt = "type_focused" if i % 2 == 0 else "standard"
            configs.append({"prompt": prompt, "temperature": 0.7})
        configs.append({"prompt": "type_focused", "temperature": 0.0})
        return configs
    else:
        configs = [{"prompt": "type_focused", "temperature": 0.7} for _ in range(n_samples - 1)]
        configs.append({"prompt": "type_focused", "temperature": 0.0})
        return configs


# ═══════════════════════════════════════════════════════════════════════
# Episode Evaluation
# ═══════════════════════════════════════════════════════════════════════

def evaluate_episode_aisap(
    client: OpenAI,
    model_name: str,
    episode: Dict,
    mode: str = "voting",
    n_samples: int = 5,
    diverse_prompts: bool = True,
    gt_history: bool = False,
    match_threshold: float = 0.5,
    consensus_threshold: float = 0.4,
    repair_n_samples: int = 10,
    max_repair_rounds: int = 2,
    image_max_pixels: Optional[int] = None,
) -> Dict[str, Any]:
    """Evaluate one episode with the AISAP framework.

    Modes:
        voting:   Phase 1+2 only (diverse sampling + consensus aggregation)
        repair:   Phase 1+2+3 (+ trajectory repair)
        adaptive: Phase 1+2+3 with adaptive N per step
    """
    episode_id = episode["episode_id"]
    goal = episode["goal"]
    steps = episode["steps"]
    num_steps = len(steps)

    history = []
    step_results = []
    prev_consensus = None

    # Phase 1+2: Forward pass with consensus
    for i, step in enumerate(steps):
        gt_action = step["action"]
        screenshot = step["screenshot"]
        image_w = step.get("image_w", 1040)
        image_h = step.get("image_h", 736)

        # Determine N for this step
        if mode == "adaptive":
            step_n = adaptive_n_for_step(i, num_steps, prev_consensus)
        else:
            step_n = n_samples

        history_text = "\n".join(history) if history else "None"
        b64 = _encode_screenshot(screenshot, image_max_pixels)

        # Sample N actions
        configs = _build_agent_configs(step_n, diverse_prompts)
        samples = sample_actions(client, model_name, b64, goal, history_text, configs)

        # Aggregate consensus
        consensus_action, consensus_score, agg_details = aggregate_consensus(samples)
        prev_consensus = consensus_score

        # Score against GT
        if consensus_action:
            fake_text = f"<action>{json.dumps(consensus_action)}</action>"
        else:
            fake_text = ""
        reward, info = compute_step_reward(fake_text, gt_action, image_w, image_h)
        success = reward >= match_threshold

        step_results.append({
            "step_idx": i,
            "success": success,
            "reward": reward,
            "consensus_action": consensus_action,
            "consensus_score": consensus_score,
            "n_samples": step_n,
            "agg_details": agg_details,
            "pred_type": info.get("pred_type"),
            "gt_type": info.get("gt_type"),
            "type_reward": info.get("type_reward", 0),
            "content_reward": info.get("content_reward", 0),
            "repaired": False,
        })

        # Update history with consensus action (cleaned history)
        if gt_history:
            history.append(_format_action_for_history(gt_action, i + 1))
        else:
            history.append(_format_action_for_history(consensus_action, i + 1))

    # Phase 3: Trajectory repair (if enabled)
    if mode in ("repair", "adaptive"):
        step_results = repair_trajectory(
            client, model_name, episode, step_results,
            consensus_threshold=consensus_threshold,
            repair_n_samples=repair_n_samples,
            max_repair_rounds=max_repair_rounds,
            diverse_prompts=diverse_prompts,
            gt_history=gt_history,
            image_max_pixels=image_max_pixels,
        )

    # Compute trajectory metrics
    first_error_step = None
    correct_steps = 0
    total_samples_used = 0
    for r in step_results:
        total_samples_used += r.get("n_samples", n_samples)
        if r["success"]:
            correct_steps += 1
        elif first_error_step is None:
            first_error_step = r["step_idx"] + 1

    progress = (first_error_step - 1) / num_steps if first_error_step else 1.0
    task_success = first_error_step is None and len(step_results) == num_steps
    avg_consensus = sum(r.get("consensus_score", 0) for r in step_results) / len(step_results) if step_results else 0

    return {
        "episode_id": episode_id,
        "goal": goal,
        "num_steps": num_steps,
        "steps_evaluated": len(step_results),
        "correct_steps": correct_steps,
        "task_success": task_success,
        "progress": progress,
        "first_error_step": first_error_step,
        "avg_consensus": avg_consensus,
        "total_samples_used": total_samples_used,
        "steps": step_results,
    }


# ═══════════════════════════════════════════════════════════════════════
# Breakdown Analysis
# ═══════════════════════════════════════════════════════════════════════

def compute_breakdown(results: Dict[str, Dict]) -> Dict[str, Any]:
    """Compute per-step-position and per-task-length metrics."""
    step_pos_correct = defaultdict(int)
    step_pos_total = defaultdict(int)
    step_pos_consensus = defaultdict(list)

    length_buckets = {"1": (1, 1), "2-3": (2, 3), "4-5": (4, 5), "6+": (6, 999)}
    length_success = defaultdict(int)
    length_total = defaultdict(int)
    length_progress = defaultdict(float)
    length_steps_correct = defaultdict(int)
    length_steps_total = defaultdict(int)

    # Consensus vs correctness correlation
    consensus_correct = []
    consensus_wrong = []

    for eid, result in results.items():
        num_steps = result["num_steps"]
        bucket = "6+"
        for bname, (lo, hi) in length_buckets.items():
            if lo <= num_steps <= hi:
                bucket = bname
                break

        length_total[bucket] += 1
        if result["task_success"]:
            length_success[bucket] += 1
        length_progress[bucket] += result["progress"]

        for step in result["steps"]:
            idx = step["step_idx"]
            step_pos_total[idx] += 1
            cs = step.get("consensus_score", 0)
            step_pos_consensus[idx].append(cs)
            if step["success"]:
                step_pos_correct[idx] += 1
                consensus_correct.append(cs)
            else:
                consensus_wrong.append(cs)
            length_steps_total[bucket] += 1
            if step["success"]:
                length_steps_correct[bucket] += 1

    # Format per-step-position
    max_step = max(step_pos_total.keys()) if step_pos_total else 0
    step_position_acc = {}
    for idx in range(min(max_step + 1, 15)):
        total = step_pos_total.get(idx, 0)
        correct = step_pos_correct.get(idx, 0)
        cs_list = step_pos_consensus.get(idx, [])
        if total > 0:
            step_position_acc[f"step_{idx}"] = {
                "accuracy": correct / total,
                "avg_consensus": sum(cs_list) / len(cs_list) if cs_list else 0,
                "total": total,
                "correct": correct,
            }

    # Format per-task-length
    task_length_metrics = {}
    for bname in length_buckets:
        total = length_total.get(bname, 0)
        if total > 0:
            s_total = length_steps_total.get(bname, 0)
            task_length_metrics[bname] = {
                "tsr": length_success[bname] / total,
                "avg_progress": length_progress[bname] / total,
                "step_sr": length_steps_correct[bname] / s_total if s_total > 0 else 0,
                "num_episodes": total,
                "num_success": length_success[bname],
            }

    # Consensus as difficulty predictor
    consensus_analysis = {}
    if consensus_correct and consensus_wrong:
        consensus_analysis = {
            "avg_consensus_correct": sum(consensus_correct) / len(consensus_correct),
            "avg_consensus_wrong": sum(consensus_wrong) / len(consensus_wrong),
            "n_correct": len(consensus_correct),
            "n_wrong": len(consensus_wrong),
        }

    return {
        "step_position_accuracy": step_position_acc,
        "task_length_metrics": task_length_metrics,
        "consensus_analysis": consensus_analysis,
    }


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="AISAP: Adaptive Inference-Time Scaling")
    parser.add_argument("--mode", required=True, choices=["voting", "repair", "adaptive"],
                        help="AISAP mode: voting=E1/E2/E3, repair=E4, adaptive=E5")
    parser.add_argument("--n_samples", type=int, default=5,
                        help="Base number of samples per step (for voting/repair modes)")
    parser.add_argument("--diverse_prompts", action="store_true", default=False,
                        help="Use diverse prompt variants (type_focused + standard)")
    parser.add_argument("--consensus_threshold", type=float, default=0.4,
                        help="Steps with consensus below this trigger repair")
    parser.add_argument("--repair_n_samples", type=int, default=10,
                        help="Number of samples for repair re-prediction")
    parser.add_argument("--max_repair_rounds", type=int, default=2,
                        help="Maximum repair iterations")
    parser.add_argument("--test_data", required=True)
    parser.add_argument("--api_url", default="http://localhost:8000/v1")
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--threads", type=int, default=32)
    parser.add_argument("--match_threshold", type=float, default=0.5)
    parser.add_argument("--gt_history", action="store_true", default=False)
    parser.add_argument("--image_max_pixels", type=int, default=None)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    args = parser.parse_args()

    episodes = []
    with open(args.test_data) as f:
        for line in f:
            episodes.append(json.loads(line))
    total_loaded = len(episodes)
    episodes = episodes[args.start:args.end]
    print(f"Loaded {len(episodes)} episodes (shard [{args.start}:{args.end}] of {total_loaded})")
    print(f"Mode: {args.mode} | N={args.n_samples} | Diverse={args.diverse_prompts}")
    print(f"GT history: {args.gt_history}")
    if args.mode in ("repair", "adaptive"):
        print(f"Repair: threshold={args.consensus_threshold} N_repair={args.repair_n_samples} max_rounds={args.max_repair_rounds}")

    client = OpenAI(base_url=args.api_url, api_key="dummy")
    os.makedirs(args.output_dir, exist_ok=True)

    results = {}
    total_success = 0
    total_progress = 0.0
    total_steps = 0
    total_correct = 0
    total_samples = 0

    with ThreadPoolExecutor(max_workers=args.threads) as executor:
        futures = {
            executor.submit(
                evaluate_episode_aisap, client, args.model_name, ep,
                args.mode, args.n_samples, args.diverse_prompts,
                args.gt_history, args.match_threshold,
                args.consensus_threshold, args.repair_n_samples,
                args.max_repair_rounds, args.image_max_pixels,
            ): ep["episode_id"]
            for ep in episodes
        }

        pbar = tqdm(total=len(episodes), desc=f"AISAP {args.mode}")
        for future in as_completed(futures):
            result = future.result()
            eid = result["episode_id"]
            results[eid] = result

            if result["task_success"]:
                total_success += 1
            total_progress += result["progress"]
            total_steps += result["steps_evaluated"]
            total_correct += result["correct_steps"]
            total_samples += result["total_samples_used"]

            n = len(results)
            pbar.update(1)
            pbar.set_postfix({
                "TSR": f"{total_success/n:.3f}",
                "StepSR": f"{total_correct/total_steps:.3f}" if total_steps > 0 else "0",
            })
        pbar.close()

    # Compute breakdowns
    breakdown = compute_breakdown(results)

    n = len(results)
    avg_samples = total_samples / total_steps if total_steps > 0 else 0
    summary = {
        "num_episodes": n,
        "tsr": total_success / n if n > 0 else 0,
        "avg_progress": total_progress / n if n > 0 else 0,
        "step_sr": total_correct / total_steps if total_steps > 0 else 0,
        "total_steps_evaluated": total_steps,
        "total_steps_correct": total_correct,
        "total_inference_calls": total_samples,
        "avg_samples_per_step": avg_samples,
        "mode": args.mode,
        "n_samples": args.n_samples,
        "diverse_prompts": args.diverse_prompts,
        "gt_history": args.gt_history,
        "match_threshold": args.match_threshold,
        **breakdown,
    }

    ts = time.strftime("%Y%m%d_%H%M%S")
    with open(os.path.join(args.output_dir, f"eval_summary_{ts}.json"), "w") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(args.output_dir, f"eval_results_{ts}.json"), "w") as f:
        json.dump(results, f, indent=2)

    # Print results
    print(f"\n{'='*65}")
    print(f"AISAP Results: mode={args.mode} N={args.n_samples} diverse={args.diverse_prompts}")
    print(f"{'='*65}")
    print(f"  TSR:            {summary['tsr']*100:.1f}%")
    print(f"  Step SR:        {summary['step_sr']*100:.1f}%")
    print(f"  Progress:       {summary['avg_progress']*100:.1f}%")
    print(f"  Avg samples/step: {avg_samples:.1f}")
    print(f"  Total inference:  {total_samples}")
    hist_mode = "GT history" if args.gt_history else "Pred history"
    print(f"  Mode: {hist_mode}")
    print(f"  (Baselines: single-pass=21.9%, type_focused=23.6%)")

    # Consensus analysis
    ca = summary.get("consensus_analysis", {})
    if ca:
        print(f"\n  --- Consensus as Difficulty Predictor ---")
        print(f"  Correct steps avg consensus: {ca.get('avg_consensus_correct', 0)*100:.1f}%")
        print(f"  Wrong steps avg consensus:   {ca.get('avg_consensus_wrong', 0)*100:.1f}%")

    print(f"\n  --- By Task Length ---")
    for bname, metrics in sorted(summary.get("task_length_metrics", {}).items()):
        print(f"  {bname:>5s} steps: TSR={metrics['tsr']*100:5.1f}%  "
              f"StepSR={metrics['step_sr']*100:5.1f}%  "
              f"Progress={metrics['avg_progress']*100:5.1f}%  "
              f"(n={metrics['num_episodes']})")

    print(f"\n  --- By Step Position ---")
    for sname, metrics in sorted(summary.get("step_position_accuracy", {}).items(),
                                  key=lambda x: int(x[0].split("_")[1])):
        print(f"  {sname:>8s}: acc={metrics['accuracy']*100:5.1f}%  "
              f"consensus={metrics['avg_consensus']*100:5.1f}%  "
              f"(n={metrics['total']})")

    print(f"{'='*65}")


if __name__ == "__main__":
    main()
