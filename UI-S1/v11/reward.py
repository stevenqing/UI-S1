#!/usr/bin/env python3
"""v11 Dense Multi-Component Reward for Trajectory GRPO.

Provides continuous per-step rewards with three components:
  - Format reward (0.1): Valid action format parsing
  - Type reward   (0.2): Action type match
  - Content reward(0.7): Continuous coordinate distance / text similarity / direction match

Total step reward = w_format * format + w_type * type + w_content * content
Range: [0, 1]

Dense continuous rewards prevent advantage collapse by providing gradient signal
even for partial matches (vs v10's discrete 0/1 which caused ~50% zero advantages).
"""

import json
import re
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ── Action parsing ──────────────────────────────────────────────────

def parse_action_from_text(text: str) -> Optional[Dict[str, Any]]:
    """Parse action dict from model output text.

    Handles formats:
      <action>{"action":"click","coordinate":[x,y]}</action>
      {"action":"click","coordinate":[x,y]}
    """
    # Try <action>...</action> tag
    m = re.search(r'<action>\s*(\{.*?\})\s*</action>', text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass

    # Try raw JSON
    m = re.search(r'\{[^{}]*"action"[^{}]*\}', text)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass

    return None


# ── Base reward components ───────────��──────────────────────────────

def _coord_distance(pred_coord, gt_coord, image_w: int, image_h: int) -> float:
    """Normalized Euclidean distance between two coordinates."""
    if pred_coord is None or gt_coord is None:
        return float("inf")
    try:
        dx = (float(pred_coord[0]) - float(gt_coord[0])) / image_w
        dy = (float(pred_coord[1]) - float(gt_coord[1])) / image_h
        return (dx ** 2 + dy ** 2) ** 0.5
    except (TypeError, IndexError, ValueError):
        return float("inf")


def _coord_reward_continuous(dist: float, threshold: float = 0.05) -> float:
    """Continuous coordinate reward: 1.0 at dist=0, linearly decays to 0 at 4*threshold."""
    max_dist = threshold * 4
    if dist >= max_dist:
        return 0.0
    return 1.0 - dist / max_dist


def _text_similarity(pred: str, gt: str) -> float:
    """Character-level similarity between two strings."""
    if not pred and not gt:
        return 1.0
    if not pred or not gt:
        return 0.0
    return SequenceMatcher(None, pred.lower(), gt.lower()).ratio()


def _direction_similarity(pred_start, pred_end, gt_start, gt_end) -> float:
    """Cosine similarity between swipe direction vectors. Returns [0, 1]."""
    try:
        gt_dx = float(gt_end[0]) - float(gt_start[0])
        gt_dy = float(gt_end[1]) - float(gt_start[1])
        pred_dx = float(pred_end[0]) - float(pred_start[0])
        pred_dy = float(pred_end[1]) - float(pred_start[1])
        gt_mag = (gt_dx ** 2 + gt_dy ** 2) ** 0.5
        pred_mag = (pred_dx ** 2 + pred_dy ** 2) ** 0.5
        if gt_mag > 0 and pred_mag > 0:
            cos_sim = (gt_dx * pred_dx + gt_dy * pred_dy) / (gt_mag * pred_mag)
            cos_sim = max(-1.0, min(1.0, cos_sim))
            return max(0.0, cos_sim)  # map [-1, 1] → [0, 1], negative = 0
    except (TypeError, IndexError, ValueError):
        pass
    return 0.0


# ── Normalize action types ─────────────────────────────────────────

_ACTION_TYPE_ALIASES = {
    "left_click": "click",
    "tap": "click",
    "input": "type",
    "open_app": "open",
    "back": "system_button",
    "home": "system_button",
    "press": "long_press",
    "scroll": "swipe",
}

VALID_ACTION_TYPES = {
    "click", "long_press", "type", "swipe", "open", "wait",
    "system_button", "terminate", "answer", "key",
}


def _normalize_action_type(atype: str) -> str:
    """Normalize action type aliases to canonical form."""
    atype = atype.strip().lower()
    return _ACTION_TYPE_ALIASES.get(atype, atype)


# ── Dense multi-component reward ───────────────────────────────────

def compute_step_reward(
    pred_text: str,
    gt_action: Dict[str, Any],
    image_w: int = 1080,
    image_h: int = 2400,
    coord_threshold: float = 0.05,
    w_format: float = 0.1,
    w_type: float = 0.2,
    w_content: float = 0.7,
) -> Tuple[float, Dict[str, Any]]:
    """Compute dense multi-component reward for a single step.

    Args:
        pred_text: Model output text (may contain <action>...</action> or raw JSON)
        gt_action: Ground truth action dict
        image_w, image_h: Image dimensions for coordinate normalization
        coord_threshold: Distance threshold for coordinate reward
        w_format, w_type, w_content: Component weights (should sum to 1.0)

    Returns:
        (total_reward, info_dict) where total_reward is in [0, 1]
    """
    info = {
        "format_reward": 0.0,
        "type_reward": 0.0,
        "content_reward": 0.0,
        "total_reward": 0.0,
        "pred_action": None,
        "gt_type": None,
        "pred_type": None,
    }

    gt_type = _normalize_action_type(gt_action.get("action", ""))
    info["gt_type"] = gt_type

    # ── Format reward: can we parse the action? ──
    pred_action = parse_action_from_text(pred_text)
    if pred_action is None:
        info["total_reward"] = 0.0
        return 0.0, info

    info["format_reward"] = 1.0
    info["pred_action"] = pred_action
    pred_type = _normalize_action_type(pred_action.get("action", ""))
    info["pred_type"] = pred_type

    # ── Type reward: does the action type match? ──
    if pred_type == gt_type:
        info["type_reward"] = 1.0
    else:
        # Partial credit for related types (click vs long_press)
        related_pairs = {
            frozenset({"click", "long_press"}),
        }
        if frozenset({pred_type, gt_type}) in related_pairs:
            info["type_reward"] = 0.5
        else:
            info["type_reward"] = 0.0

    # ── Content reward: type-specific matching ──
    content = _compute_content_reward(
        pred_action, gt_action, pred_type, gt_type,
        image_w, image_h, coord_threshold
    )
    info["content_reward"] = content

    # ── Total ──
    total = (w_format * info["format_reward"]
             + w_type * info["type_reward"]
             + w_content * content)
    info["total_reward"] = total
    return total, info


def _compute_content_reward(
    pred_action: Dict, gt_action: Dict,
    pred_type: str, gt_type: str,
    image_w: int, image_h: int,
    coord_threshold: float,
) -> float:
    """Compute content-specific reward based on ground truth action type."""

    if gt_type in ("click", "long_press"):
        gt_coord = gt_action.get("coordinate")
        pred_coord = pred_action.get("coordinate")
        if gt_coord is None:
            return 1.0 if pred_type == gt_type else 0.0
        dist = _coord_distance(pred_coord, gt_coord, image_w, image_h)
        return _coord_reward_continuous(dist, coord_threshold)

    elif gt_type in ("type", "open", "answer", "key"):
        gt_text = str(gt_action.get("text", "")).strip()
        pred_text = str(pred_action.get("text", "")).strip()
        return _text_similarity(pred_text, gt_text)

    elif gt_type == "swipe":
        gt_start = gt_action.get("startCoordinate") or gt_action.get("coordinate")
        gt_end = gt_action.get("endCoordinate")
        pred_start = pred_action.get("startCoordinate") or pred_action.get("coordinate")
        pred_end = pred_action.get("endCoordinate")
        if gt_start and gt_end and pred_start and pred_end:
            return _direction_similarity(pred_start, pred_end, gt_start, gt_end)
        return 1.0 if pred_type == gt_type else 0.0

    elif gt_type == "system_button":
        gt_btn = str(gt_action.get("button", "")).strip().lower()
        pred_btn = str(pred_action.get("button", "")).strip().lower()
        return _text_similarity(pred_btn, gt_btn)

    elif gt_type == "terminate":
        gt_status = str(gt_action.get("status", "")).strip().lower()
        pred_status = str(pred_action.get("status", "")).strip().lower()
        return _text_similarity(pred_status, gt_status)

    elif gt_type == "wait":
        return 1.0

    else:
        # Unknown type — only format+type matter
        return 0.0


# ── Trajectory-level reward computation ────────────────────────────

def compute_episode_rewards(
    pred_texts: List[str],
    gt_actions: List[Dict[str, Any]],
    image_ws: List[int],
    image_hs: List[int],
    **kwargs,
) -> Tuple[List[float], List[Dict]]:
    """Compute per-step rewards for an entire episode.

    Args:
        pred_texts: List of model output texts (one per step)
        gt_actions: List of GT action dicts (one per step)
        image_ws, image_hs: Per-step image dimensions

    Returns:
        (rewards, infos) — both lists of length num_steps
    """
    rewards, infos = [], []
    for pred, gt, w, h in zip(pred_texts, gt_actions, image_ws, image_hs):
        r, info = compute_step_reward(pred, gt, w, h, **kwargs)
        rewards.append(r)
        infos.append(info)
    return rewards, infos


def compute_discounted_returns(
    step_rewards: List[float],
    gamma: float = 0.5,
) -> List[float]:
    """Compute discounted returns with backward pass within a trajectory.

    G_t = r_t + γ * G_{t+1}

    Args:
        step_rewards: Per-step rewards [r_0, r_1, ..., r_T]
        gamma: Discount factor

    Returns:
        Discounted returns [G_0, G_1, ..., G_T]
    """
    T = len(step_rewards)
    returns = [0.0] * T
    running = 0.0
    for t in reversed(range(T)):
        running = step_rewards[t] + gamma * running
        returns[t] = running
    return returns


def normalize_advantages(
    returns_per_sample: np.ndarray,
    epsilon: float = 1e-6,
) -> np.ndarray:
    """Normalize advantages across K samples for one episode.

    Args:
        returns_per_sample: [K, T] array of discounted returns per rollout per step

    Returns:
        [K, T] normalized advantages
    """
    # Episode-level normalization: across K rollouts
    # Sum returns across steps for each rollout to get episode-level score
    episode_scores = returns_per_sample.sum(axis=1)  # [K]
    mean = episode_scores.mean()
    std = episode_scores.std()
    if std > epsilon:
        episode_adv = (episode_scores - mean) / std
    else:
        episode_adv = episode_scores - mean

    # Broadcast episode advantage to all steps
    episode_advantages = np.tile(episode_adv[:, None], (1, returns_per_sample.shape[1]))

    # Step-level normalization: across K rollouts per step
    step_means = returns_per_sample.mean(axis=0)  # [T]
    step_stds = returns_per_sample.std(axis=0)    # [T]
    step_stds = np.maximum(step_stds, epsilon)
    step_advantages = (returns_per_sample - step_means[None, :]) / step_stds[None, :]

    # Combine: 50/50 episode + step advantages
    advantages = 0.5 * episode_advantages + 0.5 * step_advantages
    return advantages


def should_filter_dapo(
    returns_per_sample: np.ndarray,
    std_threshold: float = 0.3,
) -> bool:
    """Check if this episode's K rollouts should be filtered (DAPO).

    Returns True if the episode should be SKIPPED (advantage collapsed).
    """
    episode_scores = returns_per_sample.sum(axis=1)  # [K]
    std = episode_scores.std()
    return std < std_threshold
