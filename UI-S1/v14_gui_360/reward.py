"""V14 GUI-360 Reward: Dense Step Rewards + SP + GiGPO + SPWA + Causal Type Masking.

Based on v12_gui_360/reward.py. V14 additions:
  - is_type_error(): detect action type mismatch from model output
  - apply_causal_type_masking(): zero out rewards after first type error per rollout,
    simulating the real agentic consequence that a wrong action type (e.g. click
    instead of type) derails the entire trajectory downstream
"""

import json
import re
from collections import defaultdict
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ═══════════════════════════════════════════════════════════════════════
# Action Parsing
# ═══════════════════════════════════════════════════════════════════════

def _convert_tool_call(tc: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Convert <tool_call> format to standard action format.

    GUI-360 SFT outputs: {"function":"click","args":{"coordinate":[x,y]}}
    We convert to: {"action":"click","coordinate":[x,y]}
    """
    func = tc.get("function", "")
    args = tc.get("args", {})
    if not func:
        return None
    action: Dict[str, Any] = {"action": func}
    if args.get("coordinate"):
        action["coordinate"] = args["coordinate"]
    if args.get("keys"):
        action["text"] = args["keys"]
    elif args.get("text"):
        action["text"] = args["text"]
    if args.get("start_coordinate"):
        action["coordinate"] = args["start_coordinate"]
    if args.get("end_coordinate"):
        action["endCoordinate"] = args["end_coordinate"]
    return action


def parse_action_from_text(text: str) -> Optional[Dict[str, Any]]:
    """Parse action dict from model output text."""
    # 1. Try <action> format (RL training)
    m = re.search(r'<action>\s*(\{.*?\})\s*</action>', text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass

    # 2. Try <tool_call> format (GUI-360 SFT)
    m = re.search(r'<tool_call>\s*(\{.*?\})\s*</tool_call>', text, re.DOTALL)
    if m:
        try:
            tc = json.loads(m.group(1))
            return _convert_tool_call(tc)
        except json.JSONDecodeError:
            pass

    # 3. Fallback: bare JSON with "action" key
    m = re.search(r'\{[^{}]*"action"[^{}]*\}', text)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    return None


# ═══════════════════════════════════════════════════════════════════════
# Base Reward Components
# ═══════════════════════════════════════════════════════════════════════

def _coord_distance(pred_coord, gt_coord, image_w: int, image_h: int) -> float:
    if pred_coord is None or gt_coord is None:
        return float("inf")
    try:
        dx = (float(pred_coord[0]) - float(gt_coord[0])) / image_w
        dy = (float(pred_coord[1]) - float(gt_coord[1])) / image_h
        return (dx ** 2 + dy ** 2) ** 0.5
    except (TypeError, IndexError, ValueError):
        return float("inf")


def _coord_reward_continuous(dist: float, threshold: float = 0.05) -> float:
    max_dist = threshold * 4
    if dist >= max_dist:
        return 0.0
    return 1.0 - dist / max_dist


def _text_similarity(pred: str, gt: str) -> float:
    if not pred and not gt:
        return 1.0
    if not pred or not gt:
        return 0.0
    return SequenceMatcher(None, pred.lower(), gt.lower()).ratio()


def _direction_similarity(pred_start, pred_end, gt_start, gt_end) -> float:
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
            return max(0.0, cos_sim)
    except (TypeError, IndexError, ValueError):
        pass
    return 0.0


_ACTION_TYPE_ALIASES = {
    "left_click": "click",
    "tap": "click",
    "double_click": "click",
    "double_click_input": "click",
    "double_click_on_coordinates": "click",
    "input": "type",
    "open_app": "open",
    "back": "system_button",
    "home": "system_button",
    "press": "long_press",
    "scroll": "swipe",
    "drag": "swipe",
    "wheel_mouse_input": "swipe",
}

VALID_ACTION_TYPES = {
    "click", "long_press", "type", "swipe", "open", "wait",
    "system_button", "terminate", "answer", "key", "drag",
}


def _normalize_action_type(atype: str) -> str:
    atype = atype.strip().lower()
    return _ACTION_TYPE_ALIASES.get(atype, atype)


# ═══════════════════════════════════════════════════════════════════════
# Dense Multi-Component Step Reward
# ═══════════════════════════════════════════════════════════════════════

def compute_step_reward(
    pred_text: str,
    gt_action: Dict[str, Any],
    image_w: int = 1040,
    image_h: int = 736,
    coord_threshold: float = 0.05,
    w_format: float = 0.1,
    w_type: float = 0.2,
    w_content: float = 0.7,
) -> Tuple[float, Dict[str, Any]]:
    """Compute dense multi-component reward for a single step. Returns (total, info)."""
    info = {
        "format_reward": 0.0, "type_reward": 0.0,
        "content_reward": 0.0, "total_reward": 0.0,
        "pred_action": None, "gt_type": None, "pred_type": None,
    }

    gt_type = _normalize_action_type(gt_action.get("action", ""))
    info["gt_type"] = gt_type

    pred_action = parse_action_from_text(pred_text)
    if pred_action is None:
        return 0.0, info

    info["format_reward"] = 1.0
    info["pred_action"] = pred_action
    pred_type = _normalize_action_type(pred_action.get("action", ""))
    info["pred_type"] = pred_type

    if pred_type == gt_type:
        info["type_reward"] = 1.0
    elif frozenset({pred_type, gt_type}) in {
        frozenset({"click", "long_press"}),
        frozenset({"click", "double_click"}),
    }:
        info["type_reward"] = 0.5
    else:
        info["type_reward"] = 0.0

    # Complete type mismatch (type_reward==0): don't reward content
    # Fixes click collapse where click on empty-text type steps got 0.8 reward
    if info["type_reward"] == 0.0:
        info["content_reward"] = 0.0
        total = w_format * info["format_reward"]
        info["total_reward"] = total
        return total, info

    content = _compute_content_reward(
        pred_action, gt_action, pred_type, gt_type,
        image_w, image_h, coord_threshold
    )
    info["content_reward"] = content

    total = w_format * info["format_reward"] + w_type * info["type_reward"] + w_content * content
    info["total_reward"] = total
    return total, info


def _compute_content_reward(
    pred_action: Dict, gt_action: Dict,
    pred_type: str, gt_type: str,
    image_w: int, image_h: int,
    coord_threshold: float,
) -> float:
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
        return 0.0


def compute_episode_rewards(
    pred_texts: List[str],
    gt_actions: List[Dict[str, Any]],
    image_ws: List[int],
    image_hs: List[int],
    **kwargs,
) -> Tuple[List[float], List[Dict]]:
    """Compute per-step rewards for an entire episode."""
    rewards, infos = [], []
    for pred, gt, w, h in zip(pred_texts, gt_actions, image_ws, image_hs):
        r, info = compute_step_reward(pred, gt, w, h, **kwargs)
        rewards.append(r)
        infos.append(info)
    return rewards, infos


# ═══════════════════════════════════════════════════════════════════════
# Causal Type Masking (V14)
# ═══════════════════════════════════════════════════════════════════════

def is_type_error(pred_text: str, gt_action: Dict[str, Any]) -> bool:
    """Check if prediction has a hard action-type mismatch with ground truth.

    A type error means the action category is fundamentally wrong (e.g. click
    instead of type). In a real deployment this would derail the trajectory
    because the UI state would diverge.

    Soft mismatches (click vs long_press) are NOT type errors — the UI effect
    is similar enough that the trajectory could still recover.
    """
    gt_type = _normalize_action_type(gt_action.get("action", ""))
    pred_action = parse_action_from_text(pred_text)

    if pred_action is None:
        # No parseable action — this is a format error, not a type error.
        # SP already handles this via low reward.
        return False

    pred_type = _normalize_action_type(pred_action.get("action", ""))

    if pred_type == gt_type:
        return False

    # Soft matches: similar UI effect, trajectory can recover
    soft_pairs = {
        frozenset({"click", "long_press"}),
        frozenset({"click", "double_click"}),
    }
    if frozenset({pred_type, gt_type}) in soft_pairs:
        return False

    return True


def apply_causal_type_masking(
    all_rewards: np.ndarray,
    step_texts: List[List[str]],
    gt_actions: List[Dict[str, Any]],
) -> Tuple[np.ndarray, np.ndarray]:
    """Zero out rewards after the first type error per rollout.

    Simulates the agentic consequence: a wrong action type (e.g. click instead
    of type) means the text was never entered, the UI state diverges, and all
    subsequent steps are invalid — regardless of what the model predicts for
    them under teacher-forced history.

    Args:
        all_rewards: [K, T] reward matrix (modified in-place and returned)
        step_texts: step_texts[t][k] = decoded text for rollout k at step t
        gt_actions: gt_actions[t] = ground truth action dict for step t

    Returns:
        all_rewards: [K, T] with causal masking applied
        type_error_steps: [K] array — step index of first type error per rollout
                          (T if no type error)
    """
    K, T = all_rewards.shape
    type_error_steps = np.full(K, T, dtype=np.int32)

    for k in range(K):
        for t in range(T):
            if is_type_error(step_texts[t][k], gt_actions[t]):
                type_error_steps[k] = t
                # Zero out all subsequent steps (current step keeps its reward)
                all_rewards[k, t + 1:] = 0.0
                break

    return all_rewards, type_error_steps


# ═══════════════════════════════════════════════════════════════════════
# Sequential Progress (SP)
# ═══════════════════════════════════════════════════════════════════════

def compute_sp_scores(
    step_rewards: np.ndarray,
    match_threshold: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute Sequential Progress for K rollouts over T steps.

    SP = first_error_step / total_steps (1.0 if all correct).

    Args:
        step_rewards: [K, T] array of per-step rewards
        match_threshold: reward >= threshold counts as correct

    Returns:
        sp_scores: [K] array of SP scores
        first_errors: [K] array of first error step indices (T if all correct)
    """
    K, T = step_rewards.shape
    matches = step_rewards >= match_threshold  # [K, T] bool

    sp_scores = np.zeros(K, dtype=np.float32)
    first_errors = np.full(K, T, dtype=np.int32)

    for k in range(K):
        for t in range(T):
            if not matches[k, t]:
                first_errors[k] = t
                break
        sp_scores[k] = first_errors[k] / T

    return sp_scores, first_errors


# ═══════════════════════════════════════════════════════════════════════
# GiGPO + SPWA Advantages
# ═══════════════════════════════════════════════════════════════════════

def compute_gigpo_spwa_advantages(
    sp_scores: np.ndarray,
    first_errors: np.ndarray,
    T: int,
    spwa_decay: float = 0.5,
    epsilon: float = 1e-6,
) -> np.ndarray:
    """Compute GiGPO cross-trajectory advantages with SPWA step weighting.

    GiGPO: advantage_k = (SP_k - mean(SP)) / (std(SP) + eps)
    SPWA:  weight_t = 1.0 if t <= first_error, else decay^(t - first_error)

    Final: advantages[k, t] = gigpo_advantage[k] * spwa_weight[k, t]

    Args:
        sp_scores: [K] SP scores
        first_errors: [K] first error step indices
        T: total number of steps
        spwa_decay: decay factor for steps after first error
        epsilon: numerical stability

    Returns:
        advantages: [K, T] array
    """
    K = sp_scores.shape[0]

    # GiGPO: normalize SP across K rollouts
    mean_sp = sp_scores.mean()
    std_sp = sp_scores.std()
    if std_sp > epsilon:
        gigpo_adv = (sp_scores - mean_sp) / (std_sp + epsilon)
    else:
        gigpo_adv = sp_scores - mean_sp  # [K]

    # SPWA: per-step weights based on first error
    spwa_weights = np.ones((K, T), dtype=np.float32)
    for k in range(K):
        fe = first_errors[k]
        for t in range(T):
            if t <= fe:
                spwa_weights[k, t] = 1.0
            else:
                w = spwa_decay ** (t - fe)
                spwa_weights[k, t] = max(w, 0.1)

    # Combine: broadcast GiGPO advantage × SPWA weights
    advantages = gigpo_adv[:, None] * spwa_weights  # [K, T]

    return advantages


# ═══════════════════════════════════════════════════════════════════════
# DAPO Filtering
# ═══════════════════════════════════════════════════════════════════════

def should_filter_episode(
    sp_scores: np.ndarray,
    std_threshold: float = 0.1,
) -> bool:
    """Check if episode should be skipped (advantage collapsed).

    Returns True if SP std across K rollouts is below threshold.
    """
    return sp_scores.std() < std_threshold


# ═══════════════════════════════════════════════════════════════════════
# Full Pipeline: step rewards -> SP -> GiGPO+SPWA advantages
# ═══════════════════════════════════════════════════════════════════════

def compute_step_advantages(
    step_rewards: np.ndarray,
    epsilon: float = 1e-6,
) -> np.ndarray:
    """Per-step cross-K normalized advantages.

    For each step t, normalize rewards across K rollouts:
        step_adv[k, t] = (reward[k,t] - mean_K) / (std_K + eps)

    Args:
        step_rewards: [K, T] dense step rewards

    Returns:
        step_advantages: [K, T] array
    """
    K, T = step_rewards.shape
    step_adv = np.zeros_like(step_rewards)
    for t in range(T):
        col = step_rewards[:, t]
        mean_r = col.mean()
        std_r = col.std()
        if std_r > epsilon:
            step_adv[:, t] = (col - mean_r) / (std_r + epsilon)
        else:
            step_adv[:, t] = col - mean_r
    return step_adv


def compute_trajectory_advantages(
    step_rewards: np.ndarray,
    match_threshold: float = 0.5,
    spwa_decay: float = 0.5,
    dapo_threshold: float = 0.1,
    step_adv_weight: float = 0.0,
    epsilon: float = 1e-6,
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Full pipeline from step rewards to advantages.

    Combines trajectory-level (SP+GiGPO+SPWA) and step-level advantages:
        advantage = (1 - step_adv_weight) * traj_adv + step_adv_weight * step_adv

    Args:
        step_rewards: [K, T] dense step rewards
        step_adv_weight: mixing weight for step-level advantage (0 = traj only)

    Returns:
        (advantages, sp_scores, first_errors) or None if DAPO filtered
    """
    sp_scores, first_errors = compute_sp_scores(step_rewards, match_threshold)

    if should_filter_episode(sp_scores, dapo_threshold):
        return None

    K, T = step_rewards.shape
    traj_adv = compute_gigpo_spwa_advantages(
        sp_scores, first_errors, T, spwa_decay, epsilon
    )

    if step_adv_weight > 0:
        step_adv = compute_step_advantages(step_rewards, epsilon)
        advantages = (1 - step_adv_weight) * traj_adv + step_adv_weight * step_adv
    else:
        advantages = traj_adv

    return advantages, sp_scores, first_errors


# ═══════════════════════════════════════════════════════════════════════
# Standard GRPO: Trajectory Return Normalization
# ═══════════════════════════════════════════════════════════════════════

def compute_grpo_advantages(
    step_rewards: np.ndarray,
    dapo_threshold: float = 0.0,
    match_threshold: float = 0.5,
    epsilon: float = 1e-6,
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Standard GRPO advantage: normalize total trajectory return across K.

    R_k = sum_t r_{k,t}   (total return per trajectory)
    A_k = (R_k - mean(R)) / (std(R) + eps)
    All steps in trajectory k share the same advantage A_k.

    Args:
        step_rewards: [K, T] dense step rewards
        dapo_threshold: filter if std(R) < threshold
        match_threshold: for SP score computation (reporting only)

    Returns:
        (advantages, sp_scores, first_errors) or None if filtered
    """
    K, T = step_rewards.shape

    # Total return per trajectory
    total_returns = step_rewards.sum(axis=1)  # [K]

    std_r = total_returns.std()
    if std_r < dapo_threshold:
        return None

    mean_r = total_returns.mean()
    if std_r > epsilon:
        adv_k = (total_returns - mean_r) / (std_r + epsilon)
    else:
        adv_k = total_returns - mean_r  # [K]

    # Broadcast same advantage to all steps
    advantages = np.repeat(adv_k[:, None], T, axis=1)  # [K, T]

    # Compute SP scores for reporting (not used in advantage)
    sp_scores, first_errors = compute_sp_scores(step_rewards, match_threshold)

    return advantages, sp_scores, first_errors
