#!/usr/bin/env python3
"""v10 Reward Functions: separate rewards for grounder and actor.

Grounder reward: coord_correct — did the actor's predicted coordinate match GT?
Actor reward:    action_correct — did the full action match GT?

Rewards are CONTINUOUS to provide gradient signal for GRPO (discrete buckets
cause all K samples to get identical rewards → zero advantages).
"""

import json
import re
from difflib import SequenceMatcher
from typing import Dict, Any, Optional, Tuple


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


def coord_distance(pred_coord, gt_coord, image_w, image_h) -> float:
    """Compute normalized Euclidean distance between two coordinates."""
    if pred_coord is None or gt_coord is None:
        return float('inf')
    dx = (pred_coord[0] - gt_coord[0]) / image_w
    dy = (pred_coord[1] - gt_coord[1]) / image_h
    return (dx ** 2 + dy ** 2) ** 0.5


def _coord_reward_continuous(dist: float, threshold: float = 0.05) -> float:
    """Continuous coordinate reward: 1.0 at dist=0, linearly decays to 0 at dist=max_dist."""
    max_dist = threshold * 4  # beyond this → 0
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


_ACTION_TYPE_ALIASES = {
    "left_click": "click",
    "tap": "click",
    "input": "type",
    "open_app": "open",
    "press": "long_press",
    "scroll": "swipe",
}

VALID_ACTION_TYPES = {
    "click", "long_press", "type", "swipe", "open", "wait",
    "system_button", "terminate", "answer", "key",
}


def _normalize_type(t: str) -> str:
    t = t.strip().lower()
    return _ACTION_TYPE_ALIASES.get(t, t)


def _parse_grounder_tags(grounder_text: str) -> Tuple[Optional[str], Optional[str]]:
    """Extract <action_type> and <target> from grounder output.

    Returns (action_type, target) — None if tag not found.
    """
    action_type = None
    target = None
    m = re.search(r'<action_type>\s*(.*?)\s*</action_type>', grounder_text, re.DOTALL)
    if m:
        action_type = m.group(1).strip()
    m = re.search(r'<target>\s*(.*?)\s*</target>', grounder_text, re.DOTALL)
    if m:
        target = m.group(1).strip()
    return action_type, target


def grounder_format_reward(
    grounder_text: str,
    gt_action: Dict[str, Any],
) -> float:
    """Reward for grounder's own output format and type accuracy.

    Components (range [0, 1]):
      - Has <action_type> tag:    0.3
      - Action type matches GT:   0.5
      - Has <target> tag:         0.2

    This directly incentivizes the grounder to output structured format,
    which was the root cause of v10's 43% type-error rate.
    """
    action_type, target = _parse_grounder_tags(grounder_text)

    reward = 0.0

    # ── Format: has <action_type> tag? ──
    if action_type is not None:
        reward += 0.3
        # ── Type accuracy: does it match GT? ──
        gt_type = _normalize_type(gt_action.get("action", ""))
        pred_type = _normalize_type(action_type)
        if pred_type == gt_type:
            reward += 0.5
        else:
            # Partial credit for related types
            related = {frozenset({"click", "long_press"})}
            if frozenset({pred_type, gt_type}) in related:
                reward += 0.25

    # ── Format: has <target> tag? ──
    if target is not None:
        reward += 0.2

    return reward


def grounder_reward(
    actor_output: str,
    gt_action: Dict[str, Any],
    image_w: int = 1080,
    image_h: int = 2400,
    threshold: float = 0.05,
    grounder_text: str = "",
    w_format: float = 0.5,
    w_downstream: float = 0.5,
) -> float:
    """Grounder reward combining format compliance + downstream actor quality.

    Two components:
      1. Format reward (w_format): Does the grounder output proper
         <action_type>/<target> tags, and is the type correct?
      2. Downstream reward (w_downstream): Did the actor (reading the
         grounder's output) produce the correct action?

    The format component provides dense gradient signal that directly
    incentivizes structured output — fixing v10's root cause where the
    grounder never learned to output <action_type> tags (0/8444).

    Args:
        actor_output: Actor's generated text
        gt_action: Ground truth action dict
        grounder_text: Grounder's generated text (for format reward)
        w_format: Weight for format reward [0, 1]
        w_downstream: Weight for downstream actor quality [0, 1]

    Returns:
        Reward in [0, 1].
    """
    # ── Format component ──
    # When no grounder_text is provided (e.g. validation, external callers),
    # skip the format/downstream split entirely and return pure downstream reward.
    if not grounder_text:
        w_format = 0.0
        w_downstream = 1.0

    format_r = 0.0
    if grounder_text:
        format_r = grounder_format_reward(grounder_text, gt_action)

    # ── Downstream component (original logic) ──
    gt_type = gt_action.get("action", "")
    if gt_type == "left_click":
        gt_type = "click"

    if gt_type not in ("click", "long_press"):
        downstream_r = actor_reward(actor_output, gt_action, image_w, image_h)
    else:
        gt_coord = gt_action.get("coordinate")
        if gt_coord is None:
            downstream_r = actor_reward(actor_output, gt_action, image_w, image_h)
        else:
            pred_action = parse_action_from_text(actor_output)
            if pred_action is None:
                downstream_r = 0.0
            else:
                pred_type = pred_action.get("action", "")
                if pred_type == "left_click":
                    pred_action["action"] = "click"
                pred_coord = pred_action.get("coordinate")
                if pred_coord is None:
                    downstream_r = 0.0
                else:
                    dist = coord_distance(pred_coord, gt_coord, image_w, image_h)
                    downstream_r = _coord_reward_continuous(dist, threshold)

    return w_format * format_r + w_downstream * downstream_r


def actor_reward(
    actor_output: str,
    gt_action: Dict[str, Any],
    image_w: int = 1080,
    image_h: int = 2400,
    threshold: float = 0.05,
) -> float:
    """Actor reward based on full action correctness (continuous).

    Checks both action type and content (coordinate/text/direction).

    Returns:
        Reward in [0, 1].
    """
    pred_action = parse_action_from_text(actor_output)
    if pred_action is None:
        return 0.0

    gt_type = gt_action.get("action", "")
    if gt_type == "left_click":
        gt_type = "click"
    pred_type = pred_action.get("action", "")
    if pred_type == "left_click":
        pred_type = "click"
        pred_action["action"] = "click"

    # Type mismatch
    if pred_type != gt_type:
        return 0.0

    # Type-specific content matching (continuous)
    if gt_type in ("click", "long_press"):
        gt_coord = gt_action.get("coordinate")
        pred_coord = pred_action.get("coordinate")
        if gt_coord is None or pred_coord is None:
            return 0.3  # type match only

        dist = coord_distance(pred_coord, gt_coord, image_w, image_h)
        # Type match base: 0.3, coordinate bonus: up to 0.7
        coord_bonus = 0.7 * _coord_reward_continuous(dist, threshold)
        return 0.3 + coord_bonus

    elif gt_type in ("type", "open", "answer", "key"):
        gt_text = gt_action.get("text", "").strip()
        pred_text = pred_action.get("text", "").strip()
        sim = _text_similarity(pred_text, gt_text)
        # Type match base: 0.3, text similarity bonus: up to 0.7
        return 0.3 + 0.7 * sim

    elif gt_type == "swipe":
        # Check direction match for finer-grained reward
        gt_start = gt_action.get("startCoordinate") or gt_action.get("coordinate")
        gt_end = gt_action.get("endCoordinate")
        pred_start = pred_action.get("startCoordinate") or pred_action.get("coordinate")
        pred_end = pred_action.get("endCoordinate")
        if gt_start and gt_end and pred_start and pred_end:
            # Direction similarity: compare vectors
            gt_dx = gt_end[0] - gt_start[0]
            gt_dy = gt_end[1] - gt_start[1]
            pred_dx = pred_end[0] - pred_start[0]
            pred_dy = pred_end[1] - pred_start[1]
            gt_mag = (gt_dx**2 + gt_dy**2)**0.5
            pred_mag = (pred_dx**2 + pred_dy**2)**0.5
            if gt_mag > 0 and pred_mag > 0:
                cos_sim = (gt_dx * pred_dx + gt_dy * pred_dy) / (gt_mag * pred_mag)
                cos_sim = max(-1.0, min(1.0, cos_sim))
                # cos_sim in [-1, 1] → reward bonus in [0, 0.7]
                return 0.3 + 0.7 * max(0.0, cos_sim)
        return 1.0  # type match, no direction info

    elif gt_type == "wait":
        return 1.0

    elif gt_type == "system_button":
        gt_btn = gt_action.get("button", "").strip().lower()
        pred_btn = pred_action.get("button", "").strip().lower()
        sim = _text_similarity(pred_btn, gt_btn)
        return 0.3 + 0.7 * sim

    elif gt_type == "terminate":
        gt_status = gt_action.get("status", "").strip().lower()
        pred_status = pred_action.get("status", "").strip().lower()
        sim = _text_similarity(pred_status, gt_status)
        return 0.3 + 0.7 * sim

    else:
        return 0.3  # unknown action type, type match only


def compute_rewards(
    actor_outputs: list,
    gt_actions: list,
    image_ws: list,
    image_hs: list,
) -> Tuple[list, list]:
    """Compute grounder and actor rewards for a batch."""
    g_rewards = []
    a_rewards = []
    for actor_out, gt_act, w, h in zip(actor_outputs, gt_actions, image_ws, image_hs):
        g_rewards.append(grounder_reward(actor_out, gt_act, w, h))
        a_rewards.append(actor_reward(actor_out, gt_act, w, h))
    return g_rewards, a_rewards
