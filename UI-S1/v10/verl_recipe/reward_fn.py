"""Cooperative reward function for verl recipe.

Wraps v10/reward.py's grounder_reward and actor_reward into verl's
compute_score interface. Returns dual rewards as a dict so the trainer
can build separate advantage signals for each adapter.
"""

import json
import re
from difflib import SequenceMatcher
from typing import Any, Dict, Optional


# ── Action parsing ──────────────────────────────────────────────────────────

_ACTION_TAG_RE = re.compile(r"<action>\s*(\{.*?\})\s*</action>", re.DOTALL)
_ACTION_RAW_RE = re.compile(r'\{[^{}]*"action"[^{}]*\}')


def parse_action_from_text(text: str) -> Optional[Dict[str, Any]]:
    m = _ACTION_TAG_RE.search(text)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    m = _ACTION_RAW_RE.search(text)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    return None


# ── Distance / similarity helpers ───────────────────────────────────────────

def coord_distance(pred_coord, gt_coord, image_w, image_h) -> float:
    if pred_coord is None or gt_coord is None:
        return float("inf")
    dx = (pred_coord[0] - gt_coord[0]) / image_w
    dy = (pred_coord[1] - gt_coord[1]) / image_h
    return (dx ** 2 + dy ** 2) ** 0.5


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


# ── Per-adapter reward functions (from v10/reward.py) ───────────────────────

def grounder_reward(
    actor_output: str,
    gt_action: Dict[str, Any],
    image_w: int = 1080,
    image_h: int = 2400,
    threshold: float = 0.05,
) -> float:
    gt_type = gt_action.get("action", "")
    if gt_type not in ("click", "long_press"):
        return actor_reward(actor_output, gt_action, image_w, image_h)
    gt_coord = gt_action.get("coordinate")
    if gt_coord is None:
        return actor_reward(actor_output, gt_action, image_w, image_h)
    pred_action = parse_action_from_text(actor_output)
    if pred_action is None:
        return 0.0
    pred_coord = pred_action.get("coordinate")
    if pred_coord is None:
        return 0.0
    dist = coord_distance(pred_coord, gt_coord, image_w, image_h)
    return _coord_reward_continuous(dist, threshold)


def actor_reward(
    actor_output: str,
    gt_action: Dict[str, Any],
    image_w: int = 1080,
    image_h: int = 2400,
    threshold: float = 0.05,
) -> float:
    pred_action = parse_action_from_text(actor_output)
    if pred_action is None:
        return 0.0

    gt_type = gt_action.get("action", "")
    pred_type = pred_action.get("action", "")
    if pred_type != gt_type:
        return 0.0

    if gt_type in ("click", "long_press"):
        gt_coord = gt_action.get("coordinate")
        pred_coord = pred_action.get("coordinate")
        if gt_coord is None or pred_coord is None:
            return 0.3
        dist = coord_distance(pred_coord, gt_coord, image_w, image_h)
        return 0.3 + 0.7 * _coord_reward_continuous(dist, threshold)

    elif gt_type in ("type", "open", "answer", "key"):
        gt_text = gt_action.get("text", "").strip()
        pred_text = pred_action.get("text", "").strip()
        return 0.3 + 0.7 * _text_similarity(pred_text, gt_text)

    elif gt_type == "swipe":
        gt_start = gt_action.get("startCoordinate") or gt_action.get("coordinate")
        gt_end = gt_action.get("endCoordinate")
        pred_start = pred_action.get("startCoordinate") or pred_action.get("coordinate")
        pred_end = pred_action.get("endCoordinate")
        if gt_start and gt_end and pred_start and pred_end:
            gt_dx = gt_end[0] - gt_start[0]
            gt_dy = gt_end[1] - gt_start[1]
            pred_dx = pred_end[0] - pred_start[0]
            pred_dy = pred_end[1] - pred_start[1]
            gt_mag = (gt_dx**2 + gt_dy**2) ** 0.5
            pred_mag = (pred_dx**2 + pred_dy**2) ** 0.5
            if gt_mag > 0 and pred_mag > 0:
                cos_sim = (gt_dx * pred_dx + gt_dy * pred_dy) / (gt_mag * pred_mag)
                cos_sim = max(-1.0, min(1.0, cos_sim))
                return 0.3 + 0.7 * max(0.0, cos_sim)
        return 1.0

    elif gt_type == "wait":
        return 1.0

    elif gt_type == "system_button":
        gt_btn = gt_action.get("button", "").strip().lower()
        pred_btn = pred_action.get("button", "").strip().lower()
        return 0.3 + 0.7 * _text_similarity(pred_btn, gt_btn)

    elif gt_type == "terminate":
        gt_status = gt_action.get("status", "").strip().lower()
        pred_status = pred_action.get("status", "").strip().lower()
        return 0.3 + 0.7 * _text_similarity(pred_status, gt_status)

    return 0.3


# ── verl compute_score interface ────────────────────────────────────────────

def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    """verl-compatible reward function.

    Returns a dict with dual scores so the reward manager can expose them
    as extra_info for the trainer to split into per-adapter advantages.
    """
    if isinstance(ground_truth, str):
        ground_truth = json.loads(ground_truth)

    image_w = 1080
    image_h = 2400
    if extra_info:
        if isinstance(extra_info, str):
            extra_info = json.loads(extra_info)
        image_w = extra_info.get("image_w", 1080)
        image_h = extra_info.get("image_h", 2400)

    g_score = grounder_reward(solution_str, ground_truth, image_w, image_h)
    a_score = actor_reward(solution_str, ground_truth, image_w, image_h)

    return {
        "score": a_score,  # primary score used by verl for token_level_scores
        "grounder_score": g_score,
        "actor_score": a_score,
    }
