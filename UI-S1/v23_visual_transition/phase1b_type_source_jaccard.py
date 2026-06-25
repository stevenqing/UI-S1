#!/usr/bin/env python3
"""Phase 1b type-mismatch source and two-source orthogonality gate.

Builds only prompt-biased far/type sources on the same frozen full-SFT model.
The script first supports type-source viability, then the Proposition 1
error-orthogonality test on a realistic TEST mix.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

_PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_DIR not in sys.path:
    sys.path.insert(0, _PROJECT_DIR)

from v13_gui_360.eval_gui360_template import (  # noqa: E402
    _format_action_for_history,
    build_step_prompt,
    parse_tool_call,
)
from v13_gui_360.reward import compute_step_reward, parse_action_from_text  # noqa: E402

FAR_GUIDANCE = """Use an element-grounding-first procedure for this step.

Before choosing coordinates, explicitly identify the semantic UI element needed for the current instruction and history. Compare nearby or visually similar candidate elements, reject distractors, and choose the element whose label, role, or context best matches the next required step. Then emit exactly one <tool_call> using the center of that selected element. Do not change the action type unless the screenshot clearly requires it; this source is only meant to improve element disambiguation for click targets."""

TYPE_GUIDANCE = """Use an affordance-and-prior-correction procedure for this step.

First decide the correct ACTION TYPE before choosing coordinates. Do not copy the most recent or most frequent action type from the history. Judge the target element's affordance from its shape, role, label, and current task context:
- choose type when the next step is entering text into an input/edit/document/cell/text field, especially if the instruction supplies text or the current target is an editable region;
- choose click when the next step is selecting, opening, pressing a button/menu/tab, focusing a cell/control, or confirming an option;
- choose drag/wheel only when the task requires scrolling or dragging.
After deciding the verb, choose the target and emit exactly one <tool_call>. This source is only meant to fix wrong action type; avoid changing a correct verb into a different one."""

ACTION_ALIASES = {
    "drag": "swipe",
    "scroll": "swipe",
    "wheel_mouse_input": "swipe",
    "input": "type",
    "left_click": "click",
    "tap": "click",
    "double_click": "click",
}


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path) as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_episodes(path: str) -> Dict[str, Dict[str, Any]]:
    return {str(row["episode_id"]): row for row in read_jsonl(path)}


def normalize_action_type(value: Any) -> str:
    text = str(value or "").strip().lower()
    return ACTION_ALIASES.get(text, text)


def action_coord(action: Optional[Dict[str, Any]]) -> Optional[Tuple[float, float]]:
    if not action:
        return None
    coord = action.get("coordinate") or action.get("start_coordinate") or action.get("startCoordinate")
    if coord is None or len(coord) < 2:
        return None
    try:
        return float(coord[0]), float(coord[1])
    except (TypeError, ValueError):
        return None


def coord_distance_px(pred_action: Optional[Dict[str, Any]], gt_action: Dict[str, Any]) -> Optional[float]:
    pred = action_coord(pred_action)
    gt = action_coord(gt_action)
    if pred is None or gt is None:
        return None
    return math.sqrt((pred[0] - gt[0]) ** 2 + (pred[1] - gt[1]) ** 2)


def first_bad_step(eval_episode: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    for step in eval_episode.get("steps", []):
        if not step.get("success"):
            return step
    return None


def parse_bbox(value: Any) -> Optional[Tuple[float, float, float, float]]:
    if not value or len(value) != 4:
        return None
    try:
        x1, y1, x2, y2 = [float(v) for v in value]
    except (TypeError, ValueError):
        return None
    return min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)


def bbox_center(bbox: Tuple[float, float, float, float]) -> Tuple[float, float]:
    return (bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0


def classify_action(
    pred_text: str,
    gt_action: Dict[str, Any],
    image_w: int,
    image_h: int,
    near_px: float,
    far_px: float,
    match_threshold: float,
) -> Dict[str, Any]:
    pred_action = parse_tool_call(pred_text)
    if pred_action is None:
        pred_action = parse_action_from_text(pred_text)
    fake_text = f"<action>{json.dumps(pred_action)}</action>" if pred_action else pred_text
    reward, info = compute_step_reward(fake_text, gt_action, image_w=image_w, image_h=image_h)
    success = reward >= match_threshold
    gt_type = normalize_action_type(info.get("gt_type") or gt_action.get("action"))
    pred_type = normalize_action_type(info.get("pred_type") or (pred_action or {}).get("action"))
    distance = None
    if success:
        bucket = "correct"
    elif not pred_action or info.get("format_reward", 0.0) <= 0:
        bucket = "format_error"
    elif gt_type != pred_type:
        bucket = "type_mismatch"
    elif gt_type != "click":
        bucket = f"same_type_non_click:{gt_type}"
    else:
        distance = coord_distance_px(pred_action, gt_action)
        if distance is None:
            bucket = "grounding_missing_coord"
        elif distance <= near_px:
            bucket = "near_miss"
        elif distance >= far_px:
            bucket = "far_miss"
        else:
            bucket = "mid_miss"
    return {
        "success": success,
        "bucket": bucket,
        "reward": reward,
        "pred_action": pred_action,
        "pred_type": pred_type,
        "gt_type": gt_type,
        "distance_px": distance,
        "format_reward": info.get("format_reward", 0.0),
        "type_reward": info.get("type_reward", 0.0),
        "content_reward": info.get("content_reward", 0.0),
        "pred_text": pred_text[:1000],
    }


def classify_eval_step(
    eval_step: Dict[str, Any],
    gt_action: Dict[str, Any],
    near_px: float,
    far_px: float,
    image_w: int,
    image_h: int,
) -> Dict[str, Any]:
    pred_action = eval_step.get("pred_action") or parse_action_from_text(eval_step.get("pred_text", "") or "")
    gt_type = normalize_action_type(eval_step.get("gt_type") or gt_action.get("action"))
    pred_type = normalize_action_type(eval_step.get("pred_type") or (pred_action or {}).get("action"))
    if eval_step.get("success"):
        bucket = "correct"
        distance = None
    elif not pred_action or eval_step.get("format_reward", 0.0) <= 0:
        bucket = "format_error"
        distance = None
    elif gt_type != pred_type:
        bucket = "type_mismatch"
        distance = None
    elif gt_type != "click":
        bucket = f"same_type_non_click:{gt_type}"
        distance = None
    else:
        distance = coord_distance_px(pred_action, gt_action)
        if distance is None:
            bucket = "grounding_missing_coord"
        elif distance <= near_px:
            bucket = "near_miss"
        elif distance >= far_px:
            bucket = "far_miss"
        else:
            bucket = "mid_miss"
    return {
        "success": bool(eval_step.get("success")),
        "bucket": bucket,
        "reward": eval_step.get("reward", 0.0),
        "pred_action": pred_action,
        "pred_type": pred_type,
        "gt_type": gt_type,
        "distance_px": distance,
        "format_reward": eval_step.get("format_reward", 0.0),
        "type_reward": eval_step.get("type_reward", 0.0),
        "content_reward": eval_step.get("content_reward", 0.0),
        "pred_text": (eval_step.get("pred_text") or "")[:1000],
    }


def state_features(episode: Dict[str, Any], step_idx: int) -> Dict[str, float]:
    steps = episode.get("steps", [])
    step = steps[step_idx]
    bbox = parse_bbox(step.get("bbox"))
    image_w = float(step.get("image_w") or 1040)
    image_h = float(step.get("image_h") or 736)
    history = Counter(normalize_action_type((s.get("action") or {}).get("action")) for s in steps[:step_idx])
    hist_total = max(sum(history.values()), 1)
    seq = Counter(normalize_action_type((s.get("action") or {}).get("action")) for s in steps)
    seq_total = max(sum(seq.values()), 1)
    features = {
        "history_click_share": history["click"] / hist_total,
        "history_type_share": history["type"] / hist_total,
        "episode_click_share": seq["click"] / seq_total,
        "episode_type_share": seq["type"] / seq_total,
        "target_bbox_height_norm": 0.0,
        "target_bbox_width_norm": 0.0,
        "proxy_similar_target_bbox_count": 0.0,
    }
    if bbox:
        x1, y1, x2, y2 = bbox
        width, height = max(0.0, x2 - x1), max(0.0, y2 - y1)
        cx, cy = bbox_center(bbox)
        features["target_bbox_height_norm"] = height / max(image_h, 1.0)
        features["target_bbox_width_norm"] = width / max(image_w, 1.0)
        similar = 0
        for idx, other in enumerate(steps):
            if idx == step_idx:
                continue
            ob = parse_bbox(other.get("bbox"))
            if not ob:
                continue
            ox, oy = bbox_center(ob)
            dist = math.sqrt(((cx - ox) / max(image_w, 1.0)) ** 2 + ((cy - oy) / max(image_h, 1.0)) ** 2)
            if dist < 0.12:
                similar += 1
        features["proxy_similar_target_bbox_count"] = float(similar)
    return features


def build_history(eval_episode: Dict[str, Any], step_idx: int) -> List[str]:
    history = []
    for prior in eval_episode.get("steps", [])[:step_idx]:
        history.append(_format_action_for_history(prior.get("pred_action"), int(prior.get("step_idx", len(history))) + 1))
    return history


def build_base_states(args: argparse.Namespace, episodes: Dict[str, Dict[str, Any]], base_results: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    buckets: Dict[str, List[Dict[str, Any]]] = {"far_miss": [], "type_mismatch": [], "base_correct": []}
    for episode_id, eval_episode in base_results.items():
        episode = episodes.get(str(episode_id))
        if episode is None:
            continue
        bad = first_bad_step(eval_episode)
        if bad is not None:
            step_idx = int(bad.get("step_idx", 0))
            if step_idx < len(episode.get("steps", [])):
                step = episode["steps"][step_idx]
                base = classify_eval_step(bad, step.get("action") or {}, args.near_px, args.far_px, int(step.get("image_w") or 1040), int(step.get("image_h") or 736))
                if base["bucket"] in {"far_miss", "type_mismatch"}:
                    buckets[base["bucket"]].append(make_state(str(episode_id), episode, eval_episode, step_idx, base, base["bucket"]))
        for result in eval_episode.get("steps", []):
            if not result.get("success"):
                continue
            step_idx = int(result.get("step_idx", 0))
            if step_idx >= len(episode.get("steps", [])):
                continue
            step = episode["steps"][step_idx]
            base = classify_eval_step(result, step.get("action") or {}, args.near_px, args.far_px, int(step.get("image_w") or 1040), int(step.get("image_h") or 736))
            if base["bucket"] == "correct":
                buckets["base_correct"].append(make_state(str(episode_id), episode, eval_episode, step_idx, base, "base_correct"))
    for label in buckets:
        buckets[label].sort(key=lambda row: (int(row["episode_id"]) if row["episode_id"].isdigit() else 0, row["step_idx"]))
    return buckets


def make_state(episode_id: str, episode: Dict[str, Any], eval_episode: Dict[str, Any], step_idx: int, base: Dict[str, Any], axis_label: str) -> Dict[str, Any]:
    step = episode["steps"][step_idx]
    return {
        "state_id": f"{episode_id}:{step_idx}",
        "episode_id": episode_id,
        "step_idx": step_idx,
        "axis_label": axis_label,
        "goal": episode.get("goal", ""),
        "screenshot": step.get("screenshot"),
        "image_w": int(step.get("image_w") or 1040),
        "image_h": int(step.get("image_h") or 736),
        "gt_action": step.get("action") or {},
        "history": build_history(eval_episode, step_idx),
        "features": state_features(episode, step_idx),
        "base": base,
    }


def select_states(states: Sequence[Dict[str, Any]], limit: int, seed: Optional[int]) -> List[Dict[str, Any]]:
    selected = list(states)
    if seed is not None:
        rng = random.Random(seed)
        rng.shuffle(selected)
    if limit:
        selected = selected[:limit]
    return selected


def build_step_source_messages(args: argparse.Namespace, state: Dict[str, Any], guidance: str) -> List[Dict[str, Any]]:
    return build_step_prompt(
        state["goal"],
        state["screenshot"],
        state["step_idx"],
        state["history"],
        guidance=guidance,
        image_max_pixels=args.image_max_pixels,
    )


def call_source(client: Any, args: argparse.Namespace, state: Dict[str, Any], guidance: str) -> Dict[str, Any]:
    messages = build_step_source_messages(args, state, guidance)
    response = client.chat.completions.create(
        model=args.model_name,
        messages=messages,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    pred_text = response.choices[0].message.content or ""
    return classify_action(pred_text, state["gt_action"], state["image_w"], state["image_h"], args.near_px, args.far_px, args.match_threshold)


def bootstrap_ci(values: Sequence[float], seed: int, samples: int = 10000) -> Tuple[float, float, float]:
    arr = np.array(values, dtype=np.float64)
    mean = float(arr.mean()) if len(arr) else 0.0
    if len(arr) == 0:
        return mean, 0.0, 0.0
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(arr), size=(samples, len(arr)))
    boot = arr[idx].mean(axis=1)
    return mean, float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def summarize_type_viability(rows: List[Dict[str, Any]], seed: int) -> Dict[str, Any]:
    n = len(rows)
    base_counts = Counter(row["base"]["bucket"] for row in rows)
    source_counts = Counter(row["type_source"]["bucket"] for row in rows)
    base_type = [1.0 if row["base"]["bucket"] == "type_mismatch" else 0.0 for row in rows]
    source_type = [1.0 if row["type_source"]["bucket"] == "type_mismatch" else 0.0 for row in rows]
    source_correct = [1.0 if row["type_source"]["bucket"] == "correct" else 0.0 for row in rows]
    reduction = [b - s for b, s in zip(base_type, source_type)]
    correct_gain = source_correct
    red_mean, red_lo, red_hi = bootstrap_ci(reduction, seed)
    corr_mean, corr_lo, corr_hi = bootstrap_ci(correct_gain, seed + 1)
    source_type_mean, source_type_lo, source_type_hi = bootstrap_ci(source_type, seed + 2)
    non_type_incorrect = sum(c for b, c in source_counts.items() if b not in {"type_mismatch", "correct"}) / max(n, 1)
    viable = red_lo > 0 and corr_mean > 0 and non_type_incorrect <= red_mean
    if viable:
        verdict = "VIABLE"
        consequent = "type-mismatch prompt source is individually viable; proceed to two-source Jaccard"
    else:
        verdict = "TYPE-SOURCE-NOT-VIABLE"
        consequent = "only far-miss source is viable under prompt bias; do not compute Jaccard unless escalating source construction"
    return {
        "n": n,
        "base_counts": dict(base_counts),
        "source_counts": dict(source_counts),
        "base_type_mismatch_rate": sum(base_type) / max(n, 1),
        "source_type_mismatch_rate": source_type_mean,
        "source_type_mismatch_rate_ci95": [source_type_lo, source_type_hi],
        "type_mismatch_reduction": red_mean,
        "type_mismatch_reduction_ci95": [red_lo, red_hi],
        "source_correct_rate": sum(source_correct) / max(n, 1),
        "correct_gain": corr_mean,
        "correct_gain_ci95": [corr_lo, corr_hi],
        "source_non_type_incorrect_rate": non_type_incorrect,
        "verdict": verdict,
        "consequent": consequent,
    }


def action_signature(result: Dict[str, Any]) -> Tuple[Any, ...]:
    action = result.get("pred_action") or {}
    atype = normalize_action_type(action.get("action") or result.get("pred_type"))
    coord = action_coord(action)
    text = str(action.get("text") or action.get("keys") or "").strip().lower()
    if coord:
        return atype, round(coord[0] / 20), round(coord[1] / 20), text
    return atype, None, None, text


def summarize_jaccard(rows: List[Dict[str, Any]], type_viability: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    a_errors = {row["state_id"] for row in rows if not row["far_source"]["success"]}
    b_errors = {row["state_id"] for row in rows if not row["type_source"]["success"]}
    union = a_errors | b_errors
    inter = a_errors & b_errors
    jaccard = len(inter) / max(len(union), 1)
    agreement = sum(1 for row in rows if action_signature(row["far_source"]) == action_signature(row["type_source"])) / max(len(rows), 1)
    coverage = Counter()
    axis = defaultdict(lambda: {"n": 0, "far_source_errors": 0, "type_source_errors": 0, "both_errors": 0, "far_source_correct": 0, "type_source_correct": 0})
    feature_by_source_error = defaultdict(list)
    for row in rows:
        a_ok = row["far_source"]["success"]
        b_ok = row["type_source"]["success"]
        if a_ok and b_ok:
            coverage["both_right"] += 1
        elif a_ok and not b_ok:
            coverage["only_far_source_right"] += 1
        elif b_ok and not a_ok:
            coverage["only_type_source_right"] += 1
        else:
            coverage["neither_right"] += 1
        label = row["axis_label"]
        axis[label]["n"] += 1
        axis[label]["far_source_errors"] += int(not a_ok)
        axis[label]["type_source_errors"] += int(not b_ok)
        axis[label]["both_errors"] += int((not a_ok) and (not b_ok))
        axis[label]["far_source_correct"] += int(a_ok)
        axis[label]["type_source_correct"] += int(b_ok)
        if not a_ok:
            feature_by_source_error["far_source_error_history_click_share"].append(row["features"].get("history_click_share", 0.0))
            feature_by_source_error["far_source_error_target_bbox_height_norm"].append(row["features"].get("target_bbox_height_norm", 0.0))
        if not b_ok:
            feature_by_source_error["type_source_error_history_click_share"].append(row["features"].get("history_click_share", 0.0))
            feature_by_source_error["type_source_error_target_bbox_height_norm"].append(row["features"].get("target_bbox_height_norm", 0.0))
    axis_rates = {}
    for label, data in axis.items():
        n = max(data["n"], 1)
        axis_rates[label] = {
            "n": data["n"],
            "far_source_error_rate": data["far_source_errors"] / n,
            "type_source_error_rate": data["type_source_errors"] / n,
            "both_error_rate": data["both_errors"] / n,
            "far_source_correct_rate": data["far_source_correct"] / n,
            "type_source_correct_rate": data["type_source_correct"] / n,
        }
    support_ok = all(axis_rates.get(label, {}).get("n", 0) >= args.min_axis_cell for label in ["far_miss", "type_mismatch", "base_correct"])
    predicted_axis_pass = False
    if support_ok:
        far_a = axis_rates["far_miss"]["far_source_error_rate"]
        type_a = axis_rates["type_mismatch"]["far_source_error_rate"]
        far_b = axis_rates["far_miss"]["type_source_error_rate"]
        type_b = axis_rates["type_mismatch"]["type_source_error_rate"]
        predicted_axis_pass = (type_a >= far_a + args.axis_margin) and (far_b >= type_b + args.axis_margin)
    fake_guard_pass = not (agreement >= args.high_agreement_threshold and jaccard <= args.jaccard_low_threshold)
    jaccard_low = jaccard <= args.jaccard_low_threshold
    if type_viability["verdict"] != "VIABLE":
        verdict = "TYPE-SOURCE-NOT-VIABLE"
        consequent = "type-mismatch source failed individual viability; no two-source claim"
    elif jaccard_low and predicted_axis_pass and fake_guard_pass and support_ok:
        verdict = "ORTHOGONAL"
        consequent = "Proposition 1 confirmed for prompt-biased sources; proceed to verifier only after review"
    else:
        verdict = "NOT ORTHOGONAL / FAKE"
        consequent = "prompt-biased sources do not show research-grade axis-predicted orthogonality; escalate to fine-tuned source before verifier"
    feature_means = {key: (float(np.mean(values)) if values else 0.0) for key, values in feature_by_source_error.items()}
    return {
        "n": len(rows),
        "error_jaccard": jaccard,
        "agreement_rate": agreement,
        "error_counts": {"far_source_errors": len(a_errors), "type_source_errors": len(b_errors), "intersection": len(inter), "union": len(union)},
        "unique_coverage": dict(coverage),
        "axis_cross_tab": axis_rates,
        "support_ok": support_ok,
        "jaccard_low_threshold": args.jaccard_low_threshold,
        "jaccard_low": jaccard_low,
        "predicted_axis_pass": predicted_axis_pass,
        "fake_diversity_guard_pass": fake_guard_pass,
        "feature_means_on_errors": feature_means,
        "verdict": verdict,
        "consequent": consequent,
    }


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def render_type_summary(args: argparse.Namespace, summary: Dict[str, Any], total_type_states: int) -> str:
    lines = [
        "# Phase 1b Type-Mismatch Source Viability",
        "",
        "## Gate Verdict",
        "",
        f"**{summary['verdict']}**",
        "",
        summary["consequent"],
        "",
        "## Construction Mechanism",
        "",
        "Section 1.1: prompt bias toward affordance + action-prior correction.",
        "",
        "The source uses the same frozen full-SFT model and adds guidance that forces verb/affordance judgment before coordinates. No fine-tune and no Jaccard in this viability-only mode.",
        "",
        "## Inputs",
        "",
        f"- total TEST type-mismatch states available: `{total_type_states}`",
        f"- evaluated states: `{summary['n']}`",
        f"- 200-slice mode: `{bool(args.limit and args.limit <= 200)}`",
        "",
        "## Type-Mismatch Rate",
        "",
        f"- base type-mismatch rate: `{summary['base_type_mismatch_rate']:.4f}`",
        f"- source type-mismatch rate: `{summary['source_type_mismatch_rate']:.4f}` (95% CI `{summary['source_type_mismatch_rate_ci95'][0]:.4f}` / `{summary['source_type_mismatch_rate_ci95'][1]:.4f}`)",
        f"- type-mismatch reduction: `{summary['type_mismatch_reduction']:.4f}` (95% CI `{summary['type_mismatch_reduction_ci95'][0]:.4f}` / `{summary['type_mismatch_reduction_ci95'][1]:.4f}`)",
        f"- source correctness: `{summary['source_correct_rate']:.4f}`",
        f"- correct gain: `{summary['correct_gain']:.4f}` (95% CI `{summary['correct_gain_ci95'][0]:.4f}` / `{summary['correct_gain_ci95'][1]:.4f}`)",
        "",
        "## Anti-Degradation Redistribution",
        "",
        "| bucket | base count | source count | source rate |",
        "|---|---:|---:|---:|",
    ]
    for bucket in sorted(set(summary["base_counts"]) | set(summary["source_counts"])):
        base = summary["base_counts"].get(bucket, 0)
        source = summary["source_counts"].get(bucket, 0)
        lines.append(f"| {bucket} | {base} | {source} | {source / max(summary['n'], 1):.4f} |")
    lines += ["", f"- source non-type incorrect rate: `{summary['source_non_type_incorrect_rate']:.4f}`", ""]
    return "\n".join(lines)


def render_jaccard_summary(args: argparse.Namespace, type_summary: Dict[str, Any], jaccard: Dict[str, Any], totals: Dict[str, int]) -> str:
    lines = [
        "# Phase 1b Type-Mismatch Source + Two-Source Error-Orthogonality",
        "",
        "## Gate Verdict",
        "",
        f"**{jaccard['verdict']}**",
        "",
        jaccard["consequent"],
        "",
        "## Type-Mismatch Source Viability",
        "",
        f"- verdict: `{type_summary['verdict']}`",
        f"- n: `{type_summary['n']}` / total TEST type states `{totals['type_mismatch']}`",
        f"- base type-mismatch rate: `{type_summary['base_type_mismatch_rate']:.4f}`",
        f"- source type-mismatch rate: `{type_summary['source_type_mismatch_rate']:.4f}`",
        f"- reduction: `{type_summary['type_mismatch_reduction']:.4f}` CI `{type_summary['type_mismatch_reduction_ci95'][0]:.4f}` / `{type_summary['type_mismatch_reduction_ci95'][1]:.4f}`",
        f"- source correct: `{type_summary['source_correct_rate']:.4f}`",
        f"- source non-type incorrect: `{type_summary['source_non_type_incorrect_rate']:.4f}`",
        "",
        "## Common Set",
        "",
        f"- far-miss states: `{totals['far_miss']}`",
        f"- type-mismatch states: `{totals['type_mismatch']}`",
        f"- base-correct states sampled: `{totals['base_correct']}`",
        f"- common set n: `{jaccard['n']}`",
        "",
        "## Error Jaccard + Fake-Diversity Guard",
        "",
        f"- error-Jaccard: `{jaccard['error_jaccard']:.4f}`",
        f"- agreement rate: `{jaccard['agreement_rate']:.4f}`",
        f"- Jaccard low threshold: `{jaccard['jaccard_low_threshold']:.4f}`",
        f"- Jaccard low: `{jaccard['jaccard_low']}`",
        f"- fake-diversity guard pass: `{jaccard['fake_diversity_guard_pass']}`",
        f"- error counts: `{jaccard['error_counts']}`",
        "",
        "## Unique Coverage",
        "",
        "| cell | count | share |",
        "|---|---:|---:|",
    ]
    for key, value in sorted(jaccard["unique_coverage"].items()):
        lines.append(f"| {key} | {value} | {value / max(jaccard['n'], 1):.4f} |")
    lines += [
        "",
        "## Predicted-Axis Cross-Tab",
        "",
        "| axis label | n | far-source error | type-source error | both error | far-source correct | type-source correct |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for label, row in sorted(jaccard["axis_cross_tab"].items()):
        lines.append(f"| {label} | {row['n']} | {row['far_source_error_rate']:.4f} | {row['type_source_error_rate']:.4f} | {row['both_error_rate']:.4f} | {row['far_source_correct_rate']:.4f} | {row['type_source_correct_rate']:.4f} |")
    lines += [
        "",
        f"- support ok: `{jaccard['support_ok']}`",
        f"- predicted-axis pass: `{jaccard['predicted_axis_pass']}`",
        "",
        "## Phase 0 Feature Tie-Back",
        "",
        "Feature means over source error sets:",
        "",
        "| feature/error set | mean |",
        "|---|---:|",
    ]
    for key, value in sorted(jaccard["feature_means_on_errors"].items()):
        lines.append(f"| {key} | {value:.4f} |")
    lines += ["", "No verifier, type-fine-tune, or gradient gate is built in this phase.", ""]
    return "\n".join(lines)


def run_type_viability(args: argparse.Namespace) -> None:
    from openai import OpenAI

    episodes = load_episodes(args.test_data)
    with open(args.base_eval_results) as handle:
        base_results = json.load(handle)
    buckets = build_base_states(args, episodes, base_results)
    total_type = len(buckets["type_mismatch"])
    states = select_states(buckets["type_mismatch"], args.limit, args.shuffle_seed)
    if len(states) < args.min_axis_cell:
        raise SystemExit(f"insufficient type-mismatch states: {len(states)}")
    client = OpenAI(base_url=args.api_url, api_key="dummy", timeout=args.request_timeout)
    rows = []
    for idx, state in enumerate(states, 1):
        row = dict(state)
        row["type_source"] = call_source(client, args, state, TYPE_GUIDANCE)
        rows.append(row)
        if args.log_every and idx % args.log_every == 0:
            print(f"type viability evaluated {idx}/{len(states)} states", flush=True)
    summary = summarize_type_viability(rows, args.bootstrap_seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "per_state.jsonl", rows)
    (output_dir / "summary.json").write_text(json.dumps({"summary": summary, "args": vars(args), "total_type_states": total_type}, ensure_ascii=False, indent=2) + "\n")
    (output_dir / "summary.md").write_text(render_type_summary(args, summary, total_type))
    print(f"Wrote {output_dir / 'summary.md'}")
    print(f"Wrote {output_dir / 'per_state.jsonl'}")
    print(f"GATE: {summary['verdict']} - {summary['consequent']}")


def run_jaccard(args: argparse.Namespace) -> None:
    from openai import OpenAI

    episodes = load_episodes(args.test_data)
    with open(args.base_eval_results) as handle:
        base_results = json.load(handle)
    buckets = build_base_states(args, episodes, base_results)
    far_states = select_states(buckets["far_miss"], args.limit_far, args.shuffle_seed)
    type_states = select_states(buckets["type_mismatch"], args.limit_type, args.shuffle_seed)
    correct_limit = args.correct_sample or max(len(far_states), len(type_states))
    correct_states = select_states(buckets["base_correct"], correct_limit, args.correct_seed)
    common = far_states + type_states + correct_states
    if min(len(far_states), len(type_states), len(correct_states)) < args.min_axis_cell:
        raise SystemExit(f"insufficient common-set support: far={len(far_states)} type={len(type_states)} correct={len(correct_states)}")
    client = OpenAI(base_url=args.api_url, api_key="dummy", timeout=args.request_timeout)
    rows = []
    for idx, state in enumerate(common, 1):
        row = dict(state)
        row["far_source"] = call_source(client, args, state, FAR_GUIDANCE)
        row["type_source"] = call_source(client, args, state, TYPE_GUIDANCE)
        rows.append(row)
        if args.log_every and idx % args.log_every == 0:
            print(f"jaccard evaluated {idx}/{len(common)} states", flush=True)
    type_rows = [row for row in rows if row["axis_label"] == "type_mismatch"]
    type_summary = summarize_type_viability(type_rows, args.bootstrap_seed)
    jaccard = summarize_jaccard(rows, type_summary, args)
    totals = {"far_miss": len(far_states), "type_mismatch": len(type_states), "base_correct": len(correct_states)}
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "per_state.jsonl", rows)
    (output_dir / "summary.json").write_text(json.dumps({"type_source": type_summary, "jaccard": jaccard, "totals": totals, "args": vars(args)}, ensure_ascii=False, indent=2) + "\n")
    (output_dir / "summary.md").write_text(render_jaccard_summary(args, type_summary, jaccard, totals))
    print(f"Wrote {output_dir / 'summary.md'}")
    print(f"Wrote {output_dir / 'per_state.jsonl'}")
    print(f"GATE: {jaccard['verdict']} - {jaccard['consequent']}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["type_viability", "jaccard"], required=True)
    parser.add_argument("--test_data", default="datasets/gui360-balanced/gui360_test_1000_balanced.jsonl")
    parser.add_argument("--base_eval_results", default="outputs/gui360_fullparam_sft_step250_balanced_8gpu64_stop_bounded/eval_results_20260624_153846.json")
    parser.add_argument("--output_dir", default="outputs/candidate_orthogonality/phase1b_jaccard")
    parser.add_argument("--api_url", default="http://localhost:8000/v1")
    parser.add_argument("--model_name", default="checkpoints/gui360-fullparam-sft-step250")
    parser.add_argument("--near_px", type=float, default=50.0)
    parser.add_argument("--far_px", type=float, default=150.0)
    parser.add_argument("--match_threshold", type=float, default=0.5)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--request_timeout", type=float, default=600.0)
    parser.add_argument("--image_max_pixels", type=int, default=None)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--limit_far", type=int, default=0)
    parser.add_argument("--limit_type", type=int, default=0)
    parser.add_argument("--correct_sample", type=int, default=0)
    parser.add_argument("--shuffle_seed", type=int, default=None)
    parser.add_argument("--correct_seed", type=int, default=33)
    parser.add_argument("--bootstrap_seed", type=int, default=23)
    parser.add_argument("--jaccard_low_threshold", type=float, default=0.60)
    parser.add_argument("--high_agreement_threshold", type=float, default=0.80)
    parser.add_argument("--axis_margin", type=float, default=0.05)
    parser.add_argument("--min_axis_cell", type=int, default=30)
    parser.add_argument("--log_every", type=int, default=25)
    args = parser.parse_args()

    if args.mode == "type_viability":
        run_type_viability(args)
    else:
        run_jaccard(args)


if __name__ == "__main__":
    main()
