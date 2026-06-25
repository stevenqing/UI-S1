#!/usr/bin/env python3
"""Failure-axis separability diagnostic for GUI-360.

Phase 0 for the candidate-orthogonality idea. Diagnostic only: no source
construction and no training. Test A uses existing first-error eval results;
Test B optionally resamples a frozen model through an OpenAI-compatible endpoint.
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


PRIMARY_LABELS = ("far_miss", "type_mismatch")
ACTION_TYPES = ("click", "type", "swipe")
INPUT_HINTS = ("edit", "document", "textbox", "text", "dataitem", "pane")
CLICK_HINTS = ("button", "menuitem", "tabitem", "hyperlink", "checkbox", "listitem")


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path) as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_episodes(path: str) -> Dict[str, Dict[str, Any]]:
    return {str(row["episode_id"]): row for row in read_jsonl(path)}


def normalize_action_type(value: Any) -> str:
    text = str(value or "").strip().lower()
    aliases = {
        "drag": "swipe",
        "scroll": "swipe",
        "wheel_mouse_input": "swipe",
        "input": "type",
        "left_click": "click",
        "tap": "click",
        "double_click": "click",
    }
    return aliases.get(text, text)


def first_bad_step(eval_episode: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    for step in eval_episode.get("steps", []):
        if not step.get("success"):
            return step
    return None


def coord_from_action(action: Optional[Dict[str, Any]]) -> Optional[Tuple[float, float]]:
    if not action:
        return None
    coord = action.get("coordinate") or action.get("start_coordinate") or action.get("startCoordinate")
    if coord is None:
        return None
    try:
        return float(coord[0]), float(coord[1])
    except (TypeError, ValueError, IndexError):
        return None


def coord_distance_px(pred_action: Optional[Dict[str, Any]], gt_action: Dict[str, Any]) -> Optional[float]:
    pred = coord_from_action(pred_action)
    gt = coord_from_action(gt_action)
    if pred is None or gt is None:
        return None
    return math.sqrt((pred[0] - gt[0]) ** 2 + (pred[1] - gt[1]) ** 2)


def classify_failure(
    pred_action: Optional[Dict[str, Any]],
    gt_action: Dict[str, Any],
    gt_type: Optional[str] = None,
    pred_type: Optional[str] = None,
    near_px: float = 50.0,
    far_px: float = 150.0,
) -> Tuple[str, Dict[str, Any]]:
    gt_type = normalize_action_type(gt_type or gt_action.get("action"))
    pred_type = normalize_action_type(pred_type or (pred_action or {}).get("action"))
    info = {"gt_type": gt_type, "pred_type": pred_type, "distance_px": None}
    if gt_type != pred_type:
        return "type_mismatch", info
    if gt_type != "click":
        return f"same_type_non_click:{gt_type}", info
    dist = coord_distance_px(pred_action, gt_action)
    info["distance_px"] = dist
    if dist is None:
        return "grounding_missing_coord", info
    if dist <= near_px:
        return "near_miss", info
    if dist >= far_px:
        return "far_miss", info
    return "mid_miss", info


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


def bbox_iou(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / max(area_a + area_b - inter, 1e-9)


def app_domain(episode: Dict[str, Any]) -> str:
    text = str(episode.get("goal", "")).lower()
    for step in episode.get("steps", []):
        text += " " + str(step.get("screenshot", "")).lower()
    if "excel" in text or "spreadsheet" in text or "cell" in text:
        return "excel"
    if "word" in text or "document" in text:
        return "word"
    if "ppt" in text or "powerpoint" in text or "slide" in text:
        return "ppt"
    if "browser" in text or "web" in text:
        return "browser"
    return "unknown"


def controls_from_step(step: Dict[str, Any]) -> List[Dict[str, Any]]:
    raw = step.get("control_infos") or step.get("a11y") or {}
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            raw = {}
    if not isinstance(raw, dict):
        return []
    controls = raw.get("merged_controls_info") or raw.get("controls") or []
    return [ctrl for ctrl in controls if isinstance(ctrl, dict)] if isinstance(controls, list) else []


def control_rect(ctrl: Dict[str, Any]) -> Optional[Tuple[float, float, float, float]]:
    return parse_bbox(ctrl.get("control_rect") or ctrl.get("bbox") or ctrl.get("bounds"))


def control_kind(ctrl: Dict[str, Any]) -> str:
    return str(ctrl.get("control_type") or ctrl.get("role") or "")


def control_text(ctrl: Dict[str, Any]) -> str:
    return str(ctrl.get("control_text") or ctrl.get("name") or ctrl.get("text") or "")


def find_target_control(controls: List[Dict[str, Any]], gt_action: Dict[str, Any], gt_bbox: Optional[Tuple[float, float, float, float]]) -> Optional[Dict[str, Any]]:
    coord = coord_from_action(gt_action)
    if coord:
        containing = []
        for ctrl in controls:
            rect = control_rect(ctrl)
            if rect and rect[0] <= coord[0] <= rect[2] and rect[1] <= coord[1] <= rect[3]:
                area = max(0.0, rect[2] - rect[0]) * max(0.0, rect[3] - rect[1])
                containing.append((area, ctrl))
        if containing:
            return sorted(containing, key=lambda item: item[0])[0][1]
    if gt_bbox:
        best_ctrl = None
        best_iou = 0.0
        for ctrl in controls:
            rect = control_rect(ctrl)
            if rect:
                overlap = bbox_iou(gt_bbox, rect)
                if overlap > best_iou:
                    best_iou = overlap
                    best_ctrl = ctrl
        return best_ctrl
    return None


def action_counts(steps: Sequence[Dict[str, Any]]) -> Counter:
    counts = Counter()
    for step in steps:
        counts[normalize_action_type((step.get("action") or {}).get("action"))] += 1
    return counts


def entropy(counts: Counter) -> float:
    total = sum(counts.values())
    if total == 0:
        return 0.0
    value = 0.0
    for count in counts.values():
        if count:
            p = count / total
            value -= p * math.log(p + 1e-12)
    return value


def extract_features(episode: Dict[str, Any], step_idx: int) -> Tuple[Dict[str, float], Dict[str, Any]]:
    steps = episode.get("steps", [])
    step = steps[step_idx]
    gt_action = step.get("action") or {}
    image_w = float(step.get("image_w") or 1040)
    image_h = float(step.get("image_h") or 736)
    bbox = parse_bbox(step.get("bbox"))
    controls = controls_from_step(step)
    target_control = find_target_control(controls, gt_action, bbox)
    domain = app_domain(episode)
    history = action_counts(steps[:step_idx])
    sequence = action_counts(steps)
    total_history = max(sum(history.values()), 1)
    total_sequence = max(sum(sequence.values()), 1)
    gt_type = normalize_action_type(gt_action.get("action"))

    features: Dict[str, float] = {
        "episode_num_steps": float(len(steps)),
        "step_idx_norm": step_idx / max(len(steps) - 1, 1),
        "history_len": float(step_idx),
        "history_action_entropy": entropy(history),
    }
    for name in ("excel", "word", "ppt", "browser", "unknown"):
        features[f"app_is_{name}"] = float(domain == name)
    for action_type in ACTION_TYPES:
        features[f"history_{action_type}_share"] = history[action_type] / total_history
        features[f"history_{action_type}_count"] = float(history[action_type])
        features[f"episode_{action_type}_share"] = sequence[action_type] / total_sequence
        features[f"sensitivity_gt_is_{action_type}"] = float(gt_type == action_type)
    gt_share = sequence[gt_type] / total_sequence if gt_type else 0.0
    shares = [sequence[action_type] / total_sequence for action_type in ACTION_TYPES]
    features["sensitivity_gt_action_share"] = gt_share
    features["sensitivity_gt_action_is_rarest_in_episode"] = float(gt_share <= min(shares) + 1e-12)

    if bbox:
        x1, y1, x2, y2 = bbox
        width, height = max(0.0, x2 - x1), max(0.0, y2 - y1)
        cx, cy = bbox_center(bbox)
        features.update({
            "target_bbox_area_norm": (width * height) / max(image_w * image_h, 1.0),
            "target_bbox_width_norm": width / max(image_w, 1.0),
            "target_bbox_height_norm": height / max(image_h, 1.0),
            "target_bbox_aspect_log": math.log((width + 1.0) / (height + 1.0)),
            "target_center_x_norm": cx / max(image_w, 1.0),
            "target_center_y_norm": cy / max(image_h, 1.0),
            "target_edge_distance_norm": min(cx, cy, image_w - cx, image_h - cy) / max(min(image_w, image_h), 1.0),
        })
    else:
        for key in ("target_bbox_area_norm", "target_bbox_width_norm", "target_bbox_height_norm", "target_bbox_aspect_log", "target_center_x_norm", "target_center_y_norm", "target_edge_distance_norm"):
            features[key] = 0.0

    similar_count = 0
    near_type_count = 0
    nearest_dist = 1.0
    if bbox:
        cx, cy = bbox_center(bbox)
        for idx, other in enumerate(steps):
            if idx == step_idx:
                continue
            other_bbox = parse_bbox(other.get("bbox"))
            if not other_bbox:
                continue
            ocx, ocy = bbox_center(other_bbox)
            dist = math.sqrt(((cx - ocx) / max(image_w, 1.0)) ** 2 + ((cy - ocy) / max(image_h, 1.0)) ** 2)
            nearest_dist = min(nearest_dist, dist)
            close = dist < 0.12 or bbox_iou(bbox, other_bbox) > 0.1
            if close:
                similar_count += 1
                if normalize_action_type((other.get("action") or {}).get("action")) == "type":
                    near_type_count += 1
    features["proxy_similar_target_bbox_count"] = float(similar_count)
    features["proxy_nearest_target_center_dist_norm"] = nearest_dist
    features["proxy_near_type_bbox_count"] = float(near_type_count)

    input_like = click_like = same_type = same_text = near_input = near_click = 0
    target_kind = control_kind(target_control).lower() if target_control else ""
    target_text = control_text(target_control).strip().lower() if target_control else ""
    target_rect = control_rect(target_control) if target_control else None
    for ctrl in controls:
        kind = control_kind(ctrl).lower()
        text = control_text(ctrl).strip().lower()
        is_input = any(hint in kind for hint in INPUT_HINTS)
        is_click = any(hint in kind for hint in CLICK_HINTS)
        input_like += int(is_input)
        click_like += int(is_click)
        same_type += int(bool(target_kind) and kind == target_kind)
        same_text += int(bool(target_text) and text == target_text)
        rect = control_rect(ctrl)
        if target_rect and rect:
            tx, ty = bbox_center(target_rect)
            rx, ry = bbox_center(rect)
            dist = math.sqrt(((tx - rx) / max(image_w, 1.0)) ** 2 + ((ty - ry) / max(image_h, 1.0)) ** 2)
            if dist < 0.12:
                near_input += int(is_input)
                near_click += int(is_click)
    features.update({
        "a11y_total_controls": float(len(controls)),
        "a11y_input_like_controls": float(input_like),
        "a11y_click_like_controls": float(click_like),
        "a11y_same_control_type_count": float(same_type),
        "a11y_same_text_count": float(same_text),
        "a11y_nearby_input_controls": float(near_input),
        "a11y_nearby_click_controls": float(near_click),
        "a11y_affordance_ambiguity": float(near_input > 0 and near_click > 0),
        "a11y_target_text_len": float(len(target_text)),
    })
    meta = {
        "goal": episode.get("goal", ""),
        "screenshot": step.get("screenshot", ""),
        "gt_action": gt_action,
        "gt_type": gt_type,
        "app_domain": domain,
        "a11y_controls_available": bool(controls),
        "target_control_type": target_kind,
        "target_control_text": target_text[:120],
        "feature_source": "a11y_control_infos" if controls else "available_train_state_proxy",
    }
    return features, meta


def build_records(episodes: Dict[str, Dict[str, Any]], eval_results: Dict[str, Any], near_px: float, far_px: float) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    counts = Counter()
    for episode_id, eval_episode in eval_results.items():
        if eval_episode.get("task_success"):
            continue
        episode = episodes.get(str(episode_id))
        bad = first_bad_step(eval_episode)
        if episode is None or bad is None:
            continue
        step_idx = int(bad.get("step_idx", 0))
        if step_idx >= len(episode.get("steps", [])):
            continue
        gt_action = episode["steps"][step_idx].get("action", {}) or {}
        pred_action = bad.get("pred_action") or parse_action_from_text(bad.get("pred_text", "") or "")
        failure_type, info = classify_failure(pred_action, gt_action, bad.get("gt_type"), bad.get("pred_type"), near_px, far_px)
        counts[failure_type] += 1
        features, meta = extract_features(episode, step_idx)
        meta.update({"pred_action": pred_action, "failure_info": info, "eval_progress": eval_episode.get("progress")})
        records.append({"episode_id": str(episode_id), "step_idx": step_idx, "observed_failure_type": failure_type, "features": features, "meta": meta, "resampling": None})
    return records, {
        "num_failed_records": len(records),
        "num_primary_records": sum(1 for row in records if row["observed_failure_type"] in PRIMARY_LABELS),
        "failure_counts": dict(counts),
        "a11y_controls_available_records": sum(1 for row in records if row["meta"].get("a11y_controls_available")),
    }


def feature_group(name: str) -> str:
    if name.startswith("sensitivity_"):
        return "sensitivity_gt_action"
    if name.startswith("a11y_same") or name.startswith("a11y_total") or name.startswith("proxy_similar") or name.startswith("proxy_nearest"):
        return "mechanism_far_disambiguation"
    if name.startswith("a11y_nearby") or name.startswith("a11y_affordance") or name.startswith("proxy_near_type") or name.startswith("history_"):
        return "mechanism_type_affordance_prior"
    if name.startswith("episode_") or name in {"history_len", "step_idx_norm", "target_bbox_area_norm", "target_bbox_width_norm", "target_bbox_height_norm"}:
        return "difficulty"
    if name.startswith("app_is"):
        return "app_context"
    if name.startswith("target_"):
        return "spatial_target"
    if name.startswith("a11y_"):
        return "a11y_other"
    return "other"


def feature_matrix(records: Sequence[Dict[str, Any]], include_sensitivity: bool) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    rows = [row for row in records if row["observed_failure_type"] in PRIMARY_LABELS]
    names = sorted({name for row in rows for name in row["features"]})
    if not include_sensitivity:
        names = [name for name in names if not name.startswith("sensitivity_")]
    X = np.array([[float(row["features"].get(name, 0.0)) for name in names] for row in rows], dtype=np.float64)
    y = np.array([1.0 if row["observed_failure_type"] == "type_mismatch" else 0.0 for row in rows], dtype=np.float64)
    return X, y, names


def standardize(X_train: np.ndarray, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std[std < 1e-8] = 1.0
    return (X_train - mean) / std, (X_test - mean) / std


def fit_logistic(X: np.ndarray, y: np.ndarray, steps: int = 2500, lr: float = 0.08, l2: float = 0.01) -> np.ndarray:
    Xb = np.concatenate([np.ones((X.shape[0], 1)), X], axis=1)
    weights = np.zeros(Xb.shape[1], dtype=np.float64)
    pos = max(float(y.sum()), 1.0)
    neg = max(float(len(y) - y.sum()), 1.0)
    sample_weights = np.where(y > 0.5, len(y) / (2.0 * pos), len(y) / (2.0 * neg))
    for _ in range(steps):
        logits = np.clip(Xb @ weights, -40.0, 40.0)
        probs = 1.0 / (1.0 + np.exp(-logits))
        grad = (Xb.T @ ((probs - y) * sample_weights)) / len(y)
        grad[1:] += l2 * weights[1:]
        weights -= lr * grad
    return weights


def predict(weights: np.ndarray, X: np.ndarray) -> np.ndarray:
    Xb = np.concatenate([np.ones((X.shape[0], 1)), X], axis=1)
    return 1.0 / (1.0 + np.exp(-np.clip(Xb @ weights, -40.0, 40.0)))


def stratified_folds(y: np.ndarray, n_folds: int, seed: int) -> List[List[int]]:
    rng = random.Random(seed)
    buckets: Dict[int, List[int]] = defaultdict(list)
    for idx, value in enumerate(y):
        buckets[int(value)].append(idx)
    folds = [[] for _ in range(n_folds)]
    for bucket in buckets.values():
        rng.shuffle(bucket)
        for pos, idx in enumerate(bucket):
            folds[pos % n_folds].append(idx)
    return folds


def metrics(y: np.ndarray, prob: np.ndarray) -> Dict[str, Any]:
    pred = (prob >= 0.5).astype(np.float64)
    tp = int(((pred == 1) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tpr = tp / max(tp + fn, 1)
    tnr = tn / max(tn + fp, 1)
    return {"accuracy": float((pred == y).mean()), "balanced_accuracy": (tpr + tnr) / 2.0, "confusion": {"tp_type": tp, "tn_far": tn, "fp_far_as_type": fp, "fn_type_as_far": fn}}


def run_test_a(records: Sequence[Dict[str, Any]], include_sensitivity: bool, seed: int, folds: int) -> Dict[str, Any]:
    X, y, names = feature_matrix(records, include_sensitivity)
    if len(y) < 30 or len(set(y.tolist())) < 2:
        return {"status": "PENDING", "reason": "insufficient far/type support", "n": int(len(y))}
    min_class = int(np.bincount(y.astype(int)).min())
    folds = max(2, min(folds, min_class))
    probs = np.zeros_like(y)
    for test_indices in stratified_folds(y, folds, seed):
        test_idx = np.array(sorted(test_indices), dtype=int)
        train_idx = np.array([idx for idx in range(len(y)) if idx not in set(test_indices)], dtype=int)
        X_train, X_test = standardize(X[train_idx], X[test_idx])
        weights = fit_logistic(X_train, y[train_idx])
        probs[test_idx] = predict(weights, X_test)
    result_metrics = metrics(y, probs)
    prior = max(float(y.mean()), 1.0 - float(y.mean()))

    X_std, _ = standardize(X, X)
    weights = fit_logistic(X_std, y)
    top_features = []
    for name, coef in sorted(zip(names, weights[1:]), key=lambda item: abs(item[1]), reverse=True):
        top_features.append({"feature": name, "coefficient_standardized": float(coef), "direction": "type_mismatch" if coef > 0 else "far_miss", "group": feature_group(name)})
    top10 = top_features[:10]
    top10_abs = sum(abs(item["coefficient_standardized"]) for item in top10) or 1.0
    difficulty_share = sum(abs(item["coefficient_standardized"]) for item in top10 if item["group"] == "difficulty") / top10_abs
    mechanism_groups = {item["group"] for item in top10 if item["group"].startswith("mechanism_")}
    passes_predictability = result_metrics["accuracy"] >= prior + 0.03 and result_metrics["balanced_accuracy"] >= 0.53
    passes_distinct = len(mechanism_groups) >= 2 and difficulty_share < 0.5
    return {
        "status": "DONE",
        "n": int(len(y)),
        "class_counts": {"far_miss": int((y == 0).sum()), "type_mismatch": int((y == 1).sum())},
        "class_prior_accuracy": prior,
        "metrics": result_metrics,
        "accuracy_margin_vs_prior": result_metrics["accuracy"] - prior,
        "balanced_accuracy_margin_vs_chance": result_metrics["balanced_accuracy"] - 0.5,
        "top_features": top_features[:20],
        "difficulty_top10_abs_share": difficulty_share,
        "passes_predictability": bool(passes_predictability),
        "passes_feature_distinctness": bool(passes_distinct),
        "passes_difficulty_confound_check": bool(passes_distinct),
        "include_sensitivity_features": include_sensitivity,
    }


def summarize(values: Sequence[Optional[float]]) -> Dict[str, float]:
    vals = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not vals:
        return {"n": 0, "mean": 0.0, "std": 0.0, "median": 0.0, "p25": 0.0, "p75": 0.0, "min": 0.0, "max": 0.0}
    arr = np.array(vals, dtype=np.float64)
    return {"n": int(len(arr)), "mean": float(arr.mean()), "std": float(arr.std()), "median": float(np.median(arr)), "p25": float(np.percentile(arr, 25)), "p75": float(np.percentile(arr, 75)), "min": float(arr.min()), "max": float(arr.max())}


def select_resample_records(records: Sequence[Dict[str, Any]], per_class: int, seed: int) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    selected: List[Dict[str, Any]] = []
    for label in PRIMARY_LABELS:
        bucket = [row for row in records if row["observed_failure_type"] == label]
        rng.shuffle(bucket)
        selected.extend(bucket[:per_class])
    rng.shuffle(selected)
    return selected


def run_test_b(records: List[Dict[str, Any]], episodes: Dict[str, Dict[str, Any]], eval_results: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    from openai import OpenAI

    client = OpenAI(base_url=args.api_url, api_key="dummy", timeout=args.request_timeout)
    selected = select_resample_records(records, args.resample_per_class, args.seed)
    selected_keys = {(row["episode_id"], row["step_idx"]) for row in selected}
    for idx, row in enumerate(selected, 1):
        episode = episodes[row["episode_id"]]
        eval_episode = eval_results[row["episode_id"]]
        step_idx = row["step_idx"]
        history = []
        for prior in eval_episode.get("steps", [])[:step_idx]:
            history.append(_format_action_for_history(prior.get("pred_action"), int(prior.get("step_idx", len(history))) + 1))
        step = episode["steps"][step_idx]
        messages = build_step_prompt(episode.get("goal", ""), step["screenshot"], step_idx, history, image_max_pixels=args.image_max_pixels)
        counts = Counter()
        samples = []
        for sample_idx in range(args.resample_k):
            try:
                response = client.chat.completions.create(model=args.model_name, messages=messages, max_tokens=args.max_tokens, temperature=args.temperature, top_p=args.top_p)
                pred_text = response.choices[0].message.content or ""
            except Exception as exc:
                counts["api_error"] += 1
                samples.append({"sample_idx": sample_idx, "error": str(exc)[:240]})
                continue
            pred_action = parse_tool_call(pred_text) or parse_action_from_text(pred_text)
            fake_text = f"<action>{json.dumps(pred_action)}</action>" if pred_action else pred_text
            reward, info = compute_step_reward(fake_text, step.get("action", {}) or {}, image_w=step.get("image_w", 1040), image_h=step.get("image_h", 736))
            if reward >= args.match_threshold:
                counts["correct"] += 1
                failure_type = "correct"
            else:
                failure_type, _ = classify_failure(pred_action, step.get("action", {}) or {}, info.get("gt_type"), info.get("pred_type"), args.near_px, args.far_px)
                counts[failure_type] += 1
            samples.append({"sample_idx": sample_idx, "failure_type": failure_type, "reward": reward, "pred_type": info.get("pred_type"), "gt_type": info.get("gt_type"), "pred_text": pred_text[:300]})
        wrong_total = sum(v for k, v in counts.items() if k != "correct")
        consistency = counts[row["observed_failure_type"]] / wrong_total if wrong_total else None
        row["resampling"] = {"counts": dict(counts), "wrong_total": wrong_total, "consistency": consistency, "samples": samples}
        if args.log_every and idx % args.log_every == 0:
            print(f"resampled {idx}/{len(selected)} states", flush=True)
    for row in records:
        if (row["episode_id"], row["step_idx"]) not in selected_keys and row.get("resampling") is None:
            row["resampling"] = None
    consistencies = [row["resampling"]["consistency"] for row in selected if row.get("resampling") and row["resampling"]["consistency"] is not None]
    by_type = {label: summarize([row["resampling"]["consistency"] for row in selected if row["observed_failure_type"] == label and row.get("resampling")]) for label in PRIMARY_LABELS}
    return {"status": "DONE", "sampled_states": len(selected), "resample_k": args.resample_k, "wrong_conditioned_states": len(consistencies), "support_ok": len(selected) >= args.min_resample_states and args.resample_k >= args.min_resample_k, "consistency": summarize(consistencies), "consistency_by_original_type": by_type}


def decide_gate(test_a: Dict[str, Any], test_b: Dict[str, Any], controls_available: int) -> Dict[str, Any]:
    a_pass = None if test_a.get("status") != "DONE" else bool(test_a.get("passes_predictability") and test_a.get("passes_feature_distinctness") and test_a.get("passes_difficulty_confound_check"))
    if test_b.get("status") == "SKIPPED":
        b_pass = None
    else:
        cons = test_b.get("consistency", {})
        b_pass = bool(test_b.get("support_ok") and cons.get("n", 0) >= 30 and cons.get("median", 0.0) >= 0.70 and cons.get("mean", 0.0) >= 0.65)
    if a_pass is True and b_pass is True:
        verdict = "SEPARABLE"
        consequent = "foundation holds; proceed to Phase 1 only after review"
    elif a_pass is False or b_pass is False:
        verdict = "NOT SEPARABLE"
        consequent = "ability-axis foundation fails under the pre-registered gate; prefer single-agent/verifier abstention"
    else:
        verdict = "MIXED"
        consequent = "do not build sources; complete missing or pending evidence first"
    return {"verdict": verdict, "test_a_pass": a_pass, "test_b_pass": b_pass, "a11y_status": "CONTROL_INFOS_AVAILABLE" if controls_available else "NO_CONTROL_INFOS_IN_INPUT", "consequent": consequent}


def pct(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def render_summary(payload: Dict[str, Any]) -> str:
    dataset = payload["dataset"]
    test_a = payload["test_a"]
    sensitivity = payload["test_a_sensitivity_with_gt_action_features"]
    test_b = payload["test_b"]
    gate = payload["gate"]
    inputs = payload["inputs"]
    lines = [
        "# Failure-Axis Separability Phase 0 (GUI-360)",
        "",
        "## Gate Verdict",
        "",
        f"**{gate['verdict']}**",
        "",
        gate["consequent"],
        "",
        "## Inputs",
        "",
        f"- split label: `{inputs['split_label']}`",
        f"- episode data: `{inputs['episode_data']}`",
        f"- eval results: `{inputs['eval_results']}`",
        f"- a11y data: `{inputs.get('a11y_data') or ''}`",
        f"- failed states: `{dataset['num_failed_records']}`",
        f"- far/type states: `{dataset['num_primary_records']}`",
        f"- records with explicit `control_infos`: `{dataset['a11y_controls_available_records']}`",
        "",
        "## Failure Support",
        "",
        "| bucket | count |",
        "|---|---:|",
    ]
    for key, value in sorted(dataset["failure_counts"].items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"| {key} | {value} |")
    lines += ["", "## Test A - Feature Separability", ""]
    if test_a.get("status") == "DONE":
        m = test_a["metrics"]
        lines += [
            f"- n: `{test_a['n']}`",
            f"- class counts: `{test_a['class_counts']}`",
            f"- accuracy: `{pct(m['accuracy'])}`",
            f"- class-prior baseline: `{pct(test_a['class_prior_accuracy'])}`",
            f"- accuracy margin vs prior: `{pct(test_a['accuracy_margin_vs_prior'])}`",
            f"- balanced accuracy: `{pct(m['balanced_accuracy'])}`",
            f"- balanced margin vs chance: `{pct(test_a['balanced_accuracy_margin_vs_chance'])}`",
            f"- difficulty top-10 coefficient share: `{test_a['difficulty_top10_abs_share']:.3f}`",
            f"- passes predictability: `{test_a['passes_predictability']}`",
            f"- passes feature distinctness: `{test_a['passes_feature_distinctness']}`",
            f"- confusion: `{m['confusion']}`",
            "",
            "| rank | feature | direction | group | standardized coef |",
            "|---:|---|---|---|---:|",
        ]
        for rank, item in enumerate(test_a["top_features"][:12], 1):
            lines.append(f"| {rank} | `{item['feature']}` | {item['direction']} | {item['group']} | {item['coefficient_standardized']:.4f} |")
    else:
        lines.append(f"Status: `{test_a.get('status')}` - {test_a.get('reason', '')}")
    lines += ["", "### Sensitivity With GT-Action Features", ""]
    if sensitivity.get("status") == "DONE":
        lines += ["Not part of the primary anti-leakage Test A.", f"- accuracy: `{pct(sensitivity['metrics']['accuracy'])}`", f"- balanced accuracy: `{pct(sensitivity['metrics']['balanced_accuracy'])}`"]
    else:
        lines.append(f"Status: `{sensitivity.get('status')}`")
    lines += ["", "## Test B - Resampling Consistency", ""]
    if test_b.get("status") == "DONE":
        c = test_b["consistency"]
        lines += [f"- sampled states: `{test_b['sampled_states']}`", f"- k per state: `{test_b['resample_k']}`", f"- wrong-conditioned states: `{test_b['wrong_conditioned_states']}`", f"- support ok: `{test_b['support_ok']}`", f"- mean/median consistency: `{c['mean']:.3f}` / `{c['median']:.3f}`", f"- p25/p75 consistency: `{c['p25']:.3f}` / `{c['p75']:.3f}`", "", "| original type | n | mean | median | p25 | p75 |", "|---|---:|---:|---:|---:|---:|"]
        for label, stats in test_b["consistency_by_original_type"].items():
            lines.append(f"| {label} | {stats['n']} | {stats['mean']:.3f} | {stats['median']:.3f} | {stats['p25']:.3f} | {stats['p75']:.3f} |")
    else:
        lines.append("Status: `SKIPPED`. Run with `--run_resampling` against frozen SFT vLLM to complete Test B.")
    lines += ["", "## Decision Rule", "", "SEPARABLE requires Test A PASS and Test B PASS. NOT SEPARABLE if either pre-registered foundation test fails. MIXED/PENDING means stop before source construction.", "", "## Notes", "", "Primary Test A excludes `sensitivity_gt_*` features to avoid circularly leaking the current GT action type.", "If explicit `control_infos` are absent, the diagnostic uses available train-state proxy features and reports `NO_CONTROL_INFOS_IN_INPUT`.", ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episode_data", default="datasets/gui360-balanced/gui360_train_from_parquet.jsonl")
    parser.add_argument("--eval_results", default="outputs/v23_visual_transition/train_eval_full_sft_8gpu64_stop/eval_results_20260625_043340.json")
    parser.add_argument("--a11y_data", default="datasets/gui360-balanced-a11y")
    parser.add_argument("--output_dir", default="outputs/failure_axis_separability")
    parser.add_argument("--split_label", default="train")
    parser.add_argument("--near_px", type=float, default=50.0)
    parser.add_argument("--far_px", type=float, default=150.0)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--run_resampling", action="store_true")
    parser.add_argument("--api_url", default="http://localhost:8000/v1")
    parser.add_argument("--model_name", default="checkpoints/gui360-fullparam-sft-step250")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--resample_k", type=int, default=8)
    parser.add_argument("--resample_per_class", type=int, default=50)
    parser.add_argument("--min_resample_states", type=int, default=100)
    parser.add_argument("--min_resample_k", type=int, default=8)
    parser.add_argument("--match_threshold", type=float, default=0.5)
    parser.add_argument("--image_max_pixels", type=int, default=None)
    parser.add_argument("--max_tokens", type=int, default=256)
    parser.add_argument("--request_timeout", type=float, default=600.0)
    parser.add_argument("--log_every", type=int, default=10)
    args = parser.parse_args()

    episodes = load_episodes(args.episode_data)
    with open(args.eval_results) as handle:
        eval_results = json.load(handle)
    records, dataset_summary = build_records(episodes, eval_results, args.near_px, args.far_px)
    test_a = run_test_a(records, include_sensitivity=False, seed=args.seed, folds=args.folds)
    sensitivity = run_test_a(records, include_sensitivity=True, seed=args.seed, folds=args.folds)
    test_b = run_test_b(records, episodes, eval_results, args) if args.run_resampling else {"status": "SKIPPED"}
    gate = decide_gate(test_a, test_b, dataset_summary["a11y_controls_available_records"])
    payload = {"inputs": vars(args), "dataset": dataset_summary, "test_a": test_a, "test_a_sensitivity_with_gt_action_features": sensitivity, "test_b": test_b, "gate": gate}
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    (output_dir / "summary.md").write_text(render_summary(payload))
    with (output_dir / "per_state.jsonl").open("w") as handle:
        for row in records:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {output_dir / 'summary.md'}")
    print(f"Wrote {output_dir / 'per_state.jsonl'}")
    print(f"GATE: {gate['verdict']} - {gate['consequent']}")


if __name__ == "__main__":
    main()
