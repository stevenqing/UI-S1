#!/usr/bin/env python3
"""Inference-time identifiability diagnostic for GUI-360 critical steps.

Critical labels come from held-out bottom-2 p_i. Features are restricted to
inference-available signals: sampled candidate disagreement/logprobs and screen
UIA complexity. Matcher rewards, correctness flags, GT actions, and p_i are not
used as features.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import minimize
from scipy.stats import rankdata

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.critstep_reward_structure_uia import (  # noqa: E402
    action_point,
    bbox_center,
    control_rect,
    control_text,
    control_type,
    controls_for_step,
    distance,
    rect_center,
)


DEFAULT_CANDIDATES = "outputs/verifier_e2e/slice200/candidates/per_step.jsonl"
DEFAULT_TEST_DATA = "outputs/gui360_history_ab/original_eval/gui360_test_1000_balanced_uia.jsonl"
DEFAULT_CRIT_TASKS = "outputs/critstep_eval/per_task.jsonl"
DEFAULT_OUTPUT_DIR = "outputs/critstep_identify"
BUDGETS = (0.10, 0.20, 0.30)
DISAGREEMENT_FEATURES = {
    "distinct_action_count",
    "distinct_action_type_count",
    "distinct_control_count",
    "action_entropy",
    "type_entropy",
    "control_entropy",
    "point_radius_mean",
    "point_radius_max",
    "point_x_std",
    "point_y_std",
    "one_minus_modal_action_frac",
    "one_minus_modal_control_frac",
}
LEAKY_NAME_PARTS = ("gt_", "reward", "success", "correct", "p_hat", "heldout", "label")


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def safe_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        value = float(value)
        return value if math.isfinite(value) else None
    except (TypeError, ValueError):
        return None


def entropy(values: Sequence[Any]) -> float:
    vals = [str(value) for value in values if value not in (None, "")]
    if not vals:
        return 0.0
    counts = Counter(vals)
    total = len(vals)
    return -sum((count / total) * math.log(count / total, 2) for count in counts.values())


def modal_fraction(values: Sequence[Any]) -> float:
    vals = [str(value) for value in values if value not in (None, "")]
    if not vals:
        return 0.0
    return Counter(vals).most_common(1)[0][1] / len(vals)


def std(values: Sequence[float]) -> float:
    vals = [float(value) for value in values if math.isfinite(float(value))]
    if len(vals) <= 1:
        return 0.0
    return float(np.std(np.asarray(vals, dtype=float)))


def top2_gap(values: Sequence[float]) -> float:
    vals = sorted([float(value) for value in values if math.isfinite(float(value))], reverse=True)
    if len(vals) < 2:
        return 0.0
    return vals[0] - vals[1]


def text_similarity(a: str, b: str) -> float:
    a = " ".join(str(a or "").lower().split())
    b = " ".join(str(b or "").lower().split())
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def read_episode_index(path: Path) -> Dict[str, Dict[str, Any]]:
    return {str(row["episode_id"]): row for row in read_jsonl(path)}


def read_task_index(path: Path) -> Dict[str, Dict[str, Any]]:
    return {str(row["episode_id"]): row for row in read_jsonl(path)}


def point_stats(points: Sequence[Tuple[float, float]]) -> Dict[str, float]:
    if not points:
        return {"point_x_std": 0.0, "point_y_std": 0.0, "point_radius_mean": 0.0, "point_radius_max": 0.0}
    xs = np.asarray([point[0] for point in points], dtype=float)
    ys = np.asarray([point[1] for point in points], dtype=float)
    cx = float(np.mean(xs))
    cy = float(np.mean(ys))
    radii = [math.hypot(float(x) - cx, float(y) - cy) for x, y in zip(xs, ys)]
    return {
        "point_x_std": float(np.std(xs)) if len(xs) > 1 else 0.0,
        "point_y_std": float(np.std(ys)) if len(ys) > 1 else 0.0,
        "point_radius_mean": float(np.mean(radii)) if radii else 0.0,
        "point_radius_max": max(radii) if radii else 0.0,
    }


def controls_near_point(point: Optional[Tuple[float, float]], controls: Sequence[Mapping[str, Any]], radius: float) -> int:
    if point is None:
        return 0
    count = 0
    for control in controls:
        center = rect_center(control if isinstance(control, dict) else None)
        dist = distance(point, center)
        if dist is not None and dist <= radius:
            count += 1
    return count


def screen_text_similarity_features(controls: Sequence[Mapping[str, Any]], greedy_control: Mapping[str, Any]) -> Dict[str, float]:
    texts = [control_text(control if isinstance(control, dict) else None) for control in controls]
    texts = [text for text in texts if text]
    n_text = len(texts)
    if n_text <= 1:
        max_pair_similarity = 0.0
        similar_pair_fraction = 0.0
    else:
        capped = texts[:80]
        pairs = 0
        similar = 0
        max_pair_similarity = 0.0
        for idx, left in enumerate(capped):
            for right in capped[idx + 1 :]:
                sim = text_similarity(left, right)
                max_pair_similarity = max(max_pair_similarity, sim)
                similar += int(sim >= 0.75)
                pairs += 1
        similar_pair_fraction = similar / pairs if pairs else 0.0
    greedy_text = str(greedy_control.get("text") or "") if isinstance(greedy_control, dict) else ""
    similar_to_greedy = 0
    if greedy_text:
        similar_to_greedy = sum(1 for text in texts if text_similarity(greedy_text, text) >= 0.70)
    return {
        "screen_n_text_controls": float(n_text),
        "screen_distinct_text_fraction": len(set(texts)) / n_text if n_text else 0.0,
        "screen_max_text_multiplicity": float(max(Counter(texts).values())) if texts else 0.0,
        "screen_max_pair_text_similarity": max_pair_similarity,
        "screen_similar_pair_fraction": similar_pair_fraction,
        "greedy_control_similar_text_count": float(similar_to_greedy),
    }


def verifier_score_values(candidate: Mapping[str, Any]) -> List[float]:
    values: List[float] = []
    for key, value in candidate.items():
        key_text = str(key)
        if key_text == "verifier_score" or key_text.startswith("stage1_score") or key_text.startswith("stage2_score"):
            num = safe_float(value)
            if num is not None:
                values.append(num)
    return values


def extract_features(row: Mapping[str, Any], episode: Mapping[str, Any]) -> Dict[str, float]:
    candidates = row.get("candidates") if isinstance(row.get("candidates"), list) else []
    step_idx = int(row.get("step_idx") or 0)
    steps = episode.get("steps") if isinstance(episode.get("steps"), list) else []
    step = steps[step_idx] if step_idx < len(steps) else {}
    controls = controls_for_step(step if isinstance(step, dict) else {})
    image_w = float(row.get("image_w") or step.get("image_w") or 1040)
    image_h = float(row.get("image_h") or step.get("image_h") or 736)
    area = max(1.0, image_w * image_h)

    actions = [candidate.get("action_signature") or json.dumps(candidate.get("action") or {}, sort_keys=True) for candidate in candidates]
    types = [candidate.get("pred_type") or (candidate.get("action") or {}).get("action") for candidate in candidates]
    controls_keys = []
    control_assignments = []
    points = []
    logprob_avg = []
    logprob_sum = []
    logprob_tokens = []
    text_lengths = []
    verifier_scores = []
    for candidate in candidates:
        control = candidate.get("control") if isinstance(candidate.get("control"), dict) else {}
        controls_keys.append(control.get("key") or "NO_CONTROL")
        control_assignments.append(control.get("assignment") or "unknown")
        point = action_point(candidate.get("action") if isinstance(candidate.get("action"), dict) else None)
        if point is not None:
            points.append(point)
        avg = safe_float(candidate.get("model_logprob_avg"))
        if avg is not None:
            logprob_avg.append(avg)
        total = safe_float(candidate.get("model_logprob_sum"))
        if total is not None:
            logprob_sum.append(total)
        tokens = safe_float(candidate.get("model_logprob_tokens"))
        if tokens is not None and tokens > 0:
            logprob_tokens.append(tokens)
        text = candidate.get("pred_text")
        if isinstance(text, str):
            text_lengths.append(float(len(text)))
        verifier_scores.extend(verifier_score_values(candidate))

    greedy = candidates[0] if candidates else {}
    greedy_action = greedy.get("action") if isinstance(greedy.get("action"), dict) else None
    greedy_point = action_point(greedy_action)
    greedy_control = greedy.get("control") if isinstance(greedy.get("control"), dict) else {}
    type_counts = Counter(str(item) for item in types if item)
    assignment_counts = Counter(str(item) for item in control_assignments if item)
    features: Dict[str, float] = {
        "n_candidates": float(len(candidates)),
        "distinct_action_count": float(len(set(actions))),
        "distinct_action_type_count": float(len(set(str(item) for item in types if item))),
        "distinct_control_count": float(len(set(key for key in controls_keys if key != "NO_CONTROL"))),
        "action_entropy": entropy(actions),
        "type_entropy": entropy(types),
        "control_entropy": entropy(controls_keys),
        "modal_action_frac": modal_fraction(actions),
        "modal_type_frac": modal_fraction(types),
        "modal_control_frac": modal_fraction(controls_keys),
        "one_minus_modal_action_frac": 1.0 - modal_fraction(actions),
        "one_minus_modal_type_frac": 1.0 - modal_fraction(types),
        "one_minus_modal_control_frac": 1.0 - modal_fraction(controls_keys),
        "no_control_frac": sum(1 for key in controls_keys if key == "NO_CONTROL") / len(candidates) if candidates else 0.0,
        "contains_assignment_frac": assignment_counts.get("contains", 0) / len(candidates) if candidates else 0.0,
        "nearest_assignment_frac": assignment_counts.get("nearest", 0) / len(candidates) if candidates else 0.0,
        "click_candidate_frac": type_counts.get("click", 0) / len(candidates) if candidates else 0.0,
        "type_candidate_frac": type_counts.get("type", 0) / len(candidates) if candidates else 0.0,
        "swipe_candidate_frac": type_counts.get("swipe", 0) / len(candidates) if candidates else 0.0,
        "point_valid_frac": len(points) / len(candidates) if candidates else 0.0,
        "sample_logprob_avg_mean": float(np.mean(logprob_avg)) if logprob_avg else 0.0,
        "sample_logprob_avg_std": std(logprob_avg),
        "sample_logprob_avg_max": max(logprob_avg) if logprob_avg else 0.0,
        "sample_logprob_avg_min": min(logprob_avg) if logprob_avg else 0.0,
        "sample_logprob_avg_gap_top2": top2_gap(logprob_avg),
        "sample_logprob_sum_mean": float(np.mean(logprob_sum)) if logprob_sum else 0.0,
        "sample_logprob_token_mean": float(np.mean(logprob_tokens)) if logprob_tokens else 0.0,
        "logprob_available_frac": len(logprob_avg) / len(candidates) if candidates else 0.0,
        "pred_text_len_mean": float(np.mean(text_lengths)) if text_lengths else 0.0,
        "pred_text_len_std": std(text_lengths),
        "screen_n_controls": float(len(controls)),
        "screen_control_density_per_mpx": float(len(controls)) / (area / 1_000_000.0),
        "screen_n_control_types": float(len(set(control_type(control) for control in controls if control_type(control)))),
        "greedy_nearby_controls_50": float(controls_near_point(greedy_point, controls, 50.0)),
        "greedy_nearby_controls_100": float(controls_near_point(greedy_point, controls, 100.0)),
        "greedy_nearby_controls_150": float(controls_near_point(greedy_point, controls, 150.0)),
        "greedy_has_control": 1.0 if isinstance(greedy_control, dict) and greedy_control.get("key") else 0.0,
        "verifier_score_available_frac": len(verifier_scores) / len(candidates) if candidates else 0.0,
        "verifier_score_mean": float(np.mean(verifier_scores)) if verifier_scores else 0.0,
        "verifier_score_std": std(verifier_scores),
        "verifier_score_max": max(verifier_scores) if verifier_scores else 0.0,
        "verifier_score_gap_top2": top2_gap(verifier_scores),
    }
    features.update(point_stats(points))
    features.update(screen_text_similarity_features(controls, greedy_control if isinstance(greedy_control, dict) else {}))
    return features


def auc_score(y: Sequence[int], scores: Sequence[float]) -> Optional[float]:
    y_arr = np.asarray(y, dtype=int)
    s_arr = np.asarray(scores, dtype=float)
    mask = np.isfinite(s_arr)
    y_arr = y_arr[mask]
    s_arr = s_arr[mask]
    n_pos = int(np.sum(y_arr == 1))
    n_neg = int(np.sum(y_arr == 0))
    if n_pos == 0 or n_neg == 0:
        return None
    ranks = rankdata(s_arr, method="average")
    pos_rank_sum = float(np.sum(ranks[y_arr == 1]))
    return (pos_rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def point_biserial(y: Sequence[int], values: Sequence[float]) -> Optional[float]:
    y_arr = np.asarray(y, dtype=float)
    x_arr = np.asarray(values, dtype=float)
    mask = np.isfinite(x_arr)
    y_arr = y_arr[mask]
    x_arr = x_arr[mask]
    if len(set(y_arr.tolist())) < 2 or float(np.std(x_arr)) == 0.0:
        return None
    return float(np.corrcoef(y_arr, x_arr)[0, 1])


def balanced_accuracy(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    y = np.asarray(y_true, dtype=int)
    pred = np.asarray(y_pred, dtype=int)
    pos = y == 1
    neg = y == 0
    tpr = float(np.mean(pred[pos] == 1)) if np.any(pos) else 0.0
    tnr = float(np.mean(pred[neg] == 0)) if np.any(neg) else 0.0
    return 0.5 * (tpr + tnr)


def make_feature_matrix(rows: Sequence[Mapping[str, Any]], feature_names: Sequence[str]) -> np.ndarray:
    matrix = []
    for row in rows:
        features = row.get("features") if isinstance(row.get("features"), dict) else {}
        matrix.append([float(features.get(name, 0.0) or 0.0) for name in feature_names])
    return np.asarray(matrix, dtype=float)


def stratified_folds(y: np.ndarray, n_folds: int, seed: int) -> List[np.ndarray]:
    rng = np.random.default_rng(seed)
    folds: List[List[int]] = [[] for _ in range(n_folds)]
    for label in (0, 1):
        indices = np.where(y == label)[0]
        rng.shuffle(indices)
        for idx, row_idx in enumerate(indices):
            folds[idx % n_folds].append(int(row_idx))
    return [np.asarray(sorted(fold), dtype=int) for fold in folds]


def standardize_train_test(x_train: np.ndarray, x_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    med = np.nanmedian(x_train, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    x_train = np.where(np.isfinite(x_train), x_train, med)
    x_test = np.where(np.isfinite(x_test), x_test, med)
    mean = np.mean(x_train, axis=0)
    scale = np.std(x_train, axis=0)
    scale = np.where(scale > 1e-8, scale, 1.0)
    return (x_train - mean) / scale, (x_test - mean) / scale


def fit_logistic(x: np.ndarray, y: np.ndarray, l2: float = 1.0) -> np.ndarray:
    y = y.astype(float)
    n_features = x.shape[1]
    pos = max(1.0, float(np.sum(y == 1)))
    neg = max(1.0, float(np.sum(y == 0)))
    weights = np.where(y == 1, 0.5 / pos, 0.5 / neg)

    def objective(params: np.ndarray) -> Tuple[float, np.ndarray]:
        bias = params[0]
        coef = params[1:]
        logits = np.clip(bias + x @ coef, -40.0, 40.0)
        probs = 1.0 / (1.0 + np.exp(-logits))
        eps = 1e-8
        loss = -np.sum(weights * (y * np.log(probs + eps) + (1.0 - y) * np.log(1.0 - probs + eps)))
        loss += 0.5 * l2 * float(np.sum(coef * coef)) / max(1, len(y))
        grad_logits = weights * (probs - y)
        grad_bias = float(np.sum(grad_logits))
        grad_coef = x.T @ grad_logits + l2 * coef / max(1, len(y))
        return float(loss), np.concatenate([[grad_bias], grad_coef])

    init = np.zeros(n_features + 1, dtype=float)
    result = minimize(lambda params: objective(params), init, jac=True, method="L-BFGS-B", options={"maxiter": 300})
    return result.x if result.success else result.x


def logistic_cv(rows: Sequence[Mapping[str, Any]], feature_names: Sequence[str], n_folds: int, seed: int) -> Dict[str, Any]:
    y = np.asarray([int(row["critical"]) for row in rows], dtype=int)
    x = make_feature_matrix(rows, feature_names)
    folds = stratified_folds(y, n_folds, seed)
    scores = np.zeros(len(rows), dtype=float)
    coef_rows = []
    for test_idx in folds:
        train_mask = np.ones(len(rows), dtype=bool)
        train_mask[test_idx] = False
        x_train, x_test = standardize_train_test(x[train_mask], x[test_idx])
        params = fit_logistic(x_train, y[train_mask])
        coef_rows.append(params[1:])
        logits = np.clip(params[0] + x_test @ params[1:], -40.0, 40.0)
        scores[test_idx] = 1.0 / (1.0 + np.exp(-logits))
    pred = (scores >= 0.5).astype(int)
    coefs = np.asarray(coef_rows) if coef_rows else np.zeros((1, len(feature_names)))
    importances = sorted(
        ({"feature": name, "importance": float(abs(np.mean(coefs[:, idx])))} for idx, name in enumerate(feature_names)),
        key=lambda item: item["importance"],
        reverse=True,
    )
    return {
        "scores": scores.tolist(),
        "auc": auc_score(y, scores.tolist()),
        "balanced_accuracy": balanced_accuracy(y, pred),
        "feature_importances": importances[:20],
    }


def decision_stump_cv(rows: Sequence[Mapping[str, Any]], feature_names: Sequence[str], n_folds: int, seed: int) -> Dict[str, Any]:
    y = np.asarray([int(row["critical"]) for row in rows], dtype=int)
    x = make_feature_matrix(rows, feature_names)
    folds = stratified_folds(y, n_folds, seed)
    scores = np.zeros(len(rows), dtype=float)
    models = []
    for test_idx in folds:
        train_mask = np.ones(len(rows), dtype=bool)
        train_mask[test_idx] = False
        x_train = x[train_mask]
        y_train = y[train_mask]
        best = {"feature_index": 0, "threshold": 0.0, "polarity": 1, "balanced_accuracy": -1.0}
        for feature_idx in range(len(feature_names)):
            values = np.unique(x_train[:, feature_idx][np.isfinite(x_train[:, feature_idx])])
            if len(values) == 0:
                continue
            if len(values) > 60:
                values = np.quantile(values, np.linspace(0.05, 0.95, 31))
            for threshold in values:
                for polarity in (1, -1):
                    pred = ((x_train[:, feature_idx] >= threshold) if polarity == 1 else (x_train[:, feature_idx] <= threshold)).astype(int)
                    bal = balanced_accuracy(y_train, pred)
                    if bal > best["balanced_accuracy"]:
                        best = {"feature_index": feature_idx, "threshold": float(threshold), "polarity": polarity, "balanced_accuracy": bal}
        models.append({**best, "feature": feature_names[int(best["feature_index"])]})
        feature_values = x[test_idx, int(best["feature_index"])]
        oriented = feature_values if int(best["polarity"]) == 1 else -feature_values
        scores[test_idx] = oriented
    pred_threshold = np.median(scores)
    pred = (scores >= pred_threshold).astype(int)
    return {"scores": scores.tolist(), "auc": auc_score(y, scores.tolist()), "balanced_accuracy": balanced_accuracy(y, pred), "models": models}


def per_signal_metrics(rows: Sequence[Mapping[str, Any]], feature_names: Sequence[str]) -> List[Dict[str, Any]]:
    y = [int(row["critical"]) for row in rows]
    out = []
    for name in feature_names:
        values = [float(row["features"].get(name, 0.0) or 0.0) for row in rows]
        auc = auc_score(y, values)
        oriented_auc = None if auc is None else max(float(auc), 1.0 - float(auc))
        direction = "high" if auc is not None and auc >= 0.5 else "low"
        out.append({
            "feature": name,
            "point_biserial_r": point_biserial(y, values),
            "auc_high_predicts_critical": auc,
            "best_oriented_auc": oriented_auc,
            "critical_direction": direction,
            "mean_critical": float(np.mean([value for value, label in zip(values, y) if label == 1])) if any(y) else None,
            "mean_noncritical": float(np.mean([value for value, label in zip(values, y) if label == 0])) if any(label == 0 for label in y) else None,
        })
    out.sort(key=lambda item: item["best_oriented_auc"] if item["best_oriented_auc"] is not None else -1.0, reverse=True)
    return out


def oriented_signal_scores(rows: Sequence[Mapping[str, Any]], signal: Mapping[str, Any]) -> List[float]:
    values = [float(row["features"].get(str(signal["feature"]), 0.0) or 0.0) for row in rows]
    if signal.get("critical_direction") == "low":
        values = [-value for value in values]
    return values


def triage_table(rows: Sequence[Mapping[str, Any]], scores: Sequence[float], budgets: Sequence[float]) -> List[Dict[str, Any]]:
    y = np.asarray([int(row["critical"]) for row in rows], dtype=int)
    order = np.argsort(-np.asarray(scores, dtype=float))
    total_critical = int(np.sum(y == 1))
    total_log_failure = sum(float(row.get("step_log_failure", 0.0) or 0.0) for row in rows)
    out = []
    for budget in budgets:
        k = max(1, int(round(len(rows) * budget)))
        selected = order[:k]
        selected_critical = int(np.sum(y[selected] == 1))
        selected_log_failure = sum(float(rows[int(idx)].get("step_log_failure", 0.0) or 0.0) for idx in selected)
        out.append({
            "budget_fraction": budget,
            "selected_steps": k,
            "recall": selected_critical / total_critical if total_critical else 0.0,
            "precision": selected_critical / k if k else 0.0,
            "random_recall": budget,
            "random_precision": total_critical / len(rows) if rows else 0.0,
            "all_steps_recall": 1.0,
            "all_steps_cost": 1.0,
            "log_failure_share": selected_log_failure / total_log_failure if total_log_failure else 0.0,
        })
    return out


def rank_rows(rows: List[Dict[str, Any]], scores: Sequence[float], score_name: str) -> None:
    order = np.argsort(-np.asarray(scores, dtype=float))
    for rank, idx in enumerate(order, 1):
        rows[int(idx)][score_name] = float(scores[int(idx)])
        rows[int(idx)]["triage_rank"] = rank
        rows[int(idx)]["triage_percentile"] = rank / len(rows) if rows else 0.0


def feature_name_is_allowed(name: str) -> bool:
    return not any(part in name for part in LEAKY_NAME_PARTS)


def build_rows(args: argparse.Namespace) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    candidate_rows = read_jsonl(Path(args.candidates))
    episodes = read_episode_index(Path(args.test_data))
    tasks = read_task_index(Path(args.crit_tasks))
    rows: List[Dict[str, Any]] = []
    skipped = Counter()
    for row in candidate_rows:
        episode_id = str(row.get("episode_id"))
        step_idx = int(row.get("step_idx") or 0)
        episode = episodes.get(episode_id)
        task = tasks.get(episode_id)
        if episode is None or task is None:
            skipped["missing_episode_or_task"] += 1
            continue
        per_p = task.get("per_step_p_heldout_cv") if isinstance(task.get("per_step_p_heldout_cv"), list) else []
        if step_idx >= len(per_p):
            skipped["missing_p_i"] += 1
            continue
        bottom2 = {int(idx) for idx in task.get("bottom2_critical_indices", [])}
        bottom1 = {int(idx) for idx in task.get("bottom1_critical_indices", [])}
        p_i = float(per_p[step_idx])
        eps = 1e-8
        step_log_failure = -math.log(max(eps, min(1.0, p_i)))
        features = extract_features(row, episode)
        rows.append({
            "target_id": row.get("target_id"),
            "episode_id": episode_id,
            "episode_order": row.get("episode_order"),
            "step_idx": step_idx,
            "task_k": int(task.get("k") or len(per_p)),
            "critical": step_idx in bottom2,
            "bottom1": step_idx in bottom1,
            "p_i_heldout_label_only": p_i,
            "step_log_failure": step_log_failure,
            "actual_task_success": bool(task.get("actual_success")),
            "features": features,
        })
    manifest = {
        "candidate_rows_in": len(candidate_rows),
        "rows_out": len(rows),
        "episodes_out": len({row["episode_id"] for row in rows}),
        "skipped": dict(skipped),
        "coverage_note": "Uses the existing sampled pool only. If the pool is a stopped slice, results are slice diagnostics, not full TEST.",
    }
    return rows, manifest


def decide_gate(classifier_auc: Optional[float], triage20: Mapping[str, Any]) -> Tuple[str, str]:
    auc = classifier_auc or 0.0
    recall = float(triage20.get("recall") or 0.0)
    random_recall = float(triage20.get("random_recall") or 0.20)
    if auc >= 0.70 and recall >= random_recall + 0.20:
        return "CRITICAL STEPS IDENTIFIABLE", "GT-free signals predict held-out low-p_i steps well and top-budget triage clearly beats random."
    if auc >= 0.60 and recall >= random_recall + 0.10:
        return "WEAKLY IDENTIFIABLE", "GT-free signals carry usable but not clean inference-time information about hard steps."
    return "NOT IDENTIFIABLE", "GT-free signals are near chance or triage does not beat random enough to operationalize critical-step gating."


def render_report(summary: Mapping[str, Any], output_dir: Path) -> str:
    lines = ["# Inference-Time Critical-Step Identifiability", ""]
    lines.append("Diagnostic only: no base/verifier training. Critical labels are bottom-2 held-out p_i; all predictor features are GT-free and inference-available.")
    lines.append("")
    lines.append("## Scope")
    lines.append("")
    lines.append(f"- sampled pool: `{summary['inputs']['candidates']}`")
    lines.append(f"- rows analyzed: `{summary['dataset']['rows']}` steps across `{summary['dataset']['episodes']}` episodes")
    lines.append(f"- critical prevalence: `{summary['dataset']['critical_prevalence']*100:.2f}%`")
    lines.append(f"- coverage note: {summary['dataset']['coverage_note']}")
    lines.append("")
    lines.append("## Metric 1: GT-Free Predictive Power")
    lines.append("")
    clf = summary["classifier"]
    lines.append("| predictor | AUC | balanced accuracy | note |")
    lines.append("|---|---:|---:|---|")
    lines.append(f"| majority all-noncritical | NA | 50.00% | chance balanced accuracy baseline |")
    lines.append(f"| best single signal: `{summary['best_signal']['feature']}` | {summary['best_signal']['auc']*100:.2f}% | NA | direction={summary['best_signal']['direction']} |")
    lines.append(f"| decision stump CV | {(summary['stump']['auc'] or 0.0)*100:.2f}% | {summary['stump']['balanced_accuracy']*100:.2f}% | GT-free one-split rule |")
    lines.append(f"| logistic CV | {(clf['auc'] or 0.0)*100:.2f}% | {clf['balanced_accuracy']*100:.2f}% | diagnostic classifier, no model training artifact |")
    lines.append("")
    lines.append("Top per-signal AUCs:")
    lines.append("")
    lines.append("| rank | signal | oriented AUC | point-biserial r | critical direction | critical mean | non-critical mean |")
    lines.append("|---:|---|---:|---:|---|---:|---:|")
    for idx, item in enumerate(summary["per_signal"][:15], 1):
        r_text = item.get("point_biserial_r_text") or "NA"
        lines.append(
            f"| {idx} | `{item['feature']}` | {(item['best_oriented_auc'] or 0.0)*100:.2f}% | "
            f"{r_text} | {item['critical_direction']} | "
            f"{(item.get('mean_critical') or 0.0):.3f} | {(item.get('mean_noncritical') or 0.0):.3f} |"
        )
    lines.append("")
    lines.append("## Metric 2: Mechanism Signal Ranking")
    lines.append("")
    mech = summary["mechanism"]
    lines.append(f"Best sampling-disagreement signal: `{mech['best_disagreement_feature']}` with standalone AUC `{mech['best_disagreement_auc']*100:.2f}%`.")
    lines.append(f"Sampling disagreement rank among all signals: `{mech['best_disagreement_rank']}`.")
    if summary["verifier_dispersion_available"]:
        lines.append("Verifier dispersion features were available and included.")
    else:
        lines.append("Verifier dispersion features were not available in this sampled pool; this run tests sampling/logprob/screen-complexity signals only.")
    lines.append("")
    lines.append("## Metric 3: Hard vs Task-Decisive")
    lines.append("")
    lines.append("Critical here means low held-out p_i (hard). Task-decisiveness is evaluated separately by log-failure mass, not used as a feature.")
    lines.append("")
    lines.append("| selected set | cost | critical recall | precision | log-failure share | random recall |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for row in summary["triage_primary"]:
        lines.append(
            f"| top {row['budget_fraction']*100:.0f}% by `{summary['primary_score']}` | {row['budget_fraction']*100:.0f}% | "
            f"{row['recall']*100:.2f}% | {row['precision']*100:.2f}% | {row['log_failure_share']*100:.2f}% | {row['random_recall']*100:.2f}% |"
        )
    lines.append(f"True bottom-2 steps in the covered slice carry `{summary['decisiveness']['true_critical_log_failure_share']*100:.2f}%` of covered log-failure mass.")
    lines.append("")
    lines.append("## Metric 4: Operational Triage")
    lines.append("")
    lines.append("Primary score table:")
    lines.append("")
    lines.append("| budget | selected steps | recall | precision | random recall | recall lift vs random |")
    lines.append("|---:|---:|---:|---:|---:|---:|")
    for row in summary["triage_primary"]:
        lift = row["recall"] / row["random_recall"] if row["random_recall"] else 0.0
        lines.append(f"| {row['budget_fraction']*100:.0f}% | {row['selected_steps']} | {row['recall']*100:.2f}% | {row['precision']*100:.2f}% | {row['random_recall']*100:.2f}% | {lift:.2f}x |")
    lines.append("")
    lines.append("Sampling-disagreement-only table:")
    lines.append("")
    lines.append("| budget | recall | precision | random recall | recall lift vs random |")
    lines.append("|---:|---:|---:|---:|---:|")
    for row in summary["triage_disagreement"]:
        lift = row["recall"] / row["random_recall"] if row["random_recall"] else 0.0
        lines.append(f"| {row['budget_fraction']*100:.0f}% | {row['recall']*100:.2f}% | {row['precision']*100:.2f}% | {row['random_recall']*100:.2f}% | {lift:.2f}x |")
    lines.append("")
    lines.append("## Leakage Audit")
    lines.append("")
    lines.append(f"- feature count: `{summary['feature_audit']['n_features']}`")
    lines.append(f"- excluded/leaky source fields: `{', '.join(summary['feature_audit']['excluded_fields'])}`")
    lines.append(f"- suspicious AUC>=0.90 features: `{summary['feature_audit']['suspicious_high_auc_features']}`")
    lines.append("- p_i / bottom-2 labels are used only for target and log-failure evaluation, never as predictor features.")
    lines.append("")
    lines.append("## Gate")
    lines.append("")
    lines.append(f"**{summary['gate']['verdict']}**")
    lines.append("")
    lines.append(summary["gate"]["reason"])
    lines.append("")
    lines.append("## Artifacts")
    lines.append("")
    lines.append(f"- `{output_dir / 'identifiability.md'}`")
    lines.append(f"- `{output_dir / 'summary.json'}`")
    lines.append(f"- `{output_dir / 'per_step.jsonl'}`")
    lines.append("")
    lines.append("STOP for review.")
    return "\n".join(lines) + "\n"


def fmt_report_numbers(summary: Dict[str, Any]) -> None:
    # Keep markdown numeric rendering simple by avoiding nested format conditionals.
    for item in summary["per_signal"]:
        if item.get("point_biserial_r") is None:
            item["point_biserial_r_text"] = "NA"
        else:
            item["point_biserial_r_text"] = f"{item['point_biserial_r']:.3f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", default=DEFAULT_CANDIDATES)
    parser.add_argument("--test-data", default=DEFAULT_TEST_DATA)
    parser.add_argument("--crit-tasks", default=DEFAULT_CRIT_TASKS)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=43)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows, manifest = build_rows(args)
    if not rows:
        raise SystemExit("no rows available for identifiability diagnostic")
    feature_names = sorted(rows[0]["features"].keys())
    blocked = [name for name in feature_names if not feature_name_is_allowed(name)]
    if blocked:
        raise SystemExit(f"refusing to use leaky feature names: {blocked}")

    y = [int(row["critical"]) for row in rows]
    per_signal = per_signal_metrics(rows, feature_names)
    best_signal = per_signal[0]
    disagreement = [item for item in per_signal if item["feature"] in DISAGREEMENT_FEATURES]
    best_disagreement = disagreement[0] if disagreement else best_signal
    logistic = logistic_cv(rows, feature_names, max(2, args.folds), args.seed)
    stump = decision_stump_cv(rows, feature_names, max(2, args.folds), args.seed)

    classifier_auc = logistic.get("auc") or 0.0
    best_signal_auc = best_signal.get("best_oriented_auc") or 0.0
    if classifier_auc >= best_signal_auc:
        primary_scores = logistic["scores"]
        primary_score = "logistic_cv_score"
    else:
        primary_scores = oriented_signal_scores(rows, best_signal)
        primary_score = f"single_signal:{best_signal['feature']}"
    disagreement_scores = oriented_signal_scores(rows, best_disagreement)
    rank_rows(rows, primary_scores, "classifier_score")
    for row, score in zip(rows, disagreement_scores):
        row["best_disagreement_score"] = float(score)

    triage_primary = triage_table(rows, primary_scores, BUDGETS)
    triage_disagreement = triage_table(rows, disagreement_scores, BUDGETS)
    triage20 = next(row for row in triage_primary if abs(row["budget_fraction"] - 0.20) < 1e-6)
    gate_verdict, gate_reason = decide_gate(logistic.get("auc"), triage20)
    total_log_failure = sum(float(row.get("step_log_failure") or 0.0) for row in rows)
    true_critical_log_failure = sum(float(row.get("step_log_failure") or 0.0) for row in rows if row["critical"])
    suspicious = [item["feature"] for item in per_signal if (item.get("best_oriented_auc") or 0.0) >= 0.90]
    summary: Dict[str, Any] = {
        "inputs": {"candidates": args.candidates, "test_data": args.test_data, "crit_tasks": args.crit_tasks},
        "dataset": {
            "rows": len(rows),
            "episodes": len({row["episode_id"] for row in rows}),
            "critical_steps": int(sum(y)),
            "critical_prevalence": float(sum(y) / len(y)),
            "coverage_note": manifest["coverage_note"],
            "manifest": manifest,
        },
        "per_signal": per_signal,
        "best_signal": {"feature": best_signal["feature"], "auc": best_signal["best_oriented_auc"], "direction": best_signal["critical_direction"]},
        "stump": {key: value for key, value in stump.items() if key != "scores"},
        "classifier": {key: value for key, value in logistic.items() if key != "scores"},
        "mechanism": {
            "best_disagreement_feature": best_disagreement["feature"],
            "best_disagreement_auc": best_disagreement["best_oriented_auc"],
            "best_disagreement_rank": 1 + next(idx for idx, item in enumerate(per_signal) if item["feature"] == best_disagreement["feature"]),
        },
        "primary_score": primary_score,
        "triage_primary": triage_primary,
        "triage_disagreement": triage_disagreement,
        "decisiveness": {
            "true_critical_log_failure_share": true_critical_log_failure / total_log_failure if total_log_failure else 0.0,
            "total_log_failure": total_log_failure,
        },
        "verifier_dispersion_available": any(row["features"].get("verifier_score_available_frac", 0.0) > 0.0 for row in rows),
        "feature_audit": {
            "n_features": len(feature_names),
            "feature_names": feature_names,
            "excluded_fields": ["row.gt_action", "row.greedy_correct", "row.correct_candidate_ids", "candidate.reward", "candidate.is_correct", "task.per_step_p_heldout_cv as feature"],
            "suspicious_high_auc_features": suspicious,
        },
        "gate": {"verdict": gate_verdict, "reason": gate_reason},
    }
    fmt_report_numbers(summary)
    compact_rows = []
    for row in rows:
        compact_rows.append({
            "target_id": row["target_id"],
            "episode_id": row["episode_id"],
            "step_idx": row["step_idx"],
            "task_k": row["task_k"],
            "critical": row["critical"],
            "bottom1": row["bottom1"],
            "p_i_heldout_label_only": row["p_i_heldout_label_only"],
            "step_log_failure": row["step_log_failure"],
            "classifier_score": row.get("classifier_score"),
            "best_disagreement_score": row.get("best_disagreement_score"),
            "triage_rank": row.get("triage_rank"),
            "triage_percentile": row.get("triage_percentile"),
            "features": row["features"],
        })
    write_jsonl(output_dir / "per_step.jsonl", compact_rows)
    write_json(output_dir / "summary.json", summary)
    report = render_report(summary, output_dir)
    (output_dir / "identifiability.md").write_text(report, encoding="utf-8")
    print(json.dumps({
        "output_dir": str(output_dir),
        "rows": len(rows),
        "critical_prevalence": summary["dataset"]["critical_prevalence"],
        "logistic_auc": summary["classifier"]["auc"],
        "best_signal": summary["best_signal"],
        "triage20_recall": triage20["recall"],
        "gate": gate_verdict,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()