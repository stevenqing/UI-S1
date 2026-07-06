#!/usr/bin/env python3
"""Critical-step separability diagnostic for Phase 0 self-distillation.

Reads already-computed Phase 0 per-state artifacts and asks whether bottom-2
REPAIR states can be separated from bottom-2 BREAK states without matcher, GT,
or teacher-outcome leakage. This trains only diagnostic classifiers over frozen
artifacts; it performs no teacher calls and no model distillation.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import minimize


DEFAULT_INPUT = "outputs/reexam_distill/phase0_full_train_20260630_162941_critstep/phase0_critstep_per_state.jsonl"
DEFAULT_REPORT = "outputs/reexam_distill/critstep_separability.md"
DEFAULT_PER_STATE = "outputs/reexam_distill/critstep_separability_per_state.jsonl"

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "in", "into",
    "is", "it", "of", "on", "or", "the", "to", "with", "you", "your", "slide",
    "step", "using", "use", "set", "add", "make", "create", "open", "select",
}


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(values, -40.0, 40.0)))


def percentile_ci(values: Sequence[float], low: float = 2.5, high: float = 97.5) -> Tuple[float, float]:
    if not values:
        return 0.0, 0.0
    arr = np.asarray(values, dtype=float)
    return float(np.percentile(arr, low)), float(np.percentile(arr, high))


def mean(values: Sequence[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def fmt_float(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def step_phase(step_idx: int, task_k: int) -> str:
    if task_k <= 1:
        return "single"
    if step_idx == 0:
        return "first"
    if step_idx == task_k - 1:
        return "last"
    frac = (step_idx + 1) / max(task_k, 1)
    if frac <= 0.33:
        return "early"
    if frac <= 0.67:
        return "middle"
    return "late"


def text_tokens(text: str) -> List[str]:
    tokens = re.findall(r"[a-z0-9_]+", text.lower())
    return [tok for tok in tokens if len(tok) >= 3 and tok not in STOPWORDS]


def pred_action(row: Dict[str, Any]) -> Dict[str, Any]:
    value = row.get("V", {}).get("pred_action")
    return value if isinstance(value, dict) else {}


def pred_coord(row: Dict[str, Any]) -> Optional[Tuple[float, float]]:
    action = pred_action(row)
    coord = action.get("coordinate") or action.get("xy")
    if not (isinstance(coord, list) and len(coord) >= 2):
        return None
    try:
        return float(coord[0]), float(coord[1])
    except (TypeError, ValueError):
        return None


def position_bin_from_pred(row: Dict[str, Any]) -> str:
    coord = pred_coord(row)
    if coord is None:
        return "missing"
    x_norm = min(0.999, max(0.0, coord[0] / 1040.0))
    y_norm = min(0.999, max(0.0, coord[1] / 736.0))
    return f"r{int(y_norm * 3)}c{int(x_norm * 3)}"


def safe_len(value: Any) -> int:
    return len(str(value or ""))


def build_raw_features(row: Dict[str, Any], mode: str) -> Tuple[Dict[str, float], Dict[str, str], List[str]]:
    numeric: Dict[str, float] = {}
    categorical: Dict[str, str] = {}
    tokens: List[str] = []

    include_base = mode in {"full", "no_difficulty"}
    include_difficulty = mode in {"full", "difficulty_only"}

    if include_base:
        step_idx = int(row.get("step_idx") or 0)
        task_k = max(1, int(row.get("task_k") or 1))
        goal = str(row.get("goal") or "")
        goal_tokens = text_tokens(goal)
        v = row.get("V", {}) if isinstance(row.get("V"), dict) else {}
        pred = pred_action(row)
        coord = pred_coord(row)
        pred_text = str(v.get("pred_text") or "")

        numeric.update({
            "step_idx": float(step_idx),
            "task_k": float(task_k),
            "step_frac": float((step_idx + 1) / task_k),
            "is_first": float(step_idx == 0),
            "is_last": float(step_idx == task_k - 1),
            "num_controls": float(row.get("num_controls") or 0),
            "log_num_controls": math.log1p(float(row.get("num_controls") or 0)),
            "a11y_present": float(bool(row.get("a11y_present"))),
            "goal_chars": float(len(goal)),
            "goal_words": float(len(goal_tokens)),
            "goal_unique_words": float(len(set(goal_tokens))),
            "goal_digit_count": float(sum(ch.isdigit() for ch in goal)),
            "goal_has_digit": float(any(ch.isdigit() for ch in goal)),
            "v_pred_has_coord": float(coord is not None),
            "v_pred_text_chars": float(len(pred_text)),
            "v_pred_text_lines": float(pred_text.count("\n") + 1 if pred_text else 0),
            "v_pred_text_has_tool_call": float("tool_call" in pred_text),
            "v_pred_text_has_continue": float("CONTINUE" in pred_text or "continue" in pred_text),
        })
        if coord is None:
            numeric["v_pred_x_norm"] = 0.0
            numeric["v_pred_y_norm"] = 0.0
        else:
            numeric["v_pred_x_norm"] = min(1.0, max(0.0, coord[0] / 1040.0))
            numeric["v_pred_y_norm"] = min(1.0, max(0.0, coord[1] / 736.0))

        categorical.update({
            "step_phase": step_phase(step_idx, task_k),
            "v_pred_type": str(v.get("pred_type") or "missing").lower(),
            "v_pred_action": str(pred.get("action") or "missing").lower(),
            "v_pred_pos_bin": position_bin_from_pred(row),
        })
        tokens = goal_tokens

    if include_difficulty:
        p_hat = row.get("p_hat_heldout_bucket")
        try:
            p_value = float(p_hat)
        except (TypeError, ValueError):
            p_value = 0.0
        source_count = row.get("p_hat_source_count")
        try:
            source_count_value = float(source_count)
        except (TypeError, ValueError):
            source_count_value = 0.0
        numeric.update({
            "p_hat_heldout_bucket": p_value,
            "p_hat_source_count": source_count_value,
            "log_p_hat_source_count": math.log1p(max(0.0, source_count_value)),
            "critical_rank": float(row.get("critical_rank") or 0),
        })
        categorical["p_hat_source"] = str(row.get("p_hat_source") or "missing")

    return numeric, categorical, tokens


@dataclass
class FeatureEncoder:
    mode: str
    max_text_features: int = 120
    numeric_names: List[str] = None  # type: ignore[assignment]
    numeric_mean: np.ndarray = None  # type: ignore[assignment]
    numeric_std: np.ndarray = None  # type: ignore[assignment]
    categorical_levels: List[Tuple[str, str]] = None  # type: ignore[assignment]
    token_vocab: List[str] = None  # type: ignore[assignment]
    feature_names: List[str] = None  # type: ignore[assignment]

    def fit(self, rows: Sequence[Dict[str, Any]]) -> "FeatureEncoder":
        numeric_keys: set[str] = set()
        categorical_counts: Counter[Tuple[str, str]] = Counter()
        doc_counts: Counter[str] = Counter()
        numeric_rows: List[Dict[str, float]] = []
        for row in rows:
            numeric, categorical, tokens = build_raw_features(row, self.mode)
            numeric_rows.append(numeric)
            numeric_keys.update(numeric)
            categorical_counts.update((key, value) for key, value in categorical.items())
            doc_counts.update(set(tokens))

        self.numeric_names = sorted(numeric_keys)
        matrix = np.asarray([[numeric.get(name, 0.0) for name in self.numeric_names] for numeric in numeric_rows], dtype=float)
        if matrix.size:
            self.numeric_mean = matrix.mean(axis=0)
            std = matrix.std(axis=0)
            self.numeric_std = np.where(std < 1e-9, 1.0, std)
        else:
            self.numeric_mean = np.zeros(0, dtype=float)
            self.numeric_std = np.ones(0, dtype=float)

        self.categorical_levels = sorted(categorical_counts)
        self.token_vocab = [token for token, _ in doc_counts.most_common(self.max_text_features)]
        self.feature_names = (
            [f"num:{name}" for name in self.numeric_names]
            + [f"cat:{key}={value}" for key, value in self.categorical_levels]
            + [f"goal:{token}" for token in self.token_vocab]
        )
        return self

    def transform(self, rows: Sequence[Dict[str, Any]]) -> np.ndarray:
        cols = len(self.feature_names)
        matrix = np.zeros((len(rows), cols), dtype=float)
        categorical_index = {pair: idx for idx, pair in enumerate(self.categorical_levels)}
        token_index = {token: idx for idx, token in enumerate(self.token_vocab)}
        cat_offset = len(self.numeric_names)
        token_offset = cat_offset + len(self.categorical_levels)
        for row_idx, row in enumerate(rows):
            numeric, categorical, tokens = build_raw_features(row, self.mode)
            if self.numeric_names:
                values = np.asarray([numeric.get(name, 0.0) for name in self.numeric_names], dtype=float)
                matrix[row_idx, : len(self.numeric_names)] = (values - self.numeric_mean) / self.numeric_std
            for item in categorical.items():
                idx = categorical_index.get(item)
                if idx is not None:
                    matrix[row_idx, cat_offset + idx] = 1.0
            token_counts = Counter(tokens)
            for token, count in token_counts.items():
                idx = token_index.get(token)
                if idx is not None:
                    matrix[row_idx, token_offset + idx] = min(3.0, float(count))
        return matrix


@dataclass
class LogisticModel:
    weights: np.ndarray

    def predict_proba(self, x_matrix: np.ndarray) -> np.ndarray:
        x_bias = np.c_[np.ones(x_matrix.shape[0]), x_matrix]
        return sigmoid(x_bias @ self.weights)

    def top_features(self, feature_names: Sequence[str], n: int = 10) -> List[Tuple[str, float]]:
        coefs = self.weights[1:]
        order = np.argsort(np.abs(coefs))[::-1][:n]
        return [(feature_names[idx], float(coefs[idx])) for idx in order]


def class_weights(labels: np.ndarray) -> np.ndarray:
    labels = labels.astype(int)
    total = len(labels)
    positives = int(labels.sum())
    negatives = total - positives
    pos_weight = total / max(1, 2 * positives)
    neg_weight = total / max(1, 2 * negatives)
    return np.where(labels == 1, pos_weight, neg_weight).astype(float)


def fit_logistic(x_matrix: np.ndarray, labels: np.ndarray, l2: float = 1.0) -> LogisticModel:
    labels = labels.astype(float)
    weights = class_weights(labels)
    weights = weights / max(weights.mean(), 1e-12)
    x_bias = np.c_[np.ones(x_matrix.shape[0]), x_matrix]
    initial = np.zeros(x_bias.shape[1], dtype=float)

    def objective(params: np.ndarray) -> Tuple[float, np.ndarray]:
        logits = x_bias @ params
        probs = sigmoid(logits)
        loss_vec = np.logaddexp(0.0, logits) - labels * logits
        penalty = 0.5 * l2 * np.dot(params[1:], params[1:]) / max(1, x_matrix.shape[0])
        loss = float(np.mean(weights * loss_vec) + penalty)
        grad = x_bias.T @ (weights * (probs - labels)) / max(1, x_matrix.shape[0])
        grad[1:] += l2 * params[1:] / max(1, x_matrix.shape[0])
        return loss, grad

    result = minimize(lambda p: objective(p), initial, method="L-BFGS-B", jac=True, options={"maxiter": 250, "ftol": 1e-8})
    params = result.x if result.success or np.all(np.isfinite(result.x)) else initial
    return LogisticModel(params)


def weighted_gini(labels: np.ndarray, weights: np.ndarray) -> float:
    total = float(weights.sum())
    if total <= 0:
        return 0.0
    pos = float(weights[labels == 1].sum()) / total
    neg = 1.0 - pos
    return 1.0 - pos * pos - neg * neg


@dataclass
class TreeNode:
    prob: float
    feature_idx: int = -1
    threshold: float = 0.0
    left: Optional["TreeNode"] = None
    right: Optional["TreeNode"] = None

    def is_leaf(self) -> bool:
        return self.left is None or self.right is None


@dataclass
class ShallowTree:
    root: TreeNode
    importances: Dict[int, float]

    def predict_one(self, values: np.ndarray) -> float:
        node = self.root
        while not node.is_leaf():
            node = node.left if values[node.feature_idx] <= node.threshold else node.right  # type: ignore[assignment]
        return node.prob

    def predict_proba(self, x_matrix: np.ndarray) -> np.ndarray:
        return np.asarray([self.predict_one(row) for row in x_matrix], dtype=float)

    def top_features(self, feature_names: Sequence[str], n: int = 10) -> List[Tuple[str, float]]:
        ordered = sorted(self.importances.items(), key=lambda item: item[1], reverse=True)[:n]
        return [(feature_names[idx], float(value)) for idx, value in ordered]


def candidate_thresholds(values: np.ndarray) -> np.ndarray:
    unique = np.unique(values)
    if len(unique) <= 1:
        return np.asarray([], dtype=float)
    if len(unique) <= 32:
        return (unique[:-1] + unique[1:]) / 2.0
    qs = np.linspace(0.05, 0.95, 19)
    return np.unique(np.quantile(values, qs))


def fit_tree_recursive(
    x_matrix: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
    depth: int,
    max_depth: int,
    min_leaf: int,
    importances: Dict[int, float],
) -> TreeNode:
    total_weight = float(weights.sum())
    prob = float(weights[labels == 1].sum() / total_weight) if total_weight > 0 else 0.0
    node = TreeNode(prob=prob)
    if depth >= max_depth or len(labels) < 2 * min_leaf or len(np.unique(labels)) == 1:
        return node

    parent_gini = weighted_gini(labels, weights)
    best_gain = 0.0
    best_feature = -1
    best_threshold = 0.0
    best_mask: Optional[np.ndarray] = None

    for feature_idx in range(x_matrix.shape[1]):
        values = x_matrix[:, feature_idx]
        for threshold in candidate_thresholds(values):
            left_mask = values <= threshold
            left_count = int(left_mask.sum())
            right_count = len(labels) - left_count
            if left_count < min_leaf or right_count < min_leaf:
                continue
            left_w = weights[left_mask]
            right_w = weights[~left_mask]
            left_weight = float(left_w.sum())
            right_weight = float(right_w.sum())
            child_gini = (
                left_weight * weighted_gini(labels[left_mask], left_w)
                + right_weight * weighted_gini(labels[~left_mask], right_w)
            ) / max(left_weight + right_weight, 1e-12)
            gain = parent_gini - child_gini
            if gain > best_gain + 1e-12:
                best_gain = gain
                best_feature = feature_idx
                best_threshold = float(threshold)
                best_mask = left_mask

    if best_mask is None or best_feature < 0:
        return node

    node.feature_idx = best_feature
    node.threshold = best_threshold
    importances[best_feature] = importances.get(best_feature, 0.0) + best_gain * float(weights.sum())
    node.left = fit_tree_recursive(x_matrix[best_mask], labels[best_mask], weights[best_mask], depth + 1, max_depth, min_leaf, importances)
    node.right = fit_tree_recursive(x_matrix[~best_mask], labels[~best_mask], weights[~best_mask], depth + 1, max_depth, min_leaf, importances)
    return node


def fit_tree(x_matrix: np.ndarray, labels: np.ndarray, max_depth: int = 2, min_leaf: int = 12) -> ShallowTree:
    weights = class_weights(labels)
    weights = weights / max(weights.mean(), 1e-12)
    importances: Dict[int, float] = {}
    root = fit_tree_recursive(x_matrix, labels.astype(int), weights, 0, max_depth, min_leaf, importances)
    return ShallowTree(root=root, importances=importances)


def stratified_folds(labels: np.ndarray, n_splits: int, seed: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(seed)
    pos = np.where(labels == 1)[0]
    neg = np.where(labels == 0)[0]
    rng.shuffle(pos)
    rng.shuffle(neg)
    pos_chunks = np.array_split(pos, n_splits)
    neg_chunks = np.array_split(neg, n_splits)
    all_indices = np.arange(len(labels))
    folds: List[Tuple[np.ndarray, np.ndarray]] = []
    for fold_idx in range(n_splits):
        valid = np.concatenate([pos_chunks[fold_idx], neg_chunks[fold_idx]])
        valid_set = set(int(idx) for idx in valid)
        train = np.asarray([idx for idx in all_indices if int(idx) not in valid_set], dtype=int)
        folds.append((train, valid.astype(int)))
    return folds


def metric_at_threshold(labels: np.ndarray, scores: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    pred = scores >= threshold
    positives = labels == 1
    negatives = labels == 0
    tp = int(np.logical_and(pred, positives).sum())
    fp = int(np.logical_and(pred, negatives).sum())
    tn = int(np.logical_and(~pred, negatives).sum())
    fn = int(np.logical_and(~pred, positives).sum())
    tpr = tp / max(1, tp + fn)
    tnr = tn / max(1, tn + fp)
    precision = tp / max(1, tp + fp)
    recall = tpr
    return {
        "balanced_accuracy": 0.5 * (tpr + tnr),
        "precision": precision,
        "recall": recall,
        "tp": float(tp),
        "fp": float(fp),
        "tn": float(tn),
        "fn": float(fn),
    }


@dataclass
class EvalResult:
    model_name: str
    mode: str
    repeat_metrics: List[Dict[str, float]]
    avg_scores: np.ndarray
    full_scores: np.ndarray
    full_top_features: List[Tuple[str, float]]
    top_feature_sets: List[List[str]]
    top_feature_counts: Counter[str]
    feature_names: List[str]

    def summary(self) -> Dict[str, Any]:
        bacc = [item["balanced_accuracy"] for item in self.repeat_metrics]
        precision = [item["precision"] for item in self.repeat_metrics]
        recall = [item["recall"] for item in self.repeat_metrics]
        bacc_ci = percentile_ci(bacc)
        precision_ci = percentile_ci(precision)
        recall_ci = percentile_ci(recall)
        return {
            "model": self.model_name,
            "mode": self.mode,
            "balanced_accuracy_mean": mean(bacc),
            "balanced_accuracy_ci": bacc_ci,
            "precision_mean": mean(precision),
            "precision_ci": precision_ci,
            "recall_mean": mean(recall),
            "recall_ci": recall_ci,
        }


def jaccard(values_a: Sequence[str], values_b: Sequence[str]) -> float:
    set_a = set(values_a)
    set_b = set(values_b)
    if not set_a and not set_b:
        return 1.0
    return len(set_a & set_b) / max(1, len(set_a | set_b))


def top_feature_stability(top_feature_sets: Sequence[Sequence[str]]) -> Dict[str, float]:
    if len(top_feature_sets) < 2:
        return {"mean_pairwise_jaccard": 0.0}
    scores = []
    for i in range(len(top_feature_sets)):
        for j in range(i + 1, len(top_feature_sets)):
            scores.append(jaccard(top_feature_sets[i], top_feature_sets[j]))
    return {"mean_pairwise_jaccard": mean(scores)}


def train_model(model_name: str, x_matrix: np.ndarray, labels: np.ndarray) -> Any:
    if model_name == "logistic":
        return fit_logistic(x_matrix, labels)
    if model_name == "tree":
        return fit_tree(x_matrix, labels)
    raise ValueError(f"unknown model {model_name}")


def evaluate_model(
    rows: Sequence[Dict[str, Any]],
    labels: np.ndarray,
    model_name: str,
    mode: str,
    repeats: int,
    folds: int,
    seed: int,
    max_text_features: int,
) -> EvalResult:
    repeat_metrics: List[Dict[str, float]] = []
    score_sum = np.zeros(len(labels), dtype=float)
    score_count = np.zeros(len(labels), dtype=float)
    top_feature_sets: List[List[str]] = []
    top_feature_counts: Counter[str] = Counter()

    for repeat in range(repeats):
        repeat_scores = np.zeros(len(labels), dtype=float)
        repeat_seen = np.zeros(len(labels), dtype=bool)
        for train_idx, valid_idx in stratified_folds(labels, folds, seed + repeat * 997):
            train_rows = [rows[int(idx)] for idx in train_idx]
            valid_rows = [rows[int(idx)] for idx in valid_idx]
            encoder = FeatureEncoder(mode=mode, max_text_features=max_text_features).fit(train_rows)
            train_x = encoder.transform(train_rows)
            valid_x = encoder.transform(valid_rows)
            model = train_model(model_name, train_x, labels[train_idx])
            scores = model.predict_proba(valid_x)
            repeat_scores[valid_idx] = scores
            repeat_seen[valid_idx] = True
            score_sum[valid_idx] += scores
            score_count[valid_idx] += 1.0
            top = [name for name, _ in model.top_features(encoder.feature_names, n=5)]
            top_feature_sets.append(top)
            top_feature_counts.update(top)
        if not repeat_seen.all():
            raise RuntimeError("not all rows received validation scores")
        repeat_metrics.append(metric_at_threshold(labels, repeat_scores, threshold=0.5))

    avg_scores = score_sum / np.maximum(score_count, 1.0)
    full_encoder = FeatureEncoder(mode=mode, max_text_features=max_text_features).fit(rows)
    full_x = full_encoder.transform(rows)
    full_model = train_model(model_name, full_x, labels)
    full_scores = full_model.predict_proba(full_x)
    full_top = full_model.top_features(full_encoder.feature_names, n=15)
    return EvalResult(
        model_name=model_name,
        mode=mode,
        repeat_metrics=repeat_metrics,
        avg_scores=avg_scores,
        full_scores=full_scores,
        full_top_features=full_top,
        top_feature_sets=top_feature_sets,
        top_feature_counts=top_feature_counts,
        feature_names=full_encoder.feature_names,
    )


def threshold_scan(labels: np.ndarray, scores: np.ndarray) -> Dict[str, Any]:
    thresholds = sorted(set(float(value) for value in scores), reverse=True)
    points: List[Dict[str, Any]] = []
    positives = int(labels.sum())
    negatives = len(labels) - positives
    for threshold in thresholds:
        selected = scores >= threshold
        tp = int(np.logical_and(selected, labels == 1).sum())
        fp = int(np.logical_and(selected, labels == 0).sum())
        if tp + fp == 0:
            continue
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, positives)
        break_leak_rate = fp / max(1, negatives)
        points.append({
            "threshold": threshold,
            "repair_captured": tp,
            "break_leaked": fp,
            "net": tp - fp,
            "precision": precision,
            "recall": recall,
            "break_leak_rate": break_leak_rate,
            "selected": tp + fp,
        })
    if not points:
        empty = {"threshold": 1.0, "repair_captured": 0, "break_leaked": 0, "net": 0, "precision": 0.0, "recall": 0.0, "break_leak_rate": 0.0, "selected": 0}
        return {"max_precision": empty, "best_net": empty, "positive_net_exists": False}
    max_precision = sorted(points, key=lambda item: (item["precision"], item["net"], item["recall"]), reverse=True)[0]
    best_net = sorted(points, key=lambda item: (item["net"], item["precision"], item["recall"]), reverse=True)[0]
    useful_points = [item for item in points if item["recall"] >= 0.10]
    best_net_recall10 = sorted(useful_points, key=lambda item: (item["net"], item["precision"], item["recall"]), reverse=True)[0] if useful_points else None
    return {
        "max_precision": max_precision,
        "best_net": best_net,
        "best_net_recall10": best_net_recall10,
        "positive_net_exists": best_net["net"] > 0,
        "positive_net_recall10_exists": bool(best_net_recall10 and best_net_recall10["net"] > 0),
    }


def select_rows(rows: Sequence[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], np.ndarray]:
    selected: List[Dict[str, Any]] = []
    labels: List[int] = []
    for row in rows:
        if not row.get("bottom2"):
            continue
        repair = bool(row.get("distill_positive_real_a11y"))
        broke = bool(row.get("V", {}).get("success") and not row.get("real_a11y", {}).get("success"))
        if repair:
            selected.append(row)
            labels.append(1)
        elif broke:
            selected.append(row)
            labels.append(0)
    return selected, np.asarray(labels, dtype=int)


def compact_features(row: Dict[str, Any]) -> Dict[str, Any]:
    numeric, categorical, tokens = build_raw_features(row, "full")
    keep_numeric = {
        key: numeric[key]
        for key in sorted(numeric)
        if key in {
            "step_idx", "task_k", "step_frac", "num_controls", "log_num_controls",
            "goal_chars", "goal_words", "v_pred_has_coord", "v_pred_x_norm",
            "v_pred_y_norm", "p_hat_heldout_bucket", "critical_rank",
            "p_hat_source_count", "v_pred_text_chars",
        }
    }
    return {
        "numeric": keep_numeric,
        "categorical": categorical,
        "goal_tokens_top": tokens[:30],
    }


def leakage_audit() -> List[str]:
    return [
        "Excluded label/outcome fields: distill_positive_*, preserve_v_correct, V.success, V.bucket, V.reward, V_bucket, real_a11y.success, real_a11y.bucket, real_a11y.reward, placebo_a11y.*.",
        "Excluded GT-derived fields as primary features: gt_action and critical_feature target/action/bbox bins. critical_feature is used only upstream for frozen bottom-2 tagging and held-out p_hat, not as classifier input.",
        "Allowed difficulty proxy: p_hat_heldout_bucket / p_hat_source / p_hat_source_count / critical_rank, because the spec permits the frozen held-out p_i used for critical tagging and it is not estimated from this state's own outcome.",
        "Unavailable in the artifact and therefore not used: control density near target, similar-element density, control-text length, a11y-vs-V control deltas, logits/confidence/entropy.",
    ]


def render_point(point: Optional[Dict[str, Any]]) -> str:
    if not point:
        return "n/a"
    return (
        f"threshold={fmt_float(point['threshold'], 4)}, precision={fmt_float(point['precision'])}, "
        f"recall={fmt_float(point['recall'])}, REPAIR captured={point['repair_captured']}, "
        f"BREAK leaked={point['break_leaked']}, net={point['net']}"
    )


def model_table(results: Sequence[EvalResult]) -> List[str]:
    lines = ["| model | feature set | balanced acc mean | 95% CI | REPAIR precision | REPAIR recall |", "|---|---|---:|---:|---:|---:|"]
    for result in results:
        summary = result.summary()
        bci = summary["balanced_accuracy_ci"]
        pci = summary["precision_ci"]
        rci = summary["recall_ci"]
        lines.append(
            "| {model} | {mode} | {b:.3f} | [{blo:.3f}, {bhi:.3f}] | {p:.3f} [{plo:.3f}, {phi:.3f}] | {r:.3f} [{rlo:.3f}, {rhi:.3f}] |".format(
                model=summary["model"],
                mode=summary["mode"],
                b=summary["balanced_accuracy_mean"],
                blo=bci[0],
                bhi=bci[1],
                p=summary["precision_mean"],
                plo=pci[0],
                phi=pci[1],
                r=summary["recall_mean"],
                rlo=rci[0],
                rhi=rci[1],
            )
        )
    return lines


def top_features_lines(result: EvalResult, max_items: int = 12) -> List[str]:
    lines = [f"### {result.model_name} / {result.mode}", "", "| feature | weight / importance |", "|---|---:|"]
    for feature, value in result.full_top_features[:max_items]:
        lines.append(f"| `{feature}` | {value:.4f} |")
    return lines


def render_report(
    args: argparse.Namespace,
    rows: Sequence[Dict[str, Any]],
    labels: np.ndarray,
    results: Dict[Tuple[str, str], EvalResult],
    scans: Dict[Tuple[str, str], Dict[str, Any]],
    gate: str,
    gate_reason: str,
) -> str:
    repair_count = int(labels.sum())
    break_count = len(labels) - repair_count
    full_results = [results[("logistic", "full")], results[("tree", "full")]]
    difficulty_results = [
        results[("logistic", "difficulty_only")],
        results[("tree", "difficulty_only")],
        results[("logistic", "no_difficulty")],
        results[("tree", "no_difficulty")],
    ]
    best_full = max(full_results, key=lambda item: item.summary()["balanced_accuracy_mean"])
    best_scan = scans[(best_full.model_name, best_full.mode)]
    lines: List[str] = []
    lines.append("# Critical-Step Separability Diagnostic")
    lines.append("")
    lines.append("Diagnostic only: no teacher calls, no distillation training. Labels are used only to evaluate whether existing bottom-2 critical REPAIR and BREAK states are separable.")
    lines.append("")
    lines.append("## Inputs")
    lines.append("")
    lines.append(f"- input artifact: `{args.input}`")
    lines.append(f"- selected bottom-2 critical states: `{len(labels)}`")
    lines.append(f"- REPAIR states: `{repair_count}`")
    lines.append(f"- BREAK states: `{break_count}`")
    lines.append(f"- CV: `{args.repeats}` repeats x `{args.folds}` stratified folds")
    lines.append(f"- majority baseline balanced accuracy: `0.500`")
    lines.append("")
    lines.append("## Leakage Audit")
    lines.append("")
    for item in leakage_audit():
        lines.append(f"- {item}")
    lines.append("")
    lines.append("Feature-set definitions:")
    lines.append("")
    lines.append("- `full`: observable goal/step/a11y-count/V-output features plus allowed held-out p_i difficulty proxy.")
    lines.append("- `difficulty_only`: held-out p_i difficulty proxy only.")
    lines.append("- `no_difficulty`: observable goal/step/a11y-count/V-output features without held-out p_i.")
    lines.append("")
    lines.append("## Test A: Feature-Based Separability")
    lines.append("")
    lines.extend(model_table(full_results))
    lines.append("")
    lines.append("## Test B: Resampling Consistency")
    lines.append("")
    lines.append("| model | feature set | balanced acc p2.5-p97.5 | top-5 feature Jaccard | most frequent top features |")
    lines.append("|---|---|---:|---:|---|")
    for result in full_results:
        bacc = [item["balanced_accuracy"] for item in result.repeat_metrics]
        bci = percentile_ci(bacc)
        stability = top_feature_stability(result.top_feature_sets)
        common = ", ".join(f"`{name}` ({count})" for name, count in result.top_feature_counts.most_common(5))
        lines.append(
            f"| {result.model_name} | {result.mode} | [{bci[0]:.3f}, {bci[1]:.3f}] | {stability['mean_pairwise_jaccard']:.3f} | {common} |"
        )
    lines.append("")
    lines.append("## Test C: Difficulty-Only Check")
    lines.append("")
    lines.extend(model_table(difficulty_results))
    lines.append("")
    lines.append("Interpretation: `difficulty_only` measures what the frozen held-out p_i axis can do by itself; `no_difficulty` asks whether other student-observable structure remains after removing that axis.")
    lines.append("")
    lines.append("## Section 5: Selective Operating Points")
    lines.append("")
    lines.append("Thresholds are scanned on repeated-CV out-of-fold scores. Predicted REPAIR means the gate would apply the real-a11y correction.")
    lines.append("")
    lines.append("| model | feature set | max precision operating point | best net operating point | best net with recall >= 0.10 |")
    lines.append("|---|---|---|---|---|")
    for key in [("logistic", "full"), ("tree", "full")]:
        scan = scans[key]
        lines.append(
            f"| {key[0]} | {key[1]} | {render_point(scan['max_precision'])} | {render_point(scan['best_net'])} | {render_point(scan.get('best_net_recall10'))} |"
        )
    lines.append("")
    lines.append(f"Best full-model operating point used for gate: `{best_full.model_name}` / `{best_full.mode}` -> {render_point(best_scan['best_net'])}.")
    lines.append("")
    lines.append("## Feature Importances")
    lines.append("")
    for result in full_results:
        lines.extend(top_features_lines(result))
        lines.append("")
    lines.append("## Gate Verdict")
    lines.append("")
    lines.append(f"**{gate}**")
    lines.append("")
    lines.append(gate_reason)
    lines.append("")
    lines.append("STOP for review.")
    return "\n".join(lines) + "\n"


def decide_gate(results: Dict[Tuple[str, str], EvalResult], scans: Dict[Tuple[str, str], Dict[str, Any]]) -> Tuple[str, str]:
    full_results = [results[("logistic", "full")], results[("tree", "full")]]
    best_full = max(full_results, key=lambda item: item.summary()["balanced_accuracy_mean"])
    best_summary = best_full.summary()
    bacc = float(best_summary["balanced_accuracy_mean"])
    ci_low = float(best_summary["balanced_accuracy_ci"][0])
    scan = scans[(best_full.model_name, best_full.mode)]
    positive_net = bool(scan["positive_net_exists"])
    positive_net_recall10 = bool(scan.get("positive_net_recall10_exists"))

    difficulty_best = max(
        [results[("logistic", "difficulty_only")], results[("tree", "difficulty_only")]],
        key=lambda item: item.summary()["balanced_accuracy_mean"],
    ).summary()["balanced_accuracy_mean"]
    no_diff_best = max(
        [results[("logistic", "no_difficulty")], results[("tree", "no_difficulty")]],
        key=lambda item: item.summary()["balanced_accuracy_mean"],
    ).summary()["balanced_accuracy_mean"]

    if bacc < 0.62 or ci_low <= 0.5:
        return (
            "NOT SEPARABLE",
            f"Best full feature-set classifier is {best_full.model_name} with balanced acc {bacc:.3f} and CI low {ci_low:.3f}; this does not meet the >=0.62 with CI excluding 0.5 separability criterion.",
        )
    if not positive_net or not positive_net_recall10:
        return (
            "NOT SEPARABLE",
            "The classifier clears the accuracy bar, but threshold scanning does not find a useful positive-net operating point on bottom-2 critical steps.",
        )
    if difficulty_best >= bacc - 0.03 and no_diff_best < 0.62:
        return (
            "SEPARABLE-BUT-DIFFICULTY-ONLY",
            f"The signal is mostly the held-out p_i difficulty axis: best difficulty-only balanced acc {difficulty_best:.3f} is close to full {bacc:.3f}, while no-difficulty best is {no_diff_best:.3f}. The method is usable only as a repairable/easier-critical-step gate.",
        )
    return (
        "SEPARABLE + SELECTIVE-FEASIBLE",
        f"Best full classifier ({best_full.model_name}) reaches balanced acc {bacc:.3f} with CI low {ci_low:.3f}, and threshold scanning finds a positive-net operating point. Phase 1 should use gated positives with anti-regression, not raw teacher outputs.",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--report", default=DEFAULT_REPORT)
    parser.add_argument("--per_state", default=DEFAULT_PER_STATE)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--max_text_features", type=int, default=120)
    args = parser.parse_args()

    all_rows = read_jsonl(Path(args.input))
    rows, labels = select_rows(all_rows)
    if int(labels.sum()) != 136 or len(labels) - int(labels.sum()) != 454:
        print(f"warning: expected REPAIR=136/BREAK=454, got REPAIR={int(labels.sum())}/BREAK={len(labels)-int(labels.sum())}")
    if len(rows) < 10 or len(np.unique(labels)) != 2:
        raise SystemExit("need both REPAIR and BREAK bottom-2 rows")

    results: Dict[Tuple[str, str], EvalResult] = {}
    for mode in ("full", "difficulty_only", "no_difficulty"):
        for model_name in ("logistic", "tree"):
            print(f"running {model_name}/{mode}...", flush=True)
            results[(model_name, mode)] = evaluate_model(
                rows,
                labels,
                model_name=model_name,
                mode=mode,
                repeats=args.repeats,
                folds=args.folds,
                seed=args.seed,
                max_text_features=args.max_text_features,
            )

    scans = {(model_name, "full"): threshold_scan(labels, results[(model_name, "full")].avg_scores) for model_name in ("logistic", "tree")}
    gate, gate_reason = decide_gate(results, scans)
    report = render_report(args, rows, labels, results, scans, gate, gate_reason)
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")

    per_state_rows = []
    logistic_full = results[("logistic", "full")]
    tree_full = results[("tree", "full")]
    for row, label, log_oof, tree_oof, log_full, tree_full_score in zip(
        rows,
        labels,
        logistic_full.avg_scores,
        tree_full.avg_scores,
        logistic_full.full_scores,
        tree_full.full_scores,
    ):
        per_state_rows.append({
            "state_id": row.get("state_id"),
            "episode_id": row.get("episode_id"),
            "step_idx": row.get("step_idx"),
            "class": "REPAIR" if int(label) == 1 else "BREAK",
            "bottom1": bool(row.get("bottom1")),
            "bottom2": bool(row.get("bottom2")),
            "features": compact_features(row),
            "logistic_oof_score": float(log_oof),
            "tree_oof_score": float(tree_oof),
            "logistic_full_score": float(log_full),
            "tree_full_score": float(tree_full_score),
        })
    write_jsonl(Path(args.per_state), per_state_rows)

    print(json.dumps({
        "report": args.report,
        "per_state": args.per_state,
        "gate": gate,
        "repair": int(labels.sum()),
        "break": int(len(labels) - labels.sum()),
        "best_full_logistic_bacc": results[("logistic", "full")].summary()["balanced_accuracy_mean"],
        "best_full_tree_bacc": results[("tree", "full")].summary()["balanced_accuracy_mean"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()