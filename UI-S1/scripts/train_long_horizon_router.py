#!/usr/bin/env python3
"""Train a lightweight selective-memory router from behavior-derived labels."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_recall_fscore_support
from sklearn.pipeline import Pipeline


JsonDict = dict[str, Any]
ROUTES = ["use_no_history", "use_segment_summary", "use_full_history", "escalate_or_replan", "avoid_segment_summary"]
MEMORY_ROUTES = {"use_segment_summary", "use_full_history"}


def iter_jsonl(path: Path) -> Iterable[JsonDict]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def split_episode_ids(examples: list[JsonDict], seed: int) -> dict[str, str]:
    episode_ids = sorted({str(example["features"].get("episode_id")) for example in examples})
    rng = random.Random(seed)
    rng.shuffle(episode_ids)
    train_end = int(0.8 * len(episode_ids))
    dev_end = int(0.9 * len(episode_ids))
    split_by_episode = {}
    for episode_id in episode_ids[:train_end]:
        split_by_episode[episode_id] = "train"
    for episode_id in episode_ids[train_end:dev_end]:
        split_by_episode[episode_id] = "dev"
    for episode_id in episode_ids[dev_end:]:
        split_by_episode[episode_id] = "test"
    return split_by_episode


def numeric(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except Exception:
        return default


def featurize(example: JsonDict, include_action_type: bool = False) -> dict[str, Any]:
    features = example.get("features", {})
    step_index = numeric(features.get("step_index"))
    total_steps = max(1.0, numeric(features.get("total_steps"), 1.0))
    prev_segments = numeric(features.get("prev_segments"))
    segment_len = numeric(features.get("segment_len_so_far"))
    carried_count = numeric(features.get("carried_value_count"))
    row: dict[str, Any] = {
        "model=" + str(example.get("model_key", "unknown")): 1,
        "thinking=" + str(example.get("thinking_mode", "unknown")): 1,
        "case=" + str(example.get("case_kind", "unknown")): 1,
        "memory_strength=" + str(features.get("memory_strength", "unknown")): 1,
        "capability=" + str(features.get("dominant_capability", "unknown")): 1,
        "is_long_horizon": int(bool(features.get("is_long_horizon"))),
        "step_index": step_index,
        "step_frac": step_index / total_steps,
        "total_steps": total_steps,
        "prev_segments": prev_segments,
        "segment_len_so_far": segment_len,
        "carried_value_count": carried_count,
        "has_carried_values": int(carried_count > 0),
        "late_step": int(step_index >= 10),
        "many_prev_segments": int(prev_segments >= 2),
        "deep_segment": int(segment_len >= 4),
    }
    if include_action_type:
        row["gt_action_type=" + str(features.get("gt_action_type", "unknown"))] = 1
    for value in features.get("carried_values", [])[:4]:
        text = str(value).lower().strip()
        if text:
            row["carried_token_count"] = row.get("carried_token_count", 0.0) + len(text.split())
    return row


def build_dataset(examples: list[JsonDict], include_action_type: bool) -> tuple[list[dict[str, Any]], list[str], list[str]]:
    x_rows = [featurize(example, include_action_type=include_action_type) for example in examples]
    y = [str(example["route_label"]) for example in examples]
    episode_ids = [str(example["features"].get("episode_id")) for example in examples]
    return x_rows, y, episode_ids


def make_model(model_type: str, seed: int) -> Pipeline:
    if model_type == "logistic":
        classifier = LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            solver="saga",
            random_state=seed,
        )
    elif model_type == "forest":
        classifier = RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=3,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=seed,
        )
    else:
        raise ValueError(f"unknown model_type: {model_type}")
    return Pipeline([("vectorizer", DictVectorizer(sparse=False)), ("classifier", classifier)])


def binary_memory_metrics(y_true: list[str], y_pred: list[str]) -> JsonDict:
    true_mem = [label in MEMORY_ROUTES for label in y_true]
    pred_mem = [label in MEMORY_ROUTES for label in y_pred]
    tp = sum(t and p for t, p in zip(true_mem, pred_mem))
    fp = sum((not t) and p for t, p in zip(true_mem, pred_mem))
    fn = sum(t and (not p) for t, p in zip(true_mem, pred_mem))
    tn = sum((not t) and (not p) for t, p in zip(true_mem, pred_mem))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": precision, "recall": recall, "f1": f1}


def evaluate(name: str, model: Pipeline, x_rows: list[dict[str, Any]], y: list[str], examples: list[JsonDict]) -> JsonDict:
    y_pred = list(model.predict(x_rows))
    labels = [label for label in ROUTES if label in set(y) or label in set(y_pred)]
    precision, recall, f1, support = precision_recall_fscore_support(y, y_pred, labels=labels, zero_division=0)
    per_label = {
        label: {"precision": float(p), "recall": float(r), "f1": float(f), "support": int(s)}
        for label, p, r, f, s in zip(labels, precision, recall, f1, support)
    }
    errors = []
    for example, true_label, pred_label in zip(examples, y, y_pred):
        if true_label != pred_label:
            errors.append(
                {
                    "episode_id": example["features"].get("episode_id"),
                    "case_id": example.get("case_id"),
                    "model_key": example.get("model_key"),
                    "thinking_mode": example.get("thinking_mode"),
                    "case_kind": example.get("case_kind"),
                    "true": true_label,
                    "pred": pred_label,
                    "reason": example.get("route_reason"),
                    "features": example.get("features"),
                }
            )
    threshold_sweep = []
    if hasattr(model.named_steps["classifier"], "predict_proba"):
        probabilities = model.predict_proba(x_rows)
        classes = list(model.named_steps["classifier"].classes_)
        memory_indices = [index for index, label in enumerate(classes) if label in MEMORY_ROUTES]
        memory_scores = probabilities[:, memory_indices].sum(axis=1) if memory_indices else np.zeros(len(y))
        true_memory = np.array([label in MEMORY_ROUTES for label in y], dtype=bool)
        for threshold in [0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.9]:
            pred_memory = memory_scores >= threshold
            tp = int(np.sum(true_memory & pred_memory))
            fp = int(np.sum((~true_memory) & pred_memory))
            fn = int(np.sum(true_memory & (~pred_memory)))
            precision_value = tp / (tp + fp) if tp + fp else 0.0
            recall_value = tp / (tp + fn) if tp + fn else 0.0
            f1_value = 2 * precision_value * recall_value / (precision_value + recall_value) if precision_value + recall_value else 0.0
            threshold_sweep.append({
                "threshold": threshold,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": precision_value,
                "recall": recall_value,
                "f1": f1_value,
                "predicted_positive": int(np.sum(pred_memory)),
            })
    return {
        "split": name,
        "n": len(y),
        "accuracy": float(np.mean([a == b for a, b in zip(y, y_pred)])) if y else 0.0,
        "macro_f1": float(f1_score(y, y_pred, average="macro", zero_division=0)) if y else 0.0,
        "weighted_f1": float(f1_score(y, y_pred, average="weighted", zero_division=0)) if y else 0.0,
        "label_counts": dict(Counter(y)),
        "prediction_counts": dict(Counter(y_pred)),
        "per_label": per_label,
        "memory_binary": binary_memory_metrics(y, y_pred),
        "memory_threshold_sweep": threshold_sweep,
        "confusion_labels": labels,
        "confusion_matrix": confusion_matrix(y, y_pred, labels=labels).tolist(),
        "errors": errors[:300],
    }


def write_report(path: Path, metadata: JsonDict, results: dict[str, JsonDict]) -> None:
    lines = ["# Long-Horizon Router Training Report", ""]
    lines.append("## Setup")
    lines.append("")
    for key, value in metadata.items():
        lines.append(f"- {key}: `{value}`")
    lines.append("")
    for split, result in results.items():
        lines.append(f"## {split.title()} Metrics")
        lines.append("")
        lines.append(f"- n: {result['n']}")
        lines.append(f"- accuracy: {result['accuracy']:.4f}")
        lines.append(f"- macro_f1: {result['macro_f1']:.4f}")
        lines.append(f"- weighted_f1: {result['weighted_f1']:.4f}")
        mem = result["memory_binary"]
        lines.append(f"- memory precision/recall/f1: {mem['precision']:.4f} / {mem['recall']:.4f} / {mem['f1']:.4f}")
        lines.append("")
        if result.get("memory_threshold_sweep"):
            lines.append("Memory probability threshold sweep:")
            lines.append("")
            lines.append("| threshold | predicted | precision | recall | f1 | tp | fp | fn |")
            lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|")
            for item in result["memory_threshold_sweep"]:
                lines.append(
                    f"| {item['threshold']:.2f} | {item['predicted_positive']} | {item['precision']:.4f} | "
                    f"{item['recall']:.4f} | {item['f1']:.4f} | {item['tp']} | {item['fp']} | {item['fn']} |"
                )
            lines.append("")
        lines.append("| label | support | precision | recall | f1 |")
        lines.append("|---|---:|---:|---:|---:|")
        for label, metrics in result["per_label"].items():
            lines.append(f"| {label} | {metrics['support']} | {metrics['precision']:.4f} | {metrics['recall']:.4f} | {metrics['f1']:.4f} |")
        lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append("The important operating point is not route-label accuracy alone. The router is useful only if it keeps memory precision high while recovering a meaningful fraction of segment/full-history rescue cases. Use the memory binary precision/recall and per-label `use_segment_summary` / `use_full_history` recall to choose thresholds or a two-stage classifier.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a selective-memory router from routing examples")
    parser.add_argument("--data", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-type", choices=["logistic", "forest"], default="logistic")
    parser.add_argument("--include-action-type", action="store_true")
    parser.add_argument("--seed", type=int, default=17)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_path = Path(args.data)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    examples = list(iter_jsonl(data_path))
    split_by_episode = split_episode_ids(examples, args.seed)
    split_examples: dict[str, list[JsonDict]] = defaultdict(list)
    for example in examples:
        split = split_by_episode[str(example["features"].get("episode_id"))]
        split_examples[split].append(example)
    x_train, y_train, _ = build_dataset(split_examples["train"], include_action_type=args.include_action_type)
    model = make_model(args.model_type, args.seed)
    model.fit(x_train, y_train)
    results = {}
    for split in ["train", "dev", "test"]:
        x_rows, y, _episode_ids = build_dataset(split_examples[split], include_action_type=args.include_action_type)
        results[split] = evaluate(split, model, x_rows, y, split_examples[split])
    metadata = {
        "data": str(data_path),
        "examples": len(examples),
        "model_type": args.model_type,
        "include_action_type": args.include_action_type,
        "seed": args.seed,
        "train_examples": len(split_examples["train"]),
        "dev_examples": len(split_examples["dev"]),
        "test_examples": len(split_examples["test"]),
    }
    joblib.dump(model, output_dir / "router_model.joblib")
    (output_dir / "metrics.json").write_text(json.dumps({"metadata": metadata, "results": results}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for split, result in results.items():
        with (output_dir / f"{split}_errors.jsonl").open("w", encoding="utf-8") as handle:
            for row in result["errors"]:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    write_report(output_dir / "router_training_report.md", metadata, results)
    print(f"trained model={args.model_type} examples={len(examples)} output={output_dir}")


if __name__ == "__main__":
    main()