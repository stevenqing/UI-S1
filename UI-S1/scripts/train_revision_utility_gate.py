#!/usr/bin/env python3
"""Train a lightweight student-conditioned revision utility gate.

Labels use the frozen matcher for research evaluation, but model features exclude
GT actions and matcher outcomes. Episodes are split disjointly to prevent the
same GUI task from appearing in train and test.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.rl_feasibility_sampling import action_key  # noqa: E402


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def paired_key(row: Mapping[str, Any]) -> tuple[str, int]:
    return str(row["correction_id"]), int(row["step_idx"])


def split_name(episode_id: str, seed: int) -> str:
    value = int(hashlib.sha256(f"{seed}:{episode_id}".encode()).hexdigest()[:16], 16) % 10
    return "train" if value < 8 else "dev" if value == 8 else "test"


def action_type(action: Any) -> str:
    return str((action or {}).get("action") or "unparsed").lower()


def outcome(student_correct: bool, revision_correct: bool) -> str:
    if not student_correct and revision_correct:
        return "rescue"
    if student_correct and not revision_correct:
        return "regress"
    if student_correct and revision_correct:
        return "preserve_correct"
    return "unresolved"


def build_examples(source_rows: Sequence[Mapping[str, Any]], eval_rows: Sequence[Mapping[str, Any]], seed: int) -> list[dict[str, Any]]:
    evaluated = {paired_key(row): row for row in eval_rows}
    if len(evaluated) != len(eval_rows):
        raise ValueError("duplicate evaluation key")
    examples = []
    for source in source_rows:
        key = paired_key(source)
        if key not in evaluated:
            raise ValueError(f"missing student evaluation: {key}")
        student = evaluated[key]
        revision_key = str(source["chosen_action_key"])
        actor_key = action_key(source.get("actor_action"), 25)
        student_key = str(student.get("student_action_key") or "__unparsed__")
        student_correct = bool(student["student_correct"])
        revision_correct = bool(source["revision_correct"])
        category = outcome(student_correct, revision_correct)
        num_steps = max(1, int(source["num_steps"]))
        step_idx = int(source["step_idx"])
        examples.append({
            "episode_id": str(source["episode_id"]),
            "split": split_name(str(source["episode_id"]), seed),
            "outcome": category,
            "label": int(category == "rescue"),
            "regress": int(category == "regress"),
            "continuous": {
                "corrector_confidence": float(source.get("correction_confidence") or 0.0),
                "relative_step": step_idx / max(1, num_steps - 1),
                "log_num_steps": math.log1p(num_steps),
                "log_prefix_wrong": math.log1p(int(source["prefix_wrong_count"])),
                "prefix_clean": float(bool(source["prefix_clean"])),
                "revision_changed": float(bool(source["revision_changed_from_actor"])),
                "student_parse_ok": float(bool(student["parse_ok"])),
                "revision_student_same": float(revision_key == student_key),
                "actor_student_same": float(actor_key == student_key),
                "actor_revision_same": float(actor_key == revision_key),
            },
            "categorical": {
                "actor": str(source["actor"]),
                "revision_type": action_type(source.get("revision_action")),
                "actor_type": action_type(source.get("actor_action")),
                "student_type": action_type(student.get("student_action")),
            },
        })
    return examples


def build_matrix(examples: Sequence[Mapping[str, Any]], train_examples: Sequence[Mapping[str, Any]]) -> tuple[torch.Tensor, list[str]]:
    continuous_names = sorted(train_examples[0]["continuous"])
    categorical_values = {
        name: sorted({str(row["categorical"][name]) for row in train_examples})
        for name in sorted(train_examples[0]["categorical"])
    }
    means = {
        name: sum(float(row["continuous"][name]) for row in train_examples) / len(train_examples)
        for name in continuous_names
    }
    stds = {}
    for name in continuous_names:
        variance = sum((float(row["continuous"][name]) - means[name]) ** 2 for row in train_examples) / len(train_examples)
        stds[name] = max(math.sqrt(variance), 1e-6)
    feature_names = [f"continuous:{name}" for name in continuous_names]
    for name, values in categorical_values.items():
        feature_names.extend(f"categorical:{name}={value}" for value in values)
        feature_names.append(f"categorical:{name}=__other__")

    vectors = []
    for row in examples:
        vector = [
            (float(row["continuous"][name]) - means[name]) / stds[name]
            for name in continuous_names
        ]
        for name, values in categorical_values.items():
            value = str(row["categorical"][name])
            vector.extend(float(value == candidate) for candidate in values)
            vector.append(float(value not in values))
        vectors.append(vector)
    return torch.tensor(vectors, dtype=torch.float32), feature_names


def average_ranks(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        rank = (start + 1 + end) / 2.0
        for idx in order[start:end]:
            ranks[idx] = rank
        start = end
    return ranks


def auc(labels: Sequence[int], scores: Sequence[float]) -> float | None:
    positives = sum(labels)
    negatives = len(labels) - positives
    if not positives or not negatives:
        return None
    ranks = average_ranks(scores)
    rank_sum = sum(rank for rank, label in zip(ranks, labels) if label)
    return (rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def average_precision(labels: Sequence[int], scores: Sequence[float]) -> float | None:
    positives = sum(labels)
    if not positives:
        return None
    order = sorted(range(len(scores)), key=lambda idx: scores[idx], reverse=True)
    true_positives = 0
    precision_sum = 0.0
    for rank, idx in enumerate(order, 1):
        if labels[idx]:
            true_positives += 1
            precision_sum += true_positives / rank
    return precision_sum / positives


def operating_points(examples: Sequence[Mapping[str, Any]], scores: Sequence[float]) -> list[dict[str, Any]]:
    rows = list(zip(examples, scores))
    points = []
    specifications = [("threshold", value) for value in (0.5, 0.7, 0.8, 0.9, 0.95)]
    specifications += [("top_fraction", value) for value in (0.01, 0.025, 0.05, 0.10, 0.20)]
    for kind, value in specifications:
        if kind == "threshold":
            selected = [item for item in rows if item[1] >= value]
            label = f"p>={value:.2f}"
        else:
            count = max(1, int(round(value * len(rows))))
            selected = sorted(rows, key=lambda item: item[1], reverse=True)[:count]
            label = f"top_{100*value:.1f}%"
        counts = Counter(item[0]["outcome"] for item in selected)
        accepted = len(selected)
        points.append({
            "selector": label,
            "accepted": accepted,
            "coverage": accepted / len(rows),
            "rescue_precision": counts["rescue"] / max(1, accepted),
            "regress_rate": counts["regress"] / max(1, accepted),
            "net_utility": (counts["rescue"] - counts["regress"]) / max(1, accepted),
            "outcomes": dict(counts),
        })
    return points


def evaluate_split(examples: Sequence[Mapping[str, Any]], scores: Sequence[float]) -> dict[str, Any]:
    labels = [int(row["label"]) for row in examples]
    counts = Counter(row["outcome"] for row in examples)
    return {
        "rows": len(examples),
        "outcomes": dict(counts),
        "rescue_base_rate": sum(labels) / len(labels),
        "roc_auc": auc(labels, scores),
        "average_precision": average_precision(labels, scores),
        "operating_points": operating_points(examples, scores),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--causal-input", required=True)
    parser.add_argument("--student-eval", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=800)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    examples = build_examples(read_jsonl(Path(args.causal_input)), read_jsonl(Path(args.student_eval)), args.seed)
    splits = {name: [row for row in examples if row["split"] == name] for name in ("train", "dev", "test")}
    if any(not rows for rows in splits.values()):
        raise ValueError({name: len(rows) for name, rows in splits.items()})
    train = splits["train"]
    matrices = {}
    feature_names = None
    for name, rows in splits.items():
        matrix, names = build_matrix(rows, train)
        matrices[name] = matrix
        feature_names = names if feature_names is None else feature_names
        if names != feature_names:
            raise AssertionError("feature schema mismatch")

    y_train = torch.tensor([float(row["label"]) for row in train])
    model = torch.nn.Linear(matrices["train"].shape[1], 1)
    positives = y_train.sum().item()
    pos_weight = torch.tensor([(len(y_train) - positives) / max(positives, 1.0)])
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    best_state = None
    best_dev_ap = -1.0
    patience = 0
    history = []
    for epoch in range(args.epochs):
        model.train()
        logits = model(matrices["train"]).squeeze(1)
        loss = F.binary_cross_entropy_with_logits(logits, y_train, pos_weight=pos_weight)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0 or epoch == args.epochs - 1:
            model.eval()
            with torch.no_grad():
                dev_scores = torch.sigmoid(model(matrices["dev"]).squeeze(1)).tolist()
            dev_ap = average_precision([int(row["label"]) for row in splits["dev"]], dev_scores) or 0.0
            history.append({"epoch": epoch, "loss": float(loss.item()), "dev_ap": dev_ap})
            if dev_ap > best_dev_ap + 1e-6:
                best_dev_ap = dev_ap
                best_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
                patience = 0
            else:
                patience += 1
            if patience >= 20:
                break
    if best_state is None:
        raise RuntimeError("no model state selected")
    model.load_state_dict(best_state)
    model.eval()

    evaluations = {}
    with torch.no_grad():
        for name, rows in splits.items():
            scores = torch.sigmoid(model(matrices[name]).squeeze(1)).tolist()
            evaluations[name] = evaluate_split(rows, scores)
    coefficients = {
        feature: float(weight)
        for feature, weight in zip(feature_names or [], model.weight.detach().squeeze(0).tolist())
    }
    summary = {
        "version": "student-conditioned-revision-utility-gate-v1",
        "features_exclude_gt_and_matcher": True,
        "label": "student_wrong_and_revision_correct",
        "episode_disjoint_split": True,
        "split_rows": {name: len(rows) for name, rows in splits.items()},
        "best_dev_ap": best_dev_ap,
        "training_history": history,
        "evaluations": evaluations,
        "coefficients": dict(sorted(coefficients.items(), key=lambda item: abs(item[1]), reverse=True)),
        "bias": float(model.bias.detach().item()),
    }
    out_dir = Path(args.output_dir)
    write_json(out_dir / "summary.json", summary)

    test = evaluations["test"]
    lines = [
        "# Student-Conditioned Revision Utility Gate",
        "",
        "Episode-disjoint logistic gate. Features exclude GT actions and matcher outcomes; labels use the matcher only for research supervision/evaluation.",
        "",
        f"Test rows: {test['rows']}. Rescue base rate: {100*test['rescue_base_rate']:.2f}%.",
        f"ROC-AUC: **{test['roc_auc']:.4f}**. Average precision: **{test['average_precision']:.4f}**.",
        "",
        "| selector | coverage | rescue precision | regress rate | net accepted utility |",
        "|---|---:|---:|---:|---:|",
    ]
    for point in test["operating_points"]:
        lines.append(
            f"| {point['selector']} | {100*point['coverage']:.2f}% | "
            f"{100*point['rescue_precision']:.2f}% | {100*point['regress_rate']:.2f}% | "
            f"{100*point['net_utility']:+.2f}pp |"
        )
    lines.extend(["", "## Largest Coefficients", "", "| feature | coefficient |", "|---|---:|"])
    for feature, value in list(summary["coefficients"].items())[:15]:
        lines.append(f"| {feature} | {value:+.4f} |")
    lines.append("")
    (out_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({
        "test_rows": test["rows"],
        "rescue_base_rate": test["rescue_base_rate"],
        "roc_auc": test["roc_auc"],
        "average_precision": test["average_precision"],
        "report": str(out_dir / "report.md"),
    }, indent=2))


if __name__ == "__main__":
    main()
