#!/usr/bin/env python3
"""Build episode-disjoint multimodal data for calibrated revision-rescue ranking."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def paired_key(row: Mapping[str, Any]) -> tuple[str, int]:
    return str(row["correction_id"]), int(row["step_idx"])


def split_name(episode_id: str, seed: int) -> str:
    value = int(hashlib.sha256(f"{seed}:{episode_id}".encode()).hexdigest()[:16], 16) % 10
    return "train" if value < 8 else "dev" if value == 8 else "test"


def compact(action: Any) -> str:
    return json.dumps(action or {"action": "unparsed"}, ensure_ascii=False, separators=(",", ":"))


def outcome(student_correct: bool, revision_correct: bool) -> str:
    if not student_correct and revision_correct:
        return "rescue"
    if student_correct and not revision_correct:
        return "regress"
    if student_correct and revision_correct:
        return "both_correct"
    return "both_wrong"


def prompt(source: Mapping[str, Any], student: Mapping[str, Any], rationale: str) -> str:
    history = "\n".join(str(item) for item in source.get("history", [])) or "None"
    return f"""<image>
You are a conservative GUI revision utility ranker. Compare an existing starting-student action with a global revision. Do not invent another action.

User goal:
{source['goal']}

Action history:
{history}

Candidate packet:
- Source actor: {source['actor']}
- Actor candidate: {compact(source.get('actor_action'))}
- Starting-student candidate: {compact(student.get('student_action'))}
- Global revision candidate: {compact(source.get('revision_action'))}
- Corrector rationale: {rationale or 'Not provided'}

Question: Should the global revision replace the starting-student candidate because it fixes a specific student error?

Answer exactly YES or NO. Say YES only when the revision is clearly a grounded repair; otherwise say NO."""


def balanced_train(
    rows: Sequence[Mapping[str, Any]], positives: int, negatives: int, seed: int, negative_mode: str
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    groups: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        groups.setdefault(str(row["utility_outcome"]), []).append(row)
    positive_group = list(groups.get("rescue", []))
    if not positive_group:
        raise ValueError("no rescue rows")
    rng.shuffle(positive_group)
    output = []
    for idx in range(positives):
        row = dict(positive_group[idx % len(positive_group)])
        row["sample_id"] = f"{row['sample_id']}:balanced-positive-{idx}"
        row["balanced_repeat"] = idx >= len(positive_group)
        output.append(row)

    negative_names = ("regress",) if negative_mode == "regress_only" else ("regress", "both_correct", "both_wrong")
    allocations = {name: negatives // len(negative_names) for name in negative_names}
    for name in negative_names[: negatives % len(negative_names)]:
        allocations[name] += 1
    for name in negative_names:
        group = list(groups.get(name, []))
        if not group:
            raise ValueError(f"empty negative group: {name}")
        rng.shuffle(group)
        for idx in range(allocations[name]):
            row = dict(group[idx % len(group)])
            row["sample_id"] = f"{row['sample_id']}:balanced-{name}-{idx}"
            row["balanced_repeat"] = idx >= len(group)
            output.append(row)
    rng.shuffle(output)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--causal-input", required=True)
    parser.add_argument("--student-eval", required=True)
    parser.add_argument("--corrections", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--balanced-positive", type=int, default=2048)
    parser.add_argument("--balanced-negative", type=int, default=2048)
    parser.add_argument("--negative-mode", choices=["balanced_subtypes", "regress_only"], default="balanced_subtypes")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    source_rows = read_jsonl(Path(args.causal_input))
    source = {paired_key(row): row for row in source_rows}
    student_rows = read_jsonl(Path(args.student_eval))
    student = {paired_key(row): row for row in student_rows}
    if len(source) != len(source_rows) or len(student) != len(student_rows) or set(source) != set(student):
        raise ValueError("causal/student grid mismatch")
    rationales = {}
    for correction in read_jsonl(Path(args.corrections)):
        if not correction.get("parse_ok"):
            continue
        for step in correction.get("revised_steps", []):
            rationales[(str(correction["correction_id"]), int(step["step_idx"]))] = str(step.get("rationale") or "")

    rows = []
    for key in source:
        src = source[key]
        stu = student[key]
        utility_outcome = outcome(bool(stu["student_correct"]), bool(src["revision_correct"]))
        label = utility_outcome == "rescue"
        text = prompt(src, stu, rationales.get(key, ""))
        rows.append({
            "sample_id": f"revision-rescue-ranker:{src['correction_id']}:{src['step_idx']}",
            "episode_id": str(src["episode_id"]),
            "correction_id": str(src["correction_id"]),
            "step_idx": int(src["step_idx"]),
            "split": split_name(str(src["episode_id"]), args.seed),
            "image": str(src["image"]),
            "images": [str(src["image"])],
            "prompt": text,
            "target_text": "YES" if label else "NO",
            "messages": [
                {"from": "human", "value": text},
                {"from": "gpt", "value": "YES" if label else "NO"},
            ],
            "label": int(label),
            "utility_outcome": utility_outcome,
            "student_correct": bool(stu["student_correct"]),
            "revision_correct": bool(src["revision_correct"]),
            "actor": str(src["actor"]),
            "actor_action": src.get("actor_action"),
            "student_action": stu.get("student_action"),
            "revision_action": src.get("revision_action"),
        })

    splits = {name: [row for row in rows if row["split"] == name] for name in ("train", "dev", "test")}
    balanced = balanced_train(
        splits["train"], args.balanced_positive, args.balanced_negative, args.seed, args.negative_mode
    )
    out_dir = Path(args.output_dir)
    for name, items in splits.items():
        write_jsonl(out_dir / f"{name}.jsonl", items)
    write_jsonl(out_dir / "train_balanced.jsonl", balanced)
    episode_sets = {name: {str(row["episode_id"]) for row in items} for name, items in splits.items()}
    if any(episode_sets[a] & episode_sets[b] for a, b in (("train", "dev"), ("train", "test"), ("dev", "test"))):
        raise AssertionError("episode leakage")
    summary = {
        "version": "revision-rescue-ranker-v1",
        "task": "binary calibrated student-rescue ranking",
        "features_exclude_gt_and_matcher": True,
        "labels_derived_from_frozen_matcher": True,
        "episode_disjoint": True,
        "split_rows": {name: len(items) for name, items in splits.items()},
        "split_episodes": {name: len(episode_sets[name]) for name in splits},
        "outcome_counts": {name: dict(Counter(str(row["utility_outcome"]) for row in items)) for name, items in splits.items()},
        "positive_rates": {name: sum(int(row["label"]) for row in items) / len(items) for name, items in splits.items()},
        "balanced_train_rows": len(balanced),
        "negative_mode": args.negative_mode,
        "balanced_train_labels": dict(Counter(str(row["target_text"]) for row in balanced)),
        "artifacts": {
            name: {"path": str(out_dir / f"{name}.jsonl"), "sha256": sha256(out_dir / f"{name}.jsonl")}
            for name in ("train", "dev", "test", "train_balanced")
        },
    }
    write_json(out_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
