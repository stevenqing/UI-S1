#!/usr/bin/env python3
"""Build multimodal verifier-agent data for student-conditioned revision routing."""

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


def compact_action(action: Any) -> str:
    return json.dumps(action or {"action": "unparsed"}, ensure_ascii=False, separators=(",", ":"))


def verifier_prompt(source: Mapping[str, Any], student: Mapping[str, Any], rationale: str) -> str:
    history = "\n".join(str(item) for item in source.get("history", [])) or "None"
    return f"""<image>
You are a conservative GUI action verifier. Select one existing candidate; do not invent a new coordinate.

User goal:
{source['goal']}

Action history:
{history}

Candidate packet:
- Source actor: {source['actor']}
- Actor candidate: {compact_action(source.get('actor_action'))}
- Global revision candidate: {compact_action(source.get('revision_action'))}
- Revision rationale: {rationale or 'Not provided'}
- Starting-student candidate: {compact_action(student.get('student_action'))}

Return strict JSON with:
- decision: use_revision | keep_student | replan
- selected_candidate: revision | student | null
- reason_codes: short list
- rationale: one concise sentence

Policy: keep a valid student action unless the revision is clearly a task-grounded repair. If neither candidate is reliable, replan."""


def target(decision: str) -> dict[str, Any]:
    if decision == "use_revision":
        return {
            "decision": decision,
            "selected_candidate": "revision",
            "reason_codes": ["student_error", "revision_rescue"],
            "rationale": "The student candidate is wrong and the revision repairs this step.",
        }
    if decision == "keep_student":
        return {
            "decision": decision,
            "selected_candidate": "student",
            "reason_codes": ["student_valid", "conservative_preservation"],
            "rationale": "The student candidate is already correct and should not be overwritten.",
        }
    return {
        "decision": "replan",
        "selected_candidate": None,
        "reason_codes": ["both_candidates_unreliable"],
        "rationale": "Neither the student candidate nor the revision is correct.",
    }


def balanced(rows: Sequence[Mapping[str, Any]], per_class: int, seed: int) -> list[dict[str, Any]]:
    groups: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        groups.setdefault(str(row["decision"]), []).append(row)
    rng = random.Random(seed)
    output = []
    for decision in ("use_revision", "keep_student", "replan"):
        group = list(groups.get(decision, []))
        if not group:
            raise ValueError(f"empty verifier class: {decision}")
        rng.shuffle(group)
        for index in range(per_class):
            source = dict(group[index % len(group)])
            source["sample_id"] = f"{source['sample_id']}:balanced-{decision}-{index}"
            source["balanced_repeat"] = index >= len(group)
            output.append(source)
    rng.shuffle(output)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--causal-input", required=True)
    parser.add_argument("--student-eval", required=True)
    parser.add_argument("--corrections", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--balanced-per-class", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    source_rows = read_jsonl(Path(args.causal_input))
    source = {paired_key(row): row for row in source_rows}
    evaluated_rows = read_jsonl(Path(args.student_eval))
    evaluated = {paired_key(row): row for row in evaluated_rows}
    if len(source) != len(source_rows) or len(evaluated) != len(evaluated_rows) or set(source) != set(evaluated):
        raise ValueError("source/student grid mismatch")

    rationale_by_key = {}
    for correction in read_jsonl(Path(args.corrections)):
        if not correction.get("parse_ok"):
            continue
        for step in correction.get("revised_steps", []):
            rationale_by_key[(str(correction["correction_id"]), int(step["step_idx"]))] = str(step.get("rationale") or "")

    rows = []
    for key in source:
        src = source[key]
        student = evaluated[key]
        student_correct = bool(student["student_correct"])
        revision_correct = bool(src["revision_correct"])
        decision = "keep_student" if student_correct else "use_revision" if revision_correct else "replan"
        target_payload = target(decision)
        rows.append({
            "sample_id": f"revision-verifier:{src['correction_id']}:{src['step_idx']}",
            "episode_id": str(src["episode_id"]),
            "correction_id": str(src["correction_id"]),
            "step_idx": int(src["step_idx"]),
            "split": split_name(str(src["episode_id"]), args.seed),
            "image": str(src["image"]),
            "images": [str(src["image"])],
            "prompt": verifier_prompt(src, student, rationale_by_key.get(key, "")),
            "target_text": json.dumps(target_payload, ensure_ascii=False),
            "messages": [
                {"from": "human", "value": verifier_prompt(src, student, rationale_by_key.get(key, ""))},
                {"from": "gpt", "value": json.dumps(target_payload, ensure_ascii=False)},
            ],
            "decision": decision,
            "actor_action": src.get("actor_action"),
            "revision_action": src.get("revision_action"),
            "student_action": student.get("student_action"),
            "correction_confidence": src.get("correction_confidence"),
            "revision_changed_from_actor": bool(src.get("revision_changed_from_actor")),
            "student_correct": student_correct,
            "revision_correct": revision_correct,
            "actor": str(src["actor"]),
            "prefix_clean_diagnostic_only": bool(src["prefix_clean"]),
        })

    out_dir = Path(args.output_dir)
    split_rows = {name: [row for row in rows if row["split"] == name] for name in ("train", "dev", "test")}
    for name, items in split_rows.items():
        write_jsonl(out_dir / f"{name}.jsonl", items)
    balanced_train = balanced(split_rows["train"], args.balanced_per_class, args.seed)
    write_jsonl(out_dir / "train_balanced.jsonl", balanced_train)

    episode_sets = {name: {str(row["episode_id"]) for row in items} for name, items in split_rows.items()}
    if any(episode_sets[left] & episode_sets[right] for left, right in (("train", "dev"), ("train", "test"), ("dev", "test"))):
        raise AssertionError("episode leakage")
    summary = {
        "version": "revision-verifier-agent-v1",
        "decision_policy": "keep correct student; use revision only for student rescue; otherwise replan",
        "features_exclude_gt_and_matcher": True,
        "labels_derived_from_frozen_matcher": True,
        "episode_disjoint": True,
        "split_rows": {name: len(items) for name, items in split_rows.items()},
        "split_episodes": {name: len(episode_sets[name]) for name in split_rows},
        "decision_counts": {name: dict(Counter(str(row["decision"]) for row in items)) for name, items in split_rows.items()},
        "balanced_train_rows": len(balanced_train),
        "balanced_train_counts": dict(Counter(str(row["decision"]) for row in balanced_train)),
        "balanced_per_class": args.balanced_per_class,
        "source": args.causal_input,
        "student_eval": args.student_eval,
        "corrections": args.corrections,
        "artifacts": {
            name: {"path": str(out_dir / f"{name}.jsonl"), "sha256": sha256(out_dir / f"{name}.jsonl")}
            for name in ("train", "dev", "test", "train_balanced")
        },
    }
    write_json(out_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
