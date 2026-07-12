#!/usr/bin/env python3
"""Build an oracle ceiling from revisions that rescue the frozen student."""

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


def key(row: Mapping[str, Any]) -> tuple[str, int]:
    return str(row["correction_id"]), int(row["step_idx"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt-history-arm", required=True)
    parser.add_argument("--student-eval", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest-output")
    parser.add_argument("--rows", type=int, default=800)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    source_rows = read_jsonl(Path(args.gt_history_arm))
    source = {key(row): row for row in source_rows}
    evaluated_rows = read_jsonl(Path(args.student_eval))
    evaluated = {key(row): row for row in evaluated_rows}
    if len(source) != len(source_rows) or len(evaluated) != len(evaluated_rows):
        raise ValueError("duplicate paired key")
    if set(source) != set(evaluated):
        raise ValueError("source/evaluation grid mismatch")

    eligible = []
    for paired_key, row in source.items():
        student = evaluated[paired_key]
        if bool(student["student_correct"]) or not bool(row["revision_correct"]):
            continue
        clone = dict(row)
        clone.update({
            "sample_id": f"a13_oracle_student_rescue:{row['correction_id']}:{row['step_idx']}",
            "treatment_arm": "a13_oracle_student_rescue_gt_history",
            "screen_arm": "a13_oracle_student_rescue_gt_history",
            "selection_policy": "oracle_student_wrong_and_revision_correct",
            "selection_uses_matcher": True,
            "semantic_quality_filter_used": True,
            "oracle_target_used": True,
            "student_correct_before": False,
            "research_role": "oracle_student_utility_ceiling",
        })
        eligible.append(clone)
    if len(eligible) < args.rows:
        raise ValueError(f"eligible rescue rows {len(eligible)} < requested {args.rows}")
    random.Random(args.seed).shuffle(eligible)
    selected = eligible[: args.rows]
    output = Path(args.output)
    write_jsonl(output, selected)
    summary = {
        "arm": "a13_oracle_student_rescue_gt_history",
        "research_role": "oracle_student_utility_ceiling",
        "eligible_rows": len(eligible),
        "rows": len(selected),
        "episodes": len({str(row["episode_id"]) for row in selected}),
        "optimizer_steps": len(selected) // 8,
        "gradient_accumulation_steps": 8,
        "diagnostic_label_accuracy": sum(bool(row["diagnostic_matcher_correct"]) for row in selected) / len(selected),
        "selection_uses_matcher": True,
        "semantic_quality_filter_used": True,
        "oracle_target_used": True,
        "prefix_clean_fraction": sum(bool(row["prefix_clean"]) for row in selected) / len(selected),
        "actor_counts": dict(Counter(str(row["actor"]) for row in selected)),
        "output": str(output),
        "output_sha256": sha256(output),
    }
    write_json(output.with_suffix(".summary.json"), summary)
    manifest_path = Path(args.manifest_output) if args.manifest_output else output.with_name("a13_screen_manifest.json")
    write_json(manifest_path, {
        "version": "student-rescue-oracle-screen-v1",
        "training_policy": {"same_update_budget": True, "optimizer_steps": summary["optimizer_steps"]},
        "arms": {summary["arm"]: summary},
    })
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
