#!/usr/bin/env python3
"""Mix oracle student-rescue rows with broad clean replay for an SFT ceiling."""

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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rescue-data", required=True)
    parser.add_argument("--clean-replay-data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--arm", default="a14_student_rescue_clean_replay")
    parser.add_argument("--rescue-rows", type=int, default=400)
    parser.add_argument("--replay-rows", type=int, default=400)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    rescue = read_jsonl(Path(args.rescue_data))
    rng.shuffle(rescue)
    rescue = [dict(row) for row in rescue[: args.rescue_rows]]
    if len(rescue) != args.rescue_rows:
        raise ValueError("insufficient rescue rows")
    rescue_keys = {paired_key(row) for row in rescue}

    replay_source = [row for row in read_jsonl(Path(args.clean_replay_data)) if paired_key(row) not in rescue_keys]
    actors = sorted({str(row["actor"]) for row in replay_source})
    if args.replay_rows % (2 * len(actors)):
        raise ValueError("replay rows must divide actor x prefix strata")
    per_stratum = args.replay_rows // (2 * len(actors))
    replay = []
    for actor in actors:
        for clean in (True, False):
            group = [row for row in replay_source if str(row["actor"]) == actor and bool(row["prefix_clean"]) is clean]
            random.Random(f"{args.seed}:{actor}:{int(clean)}").shuffle(group)
            if len(group) < per_stratum:
                raise ValueError(f"insufficient replay stratum {actor}/{clean}")
            replay.extend(dict(row) for row in group[:per_stratum])

    rows = []
    for source_kind, source_rows in (("student_rescue", rescue), ("clean_replay", replay)):
        for row in source_rows:
            row.update({
                "sample_id": f"{args.arm}:{source_kind}:{row['correction_id']}:{row['step_idx']}",
                "treatment_arm": args.arm,
                "screen_arm": args.arm,
                "research_role": "oracle_student_rescue_with_clean_replay",
                "mixture_source": source_kind,
                "oracle_target_used": True,
            })
            rows.append(row)
    rng.shuffle(rows)
    if len({str(row["sample_id"]) for row in rows}) != len(rows):
        raise ValueError("duplicate mixed sample_id")
    output = Path(args.output)
    write_jsonl(output, rows)
    summary = {
        "arm": args.arm,
        "research_role": "oracle_student_rescue_with_clean_replay",
        "rows": len(rows),
        "optimizer_steps": len(rows) // 8,
        "gradient_accumulation_steps": 8,
        "mixture_counts": dict(Counter(str(row["mixture_source"]) for row in rows)),
        "diagnostic_label_accuracy": sum(bool(row["diagnostic_matcher_correct"]) for row in rows) / len(rows),
        "actor_counts": dict(Counter(str(row["actor"]) for row in rows)),
        "selection_uses_matcher": True,
        "semantic_quality_filter_used": True,
        "oracle_target_used": True,
        "output": str(output),
        "output_sha256": sha256(output),
    }
    write_json(output.with_suffix(".summary.json"), summary)
    manifest = {
        "version": "student-rescue-clean-replay-screen-v1",
        "training_policy": {"same_update_budget": True, "optimizer_steps": summary["optimizer_steps"]},
        "arms": {summary["arm"]: summary},
    }
    write_json(output.with_name(f"{args.arm}_screen_manifest.json"), manifest)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
