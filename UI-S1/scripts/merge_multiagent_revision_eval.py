#!/usr/bin/env python3
"""Merge and validate sharded multi-agent revision evaluation outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode-data", required=True)
    parser.add_argument("--shards", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--expected-episodes", type=int, required=True)
    parser.add_argument("--expected-steps", type=int, required=True)
    args = parser.parse_args()

    source = read_jsonl(Path(args.episode_data))
    source_ids = [str(row["episode_id"]) for row in source]
    source_steps = {str(row["episode_id"]): len(row.get("steps", [])) for row in source}
    rows = []
    shard_meta = []
    for path_text in args.shards:
        path = Path(path_text)
        shard_rows = read_jsonl(path)
        rows.extend(shard_rows)
        shard_meta.append({"path": str(path), "sha256": sha256(path), "episodes": len(shard_rows)})
    by_id = {str(row["episode_id"]): row for row in rows}
    if len(rows) != len(by_id):
        raise ValueError("duplicate episodes across shards")
    if set(by_id) != set(source_ids) or len(rows) != args.expected_episodes:
        raise ValueError("shards do not exactly cover the source episode IDs")
    ordered = [by_id[episode_id] for episode_id in source_ids]
    if any(int(row.get("num_steps") or -1) != source_steps[str(row["episode_id"])] for row in ordered):
        raise ValueError("episode step count mismatch")
    all_steps = [step for row in ordered for step in row.get("steps", [])]
    if len(all_steps) != args.expected_steps:
        raise ValueError(f"expected {args.expected_steps} steps, found {len(all_steps)}")
    output = Path(args.output)
    write_jsonl(output, ordered)
    summary = {
        "episodes": len(ordered),
        "steps": len(all_steps),
        "task_successes": sum(bool(row.get("task_success")) for row in ordered),
        "tsr": sum(bool(row.get("task_success")) for row in ordered) / len(ordered),
        "step_accuracy": sum(bool(step.get("success")) for step in all_steps) / len(all_steps),
        "parse_rate": sum(bool(step.get("parse_ok")) for step in all_steps) / len(all_steps),
        "mean_reward": sum(float(step.get("reward") or 0.0) for step in all_steps) / len(all_steps),
        "episode_data": args.episode_data,
        "episode_data_sha256": sha256(Path(args.episode_data)),
        "shards": shard_meta,
        "output_sha256": sha256(output),
    }
    write_json(output.with_suffix(".summary.json"), summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
