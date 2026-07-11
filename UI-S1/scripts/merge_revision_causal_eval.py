#!/usr/bin/env python3
"""Merge deterministic revision-causal evaluation shards with exact grid checks."""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
from collections import Counter
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


def group_metrics(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    trajectories: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        trajectories.setdefault(str(row["correction_id"]), []).append(row)
    complete = sum(all(bool(row["student_correct"]) for row in group) for group in trajectories.values())
    return {
        "rows": len(rows),
        "accuracy": sum(bool(row["student_correct"]) for row in rows) / max(1, len(rows)),
        "parse_rate": sum(bool(row["parse_ok"]) for row in rows) / max(1, len(rows)),
        "mean_reward": sum(float(row["reward"]) for row in rows) / max(1, len(rows)),
        "trajectories": len(trajectories),
        "complete_trajectories": complete,
        "trajectory_success_rate": complete / max(1, len(trajectories)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Causal-arm source JSONL")
    parser.add_argument("--shards", required=True, help="Glob for evaluation shard JSONLs")
    parser.add_argument("--output", required=True)
    parser.add_argument("--expected-rows", type=int, default=0)
    parser.add_argument("--max-rows", type=int, default=0)
    args = parser.parse_args()

    input_path = Path(args.input)
    source_rows = read_jsonl(input_path)
    if args.max_rows > 0:
        source_rows = source_rows[: args.max_rows]
    source_ids = [str(row["sample_id"]) for row in source_rows]
    if len(source_ids) != len(set(source_ids)):
        raise ValueError("duplicate source sample_id")
    if args.expected_rows > 0 and len(source_rows) != args.expected_rows:
        raise ValueError(f"source row mismatch: {len(source_rows)} != {args.expected_rows}")

    shard_paths = [Path(path) for path in sorted(glob.glob(args.shards))]
    if not shard_paths:
        raise ValueError("no shard files matched")
    merged_by_id: dict[str, dict[str, Any]] = {}
    shard_manifest = []
    for path in shard_paths:
        rows = read_jsonl(path)
        for row in rows:
            sample_id = str(row["sample_id"])
            if sample_id in merged_by_id:
                raise ValueError(f"duplicate shard sample_id: {sample_id}")
            merged_by_id[sample_id] = row
        shard_manifest.append({"path": str(path), "rows": len(rows), "sha256": sha256(path)})
    if set(merged_by_id) != set(source_ids):
        missing = sorted(set(source_ids) - set(merged_by_id))[:10]
        extra = sorted(set(merged_by_id) - set(source_ids))[:10]
        raise ValueError(f"shard grid mismatch missing={missing} extra={extra}")

    merged = [merged_by_id[sample_id] for sample_id in source_ids]
    output_path = Path(args.output)
    write_jsonl(output_path, merged)
    by_actor = {
        actor: group_metrics([row for row in merged if str(row["actor"]) == actor])
        for actor in sorted({str(row["actor"]) for row in merged})
    }
    by_prefix = {
        name: group_metrics([row for row in merged if bool(row["prefix_clean"]) is clean])
        for name, clean in (("clean", True), ("dirty", False))
    }
    summary = {
        "arm": sorted({str(row["arm"]) for row in merged}),
        **group_metrics(merged),
        "by_actor": by_actor,
        "by_prefix": by_prefix,
        "predicted_action_types": dict(Counter(str((row.get("student_action") or {}).get("action") or "unparsed") for row in merged)),
        "input": str(input_path),
        "input_sha256": sha256(input_path),
        "shards": shard_manifest,
        "output": str(output_path),
        "output_sha256": sha256(output_path),
    }
    write_json(output_path.with_suffix(".summary.json"), summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
