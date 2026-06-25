#!/usr/bin/env python3
"""Build a stratified GUI-Odyssey pilot for text-space error-horizon probing."""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


JsonDict = dict[str, Any]


def iter_jsonl(path: Path) -> Iterable[JsonDict]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def length_bucket(num_steps: int) -> str:
    if num_steps <= 3:
        return "short(1-3)"
    if num_steps <= 7:
        return "medium(4-7)"
    if num_steps <= 15:
        return "long(8-15)"
    return "vlong(16+)"


def load_rollout_by_id(path: Path) -> dict[str, JsonDict]:
    return {str(row["episode_id"]): row for row in iter_jsonl(path)}


def write_jsonl(path: Path, rows: list[JsonDict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def summarize(rows: list[JsonDict]) -> JsonDict:
    by_category: dict[str, int] = defaultdict(int)
    by_bucket: dict[str, int] = defaultdict(int)
    by_pair: dict[str, int] = defaultdict(int)
    for row in rows:
        by_category[str(row.get("category", "unknown"))] += 1
        by_bucket[str(row.get("length_bucket", "unknown"))] += 1
        by_pair[f"{row.get('category', 'unknown')}::{row.get('length_bucket', 'unknown')}"] += 1
    return {
        "episodes": len(rows),
        "steps": sum(int(row.get("num_steps") or 0) for row in rows),
        "by_category": dict(sorted(by_category.items())),
        "by_length_bucket": dict(sorted(by_bucket.items())),
        "by_category_length": dict(sorted(by_pair.items())),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build stratified pilot episodes for text-space error-horizon probe")
    parser.add_argument("--jsonl-file", type=Path, required=True)
    parser.add_argument("--rollout-results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-episodes", type=int, default=300)
    parser.add_argument("--seed", type=int, default=17)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    rollout_by_id = load_rollout_by_id(args.rollout_results)
    groups: dict[tuple[str, str], list[JsonDict]] = defaultdict(list)
    all_rows = []
    for index, episode in enumerate(iter_jsonl(args.jsonl_file)):
        episode_id = str(episode.get("episode_id"))
        rollout = rollout_by_id.get(episode_id, {})
        num_steps = len(episode.get("steps", []) or [])
        row = {
            "episode_index": index,
            "episode_id": episode_id,
            "num_steps": num_steps,
            "category": episode.get("category", "unknown"),
            "length_bucket": length_bucket(num_steps),
            "rollout_task_success": bool(rollout.get("task_success")),
        }
        groups[(row["category"], row["length_bucket"])].append(row)
        all_rows.append(row)

    for items in groups.values():
        rng.shuffle(items)

    selected = []
    seen = set()
    group_keys = sorted(groups, key=lambda key: (key[0], key[1]))
    while len(selected) < args.max_episodes:
        made_progress = False
        for key in group_keys:
            items = groups[key]
            while items and items[-1]["episode_id"] in seen:
                items.pop()
            if not items:
                continue
            row = items.pop()
            selected.append(row)
            seen.add(row["episode_id"])
            made_progress = True
            if len(selected) >= args.max_episodes:
                break
        if not made_progress:
            break

    if len(selected) < args.max_episodes:
        remaining = [row for row in all_rows if row["episode_id"] not in seen]
        rng.shuffle(remaining)
        selected.extend(remaining[: args.max_episodes - len(selected)])

    selected.sort(key=lambda row: row["episode_index"])
    write_jsonl(args.output, selected)
    stats = summarize(selected)
    stats_path = args.output.with_suffix(args.output.suffix + ".summary.json")
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "summary": stats}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()