#!/usr/bin/env python3
"""Extract verifier-agent packet rows matching episode/step keys.

This is useful for aligning a row-level GUI-Odyssey baseline result set with
the already-built verifier-agent packet format. It can preserve split labels so
we do not accidentally mix training rows into a held-out evaluation.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


JsonDict = dict[str, Any]


def iter_jsonl(path: Path) -> Iterable[JsonDict]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: Path, rows: Iterable[JsonDict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def key_from_key_row(row: JsonDict, include_thinking_mode: bool) -> tuple[Any, ...] | None:
    metadata = row.get("metadata") or {}
    episode_id = row.get("episode_id", metadata.get("episode_id"))
    step_index = row.get("step_index", metadata.get("step_index"))
    thinking_mode = row.get("thinking_mode", metadata.get("thinking_mode"))
    if episode_id is None or step_index is None:
        return None
    if include_thinking_mode:
        return (str(episode_id), int(step_index), str(thinking_mode))
    return (str(episode_id), int(step_index))


def key_from_packet(row: JsonDict, include_thinking_mode: bool) -> tuple[Any, ...] | None:
    metadata = row.get("metadata") or {}
    episode_id = metadata.get("episode_id")
    step_index = metadata.get("step_index")
    thinking_mode = metadata.get("thinking_mode")
    if episode_id is None or step_index is None:
        return None
    if include_thinking_mode:
        return (str(episode_id), int(step_index), str(thinking_mode))
    return (str(episode_id), int(step_index))


def load_keys(path: Path, include_thinking_mode: bool) -> set[tuple[Any, ...]]:
    keys = set()
    for row in iter_jsonl(path):
        key = key_from_key_row(row, include_thinking_mode)
        if key is not None:
            keys.add(key)
    return keys


def parse_source(value: str) -> tuple[str, Path]:
    if "=" not in value:
        path = Path(value)
        return path.stem, path
    split, path = value.split("=", 1)
    return split, Path(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract verifier-agent rows by episode/step keys")
    parser.add_argument("--keys", required=True, type=Path, help="JSONL containing episode_id and step_index")
    parser.add_argument("--source", action="append", required=True, help="split=path JSONL; may be repeated")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--ignore-thinking-mode", action="store_true", help="Match only episode_id+step_index")
    args = parser.parse_args()

    include_thinking_mode = not args.ignore_thinking_mode
    wanted = load_keys(args.keys, include_thinking_mode)
    seen: set[tuple[Any, ...]] = set()
    rows = []
    split_counts: Counter[str] = Counter()
    decision_counts: Counter[str] = Counter()
    duplicate_keys = 0
    for source_arg in args.source:
        split, path = parse_source(source_arg)
        for row in iter_jsonl(path):
            key = key_from_packet(row, include_thinking_mode)
            if key is None or key not in wanted:
                continue
            if key in seen:
                duplicate_keys += 1
                continue
            seen.add(key)
            row = {**row, "source_split": split}
            rows.append(row)
            split_counts[split] += 1
            decision_counts[str((row.get("target") or {}).get("decision"))] += 1
    rows.sort(key=lambda row: (str((row.get("metadata") or {}).get("episode_id")), int((row.get("metadata") or {}).get("step_index", -1)), str((row.get("metadata") or {}).get("thinking_mode"))))
    written = write_jsonl(args.output, rows)
    summary = {
        "keys": str(args.keys),
        "wanted_keys": len(wanted),
        "matched_rows": written,
        "missing_keys": len(wanted - seen),
        "duplicate_keys_skipped": duplicate_keys,
        "include_thinking_mode": include_thinking_mode,
        "split_counts": dict(sorted(split_counts.items())),
        "target_decisions": dict(sorted(decision_counts.items())),
        "output": str(args.output),
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()