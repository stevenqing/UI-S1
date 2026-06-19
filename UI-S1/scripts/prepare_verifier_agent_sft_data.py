#!/usr/bin/env python3
"""Prepare class-balanced SFT data for the Verifier Agent."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


JsonDict = dict[str, Any]


def iter_jsonl(path: Path) -> list[JsonDict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def decision(row: JsonDict) -> str:
    return str((row.get("target", {}) or {}).get("decision", "unknown"))


def sample_balanced(rows: list[JsonDict], per_class: int, seed: int) -> list[JsonDict]:
    grouped: dict[str, list[JsonDict]] = defaultdict(list)
    for row in rows:
        grouped[decision(row)].append(row)
    rng = random.Random(seed)
    sampled = []
    for label in sorted(grouped):
        items = grouped[label]
        if not items:
            continue
        if len(items) >= per_class:
            sampled.extend(rng.sample(items, per_class))
        else:
            sampled.extend(items)
            sampled.extend(rng.choice(items) for _ in range(per_class - len(items)))
    rng.shuffle(sampled)
    return sampled


def sample_cap(rows: list[JsonDict], max_rows: int, seed: int) -> list[JsonDict]:
    if max_rows <= 0 or len(rows) <= max_rows:
        return list(rows)
    rng = random.Random(seed)
    return rng.sample(rows, max_rows)


def write_jsonl(path: Path, rows: list[JsonDict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def parquet_row(row: JsonDict) -> JsonDict:
    messages = row.get("messages", [])
    return {
        "messages": json.dumps(messages, ensure_ascii=False),
        "decision": decision(row),
        "metadata": json.dumps(row.get("metadata", {}), ensure_ascii=False),
    }


def write_parquet(path: Path, rows: list[JsonDict]) -> bool:
    try:
        import pandas as pd
    except Exception as exc:
        print(f"warning: pandas unavailable; skip parquet {path}: {exc}")
        return False
    try:
        frame = pd.DataFrame([parquet_row(row) for row in rows])
        frame["messages"] = frame["messages"].astype("object")
        frame.to_parquet(path, index=False, engine="pyarrow")
    except Exception as exc:
        print(f"warning: parquet write failed for {path}: {exc}")
        return False
    return True


def summarize(rows: list[JsonDict]) -> JsonDict:
    decisions = Counter(decision(row) for row in rows)
    capabilities = Counter(str((row.get("metadata", {}) or {}).get("dominant_capability", "unknown")) for row in rows)
    utility = Counter(str((row.get("metadata", {}) or {}).get("utility_label", "unknown")) for row in rows)
    user_lengths = []
    assistant_lengths = []
    for row in rows:
        messages = row.get("messages", [])
        for message in messages:
            role = message.get("role")
            if role == "user":
                user_lengths.append(len(str(message.get("content", ""))))
            elif role == "assistant":
                assistant_lengths.append(len(str(message.get("content", ""))))
    return {
        "rows": len(rows),
        "decisions": dict(decisions.most_common()),
        "capabilities": dict(capabilities.most_common(20)),
        "utility_labels": dict(utility.most_common()),
        "avg_user_chars": sum(user_lengths) / len(user_lengths) if user_lengths else 0.0,
        "max_user_chars": max(user_lengths) if user_lengths else 0,
        "avg_assistant_chars": sum(assistant_lengths) / len(assistant_lengths) if assistant_lengths else 0.0,
        "max_assistant_chars": max(assistant_lengths) if assistant_lengths else 0,
    }


def write_manifest(output_dir: Path, stats: JsonDict, args: argparse.Namespace, parquet_written: dict[str, bool]) -> None:
    manifest = {
        "source_dir": args.input_dir,
        "train_per_class": args.train_per_class,
        "balanced_eval_per_class": args.balanced_eval_per_class,
        "seed": args.seed,
        "parquet_written": parquet_written,
        "splits": stats,
        "recommended_training": {
            "train_file": str(output_dir / "train_balanced.parquet"),
            "val_file": str(output_dir / "dev.parquet"),
            "messages_key": "messages",
            "task": "Verifier Agent route-decision SFT",
        },
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_report(output_dir: Path, stats: JsonDict) -> None:
    lines = ["# Verifier Agent SFT Data", ""]
    lines.append("## Splits")
    lines.append("")
    lines.append("| split | rows | decisions | avg user chars | max user chars |")
    lines.append("|---|---:|---|---:|---:|")
    for split, item in stats.items():
        decisions = ", ".join(f"{key}:{value}" for key, value in item["decisions"].items())
        lines.append(f"| {split} | {item['rows']} | {decisions} | {item['avg_user_chars']:.1f} | {item['max_user_chars']} |")
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- `train_balanced` uses class-balanced oversampling for verifier decisions.")
    lines.append("- `dev` and `test` preserve the original hard-only distribution for realistic evaluation.")
    lines.append("- `dev_balanced` and `test_balanced` are diagnostic only and should not replace original-distribution evaluation.")
    (output_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare Verifier Agent SFT data")
    parser.add_argument("--input-dir", required=True, help="Verifier-agent JSONL directory from build_verifier_agent_data.py")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--train-per-class", type=int, default=1024)
    parser.add_argument("--balanced-eval-per-class", type=int, default=0, help="Optional per-class cap for balanced dev/test diagnostics")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--no-parquet", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw = {split: iter_jsonl(input_dir / f"{split}.jsonl") for split in ["train", "dev", "test"]}
    splits = {
        "train_balanced": sample_balanced(raw["train"], args.train_per_class, args.seed),
        "dev": raw["dev"],
        "test": raw["test"],
    }
    if args.balanced_eval_per_class > 0:
        splits["dev_balanced"] = sample_balanced(raw["dev"], args.balanced_eval_per_class, args.seed + 1)
        splits["test_balanced"] = sample_balanced(raw["test"], args.balanced_eval_per_class, args.seed + 2)
    else:
        splits["dev_balanced"] = sample_cap(raw["dev"], 0, args.seed + 1)
        splits["test_balanced"] = sample_cap(raw["test"], 0, args.seed + 2)
    stats = {}
    parquet_written = {}
    for split, rows in splits.items():
        write_jsonl(output_dir / f"{split}.jsonl", rows)
        if not args.no_parquet:
            parquet_written[split] = write_parquet(output_dir / f"{split}.parquet", rows)
        else:
            parquet_written[split] = False
        stats[split] = summarize(rows)
    write_manifest(output_dir, stats, args, parquet_written)
    write_report(output_dir, stats)
    print(json.dumps({"output_dir": str(output_dir), "splits": stats}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
