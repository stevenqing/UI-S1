#!/usr/bin/env python3
"""Apply Verifier Agent predictions to produce executable coordinator commands."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from verifier_agent_runtime import coordinator_command, iter_jsonl, summarize_commands, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert verifier predictions into coordinator commands")
    parser.add_argument("--data", required=True, help="Verifier Agent JSONL with packets and metadata")
    parser.add_argument("--predictions", required=True, help="Verifier prediction JSONL")
    parser.add_argument("--output", required=True, help="Output JSONL of coordinator commands")
    parser.add_argument("--summary", default="", help="Optional JSON summary path")
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_rows = iter_jsonl(Path(args.data))
    prediction_rows = iter_jsonl(Path(args.predictions))
    if args.limit > 0:
        data_rows = data_rows[: args.limit]
        prediction_rows = prediction_rows[: args.limit]
    if len(data_rows) != len(prediction_rows):
        raise SystemExit(f"prediction count mismatch: {len(prediction_rows)} != {len(data_rows)}")
    commands = [
        coordinator_command(data_row, prediction_row)
        for data_row, prediction_row in zip(data_rows, prediction_rows, strict=True)
    ]
    output_path = Path(args.output)
    write_jsonl(output_path, commands)
    summary = summarize_commands(commands)
    if args.summary:
        summary_path = Path(args.summary)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output_path), "summary": summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()