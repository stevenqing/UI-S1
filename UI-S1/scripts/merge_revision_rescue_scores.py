#!/usr/bin/env python3
"""Exact-merge rescue-ranker score shards and recompute ranking metrics."""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

from score_revision_rescue_ranker import auc, average_precision, read_jsonl, sha256, write_json, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--shards", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    source = read_jsonl(Path(args.input))
    source_ids = [str(row["sample_id"]) for row in source]
    if len(source_ids) != len(set(source_ids)):
        raise ValueError("duplicate source sample_id")
    merged = {}
    manifest = []
    for path_name in sorted(glob.glob(args.shards)):
        path = Path(path_name); rows = read_jsonl(path)
        for row in rows:
            sample_id = str(row["sample_id"])
            if sample_id in merged:
                raise ValueError(f"duplicate score sample: {sample_id}")
            merged[sample_id] = row
        manifest.append({"path": str(path), "rows": len(rows), "sha256": sha256(path)})
    if set(merged) != set(source_ids):
        raise ValueError(f"score grid mismatch missing={list(set(source_ids)-set(merged))[:10]} extra={list(set(merged)-set(source_ids))[:10]}")
    rows = [merged[sample_id] for sample_id in source_ids]
    output = Path(args.output); write_jsonl(output, rows)
    labels = [int(row["label"]) for row in rows]; scores = [float(row["score"]) for row in rows]
    summary = {
        "rows": len(rows), "positive_rate": sum(labels)/len(labels),
        "roc_auc": auc(labels,scores), "average_precision": average_precision(labels,scores),
        "input": args.input, "input_sha256": sha256(Path(args.input)),
        "shards": manifest, "output": str(output), "output_sha256": sha256(output),
    }
    write_json(output.with_suffix(".summary.json"), summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
