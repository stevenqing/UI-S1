import argparse
import hashlib
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def candidate_hash(candidates):
    return hashlib.sha256(
        json.dumps(candidates, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-shards", type=int, default=8)
    args = parser.parse_args()
    rows = {}
    for shard in range(args.num_shards):
        path = args.shard_root / f"shard-{shard}.jsonl"
        for line in path.read_text().splitlines():
            if not line.strip(): continue
            row = json.loads(line)
            if row["id"] in rows or row["shard_index"] != shard:
                raise ValueError("GTA1 N12 shard identity mismatch")
            if candidate_hash(row["candidates"]) != row["candidate_sha256"]:
                raise ValueError("GTA1 N12 source hash mismatch")
            if len(row["candidates"]) < 12:
                raise ValueError(f"GTA1 N12 insufficient candidates: {row['id']}")
            candidates = row["candidates"][:12]
            rows[row["id"]] = {
                **{key: value for key, value in row.items() if key not in {"candidates", "candidate_sha256", "candidate_count"}},
                "candidates": candidates,
                "candidate_count": 12,
                "candidate_sha256": candidate_hash(candidates),
                "derivation": "official_ordered_prefix_N12",
            }
    if len(rows) != 1581:
        raise ValueError(f"GTA1 N12 requires 1,581 rows, found {len(rows)}")
    ordered = [rows[key] for key in sorted(rows)]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(ordered), args.output, compression="zstd")
    print(json.dumps({
        "status": "PASS", "rows": len(ordered), "output": str(args.output),
        "sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
    }, indent=2))


if __name__ == "__main__":
    main()
