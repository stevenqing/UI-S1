import argparse
import hashlib
import json
from pathlib import Path


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
                raise ValueError("Allocation-Law region identity mismatch")
            if candidate_hash(row["candidates"]) != row["candidate_sha256"]:
                raise ValueError("Allocation-Law source candidate hash mismatch")
            if len(row["candidates"]) < 12:
                raise ValueError(f"Allocation-Law requires 12 regions: {row['id']}")
            rows[row["id"]] = row
    if len(rows) != 1581:
        raise ValueError(f"Allocation-Law region export requires 1,581 rows, found {len(rows)}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as output:
        for stable_index, row_id in enumerate(sorted(rows)):
            row = rows[row_id]
            candidates = row["candidates"][:12]
            value = {
                "stable_index": stable_index,
                "id": row["id"],
                "application": row["application"],
                "img_filename": row["img_filename"],
                "img_size": row["img_size"],
                "target_bbox": row["target_bbox"],
                "instruction": row["instruction"],
                "shared_region_candidate_sha256": candidate_hash(candidates),
                "regions": [candidate["region"] for candidate in candidates],
            }
            output.write(json.dumps(value, ensure_ascii=True) + "\n")
    print(json.dumps({
        "status": "PASS", "rows": len(rows), "views": 12,
        "sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
        "output": str(args.output),
    }, indent=2))


if __name__ == "__main__":
    main()
