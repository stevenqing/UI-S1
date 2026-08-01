import argparse
import hashlib
import json
from pathlib import Path

import pyarrow.parquet as pq


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = pq.read_table(args.input).to_pylist()
    rows.sort(key=lambda row: row["id"])
    if len(rows) != 1581 or any(len(row["candidates"]) != 4 for row in rows):
        raise ValueError("H3 region export requires complete N4 rows")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as output:
        for stable_index, row in enumerate(rows):
            value = {
                "stable_index": stable_index,
                "id": row["id"],
                "application": row["application"],
                "img_filename": row["img_filename"],
                "img_size": row["img_size"],
                "target_bbox": row["target_bbox"],
                "instruction": row["instruction"],
                "shared_region_candidate_sha256": row["candidate_sha256"],
                "regions": [candidate["region"] for candidate in row["candidates"]],
            }
            output.write(json.dumps(value, ensure_ascii=True) + "\n")
    print(json.dumps({
        "status": "PASS", "rows": len(rows), "output": str(args.output),
        "sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
    }, indent=2))


if __name__ == "__main__":
    main()
