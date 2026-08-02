import argparse
import hashlib
import json
from pathlib import Path


EXPECTED_SOURCE_SHA256 = "2a7233cf0fbab109ea481ba891d2d7e65a481c48cd73c623d7d76fc61c06ae17"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source_hash = hashlib.sha256(args.source.read_bytes()).hexdigest()
    if source_hash != EXPECTED_SOURCE_SHA256:
        raise ValueError(f"Closing input source hash mismatch: {source_hash}")
    rows = [json.loads(line) for line in args.source.read_text().splitlines() if line.strip()]
    if len(rows) != 1581 or [row["stable_index"] for row in rows] != list(range(1581)):
        raise ValueError("Closing input identity mismatch")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as output:
        for row in rows:
            value = {
                key: row[key]
                for key in ("stable_index", "id", "application", "img_filename", "img_size", "instruction")
            }
            output.write(json.dumps(value, ensure_ascii=True) + "\n")
    print(json.dumps({
        "status": "PASS",
        "rows": len(rows),
        "source_sha256": source_hash,
        "output_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
        "target_fields": 0,
    }, indent=2))


if __name__ == "__main__":
    main()