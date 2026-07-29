import argparse
import json
import os
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON at {path}:{line_number}") from error
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-shards", type=int, default=4)
    parser.add_argument("--expected-rows", type=int, default=7708)
    args = parser.parse_args()

    by_index = {}
    for shard_index in range(args.num_shards):
        path = args.shard_root / f"shard-{shard_index}.jsonl"
        for row in read_jsonl(path):
            index = row["index"]
            if row["shard_index"] != shard_index or index % args.num_shards != shard_index:
                raise ValueError(f"shard assignment mismatch at index {index}")
            if index in by_index:
                raise ValueError(f"duplicate index {index}")
            by_index[index] = row
    expected = set(range(args.expected_rows))
    if set(by_index) != expected:
        missing = sorted(expected - set(by_index))[:10]
        extra = sorted(set(by_index) - expected)[:10]
        raise ValueError(f"incomplete merge: missing={missing}, extra={extra}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as output:
        for index in range(args.expected_rows):
            output.write(json.dumps(by_index[index], ensure_ascii=True) + "\n")
        output.flush()
        os.fsync(output.fileno())
    print(json.dumps({"status": "PASS", "rows": args.expected_rows, "shards": args.num_shards}))


if __name__ == "__main__":
    main()