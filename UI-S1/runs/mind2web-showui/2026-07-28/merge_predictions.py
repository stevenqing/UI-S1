import argparse
import json
from pathlib import Path


EXPECTED_ROWS = 2080


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard-root", type=Path, required=True)
    parser.add_argument("--num-shards", type=int, default=4)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    rows = {}
    for shard_index in range(args.num_shards):
        path = args.shard_root / f"shard-{shard_index}" / "predictions.jsonl"
        if not path.exists():
            raise FileNotFoundError(path)
        with path.open() as handle:
            for line in handle:
                row = json.loads(line)
                index = row["index"]
                if index in rows:
                    raise ValueError(f"duplicate index {index}")
                if row["shard_index"] != shard_index:
                    raise ValueError(f"row {index} has wrong shard identity")
                if index % args.num_shards != shard_index:
                    raise ValueError(f"row {index} is in the wrong shard")
                rows[index] = row

    expected = set(range(EXPECTED_ROWS))
    actual = set(rows)
    if actual != expected:
        raise ValueError(
            f"coverage mismatch: missing={sorted(expected - actual)[:20]}, "
            f"extra={sorted(actual - expected)[:20]}"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "predictions.jsonl"
    if output_path.exists():
        raise FileExistsError(output_path)
    with output_path.open("w") as output:
        for index in range(EXPECTED_ROWS):
            output.write(json.dumps(rows[index], ensure_ascii=True) + "\n")
    print(json.dumps({"rows": len(rows), "output": str(output_path)}, indent=2))


if __name__ == "__main__":
    main()
