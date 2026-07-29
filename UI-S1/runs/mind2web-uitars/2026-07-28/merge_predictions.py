import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-shards", type=int, default=4)
    args = parser.parse_args()
    rows = {}
    for shard_index in range(args.num_shards):
        path = args.shard_root / f"shard-{shard_index}" / "predictions.jsonl"
        if not path.is_file():
            raise FileNotFoundError(path)
        for line in path.read_text().splitlines():
            row = json.loads(line)
            index = row["index"]
            if index in rows:
                raise ValueError(f"duplicate index {index}")
            if row["shard_index"] != shard_index or index % args.num_shards != shard_index:
                raise ValueError(f"row {index} has wrong shard identity")
            rows[index] = row
    if set(rows) != set(range(2080)):
        raise ValueError("merged prediction coverage is not exactly 0..2079")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "predictions.jsonl"
    if output_path.exists():
        raise FileExistsError(output_path)
    with output_path.open("w") as output:
        for index in range(2080):
            output.write(json.dumps(rows[index], ensure_ascii=True) + "\n")
    print(json.dumps({"rows": len(rows), "output": str(output_path)}, indent=2))


if __name__ == "__main__":
    main()