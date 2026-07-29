import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data/prepared/ac_high.jsonl"
PREFIX = ROOT / "artifacts/full/predictions.jsonl"
SHARDS = [ROOT / f"artifacts/shards/shard_{index}/predictions.jsonl" for index in range(4)]
OUTPUT_DIR = ROOT / "artifacts/merged"


def read_jsonl(path):
    with path.open() as input_file:
        return [json.loads(line) for line in input_file if line.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--prefix", type=Path, default=PREFIX)
    parser.add_argument("--shard-root", type=Path, default=ROOT / "artifacts/shards")
    parser.add_argument("--no-prefix", action="store_true")
    args = parser.parse_args()
    args.output_dir = args.output_dir.resolve()
    args.prefix = args.prefix.resolve()
    args.shard_root = args.shard_root.resolve()

    expected = [row["identity"] for row in read_jsonl(DATA)]
    shards = [args.shard_root / f"shard_{index}/predictions.jsonl" for index in range(4)]
    sources = shards if args.no_prefix else [args.prefix, *shards]
    predictions = []
    for source in sources:
        predictions.extend(read_jsonl(source))

    identities = [row["identity"] for row in predictions]
    duplicates = len(identities) - len(set(identities))
    missing = sorted(set(expected) - set(identities))
    extra = sorted(set(identities) - set(expected))
    if duplicates or missing or extra or len(predictions) != len(expected):
        raise RuntimeError(
            f"incomplete coverage: predictions={len(predictions)} duplicates={duplicates} "
            f"missing={len(missing)} extra={len(extra)}"
        )

    by_identity = {row["identity"]: row for row in predictions}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "predictions.jsonl"
    with output_path.open("w") as output_file:
        for identity in expected:
            output_file.write(json.dumps(by_identity[identity], ensure_ascii=False) + "\n")

    summary = {
        "status": "PASS",
        "predictions": len(predictions),
        "unique_identities": len(by_identity),
        "duplicates": duplicates,
        "missing": len(missing),
        "extra": len(extra),
        "order_matches_prepared_data": True,
        "sources": [str(source.relative_to(ROOT)) for source in sources],
    }
    (args.output_dir / "merge_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()