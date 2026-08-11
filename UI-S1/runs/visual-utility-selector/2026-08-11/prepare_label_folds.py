import argparse
import json
from pathlib import Path

from vus_data import sha256_file


RUN_DIR = Path(__file__).resolve().parent


def load_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--public", type=Path, default=RUN_DIR / "data/public_records.jsonl")
    parser.add_argument("--private", type=Path, default=RUN_DIR / "data/private_labels.jsonl")
    parser.add_argument("--output-dir", type=Path, default=RUN_DIR / "data")
    args = parser.parse_args()
    public_rows = load_jsonl(args.public)
    private_rows = load_jsonl(args.private)
    fold_by_key = {row["sample_key"]: int(row["fold"]) for row in public_rows}
    if len(fold_by_key) != len(public_rows):
        raise ValueError("duplicate public sample keys")
    private_by_key = {row["sample_key"]: row for row in private_rows}
    if len(private_by_key) != len(private_rows) or set(private_by_key) != set(fold_by_key):
        raise ValueError("private/public label coverage mismatch")
    manifest = {
        "schema_version": 1,
        "status": "PASS_FOLD_SEALED_LABELS",
        "source_private_sha256": sha256_file(args.private),
        "source_public_sha256": sha256_file(args.public),
        "folds": {},
    }
    total = 0
    for fold in range(5):
        rows = [private_by_key[key] for key in sorted(private_by_key) if fold_by_key[key] == fold]
        path = args.output_dir / f"private_labels_fold-{fold}.jsonl"
        write_jsonl(path, rows)
        manifest["folds"][str(fold)] = {
            "path": str(path.relative_to(RUN_DIR)),
            "records": len(rows),
            "sha256": sha256_file(path),
        }
        total += len(rows)
    if total != len(private_rows):
        raise ValueError(f"fold-sealed label count mismatch: {total}/{len(private_rows)}")
    manifest["records"] = total
    path = args.output_dir / "private_label_folds.manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
