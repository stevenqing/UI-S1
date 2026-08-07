import argparse
import hashlib
import json
from pathlib import Path

import yaml


RUN_DIR = Path(__file__).resolve().parent
ROOT = RUN_DIR.parents[2]
MODEL_DIRS = {
    "TongUI-7B": "tongui",
    "CogAgent-18B": "cogagent",
    "UI-TARS-7B": "uitars",
}
EXPECTED_SHARDS = {
    "TongUI-7B": 2,
    "CogAgent-18B": 4,
    "UI-TARS-7B": 2,
}


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_lane(directory, model_spec, canonical_rows, crop_set=None):
    expected_shards = EXPECTED_SHARDS[model_spec["id"]]
    paths = sorted(directory.glob("shard-*.jsonl"))
    if len(paths) != expected_shards:
        raise ValueError(f"shard count mismatch: {directory}, found={len(paths)}")
    model_dir = ROOT / model_spec["local_path"]
    index_hash = sha256_file(model_dir / "model.safetensors.index.json")
    seen = set()
    parse_ok = 0
    for path in paths:
        file_shard = int(path.stem.removeprefix("shard-"))
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            stable_index = row["stable_index"]
            source = canonical_rows[stable_index]
            if row["id"] in seen:
                raise ValueError(f"duplicate id: {model_spec['id']}/{row['id']}")
            seen.add(row["id"])
            expected = {
                "id": source["id"],
                "image_sha256": source["image_sha256"],
                "model_id": model_spec["id"],
                "model_revision": model_spec["revision"],
                "model_index_sha256": index_hash,
                "shard_index": file_shard,
                "num_shards": expected_shards,
            }
            for key, value in expected.items():
                if row.get(key) != value:
                    raise ValueError(f"provenance mismatch: {model_spec['id']}/{source['id']}/{key}")
            if stable_index % expected_shards != file_shard:
                raise ValueError(f"shard modulo mismatch: {model_spec['id']}/{source['id']}")
            if crop_set is None:
                prediction = row["prediction"]
            else:
                if row.get("sets") != [crop_set] or len(row["predictions"][crop_set]) != 1:
                    raise ValueError(f"crop set mismatch: {model_spec['id']}/{source['id']}")
                prediction = row["predictions"][crop_set][0]["prediction"]
            if set(prediction) < {"action", "value", "position", "parse_ok"}:
                raise ValueError(f"prediction schema mismatch: {model_spec['id']}/{source['id']}")
            parse_ok += int(prediction["parse_ok"])
    if len(seen) != len(canonical_rows) or seen != {row["id"] for row in canonical_rows}:
        raise ValueError(f"lane coverage mismatch: {model_spec['id']}, rows={len(seen)}")
    return {
        "rows": len(seen),
        "parse_ok": parse_ok,
        "parse_rate": parse_ok / len(seen),
        "shards": expected_shards,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--crop-set")
    args = parser.parse_args()
    roster = yaml.safe_load((RUN_DIR / "configs/xfer_roster.yaml").read_text())
    specs = {model["id"]: model for model in roster["mind2web"]["models"]}
    canonical_rows = [json.loads(line) for line in (RUN_DIR / "data/mind2web/mind2web_test_task.jsonl").read_text().splitlines() if line.strip()]
    base = RUN_DIR / "raw/stage1"
    if args.crop_set:
        base = base / args.crop_set
    report = {
        model: validate_lane(base / directory, specs[model], canonical_rows, args.crop_set)
        for model, directory in MODEL_DIRS.items()
    }
    print(json.dumps({"status": "PASS", "crop_set": args.crop_set, "lanes": report}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()