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


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def load_proposer_regions():
    rows = {}
    for path in sorted((RUN_DIR / "raw/proposer-regions").glob("shard-*.jsonl")):
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row["id"] in rows:
                raise ValueError(f"duplicate proposer id: {row['id']}")
            rows[row["id"]] = row
    return rows


def expected_regions(source, set_name):
    if set_name.startswith("view"):
        view_index = int(set_name.removeprefix("view"))
        return [source["regions"][view_index - 1]["region"]]
    return source["arms"][set_name]


def validate_lane(directory, model_spec, canonical_rows, crop_sets=None, expected_shards=None):
    expected_shards = expected_shards or EXPECTED_SHARDS[model_spec["id"]]
    crop_sets = crop_sets or []
    proposer_regions = load_proposer_regions() if crop_sets else None
    paths = sorted(directory.glob("shard-*.jsonl"))
    if len(paths) != expected_shards:
        raise ValueError(f"shard count mismatch: {directory}, found={len(paths)}")
    model_dir = ROOT / model_spec["local_path"]
    index_hash = sha256_file(model_dir / "model.safetensors.index.json")
    seen = set()
    parse_ok = 0
    prediction_count = 0
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
            if not crop_sets:
                prediction = row["prediction"]
                predictions = [prediction]
            else:
                if row.get("sets") != crop_sets:
                    raise ValueError(f"crop set mismatch: {model_spec['id']}/{source['id']}")
                proposer = proposer_regions[row["id"]]
                predictions = []
                for set_name in crop_sets:
                    expected_hash = proposer["regions_sha256"] if set_name.startswith("view") else proposer["arms_sha256"]
                    if row["source_hashes"][set_name] != expected_hash:
                        raise ValueError(f"crop source mismatch: {model_spec['id']}/{source['id']}/{set_name}")
                    values = row["predictions"][set_name]
                    if [value["region"] for value in values] != expected_regions(proposer, set_name):
                        raise ValueError(f"crop geometry mismatch: {model_spec['id']}/{source['id']}/{set_name}")
                    predictions.extend(value["prediction"] for value in values)
                if canonical_hash(row["predictions"]) != row["predictions_sha256"]:
                    raise ValueError(f"prediction hash mismatch: {model_spec['id']}/{source['id']}")
            for prediction in predictions:
                if set(prediction) < {"action", "value", "position", "parse_ok"}:
                    raise ValueError(f"prediction schema mismatch: {model_spec['id']}/{source['id']}")
                parse_ok += int(prediction["parse_ok"])
                prediction_count += 1
    if len(seen) != len(canonical_rows) or seen != {row["id"] for row in canonical_rows}:
        raise ValueError(f"lane coverage mismatch: {model_spec['id']}, rows={len(seen)}")
    return {
        "rows": len(seen),
        "parse_ok": parse_ok,
        "prediction_count": prediction_count,
        "parse_rate": parse_ok / prediction_count,
        "shards": expected_shards,
        "crop_sets": crop_sets,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--crop-set")
    parser.add_argument("--crop-sets")
    parser.add_argument("--models", default=",".join(MODEL_DIRS))
    parser.add_argument("--num-shards", type=int)
    args = parser.parse_args()
    roster = yaml.safe_load((RUN_DIR / "configs/xfer_roster.yaml").read_text())
    specs = {model["id"]: model for model in roster["mind2web"]["models"]}
    canonical_rows = [json.loads(line) for line in (RUN_DIR / "data/mind2web/mind2web_test_task.jsonl").read_text().splitlines() if line.strip()]
    base = RUN_DIR / "raw/stage1"
    if args.crop_set and args.crop_sets:
        raise ValueError("use either --crop-set or --crop-sets")
    crop_sets = args.crop_sets.split(",") if args.crop_sets else [args.crop_set] if args.crop_set else []
    if args.crop_set:
        base = base / args.crop_set
    selected_models = args.models.split(",")
    if not selected_models or len(selected_models) != len(set(selected_models)):
        raise ValueError("models must be a non-empty unique comma-separated list")
    unknown_models = set(selected_models) - set(MODEL_DIRS)
    if unknown_models:
        raise ValueError(f"unknown models: {sorted(unknown_models)}")
    report = {
        model: validate_lane(base / directory, specs[model], canonical_rows, crop_sets, args.num_shards)
        for model, directory in MODEL_DIRS.items()
        if model in selected_models
    }
    print(json.dumps({"status": "PASS", "crop_sets": crop_sets, "lanes": report}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()