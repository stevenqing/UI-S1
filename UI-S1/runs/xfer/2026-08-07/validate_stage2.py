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
EXPECTED_SHARDS = {"TongUI-7B": 8, "CogAgent-18B": 8, "UI-TARS-7B": 8}
ARMS = ["C_uni", "C_cond", "C_rand", "C_self"]


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def load_consensus():
    path = RUN_DIR / "raw/mind2web-consensus-roi.jsonl"
    rows = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row["id"] in rows:
            raise ValueError(f"duplicate consensus id: {row['id']}")
        rows[row["id"]] = row
    if len(rows) != 2080:
        raise ValueError(f"consensus requires 2,080 rows, found {len(rows)}")
    return rows


def validate_lane(directory, model_spec, canonical_rows, consensus, expected_shards):
    paths = sorted(directory.glob("shard-*.jsonl"))
    if len(paths) != expected_shards:
        raise ValueError(f"stage2 shard count mismatch: {directory}, found={len(paths)}")
    model_dir = ROOT / model_spec["local_path"]
    index_hash = sha256_file(model_dir / "model.safetensors.index.json")
    seen = set()
    executed = 0
    parse_ok = 0
    for path in paths:
        file_shard = int(path.stem.removeprefix("shard-"))
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            stable_index = row["stable_index"]
            source = canonical_rows[stable_index]
            arm_source = consensus[source["id"]]
            if row["id"] in seen:
                raise ValueError(f"duplicate stage2 id: {model_spec['id']}/{row['id']}")
            seen.add(row["id"])
            expected = {
                "id": source["id"],
                "image_sha256": source["image_sha256"],
                "model_id": model_spec["id"],
                "model_revision": model_spec["revision"],
                "model_index_sha256": index_hash,
                "sets": ARMS,
                "source_hashes": {arm: arm_source["arms_sha256"] for arm in ARMS},
                "shard_index": file_shard,
                "num_shards": expected_shards,
            }
            for key, value in expected.items():
                if row.get(key) != value:
                    raise ValueError(f"stage2 provenance mismatch: {model_spec['id']}/{source['id']}/{key}")
            if stable_index % expected_shards != file_shard:
                raise ValueError(f"stage2 shard modulo mismatch: {model_spec['id']}/{source['id']}")
            if canonical_hash(row["predictions"]) != row["predictions_sha256"]:
                raise ValueError(f"stage2 prediction hash mismatch: {model_spec['id']}/{source['id']}")
            for arm in ARMS:
                values = row["predictions"][arm]
                expected_regions = arm_source["arms"][arm]
                if [value["crop_index"] for value in values] != list(range(len(expected_regions))):
                    raise ValueError(f"stage2 crop index mismatch: {model_spec['id']}/{source['id']}/{arm}")
                if [value["region"] for value in values] != expected_regions:
                    raise ValueError(f"stage2 crop geometry mismatch: {model_spec['id']}/{source['id']}/{arm}")
                if arm_source["stage2_trigger"] and len(values) != 2:
                    raise ValueError(f"triggered stage2 arm lacks two crops: {model_spec['id']}/{source['id']}/{arm}")
                if not arm_source["stage2_trigger"] and values:
                    raise ValueError(f"nontriggered stage2 arm is nonempty: {model_spec['id']}/{source['id']}/{arm}")
                for value in values:
                    prediction = value["prediction"]
                    if set(prediction) < {"action", "value", "position", "parse_ok"}:
                        raise ValueError(f"stage2 prediction schema mismatch: {model_spec['id']}/{source['id']}/{arm}")
                    executed += 1
                    parse_ok += int(prediction["parse_ok"])
    expected_ids = {row["id"] for row in canonical_rows}
    if seen != expected_ids:
        raise ValueError(f"stage2 lane coverage mismatch: {model_spec['id']}, rows={len(seen)}")
    return {
        "rows": len(seen),
        "executed_forwards": executed,
        "parse_ok": parse_ok,
        "parse_rate": parse_ok / executed if executed else None,
        "shards": expected_shards,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage2-root", type=Path, required=True)
    parser.add_argument("--models", default=",".join(MODEL_DIRS))
    args = parser.parse_args()
    selected_models = args.models.split(",")
    if not selected_models or len(selected_models) != len(set(selected_models)):
        raise ValueError("models must be a non-empty unique comma-separated list")
    unknown = set(selected_models) - set(MODEL_DIRS)
    if unknown:
        raise ValueError(f"unknown models: {sorted(unknown)}")
    roster = yaml.safe_load((RUN_DIR / "configs/xfer_roster.yaml").read_text())
    specs = {model["id"]: model for model in roster["mind2web"]["models"]}
    canonical_rows = [json.loads(line) for line in (RUN_DIR / "data/mind2web/mind2web_test_task.jsonl").read_text().splitlines() if line.strip()]
    consensus = load_consensus()
    report = {
        model: validate_lane(
            args.stage2_root / MODEL_DIRS[model], specs[model], canonical_rows,
            consensus, EXPECTED_SHARDS[model],
        )
        for model in selected_models
    }
    print(json.dumps({"status": "PASS", "lanes": report}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()