import hashlib
from pathlib import Path

from recovery_common import (
    ROOT, RUN_DIR, assert_protected_process, atomic_json, load_config, load_jsonl, references, sha256_file,
    validate_lane_rows,
)


def identity_hash(rows):
    return hashlib.sha256("\n".join(row["id"] for row in rows).encode()).hexdigest()


def main():
    config = load_config()
    protected_process = assert_protected_process(config)
    report = {}
    for name, lane in config["lanes"].items():
        path = ROOT / lane["destination"]
        rows = validate_lane_rows(
            load_jsonl(path), references(config, lane["setting"]), lane, require_complete=True
        )
        report[name] = {
            "model_id": lane["model_id"],
            "setting": lane["setting"],
            "path": lane["destination"],
            "rows": len(rows),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
            "row_ids_sha256": identity_hash(rows),
            "recovered_from_rows": lane["seed_rows"],
        }
    for name, lane in config["complete_lanes"].items():
        rows = []
        files = []
        for item in lane["shards"]:
            path = ROOT / item["path"]
            rows.extend(load_jsonl(path))
            files.append({
                "path": item["path"], "rows": item["rows"], "bytes": item["bytes"],
                "sha256": item["sha256"], "shard_index": item["shard_index"],
            })
        rows = validate_lane_rows(rows, references(config, lane["setting"]), lane, require_complete=True)
        report[name] = {
            "model_id": lane["model_id"],
            "setting": lane["setting"],
            "rows": len(rows),
            "row_ids_sha256": identity_hash(rows),
            "files": files,
            "preexisting_complete": True,
        }
    result = {
        "schema_version": 1,
        "status": "PASS_TRIVUS_R0_RESULT_BLIND_RECOVERY",
        "reference_rows_contain_gt_fields": True,
        "ground_truth_fields_used": False,
        "scorer_or_evaluator_imported": False,
        "accuracy_or_oracle_computed": False,
        "historical_files_modified": False,
        "expected_rows_per_lane": config["expected_rows_per_lane"],
        "protected_process": protected_process,
        "lanes": report,
    }
    output = RUN_DIR / "RECOVERY_MANIFEST.json"
    if output.exists():
        raise FileExistsError(output)
    atomic_json(output, result)
    print("TRIVUS_R0_FINALIZED")


if __name__ == "__main__":
    main()