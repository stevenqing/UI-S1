import shutil
from pathlib import Path

from recovery_common import ROOT, RUN_DIR, assert_protected_process, atomic_json, load_config, load_jsonl, references, sha256_file, validate_lane_rows


def main():
    config = load_config()
    assert_protected_process(config)
    prepared = RUN_DIR / "recovery/PREPARED.json"
    if prepared.exists():
        raise FileExistsError(prepared)
    report = {}
    for name, lane in config["lanes"].items():
        source = ROOT / lane["seed_path"]
        destination = ROOT / lane["destination"]
        if destination.exists():
            raise FileExistsError(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        if sha256_file(destination) != lane["seed_sha256"] or destination.stat().st_size != lane["seed_bytes"]:
            raise ValueError(f"TriVUS R0 copied seed mismatch: {name}")
        validate_lane_rows(
            load_jsonl(destination), references(config, lane["setting"]), lane, require_complete=False
        )
        report[name] = {
            "source": lane["seed_path"],
            "destination": lane["destination"],
            "sha256": lane["seed_sha256"],
            "bytes": lane["seed_bytes"],
            "rows": lane["seed_rows"],
        }
    atomic_json(prepared, {
        "schema_version": 1,
        "status": "PASS_RESULT_BLIND_SEEDS_COPIED",
        "ground_truth_fields_used": False,
        "scoring_imported": False,
        "accuracy_computed": False,
        "lanes": report,
    })
    print("TRIVUS_R0_PREPARED")


if __name__ == "__main__":
    main()