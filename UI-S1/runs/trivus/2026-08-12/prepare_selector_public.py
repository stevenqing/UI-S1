import json
import sys
from pathlib import Path


RUN_DIR = Path(__file__).resolve().parent
ROOT = RUN_DIR.parents[2]
sys.path.insert(0, str(RUN_DIR))

from r1_headroom import load_lane_maps, load_locked_recovery
from recovery_common import assert_protected_process, load_jsonl
from selector_data import (
    assert_selector_environment, audit_public_record, atomic_json, load_config,
    normalize_candidate, public_candidate_permutation, sha256_file, write_jsonl,
)


def build_public_records():
    config = load_config()
    assert_selector_environment(config)
    recovery_config, _ = load_locked_recovery()
    assert_protected_process(recovery_config)
    lanes = load_lane_maps(recovery_config)
    folds = json.loads((ROOT / config["folds"]["path"]).read_text())["pools"]
    records = []
    paired = {}
    for setting in config["settings"]:
        references = load_jsonl(ROOT / recovery_config["references"][setting]["path"])
        fold_map = folds[f"androidcontrol/{setting}"]["group_to_fold"]
        for reference in references:
            row_id = reference["id"]
            candidates = [
                normalize_candidate(lanes[setting][model][row_id]["prediction"], config)
                for model in config["canonical_private_source_order"]
            ]
            sample_key = f"androidcontrol/{setting}/{row_id}"
            public_order = public_candidate_permutation(sample_key, config["seed"])
            candidates = [candidates[index] for index in public_order]
            record = {
                "schema_version": 1,
                "sample_key": sample_key,
                "benchmark": "androidcontrol",
                "setting": setting,
                "row_id": row_id,
                "fold": int(fold_map[reference["episode_id"]]),
                "group": reference["episode_id"],
                "image_path": reference["image"],
                "image_sha256": reference["image_sha256"],
                "instruction": str(reference["instruction"]),
                "history": str(reference.get("history") or "")[-config["candidate_normalization"]["history_max_chars"]:],
                "candidates": candidates,
            }
            audit_public_record(record)
            records.append(record)
            pair = paired.setdefault(row_id, {})
            pair[setting] = (record["image_sha256"], record["group"], record["fold"])
    if len(records) != config["expected_records"] or len({row["sample_key"] for row in records}) != len(records):
        raise ValueError("TriVUS public coverage mismatch")
    if any(set(value) != {"low", "high"} or value["low"] != value["high"] for value in paired.values()):
        raise ValueError("TriVUS Low/High pairing mismatch")
    return sorted(records, key=lambda row: row["sample_key"])


def main():
    output = RUN_DIR / "data/public_records.jsonl"
    manifest_path = RUN_DIR / "data/PUBLIC_MANIFEST.json"
    if output.exists() or manifest_path.exists():
        raise FileExistsError(output)
    if any((RUN_DIR / "data").glob("private*")):
        raise PermissionError("TriVUS private labels must not exist before blind lock")
    rows = build_public_records()
    write_jsonl(output, rows)
    manifest = {
        "schema_version": 1,
        "status": "PASS_TRIVUS_PUBLIC_BANK_LOCKED",
        "records": len(rows),
        "settings": {setting: sum(row["setting"] == setting for row in rows) for setting in ("low", "high")},
        "candidates_per_record": 3,
        "public_sha256": sha256_file(output),
        "sample_keys_sha256": __import__("hashlib").sha256("\n".join(row["sample_key"] for row in rows).encode()).hexdigest(),
        "public_candidate_order": config["public_candidate_order"],
        "python": config["python"],
        "private_labels_created": False,
        "ground_truth_fields_used": False,
        "scorer_or_evaluator_imported": False,
        "label_metrics_computed": False,
    }
    atomic_json(manifest_path, manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()