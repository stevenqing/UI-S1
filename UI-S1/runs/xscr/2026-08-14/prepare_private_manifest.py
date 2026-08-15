import glob
import hashlib
import json
import os
from pathlib import Path

import yaml


RUN_DIR = Path(__file__).resolve().parent
ROOT = RUN_DIR.parents[2]
CONFIG_PATH = RUN_DIR / "configs/xscr_prereg.yaml"
PUBLIC_MANIFEST_PATH = RUN_DIR / "INPUT_MANIFEST.json"
Q2_PATH = RUN_DIR / "Q2.json"
DECISION_PATH = RUN_DIR / "DECISION_Q2.json"
OUTPUT_PATH = RUN_DIR / "PRIVATE_INPUT_MANIFEST.json"
MIND_REFERENCE = ROOT / "runs/xfer/2026-08-07/data/mind2web/mind2web_test_task.jsonl"


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def inspect_label_files(pattern, prefix, candidate_count, expected_records):
    paths = [Path(path) for path in sorted(glob.glob(str(ROOT / pattern)))]
    if len(paths) != 5:
        raise ValueError(f"expected five private label folds: {pattern}")
    seen = set()
    files = []
    selected = 0
    for path in paths:
        rows = read_jsonl(path)
        for row in rows:
            if set(row) != {"schema_version", "sample_key", "candidate_success"}:
                raise ValueError(f"unexpected private label schema: {path}")
            if len(row["candidate_success"]) != candidate_count:
                raise ValueError(f"candidate label width mismatch: {path}")
            key = row["sample_key"]
            if key in seen:
                raise ValueError(f"duplicate private sample key: {key}")
            seen.add(key)
            selected += int(key.startswith(prefix))
        files.append({
            "path": str(path.relative_to(ROOT)),
            "rows": len(rows),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        })
    if selected != expected_records:
        raise ValueError(f"private label identity mismatch: {prefix} {selected}")
    return files


def reference_record(path, expected_rows):
    rows = read_jsonl(path)
    if len(rows) != expected_rows:
        raise ValueError(f"reference row mismatch: {path}")
    row_ids = [str(row["id"]) for row in rows]
    if len(row_ids) != len(set(row_ids)):
        raise ValueError(f"duplicate reference row IDs: {path}")
    return {
        "path": str(path.relative_to(ROOT)),
        "rows": len(rows),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "row_ids_sha256": hashlib.sha256("\n".join(sorted(row_ids)).encode()).hexdigest(),
    }


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def main():
    if OUTPUT_PATH.exists():
        raise FileExistsError(OUTPUT_PATH)
    config = yaml.safe_load(CONFIG_PATH.read_text())
    public_manifest = json.loads(PUBLIC_MANIFEST_PATH.read_text())
    q2 = json.loads(Q2_PATH.read_text())
    decision = json.loads(DECISION_PATH.read_text())
    if (
        public_manifest["status"] != "LOCKED_XSCR_PUBLIC_INPUTS_AND_SCREEN_SEAL"
        or q2["status"] != "PASS_XSCR_Q2_COMPLETE_AWAITING_HUMAN_DECISION"
        or q2["private_labels_opened"] is not False
        or decision["decision"] != "PROCEED"
        or decision["private_labels_authorized"] is not True
        or decision["q3_q4_authorized"] is not True
    ):
        raise PermissionError("XSCR private-input authorization mismatch")

    mind_labels = inspect_label_files(
        config["lanes"]["mind2web"]["private_label_glob"],
        "mind2web/C_uni/",
        12,
        2080,
    )
    android_labels = inspect_label_files(
        config["lanes"]["androidcontrol_low"]["private_label_glob"],
        "androidcontrol/",
        3,
        4000,
    )
    low_reference = ROOT / config["lanes"]["androidcontrol_low"]["reference"]
    high_reference = ROOT / config["lanes"]["androidcontrol_high"]["reference"]
    output = {
        "schema_version": 1,
        "status": "LOCKED_XSCR_PRIVATE_INPUTS_BEFORE_Q3_Q4_STATISTICS",
        "candidate_success_values_aggregated": False,
        "q3_q4_computed": False,
        "dependencies": {
            "config": {"path": str(CONFIG_PATH.relative_to(ROOT)), "sha256": sha256_file(CONFIG_PATH)},
            "public_manifest": {"path": str(PUBLIC_MANIFEST_PATH.relative_to(ROOT)), "sha256": sha256_file(PUBLIC_MANIFEST_PATH)},
            "q2": {"path": str(Q2_PATH.relative_to(ROOT)), "sha256": sha256_file(Q2_PATH)},
            "decision_q2": {"path": str(DECISION_PATH.relative_to(ROOT)), "sha256": sha256_file(DECISION_PATH)},
        },
        "private_labels": {
            "mind2web": mind_labels,
            "androidcontrol": android_labels,
        },
        "references": {
            "mind2web": reference_record(MIND_REFERENCE, 2080),
            "androidcontrol_low": reference_record(low_reference, 2000),
            "androidcontrol_high": reference_record(high_reference, 2000),
        },
    }
    atomic_json(OUTPUT_PATH, output)
    print(json.dumps({
        "status": output["status"],
        "candidate_success_values_aggregated": False,
        "q3_q4_computed": False,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()