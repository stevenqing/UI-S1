import hashlib
import json
import os
from collections import Counter
from pathlib import Path

import numpy as np
import yaml


RUN_DIR = Path(__file__).resolve().parent
ROOT = RUN_DIR.parents[2]
CONFIG_PATH = RUN_DIR / "configs/xscr_prereg.yaml"
SEAL_PATH = RUN_DIR / "SCREEN_SEAL.json"
MANIFEST_PATH = RUN_DIR / "INPUT_MANIFEST.json"
OUTPUT_PATH = RUN_DIR / "Q1.json"
RAW_PATH = RUN_DIR / "raw/q1_screen_multiplicity.jsonl"


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def summarize(counts):
    values = np.asarray(list(counts.values()), dtype=np.int64)
    if not len(values):
        raise ValueError("Q1 screen set is empty")
    quartiles = np.quantile(values, [0.25, 0.5, 0.75], method="linear")
    singleton_screens = int(np.sum(values == 1))
    rows = int(np.sum(values))
    return {
        "rows": rows,
        "screens": int(len(values)),
        "rows_per_screen_q1": float(quartiles[0]),
        "rows_per_screen_median": float(quartiles[1]),
        "rows_per_screen_q3": float(quartiles[2]),
        "singleton_screens": singleton_screens,
        "singleton_screen_fraction": float(singleton_screens / len(values)),
        "rows_on_singleton_screens": singleton_screens,
        "rows_on_singleton_screen_fraction": float(singleton_screens / rows),
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
    if OUTPUT_PATH.exists() or RAW_PATH.exists():
        raise FileExistsError("XSCR Q1 output exists")
    config = yaml.safe_load(CONFIG_PATH.read_text())
    seal = json.loads(SEAL_PATH.read_text())
    manifest = json.loads(MANIFEST_PATH.read_text())
    if (
        seal["status"] != "SEALED_BEFORE_XSCR_Q1_AND_ANY_PRIVATE_LABEL_ACCESS"
        or seal["private_labels_opened"] is not False
        or manifest["status"] != "LOCKED_XSCR_PUBLIC_INPUTS_AND_SCREEN_SEAL"
        or manifest["dependencies"]["screen_seal"]["sha256"] != sha256_file(SEAL_PATH)
        or manifest["private_labels_opened"] is not False
        or manifest["statistics_computed"] is not False
    ):
        raise PermissionError("XSCR Q1 seal contract mismatch")

    mind_path = ROOT / config["lanes"]["mind2web"]["public_records"]
    android_path = ROOT / config["lanes"]["androidcontrol_low"]["public_records"]
    if sha256_file(mind_path) != manifest["dependencies"]["mind2web_public"]["sha256"]:
        raise ValueError("Mind2Web public bank changed")
    if sha256_file(android_path) != manifest["dependencies"]["androidcontrol_public"]["sha256"]:
        raise ValueError("AndroidControl public bank changed")

    mind = [row for row in read_jsonl(mind_path) if row.get("sample_key", "").startswith("mind2web/C_uni/")]
    android = read_jsonl(android_path)
    rows_by_lane = {
        "mind2web": mind,
        "androidcontrol_low": [row for row in android if row.get("setting") == "low"],
        "androidcontrol_high": [row for row in android if row.get("setting") == "high"],
    }
    seal_maps = {
        "mind2web": {row["image_sha256"]: row for row in seal["benchmarks"]["mind2web"]["assignments"]},
        "androidcontrol_low": {row["image_sha256"]: row for row in seal["benchmarks"]["androidcontrol"]["assignments"]},
        "androidcontrol_high": {row["image_sha256"]: row for row in seal["benchmarks"]["androidcontrol"]["assignments"]},
    }
    summaries = {}
    raw_rows = []
    for lane, rows in rows_by_lane.items():
        exploratory = [row for row in rows if seal_maps[lane][row["image_sha256"]]["side"] == "exploratory"]
        counts = Counter(row["image_sha256"] for row in exploratory)
        summaries[lane] = summarize(counts)
        raw_rows.extend(
            {
                "lane": lane,
                "image_sha256": screen,
                "rows": count,
                "stratum": seal_maps[lane][screen]["stratum"],
            }
            for screen, count in sorted(counts.items())
        )

    RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
    with RAW_PATH.open("x", encoding="utf-8") as handle:
        for row in raw_rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
    output = {
        "schema_version": 1,
        "status": "PASS_XSCR_Q1_COMPLETE_AWAITING_HUMAN_DECISION",
        "evidence_status": "POST_SELECTION_FEASIBILITY",
        "private_labels_opened": False,
        "q2_computed": False,
        "q3_q4_computed": False,
        "screen_seal_sha256": sha256_file(SEAL_PATH),
        "raw": {
            "path": str(RAW_PATH.relative_to(ROOT)),
            "rows": len(raw_rows),
            "bytes": RAW_PATH.stat().st_size,
            "sha256": sha256_file(RAW_PATH),
            "write_flush_fsync_per_row": True,
        },
        "lanes": summaries,
    }
    atomic_json(OUTPUT_PATH, output)
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()