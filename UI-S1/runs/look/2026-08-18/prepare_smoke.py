import importlib.util
import json
import sys
from pathlib import Path

from look_common import ROOT, RUN_DIR, atomic_json, read_jsonl, sha256_file, write_jsonl_fsynced


PRIVATE_PATH = RUN_DIR / "raw/private_preflight_rows.jsonl"
SAMPLE_PATH = RUN_DIR / "SAMPLE_WINDOW_MANIFEST.jsonl"
IMAGE_PATH = ROOT / "runs/owin/2026-08-17/INPUT_IMAGE_MANIFEST.jsonl"
GTA1_ROOT = ROOT / "runs/ccm-h2h/2026-07-31/h1/shards/top18"
OUTPUT_PATH = RUN_DIR / "SMOKE_INPUT_MANIFEST.jsonl"
SUMMARY_PATH = RUN_DIR / "SMOKE_INPUT_SUMMARY.json"


def main():
    if OUTPUT_PATH.exists() or SUMMARY_PATH.exists():
        raise FileExistsError("LOOK smoke input exists")
    private = {row["row_id"]: row for row in read_jsonl(PRIVATE_PATH)}
    sampled = {row["row_id"] for row in read_jsonl(SAMPLE_PATH)}
    images = {row["row_id"]: row for row in read_jsonl(IMAGE_PATH)}
    gta1 = {}
    for path in sorted(GTA1_ROOT.glob("shard-*.jsonl")):
        for row in read_jsonl(path):
            gta1[row["id"]] = row
    eligible = sorted((row_id for row_id, row in private.items() if row["eligible"] and row_id not in sampled), key=lambda row_id: (__import__("hashlib").sha256(f"LOOK|20260818|SMOKE|{row_id}".encode()).hexdigest(), row_id))[:3]
    if len(eligible) != 3:
        raise ValueError("LOOK requires three nonsample smoke rows")
    records = []
    for row_id in eligible:
        row = private[row_id]
        windows = []
        for kind, window in (("main", row["main_window"]), ("sensitivity", row["sensitivity_window"]), ("null", row["null_window"]["window"])):
            windows.append({"kind": kind, "final_window": window["final_window"], "dimensions": window["dimensions"], "area_fraction": window["area_fraction"]})
        records.append({"sample_id": f"look-smoke-{row_id}", "row_id": row_id, "instruction": gta1[row_id]["instruction"], "image": images[row_id], "windows": windows})
    write_jsonl_fsynced(OUTPUT_PATH, records)
    summary = {"schema_version": 1, "status": "PASS_LOOK_SMOKE_INPUT_FROZEN", "gpu_used": False, "gpu_authorized": False, "rows": 3, "calls": 9, "row_ids": eligible, "overlap_with_formal": sorted(set(eligible) & sampled), "manifest": {"path": str(OUTPUT_PATH.relative_to(ROOT)), "bytes": OUTPUT_PATH.stat().st_size, "sha256": sha256_file(OUTPUT_PATH)}, "next_action": "COMMIT_RUNNER_TESTS_AND_SEPARATE_AUTHORIZATION"}
    atomic_json(SUMMARY_PATH, summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()