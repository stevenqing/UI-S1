import json
from collections import Counter
from pathlib import Path

from owin_common import (
    ROOT,
    RUN_DIR,
    atomic_json,
    contains_bbox,
    contains_center,
    read_jsonl,
    sha256_file,
    summarize,
    tiling_layout,
    union_area,
    write_jsonl_fsynced,
)


PREFLIGHT_PATH = RUN_DIR / "PREFLIGHT.json"
REGION_PATH = ROOT / "runs/allocation-law/2026-08-01/raw/shared_regions_n12.jsonl"
COVER_ROWS_PATH = ROOT / "runs/cover/2026-08-16/raw/arm_a_rows.jsonl"
GTA1_ROOT = ROOT / "runs/ccm-h2h/2026-07-31/h1/shards/top18"
OUTPUT_PATH = RUN_DIR / "ARM_B.json"
RAW_PATH = RUN_DIR / "raw/arm_b_rows.jsonl"


def evaluate_rectangles(rectangles, target_bbox):
    return {
        "union_area": union_area(rectangles),
        "center_count": sum(contains_center(rectangle, target_bbox) for rectangle in rectangles),
        "full_bbox_count": sum(contains_bbox(rectangle, target_bbox) for rectangle in rectangles),
    }


def distribution(count):
    if count == 0:
        return "uncovered_0"
    if count == 11:
        return "common_11"
    return "partial_1_10"


def main():
    if OUTPUT_PATH.exists() or RAW_PATH.exists():
        raise FileExistsError("OWIN Arm B output exists")
    preflight = json.loads(PREFLIGHT_PATH.read_text())
    if preflight["status"] != "PASS_OWIN_GEOMETRY_CALIBRATION_PREFLIGHT_NO_MODEL_OUTPUT" or preflight["gpu_authorized"] is not False:
        raise PermissionError("OWIN Arm B preflight mismatch")
    regions = {row["id"]: row for row in read_jsonl(REGION_PATH)}
    cover = {row["row_id"]: row for row in read_jsonl(COVER_ROWS_PATH)}
    gta1_rows = {}
    for path in sorted(GTA1_ROOT.glob("shard-*.jsonl")):
        for row in read_jsonl(path):
            if row["id"] in gta1_rows:
                raise ValueError(f"OWIN duplicate GTA1 row: {row['id']}")
            gta1_rows[row["id"]] = row
    if set(gta1_rows) != set(regions) or set(cover) != set(regions):
        raise ValueError("OWIN Arm B identity mismatch")
    raw = []
    for row_id in sorted(regions):
        width, height = regions[row_id]["img_size"]
        target_bbox = gta1_rows[row_id]["target_bbox"]
        for count in range(4, 12):
            tiling = tiling_layout(width, height, count)
            prefix = regions[row_id]["regions"][1 : count + 1]
            raw.append({"row_id": row_id, "application": cover[row_id]["application"], "existing_center_stratum": cover[row_id]["target_stratum"], "N": count, "width": width, "height": height, "tiling": {**tiling, **evaluate_rectangles(tiling["rectangles"], target_bbox)}, "existing_prefix": {"rectangles": prefix, **evaluate_rectangles(prefix, target_bbox)}})
    write_jsonl_fsynced(RAW_PATH, raw)
    summaries = {}
    for count in range(4, 12):
        rows = [row for row in raw if row["N"] == count]
        summaries[str(count)] = {}
        for name in ("tiling", "existing_prefix"):
            summaries[str(count)][name] = {
                "union_fraction": summarize([row[name]["union_area"] / (row["width"] * row["height"]) for row in rows]),
                "center_covered_rows": sum(row[name]["center_count"] > 0 for row in rows),
                "center_uncovered_rows": sum(row[name]["center_count"] == 0 for row in rows),
                "full_bbox_covered_rows": sum(row[name]["full_bbox_count"] > 0 for row in rows),
                "full_bbox_uncovered_rows": sum(row[name]["full_bbox_count"] == 0 for row in rows),
                "center_transition": dict(Counter(f"{row['existing_center_stratum']}->{('covered' if row[name]['center_count'] > 0 else 'uncovered')}" for row in rows)),
            }
    output = {"schema_version": 1, "status": "PASS_OWIN_ARM_B_COMPLETE_ZERO_GPU", "gpu_used": False, "gpu_authorized": False, "rows": 1581, "window_counts": list(range(4, 12)), "summaries": summaries, "raw": {"path": str(RAW_PATH.relative_to(ROOT)), "rows": len(raw), "bytes": RAW_PATH.stat().st_size, "sha256": sha256_file(RAW_PATH), "write_flush_fsync_per_row": True}}
    atomic_json(OUTPUT_PATH, output)
    print(json.dumps({"status": output["status"], "summaries": summaries, "gpu_authorized": False}, indent=2))


if __name__ == "__main__":
    main()