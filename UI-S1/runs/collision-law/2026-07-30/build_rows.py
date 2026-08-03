import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from scoring import EXPECTED_SUMMARY_SHA256, sha256_file, verify_locked_summaries


RUN_DIR = Path(__file__).resolve().parent
ROOT = RUN_DIR.parents[2]
UPSTREAM_ROWS = ROOT / "runs/complementarity/2026-07-30/rows.parquet"
UPSTREAM_MANIFEST = ROOT / "runs/complementarity/2026-07-30/rows_manifest.json"
EXPECTED_ROWS = 102054


def gt_element_area(bench: str, bbox) -> float:
    if bench != "mind2web" or bbox is None:
        return float("nan")
    if len(bbox) != 4:
        raise ValueError(f"invalid Mind2Web bbox: {bbox}")
    x0, y0, x1, y1 = bbox
    area = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    if not math.isfinite(area):
        raise ValueError(f"non-finite Mind2Web bbox area: {area}")
    return area


def expected_lane_successes(manifest: dict) -> dict[tuple[str, str, str], int]:
    output = {}
    for lane, values in manifest["androidcontrol"]["lanes"].items():
        setting, model = lane.split("/", 1)
        output[("androidcontrol", setting, model)] = values["successes"]
    for lane, values in manifest["mind2web"]["lanes"].items():
        setting, model = lane.split("/", 1)
        output[("mind2web", setting, model)] = values["successes"]
    return output


def validate_source(table: pa.Table, manifest: dict) -> None:
    if table.num_rows != EXPECTED_ROWS or manifest["rows"] != EXPECTED_ROWS:
        raise ValueError("upstream tidy row count mismatch")
    actual_hash = sha256_file(UPSTREAM_ROWS)
    if actual_hash != manifest["rows_parquet_sha256"]:
        raise ValueError(f"upstream rows hash mismatch: {actual_hash}")
    if manifest["locked_input_sha256"]["androidcontrol_summary.json"] != EXPECTED_SUMMARY_SHA256["androidcontrol"]:
        raise ValueError("upstream AndroidControl lock differs from collision lock")
    if manifest["locked_input_sha256"]["mind2web_summary.json"] != EXPECTED_SUMMARY_SHA256["mind2web"]:
        raise ValueError("upstream Mind2Web lock differs from collision lock")

    clean = table.filter(pc.invert(table["quarantine"]))
    successes = Counter()
    for bench, setting, model, success in zip(
        clean["bench"].to_pylist(), clean["setting"].to_pylist(),
        clean["model"].to_pylist(), clean["success"].to_pylist(),
    ):
        successes[(bench, setting, model)] += int(success)
    expected = expected_lane_successes(manifest)
    for key, expected_successes in expected.items():
        quarantine_successes = sum(
            int(success)
            for bench, setting, model, success, quarantine in zip(
                table["bench"].to_pylist(), table["setting"].to_pylist(),
                table["model"].to_pylist(), table["success"].to_pylist(),
                table["quarantine"].to_pylist(),
            )
            if (bench, setting, model) == key and quarantine
        )
        if successes[key] + quarantine_successes != expected_successes:
            raise ValueError(f"lane success mismatch: {key}")


def build_table() -> tuple[pa.Table, dict]:
    verify_locked_summaries()
    manifest = json.loads(UPSTREAM_MANIFEST.read_text())
    table = pq.read_table(UPSTREAM_ROWS)
    validate_source(table, manifest)
    benches = table["bench"].to_pylist()
    models = table["model"].to_pylist()
    bboxes = table["gt_bbox"].to_pylist()
    view_id = pa.array(["full"] * table.num_rows, type=pa.string())
    pred_source = pa.array([f"{model}__full" for model in models], type=pa.string())
    areas = pa.array(
        [gt_element_area(bench, bbox) for bench, bbox in zip(benches, bboxes)],
        type=pa.float64(),
    )
    extended = table.append_column("view_id", view_id)
    extended = extended.append_column("pred_source", pred_source)
    extended = extended.append_column("gt_element_area", areas)
    if len(set(extended.schema.names)) != len(extended.schema.names):
        raise ValueError("duplicate output schema field")
    area_values = extended["gt_element_area"].to_pylist()
    m2w_areas = [area for bench, area in zip(benches, area_values) if bench == "mind2web"]
    ac_areas = [area for bench, area in zip(benches, area_values) if bench == "androidcontrol"]
    if any(math.isnan(area) for area in m2w_areas[:22880]):
        raise ValueError("visual Mind2Web row has missing GT area")
    if any(not math.isnan(area) for area in ac_areas):
        raise ValueError("AndroidControl row unexpectedly has GT area")
    out_of_unit_identities = {
        row_id
        for bench, row_id, area in zip(benches, table["row_id"].to_pylist(), area_values)
        if bench == "mind2web" and not math.isnan(area) and area > 1.0
    }
    details = {
        "source_rows_sha256": manifest["rows_parquet_sha256"],
        "source_manifest_sha256": sha256_file(UPSTREAM_MANIFEST),
        "rows": extended.num_rows,
        "new_columns": ["view_id", "pred_source", "gt_element_area"],
        "view_counts": dict(Counter(extended["view_id"].to_pylist())),
        "pred_sources": len(set(extended["pred_source"].to_pylist())),
        "mind2web_area": {
            "rows_with_area": sum(not math.isnan(value) for value in m2w_areas),
            "minimum": min(value for value in m2w_areas if not math.isnan(value)),
            "maximum": max(value for value in m2w_areas if not math.isnan(value)),
            "out_of_unit_unique_identities": len(out_of_unit_identities),
            "out_of_unit_policy": "preserve released normalized bbox area; flag for annotation audit; assign to regular bin",
        },
    }
    return extended, details


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    table, details = build_table()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, args.output, compression="zstd", use_dictionary=True)
    result = {
        "status": "PASS",
        **details,
        "output_sha256": sha256_file(args.output),
        "schema": str(table.schema),
        "default_filter": "quarantine == false",
    }
    args.manifest.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: result[key] for key in ("status", "rows", "new_columns", "view_counts", "pred_sources")}, indent=2))


if __name__ == "__main__":
    main()