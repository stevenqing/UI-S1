import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np


RUN_DIR = Path(__file__).resolve().parent
ROOT = RUN_DIR.parents[2]
DIVERSITY_DIR = ROOT / "runs/diversity-axis/2026-08-02"
sys.path.insert(0, str(RUN_DIR))
from closing_common import load_closing_pools


def sha256_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    gta1, pools = load_closing_pools()
    area = {}
    for row_id, row in gta1.items():
        width, height = row["img_size"]
        left, top, right, bottom = row["target_bbox"]
        value = (right - left) * (bottom - top) / (width * height)
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"F4 invalid target area: {row_id}")
        area[row_id] = value
    ordered = sorted(area, key=lambda row_id: (area[row_id], row_id))
    bins = np.array_split(np.asarray(ordered, dtype=object), 5)
    x3_path = DIVERSITY_DIR / "x3_curve_stats.json"
    x3 = json.loads(x3_path.read_text())
    reports = []
    for index, values in enumerate(bins):
        ids = values.tolist()
        frozen = x3["area_strata"][index]
        minimum, maximum = min(area[row_id] for row_id in ids), max(area[row_id] for row_id in ids)
        if (
            len(ids) != frozen["rows"]
            or abs(minimum - frozen["area_ratio_min"]) > 1e-18
            or abs(maximum - frozen["area_ratio_max"]) > 1e-18
        ):
            raise ValueError(f"F4/X3 area bin mismatch: {index}")
        record = {
            "bin": index,
            "rows": len(ids),
            "area_ratio_min": minimum,
            "area_ratio_mean": float(np.mean([area[row_id] for row_id in ids])),
            "area_ratio_max": maximum,
        }
        for method in ("pass_at_n", "B3_mvp", "M1_ccm"):
            v_output = pools["v_only_N12"]["evaluation"]["outputs"][method]
            mixed_output = pools["mixed_N12"]["evaluation"]["outputs"][method]
            v_accuracy = float(np.mean([v_output[row_id] for row_id in ids]))
            mixed_accuracy = float(np.mean([mixed_output[row_id] for row_id in ids]))
            record[method] = {
                "v_only": v_accuracy,
                "mixed": mixed_accuracy,
                "mixed_minus_v_only": mixed_accuracy - v_accuracy,
            }
        reports.append(record)
    smallest_delta = reports[0]["pass_at_n"]["mixed_minus_v_only"]
    result = {
        "schema_version": 1,
        "status": "PASS",
        "area_strata": reports,
        "hypothesis": {
            "name": "small_target_candidate_coverage_limited",
            "criterion": "smallest_quintile_mixed_minus_v_only_pass_at_12_le_zero",
            "smallest_quintile_delta": smallest_delta,
            "supported": smallest_delta <= 0,
        },
        "source": {"X3_sha256": sha256_file(x3_path)},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "pass_at_12": [record["pass_at_n"] for record in reports],
        "hypothesis": result["hypothesis"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()