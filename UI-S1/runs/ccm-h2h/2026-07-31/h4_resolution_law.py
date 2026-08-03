import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from common import ROOT, UPSTREAM, write_json


def equal_count_bins(rows, bins=5):
    ordered = sorted(rows, key=lambda row: row["x"])
    groups = np.array_split(np.asarray(ordered, dtype=object), bins)
    output = []
    for index, group in enumerate(groups):
        members = list(group)
        output.append({
            "bin": index,
            "rows": len(members),
            "x_mean": float(np.mean([row["x"] for row in members])),
            "bare_accuracy": float(np.mean([row["bare"] for row in members])),
            "zoom_accuracy": float(np.mean([row["zoom"] for row in members])),
            "delta": float(np.mean([row["zoom"] - row["bare"] for row in members])),
        })
    return output


def screenspot_rows():
    source = json.loads((UPSTREAM / "w3_artifacts/mvp_official_gta1_screenspot_pro/source_result.json").read_text())
    annotations = {}
    for path in (UPSTREAM / "w3_assets/ScreenSpot-Pro/annotations").glob("*.json"):
        for row in json.loads(path.read_text()):
            if row["id"] in annotations:
                raise ValueError(f"duplicate ScreenSpot-Pro id: {row['id']}")
            annotations[row["id"]] = row
    if len(annotations) != 1581:
        raise ValueError("ScreenSpot-Pro annotation coverage mismatch")
    output = []
    for row in source["detailed_results"]:
        annotation = annotations[row["id"]]
        width, height = annotation["img_size"]
        x0, y0, x1, y1 = row["target_bbox"]
        area = (x1 - x0) * (y1 - y0)
        if area <= 0:
            raise ValueError("ScreenSpot-Pro target area must be positive")
        bare = bool(row["all_predictions"][0]["in_bbox"])
        final_x, final_y = row["final_prediction"]["point"]
        zoom = x0 <= final_x <= x1 and y0 <= final_y <= y1
        output.append({
            "id": row["id"],
            "group": row["application"],
            "x": math.log(width * height / area),
            "bare": int(bare),
            "zoom": int(zoom),
        })
    if len(output) != 1581:
        raise ValueError("ScreenSpot-Pro coverage mismatch")
    return output


def mind2web_rows():
    import pyarrow.parquet as pq
    table = pq.read_table(
        UPSTREAM / "rows.parquet",
        filters=[("bench", "=", "mind2web"), ("setting", "=", "visual"), ("model", "=", "tongui-7b")],
        columns=["row_id", "episode_id", "gt_element_area", "success"],
    ).to_pylist()
    v2 = {
        f"{row['annot_id']}__{row['action_uid']}": row
        for row in [json.loads(line) for line in (UPSTREAM / "w2_artifacts/mind2web/tongui-7b/v2/scored_rows.jsonl").read_text().splitlines() if line.strip()]
    }
    v3 = {
        f"{row['annot_id']}__{row['action_uid']}": row
        for row in [json.loads(line) for line in (UPSTREAM / "w2_artifacts/mind2web/tongui-7b/v3/scored_rows.jsonl").read_text().splitlines() if line.strip()]
    }
    output = []
    excluded = []
    for row in table:
        area = row["gt_element_area"]
        if area is None or not math.isfinite(area) or area <= 0:
            excluded.append({"row_id": row["row_id"], "area": area, "reason": "NONPOSITIVE_AREA"})
            continue
        if row["row_id"] not in v2 or row["row_id"] not in v3:
            raise ValueError("Mind2Web view identity mismatch")
        output.append({
            "id": row["row_id"],
            "group": row["episode_id"],
            "x": math.log(1.0 / area),
            "bare": int(row["success"]),
            "zoom": (int(v2[row["row_id"]]["success"]) + int(v3[row["row_id"]]["success"])) / 2,
        })
    if len(output) + len(excluded) != 2080 or len(excluded) != 1:
        raise ValueError("Mind2Web area audit mismatch")
    return output, excluded


def analyze(name, rows):
    deltas = [row["zoom"] - row["bare"] for row in rows]
    correlation = spearmanr([row["x"] for row in rows], deltas)
    return {
        "rows": len(rows),
        "bare_accuracy": float(np.mean([row["bare"] for row in rows])),
        "zoom_accuracy": float(np.mean([row["zoom"] for row in rows])),
        "delta": float(np.mean(deltas)),
        "row_spearman_rho": float(correlation.statistic),
        "row_spearman_p": float(correlation.pvalue),
        "equal_count_bins": equal_count_bins(rows),
        "name": name,
    }


def pdf_escape(value):
    return value.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def plot(results, output):
    width, height = 720, 460
    left, right, bottom, top = 80, 680, 70, 420
    series = {
        name: [(item["x_mean"], 100 * item["delta"]) for item in result["equal_count_bins"]]
        for name, result in results.items()
    }
    all_x = [x for values in series.values() for x, _ in values]
    all_y = [y for values in series.values() for _, y in values] + [0.0]
    x_min, x_max = min(all_x), max(all_x)
    y_min, y_max = min(all_y), max(all_y)
    y_pad = max((y_max - y_min) * 0.1, 1.0)
    y_min -= y_pad
    y_max += y_pad

    def px(value):
        return left + (value - x_min) / (x_max - x_min) * (right - left)

    def py(value):
        return bottom + (value - y_min) / (y_max - y_min) * (top - bottom)

    content = ["0.8 w", f"{left} {bottom} m {left} {top} l {right} {top} l {right} {bottom} l h S"]
    if y_min <= 0 <= y_max:
        content.extend(["0.7 0.7 0.7 RG", f"{left} {py(0):.2f} m {right} {py(0):.2f} l S", "0 0 0 RG"])
    colors = ((0.1, 0.35, 0.7), (0.75, 0.2, 0.15))
    for series_index, (name, values) in enumerate(series.items()):
        red, green, blue = colors[series_index % len(colors)]
        content.append(f"{red} {green} {blue} RG 1.8 w")
        for index, (x_value, y_value) in enumerate(values):
            command = "m" if index == 0 else "l"
            content.append(f"{px(x_value):.2f} {py(y_value):.2f} {command}")
        content.append("S")
        for x_value, y_value in values:
            content.append(f"{px(x_value)-2:.2f} {py(y_value)-2:.2f} 4 4 re f")
        legend_y = top - 18 * series_index
        content.extend([
            f"{red} {green} {blue} RG {right-150} {legend_y} m {right-130} {legend_y} l S",
            "0 0 0 RG", "BT /F1 10 Tf", f"{right-125} {legend_y-3} Td ({pdf_escape(name)}) Tj ET",
        ])
    content.extend([
        "0 0 0 RG", "BT /F1 11 Tf", f"{left+150} 30 Td (log\\(screen area / target bbox area\\)) Tj ET",
        "BT /F1 11 Tf 0 1 -1 0 20 130 Tm (Zoom minus bare accuracy \\(pp\\)) Tj ET",
        "BT /F1 9 Tf", f"{left} {bottom-15} Td ({x_min:.2f}) Tj ET",
        "BT /F1 9 Tf", f"{right-20} {bottom-15} Td ({x_max:.2f}) Tj ET",
        "BT /F1 9 Tf", f"{left-45} {bottom} Td ({y_min:.1f}) Tj ET",
        "BT /F1 9 Tf", f"{left-45} {top} Td ({y_max:.1f}) Tj ET",
    ])
    stream = "\n".join(content).encode("ascii")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {width} {height}] /Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>".encode(),
        f"<< /Length {len(stream)} >>\nstream\n".encode() + stream + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    document = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for index, value in enumerate(objects, start=1):
        offsets.append(len(document))
        document.extend(f"{index} 0 obj\n".encode() + value + b"\nendobj\n")
    xref = len(document)
    document.extend(f"xref\n0 {len(objects)+1}\n0000000000 65535 f\n".encode())
    for offset in offsets[1:]:
        document.extend(f"{offset:010d} 00000 n\n".encode())
    document.extend(f"trailer << /Size {len(objects)+1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode())
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(document)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--figure", type=Path, required=True)
    args = parser.parse_args()
    mind2web, mind2web_excluded = mind2web_rows()
    results = {
        "screenspot-pro": analyze("ScreenSpot-Pro", screenspot_rows()),
        "mind2web": analyze("Mind2Web", mind2web),
    }
    plot(results, args.figure)
    result = {
        "status": "PASS_DESCRIPTIVE_PRIMARY_BLOCKED",
        "definition": "x=log(screen_area/target_bbox_area); y=zoom_correct-bare_correct",
        "benchmarks": results,
        "androidcontrol": {
            "status": "UNAVAILABLE",
            "reason": "W2 has point GT only; exact SHA256 link to Curated images matched 0/7708",
        },
        "mind2web_excluded": mind2web_excluded,
        "primary": {
            "required_benchmarks": 3,
            "available_benchmarks": 2,
            "minimum_rho": 0.7,
            "prediction_satisfied": False,
            "kill_condition": "H-K4_TRIGGERED_AREA_AXIS_UNAUDITABLE",
        },
        "figure": str(args.figure),
    }
    write_json(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
