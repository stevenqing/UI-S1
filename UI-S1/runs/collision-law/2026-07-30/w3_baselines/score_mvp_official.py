import argparse
import hashlib
import json
from pathlib import Path

from mvp_port import mvp_official_code, mvp_paper_centroid, multi_coordinate_clustering


RUN_DIR = Path(__file__).resolve().parent
DATA_DIR = RUN_DIR.parent / "w3_assets/ScreenSpot-Pro"


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def expected_ids():
    rows = []
    for path in sorted((DATA_DIR / "annotations").glob("*.json")):
        rows.extend(json.loads(path.read_text()))
    if len(rows) != 1581 or len({row["id"] for row in rows}) != 1581:
        raise ValueError("ScreenSpot-Pro identity coverage mismatch")
    return {row["id"] for row in rows}


def point_in_bbox(point, bbox):
    return bbox[0] <= point[0] <= bbox[2] and bbox[1] <= point[1] <= bbox[3]


def score_rows(rows, identities):
    if len(rows) != len(identities) or {row["id"] for row in rows} != identities:
        raise ValueError("official MVP result identity mismatch")
    counts = {"bare_full": 0, "official_code": 0, "paper_centroid": 0, "graph_centroid_ablation": 0}
    for row in rows:
        predictions = row["all_predictions"]
        points = [tuple(map(float, prediction["point"])) for prediction in predictions]
        coverage = [float(prediction["coverage"]) if not isinstance(prediction["coverage"], str) else 0.0 for prediction in predictions]
        official = mvp_official_code(points, coverage)
        source_final = tuple(map(float, row["final_prediction"]["point"]))
        if official.coordinate != source_final:
            raise ValueError(f"official MVP aggregation mismatch for {row['id']}")
        paper = mvp_paper_centroid(points, coverage)
        width, height = row.get("img_size", row.get("image_size", (0, 0)))
        if not width or not height:
            width = max(point[0] for point in points) + 1
            height = max(point[1] for point in points) + 1
        normalized = [(point[0] / width, point[1] / height) for point in points]
        graph = multi_coordinate_clustering(normalized, (width, height))
        bbox = row["target_bbox"]
        counts["bare_full"] += point_in_bbox(points[0], bbox)
        counts["official_code"] += point_in_bbox(official.coordinate, bbox)
        counts["paper_centroid"] += point_in_bbox(paper.coordinate, bbox)
        graph_pixels = (graph.coordinate[0] * width, graph.coordinate[1] * height)
        counts["graph_centroid_ablation"] += point_in_bbox(graph_pixels, bbox)
    total = len(rows)
    return {
        "status": "PASS",
        "rows": total,
        "accuracy": {name: correct / total for name, correct in counts.items()},
        "correct": counts,
        "reported_anchor": 0.617,
        "official_code_delta_to_anchor": counts["official_code"] / total - 0.617,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = json.loads(args.input.read_text())
    result = score_rows(payload["detailed_results"], expected_ids())
    result["source_sha256"] = sha256_file(args.input)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()