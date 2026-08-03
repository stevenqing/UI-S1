import argparse
import hashlib
import json
import sys
from pathlib import Path


RUN_DIR = Path(__file__).resolve().parents[1]
ROOT = RUN_DIR.parents[2]
H1_DIR = ROOT / "runs/ccm-h2h/2026-07-31/h1"
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(H1_DIR))
from guirc_port import POINT_EXPAND_SIZE, region_consistency_vote
from aggregators_coord import mvp_official


EXPECTED_GUI_RC_COMMIT = "af15ed5fe8b89b0fe5043a3e94f2984c7b126a4b"
EXPECTED_GUI_RC_BLOB = "20a577695c488ea5a75fe685dabf9e5bc1d50757"


def sha256_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def original_points(row, budget):
    original_width, original_height = row["img_size"]
    resized_width, resized_height = row["resized_size"]
    points = []
    for point in row["sample_points_resized"][:budget]:
        if point is None:
            points.append(None)
        else:
            points.append([
                point[0] * original_width / resized_width,
                point[1] * original_height / resized_height,
            ])
    return points


def point_in_bbox(point, bbox):
    return bbox[0] <= point[0] <= bbox[2] and bbox[1] <= point[1] <= bbox[3]


def evaluate(rows, budget):
    guirc_success = 0
    b3_success = 0
    pass_at_n = 0
    guirc_outputs = {}
    b3_outputs = {}
    parse_counts = []
    max_votes = []
    for row in rows:
        points = original_points(row, budget)
        valid_points = [point for point in points if point is not None]
        width, height = row["img_size"]
        guirc = region_consistency_vote(points, width, height)
        b3_candidates = [
            {"coverage": 0, "region": [0, 0, width, height]}
            for _ in valid_points
        ]
        b3_point = mvp_official(valid_points, b3_candidates) if valid_points else (0.0, 0.0)
        guirc_hit = point_in_bbox(guirc["point"], row["bbox"])
        b3_hit = point_in_bbox(b3_point, row["bbox"])
        oracle = any(point_in_bbox(point, row["bbox"]) for point in valid_points)
        guirc_outputs[row["id"]] = bool(guirc_hit)
        b3_outputs[row["id"]] = bool(b3_hit)
        guirc_success += int(guirc_hit)
        b3_success += int(b3_hit)
        pass_at_n += int(oracle)
        parse_counts.append(len(valid_points))
        max_votes.append(guirc["max_votes"])
    return {
        "budget": budget,
        "rows": len(rows),
        "accuracy": {
            "GUI_RC": guirc_success / len(rows),
            "B3_mvp": b3_success / len(rows),
            "pass_at_n": pass_at_n / len(rows),
        },
        "parse": {
            "mean_valid_samples": sum(parse_counts) / len(rows),
            "rows_with_all_samples_valid": sum(count == budget for count in parse_counts),
        },
        "GUI_RC_mean_max_votes": sum(max_votes) / len(rows),
        "outputs": {"GUI_RC": guirc_outputs, "B3_mvp": b3_outputs},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.predictions.read_text().splitlines() if line.strip()]
    if len(rows) != 1581 or [row["index"] for row in rows] != list(range(1581)):
        raise ValueError("X1 requires ordered ScreenSpot-Pro identities 0..1580")
    if any(row["samples"] != 5 or len(row["sample_points_resized"]) != 5 for row in rows):
        raise ValueError("X1 source trace requires exactly five raw samples per identity")
    if any(row["temperature"] != 0.7 for row in rows):
        raise ValueError("X1 source temperature mismatch")

    n4 = evaluate(rows, 4)
    compact_n4 = {key: value for key, value in n4.items() if key != "outputs"}
    result = {
        "schema_version": 1,
        "status": "BLOCKED_INSUFFICIENT_SAMPLES",
        "source": {
            "path": str(args.predictions),
            "sha256": sha256_file(args.predictions),
            "available_samples_per_row": 5,
            "temperature": 0.7,
            "top_p": "UNRECORDED",
        },
        "port": {
            "upstream_repository": "ZJU-REAL/GUI-RCPO",
            "upstream_commit": EXPECTED_GUI_RC_COMMIT,
            "source_blob": EXPECTED_GUI_RC_BLOB,
            "point_expand_size": POINT_EXPAND_SIZE,
            "tie_rule": "largest_max_vote_component_first_in_scan_order",
            "status": "ALGORITHM_LEVEL_PORT",
        },
        "sanity_anchor": {
            "model": "Qwen2.5-VL-3B-Instruct",
            "benchmark": "ScreenSpot-v2",
            "baseline": 0.8011,
            "GUI_RC": 0.8263,
            "tolerance_absolute": 0.01,
            "status": "NOT_RUN_MISSING_LOCAL_SCREENSPOT_V2_GENERATIONS",
        },
        "available_budget_result": {"S_only": {"4": compact_n4}},
        "required_budgets": [4, 8, 12, 16],
        "unavailable_budgets": [8, 12, 16],
        "mixed_sampling": "UNAVAILABLE_NO_CROSS_LINEAGE_SAMPLING_TRACES",
        "sampling_plus_views": "UNAVAILABLE_NO_MATCHED_SAMPLING_VIEW_TRACES",
        "slope": "NOT_EVALUATED",
        "prediction_X1": "NOT_EVALUATED",
        "kill_conditions": {"X-K3": "NOT_EVALUATED"},
        "required_followup": "freeze and generate at least 16 samples per row for each required lineage; never pad the five-sample trace",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": result["status"], "N4": compact_n4, "slope": result["slope"]}, indent=2))


if __name__ == "__main__":
    main()
