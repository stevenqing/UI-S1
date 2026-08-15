import glob
import hashlib
import json
import math
import os
from collections import defaultdict
from pathlib import Path


RUN_DIR = Path(__file__).resolve().parent
ROOT = RUN_DIR.parents[2]
PRIVATE_MANIFEST_PATH = RUN_DIR / "PRIVATE_INPUT_MANIFEST.json"
Q2_PATH = RUN_DIR / "Q2.json"
Q2_RAW_PATH = RUN_DIR / "raw/q2_collisions.jsonl"
OUTPUT_PATH = RUN_DIR / "Q3_Q4.json"
RAW_PATH = RUN_DIR / "raw/q3_q4_rows.jsonl"


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_labels(records):
    labels = {}
    for record in records:
        path = ROOT / record["path"]
        if path.stat().st_size != record["bytes"] or sha256_file(path) != record["sha256"]:
            raise ValueError(f"private label file changed: {record['path']}")
        for row in read_jsonl(path):
            key = row["sample_key"]
            if key in labels:
                raise ValueError(f"duplicate private label: {key}")
            labels[key] = tuple(bool(value) for value in row["candidate_success"])
    return labels


def load_reference(record):
    path = ROOT / record["path"]
    if path.stat().st_size != record["bytes"] or sha256_file(path) != record["sha256"]:
        raise ValueError(f"reference file changed: {record['path']}")
    rows = read_jsonl(path)
    return {str(row["id"]): row for row in rows}


def loses_location(row, screen_rows, tolerance):
    coordinate = row["representative_coordinate"]
    return bool(
        coordinate is not None
        and any(
            other["sample_key"] != row["sample_key"]
            and other["representative_coordinate"] is not None
            and other["mode_weight"] > row["mode_weight"]
            and math.dist(coordinate, other["representative_coordinate"]) <= tolerance
            for other in screen_rows
        )
    )


def shared_target(row, screen_rows, tolerance):
    target = row.get("target_coordinate")
    return bool(
        target is not None
        and any(
            other["sample_key"] != row["sample_key"]
            and other.get("target_coordinate") is not None
            and math.dist(target, other["target_coordinate"]) <= tolerance
            for other in screen_rows
        )
    )


def mind_target(reference, image_record):
    bbox = reference["step"].get("bbox")
    if not bbox:
        return None
    diagonal = math.hypot(image_record["width"], image_record["height"])
    return [
        (float(bbox["x"]) + float(bbox["width"]) / 2) / diagonal,
        (float(bbox["y"]) + float(bbox["height"]) / 2) / diagonal,
    ]


def android_target(reference):
    coordinate = reference.get("gt_bbox")
    width, height = reference["image_size"]
    if not coordinate or coordinate[0] < 0 or coordinate[1] < 0:
        return None
    return [float(coordinate[0]) / width, float(coordinate[1]) / height]


def summarize(rows):
    total = len(rows)
    selected_correct = sum(row["selected_correct"] for row in rows)
    recoverable = sum(row["recoverable"] for row in rows)
    repairable = sum(row["repairable"] for row in rows)
    damageable = sum(row["damageable"] for row in rows)
    coordinate_target_rows = [row for row in rows if row["target_coordinate"] is not None and row["multi_row_screen"]]
    shared = sum(row["shared_target"] for row in coordinate_target_rows)
    return {
        "rows": total,
        "selected_correct_rows": selected_correct,
        "recoverable_rows": recoverable,
        "repairable_rows": repairable,
        "repairable_over_recoverable": float(repairable / recoverable) if recoverable else None,
        "repairable_over_all": float(repairable / total),
        "damageable_rows": damageable,
        "damageable_over_selected_correct": float(damageable / selected_correct) if selected_correct else None,
        "damageable_over_all": float(damageable / total),
        "signed_screening_proxy_pp": float(100 * (repairable - damageable) / total),
        "multi_row_coordinate_target_rows": len(coordinate_target_rows),
        "shared_target_rows": shared,
        "shared_target_fraction": float(shared / len(coordinate_target_rows)) if coordinate_target_rows else None,
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
        raise FileExistsError("XSCR Q3/Q4 output exists")
    private_manifest = json.loads(PRIVATE_MANIFEST_PATH.read_text())
    q2 = json.loads(Q2_PATH.read_text())
    if (
        private_manifest["status"] != "LOCKED_XSCR_PRIVATE_INPUTS_BEFORE_Q3_Q4_STATISTICS"
        or private_manifest["candidate_success_values_aggregated"] is not False
        or private_manifest["q3_q4_computed"] is not False
        or private_manifest["dependencies"]["q2"]["sha256"] != sha256_file(Q2_PATH)
        or q2["status"] != "PASS_XSCR_Q2_COMPLETE_AWAITING_HUMAN_DECISION"
        or q2["raw"]["sha256"] != sha256_file(Q2_RAW_PATH)
    ):
        raise PermissionError("XSCR Q3/Q4 input lock mismatch")

    mind_labels = load_labels(private_manifest["private_labels"]["mind2web"])
    android_labels = load_labels(private_manifest["private_labels"]["androidcontrol"])
    references = {
        "mind2web": load_reference(private_manifest["references"]["mind2web"]),
        "androidcontrol_low": load_reference(private_manifest["references"]["androidcontrol_low"]),
        "androidcontrol_high": load_reference(private_manifest["references"]["androidcontrol_high"]),
    }
    public_manifest = json.loads((RUN_DIR / "INPUT_MANIFEST.json").read_text())
    image_records = {
        "mind2web": public_manifest["dataset_snapshot"]["mind2web"]["images"],
        "androidcontrol_low": public_manifest["dataset_snapshot"]["androidcontrol"]["images"],
        "androidcontrol_high": public_manifest["dataset_snapshot"]["androidcontrol"]["images"],
    }
    q2_rows = read_jsonl(Q2_RAW_PATH)
    grouped = defaultdict(list)
    for row in q2_rows:
        grouped[(row["lane"], row["tolerance"], row["image_sha256"])].append(row)

    derived = []
    for (lane, tolerance_label, _), screen_rows in sorted(grouped.items()):
        tolerance = float(tolerance_label)
        labels = mind_labels if lane == "mind2web" else android_labels
        for row in screen_rows:
            candidate_success = labels[row["sample_key"]]
            selected_index = row["representative_candidate_index"]
            selected_correct = bool(selected_index is not None and candidate_success[selected_index])
            recoverable = bool(not selected_correct and any(candidate_success))
            loses = loses_location(row, screen_rows, tolerance)
            base_id = row["sample_key"].rsplit("/", 1)[1]
            reference = references[lane][base_id]
            target = (
                mind_target(reference, image_records[lane][row["image_sha256"]])
                if lane == "mind2web"
                else android_target(reference)
            )
            derived.append({
                **row,
                "selected_correct": selected_correct,
                "recoverable": recoverable,
                "loses_to_stronger_same_screen_mode": loses,
                "repairable": bool(recoverable and loses),
                "damageable": bool(selected_correct and loses),
                "multi_row_screen": len(screen_rows) > 1,
                "target_coordinate": target,
            })
        for row in derived[-len(screen_rows):]:
            row["shared_target"] = shared_target(row, derived[-len(screen_rows):], tolerance)

    RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
    with RAW_PATH.open("x", encoding="utf-8") as handle:
        for row in derived:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
    by_lane_tolerance = defaultdict(list)
    for row in derived:
        by_lane_tolerance[(row["lane"], row["tolerance"])].append(row)
    summaries = defaultdict(list)
    for (lane, tolerance), rows in sorted(by_lane_tolerance.items()):
        summaries[lane].append({"tolerance": float(tolerance), **summarize(rows)})
    output = {
        "schema_version": 1,
        "status": "PASS_XSCR_Q3_Q4_COMPLETE",
        "evidence_status": "POST_SELECTION_FEASIBILITY",
        "method_claim_allowed": False,
        "confirmatory_claim_allowed": False,
        "q2_sha256": sha256_file(Q2_PATH),
        "private_input_manifest_sha256": sha256_file(PRIVATE_MANIFEST_PATH),
        "raw": {
            "path": str(RAW_PATH.relative_to(ROOT)),
            "rows": len(derived),
            "bytes": RAW_PATH.stat().st_size,
            "sha256": sha256_file(RAW_PATH),
            "write_flush_fsync_per_row": True,
        },
        "lanes": dict(summaries),
    }
    atomic_json(OUTPUT_PATH, output)
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()