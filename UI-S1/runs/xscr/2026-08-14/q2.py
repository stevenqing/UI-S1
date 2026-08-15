import hashlib
import json
import math
import os
from collections import defaultdict
from pathlib import Path

import yaml


RUN_DIR = Path(__file__).resolve().parent
ROOT = RUN_DIR.parents[2]
CONFIG_PATH = RUN_DIR / "configs/xscr_prereg.yaml"
SEAL_PATH = RUN_DIR / "SCREEN_SEAL.json"
MANIFEST_PATH = RUN_DIR / "INPUT_MANIFEST.json"
Q1_PATH = RUN_DIR / "Q1.json"
DECISION_PATH = RUN_DIR / "DECISION_Q1.json"
OUTPUT_PATH = RUN_DIR / "Q2.json"
RAW_PATH = RUN_DIR / "raw/q2_collisions.jsonl"


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def candidate_equivalent(left, right, tolerance):
    if left["action"] != right["action"]:
        return False
    left_coordinate = left["coordinate"]
    right_coordinate = right["coordinate"]
    if left_coordinate is None or right_coordinate is None:
        return left_coordinate is None and right_coordinate is None and left["parameter"] == right["parameter"]
    return math.dist(left_coordinate, right_coordinate) <= tolerance


def complete_link_classes(candidates, tolerance):
    classes = []
    for index, candidate in enumerate(candidates):
        for members in classes:
            if all(candidate_equivalent(candidate, candidates[member], tolerance) for member in members):
                members.append(index)
                break
        else:
            classes.append([index])
    return classes


def select_mode(candidates, tolerance):
    parsed = [candidate for candidate in candidates if candidate["parse_ok"]]
    if not parsed:
        return None
    classes = complete_link_classes(parsed, tolerance)
    winner = max(classes, key=lambda members: (len(members), -min(parsed[index]["order"] for index in members)))
    representative_index = min(winner, key=lambda index: parsed[index]["order"])
    representative = parsed[representative_index]
    return {
        "action": representative["action"],
        "coordinate": representative["coordinate"],
        "mode_weight": len(winner),
        "representative_candidate_index": representative["order"],
    }


def transformed_candidates(row, lane, image_record):
    output = []
    diagonal = math.hypot(image_record["width"], image_record["height"])
    for order, candidate in enumerate(row["candidates"]):
        coordinate = candidate.get("coordinate")
        if coordinate is not None:
            if lane == "mind2web":
                coordinate = [
                    float(coordinate[0]) * image_record["width"] / diagonal,
                    float(coordinate[1]) * image_record["height"] / diagonal,
                ]
            else:
                coordinate = [float(coordinate[0]), float(coordinate[1])]
        output.append({
            "action": str(candidate.get("action") or ""),
            "coordinate": coordinate,
            "parameter": str(candidate.get("parameter") or ""),
            "parse_ok": bool(candidate.get("parse_ok")),
            "order": order,
        })
    return output


def collision_flags(modes, tolerance):
    by_screen = defaultdict(list)
    for row_id, mode in modes.items():
        by_screen[mode["image_sha256"]].append((row_id, mode))
    flags = {}
    for members in by_screen.values():
        for row_id, mode in members:
            coordinate = mode["coordinate"]
            flags[row_id] = bool(
                coordinate is not None
                and any(
                    other_id != row_id
                    and other["coordinate"] is not None
                    and math.dist(coordinate, other["coordinate"]) <= tolerance
                    for other_id, other in members
                )
            )
    return flags


def tolerance_label(value):
    return format(float(value), ".17g")


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
        raise FileExistsError("XSCR Q2 output exists")
    config = yaml.safe_load(CONFIG_PATH.read_text())
    seal = json.loads(SEAL_PATH.read_text())
    manifest = json.loads(MANIFEST_PATH.read_text())
    q1 = json.loads(Q1_PATH.read_text())
    decision = json.loads(DECISION_PATH.read_text())
    if (
        seal["private_labels_opened"] is not False
        or manifest["private_labels_opened"] is not False
        or q1["status"] != "PASS_XSCR_Q1_COMPLETE_AWAITING_HUMAN_DECISION"
        or q1["private_labels_opened"] is not False
        or decision["decision"] != "PROCEED"
        or decision["q2_authorized"] is not True
        or decision["q3_q4_authorized"] is not False
    ):
        raise PermissionError("XSCR Q2 authorization mismatch")

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
    image_records = {
        "mind2web": manifest["dataset_snapshot"]["mind2web"]["images"],
        "androidcontrol_low": manifest["dataset_snapshot"]["androidcontrol"]["images"],
        "androidcontrol_high": manifest["dataset_snapshot"]["androidcontrol"]["images"],
    }

    raw_rows = []
    summaries = {}
    for lane, rows in rows_by_lane.items():
        exploratory = [row for row in rows if seal_maps[lane][row["image_sha256"]]["side"] == "exploratory"]
        lane_results = []
        for tolerance in config["lanes"][lane]["tolerances"]:
            modes = {}
            for row in exploratory:
                mode = select_mode(
                    transformed_candidates(row, lane, image_records[lane][row["image_sha256"]]),
                    float(tolerance),
                )
                if mode is None:
                    mode = {"action": None, "coordinate": None, "mode_weight": 0, "representative_candidate_index": None}
                modes[row["sample_key"]] = {**mode, "image_sha256": row["image_sha256"]}
            flags = collision_flags(modes, float(tolerance))
            collision_rows = sum(flags.values())
            collision_screens = len({modes[row_id]["image_sha256"] for row_id, value in flags.items() if value})
            label = tolerance_label(tolerance)
            lane_results.append({
                "tolerance": float(tolerance),
                "rows": len(exploratory),
                "collision_rows": collision_rows,
                "collision_row_fraction": float(collision_rows / len(exploratory)),
                "collision_screens": collision_screens,
            })
            raw_rows.extend(
                {
                    "lane": lane,
                    "sample_key": row_id,
                    "image_sha256": mode["image_sha256"],
                    "tolerance": label,
                    "mode_weight": mode["mode_weight"],
                    "representative_candidate_index": mode["representative_candidate_index"],
                    "representative_action": mode["action"],
                    "representative_coordinate": mode["coordinate"],
                    "collision": flags[row_id],
                }
                for row_id, mode in sorted(modes.items())
            )
        summaries[lane] = lane_results

    RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
    with RAW_PATH.open("x", encoding="utf-8") as handle:
        for row in raw_rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
    output = {
        "schema_version": 1,
        "status": "PASS_XSCR_Q2_COMPLETE_AWAITING_HUMAN_DECISION",
        "evidence_status": "POST_SELECTION_FEASIBILITY",
        "private_labels_opened": False,
        "q3_q4_computed": False,
        "screen_seal_sha256": sha256_file(SEAL_PATH),
        "q1_sha256": sha256_file(Q1_PATH),
        "decision_q1_sha256": sha256_file(DECISION_PATH),
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