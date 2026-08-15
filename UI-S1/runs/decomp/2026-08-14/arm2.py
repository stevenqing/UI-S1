import builtins
import hashlib
import io
import json
import os
from collections import Counter, defaultdict
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import yaml


RUN_DIR = Path(__file__).resolve().parent
ROOT = RUN_DIR.parents[2]
CONFIG_PATH = RUN_DIR / "configs/decomp_prereg.yaml"
PREFLIGHT_PATH = RUN_DIR / "PREFLIGHT.json"
PUBLIC_PATH = ROOT / "runs/visual-utility-selector/2026-08-11/data/public_records.jsonl"
OUTPUT_PATH = RUN_DIR / "ARM2.json"
RAW_Q1_PATH = RUN_DIR / "raw/arm2_q1_screens.jsonl"
RAW_Q2_PATH = RUN_DIR / "raw/arm2_q2_rows.jsonl"

FORBIDDEN_OPEN_PARTS = (
    "annotation", "candidate_success", "private_label", "private_labels",
    "q3_q4", "target_bbox",
)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@contextmanager
def audited_open():
    original_builtin = builtins.open
    original_io = io.open
    opened = []

    def guard(file, *args, **kwargs):
        path = os.fspath(file)
        normalized = path.lower()
        if any(part in normalized for part in FORBIDDEN_OPEN_PARTS):
            raise PermissionError(f"DECOMP Arm 2 prohibited open: {path}")
        opened.append(path)
        return original_io(file, *args, **kwargs)

    builtins.open = guard
    io.open = guard
    try:
        yield opened
    finally:
        builtins.open = original_builtin
        io.open = original_io


def source_id(row):
    return row["image_path"].split("/images/", 1)[-1]


def summarize_counts(counts):
    values = np.asarray(list(counts.values()), dtype=np.int64)
    if not len(values):
        raise ValueError("DECOMP Arm 2 empty screen grouping")
    q1, median, q3 = np.quantile(values, [0.25, 0.5, 0.75], method="linear")
    singleton = int(np.sum(values == 1))
    rows = int(np.sum(values))
    return {
        "rows": rows,
        "screens": int(len(values)),
        "rows_per_screen_q1": float(q1),
        "rows_per_screen_median": float(median),
        "rows_per_screen_q3": float(q3),
        "singleton_screens": singleton,
        "singleton_screen_fraction": float(singleton / len(values)),
        "rows_on_singleton_screens": singleton,
        "rows_on_singleton_screen_fraction": float(singleton / rows),
    }


def equivalent(left, right, tolerance):
    if left["action"] != right["action"]:
        return False
    left_coordinate = left["coordinate"]
    right_coordinate = right["coordinate"]
    if left_coordinate is None or right_coordinate is None:
        return left_coordinate is None and right_coordinate is None and left["parameter"] == right["parameter"]
    return (
        abs(left_coordinate[0] - right_coordinate[0]) <= tolerance
        and abs(left_coordinate[1] - right_coordinate[1]) <= tolerance
    )


def complete_link_classes(candidates, tolerance):
    classes = []
    for index, candidate in enumerate(candidates):
        for members in classes:
            if all(equivalent(candidate, candidates[member], tolerance) for member in members):
                members.append(index)
                break
        else:
            classes.append([index])
    return classes


def select_mode(candidates, tolerance):
    parsed = [candidate for candidate in candidates if candidate["parse_ok"]]
    if not parsed:
        return {"coordinate": None, "mode_weight": 0, "representative_index": None}
    classes = complete_link_classes(parsed, tolerance)
    winner = max(classes, key=lambda members: (len(members), -min(parsed[index]["order"] for index in members)))
    representative = min(winner, key=lambda index: parsed[index]["order"])
    return {
        "coordinate": parsed[representative]["coordinate"],
        "mode_weight": len(winner),
        "representative_index": parsed[representative]["order"],
    }


def pixel_candidates(row, image_record):
    output = []
    for order, candidate in enumerate(row["candidates"]):
        coordinate = candidate.get("coordinate")
        if coordinate is not None:
            coordinate = [
                float(coordinate[0]) * image_record["width"],
                float(coordinate[1]) * image_record["height"],
            ]
        output.append({
            "action": str(candidate.get("action") or ""),
            "coordinate": coordinate,
            "parameter": str(candidate.get("parameter") or ""),
            "parse_ok": bool(candidate.get("parse_ok")),
            "order": order,
        })
    return output


def collide(left, right, tolerance):
    if left is None or right is None:
        return False
    return abs(left[0] - right[0]) <= tolerance and abs(left[1] - right[1]) <= tolerance


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def write_jsonl_fsynced(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())


def main():
    if OUTPUT_PATH.exists() or RAW_Q1_PATH.exists() or RAW_Q2_PATH.exists():
        raise FileExistsError("DECOMP Arm 2 output exists")
    with audited_open() as opened_paths:
        config = yaml.safe_load(CONFIG_PATH.read_text())
        preflight = json.loads(PREFLIGHT_PATH.read_text())
        if (
            config["arm2"]["evidence_status"] != "LABEL_FREE_PUBLIC_STRUCTURE"
            or config["arm2"]["labels_opened"] is not False
            or config["arm2"]["target_bbox_opened"] is not False
            or preflight["status"] != "PASS_DECOMP_PREFLIGHT_NO_ARM_STARTED"
            or preflight["labels_opened"] is not False
            or preflight["dependencies"]["public_records"]["sha256"] != sha256_file(PUBLIC_PATH)
        ):
            raise PermissionError("DECOMP Arm 2 authorization mismatch")
        rows = []
        with PUBLIC_PATH.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    if row.get("benchmark") == "screenspot_pro" and row.get("arm") == "C_uni":
                        rows.append(row)
        if len(rows) != 1581:
            raise ValueError("DECOMP Arm 2 public row mismatch")
        images = preflight["dataset_snapshot"]["images"]

        hash_counts = Counter(row["image_sha256"] for row in rows)
        source_counts = Counter(source_id(row) for row in rows)
        q1_rows = []
        for kind, counts in (("image_sha256", hash_counts), ("img_filename", source_counts)):
            q1_rows.extend({"grouping": kind, "screen_key": key, "rows": count} for key, count in sorted(counts.items()))

        q2_rows = []
        q2_summary = []
        by_screen = defaultdict(list)
        for row in rows:
            by_screen[row["image_sha256"]].append(row)
        for tolerance in config["arm2"]["tolerances_pixels"]:
            modes = {}
            for row in rows:
                modes[row["sample_key"]] = {
                    **select_mode(pixel_candidates(row, images[row["image_sha256"]]), float(tolerance)),
                    "image_sha256": row["image_sha256"],
                }
            collision_flags = {}
            for screen_rows in by_screen.values():
                for row in screen_rows:
                    key = row["sample_key"]
                    coordinate = modes[key]["coordinate"]
                    collision_flags[key] = any(
                        other["sample_key"] != key
                        and collide(coordinate, modes[other["sample_key"]]["coordinate"], float(tolerance))
                        for other in screen_rows
                    )
            collision_rows = sum(collision_flags.values())
            collision_screens = len({modes[key]["image_sha256"] for key, value in collision_flags.items() if value})
            q2_summary.append({
                "tolerance_pixels": float(tolerance),
                "rows": len(rows),
                "collision_rows": collision_rows,
                "collision_row_fraction": float(collision_rows / len(rows)),
                "screens": len(by_screen),
                "collision_screens": collision_screens,
                "collision_screen_fraction": float(collision_screens / len(by_screen)),
            })
            q2_rows.extend({
                "sample_key": key,
                "image_sha256": mode["image_sha256"],
                "tolerance_pixels": float(tolerance),
                "mode_weight": mode["mode_weight"],
                "representative_index": mode["representative_index"],
                "representative_coordinate_pixels": mode["coordinate"],
                "collision": collision_flags[key],
            } for key, mode in sorted(modes.items()))

        write_jsonl_fsynced(RAW_Q1_PATH, q1_rows)
        write_jsonl_fsynced(RAW_Q2_PATH, q2_rows)
        permitted_roots = (
            str(RUN_DIR), str(PUBLIC_PATH),
        )
        opened_unique = sorted(set(opened_paths))
        if any(any(part in path.lower() for part in FORBIDDEN_OPEN_PARTS) for path in opened_unique):
            raise PermissionError("DECOMP Arm 2 open audit detected prohibited path")
        output = {
            "schema_version": 1,
            "status": "PASS_DECOMP_ARM2_LABEL_FREE_COMPLETE_AWAITING_HUMAN_DECISION",
            "evidence_status": "LABEL_FREE_PUBLIC_STRUCTURE",
            "labels_opened": False,
            "target_bbox_opened": False,
            "evaluator_imported": False,
            "open_audit": {
                "enabled": True,
                "forbidden_open_count": 0,
                "opened_paths_sha256": hashlib.sha256("\n".join(opened_unique).encode()).hexdigest(),
                "opened_path_count": len(opened_unique),
            },
            "q1": {
                "image_sha256": summarize_counts(hash_counts),
                "img_filename": summarize_counts(source_counts),
                "partition_disagreement": bool(
                    sorted(hash_counts.values()) != sorted(source_counts.values())
                    or len(hash_counts) != len(source_counts)
                ),
            },
            "q2": q2_summary,
            "raw": {
                "q1": {"path": str(RAW_Q1_PATH.relative_to(ROOT)), "rows": len(q1_rows), "bytes": RAW_Q1_PATH.stat().st_size, "sha256": sha256_file(RAW_Q1_PATH)},
                "q2": {"path": str(RAW_Q2_PATH.relative_to(ROOT)), "rows": len(q2_rows), "bytes": RAW_Q2_PATH.stat().st_size, "sha256": sha256_file(RAW_Q2_PATH)},
                "write_flush_fsync_per_row": True,
            },
        }
        atomic_json(OUTPUT_PATH, output)
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()