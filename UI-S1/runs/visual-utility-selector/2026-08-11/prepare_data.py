import argparse
import hashlib
import json
import sys
from pathlib import Path

from PIL import Image


RUN_DIR = Path(__file__).resolve().parent
ROOT = RUN_DIR.parents[2]
UTILITY_DIR = ROOT / "runs/lsa-utility/2026-08-11"
LSA_DIR = ROOT / "runs/lsa/2026-08-10"
SCREEN_REGIONS = ROOT / "runs/allocation-law/2026-08-01/raw/shared_regions_n12.jsonl"
SCREEN_IMAGES = ROOT / "runs/collision-law/2026-07-30/w3_assets/ScreenSpot-Pro/images"
MIND_ROWS = ROOT / "runs/xfer/2026-08-07/data/mind2web/mind2web_test_task.jsonl"
sys.path.insert(0, str(UTILITY_DIR))
sys.path.insert(0, str(LSA_DIR))

from utility_common import ARMS, BENCHMARKS, load_banks
from vus_data import audit_public_record, sha256_file


def load_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def compact_history(row, limit):
    output = []
    for step in row.get("step_history", [])[-limit:]:
        operation = step.get("operation") or {}
        action = str(operation.get("op") or operation.get("original_op") or "UNKNOWN")
        parameter = str(operation.get("value") or "")
        output.append(f"{action} {parameter}".strip())
    return output


def normalized_coordinate(candidate, benchmark, width, height):
    coordinate = candidate.baseline_coordinate
    if coordinate is None:
        return None
    if benchmark == "mind2web":
        x, y = coordinate
    else:
        x, y = coordinate[0] / width, coordinate[1] / height
    if not all(__import__("math").isfinite(float(value)) for value in (x, y)):
        raise ValueError(f"non-finite candidate coordinate: {(x, y)}")
    return [float(x), float(y)]


def public_candidate(candidate, benchmark, width, height):
    return {
        "action": candidate.action,
        "coordinate": normalized_coordinate(candidate, benchmark, width, height),
        "parameter": candidate.parameter[:256],
        "parse_ok": candidate.parse_ok,
    }


def hash_records(records):
    digest = hashlib.sha256()
    for record in records:
        digest.update(json.dumps(record, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def build_records(limit=None):
    banks = load_banks()
    mind = {row["id"]: row for row in load_jsonl(MIND_ROWS)}
    screen = {row["id"]: row for row in load_jsonl(SCREEN_REGIONS)}
    public = []
    private = []
    for benchmark in BENCHMARKS:
        row_ids = sorted(banks["C_uni"][benchmark])
        if limit is not None:
            row_ids = row_ids[:limit]
        for row_id in row_ids:
            if benchmark == "mind2web":
                source = mind[row_id]
                image_path = ROOT / source["image"]
                instruction = source["task"]
                history = compact_history(source, 4)
            else:
                source = screen[row_id]
                image_path = SCREEN_IMAGES / source["img_filename"]
                instruction = source["instruction"]
                history = []
            if not image_path.is_file():
                raise FileNotFoundError(image_path)
            width, height = source["img_size"] if benchmark == "screenspot_pro" else Image.open(image_path).size
            image_sha256 = sha256_file(image_path)
            for arm in ARMS:
                row = banks[arm][benchmark][row_id]
                sample_key = f"{benchmark}/{arm}/{row_id}"
                record = {
                    "schema_version": 1,
                    "sample_key": sample_key,
                    "benchmark": benchmark,
                    "arm": arm,
                    "row_id": row_id,
                    "fold": row.fold,
                    "group": row.group,
                    "image_path": str(image_path),
                    "image_sha256": image_sha256,
                    "instruction": instruction,
                    "history": history,
                    "candidates": [
                        public_candidate(candidate, benchmark, width, height)
                        for candidate in row.candidates
                    ],
                }
                audit_public_record(record)
                public.append(record)
                private.append({
                    "schema_version": 1,
                    "sample_key": sample_key,
                    "candidate_success": [bool(candidate.success) for candidate in row.candidates],
                })
    return public, private


def write_jsonl(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True, sort_keys=True) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=RUN_DIR / "data")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        raise ValueError("limit must be positive")
    public, private = build_records(args.limit)
    public_path = args.output_dir / "public_records.jsonl"
    private_path = args.output_dir / "private_labels.jsonl"
    write_jsonl(public_path, public)
    write_jsonl(private_path, private)
    manifest = {
        "schema_version": 1,
        "status": "PASS",
        "limited_rows_per_benchmark": args.limit,
        "public_records": len(public),
        "private_records": len(private),
        "public_sha256": sha256_file(public_path),
        "private_sha256": sha256_file(private_path),
        "canonical_public_sha256": hash_records(public),
        "canonical_private_sha256": hash_records(private),
        "public_forbidden_fields": "PASS",
        "candidate_count": 12,
    }
    (args.output_dir / "MANIFEST.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
