import hashlib
import json
from pathlib import Path


RUN_DIR = Path(__file__).resolve().parent
RAW_ROOT = RUN_DIR / "raw"
OUTPUT_PATH = RUN_DIR / "OCR_RAW_MANIFEST.json"


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    if OUTPUT_PATH.exists():
        raise FileExistsError(OUTPUT_PATH)
    engines = {}
    all_ids = {}
    for engine in ("easyocr", "rapidocr"):
        root = RAW_ROOT / engine
        artifacts = {}
        row_ids = []
        boxes = 0
        errors = 0
        for shard in range(12):
            path = root / f"shard-{shard}.jsonl"
            manifest_path = path.with_suffix(".manifest.json")
            manifest = json.loads(manifest_path.read_text())
            if (
                manifest.get("status") != "PASS_ORTH_OCR_LANE_COMPLETE"
                or manifest.get("rows") != sum(bool(line.strip()) for line in path.read_text().splitlines())
                or manifest.get("bytes") != path.stat().st_size
                or manifest.get("sha256") != sha256_file(path)
            ):
                raise ValueError(f"ORTH OCR lane mismatch: {engine}/{shard}")
            artifacts[path.relative_to(RUN_DIR).as_posix()] = {
                "bytes": path.stat().st_size, "sha256": sha256_file(path),
            }
            artifacts[manifest_path.relative_to(RUN_DIR).as_posix()] = {
                "bytes": manifest_path.stat().st_size, "sha256": sha256_file(manifest_path),
            }
            for line in path.read_text().splitlines():
                if line.strip():
                    row = json.loads(line); row_ids.append(row["row_id"])
                    boxes += len(row["boxes"]); errors += row["status"] != "PASS"
        if len(row_ids) != len(set(row_ids)) != 1581:
            raise ValueError(f"ORTH OCR row coverage mismatch: {engine}")
        if len(row_ids) != 1581:
            raise ValueError(f"ORTH OCR expected 1581 rows: {engine}")
        all_ids[engine] = set(row_ids)
        engines[engine] = {
            "rows": len(row_ids), "boxes": boxes, "errors": errors,
            "artifacts": artifacts,
            "total_bytes": sum(value["bytes"] for value in artifacts.values()),
        }
    if all_ids["easyocr"] != all_ids["rapidocr"]:
        raise ValueError("ORTH OCR engine row sets differ")
    result = {
        "schema_version": 1,
        "status": "LOCKED_ORTH_RAW_OCR_COMPLETE",
        "engines": engines,
        "common_rows": len(all_ids["easyocr"]),
        "raw_contains_labels_or_instruction": False,
    }
    OUTPUT_PATH.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({engine: {key: value[key] for key in ("rows", "boxes", "errors", "total_bytes")} for engine, value in engines.items()}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()