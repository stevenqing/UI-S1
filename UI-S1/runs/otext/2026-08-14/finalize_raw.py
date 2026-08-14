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
    engines = {}; row_sets = {}
    for engine, shards in (("easyocr", 96), ("rapidocr", 12)):
        artifacts = {}; row_ids = []; boxes = errors = 0
        for shard in range(shards):
            path = RAW_ROOT / engine / f"shard-{shard}.jsonl"
            manifest_path = path.with_suffix(".manifest.json")
            manifest = json.loads(manifest_path.read_text())
            if manifest.get("status") != "PASS_OTEXT_OCR_LANE_COMPLETE" or manifest["bytes"] != path.stat().st_size or manifest["sha256"] != sha256_file(path):
                raise ValueError(f"OTEXT raw lane mismatch: {engine}/{shard}")
            artifacts[path.relative_to(RUN_DIR).as_posix()] = {"bytes": path.stat().st_size, "sha256": sha256_file(path), "rows": manifest["rows"]}
            artifacts[manifest_path.relative_to(RUN_DIR).as_posix()] = {"bytes": manifest_path.stat().st_size, "sha256": sha256_file(manifest_path)}
            for line in path.read_text().splitlines():
                if line.strip():
                    row = json.loads(line); row_ids.append(row["row_id"]); boxes += len(row["boxes"]); errors += row["status"] != "PASS"
        if len(row_ids) != len(set(row_ids)) or len(row_ids) != 1581:
            raise ValueError(f"OTEXT raw coverage mismatch: {engine}")
        row_sets[engine] = set(row_ids)
        engines[engine] = {"rows": 1581, "boxes": boxes, "errors": errors, "shards": shards, "total_bytes": sum(value["bytes"] for value in artifacts.values()), "artifacts": artifacts}
    if row_sets["easyocr"] != row_sets["rapidocr"]:
        raise ValueError("OTEXT engine row sets differ")
    result = {"schema_version": 1, "status": "LOCKED_OTEXT_RAW_OCR_COMPLETE", "engines": engines, "common_rows": 1581, "raw_regenerated_independently_from_ORTH": True}
    OUTPUT_PATH.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({engine: {key: value[key] for key in ("rows", "boxes", "errors", "total_bytes")} for engine, value in engines.items()}, indent=2))


if __name__ == "__main__":
    main()