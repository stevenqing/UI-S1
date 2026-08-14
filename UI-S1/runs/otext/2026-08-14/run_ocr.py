import argparse
import hashlib
import importlib.util
import json
import os
import time
from pathlib import Path

import numpy as np
from PIL import Image


RUN_DIR = Path(__file__).resolve().parent
ROOT = RUN_DIR.parents[2]
PREFLIGHT_PATH = RUN_DIR / "PREFLIGHT.json"
ORTH_PREFLIGHT_PATH = ROOT / "runs/orth/2026-08-14/PREFLIGHT.json"
ORTH_WRITER_PATH = ROOT / "runs/orth/2026-08-14/run_ocr.py"
ORTH_PREFLIGHT_SHA256 = "8b32da564fe5ebf1bd7d5127d86064bd40a4e8a0ebed240c662c5004edd50358"


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_orth_writer():
    spec = importlib.util.spec_from_file_location("otext_orth_ocr_writer", ORTH_WRITER_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(ORTH_WRITER_PATH)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", choices=("easyocr", "rapidocr"), required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    expected_shards = 96 if args.engine == "easyocr" else 48
    if args.num_shards != expected_shards or not 0 <= args.shard_index < expected_shards:
        raise ValueError(f"OTEXT requires {expected_shards} shards for {args.engine}")
    manifest_path = args.output.with_suffix(".manifest.json")
    if args.output.exists() or manifest_path.exists():
        raise FileExistsError(args.output)
    preflight = json.loads(PREFLIGHT_PATH.read_text())
    if preflight.get("status") != "PASS_OTEXT_PREFLIGHT_POST_SELECTION_VALIDATION" or preflight.get("ocr_forward_started") is not False:
        raise PermissionError("OTEXT OCR preflight mismatch")
    if sha256_file(ORTH_PREFLIGHT_PATH) != ORTH_PREFLIGHT_SHA256:
        raise PermissionError("OTEXT locked OCR parameter source drift")
    orth_preflight = json.loads(ORTH_PREFLIGHT_PATH.read_text())
    orth_writer = load_orth_writer()
    infer = orth_writer.easyocr_engine(orth_preflight) if args.engine == "easyocr" else orth_writer.rapidocr_engine(orth_preflight)
    split = json.loads((ROOT / preflight["upstream"]["split_preflight"]["path"]).read_text())
    row_ids = sorted(split["images"])
    indices = range(args.shard_index, len(row_ids), args.num_shards)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    written = 0
    with temporary.open("w", buffering=1) as handle:
        for stable_index in indices:
            row_id = row_ids[stable_index]; image_record = split["images"][row_id]
            started = time.perf_counter()
            try:
                image = Image.open(ROOT / image_record["path"]).convert("RGB")
                boxes, engine_elapsed = infer(image)
                status, error = "PASS", None
            except Exception as exception:
                image = None; boxes = []; engine_elapsed = None; status = "ERROR"
                error = {"type": type(exception).__name__, "message": str(exception)}
            record = {
                "schema_version": 1, "status": status, "engine": args.engine,
                "row_id": row_id, "stable_index": stable_index,
                "shard_index": args.shard_index, "num_shards": args.num_shards,
                "image_path": image_record["path"], "image_sha256": image_record["sha256"],
                "image_size": list(image.size) if image else image_record["declared_size"],
                "boxes": boxes, "engine_elapsed": engine_elapsed,
                "wall_seconds": time.perf_counter() - started, "error": error,
            }
            handle.write(json.dumps(record, sort_keys=True) + "\n"); handle.flush(); os.fsync(handle.fileno()); written += 1
    temporary.replace(args.output)
    manifest = {
        "schema_version": 1, "status": "PASS_OTEXT_OCR_LANE_COMPLETE",
        "engine": args.engine, "shard_index": args.shard_index,
        "num_shards": args.num_shards, "rows": written,
        "output": str(args.output), "bytes": args.output.stat().st_size,
        "sha256": sha256_file(args.output),
        "otext_preflight_sha256": sha256_file(PREFLIGHT_PATH),
        "orth_parameter_preflight_sha256": ORTH_PREFLIGHT_SHA256,
        "row_write_flush_fsync": True,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()