import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
from PIL import Image


RUN_DIR = Path(__file__).resolve().parent
ROOT = RUN_DIR.parents[2]
PREFLIGHT_PATH = RUN_DIR / "PREFLIGHT.json"


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_polygon(value):
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (4, 2) or not np.isfinite(array).all():
        raise ValueError("ORTH OCR invalid polygon")
    return [[float(coordinate) for coordinate in point] for point in array]


def easyocr_engine(preflight):
    import torch
    import easyocr
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    model_path = Path(preflight["ocr_engines"]["model_assets"]["easyocr_craft"]["path"]).parent
    reader = easyocr.Reader(
        ["en"], gpu=False, model_storage_directory=str(model_path),
        download_enabled=False, verbose=False, quantize=True,
    )
    parameters = preflight["ocr_engines"]["easyocr"]["call"]
    def infer(image):
        values = reader.readtext(np.asarray(image), **parameters)
        return [
            {
                "polygon": normalize_polygon(polygon),
                "text": str(text),
                "confidence": float(confidence),
                "orientation": None,
                "engine_order": index,
            }
            for index, (polygon, text, confidence) in enumerate(values)
        ], None
    return infer


def rapidocr_engine(preflight):
    from rapidocr_onnxruntime import RapidOCR
    config_path = preflight["ocr_engines"]["model_assets"]["rapidocr_config"]["path"]
    engine = RapidOCR(
        config_path=config_path,
        intra_op_num_threads=1,
        inter_op_num_threads=1,
    )
    def infer(image):
        bgr = np.asarray(image)[:, :, ::-1].copy()
        values, elapsed = engine(bgr, use_det=True, use_cls=True, use_rec=True)
        output = []
        for index, value in enumerate(values or []):
            polygon, text, confidence = value[:3]
            output.append({
                "polygon": normalize_polygon(polygon),
                "text": str(text),
                "confidence": float(confidence),
                "orientation": None,
                "engine_order": index,
            })
        return output, elapsed
    return infer


def write_record(handle, row):
    handle.write(json.dumps(row, sort_keys=True) + "\n")
    handle.flush()
    os.fsync(handle.fileno())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", choices=("easyocr", "rapidocr"), required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    expected_shards = 48 if args.engine == "easyocr" else 12
    if args.num_shards != expected_shards or not 0 <= args.shard_index < args.num_shards:
        raise ValueError(f"ORTH OCR requires {expected_shards} shards for {args.engine}")
    if args.output.exists() or args.output.with_suffix(".manifest.json").exists():
        raise FileExistsError(args.output)
    preflight = json.loads(PREFLIGHT_PATH.read_text())
    if preflight.get("status") != "PASS_ORTH_PREFLIGHT_CPU_OCR_READY" or preflight.get("ocr_forward_started") is not False:
        raise PermissionError("ORTH OCR preflight boundary mismatch")
    split = json.loads((ROOT / preflight["inputs"]["split_preflight"]["path"]).read_text())
    row_ids = sorted(split["images"])
    indices = list(range(args.shard_index, len(row_ids), args.num_shards))
    if args.limit is not None:
        indices = indices[:args.limit]
    infer = easyocr_engine(preflight) if args.engine == "easyocr" else rapidocr_engine(preflight)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    written = 0
    with temporary.open("w", buffering=1) as handle:
        for stable_index in indices:
            row_id = row_ids[stable_index]
            image_record = split["images"][row_id]
            image_path = ROOT / image_record["path"]
            started = time.perf_counter()
            try:
                image = Image.open(image_path).convert("RGB")
                boxes, engine_elapsed = infer(image)
                record = {
                    "schema_version": 1,
                    "status": "PASS",
                    "engine": args.engine,
                    "row_id": row_id,
                    "stable_index": stable_index,
                    "shard_index": args.shard_index,
                    "num_shards": args.num_shards,
                    "image_path": image_record["path"],
                    "image_sha256": image_record["sha256"],
                    "image_size": list(image.size),
                    "boxes": boxes,
                    "engine_elapsed": engine_elapsed,
                    "wall_seconds": time.perf_counter() - started,
                    "error": None,
                }
            except Exception as error:
                record = {
                    "schema_version": 1,
                    "status": "ERROR",
                    "engine": args.engine,
                    "row_id": row_id,
                    "stable_index": stable_index,
                    "shard_index": args.shard_index,
                    "num_shards": args.num_shards,
                    "image_path": image_record["path"],
                    "image_sha256": image_record["sha256"],
                    "image_size": image_record["declared_size"],
                    "boxes": [],
                    "engine_elapsed": None,
                    "wall_seconds": time.perf_counter() - started,
                    "error": {"type": type(error).__name__, "message": str(error)},
                }
            write_record(handle, record)
            written += 1
    temporary.replace(args.output)
    manifest = {
        "schema_version": 1,
        "status": "PASS_ORTH_OCR_LANE_COMPLETE",
        "engine": args.engine,
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "rows": written,
        "output": str(args.output),
        "bytes": args.output.stat().st_size,
        "sha256": sha256_file(args.output),
        "preflight_sha256": sha256_file(PREFLIGHT_PATH),
        "row_write_flush_fsync": True,
    }
    manifest_path = args.output.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()