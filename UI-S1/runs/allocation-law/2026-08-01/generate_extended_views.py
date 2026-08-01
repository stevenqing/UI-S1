import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor


ROOT = Path(__file__).resolve().parents[3]
H3_DIR = ROOT / "runs/ccm-h2h/2026-07-31/h3"
DATA_ROOT = ROOT / "runs/collision-law/2026-07-30/w3_assets/ScreenSpot-Pro"
sys.path.insert(0, str(H3_DIR))
from generate_fixed_regions import process_qwen3_subimage, process_uitars_subimage


def completed_ids(path):
    if not path.exists(): return set()
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    ids = [row["id"] for row in rows]
    if len(ids) != len(set(ids)): raise ValueError("duplicate extended-view ids")
    return set(ids)


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--regions", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--model-type", choices=("qwen3", "uitars"), required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--view-start", type=int, default=4)
    parser.add_argument("--view-stop", type=int, default=12)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-shards", type=int, default=4)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not 0 <= args.view_start < args.view_stop <= 12:
        raise ValueError("invalid view range")
    rows = [json.loads(line) for line in args.regions.read_text().splitlines() if line.strip()]
    if len(rows) != 1581 or any(len(row["regions"]) != 12 for row in rows):
        raise ValueError("extended views require complete 12-region manifest")
    indices = list(range(args.shard_index, len(rows), args.num_shards))
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("limit must be positive")
        indices = indices[:args.limit]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists() and not args.resume: raise FileExistsError(args.output)
    completed = completed_ids(args.output) if args.resume else set()

    if args.model_type == "qwen3":
        from transformers import Qwen3VLForConditionalGeneration
        model_class = Qwen3VLForConditionalGeneration
        process = process_qwen3_subimage
        model = model_class.from_pretrained(args.model_dir, torch_dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda:0").eval()
        processor = AutoProcessor.from_pretrained(args.model_dir, min_pixels=3136, max_pixels=4096 * 2160, use_fast=False)
    else:
        from transformers import Qwen2VLForConditionalGeneration
        model_class = Qwen2VLForConditionalGeneration
        process = process_uitars_subimage
        model = model_class.from_pretrained(args.model_dir, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2", device_map="cuda:0").eval()
        processor = AutoProcessor.from_pretrained(
            args.model_dir, min_pixels=3136, max_pixels=4096 * 2160, use_fast=False,
            size={"shortest_edge": 3136, "longest_edge": 2116800},
        )
    index_hash = sha256_file(args.model_dir / "model.safetensors.index.json")
    written = 0
    with args.output.open("a", buffering=1) as output:
        for stable_index in indices:
            source = rows[stable_index]
            if source["id"] in completed: continue
            image = Image.open(DATA_ROOT / "images" / source["img_filename"]).convert("RGB")
            predictions = []
            for view_index in range(args.view_start, args.view_stop):
                left, top, right, bottom = source["regions"][view_index]
                crop = image.crop((left, top, right, bottom))
                point, response = process(
                    crop, source["instruction"], processor, model, 0,
                    offset_x=left, offset_y=top, resize=True,
                )
                predictions.append({
                    "view_index": view_index,
                    "region": [left, top, right, bottom],
                    "point": list(map(float, point)),
                    "response": response,
                })
            artifact = {
                "stable_index": stable_index,
                "id": source["id"],
                "application": source["application"],
                "img_filename": source["img_filename"],
                "img_size": source["img_size"],
                "instruction": source["instruction"],
                "model_id": args.model_id,
                "model_revision": args.model_revision,
                "model_index_sha256": index_hash,
                "shared_region_candidate_sha256": source["shared_region_candidate_sha256"],
                "view_start": args.view_start,
                "view_stop": args.view_stop,
                "predictions": predictions,
                "shard_index": args.shard_index,
                "num_shards": args.num_shards,
            }
            artifact["prediction_sha256"] = hashlib.sha256(
                json.dumps(predictions, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()
            output.write(json.dumps(artifact, ensure_ascii=True) + "\n")
            output.flush(); os.fsync(output.fileno())
            written += 1
            if written % 25 == 0:
                print(json.dumps({
                    "model": args.model_id,
                    "shard": args.shard_index,
                    "written_this_run": written,
                    "total_assigned": len(indices),
                }), flush=True)

    print(json.dumps({
        "status": "PASS",
        "model": args.model_id,
        "shard": args.shard_index,
        "written_this_run": written,
        "completed": len(completed_ids(args.output)),
        "total_assigned": len(indices),
    }), flush=True)


if __name__ == "__main__":
    main()
