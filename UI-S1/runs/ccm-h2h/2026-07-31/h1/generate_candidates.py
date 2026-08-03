import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import torch

from generation_contract import generation_row


RUN_DIR = Path(__file__).resolve().parents[1]
ROOT = RUN_DIR.parents[2]
UPSTREAM = ROOT / "runs/collision-law/2026-07-30"
MVP_ROOT = UPSTREAM / "w3_assets/MVP"
MODEL_ROOT = UPSTREAM / "w3_assets/GTA1-7B"
DATA_ROOT = UPSTREAM / "w3_assets/ScreenSpot-Pro"
SOURCE_REVISION = "988ff3c61b9f7632d780ae27c83260de75b3c95f"
MODEL_REVISION = "701bedc80b447863bd60e3318ae44f6cbbfafd78"

sys.path.insert(0, str(MVP_ROOT.resolve()))
from mvp_sspro import (
    Qwen2_5_VLConfig,
    Qwen2_5_VLForConditionalGeneration,
    Qwen2_5_VLProcessor,
    process_single_image,
)


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def samples():
    rows = []
    for path in sorted((DATA_ROOT / "annotations").glob("*.json")):
        for row in json.loads(path.read_text()):
            rows.append({"annotation_file": path.name, **row})
    rows.sort(key=lambda row: row["id"])
    if len(rows) != 1581 or len({row["id"] for row in rows}) != 1581:
        raise ValueError("ScreenSpot-Pro identity coverage mismatch")
    return rows


def completed_ids(path):
    if not path.exists():
        return set()
    ids = [json.loads(line)["id"] for line in path.read_text().splitlines() if line.strip()]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate resumed H1 ids")
    return set(ids)


def normalize_result(row, result, max_subimages, shard_index, num_shards):
    predictions = result["all_predictions"]
    if len(predictions) < 10 or len(predictions) > max_subimages + 1:
        raise ValueError(f"candidate count outside [10,{max_subimages + 1}]: {row['id']} {len(predictions)}")
    normalized = {
        "id": row["id"],
        "annotation_file": row["annotation_file"],
        "application": row["application"],
        "platform": row["platform"],
        "ui_type": row["ui_type"],
        "group": row["group"],
        "img_filename": row["img_filename"],
        "img_size": row["img_size"],
        "instruction": row["instruction"],
        "target_bbox": row["bbox"],
        "candidates": [
            {
                "candidate_index": index,
                "point": list(map(float, prediction["point"])),
                "coverage": float(prediction["coverage"]) if not isinstance(prediction["coverage"], str) else 0.0,
                "region": list(map(int, prediction["region"])),
                "stage": prediction["stage"],
                "output": prediction["output"],
            }
            for index, prediction in enumerate(predictions)
        ],
        "candidate_count": len(predictions),
        "requested_candidate_count": max_subimages + 1,
        "official_source_revision": SOURCE_REVISION,
        "model_revision": MODEL_REVISION,
        "attention_layer": 20,
        "target_token": ",",
        "max_subimages": max_subimages,
        "shard_index": shard_index,
        "num_shards": num_shards,
    }
    candidate_payload = json.dumps(normalized["candidates"], sort_keys=True, separators=(",", ":"))
    normalized["candidate_sha256"] = hashlib.sha256(candidate_payload.encode()).hexdigest()
    return normalized


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-shards", type=int, default=8)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--max-subimages", type=int, choices=(9, 18), required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.shard_index < 0 or args.shard_index >= args.num_shards:
        raise ValueError("invalid shard index")
    rows = [row for index, row in enumerate(samples()) if index % args.num_shards == args.shard_index]
    if args.limit is not None:
        rows = rows[:args.limit]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists() and not args.resume:
        raise FileExistsError(args.output)
    done = completed_ids(args.output) if args.resume else set()

    config = Qwen2_5_VLConfig.from_pretrained(MODEL_ROOT)
    config.target_token_id = ","
    config.target_layer_idx = 20
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ROOT,
        config=config,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="cuda:0",
    ).eval()
    processor = Qwen2_5_VLProcessor.from_pretrained(
        MODEL_ROOT, min_pixels=3136, max_pixels=4096 * 2160,
    )
    with args.output.open("a", buffering=1) as output:
        for row in rows:
            if row["id"] in done:
                continue
            result = process_single_image(
                generation_row(row), model, processor, str((DATA_ROOT / "images").resolve()), 0,
                max_inferences=args.max_subimages,
            )
            if result is None:
                raise RuntimeError(f"official MVP generation failed: {row['id']}")
            output.write(json.dumps(
                normalize_result(row, result, args.max_subimages, args.shard_index, args.num_shards),
                ensure_ascii=True,
            ) + "\n")
            output.flush()
            os.fsync(output.fileno())


if __name__ == "__main__":
    main()
