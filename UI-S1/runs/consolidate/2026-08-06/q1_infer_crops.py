import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor


RUN_DIR = Path(__file__).resolve().parent
ROOT = RUN_DIR.parents[2]
DATA_ROOT = ROOT / "runs/collision-law/2026-07-30/w3_assets/ScreenSpot-Pro"


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def completed_ids(path):
    if not path.exists():
        return set()
    ids = [json.loads(line)["id"] for line in path.read_text().splitlines() if line.strip()]
    if len(ids) != len(set(ids)):
        raise ValueError("Q1 duplicate resumed ids")
    return set(ids)


def load_model(args):
    if args.model_type == "gta1":
        mvp_root = ROOT / "runs/collision-law/2026-07-30/w3_assets/MVP"
        sys.path.insert(0, str(mvp_root))
        from mvp_sspro import Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLProcessor, process_single_subimage
        config = Qwen2_5_VLConfig.from_pretrained(args.model_dir)
        config.target_token_id = ","
        config.target_layer_idx = 20
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            args.model_dir, config=config, torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2", device_map="cuda:0",
        ).eval()
        processor = Qwen2_5_VLProcessor.from_pretrained(args.model_dir, min_pixels=3136, max_pixels=4096 * 2160)
        return model, processor, process_single_subimage
    h3_dir = ROOT / "runs/ccm-h2h/2026-07-31/h3"
    sys.path.insert(0, str(h3_dir))
    from generate_fixed_regions import process_qwen3_subimage, process_uitars_subimage
    if args.model_type == "qwen3":
        from transformers import Qwen3VLForConditionalGeneration
        model = Qwen3VLForConditionalGeneration.from_pretrained(args.model_dir, torch_dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda:0").eval()
        processor = AutoProcessor.from_pretrained(args.model_dir, min_pixels=3136, max_pixels=4096 * 2160, use_fast=False)
        return model, processor, process_qwen3_subimage
    from transformers import Qwen2VLForConditionalGeneration
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        args.model_dir, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2", device_map="cuda:0",
    ).eval()
    processor = AutoProcessor.from_pretrained(
        args.model_dir, min_pixels=3136, max_pixels=4096 * 2160, use_fast=False,
        size={"shortest_edge": 3136, "longest_edge": 2116800},
    )
    return model, processor, process_uitars_subimage


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--regions", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--model-type", choices=("gta1", "qwen3", "uitars"), required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-shards", type=int, default=8)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.regions.read_text().splitlines() if line.strip()]
    if len(rows) != 1581 or any(set(row["arms"]) != {"C_cond", "C_rand", "C_self"} for row in rows):
        raise ValueError("Q1 inference requires complete prepared regions")
    indices = list(range(args.shard_index, len(rows), args.num_shards))
    if args.limit is not None:
        indices = indices[:args.limit]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists() and not args.resume:
        raise FileExistsError(args.output)
    completed = completed_ids(args.output) if args.resume else set()
    model, processor, process = load_model(args)
    index_hash = sha256_file(args.model_dir / "model.safetensors.index.json")
    written = 0
    with args.output.open("a", buffering=1) as output:
        for stable_index in indices:
            source = rows[stable_index]
            if source["id"] in completed:
                continue
            image = Image.open(DATA_ROOT / "images" / source["img_filename"]).convert("RGB")
            predictions = {}
            for arm in ("C_cond", "C_rand", "C_self"):
                predictions[arm] = []
                for crop_index, region in enumerate(source["arms"][arm]):
                    left, top, right, bottom = region
                    crop = image.crop(region)
                    point, response = process(crop, source["instruction"], processor, model, 0, offset_x=left, offset_y=top, resize=True)
                    predictions[arm].append({
                        "crop_index": crop_index, "region": region,
                        "point": list(map(float, point)), "response": response,
                    })
            artifact = {
                "stable_index": stable_index, "id": source["id"], "model_id": args.model_id,
                "model_revision": args.model_revision, "model_index_sha256": index_hash,
                "arms_sha256": source["arms_sha256"], "predictions": predictions,
                "shard_index": args.shard_index, "num_shards": args.num_shards,
            }
            artifact["predictions_sha256"] = hashlib.sha256(json.dumps(predictions, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
            output.write(json.dumps(artifact, ensure_ascii=True) + "\n")
            output.flush(); os.fsync(output.fileno())
            written += 1
            if written % 10 == 0:
                print(json.dumps({"model": args.model_id, "shard": args.shard_index, "written": written}), flush=True)
    print(json.dumps({"status": "PASS", "model": args.model_id, "shard": args.shard_index, "written": written, "completed": len(completed_ids(args.output))}), flush=True)


if __name__ == "__main__":
    main()