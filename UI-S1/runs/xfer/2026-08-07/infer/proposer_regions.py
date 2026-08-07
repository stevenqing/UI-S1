import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import torch
import yaml
from PIL import Image


RUN_DIR = Path(__file__).resolve().parents[1]
ROOT = RUN_DIR.parents[2]
MVP_ROOT = ROOT / "runs/collision-law/2026-07-30/w3_assets/MVP"
sys.path.insert(0, str(MVP_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from mvp_sspro import Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLProcessor
from proposer_ablation import generate_multilayer, prompt_text, sha256_file


def completed_ids(path):
    if not path.exists():
        return set()
    ids = [json.loads(line)["id"] for line in path.read_text().splitlines() if line.strip()]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate proposer-region ids")
    return set(ids)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-shards", type=int, default=2)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("invalid shard index")
    roster = yaml.safe_load((RUN_DIR / "configs/xfer_roster.yaml").read_text())
    proposer = roster["mind2web"]["proposer"]
    if proposer["selection_status"] != "FROZEN_AFTER_DEV_ABLATION":
        raise ValueError("Mind2Web proposer layer is not frozen")
    layer = proposer["selected_layer"]
    model_spec = next(model for model in roster["mind2web"]["models"] if model["id"] == proposer["model"])
    model_dir = ROOT / model_spec["local_path"]
    rows = [json.loads(line) for line in (RUN_DIR / "data/mind2web/mind2web_test_task.jsonl").read_text().splitlines() if line.strip()]
    indices = list(range(args.shard_index, len(rows), args.num_shards))
    if args.limit is not None:
        indices = indices[:args.limit]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists() and not args.resume:
        raise FileExistsError(args.output)
    completed = completed_ids(args.output) if args.resume else set()
    config = Qwen2_5_VLConfig.from_pretrained(model_dir)
    config.target_token_id = ","
    config.target_layer_idx = layer
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_dir, config=config, torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2", device_map="cuda:0",
    ).eval()
    processor = Qwen2_5_VLProcessor.from_pretrained(
        model_dir, min_pixels=256 * 28 * 28, max_pixels=1344 * 28 * 28,
    )
    index_hash = sha256_file(model_dir / "model.safetensors.index.json")
    with args.output.open("a", buffering=1) as output:
        for index in indices:
            row = rows[index]
            if row["id"] in completed:
                continue
            image = Image.open(ROOT / row["image"]).convert("RGB")
            response, resized_size, layers = generate_multilayer(
                image, prompt_text(roster, row), processor, model, [layer], proposer
            )
            regions = layers[str(layer)]
            if len(regions) != proposer["attention"]["max_regions"]:
                raise ValueError(f"incomplete proposer regions: {row['id']}")
            artifact = {
                "stable_index": index,
                "id": row["id"],
                "image_sha256": row["image_sha256"],
                "model_id": model_spec["id"],
                "model_revision": model_spec["revision"],
                "model_index_sha256": index_hash,
                "selected_layer": layer,
                "selected_query_token": proposer["selected_query_token"],
                "resized_size": resized_size,
                "response": response,
                "regions": regions,
                "regions_sha256": hashlib.sha256(json.dumps(regions, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
                "shard_index": args.shard_index,
                "num_shards": args.num_shards,
            }
            if any("target" in key or "bbox" in key or key == "step" for key in artifact):
                raise ValueError("proposer region target leak")
            output.write(json.dumps(artifact, ensure_ascii=True) + "\n")
            output.flush()
            os.fsync(output.fileno())
    print(json.dumps({
        "status": "PASS", "shard": args.shard_index,
        "completed": len(completed_ids(args.output)),
    }))


if __name__ == "__main__":
    main()