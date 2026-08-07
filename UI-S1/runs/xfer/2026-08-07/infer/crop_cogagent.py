import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import torch
import yaml
from PIL import Image
from transformers import AutoModelForCausalLM, LlamaTokenizer


RUN_DIR = Path(__file__).resolve().parents[1]
ROOT = RUN_DIR.parents[2]
sys.path.insert(0, str(RUN_DIR))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from crop_qwen import completed_ids, load_sources, regions_for, remap_prediction, source_hash
from stage1_cogagent import prompt_text, sha256_file
from xfer_common import MIND2WEB_ACTIONS, parse_cogagent_response, parse_product_response


def infer(image, prompt, tokenizer, model):
    conversation = model.build_conversation_input_ids(
        tokenizer, query=prompt, history=[], images=[image], template_version="chat"
    )
    inputs = {
        "input_ids": conversation["input_ids"].unsqueeze(0).to("cuda:0"),
        "token_type_ids": conversation["token_type_ids"].unsqueeze(0).to("cuda:0"),
        "attention_mask": conversation["attention_mask"].unsqueeze(0).to("cuda:0"),
        "images": [[conversation["images"][0].to("cuda:0", dtype=torch.bfloat16)]],
        "cross_images": [[conversation["cross_images"][0].to("cuda:0", dtype=torch.bfloat16)]],
    }
    with torch.inference_mode():
        generated = model.generate(
            **inputs, max_new_tokens=256, do_sample=False,
            eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id,
        )
    response = tokenizer.decode(generated[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    try:
        prediction = (
            parse_product_response(response, MIND2WEB_ACTIONS)
            if response.strip().startswith("{")
            else parse_cogagent_response(response, MIND2WEB_ACTIONS)
        )
    except (json.JSONDecodeError, TypeError, ValueError):
        prediction = {"action": None, "value": None, "position": None, "parse_ok": False}
    return response, prediction


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--regions", type=Path, required=True)
    parser.add_argument("--sets", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-shards", type=int, default=4)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("invalid shard index")
    sets = args.sets.split(",")
    if len(sets) != len(set(sets)) or not sets:
        raise ValueError("invalid crop set list")
    roster = yaml.safe_load((RUN_DIR / "configs/xfer_roster.yaml").read_text())
    model_spec = next(model for model in roster["mind2web"]["models"] if model["id"] == "CogAgent-18B")
    canonical_rows = [json.loads(line) for line in (RUN_DIR / "data/mind2web/mind2web_test_task.jsonl").read_text().splitlines() if line.strip()]
    sources = load_sources(args.regions)
    if set(sources) != {row["id"] for row in canonical_rows}:
        raise ValueError("crop source identity mismatch")
    indices = list(range(args.shard_index, len(canonical_rows), args.num_shards))
    if args.limit is not None:
        indices = indices[:args.limit]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists() and not args.resume:
        raise FileExistsError(args.output)
    completed = completed_ids(args.output) if args.resume else set()
    tokenizer = LlamaTokenizer.from_pretrained(RUN_DIR / "models/vicuna-7b-v1.5")
    model_dir = ROOT / model_spec["local_path"]
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True,
    ).to("cuda:0").eval()
    index_hash = sha256_file(model_dir / "model.safetensors.index.json")
    with args.output.open("a", buffering=1) as output:
        for index in indices:
            row = canonical_rows[index]
            if row["id"] in completed:
                continue
            source = sources[row["id"]]
            image = Image.open(ROOT / row["image"]).convert("RGB")
            predictions = {}
            for set_name in sets:
                predictions[set_name] = []
                for crop_index, region in enumerate(regions_for(source, set_name)):
                    response, prediction = infer(
                        image.crop(region), prompt_text(roster, row), tokenizer, model
                    )
                    predictions[set_name].append({
                        "crop_index": crop_index,
                        "region": region,
                        "response": response,
                        "prediction": remap_prediction(prediction, region, image.size),
                    })
            artifact = {
                "stable_index": index,
                "id": row["id"],
                "image_sha256": row["image_sha256"],
                "model_id": "CogAgent-18B",
                "model_revision": model_spec["revision"],
                "model_index_sha256": index_hash,
                "sets": sets,
                "source_hashes": {set_name: source_hash(source, set_name) for set_name in sets},
                "predictions": predictions,
                "predictions_sha256": hashlib.sha256(json.dumps(predictions, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
                "shard_index": args.shard_index,
                "num_shards": args.num_shards,
            }
            output.write(json.dumps(artifact, ensure_ascii=True) + "\n")
            output.flush()
            os.fsync(output.fileno())
    print(json.dumps({
        "status": "PASS", "model": "CogAgent-18B", "shard": args.shard_index,
        "completed": len(completed_ids(args.output)),
    }))


if __name__ == "__main__":
    main()