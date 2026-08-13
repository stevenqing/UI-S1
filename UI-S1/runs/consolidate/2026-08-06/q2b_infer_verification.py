import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path

import torch
import yaml
from PIL import Image
from qwen_vl_utils import process_vision_info, smart_resize
from transformers import AutoProcessor


RUN_DIR = Path(__file__).resolve().parent
ROOT = RUN_DIR.parents[2]
DATA_ROOT = ROOT / "runs/collision-law/2026-07-30/w3_assets/ScreenSpot-Pro"
DECISION_PATTERN = re.compile(r"\b(YES|NO)\b", re.IGNORECASE)


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def completed_ids(path):
    if not path.exists():
        return set()
    ids = [json.loads(line)["id"] for line in path.read_text().splitlines() if line.strip()]
    if len(ids) != len(set(ids)):
        raise ValueError("Q2b duplicate resumed ids")
    return set(ids)


def load_model(args):
    if args.model_type == "gta1":
        mvp_root = ROOT / "runs/collision-law/2026-07-30/w3_assets/MVP"
        sys.path.insert(0, str(mvp_root))
        from mvp_sspro import Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLProcessor
        config = Qwen2_5_VLConfig.from_pretrained(args.model_dir)
        config.target_token_id = ","
        config.target_layer_idx = 20
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            args.model_dir, config=config, torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2", device_map="cuda:0",
        ).eval()
        processor = Qwen2_5_VLProcessor.from_pretrained(args.model_dir, min_pixels=3136, max_pixels=4096 * 2160)
        return model, processor
    if args.model_type == "qwen3":
        from transformers import Qwen3VLForConditionalGeneration
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            args.model_dir, torch_dtype=torch.bfloat16, attn_implementation="sdpa"
        ).to("cuda:0").eval()
        return model, AutoProcessor.from_pretrained(args.model_dir, min_pixels=3136, max_pixels=4096 * 2160, use_fast=False)
    from transformers import Qwen2VLForConditionalGeneration
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        args.model_dir, torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2", device_map="cuda:0",
    ).eval()
    processor = AutoProcessor.from_pretrained(
        args.model_dir, min_pixels=3136, max_pixels=4096 * 2160, use_fast=False,
        size={"shortest_edge": 3136, "longest_edge": 2116800},
    )
    return model, processor


def verify(crop, instruction, prompt_template, processor, model):
    resized_height, resized_width = smart_resize(
        crop.height, crop.width,
        factor=processor.image_processor.patch_size * processor.image_processor.merge_size,
        min_pixels=processor.image_processor.min_pixels,
        max_pixels=processor.image_processor.max_pixels,
    )
    resized = crop.resize((resized_width, resized_height))
    question = prompt_template.format(instruction=instruction)
    messages = [
        {"role": "user", "content": [
            {"type": "image", "image": resized},
            {"type": "text", "text": question},
        ]},
    ]
    image_inputs, video_inputs = process_vision_info(messages)
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        output_ids = model.generate(**inputs, max_new_tokens=4, do_sample=False, use_cache=True)
    generated = output_ids[:, inputs.input_ids.shape[1]:]
    response = processor.batch_decode(generated, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]
    match = DECISION_PATTERN.search(response)
    return (match.group(1).upper() == "YES" if match else False), bool(match), response


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--crops", type=Path, required=True)
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
    config = yaml.safe_load((RUN_DIR / "configs/q2b_verification.yaml").read_text())
    if config["status"] != "result_blind_design_freeze":
        raise ValueError("Q2b config is not frozen")
    rows = [json.loads(line) for line in args.crops.read_text().splitlines() if line.strip()]
    if len(rows) != 1581 or any(len(row["checks"]) != 6 for row in rows):
        raise ValueError("Q2b requires complete prepared crops")
    indices = list(range(args.shard_index, len(rows), args.num_shards))
    if args.limit is not None:
        indices = indices[:args.limit]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists() and not args.resume:
        raise FileExistsError(args.output)
    completed = completed_ids(args.output) if args.resume else set()
    model, processor = load_model(args)
    index_hash = sha256_file(args.model_dir / "model.safetensors.index.json")
    written = 0
    with args.output.open("a", buffering=1) as output:
        for stable_index in indices:
            source = rows[stable_index]
            if source["id"] in completed:
                continue
            image = Image.open(DATA_ROOT / "images" / source["img_filename"]).convert("RGB")
            checks = []
            for check in source["checks"]:
                if check["verifier_model"] != args.model_id:
                    continue
                crop = image.crop(check["verification_crop"])
                decision, parse_ok, response = verify(
                    crop, source["instruction"], config["stage_2"]["prompt_template"], processor, model
                )
                checks.append({
                    "check_index": check["check_index"],
                    "candidate_model": check["candidate_model"],
                    "candidate_view": check["candidate_view"],
                    "verification_crop": check["verification_crop"],
                    "decision": decision,
                    "parse_ok": parse_ok,
                    "response": response,
                })
            if len(checks) != 2:
                raise ValueError(f"Q2b verifier check count mismatch: {args.model_id}/{source['id']}")
            artifact = {
                "stable_index": stable_index,
                "id": source["id"],
                "verifier_model": args.model_id,
                "model_revision": args.model_revision,
                "model_index_sha256": index_hash,
                "checks_sha256": source["checks_sha256"],
                "checks": checks,
                "shard_index": args.shard_index,
                "num_shards": args.num_shards,
            }
            artifact["outputs_sha256"] = canonical_hash(checks)
            output.write(json.dumps(artifact, ensure_ascii=True) + "\n")
            output.flush()
            os.fsync(output.fileno())
            written += 1
            if written % 10 == 0:
                print(json.dumps({"verifier": args.model_id, "shard": args.shard_index, "written": written}), flush=True)
    print(json.dumps({
        "status": "PASS", "verifier": args.model_id, "shard": args.shard_index,
        "written": written, "completed": len(completed_ids(args.output)),
    }), flush=True)


if __name__ == "__main__":
    main()