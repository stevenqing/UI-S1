import argparse
import hashlib
import json
import os
import re
from pathlib import Path

import torch
from PIL import Image
from qwen_vl_utils import process_vision_info, smart_resize
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


RUN_DIR = Path(__file__).resolve().parent
ROOT = RUN_DIR.parents[2]
MODEL_DIR = ROOT / "runs/collision-law/2026-07-30/w3_assets/GTA1-7B"
DATA_DIR = ROOT / "runs/collision-law/2026-07-30/w3_assets/ScreenSpot-Pro"
MODEL_REVISION = "701bedc80b447863bd60e3318ae44f6cbbfafd78"
SAMPLES = 16
TEMPERATURE = 0.5
TOP_P = 0.95
SEED = 20260802
SYSTEM_PROMPT = """You are an expert UI element locator. Given a GUI image and a user's element description, provide the coordinates of the specified element as a single (x,y) point. The image resolution is height {height} and width {width}. For elements with area, return the center point.

Output the coordinate pair exactly:
(x,y)"""


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_coordinate(response):
    matches = re.findall(r"\((-?\d*\.?\d+),\s*(-?\d*\.?\d+)\)", response)
    if not matches:
        return None
    try:
        return tuple(map(float, matches[0]))
    except ValueError:
        return None


def sample_seed(stable_index, sample_index):
    return SEED + stable_index * SAMPLES + sample_index


def completed_ids(path):
    if not path.exists():
        return set()
    ids = [json.loads(line)["id"] for line in path.read_text().splitlines() if line.strip()]
    if len(ids) != len(set(ids)):
        raise ValueError("F2 duplicate resumed identities")
    return set(ids)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-shards", type=int, default=8)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.inputs.read_text().splitlines() if line.strip()]
    if len(rows) != 1581 or any("bbox" in row or "target_bbox" in row for row in rows):
        raise ValueError("F2 requires complete label-free inputs")
    indices = list(range(args.shard_index, len(rows), args.num_shards))
    if args.limit is not None:
        indices = indices[:args.limit]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists() and not args.resume:
        raise FileExistsError(args.output)
    completed = completed_ids(args.output) if args.resume else set()
    processor = AutoProcessor.from_pretrained(MODEL_DIR, min_pixels=3136, max_pixels=4096 * 2160, use_fast=False)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_DIR, torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2", device_map="cuda:0",
    ).eval()
    model_hash = sha256_file(MODEL_DIR / "model.safetensors.index.json")
    written = 0
    with args.output.open("a", buffering=1) as output:
        for stable_index in indices:
            source = rows[stable_index]
            if source["id"] in completed:
                continue
            image = Image.open(DATA_DIR / "images" / source["img_filename"]).convert("RGB")
            resized_height, resized_width = smart_resize(
                image.height, image.width,
                factor=processor.image_processor.patch_size * processor.image_processor.merge_size,
                min_pixels=processor.image_processor.min_pixels,
                max_pixels=processor.image_processor.max_pixels,
            )
            resized = image.resize((resized_width, resized_height))
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT.format(height=resized_height, width=resized_width)},
                {"role": "user", "content": [
                    {"type": "image", "image": resized},
                    {"type": "text", "text": source["instruction"]},
                ]},
            ]
            image_inputs, video_inputs = process_vision_info(messages)
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(
                text=[text], images=image_inputs, videos=video_inputs,
                padding=True, return_tensors="pt",
            ).to(model.device)
            samples = []
            for sample_index in range(SAMPLES):
                seed = sample_seed(stable_index, sample_index)
                torch.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
                with torch.inference_mode():
                    output_ids = model.generate(
                        **inputs, max_new_tokens=32, do_sample=True,
                        temperature=TEMPERATURE, top_p=TOP_P, use_cache=True,
                    )
                generated = output_ids[:, inputs.input_ids.shape[1]:]
                response = processor.batch_decode(
                    generated, skip_special_tokens=True,
                    clean_up_tokenization_spaces=True,
                )[0]
                resized_point = parse_coordinate(response)
                point = None if resized_point is None else [
                    resized_point[0] * image.width / resized_width,
                    resized_point[1] * image.height / resized_height,
                ]
                samples.append({
                    "sample_index": sample_index,
                    "seed": seed,
                    "response": response,
                    "point": point,
                })
            artifact = {
                **source,
                "model_id": "GTA1-7B",
                "model_revision": MODEL_REVISION,
                "model_index_sha256": model_hash,
                "samples": SAMPLES,
                "temperature": TEMPERATURE,
                "top_p": TOP_P,
                "resized_size": [resized_width, resized_height],
                "predictions": samples,
                "valid_predictions": sum(sample["point"] is not None for sample in samples),
                "shard_index": args.shard_index,
                "num_shards": args.num_shards,
            }
            artifact["prediction_sha256"] = hashlib.sha256(
                json.dumps(samples, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()
            output.write(json.dumps(artifact, ensure_ascii=True) + "\n")
            output.flush()
            os.fsync(output.fileno())
            written += 1
            if written % 25 == 0:
                print(json.dumps({"shard": args.shard_index, "written": written}), flush=True)
    print(json.dumps({"status": "PASS", "shard": args.shard_index, "written": written}), flush=True)


if __name__ == "__main__":
    main()