import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path

import torch
from PIL import Image
from qwen_vl_utils import process_vision_info, smart_resize
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration, Qwen3VLForConditionalGeneration


ROOT = Path(__file__).resolve().parents[4]
MVP_ROOT = ROOT / "runs/collision-law/2026-07-30/w3_assets/MVP"
DATA_ROOT = ROOT / "runs/collision-law/2026-07-30/w3_assets/ScreenSpot-Pro"
sys.path.insert(0, str(MVP_ROOT))


def load_backend(model_type):
    if model_type == "uitars":
        return process_uitars_subimage, None, Qwen2VLForConditionalGeneration
    if model_type == "qwen3":
        return process_qwen3_subimage, None, Qwen3VLForConditionalGeneration
    raise ValueError(model_type)


SYSTEM_PROMPT = """You are an expert UI element locator. Given a GUI image and a user's element description, provide the coordinates of the specified element as a single (x,y) point. For elements with area, return the center point. Output only one coordinate pair."""
POINT_PATTERN = re.compile(
    r"(?:<point>\s*)?([0-9]{1,4})[\s,]+([0-9]{1,4})(?:\s*</point>)?"
    r"|\[\[?\s*([0-9]{1,4})\s*,\s*([0-9]{1,4})\s*\]?\]"
    r"|\(\s*([0-9]{1,4})\s*,\s*([0-9]{1,4})\s*\)"
)


def parse_uitars_point(response):
    match = POINT_PATTERN.search(response)
    if not match:
        return None
    values = [value for value in match.groups() if value is not None]
    x, y = map(int, values[:2])
    if x > 1000 or y > 1000:
        return None
    return x, y


def process_uitars_subimage(subimage, instruction, processor, model, device, offset_x=0, offset_y=0, resize=False):
    resized_height, resized_width = smart_resize(
        subimage.height, subimage.width,
        factor=processor.image_processor.patch_size * processor.image_processor.merge_size,
        min_pixels=processor.image_processor.min_pixels,
        max_pixels=processor.image_processor.max_pixels,
    )
    resized = subimage.resize((resized_width, resized_height))
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": [
            {"type": "image", "image": resized},
            {"type": "text", "text": instruction},
        ]},
    ]
    image_inputs, video_inputs = process_vision_info(messages)
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        output_ids = model.generate(**inputs, max_new_tokens=32, do_sample=False, use_cache=True)
    generated = output_ids[:, inputs.input_ids.shape[1]:]
    response = processor.batch_decode(generated, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]
    point = parse_uitars_point(response)
    if point is None:
        return (0, 0), response
    x, y = point
    return (
        int(x / 1000 * subimage.width + offset_x),
        int(y / 1000 * subimage.height + offset_y),
    ), response


def process_qwen3_subimage(subimage, instruction, processor, model, device, offset_x=0, offset_y=0, resize=False):
    resized_height, resized_width = smart_resize(
        subimage.height, subimage.width,
        factor=processor.image_processor.patch_size * processor.image_processor.merge_size,
        min_pixels=processor.image_processor.min_pixels,
        max_pixels=processor.image_processor.max_pixels,
    )
    resized = subimage.resize((resized_width, resized_height))
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": [
            {"type": "image", "image": resized},
            {"type": "text", "text": instruction},
        ]},
    ]
    image_inputs, video_inputs = process_vision_info(messages)
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        output_ids = model.generate(**inputs, max_new_tokens=32, do_sample=False, use_cache=True)
    generated = output_ids[:, inputs.input_ids.shape[1]:]
    response = processor.batch_decode(generated, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]
    point = parse_uitars_point(response)
    if point is None:
        return (0, 0), response
    x, y = point
    return (
        int(x / 1000 * subimage.width + offset_x),
        int(y / 1000 * subimage.height + offset_y),
    ), response


def completed_ids(path):
    if not path.exists():
        return set()
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    ids = [row["id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate H3 resume ids")
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
    parser.add_argument("--model-type", choices=("uitars", "qwen3"), required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--views", type=int, choices=(1, 4), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-shards", type=int, default=8)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.regions.read_text().splitlines() if line.strip()]
    if len(rows) != 1581 or any(len(row["regions"]) != 4 for row in rows):
        raise ValueError("H3 requires complete N4 shared regions")
    indices = list(range(args.shard_index, len(rows), args.num_shards))
    if args.limit is not None:
        indices = indices[:args.limit]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists() and not args.resume:
        raise FileExistsError(args.output)
    completed = completed_ids(args.output) if args.resume else set()

    process_subimage, config_class, model_class = load_backend(args.model_type)
    config = config_class.from_pretrained(args.model_dir) if config_class is not None else None
    model_kwargs = {
        **({"config": config} if config is not None else {}),
        "torch_dtype": torch.bfloat16,
        "attn_implementation": "sdpa" if args.model_type == "qwen3" else "flash_attention_2",
    }
    if args.model_type == "uitars":
        model_kwargs["device_map"] = "cuda:0"
    model = model_class.from_pretrained(args.model_dir, **model_kwargs)
    if args.model_type == "qwen3":
        model = model.to("cuda:0")
    model = model.eval()
    processor_kwargs = {
        "min_pixels": 3136, "max_pixels": 4096 * 2160, "use_fast": False,
    }
    if args.model_type == "uitars":
        processor_kwargs["size"] = {"shortest_edge": 3136, "longest_edge": 2116800}
    processor = AutoProcessor.from_pretrained(args.model_dir, **processor_kwargs)
    model_hash = sha256_file(args.model_dir / "model.safetensors.index.json")
    with args.output.open("a", buffering=1) as output:
        for index in indices:
            source = rows[index]
            if source["id"] in completed:
                continue
            image = Image.open(DATA_ROOT / "images" / source["img_filename"]).convert("RGB")
            predictions = []
            for view_index in range(args.views):
                region = source["regions"][view_index]
                left, top, right, bottom = region
                crop = image if view_index == 0 else image.crop(region)
                point, response = process_subimage(
                    crop, source["instruction"], processor, model, 0,
                    offset_x=0 if view_index == 0 else left,
                    offset_y=0 if view_index == 0 else top,
                    resize=view_index != 0,
                )
                predictions.append({
                    "view_index": view_index,
                    "region": region,
                    "point": list(map(float, point)),
                    "response": response,
                })
            artifact = {
                "stable_index": index,
                "id": source["id"],
                "application": source["application"],
                "img_filename": source["img_filename"],
                "img_size": source["img_size"],
                "target_bbox": source["target_bbox"],
                "instruction": source["instruction"],
                "model_id": args.model_id,
                "model_revision": args.model_revision,
                "model_index_sha256": model_hash,
                "shared_region_candidate_sha256": source["shared_region_candidate_sha256"],
                "predictions": predictions,
                "views": args.views,
                "shard_index": args.shard_index,
                "num_shards": args.num_shards,
            }
            artifact["prediction_sha256"] = hashlib.sha256(
                json.dumps(predictions, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()
            output.write(json.dumps(artifact, ensure_ascii=True) + "\n")
            output.flush()
            os.fsync(output.fileno())


if __name__ == "__main__":
    main()
