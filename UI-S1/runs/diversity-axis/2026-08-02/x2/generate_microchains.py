import argparse
import hashlib
import json
import math
import os
import re
import sys
from pathlib import Path

import torch
from PIL import Image
from qwen_vl_utils import process_vision_info, smart_resize
from transformers import AutoProcessor


RUN_DIR = Path(__file__).resolve().parents[1]
ROOT = RUN_DIR.parents[2]
DATA_ROOT = ROOT / "runs/collision-law/2026-07-30/w3_assets/ScreenSpot-Pro"
sys.path.insert(0, str(Path(__file__).parent))
from zoom_port import adaptive_crop, deterministic_seed, gate, point_to_box


MODEL_SPECS = {
    "gta1": {
        "id": "GTA1-7B",
        "revision": "701bedc80b447863bd60e3318ae44f6cbbfafd78",
        "path": ROOT / "runs/collision-law/2026-07-30/w3_assets/GTA1-7B",
    },
    "qwen3": {
        "id": "Qwen3-VL-8B-Instruct",
        "revision": "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b",
        "path": ROOT / "runs/ccm-h2h/2026-07-31/h3/models/Qwen3-VL-8B-Instruct",
    },
    "uitars": {
        "id": "UI-TARS-7B-SFT",
        "revision": "3434901a9dd04dd3625617d839a5724fe5e2db20",
        "path": ROOT / "runs/mind2web-tongui/2026-07-28/models/UI-TARS-7B-SFT",
    },
}
NORMALIZED_SYSTEM_PROMPT = """You are an expert UI element locator. Given a GUI image and a user's element description, provide the coordinates of the specified element as a single (x,y) point. For elements with area, return the center point. Output only one coordinate pair."""
PIXEL_SYSTEM_PROMPT = """You are an expert UI element locator. Given a GUI image and a user's element description, provide the coordinates of the specified element as a single (x,y) point. The image resolution is height {height} and width {width}. For elements with area, return the center point.\n\nOutput the coordinate pair exactly:\n(x,y)"""
POINT_PATTERN = re.compile(
    r"(?:<point>\s*)?([0-9]{1,5})[\s,]+([0-9]{1,5})(?:\s*</point>)?"
    r"|\[\[?\s*([0-9]{1,5})\s*,\s*([0-9]{1,5})\s*\]?\]"
    r"|\(\s*([0-9]{1,5})\s*,\s*([0-9]{1,5})\s*\)"
)


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_pair(response):
    match = POINT_PATTERN.search(response)
    if not match:
        return None
    values = [value for value in match.groups() if value is not None]
    return tuple(map(float, values[:2]))


def load_backend(model_type):
    spec = MODEL_SPECS[model_type]
    if model_type == "gta1":
        from transformers import Qwen2_5_VLForConditionalGeneration
        model_class = Qwen2_5_VLForConditionalGeneration
    elif model_type == "uitars":
        from transformers import Qwen2VLForConditionalGeneration
        model_class = Qwen2VLForConditionalGeneration
    else:
        from transformers import Qwen3VLForConditionalGeneration
        model_class = Qwen3VLForConditionalGeneration
    kwargs = {
        "torch_dtype": torch.bfloat16,
        "attn_implementation": "sdpa" if model_type == "qwen3" else "flash_attention_2",
    }
    if model_type != "qwen3":
        kwargs["device_map"] = "cuda:0"
    model = model_class.from_pretrained(spec["path"], **kwargs)
    if model_type == "qwen3":
        model = model.to("cuda:0")
    model = model.eval()
    processor_kwargs = {"min_pixels": 10000, "max_pixels": 5000000, "use_fast": False}
    if model_type == "uitars":
        processor_kwargs["size"] = {"shortest_edge": 10000, "longest_edge": 5000000}
    processor = AutoProcessor.from_pretrained(spec["path"], **processor_kwargs)
    return spec, model, processor


def confidence_from_generation(model, generated, generated_ids, pad_token_id):
    transition = model.compute_transition_scores(generated.sequences, generated.scores, normalize_logits=True)
    token_ids = generated_ids[0, :transition.shape[1]]
    scores = transition[0]
    mask = token_ids.ne(pad_token_id) if pad_token_id is not None else torch.ones_like(token_ids, dtype=torch.bool)
    finite = scores[mask & torch.isfinite(scores)]
    return float(torch.exp(finite.mean()).item()) if finite.numel() else 0.0


def infer(image, instruction, model_type, processor, model, seed, temperature, offset_x=0, offset_y=0, global_image=True):
    minimum, maximum = ((2000000, 4800000) if global_image else (1000000, 4000000))
    factor = processor.image_processor.patch_size * processor.image_processor.merge_size
    resized_height, resized_width = smart_resize(
        image.height, image.width, factor=factor, min_pixels=minimum, max_pixels=maximum
    )
    resized = image.resize((resized_width, resized_height), Image.Resampling.BICUBIC)
    system_prompt = (
        PIXEL_SYSTEM_PROMPT.format(height=resized_height, width=resized_width)
        if model_type == "gta1" else NORMALIZED_SYSTEM_PROMPT
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": [
            {"type": "image", "image": resized},
            {"type": "text", "text": instruction},
        ]},
    ]
    image_inputs, video_inputs = process_vision_info(messages)
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt").to(model.device)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    kwargs = {
        "max_new_tokens": 32,
        "do_sample": temperature > 0,
        "use_cache": True,
        "return_dict_in_generate": True,
        "output_scores": True,
        "pad_token_id": processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id,
    }
    if temperature > 0:
        kwargs.update({"temperature": temperature, "top_p": 1.0})
    with torch.inference_mode():
        generated = model.generate(**inputs, **kwargs)
    generated_ids = generated.sequences[:, inputs.input_ids.shape[1]:]
    response = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]
    confidence = confidence_from_generation(model, generated, generated_ids, kwargs["pad_token_id"])
    parsed = parse_pair(response)
    if parsed is None:
        point = None
    elif model_type == "gta1":
        point = [parsed[0] * image.width / resized_width + offset_x, parsed[1] * image.height / resized_height + offset_y]
    elif 0 <= parsed[0] <= 1000 and 0 <= parsed[1] <= 1000:
        point = [parsed[0] / 1000 * image.width + offset_x, parsed[1] / 1000 * image.height + offset_y]
    else:
        point = None
    return {
        "point": point,
        "response": response,
        "confidence": confidence,
        "seed": seed,
        "temperature": temperature,
        "resized_size": [resized_width, resized_height],
    }


def run_chain(source, image, cell, model_type, model_id, chain_index, processor, model):
    width, height = image.size
    predictions = []
    candidates = []
    for slot in range(3):
        seed = deterministic_seed(source["id"], cell, model_id, chain_index, slot)
        prediction = infer(image, source["instruction"], model_type, processor, model, seed, 0.9)
        box = point_to_box(prediction["point"], width, height)
        candidates.append({"box": box, "confidence": prediction["confidence"]})
        predictions.append({
            **prediction,
            "chain_index": chain_index,
            "slot": slot,
            "branch": "global_sample",
            "region": [0, 0, width, height],
            "box": box,
        })
    gate_report = gate(candidates)
    crop = adaptive_crop(candidates, width, height) if not gate_report["reliable"] else None
    seed = deterministic_seed(source["id"], cell, model_id, chain_index, 3)
    if crop is not None:
        left, top, right, bottom = crop
        branch = infer(
            image.crop(crop), source["instruction"], model_type, processor, model,
            seed, 0.0, offset_x=left, offset_y=top, global_image=False,
        )
        branch_kind = "adaptive_crop_refine"
        region = crop
    else:
        branch = infer(image, source["instruction"], model_type, processor, model, seed, 0.9)
        branch_kind = "global_confirmation"
        region = [0, 0, width, height]
    predictions.append({
        **branch,
        "chain_index": chain_index,
        "slot": 3,
        "branch": branch_kind,
        "region": region,
        "box": point_to_box(branch["point"], width, height),
    })
    return predictions, {**gate_report, "chain_index": chain_index, "branch": branch_kind, "crop": crop}


def completed_ids(path):
    if not path.exists():
        return set()
    ids = [json.loads(line)["id"] for line in path.read_text().splitlines() if line.strip()]
    if len(ids) != len(set(ids)):
        raise ValueError("X2 duplicate resume identities")
    return set(ids)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--cell", choices=("Q2", "Q4"), required=True)
    parser.add_argument("--model-type", choices=tuple(MODEL_SPECS), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-shards", type=int, default=2)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.cell == "Q2" and args.model_type != "gta1":
        raise ValueError("Q2 uses GTA1 only")
    rows = [json.loads(line) for line in args.inputs.read_text().splitlines() if line.strip()]
    if len(rows) != 1581 or any("target_bbox" in row or "bbox" in row for row in rows):
        raise ValueError("X2 requires complete label-free input manifest")
    indices = list(range(args.shard_index, len(rows), args.num_shards))
    if args.limit is not None:
        indices = indices[:args.limit]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists() and not args.resume:
        raise FileExistsError(args.output)
    completed = completed_ids(args.output) if args.resume else set()
    spec, model, processor = load_backend(args.model_type)
    model_hash = sha256_file(spec["path"] / "model.safetensors.index.json")
    chains = range(3) if args.cell == "Q2" else range(1)
    written = 0
    with args.output.open("a", buffering=1) as output:
        for stable_index in indices:
            source = rows[stable_index]
            if source["id"] in completed:
                continue
            image = Image.open(DATA_ROOT / "images" / source["img_filename"]).convert("RGB")
            predictions = []
            chain_reports = []
            for chain_index in chains:
                chain_predictions, chain_report = run_chain(
                    source, image, args.cell, args.model_type, spec["id"], chain_index, processor, model
                )
                predictions.extend(chain_predictions)
                chain_reports.append(chain_report)
            artifact = {
                **source,
                "cell": args.cell,
                "model_id": spec["id"],
                "model_revision": spec["revision"],
                "model_index_sha256": model_hash,
                "policy": "fixed12_microchains",
                "forward_count": len(predictions),
                "valid_candidate_count": sum(prediction["point"] is not None for prediction in predictions),
                "predictions": predictions,
                "chain_reports": chain_reports,
                "shard_index": args.shard_index,
                "num_shards": args.num_shards,
            }
            artifact["prediction_sha256"] = hashlib.sha256(
                json.dumps(predictions, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()
            output.write(json.dumps(artifact, ensure_ascii=True) + "\n")
            output.flush()
            os.fsync(output.fileno())
            written += 1
            if written % 25 == 0:
                print(json.dumps({"cell": args.cell, "model": spec["id"], "shard": args.shard_index, "written": written}), flush=True)
    print(json.dumps({"status": "PASS", "cell": args.cell, "model": spec["id"], "shard": args.shard_index, "written": written}), flush=True)


if __name__ == "__main__":
    main()