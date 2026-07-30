import argparse
import hashlib
import importlib.util
import json
import math
import os
from io import BytesIO
from pathlib import Path

import pyarrow.parquet as parquet
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor
from vllm import LLM, SamplingParams

from views import generate_view, max_visual_tokens


RUN_DIR = Path(__file__).resolve().parent
ROOT = RUN_DIR.parents[3]
UPSTREAM_DIR = ROOT / "runs/androidcontrol-rft/2026-07-29"
MODEL_CONFIG = {
    "gui-r1-7b": {
        "model_dir": UPSTREAM_DIR / "models/GUI-R1/GUI-R1-7B",
        "model_name": "ritzzai/GUI-R1:GUI-R1-7B",
        "prompt_template": "gui_r1",
    },
    "ui-agile-7b": {
        "model_dir": UPSTREAM_DIR / "models/UI-AGILE-7B",
        "model_name": "KDEGroup/UI-AGILE",
        "prompt_template": "android_control_detailed",
    },
}


def load_upstream():
    spec = importlib.util.spec_from_file_location("collision_ac_upstream_infer", UPSTREAM_DIR / "infer.py")
    if spec is None or spec.loader is None:
        raise ImportError(UPSTREAM_DIR / "infer.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


UPSTREAM = load_upstream()


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def build_request(sample, processor, prompt_template, view_id, full_prediction_center):
    original = Image.open(BytesIO(sample["image"]["bytes"])).convert("RGB")
    generated = generate_view(original, view_id, full_prediction_center)
    history = sample.get("history", "None")
    text_prompt = prompt_template.format(instruction=sample["instruction"], history=history)
    full_prompt = "<image>\n" + text_prompt
    message = [{
        "role": "user",
        "content": [
            {"type": "image", "image": generated.image},
            {"type": "text", "text": full_prompt},
        ],
    }]
    prompt = processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
    image_inputs, _, video_kwargs = process_vision_info(message, return_video_kwargs=True)
    inputs = processor(text=[prompt], images=image_inputs, padding=True, return_tensors="pt")
    image_grid = inputs["image_grid_thw"][0]
    patch_size = processor.image_processor.patch_size
    resized_height = float(image_grid[1] * patch_size)
    resized_width = float(image_grid[2] * patch_size)
    view_width, view_height = generated.image.size
    return {
        "prompt": prompt,
        "multi_modal_data": {"image": image_inputs},
        "mm_processor_kwargs": video_kwargs,
    }, {
        "geometry": generated.geometry,
        "view_scale": [view_width / resized_width, view_height / resized_height],
        "text_prompt_sha256": sha256_bytes(text_prompt.encode()),
        "model_prompt_sha256": sha256_bytes(prompt.encode()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODEL_CONFIG, required=True)
    parser.add_argument("--setting", choices=("low", "high"), required=True)
    parser.add_argument("--view", choices=("v1", "v2", "v3", "v4"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-shards", type=int, default=4)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--kv-cache-memory-bytes", type=int, default=2 * 1024**3)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("shard index outside range")
    config = MODEL_CONFIG[args.model]
    data_path = UPSTREAM_DIR / f"data/UI-AGILE-Data/android_control/androidcontrol_{args.setting}_test.parquet"
    rows = parquet.read_table(data_path).to_pylist()
    if len(rows) != 7708:
        raise ValueError("AndroidControl W2 requires exactly 7,708 rows")
    full_path = UPSTREAM_DIR / f"artifacts/{args.model}/{args.setting}/predictions.jsonl"
    full_rows = read_jsonl(full_path)
    if len(full_rows) != 7708 or [row["index"] for row in full_rows] != list(range(7708)):
        raise ValueError("full-view prediction coverage mismatch")

    prompt_template = UPSTREAM.load_prompt_template(config["prompt_template"])
    extract_action, extract_coordinates, extract_parameter, extract_gui_r1_parameter = UPSTREAM.load_official_parsers()
    visual_tokens = max_visual_tokens("androidcontrol", args.view)
    processor = AutoProcessor.from_pretrained(
        config["model_dir"].resolve(), trust_remote_code=True, use_fast=False,
        min_pixels=256 * 28 * 28, max_pixels=visual_tokens * 28 * 28,
    )
    model = LLM(
        model=str(config["model_dir"].resolve()), trust_remote_code=True,
        tensor_parallel_size=1, dtype="bfloat16", max_model_len=8192,
        gpu_memory_utilization=0.65, kv_cache_memory_bytes=args.kv_cache_memory_bytes,
        limit_mm_per_prompt={"image": 1}, enforce_eager=True,
    )
    sampling = SamplingParams(temperature=0.0, max_tokens=256, skip_special_tokens=False)

    indices = list(range(args.shard_index, len(rows), args.num_shards))
    if args.limit is not None:
        indices = indices[:args.limit]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    existing = read_jsonl(args.output) if args.resume else []
    if args.output.exists() and not args.resume:
        raise FileExistsError(args.output)
    completed = {row["index"] for row in existing}
    if len(completed) != len(existing):
        raise ValueError("duplicate resumed indices")
    pending = [index for index in indices if index not in completed]

    with args.output.open("a", buffering=1) as output:
        for start in range(0, len(pending), args.batch_size):
            batch_indices = pending[start:start + args.batch_size]
            requests, metadata = [], []
            for index in batch_indices:
                full_center = tuple(full_rows[index]["pred_coord"][:2])
                request, item_metadata = build_request(
                    rows[index], processor, prompt_template, args.view, full_center,
                )
                requests.append(request)
                metadata.append(item_metadata)
            responses = model.generate(requests, sampling_params=sampling, use_tqdm=False)
            for index, item_metadata, response in zip(batch_indices, metadata, responses):
                sample = rows[index]
                raw = response.outputs[0].text
                action_raw = extract_action(raw)
                action = UPSTREAM.MODEL_TO_GT_ACTION_MAP.get(action_raw, action_raw)
                coordinate, _, _ = extract_coordinates(raw)
                if coordinate is None:
                    coordinate = [0, 0]
                view_coordinate = (
                    coordinate[0] * item_metadata["view_scale"][0],
                    coordinate[1] * item_metadata["view_scale"][1],
                )
                normalized = item_metadata["geometry"].view_to_original_normalized(*view_coordinate)
                original_width, original_height = item_metadata["geometry"].original_size
                original_coordinate = [normalized[0] * original_width, normalized[1] * original_height]
                parameter = extract_parameter(raw)
                if config["prompt_template"] == "gui_r1":
                    parameter = extract_gui_r1_parameter(raw)
                gt_input = sample["gt_input_text"].lower() if sample["gt_action"] == "scroll" else sample["gt_input_text"]
                result = {
                    "index": index, "data_setting": args.setting,
                    "instruction": sample["instruction"], "history": sample.get("history"),
                    "gt_action": sample["gt_action"], "gt_bbox": sample["gt_bbox"],
                    "gt_input_text": gt_input, "group": sample["group"], "ui_type": sample["ui_type"],
                    "image_size": list(item_metadata["geometry"].original_size),
                    "pred_raw": raw, "pred_action": action,
                    "pred_coord": original_coordinate, "pred_x": normalized[0], "pred_y": normalized[1],
                    "pred_input_text": parameter,
                    "image_sha256": sha256_bytes(sample["image"]["bytes"]),
                    "view_id": args.view, "pred_source": f"{args.model}__{args.view}",
                    "view_size": list(item_metadata["geometry"].view_size),
                    "view_offset": [item_metadata["geometry"].offset_x, item_metadata["geometry"].offset_y],
                    "center_fallback": item_metadata["geometry"].center_fallback,
                    "full_prediction_source_sha256": sha256_bytes(json.dumps(full_rows[index], sort_keys=True).encode()),
                    "text_prompt_sha256": item_metadata["text_prompt_sha256"],
                    "model_prompt_sha256": item_metadata["model_prompt_sha256"],
                    "model": args.model, "model_name": config["model_name"],
                    "model_revision": UPSTREAM.MODEL_REVISIONS[config["model_name"]],
                    "prompt_template": config["prompt_template"],
                    "max_visual_tokens": visual_tokens, "generation": "greedy_max_tokens_256",
                    "num_shards": args.num_shards, "shard_index": args.shard_index,
                    "kv_cache_memory_bytes": args.kv_cache_memory_bytes,
                }
                output.write(json.dumps(result, ensure_ascii=True) + "\n")
            output.flush()
            os.fsync(output.fileno())


if __name__ == "__main__":
    main()