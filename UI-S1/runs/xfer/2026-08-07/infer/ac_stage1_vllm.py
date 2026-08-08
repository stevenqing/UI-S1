import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor
from vllm import LLM, SamplingParams


RUN_DIR = Path(__file__).resolve().parents[1]
ROOT = RUN_DIR.parents[2]
sys.path.insert(0, str(RUN_DIR))

from ac_common import COORDINATE_ACTIONS, load_prompt_templates, parse_response, prompt_text


MODEL_SPECS = {
    "UI-AGILE-7B": {
        "local_path": "runs/xfer/2026-08-07/models/UI-AGILE-7B",
        "prompt": "UI-AGILE-7B",
    },
    "GUI-R1-7B": {
        "local_path": "runs/xfer/2026-08-07/models/GUI-R1/GUI-R1-7B",
        "prompt": "GUI-R1-7B",
    },
    "UI-R1-E-3B": {
        "local_path": "runs/xfer/2026-08-07/models/UI-R1-E-3B",
        "prompt": "UI-R1-E-3B",
    },
}


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
        raise ValueError("duplicate AC stage1 ids")
    return set(ids)


def build_request(row, model_id, processor, templates):
    image = Image.open(ROOT / row["image"]).convert("RGB")
    text_prompt = prompt_text(model_id, row, templates)
    full_prompt = "<image>\n" + text_prompt
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": full_prompt},
    ]}]
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, _, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
    processor_inputs = processor(
        text=[prompt], images=image_inputs, padding=True, return_tensors="pt"
    )
    grid = processor_inputs["image_grid_thw"][0]
    patch_size = processor.image_processor.patch_size
    resized_height = float(grid[1] * patch_size)
    resized_width = float(grid[2] * patch_size)
    scale = [image.width / resized_width, image.height / resized_height]
    return {
        "prompt": prompt,
        "multi_modal_data": {"image": image_inputs},
        "mm_processor_kwargs": video_kwargs,
    }, {
        "image_size": list(image.size),
        "coordinate_scale": scale,
        "text_prompt_sha256": hashlib.sha256(text_prompt.encode()).hexdigest(),
        "model_prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
    }


def normalize_prediction(prediction, image_size):
    value = dict(prediction)
    value["pixel_position"] = prediction["position"]
    if prediction["position"] is not None and prediction["action"] in COORDINATE_ACTIONS:
        value["position"] = [
            prediction["position"][0] / image_size[0],
            prediction["position"][1] / image_size[1],
        ]
    else:
        value["position"] = None
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", choices=MODEL_SPECS, required=True)
    parser.add_argument("--setting", choices=("low", "high"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("invalid shard index")
    if args.batch_size < 1:
        raise ValueError("batch size must be positive")
    roster = __import__("yaml").safe_load((RUN_DIR / "configs/xfer_roster.yaml").read_text())
    model_spec = next(model for model in roster["androidcontrol"]["models"] if model["id"] == args.model_id)
    expected_path = ROOT / model_spec["local_path"]
    configured_path = ROOT / MODEL_SPECS[args.model_id]["local_path"]
    if expected_path.resolve() != configured_path.resolve():
        raise ValueError("AC model path differs from frozen roster")
    rows = [json.loads(line) for line in (RUN_DIR / f"data/androidcontrol/{args.setting}_sample.jsonl").read_text().splitlines() if line.strip()]
    if len(rows) != 2000:
        raise ValueError(f"AC stage1 expected 2,000 rows, found {len(rows)}")
    indices = list(range(args.shard_index, len(rows), args.num_shards))
    if args.limit is not None:
        indices = indices[:args.limit]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists() and not args.resume:
        raise FileExistsError(args.output)
    completed = completed_ids(args.output) if args.resume else set()
    processor = AutoProcessor.from_pretrained(
        configured_path, trust_remote_code=True, use_fast=False,
        min_pixels=256 * 28 * 28, max_pixels=12800 * 28 * 28,
    )
    model = LLM(
        model=str(configured_path), trust_remote_code=True,
        tensor_parallel_size=1, dtype="bfloat16", max_model_len=8192,
        gpu_memory_utilization=0.65, kv_cache_memory_bytes=2 * 1024**3,
        limit_mm_per_prompt={"image": 1}, enforce_eager=True,
    )
    sampling = SamplingParams(temperature=0.0, max_tokens=256, skip_special_tokens=False)
    templates = load_prompt_templates()
    index_hash = sha256_file(configured_path / "model.safetensors.index.json")
    pending = [index for index in indices if rows[index]["id"] not in completed]
    with args.output.open("a", buffering=1) as output:
        for start in range(0, len(pending), args.batch_size):
            batch_indices = pending[start:start + args.batch_size]
            requests, metadata = [], []
            for index in batch_indices:
                request, values = build_request(rows[index], args.model_id, processor, templates)
                requests.append(request)
                metadata.append(values)
            responses = model.generate(requests, sampling_params=sampling, use_tqdm=False)
            for index, values, response in zip(batch_indices, metadata, responses):
                row = rows[index]
                generated = response.outputs[0].text
                prediction = normalize_prediction(
                    parse_response(generated, args.model_id, values["coordinate_scale"]),
                    values["image_size"],
                )
                artifact = {
                    "stable_index": row["stable_index"],
                    "id": row["id"],
                    "episode_id": row["episode_id"],
                    "setting": args.setting,
                    "source_index": row["source_index"],
                    "source_sha256": row["source_sha256"],
                    "image_sha256": row["image_sha256"],
                    "image_size": values["image_size"],
                    "model_id": args.model_id,
                    "model_revision": model_spec["revision"],
                    "model_index_sha256": index_hash,
                    "response": generated,
                    "prediction": prediction,
                    "text_prompt_sha256": values["text_prompt_sha256"],
                    "model_prompt_sha256": values["model_prompt_sha256"],
                    "shard_index": args.shard_index,
                    "num_shards": args.num_shards,
                }
                output.write(json.dumps(artifact, ensure_ascii=True) + "\n")
                output.flush()
                os.fsync(output.fileno())
    print(json.dumps({
        "status": "PASS", "model": args.model_id, "setting": args.setting,
        "completed": len(completed_ids(args.output)),
    }))


if __name__ == "__main__":
    main()