import argparse
import hashlib
import json
import os
import re
from pathlib import Path

from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor
from vllm import LLM, SamplingParams


MODEL_NAME = "LZXzju/Qwen2.5-VL-3B-UI-R1"
MODEL_REVISION = "9cc2fbb7d99ffe90c21f9cd0eb19c45380f8bb0f"
SOURCE_REVISION = "2fe9bf00c8aae85fcedaf56eb3e2780dbfe8075e"
MAX_PIXELS = 14 * 14 * 4 * 1280


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def format_prompt(instruction: str) -> str:
    return (
        f"In this UI screenshot, I want to perform the command '{instruction}'.\n"
        "Please provide the action to perform (enumerate in ['click', 'open_app', 'scroll', 'navigate_back', 'input_text]')"
        "and the coordinate where the cursor is moved to(integer) if click is performed.\n"
        "Output the thinking process in <think> </think> and final answer in <answer> </answer> tags."
        "The output answer format should be as follows:\n"
        "<think> ... </think> <answer>[{'action': enum['click', 'open_app', 'scroll', 'navigate_back', 'input_text], 'coordinate': [x, y]}]</answer>\n"
        "Please strictly follow the format."
    )


def extract_action(content: str):
    answer = re.search(r"<answer>(.*?)</answer>", content, re.DOTALL)
    if not answer:
        return None
    match = re.search(r"'action':\s*'(\w+)'", answer.group(1).strip())
    return match.group(1) if match else None


def extract_coordinate(content: str) -> list[int]:
    answer = re.search(r"<answer>(.*?)</answer>", content, re.DOTALL)
    if answer:
        match = re.search(r"\{.*\[(\d+),\s*(\d+)]\s*.*\}", answer.group(1).strip())
        if match:
            return [int(match.group(1)), int(match.group(2))]
    return [0, 0, 0, 0]


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines()]


def resolve_image(name: str, prepared_images: Path, recovered_images: Path) -> Path:
    canonical = name.replace("-screenshot_", "_screenshot_")
    prepared = prepared_images / canonical
    if prepared.is_file():
        return prepared
    recovered = recovered_images / canonical
    if recovered.is_file():
        return recovered
    raise FileNotFoundError(canonical)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--source-json", type=Path, required=True)
    parser.add_argument("--prepared-images", type=Path, required=True)
    parser.add_argument("--recovered-images", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-shards", type=int, default=4)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--kv-cache-memory-bytes", type=int, default=2 * 1024**3)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("shard-index must be in [0, num-shards)")
    source = json.loads(args.source_json.read_text())
    if len(source) != 7868:
        raise ValueError(f"expected 7868 selected AndroidControl rows, found {len(source)}")
    indices = list(range(args.shard_index, len(source), args.num_shards))
    if args.limit is not None:
        indices = indices[: args.limit]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists() and not args.resume:
        raise FileExistsError(f"output exists without --resume: {args.output}")
    existing = read_jsonl(args.output) if args.resume else []
    completed = {row["index"] for row in existing}
    if len(completed) != len(existing):
        raise ValueError("duplicate existing indices")

    processor = AutoProcessor.from_pretrained(
        args.model_dir.resolve(),
        min_pixels=3136,
        max_pixels=MAX_PIXELS,
        size={"shortest_edge": 3136, "longest_edge": MAX_PIXELS},
        use_fast=False,
    )
    model = LLM(
        model=str(args.model_dir.resolve()),
        tensor_parallel_size=1,
        dtype="bfloat16",
        max_model_len=8192,
        gpu_memory_utilization=0.65,
        kv_cache_memory_bytes=args.kv_cache_memory_bytes,
        limit_mm_per_prompt={"image": 1},
        enforce_eager=True,
        trust_remote_code=True,
    )
    sampling = SamplingParams(
        temperature=0.1,
        top_k=1,
        top_p=0.001,
        repetition_penalty=1.05,
        max_tokens=1024,
        skip_special_tokens=True,
    )
    pending = [index for index in indices if index not in completed]
    with args.output.open("a", buffering=1) as output:
        for start in range(0, len(pending), args.batch_size):
            batch_indices = pending[start : start + args.batch_size]
            requests = []
            metadata = []
            for index in batch_indices:
                row = source[index]
                image_path = resolve_image(
                    row["image"], args.prepared_images, args.recovered_images
                )
                image_bytes = image_path.read_bytes()
                image = Image.open(image_path).convert("RGB")
                prompt_text = format_prompt(row["task"])
                query = "<image>\n" + prompt_text
                messages = [{
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": query},
                    ],
                }]
                model_prompt = processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                image_inputs, _, video_kwargs = process_vision_info(
                    messages, return_video_kwargs=True
                )
                inputs = processor(
                    text=[model_prompt], images=image_inputs, padding=True, return_tensors="pt"
                )
                grid = inputs["image_grid_thw"][0]
                patch_size = processor.image_processor.patch_size
                resized = [int(grid[2] * patch_size), int(grid[1] * patch_size)]
                requests.append({
                    "prompt": model_prompt,
                    "multi_modal_data": {"image": image_inputs},
                    "mm_processor_kwargs": {
                        **video_kwargs,
                        "min_pixels": 3136,
                        "max_pixels": MAX_PIXELS,
                    },
                })
                metadata.append({
                    "image_path": str(image_path),
                    "image_sha256": sha256_bytes(image_bytes),
                    "image_size": list(image.size),
                    "resized_image_size": resized,
                    "prompt_sha256": sha256_bytes(prompt_text.encode()),
                    "model_prompt_sha256": sha256_bytes(model_prompt.encode()),
                })
            responses = model.generate(requests, sampling_params=sampling, use_tqdm=False)
            for index, meta, response in zip(batch_indices, metadata, responses):
                source_row = source[index]
                generated = response.outputs[0].text
                coordinate = extract_coordinate(generated)
                scaled = [
                    int(coordinate[0] * 1080 / 644),
                    int(coordinate[1] * 2400 / 1484),
                ]
                result = {
                    "index": index,
                    "image": source_row["image"],
                    "task": source_row["task"],
                    "gt": source_row["gt"],
                    **meta,
                    "response": generated,
                    "pred_action": extract_action(generated),
                    "pred_coordinate_resized": coordinate,
                    "pred_coordinate_original": scaled,
                    "model_name": MODEL_NAME,
                    "model_revision": MODEL_REVISION,
                    "source_revision": SOURCE_REVISION,
                    "prompt_contract": "released_test_androidcontrol",
                    "coordinate_contract": "observed_672x1484_input_released_eval_ac_644x1484_scale",
                    "generation": "temperature_0.1_top_k_1_top_p_0.001_repetition_1.05_max_1024",
                    "tensor_parallel_size": 1,
                    "kv_cache_memory_bytes": args.kv_cache_memory_bytes,
                    "num_shards": args.num_shards,
                    "shard_index": args.shard_index,
                }
                output.write(json.dumps(result, ensure_ascii=True) + "\n")
            output.flush()
            os.fsync(output.fileno())


if __name__ == "__main__":
    main()