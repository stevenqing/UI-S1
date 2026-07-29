import argparse
import json
import os
from pathlib import Path

from PIL import Image
from transformers import AutoProcessor
from vllm import LLM, SamplingParams

from common import MODEL_REVISIONS, expected_answer, format_prompt, prompt_sha256, read_json, read_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--model-name", choices=MODEL_REVISIONS, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--kv-cache-memory-bytes", type=int, default=4 * 1024**3)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("shard-index must be in [0, num-shards)")
    metadata = read_json(args.metadata)
    if len(metadata) != 2080:
        raise ValueError(f"expected 2080 prepared rows, found {len(metadata)}")
    indices = [index for index in range(len(metadata)) if index % args.num_shards == args.shard_index]
    if args.limit is not None:
        indices = indices[: args.limit]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "predictions.jsonl"
    if output_path.exists() and not args.resume:
        raise FileExistsError(f"output exists without --resume: {output_path}")
    existing = read_jsonl(output_path) if args.resume else []
    completed = {row["index"] for row in existing}
    if len(completed) != len(existing):
        raise ValueError("duplicate indices in existing predictions")

    processor = AutoProcessor.from_pretrained(
        args.model_dir.resolve(),
        size={"shortest_edge": 3136, "longest_edge": 2116800},
        min_pixels=3136,
        max_pixels=2116800,
        use_fast=False,
    )
    model = LLM(
        model=str(args.model_dir.resolve()),
        tensor_parallel_size=args.tensor_parallel_size,
        dtype="bfloat16",
        max_model_len=8192,
        gpu_memory_utilization=0.65,
        kv_cache_memory_bytes=args.kv_cache_memory_bytes,
        limit_mm_per_prompt={"image": 1},
        trust_remote_code=True,
        enforce_eager=args.enforce_eager,
    )
    sampling = SamplingParams(temperature=0, frequency_penalty=1, max_tokens=128)

    pending = [index for index in indices if index not in completed]
    with output_path.open("a", buffering=1) as output:
        for start in range(0, len(pending), args.batch_size):
            batch_indices = pending[start : start + args.batch_size]
            requests = []
            for index in batch_indices:
                sample = metadata[index]
                image_path = args.image_root / sample["img_url"]
                if not image_path.is_file():
                    raise FileNotFoundError(image_path)
                image = Image.open(image_path).convert("RGB")
                messages = [{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": format_prompt(sample)},
                        {"type": "image"},
                    ],
                }]
                prompt = processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                requests.append({"prompt": prompt, "multi_modal_data": {"image": image}})
            responses = model.generate(requests, sampling, use_tqdm=False)
            for index, response in zip(batch_indices, responses):
                sample = metadata[index]
                row = {
                    "index": index,
                    "annot_id": sample["annot_id"],
                    "action_uid": sample["action_uid"],
                    "split": sample["split"],
                    "image": sample["img_url"],
                    "answer": expected_answer(sample),
                    "bbox": sample["step"]["bbox"],
                    "image_size": sample["img_size"],
                    "response": response.outputs[0].text,
                    "prompt_sha256": prompt_sha256(sample),
                    "model_name": args.model_name,
                    "model_revision": MODEL_REVISIONS[args.model_name],
                    "prompt_contract": "official_v1_computer_use_single_round",
                    "coordinate_space": "point_0_1000",
                    "generation": "greedy_frequency_penalty_1",
                    "max_new_tokens": 128,
                    "tensor_parallel_size": args.tensor_parallel_size,
                    "kv_cache_memory_bytes": args.kv_cache_memory_bytes,
                    "enforce_eager": args.enforce_eager,
                    "shard_index": args.shard_index,
                    "num_shards": args.num_shards,
                }
                output.write(json.dumps(row, ensure_ascii=True) + "\n")
                output.flush()
                os.fsync(output.fileno())


if __name__ == "__main__":
    main()