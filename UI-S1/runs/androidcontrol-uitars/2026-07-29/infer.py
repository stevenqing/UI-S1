import argparse
import hashlib
import json
import os
from pathlib import Path

from PIL import Image
from transformers import AutoProcessor
from vllm import LLM, SamplingParams

from common import MODEL_REVISIONS, format_prompt, read_jsonl


GENERATION = {
    "temperature": 0,
    "frequency_penalty": 1,
    "max_tokens": 128,
    "seed": 0,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--model-name", choices=MODEL_REVISIONS, required=True)
    parser.add_argument("--setting", choices=("low", "high"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("shard-index must be in [0, num-shards)")
    rows = read_jsonl(args.data)
    if len(rows) != 7708:
        raise ValueError(f"expected 7708 prepared rows, found {len(rows)}")
    indices = list(range(args.shard_index, len(rows), args.num_shards))
    if args.limit is not None:
        indices = indices[: args.limit]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "predictions.jsonl"
    if output_path.exists() and not args.resume:
        raise FileExistsError(output_path)
    existing = read_jsonl(output_path) if args.resume else []
    completed = {row["identity"] for row in existing}
    if len(completed) != len(existing):
        raise ValueError("duplicate identities in resume artifact")

    processor = AutoProcessor.from_pretrained(
        args.model_dir.resolve(),
        size={"shortest_edge": 3136, "longest_edge": 2116800},
        min_pixels=3136,
        max_pixels=2116800,
        use_fast=False,
    )
    engine = LLM(
        model=str(args.model_dir.resolve()),
        dtype="bfloat16",
        tensor_parallel_size=1,
        max_model_len=8192,
        gpu_memory_utilization=0.65,
        kv_cache_memory_bytes=4 * 1024**3,
        limit_mm_per_prompt={"image": 1},
    )
    sampling = SamplingParams(**GENERATION)
    pending = [index for index in indices if rows[index]["identity"] not in completed]
    with output_path.open("a", buffering=1) as output:
        for start in range(0, len(pending), args.batch_size):
            batch_indices = pending[start : start + args.batch_size]
            requests = []
            hashes = []
            for index in batch_indices:
                row = rows[index]
                prompt = format_prompt(row, args.setting)
                messages = [{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image"},
                    ],
                }]
                serialized = processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                image_path = args.image_root / row["image"]
                if not image_path.is_file():
                    raise FileNotFoundError(image_path)
                requests.append({
                    "prompt": serialized,
                    "multi_modal_data": {"image": Image.open(image_path).convert("RGB")},
                })
                hashes.append(hashlib.sha256(serialized.encode()).hexdigest())
            responses = engine.generate(requests, sampling, use_tqdm=False)
            for index, prompt_hash, response in zip(batch_indices, hashes, responses):
                source = rows[index]
                artifact = {
                    "index": index,
                    "identity": source["identity"],
                    "episode_id": source["episode_id"],
                    "step_id": source["step_id"],
                    "gt_action": source["gt_action"],
                    "response": response.outputs[0].text,
                    "prompt_sha256": prompt_hash,
                    "model_name": args.model_name,
                    "model_revision": MODEL_REVISIONS[args.model_name],
                    "prompt_contract": "official_v1_mobile_use_with_androidcontrol_wait",
                    "coordinate_space": "point_0_1000",
                    "generation": GENERATION,
                    "setting": args.setting,
                    "shard_index": args.shard_index,
                    "num_shards": args.num_shards,
                }
                output.write(json.dumps(artifact, ensure_ascii=True) + "\n")
                output.flush()
                os.fsync(output.fileno())


if __name__ == "__main__":
    main()