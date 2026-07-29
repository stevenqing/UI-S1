import argparse
import hashlib
import json
import os
from pathlib import Path

from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor
from vllm import LLM, SamplingParams

from infer import build_messages, load_completed, prompt_prefix, read_jsonl


GENERATION_CONFIG = {
    "max_tokens": 128,
    "temperature": 0.01,
    "top_k": 1,
    "top_p": 0.001,
    "seed": 0,
    "skip_special_tokens": False,
    "spaces_between_special_tokens": False,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--setting", choices=("high", "low"), required=True)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.65)
    parser.add_argument("--kv-cache-memory-bytes", type=int)
    parser.add_argument("--indices", help="comma-separated global row indices")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("shard-index must be in [0, num-shards)")
    rows = read_jsonl(args.data)
    if args.indices:
        indices = [int(value) for value in args.indices.split(",")]
        if len(indices) != len(set(indices)) or any(index < 0 or index >= len(rows) for index in indices):
            raise ValueError("indices must be unique valid global row indices")
        rows = [rows[index] for index in indices]
    if args.limit is not None:
        rows = rows[: args.limit]
    rows = rows[args.shard_index :: args.num_shards]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = args.output_dir / "predictions.jsonl"
    if predictions_path.exists() and not args.resume:
        raise FileExistsError(f"output exists without --resume: {predictions_path}")
    completed = load_completed(predictions_path) if args.resume and predictions_path.exists() else set()
    pending = [row for row in rows if row["identity"] not in completed]

    processor = AutoProcessor.from_pretrained(args.model.resolve(), use_fast=False)
    prefix = prompt_prefix()
    engine = LLM(
        model=str(args.model.resolve()),
        dtype="bfloat16",
        seed=0,
        gpu_memory_utilization=args.gpu_memory_utilization,
        kv_cache_memory_bytes=args.kv_cache_memory_bytes,
        max_model_len=8192,
        limit_mm_per_prompt={"image": 1},
        mm_processor_kwargs={"max_pixels": 1024 * 1024},
    )
    sampling = SamplingParams(**GENERATION_CONFIG)

    mode = "a" if args.resume else "w"
    with predictions_path.open(mode, buffering=1) as predictions_file:
        for start in range(0, len(pending), args.batch_size):
            batch = pending[start : start + args.batch_size]
            conversations = [build_messages(row, prefix, args.setting) for row in batch]
            texts = [
                processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
                for conversation in conversations
            ]
            prompt_hashes = [hashlib.sha256(text.encode()).hexdigest() for text in texts]
            image_inputs, _ = process_vision_info(conversations)
            prompts = [
                {"prompt": text, "multi_modal_data": {"image": image}}
                for text, image in zip(texts, image_inputs)
            ]
            outputs = engine.generate(prompts, sampling, use_tqdm=False)
            for row, prompt_hash, output in zip(batch, prompt_hashes, outputs):
                artifact = {
                    "identity": row["identity"],
                    "episode_id": row["episode_id"],
                    "step_id": row["step_id"],
                    "gt_action": row["gt_action"],
                    "response": output.outputs[0].text,
                    "prompt_sha256": prompt_hash,
                    "model_name": args.model_name,
                    "model_revision": args.model_revision,
                    "model_family": "qwen2.5",
                    "processor_use_fast": False,
                    "backend": "vllm-0.11.0",
                    "generation": GENERATION_CONFIG,
                    "setting": args.setting,
                }
                predictions_file.write(json.dumps(artifact, ensure_ascii=False) + "\n")
                predictions_file.flush()
                os.fsync(predictions_file.fileno())

    artifacts = read_jsonl(predictions_path)
    summary = {
        "status": "COMPLETE" if len(artifacts) == len(rows) else "PARTIAL",
        "rows_requested": len(rows),
        "predictions": len(artifacts),
        "unique_identities": len({row["identity"] for row in artifacts}),
        "model_name": args.model_name,
        "model_revision": args.model_revision,
        "backend": "vllm-0.11.0",
        "batch_size": args.batch_size,
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "setting": args.setting,
    }
    (args.output_dir / "inference_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()