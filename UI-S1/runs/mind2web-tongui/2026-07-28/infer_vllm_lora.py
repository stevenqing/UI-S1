import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest


def load_completed(path: Path) -> set[int]:
    if not path.exists():
        return set()
    with path.open() as handle:
        indices = [json.loads(line)["index"] for line in handle if line.strip()]
    if len(indices) != len(set(indices)):
        raise ValueError("duplicate indices in resume artifact")
    return set(indices)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--lora-path", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--repo-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--tensor-parallel-size", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    sys.path.insert(0, str(args.repo_dir.resolve()))
    from tongui.data.dset_mind2web import Mind2WebDataset

    processor = AutoProcessor.from_pretrained(
        args.base_model.resolve(),
        min_pixels=256 * 28 * 28,
        max_pixels=1344 * 28 * 28,
        model_max_length=8196,
        use_fast=False,
    )
    dataset = Mind2WebDataset(
        str(args.data_root.resolve()),
        "Mind2Web",
        "hf_test_task_with_thoughts",
        processor,
        inference=True,
        args_dict={"num_history": 2, "interleaved_history": "vtvt", "version": "v2"},
    )
    if len(dataset) != 2080:
        raise ValueError(f"expected 2080 rows, found {len(dataset)}")
    indices = list(range(len(dataset)))
    if args.limit is not None:
        indices = indices[: args.limit]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "predictions.jsonl"
    if output_path.exists() and not args.resume:
        raise FileExistsError(output_path)
    completed = load_completed(output_path) if args.resume else set()
    pending = [index for index in indices if index not in completed]

    engine = LLM(
        model=str(args.base_model.resolve()),
        tensor_parallel_size=args.tensor_parallel_size,
        dtype="bfloat16",
        seed=42,
        max_model_len=8196,
        gpu_memory_utilization=0.65,
        kv_cache_memory_bytes=2 * 1024**3,
        limit_mm_per_prompt={"image": 3},
        enable_lora=True,
        max_lora_rank=64,
        max_loras=1,
    )
    sampling = SamplingParams(temperature=0, max_tokens=128, seed=42)
    lora_request = LoRARequest(
        lora_name="tongui-32b",
        lora_int_id=1,
        lora_path=str(args.lora_path.resolve()),
        base_model_name=str(args.base_model.resolve()),
    )

    with output_path.open("a", buffering=1) as output:
        for start in range(0, len(pending), args.batch_size):
            batch_indices = pending[start : start + args.batch_size]
            requests = []
            metadata = []
            input_hashes = []
            for index in batch_indices:
                data_dict, meta = dataset[index]
                source = meta["source"]
                prompt = processor.apply_chat_template(
                    source, tokenize=False, add_generation_prompt=True
                )
                image_inputs, _ = process_vision_info(source)
                image_data = image_inputs[0] if len(image_inputs) == 1 else image_inputs
                requests.append({
                    "prompt": prompt,
                    "multi_modal_data": {"image": image_data},
                })
                metadata.append(meta)
                input_hashes.append(hashlib.sha256(data_dict["input_ids"].numpy().tobytes()).hexdigest())
            responses = engine.generate(
                requests,
                sampling,
                lora_request=lora_request,
                use_tqdm=False,
            )
            for index, meta, input_hash, response in zip(
                batch_indices, metadata, input_hashes, responses
            ):
                row = {
                    "index": index,
                    "annot_id": meta["annot_id"],
                    "action_uid": meta["action_uid"],
                    "split": meta["split"],
                    "image": meta["img_url"],
                    "answer": meta["answer"],
                    "bbox": meta["step"]["bbox"],
                    "image_size": meta["img_size"],
                    "response": response.outputs[0].text,
                    "input_ids_sha256": input_hash,
                    "model_name": args.model_name,
                    "model_revision": args.model_revision,
                    "version": "v2",
                    "num_history": 2,
                    "interleaved_history": "vtvt",
                    "min_visual_tokens": 256,
                    "max_visual_tokens": 1344,
                    "attention_backend": "vllm-0.11.0-tp4-native-lora",
                    "generation": "greedy",
                    "shard_index": 0,
                    "num_shards": 1,
                }
                output.write(json.dumps(row, ensure_ascii=True) + "\n")
                output.flush()
                os.fsync(output.fileno())


if __name__ == "__main__":
    main()