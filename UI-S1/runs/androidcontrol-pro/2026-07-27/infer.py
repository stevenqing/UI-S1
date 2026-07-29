import argparse
import hashlib
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from qwen_vl_utils import process_vision_info
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, Qwen2VLForConditionalGeneration


ROOT = Path(__file__).resolve().parent
MODEL_REVISION = "6c0135de0627db98533ac4b47ae71fa17cf21c48"
SAMPLE_PATH = ROOT.parent.parent / "androidcontrol/2026-07-27/repos/OS-Atlas/eval/data/ac_test.jsonl"
GENERATION_CONFIG = {
    "max_new_tokens": 128,
    "do_sample": True,
    "temperature": 0.01,
    "top_k": 1,
    "top_p": 0.001,
    "use_cache": True,
}


def prompt_prefix():
    sample = json.loads(SAMPLE_PATH.read_text().splitlines()[0])
    human = sample["conversations"][0]["value"]
    return human.split("Screenshot: <image>", 1)[0] + "Screenshot: "


def build_messages(row, prefix, setting):
    image_path = ROOT / row["image"]
    instruction = row["goal"]
    if setting == "low":
        instruction += f' You need to: {row["low_instruction_audit_only"]}'
    history_terminator = "\n" if row["history"] == "None" else "\n\n"
    suffix = f'\nTask: {instruction}\nHistory: \n{row["history"]}{history_terminator}'
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prefix},
                {
                    "type": "image",
                    "image": str(image_path),
                    "max_pixels": 1024 * 1024,
                },
                {"type": "text", "text": suffix},
            ],
        }
    ]


def read_jsonl(path):
    with path.open() as input_file:
        return [json.loads(line) for line in input_file if line.strip()]


def load_completed(path):
    identities = [row["identity"] for row in read_jsonl(path)]
    if len(identities) != len(set(identities)):
        raise ValueError(f"duplicate identities in resume file: {path}")
    return set(identities)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=ROOT / "data/prepared/ac_high.jsonl")
    parser.add_argument("--model", type=Path, default=ROOT / "models/OS-Atlas-Pro-7B")
    parser.add_argument("--model-family", choices=("qwen2", "qwen2.5"), default="qwen2")
    parser.add_argument("--model-name", default="OS-Atlas-Pro-7B")
    parser.add_argument("--model-revision", default=MODEL_REVISION)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--setting", choices=("high", "low"), default="high")
    args = parser.parse_args()

    if torch.cuda.device_count() != 1:
        raise RuntimeError("AndroidControl evaluation requires exactly one visible GPU")
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("shard-index must be in [0, num-shards)")
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    rows = read_jsonl(args.data)
    if args.limit is not None:
        rows = rows[: args.limit]
    rows = rows[args.shard_index :: args.num_shards]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = args.output_dir / "predictions.jsonl"
    if predictions_path.exists() and not args.resume:
        raise FileExistsError(f"output exists without --resume: {predictions_path}")
    completed = set()
    if args.resume and predictions_path.exists():
        completed = load_completed(predictions_path)
    mode = "a" if args.resume else "w"

    model_class = {
        "qwen2": Qwen2VLForConditionalGeneration,
        "qwen2.5": Qwen2_5_VLForConditionalGeneration,
    }[args.model_family]
    model = model_class.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        attn_implementation="sdpa",
    ).eval()
    processor = AutoProcessor.from_pretrained(args.model, use_fast=False)
    processor.tokenizer.padding_side = "left"
    prefix = prompt_prefix()
    pending = [row for row in rows if row["identity"] not in completed]

    with predictions_path.open(mode) as predictions_file:
        for start in tqdm(range(0, len(pending), args.batch_size)):
            batch = pending[start : start + args.batch_size]
            conversations = [build_messages(row, prefix, args.setting) for row in batch]
            texts = [
                processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
                for conversation in conversations
            ]
            prompt_hashes = [hashlib.sha256(text.encode()).hexdigest() for text in texts]
            image_inputs, video_inputs = process_vision_info(conversations)
            inputs = processor(
                text=texts,
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            ).to("cuda")
            with torch.inference_mode():
                generated_ids = model.generate(**inputs, **GENERATION_CONFIG)
            generated_ids = generated_ids[:, inputs.input_ids.shape[1] :]
            trimmed_ids = []
            for output_ids in generated_ids:
                end = len(output_ids)
                while end > 0 and output_ids[end - 1] == processor.tokenizer.pad_token_id:
                    end -= 1
                trimmed_ids.append(output_ids[:end])
            responses = [
                processor.decode(
                    output_ids,
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                )
                for output_ids in trimmed_ids
            ]
            for row, prompt_hash, response in zip(batch, prompt_hashes, responses):
                artifact = {
                    "identity": row["identity"],
                    "episode_id": row["episode_id"],
                    "step_id": row["step_id"],
                    "gt_action": row["gt_action"],
                    "response": response,
                    "prompt_sha256": prompt_hash,
                    "model_name": args.model_name,
                    "model_revision": args.model_revision,
                    "model_family": args.model_family,
                    "processor_use_fast": False,
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
        "model_family": args.model_family,
        "batch_size": args.batch_size,
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "setting": args.setting,
    }
    (args.output_dir / "inference_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()