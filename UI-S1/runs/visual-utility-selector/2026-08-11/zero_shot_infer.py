import argparse
import hashlib
import json
import os
from pathlib import Path

import torch
import yaml
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


RUN_DIR = Path(__file__).resolve().parent
ROOT = RUN_DIR.parents[2]
from vus_data import (
    CANDIDATE_LABELS,
    build_candidate_prompt,
    deterministic_permutation,
    render_overlay,
    sha256_file,
)


SYSTEM_PROMPT = (
    "You are a careful GUI action adjudicator. Inspect the task and screenshot, compare every labeled "
    "candidate, and answer with exactly one capital letter A through L."
)


def load_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def completed_keys(path):
    if not path.exists():
        return set()
    rows = load_jsonl(path)
    keys = [row["sample_key"] for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError(f"duplicate zero-shot output keys: {path}")
    return set(keys)


def model_index_sha256(model_dir):
    index = model_dir / "model.safetensors.index.json"
    if not index.is_file():
        raise FileNotFoundError(index)
    return sha256_file(index)


def label_token_ids(tokenizer):
    output = []
    for label in CANDIDATE_LABELS:
        values = tokenizer.encode(label, add_special_tokens=False)
        if len(values) != 1:
            raise ValueError(f"VUS label is not one token: {label}/{values}")
        output.append(values[0])
    if len(set(output)) != len(output):
        raise ValueError("VUS label token collision")
    return output


def prepare_batch(records, processor, seed, max_edge):
    messages = []
    metadata = []
    for record in records:
        permutation = deterministic_permutation(record["sample_key"], 0, seed)
        overlay = render_overlay(record, permutation, max_edge=max_edge)
        prompt = build_candidate_prompt(record, permutation)
        message = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "image", "image": overlay},
                {"type": "text", "text": prompt},
            ]},
        ]
        messages.append(message)
        metadata.append((record, permutation, prompt))
    texts = [processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True) for message in messages]
    image_inputs = []
    video_inputs = []
    for message in messages:
        images, videos = process_vision_info(message)
        image_inputs.extend(images or [])
        video_inputs.extend(videos or [])
    inputs = processor(
        text=texts,
        images=image_inputs or None,
        videos=video_inputs or None,
        padding=True,
        return_tensors="pt",
    )
    return inputs, metadata


def score_batch(model, inputs, token_ids):
    device = next(model.parameters()).device
    inputs = {key: value.to(device, non_blocking=True) if hasattr(value, "to") else value for key, value in inputs.items()}
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        logits = model(**inputs, use_cache=False).logits
    mask = inputs["attention_mask"].bool()
    positions = mask.shape[1] - 1 - torch.argmax(mask.flip(dims=(1,)).to(torch.int64), dim=1)
    rows = torch.arange(logits.shape[0], device=logits.device)
    label_logits = logits[rows, positions][:, token_ids].float()
    probabilities = torch.softmax(label_logits, dim=-1)
    return label_logits.cpu().tolist(), probabilities.cpu().tolist()


def output_record(metadata, logits, probabilities, model_hash):
    record, permutation, prompt = metadata
    selected = int(max(range(len(probabilities)), key=probabilities.__getitem__))
    selected_candidate = permutation[selected]
    return {
        "schema_version": 1,
        "sample_key": record["sample_key"],
        "benchmark": record["benchmark"],
        "arm": record["arm"],
        "row_id": record["row_id"],
        "fold": record["fold"],
        "group": record["group"],
        "display_to_candidate": list(permutation),
        "selected_label": CANDIDATE_LABELS[selected],
        "selected_candidate_index": int(selected_candidate),
        "label_logits": [float(value) for value in logits],
        "label_probabilities": [float(value) for value in probabilities],
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "image_sha256": record["image_sha256"],
        "model_index_sha256": model_hash,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", type=Path, default=RUN_DIR / "data/public_records.jsonl")
    parser.add_argument("--model-dir", type=Path, default=ROOT / "runs/ccm-h2h/2026-07-31/h3/models/Qwen3-VL-8B-Instruct")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-shards", type=int, default=8)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-edge", type=int, default=1600)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("invalid shard index")
    if args.batch_size < 1 or args.max_edge < 224:
        raise ValueError("invalid batch or image size")
    if not Path("/proc/2274").exists():
        raise RuntimeError("protected PID 2274 is unexpectedly absent")
    config = yaml.safe_load((RUN_DIR / "configs/vus_prereg.yaml").read_text())
    if config["status"] != "FROZEN_BEFORE_VUS_RESULTS":
        raise ValueError("VUS protocol is not frozen")
    records = load_jsonl(args.records)
    records.sort(key=lambda row: row["sample_key"])
    records = records[args.shard_index::args.num_shards]
    if args.limit is not None:
        records = records[:args.limit]
    if args.output.exists() and not args.resume:
        raise FileExistsError(args.output)
    done = completed_keys(args.output) if args.resume else set()
    records = [record for record in records if record["sample_key"] not in done]
    processor = AutoProcessor.from_pretrained(
        args.model_dir,
        min_pixels=256 * 28 * 28,
        max_pixels=1280 * 28 * 28,
        use_fast=False,
    )
    token_ids = label_token_ids(processor.tokenizer)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_dir,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    ).to("cuda:0").eval()
    model_hash = model_index_sha256(args.model_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with args.output.open("a", buffering=1) as output:
        for start in range(0, len(records), args.batch_size):
            batch = records[start:start + args.batch_size]
            inputs, metadata = prepare_batch(batch, processor, config["seed"], args.max_edge)
            logits, probabilities = score_batch(model, inputs, token_ids)
            for values in zip(metadata, logits, probabilities):
                output.write(json.dumps(output_record(*values, model_hash), ensure_ascii=True, sort_keys=True) + "\n")
                written += 1
            output.flush()
            if written % 25 == 0:
                os.fsync(output.fileno())
                print(json.dumps({"shard": args.shard_index, "written": written, "assigned": len(records)}), flush=True)
        os.fsync(output.fileno())
    print(json.dumps({
        "status": "PASS",
        "shard": args.shard_index,
        "written_this_run": written,
        "completed": len(completed_keys(args.output)),
        "assigned_after_resume": len(records),
        "model_index_sha256": model_hash,
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
