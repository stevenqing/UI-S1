import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import torch
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


RUN_DIR = Path(__file__).resolve().parent
ROOT = RUN_DIR.parents[2]
sys.path.insert(0, str(RUN_DIR))

from recovery_common import assert_protected_process, load_config as load_recovery_config
from selector_data import (
    LABELS, assert_selector_environment, audit_public_record, build_prompt,
    deterministic_permutation, load_config, load_jsonl, render_overlay,
    rendered_image_sha256,
)


SYSTEM_PROMPT = (
    "You are a careful GUI action adjudicator. Inspect the task and screenshot, "
    "compare all three labeled actions, and answer with exactly A, B, or C."
)


def completed_keys(path):
    if not path.exists():
        return set()
    rows = load_jsonl(path)
    keys = [row["sample_key"] for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError(f"TriVUS duplicate selector keys: {path}")
    return set(keys)


def label_token_ids(tokenizer):
    output = []
    for label in LABELS:
        values = tokenizer.encode(label, add_special_tokens=False)
        if len(values) != 1:
            raise ValueError(f"TriVUS label is not one token: {label}/{values}")
        output.append(values[0])
    if len(set(output)) != len(output):
        raise ValueError("TriVUS label token collision")
    return output


def prepare_record(record, processor, config):
    permutation = deterministic_permutation(record["sample_key"], config["seed"])
    overlay = render_overlay(record, permutation, config["processor"]["max_rendered_edge"])
    prompt = build_prompt(record, permutation)
    message = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": [
            {"type": "image", "image": overlay},
            {"type": "text", "text": prompt},
        ]},
    ]
    text = processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
    images, videos = process_vision_info(message)
    inputs = processor(
        text=[text], images=images or None, videos=videos or None,
        padding=True, return_tensors="pt",
    )
    return inputs, permutation, prompt, rendered_image_sha256(overlay)


def score_record(model, inputs, token_ids):
    device = next(model.parameters()).device
    inputs = {key: value.to(device, non_blocking=True) if hasattr(value, "to") else value for key, value in inputs.items()}
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        logits = model(**inputs, use_cache=False).logits
    mask = inputs["attention_mask"].bool()
    position = mask.shape[1] - 1 - torch.argmax(mask.flip(dims=(1,)).to(torch.int64), dim=1)
    label_logits = logits[0, position[0], token_ids].float()
    probabilities = torch.softmax(label_logits, dim=-1)
    return label_logits.cpu().tolist(), probabilities.cpu().tolist()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shard-index", type=int, choices=range(8), required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config = load_config()
    assert_selector_environment(config)
    assert_protected_process(load_recovery_config())
    if any((RUN_DIR / "data").glob("private*")):
        raise PermissionError("TriVUS selector inference sealed from private labels")
    records = load_jsonl(RUN_DIR / "data/public_records.jsonl")
    records.sort(key=lambda row: row["sample_key"])
    records = records[args.shard_index::config["inference"]["num_shards"]]
    for record in records:
        audit_public_record(record)
    if args.output.exists() and not args.resume:
        raise FileExistsError(args.output)
    done = completed_keys(args.output) if args.resume else set()
    pending = [record for record in records if record["sample_key"] not in done]
    model_dir = ROOT / config["model"]["path"]
    processor = AutoProcessor.from_pretrained(
        model_dir,
        min_pixels=config["processor"]["min_pixels"],
        max_pixels=config["processor"]["max_pixels"],
        use_fast=False,
    )
    token_ids = label_token_ids(processor.tokenizer)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_dir, torch_dtype=torch.bfloat16,
        attn_implementation=config["inference"]["attention"],
        low_cpu_mem_usage=True,
    ).to("cuda:0").eval()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with args.output.open("a", buffering=1) as handle:
        for record in pending:
            inputs, permutation, prompt, overlay_hash = prepare_record(record, processor, config)
            logits, probabilities = score_record(model, inputs, token_ids)
            selected = max(range(3), key=probabilities.__getitem__)
            output = {
                "schema_version": 1,
                "sample_key": record["sample_key"],
                "benchmark": record["benchmark"],
                "setting": record["setting"],
                "row_id": record["row_id"],
                "fold": record["fold"],
                "group": record["group"],
                "display_to_candidate": list(permutation),
                "selected_label": LABELS[selected],
                "selected_candidate_index": int(permutation[selected]),
                "label_logits": [float(value) for value in logits],
                "label_probabilities": [float(value) for value in probabilities],
                "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                "overlay_sha256": overlay_hash,
                "image_sha256": record["image_sha256"],
                "model_index_sha256": config["model"]["index_sha256"],
            }
            handle.write(json.dumps(output, ensure_ascii=True, sort_keys=True) + "\n")
            written += 1
            if written % config["inference"]["fsync_every_rows"] == 0:
                handle.flush()
                os.fsync(handle.fileno())
                print(json.dumps({"shard": args.shard_index, "written": written, "assigned": len(records)}), flush=True)
        handle.flush()
        os.fsync(handle.fileno())
    assert_protected_process(load_recovery_config())
    print(json.dumps({
        "status": "PASS", "shard": args.shard_index,
        "written_this_run": written, "completed": len(completed_keys(args.output)),
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()