import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import torch
import yaml
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


RUN_DIR = Path(__file__).resolve().parent
ROOT = RUN_DIR.parents[2]
VUS = ROOT / "runs/visual-utility-selector/2026-08-11"
sys.path.insert(0, str(VUS))

from vus_data import CANDIDATE_LABELS, sha256_file
from evidence_data import MODES, RAVEL_MAX_PIXELS, RAVEL_MIN_PIXELS, render_evidence


SYSTEM_PROMPT = (
    "You are a careful GUI action adjudicator. Compare all candidate evidence and answer with "
    "exactly one capital letter A through L."
)


def load_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def completed_keys(path):
    if not path.exists():
        return set()
    rows = load_jsonl(path)
    keys = [row["sample_key"] for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError(f"duplicate RAVEL output keys: {path}")
    return set(keys)


def label_token_ids(tokenizer):
    output = []
    for label in CANDIDATE_LABELS:
        values = tokenizer.encode(label, add_special_tokens=False)
        if len(values) != 1:
            raise ValueError(f"RAVEL label is not one token: {label}/{values}")
        output.append(values[0])
    return output


def prepare(record, mode, processor, seed):
    images, permutation, prompt, budget = render_evidence(record, mode, seed)
    message = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": [
            *[{"type": "image", "image": image} for image in images],
            {"type": "text", "text": prompt},
        ]},
    ]
    image_inputs, video_inputs = process_vision_info(message)
    text = processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
    inputs = processor(
        text=[text], images=image_inputs, videos=video_inputs,
        padding=True, return_tensors="pt",
    )
    grid = inputs["image_grid_thw"]
    actual_pixels = int(sum(int(t) * int(h) * int(w) * 16 * 16 for t, h, w in grid.tolist()))
    actual_tokens = int(sum(int(t) * int(h) * int(w) // 4 for t, h, w in grid.tolist()))
    if actual_pixels != budget["expected_total_processed_pixels"]:
        raise ValueError(f"RAVEL processor pixel mismatch: {actual_pixels}/{budget['expected_total_processed_pixels']}")
    actual_ratio = actual_pixels / budget["baseline_vus_pixels"]
    if actual_ratio > 1.02 + 1e-12:
        raise ValueError(f"RAVEL-K1 actual pixel ratio: {actual_ratio}")
    budget.update({
        "actual_processed_pixels": actual_pixels,
        "actual_visual_tokens": actual_tokens,
        "actual_pixel_ratio_vs_vus": actual_ratio,
        "image_grid_thw": grid.tolist(),
    })
    return inputs, permutation, prompt, budget


def score(model, inputs, token_ids):
    device = next(model.parameters()).device
    inputs = {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        logits = model(**inputs, use_cache=False).logits
    mask = inputs["attention_mask"].bool()
    position = mask.shape[1] - 1 - torch.argmax(mask.flip(dims=(1,)).to(torch.int64), dim=1)
    label_logits = logits[0, position[0], token_ids].float()
    probabilities = torch.softmax(label_logits, dim=-1)
    return label_logits.cpu().tolist(), probabilities.cpu().tolist()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--records", type=Path, default=VUS / "data/public_records.jsonl")
    parser.add_argument("--model-dir", type=Path, default=ROOT / "runs/ccm-h2h/2026-07-31/h3/models/Qwen3-VL-8B-Instruct")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-shards", type=int, default=8)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("invalid shard index")
    if not Path("/proc/2274").exists():
        raise RuntimeError("protected PID 2274 absent")
    config = yaml.safe_load((RUN_DIR / "configs/ravel_prereg.yaml").read_text())
    if config["status"] != "FROZEN_AFTER_CARE_A1_BEFORE_RAVEL_RESULTS":
        raise ValueError("RAVEL protocol is not frozen")
    rows = sorted(load_jsonl(args.records), key=lambda row: row["sample_key"])
    rows = rows[args.shard_index::args.num_shards]
    if args.limit is not None:
        rows = rows[:args.limit]
    if args.output.exists() and not args.resume:
        raise FileExistsError(args.output)
    done = completed_keys(args.output) if args.resume else set()
    rows = [row for row in rows if row["sample_key"] not in done]
    processor = AutoProcessor.from_pretrained(
        args.model_dir, min_pixels=RAVEL_MIN_PIXELS,
        max_pixels=RAVEL_MAX_PIXELS, use_fast=False,
    )
    token_ids = label_token_ids(processor.tokenizer)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_dir, torch_dtype=torch.bfloat16,
        attn_implementation="sdpa", low_cpu_mem_usage=True,
    ).to("cuda:0").eval()
    model_hash = sha256_file(args.model_dir / "model.safetensors.index.json")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with args.output.open("a", buffering=1) as output:
        for record in rows:
            inputs, permutation, prompt, budget = prepare(record, args.mode, processor, 20260811)
            logits, probabilities = score(model, inputs, token_ids)
            selected = int(max(range(12), key=probabilities.__getitem__))
            value = {
                "schema_version": 1,
                "sample_key": record["sample_key"],
                "benchmark": record["benchmark"],
                "arm": record["arm"],
                "row_id": record["row_id"],
                "fold": record["fold"],
                "group": record["group"],
                "mode": args.mode,
                "display_to_candidate": list(permutation),
                "selected_label": CANDIDATE_LABELS[selected],
                "selected_candidate_index": int(permutation[selected]),
                "label_logits": [float(item) for item in logits],
                "label_probabilities": [float(item) for item in probabilities],
                "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                "image_sha256": record["image_sha256"],
                "model_index_sha256": model_hash,
                "visual_budget": budget,
            }
            output.write(json.dumps(value, ensure_ascii=True, sort_keys=True) + "\n")
            output.flush()
            written += 1
            if written % 25 == 0:
                os.fsync(output.fileno())
                print(json.dumps({"mode": args.mode, "shard": args.shard_index, "written": written, "assigned": len(rows)}), flush=True)
        os.fsync(output.fileno())
    print(json.dumps({
        "status": "PASS", "mode": args.mode, "shard": args.shard_index,
        "written": written, "completed": len(completed_keys(args.output)),
        "model_index_sha256": model_hash,
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
