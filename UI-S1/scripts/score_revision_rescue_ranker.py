#!/usr/bin/env python3
"""Score YES versus NO conditional likelihood for a multimodal rescue ranker."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from PIL import Image
from peft import PeftModel
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from v15_gui_360.train_trajectory_gspo import V15TrajectoryGSPOTrainer  # noqa: E402

CANDIDATES = ("YES", "NO")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def average_ranks(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        rank = (start + 1 + end) / 2.0
        for idx in order[start:end]:
            ranks[idx] = rank
        start = end
    return ranks


def auc(labels: Sequence[int], scores: Sequence[float]) -> float | None:
    positive = sum(labels); negative = len(labels) - positive
    if not positive or not negative:
        return None
    ranks = average_ranks(scores)
    rank_sum = sum(rank for rank, label in zip(ranks, labels) if label)
    return (rank_sum - positive * (positive + 1) / 2) / (positive * negative)


def average_precision(labels: Sequence[int], scores: Sequence[float]) -> float | None:
    positive = sum(labels)
    if not positive:
        return None
    order = sorted(range(len(scores)), key=lambda idx: scores[idx], reverse=True)
    hits = 0; total = 0.0
    for rank, idx in enumerate(order, 1):
        if labels[idx]:
            hits += 1; total += hits / rank
    return total / positive


def differing_spans(left: Sequence[int], right: Sequence[int]) -> tuple[int, int, int]:
    """Return common-prefix end and candidate-specific ends before common suffix."""
    prefix = 0
    while prefix < min(len(left), len(right)) and left[prefix] == right[prefix]:
        prefix += 1
    suffix = 0
    max_suffix = min(len(left), len(right)) - prefix
    while suffix < max_suffix and left[len(left) - suffix - 1] == right[len(right) - suffix - 1]:
        suffix += 1
    left_end = len(left) - suffix
    right_end = len(right) - suffix
    if prefix >= left_end or prefix >= right_end:
        raise ValueError("YES/NO tokenizations have no candidate-specific span")
    return prefix, left_end, right_end


def build_messages(row: Mapping[str, Any], answer: str | None = None) -> list[dict[str, Any]]:
    user = {
        "role": "user",
        "content": [
            {"type": "image", "image": str(row["image"])},
            {"type": "text", "text": str(row["prompt"]).removeprefix("<image>\n")},
        ],
    }
    messages = [user]
    if answer is not None:
        messages.append({"role": "assistant", "content": [{"type": "text", "text": answer}]})
    return messages


def score_batch(model: Any, processor: Any, device: torch.device, rows: Sequence[Mapping[str, Any]], max_length: int) -> list[dict[str, float]]:
    expanded = [(row, answer) for row in rows for answer in CANDIDATES]
    full_texts = [processor.apply_chat_template(build_messages(row, answer), tokenize=False, add_generation_prompt=False) for row, answer in expanded]
    images = [Image.open(row["image"]).convert("RGB") for row, _ in expanded]
    full_inputs = processor(text=full_texts, images=images, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
    full_lengths = full_inputs["attention_mask"].sum(dim=1).tolist()
    if any(int(length) >= max_length for length in full_lengths):
        raise ValueError(f"ranker input reached max_length={max_length}; answer span may be truncated")
    full_inputs = {name: value.to(device) for name, value in full_inputs.items()}
    with torch.inference_mode(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        logits = model(**full_inputs).logits.float()
    log_probs = torch.log_softmax(logits, dim=-1)
    full_ids = full_inputs["input_ids"]
    output = []
    for pair_idx in range(len(rows)):
        yes_idx = 2 * pair_idx
        no_idx = yes_idx + 1
        yes_ids = full_ids[yes_idx, : int(full_lengths[yes_idx])].tolist()
        no_ids = full_ids[no_idx, : int(full_lengths[no_idx])].tolist()
        start, yes_end, no_end = differing_spans(yes_ids, no_ids)
        yes = sum(
            float(log_probs[yes_idx, position - 1, yes_ids[position]].item())
            for position in range(start, yes_end)
        )
        no = sum(
            float(log_probs[no_idx, position - 1, no_ids[position]].item())
            for position in range(start, no_end)
        )
        probability = 1.0 / (1.0 + math.exp(max(-60.0, min(60.0, no - yes))))
        output.append({
            "logp_yes": yes,
            "logp_no": no,
            "score": probability,
            "candidate_start": start,
            "yes_token_count": yes_end - start,
            "no_token_count": no_end - start,
        })
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-path", default="checkpoints/gui360-fullparam-sft-step250")
    parser.add_argument("--adapter-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--image-max-pixels", type=int, default=200704)
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--max-rows", type=int, default=0)
    args = parser.parse_args()

    from transformers import AutoConfig, AutoProcessor, Qwen2_5_VLForConditionalGeneration

    all_rows = read_jsonl(Path(args.input))
    if args.max_rows > 0:
        all_rows = all_rows[: args.max_rows]
    rows = [row for idx, row in enumerate(all_rows) if idx % args.shard_count == args.shard_index]
    if not rows:
        raise ValueError("empty score shard")
    device = torch.device(args.device)
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    processor.tokenizer.padding_side = "right"
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    processor.image_processor.max_pixels = args.image_max_pixels
    config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
    V15TrajectoryGSPOTrainer._patch_legacy_mrope_config(config)
    base = Qwen2_5_VLForConditionalGeneration.from_pretrained(args.model_path, config=config, torch_dtype=torch.bfloat16, trust_remote_code=True)
    model = PeftModel.from_pretrained(base, args.adapter_dir).to(device).eval()

    outputs = []
    for start in tqdm(range(0, len(rows), args.batch_size), desc=f"ranker:{args.shard_index}"):
        batch = rows[start : start + args.batch_size]
        scores = score_batch(model, processor, device, batch, args.max_length)
        for row, score in zip(batch, scores):
            outputs.append({
                "sample_id": row["sample_id"],
                "episode_id": row["episode_id"],
                "correction_id": row["correction_id"],
                "step_idx": row["step_idx"],
                "label": int(row["label"]),
                "utility_outcome": row["utility_outcome"],
                **score,
            })
        torch.cuda.empty_cache()
    output = Path(args.output)
    write_jsonl(output, outputs)
    labels = [int(row["label"]) for row in outputs]; scores = [float(row["score"]) for row in outputs]
    summary = {
        "rows": len(outputs),
        "positive_rate": sum(labels) / len(labels),
        "roc_auc": auc(labels, scores),
        "average_precision": average_precision(labels, scores),
        "input": args.input,
        "input_sha256": sha256(Path(args.input)),
        "adapter_dir": args.adapter_dir,
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "output": str(output),
        "output_sha256": sha256(output),
    }
    write_json(output.with_suffix(".summary.json"), summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
