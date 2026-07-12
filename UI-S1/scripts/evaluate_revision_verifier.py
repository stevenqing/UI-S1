#!/usr/bin/env python3
"""Evaluate a multimodal revision verifier LoRA on episode-disjoint packets."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
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

DECISIONS = ("keep_student", "use_revision", "replan")


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


def parse_decision(text: str) -> tuple[str | None, str]:
    candidates = [text.strip()]
    candidates.extend(re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.S | re.I))
    start, end = text.find("{"), text.rfind("}")
    if start >= 0 and end > start:
        candidates.append(text[start : end + 1])
    for candidate in candidates:
        try:
            payload = json.loads(candidate)
        except Exception:
            continue
        decision = str(payload.get("decision") or "").strip().lower()
        if decision in DECISIONS:
            return decision, "json"
    lowered = text.lower()
    found = [decision for decision in DECISIONS if decision in lowered]
    if len(found) == 1:
        return found[0], "lenient_text"
    return None, "invalid"


def metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    confusion = {truth: {pred: 0 for pred in (*DECISIONS, "invalid")} for truth in DECISIONS}
    for row in rows:
        truth = str(row["target_decision"])
        pred = str(row.get("predicted_decision") or "invalid")
        confusion[truth][pred] += 1
    per_class = {}
    f1_values = []
    for decision in DECISIONS:
        tp = confusion[decision][decision]
        fp = sum(confusion[truth][decision] for truth in DECISIONS if truth != decision)
        fn = sum(confusion[decision][pred] for pred in (*DECISIONS, "invalid") if pred != decision)
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1 = 2 * precision * recall / max(1e-12, precision + recall)
        per_class[decision] = {"precision": precision, "recall": recall, "f1": f1, "support": sum(confusion[decision].values())}
        f1_values.append(f1)
    valid = [row for row in rows if row.get("predicted_decision") in DECISIONS]
    committed = [row for row in valid if row["predicted_decision"] != "replan"]
    committed_correct = []
    fallback_correct = []
    unsafe_overwrites = 0
    for row in rows:
        decision = row.get("predicted_decision")
        if decision == "use_revision":
            committed_correct.append(int(bool(row["revision_correct"])))
            fallback_correct.append(int(bool(row["revision_correct"])))
            unsafe_overwrites += int(bool(row["student_correct"]) and not bool(row["revision_correct"]))
        elif decision == "keep_student":
            committed_correct.append(int(bool(row["student_correct"])))
            fallback_correct.append(int(bool(row["student_correct"])))
        else:
            fallback_correct.append(int(bool(row["student_correct"])))
    accuracy = sum(str(row.get("predicted_decision")) == str(row["target_decision"]) for row in rows) / len(rows)
    return {
        "rows": len(rows),
        "accuracy": accuracy,
        "macro_f1": sum(f1_values) / len(f1_values),
        "invalid_rate": sum(row.get("predicted_decision") not in DECISIONS for row in rows) / len(rows),
        "per_class": per_class,
        "confusion": confusion,
        "commit_rate": len(committed) / len(rows),
        "committed_action_accuracy": sum(committed_correct) / max(1, len(committed_correct)),
        "fallback_student_accuracy": sum(fallback_correct) / len(fallback_correct),
        "student_baseline_accuracy": sum(bool(row["student_correct"]) for row in rows) / len(rows),
        "unsafe_overwrite_rate": unsafe_overwrites / len(rows),
        "predicted_decisions": dict(Counter(str(row.get("predicted_decision") or "invalid") for row in rows)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-path", default="checkpoints/gui360-fullparam-sft-step250")
    parser.add_argument("--adapter-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--image-max-pixels", type=int, default=200704)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    args = parser.parse_args()

    from transformers import AutoConfig, AutoProcessor, Qwen2_5_VLForConditionalGeneration

    all_rows = read_jsonl(Path(args.input))
    rows = [row for idx, row in enumerate(all_rows) if idx % args.shard_count == args.shard_index]
    if not rows:
        raise ValueError("empty verifier shard")
    device = torch.device(args.device)
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    processor.tokenizer.padding_side = "left"
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    processor.image_processor.max_pixels = args.image_max_pixels
    config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
    V15TrajectoryGSPOTrainer._patch_legacy_mrope_config(config)
    base = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path, config=config, torch_dtype=torch.bfloat16, trust_remote_code=True
    )
    model = PeftModel.from_pretrained(base, args.adapter_dir).to(device).eval()

    outputs = []
    for start in tqdm(range(0, len(rows), args.batch_size), desc=f"verifier:{args.shard_index}"):
        batch = rows[start : start + args.batch_size]
        messages = [
            [{"role": "user", "content": [
                {"type": "image", "image": row["image"]},
                {"type": "text", "text": str(row["prompt"]).removeprefix("<image>\n")},
            ]}]
            for row in batch
        ]
        prompts = [processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True) for message in messages]
        images = [Image.open(row["image"]).convert("RGB") for row in batch]
        inputs = processor(text=prompts, images=images, return_tensors="pt", padding=True)
        inputs = {name: value.to(device) for name, value in inputs.items()}
        prompt_width = int(inputs["input_ids"].shape[1])
        with torch.inference_mode(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            generated = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False, eos_token_id=processor.tokenizer.eos_token_id)
        for idx, row in enumerate(batch):
            text = processor.tokenizer.decode(generated[idx, prompt_width:], skip_special_tokens=True)
            decision, parser_mode = parse_decision(text)
            outputs.append({
                "sample_id": row["sample_id"],
                "episode_id": row["episode_id"],
                "correction_id": row["correction_id"],
                "step_idx": row["step_idx"],
                "target_decision": row["decision"],
                "predicted_decision": decision,
                "parser_mode": parser_mode,
                "student_correct": bool(row["student_correct"]),
                "revision_correct": bool(row["revision_correct"]),
                "raw_output": text,
            })
        torch.cuda.empty_cache()

    output = Path(args.output)
    write_jsonl(output, outputs)
    summary = {
        **metrics(outputs),
        "input": args.input,
        "input_sha256": sha256(Path(args.input)),
        "model_path": args.model_path,
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
