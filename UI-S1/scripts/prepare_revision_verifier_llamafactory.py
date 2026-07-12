#!/usr/bin/env python3
"""Prepare multimodal revision-verifier data and LLaMA-Factory LoRA config."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def write_json(path: Path, payload: Any, compact: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=None if compact else 2) + ("" if compact else "\n"), encoding="utf-8")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="datasets/revision_verifier_agent_v1")
    parser.add_argument("--output-dir", default="outputs/multiagent_trajectory_revision/full_v1/revision_verifier")
    parser.add_argument("--model-path", default="checkpoints/gui360-fullparam-sft-step250")
    parser.add_argument("--model-output", default="outputs/multiagent_trajectory_revision/full_v1/revision_verifier/model")
    parser.add_argument("--gpus", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--image-max-pixels", type=int, default=200704)
    args = parser.parse_args()

    source_dir = Path(args.data_dir)
    source_summary = read_json(source_dir / "summary.json")
    if not source_summary.get("episode_disjoint") or not source_summary.get("features_exclude_gt_and_matcher"):
        raise ValueError("verifier data isolation/provenance check failed")
    out_dir = Path(args.output_dir)
    data_out = out_dir / "data"
    split_manifest = {}
    for split, source_name in (("train", "train_balanced.jsonl"), ("dev", "dev.jsonl"), ("test", "test.jsonl")):
        source_path = source_dir / source_name
        rows = read_jsonl(source_path)
        examples = []
        for row in rows:
            messages = row["messages"]
            if not messages or messages[0].get("from") != "human" or messages[-1].get("from") != "gpt":
                raise ValueError(f"invalid messages: {row.get('sample_id')}")
            image = Path(row["image"])
            if not image.is_file():
                raise FileNotFoundError(image)
            examples.append({
                "conversations": messages,
                "images": [str(image.resolve())],
                "sample_id": str(row["sample_id"]),
                "episode_id": str(row["episode_id"]),
                "decision": str(row["decision"]),
            })
        target = data_out / f"{split}.json"
        write_json(target, examples, compact=True)
        split_manifest[split] = {"rows": len(examples), "path": str(target), "sha256": sha256(target)}

    dataset_info = {}
    for split in ("train", "dev", "test"):
        dataset_info[f"revision_verifier_{split}"] = {
            "file_name": f"{split}.json",
            "formatting": "sharegpt",
            "columns": {"messages": "conversations", "images": "images"},
            "tags": {"role_tag": "from", "content_tag": "value", "user_tag": "human", "assistant_tag": "gpt"},
        }
    write_json(data_out / "dataset_info.json", dataset_info)

    effective_batch = args.gpus * args.gradient_accumulation_steps
    optimizer_steps = (split_manifest["train"]["rows"] + effective_batch - 1) // effective_batch
    config_path = out_dir / "train_lora.yaml"
    config_path.write_text(f"""### model
model_name_or_path: {args.model_path}
image_max_pixels: {args.image_max_pixels}
video_max_pixels: 16384
trust_remote_code: true

### method
stage: sft
do_train: true
finetuning_type: lora
lora_rank: 32
lora_alpha: 64
lora_dropout: 0.05
lora_target: q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj

### data
dataset: revision_verifier_train
dataset_dir: {data_out}
template: qwen2_vl
cutoff_len: 8192
preprocessing_num_workers: 16
dataloader_num_workers: 4

### output
output_dir: {args.model_output}
logging_steps: 10
save_strategy: 'no'
plot_loss: true
overwrite_output_dir: true
save_only_model: true
report_to: none
run_name: revision_verifier_lora

### train
per_device_train_batch_size: 1
gradient_accumulation_steps: {args.gradient_accumulation_steps}
learning_rate: {args.learning_rate}
num_train_epochs: 1.0
lr_scheduler_type: cosine
warmup_ratio: 0.03
weight_decay: 0.01
max_grad_norm: 1.0
bf16: true
gradient_checkpointing: true
ddp_timeout: 180000000

### eval
eval_strategy: 'no'
""", encoding="utf-8")
    manifest = {
        "version": "revision-verifier-llamafactory-v1",
        "source_summary": str(source_dir / "summary.json"),
        "source_summary_sha256": sha256(source_dir / "summary.json"),
        "splits": split_manifest,
        "dataset_info": str(data_out / "dataset_info.json"),
        "config": str(config_path),
        "model_input": args.model_path,
        "model_output": args.model_output,
        "gpus": args.gpus,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "effective_batch": effective_batch,
        "optimizer_steps": optimizer_steps,
        "finetuning_type": "lora",
        "features_exclude_gt_and_matcher": True,
        "labels_derived_from_frozen_matcher": True,
    }
    write_json(out_dir / "preparation_manifest.json", manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
