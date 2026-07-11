#!/usr/bin/env python3
"""Prepare noisy global revisions for 6-GPU LLaMA-Factory full-parameter SFT."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from v13_gui_360.eval_gui360_template import SUPPORTED_ACTIONS, USER_PROMPT_TEMPLATE  # noqa: E402


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def write_json(path: Path, payload: Any, *, compact: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=None if compact else 2) + ("" if compact else "\n"), encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="outputs/multiagent_trajectory_revision/full_v1/noisy_global_corrections_train.jsonl")
    parser.add_argument("--input-summary", default="outputs/multiagent_trajectory_revision/full_v1/noisy_global_corrections_train.summary.json")
    parser.add_argument("--output-dir", default="outputs/multiagent_trajectory_revision/full_v1/llamafactory_data")
    parser.add_argument("--config-out", default="outputs/multiagent_trajectory_revision/full_v1/fullparam_6gpu.yaml")
    parser.add_argument("--model-path", default="checkpoints/gui360-fullparam-sft-step250")
    parser.add_argument("--model-output", default="outputs/multiagent_trajectory_revision/full_v1/fullparam_model")
    parser.add_argument("--deepspeed-config", default="train_GUI_360/llamafactory/ds_z3_config.json")
    parser.add_argument("--pad-to-multiple", type=int, default=48)
    parser.add_argument("--learning-rate", type=float, default=6e-6)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--image-max-pixels", type=int, default=200704)
    args = parser.parse_args()

    input_path = Path(args.input)
    source_rows = read_jsonl(input_path)
    source_summary = read_json(Path(args.input_summary))
    if not source_rows or source_summary.get("semantic_quality_filter_used") is not False:
        raise ValueError("noisy source is empty or semantic filtering was enabled")
    if source_summary.get("selection_uses_matcher") is not False:
        raise ValueError("matcher-selected labels are not allowed in this arm")
    if source_summary.get("output_sha256") != sha256(input_path):
        raise ValueError("noisy source hash does not match its summary")
    if any(not str(row.get("image") or "").startswith("outputs/validation_2k/data/images/train/") for row in source_rows):
        raise ValueError("non-train image detected")
    sample_ids = [str(row.get("sample_id")) for row in source_rows if not row.get("padding_repeat")]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("duplicate unique sample IDs")

    examples = []
    for row in source_rows:
        history = list(row.get("history") or [])
        history_text = "\n".join(history) if history else "None"
        prompt = "<image>\n" + USER_PROMPT_TEMPLATE.format(
            instruction=str(row.get("goal") or ""),
            history=history_text,
            actions=SUPPORTED_ACTIONS,
        )
        examples.append({
            "conversations": [
                {"from": "human", "value": prompt},
                {"from": "gpt", "value": str(row["target_text"])},
            ],
            "images": [str(Path(row["image"]).resolve())],
            "sample_id": str(row.get("sample_id")),
            "padding_repeat": bool(row.get("padding_repeat")),
        })
    unique_examples = len(examples)
    padding = (-len(examples)) % args.pad_to_multiple if args.pad_to_multiple > 0 else 0
    for index in range(padding):
        clone = dict(examples[index % unique_examples])
        clone["sample_id"] = f"{clone['sample_id']}:fullparam-padding-{index}"
        clone["padding_repeat"] = True
        examples.append(clone)

    out_dir = Path(args.output_dir)
    train_path = out_dir / "noisy_global_revision_train.json"
    write_json(train_path, examples, compact=True)
    dataset_info = {
        "multiagent_noisy_global_revision_full": {
            "file_name": train_path.name,
            "formatting": "sharegpt",
            "columns": {"messages": "conversations", "images": "images"},
            "tags": {
                "role_tag": "from",
                "content_tag": "value",
                "user_tag": "human",
                "assistant_tag": "gpt",
            },
        }
    }
    write_json(out_dir / "dataset_info.json", dataset_info)

    config_path = Path(args.config_out)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(f"""### model
model_name_or_path: {args.model_path}
image_max_pixels: {args.image_max_pixels}
video_max_pixels: 16384
trust_remote_code: true

### full-parameter method
stage: sft
do_train: true
finetuning_type: full
freeze_vision_tower: false
freeze_multi_modal_projector: false
freeze_language_model: false
deepspeed: {args.deepspeed_config}

### noisy train-only dataset
dataset: multiagent_noisy_global_revision_full
dataset_dir: {args.output_dir}
template: qwen2_vl
cutoff_len: 16384
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
run_name: multiagent_noisy_global_revision_fullparam_6gpu

### train
per_device_train_batch_size: 1
gradient_accumulation_steps: {args.gradient_accumulation_steps}
learning_rate: {args.learning_rate}
num_train_epochs: 1.0
lr_scheduler_type: cosine
warmup_ratio: 0.03
weight_decay: 0.1
max_grad_norm: 1.0
bf16: true
gradient_checkpointing: true
ddp_timeout: 180000000

### eval
eval_strategy: 'no'
""", encoding="utf-8")

    manifest = {
        "source": str(input_path),
        "source_sha256": sha256(input_path),
        "source_rows": len(source_rows),
        "sharegpt_unique_examples": unique_examples,
        "sharegpt_padding_examples": padding,
        "sharegpt_examples": len(examples),
        "pad_to_multiple": args.pad_to_multiple,
        "effective_batch": 6 * args.gradient_accumulation_steps,
        "optimizer_steps": len(examples) // (6 * args.gradient_accumulation_steps),
        "dataset_json": str(train_path),
        "dataset_json_sha256": sha256(train_path),
        "dataset_info": str(out_dir / "dataset_info.json"),
        "config": str(config_path),
        "model_input": args.model_path,
        "model_output": args.model_output,
        "finetuning_type": "full",
        "freeze_vision_tower": False,
        "freeze_multi_modal_projector": False,
        "freeze_language_model": False,
        "gpus": 6,
        "semantic_quality_filter_used": False,
        "matcher_used_for_selection": False,
    }
    write_json(out_dir / "preparation_manifest.json", manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
