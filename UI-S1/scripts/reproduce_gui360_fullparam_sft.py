#!/usr/bin/env python3
"""Reproduce the GUI-360 full-parameter SFT checkpoint setup.

The checkpoint used in this workspace is:

    checkpoints/gui360-fullparam-sft-step250

Its README says it was trained as full-parameter SFT from
Qwen/Qwen2.5-VL-7B-Instruct on Stevenshuqing/gui360-balanced, using
LLaMA-Factory + DeepSpeed ZeRO-3. This script recreates that pipeline in a
single Python entrypoint:

1. Convert `datasets/gui360-balanced` parquet episodes into LLaMA-Factory
   ShareGPT-style JSON files with extracted screenshot PNGs.
2. Write `dataset_info.json` entries for LLaMA-Factory.
3. Write the full-parameter SFT YAML matching the original config.
4. Optionally launch `llamafactory-cli train <yaml>`.

Example:

    python scripts/reproduce_gui360_fullparam_sft.py \
      --base-model checkpoints/Qwen2.5-VL-7B-Instruct \
      --prepare-data \
      --write-config

Then train with:

    llamafactory-cli train train_GUI_360/llamafactory/qwen25vl_gui360_balanced_full_sft_repro.yaml

For the original multi-node run, use the generated YAML with the same cluster
launcher/environment used for LLaMA-Factory ZeRO-3 jobs.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    import pandas as pd
except ImportError:  # pragma: no cover - exercised in lightweight/offline envs
    pd = None


DEFAULT_BALANCED_DIR = "datasets/gui360-balanced/data"
DEFAULT_DATA_DIR = "train_GUI_360/llamafactory/data"
DEFAULT_IMAGE_DIR = "train_GUI_360/llamafactory/data/gui360_balanced_images"
DEFAULT_CONFIG_PATH = "train_GUI_360/llamafactory/qwen25vl_gui360_balanced_full_sft_repro.yaml"
DEFAULT_DS_CONFIG = "train_GUI_360/llamafactory/ds_z3_config.json"
DEFAULT_OUTPUT_DIR = "train_GUI_360/llamafactory/output/gui360_balanced_full_sft_repro"
DEFAULT_BASE_MODEL = "checkpoints/Qwen2.5-VL-7B-Instruct"


def parquet_files(data_dir: Path, split: str) -> List[Path]:
    return sorted(data_dir.glob(f"{split}-*.parquet"))


def _require_pandas() -> Any:
    if pd is None:
        raise ImportError("pandas is required to read GUI-360 balanced parquet files")
    return pd


def count_rows(data_dir: Path, split: str) -> int:
    pandas = _require_pandas()
    return sum(len(pandas.read_parquet(path)) for path in parquet_files(data_dir, split))


def load_rows(data_dir: Path, split: str, max_episodes: int) -> Iterable[Dict[str, Any]]:
    files = parquet_files(data_dir, split)
    if not files:
        raise FileNotFoundError(f"no {split}-*.parquet files under {data_dir}")
    pandas = _require_pandas()
    seen = 0
    for path in files:
        frame = pandas.read_parquet(path)
        for row in frame.to_dict(orient="records"):
            yield row
            seen += 1
            if max_episodes > 0 and seen >= max_episodes:
                return


def image_bytes_from_cell(cell: Any) -> Optional[bytes]:
    if cell is None:
        return None
    if isinstance(cell, dict):
        data = cell.get("bytes")
        return data if isinstance(data, (bytes, bytearray)) else None
    if hasattr(cell, "get"):
        try:
            data = cell.get("bytes")
            return data if isinstance(data, (bytes, bytearray)) else None
        except Exception:
            return None
    return None


def safe_rel_image_path(split: str, episode_id: Any, step_idx: Any) -> str:
    return f"{split}/episode_{episode_id}/step_{int(step_idx):04d}.png"


def write_image(image_root: Path, rel_path: str, image_bytes: Optional[bytes], source_path: str, require_image: bool) -> Optional[str]:
    target = image_root / rel_path
    if image_bytes:
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists() or target.stat().st_size != len(image_bytes):
            target.write_bytes(bytes(image_bytes))
        return str(target.resolve())
    if source_path and Path(source_path).exists():
        return str(Path(source_path).resolve())
    if require_image:
        raise FileNotFoundError(f"missing image for {rel_path}; source={source_path!r}")
    return None


def normalize_conversation(human: str, assistant: str) -> List[Dict[str, str]]:
    human = str(human or "").strip()
    assistant = str(assistant or "").strip()
    if not human or not assistant:
        raise ValueError("missing conversation_human or conversation_gpt")
    return [{"from": "human", "value": human}, {"from": "gpt", "value": assistant}]


def convert_balanced_split(
    *,
    data_dir: Path,
    split: str,
    output_json: Path,
    image_root: Path,
    max_episodes: int,
    max_steps: int,
    require_images: bool,
) -> Dict[str, Any]:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    image_root.mkdir(parents=True, exist_ok=True)

    examples: List[Dict[str, Any]] = []
    n_episodes = 0
    n_steps = 0
    skipped = 0

    for row in load_rows(data_dir, split, max_episodes=max_episodes):
        n_episodes += 1
        episode_id = row.get("episode_id", n_episodes)
        try:
            steps = json.loads(row.get("steps") or "[]")
        except json.JSONDecodeError:
            skipped += 1
            continue
        screenshots_value = row.get("screenshots")
        screenshots = list(screenshots_value) if screenshots_value is not None else []
        for step_pos, step in enumerate(steps):
            if max_steps > 0 and n_steps >= max_steps:
                break
            try:
                human = step.get("conversation_human")
                assistant = step.get("conversation_gpt")
                conversation = normalize_conversation(human, assistant)
                step_idx = step.get("step_idx", step_pos)
                rel_image = safe_rel_image_path(split, episode_id, step_idx)
                img_cell = screenshots[step_pos] if step_pos < len(screenshots) else None
                img_path = write_image(
                    image_root,
                    rel_image,
                    image_bytes_from_cell(img_cell),
                    str(step.get("screenshot") or ""),
                    require_image=require_images,
                )
                if not img_path:
                    skipped += 1
                    continue
                examples.append({"conversations": conversation, "images": [img_path]})
                n_steps += 1
            except Exception:
                skipped += 1
                if require_images:
                    raise
        if max_steps > 0 and n_steps >= max_steps:
            break

    output_json.write_text(json.dumps(examples, ensure_ascii=False), encoding="utf-8")
    return {
        "split": split,
        "episodes_read": n_episodes,
        "examples_written": len(examples),
        "steps_seen": n_steps,
        "skipped": skipped,
        "output_json": str(output_json),
        "image_root": str(image_root),
    }


def write_dataset_info(data_dir: Path, train_name: str, val_name: str, train_file: str, val_file: str) -> Path:
    path = data_dir / "dataset_info.json"
    current: Dict[str, Any] = {}
    if path.exists():
        try:
            current = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            current = {}

    def entry(file_name: str) -> Dict[str, Any]:
        return {
            "file_name": file_name,
            "formatting": "sharegpt",
            "columns": {"messages": "conversations", "images": "images"},
            "tags": {
                "role_tag": "from",
                "content_tag": "value",
                "user_tag": "human",
                "assistant_tag": "gpt",
            },
        }

    current[train_name] = entry(train_file)
    current[val_name] = entry(val_file)
    path.write_text(json.dumps(current, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def write_train_yaml(
    *,
    path: Path,
    base_model: str,
    dataset_dir: str,
    output_dir: str,
    ds_config: str,
    train_dataset: str,
    val_dataset: str,
    image_max_pixels: int,
    cutoff_len: int,
    epochs: float,
    learning_rate: float,
    gradient_accumulation_steps: int,
    save_strategy: str,
    save_steps: int,
    eval_strategy: str,
    eval_steps: int,
    report_to: str,
    run_name: str = "gui360_balanced_full_sft_repro",
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = f"""### model - Qwen2.5-VL-7B base
model_name_or_path: {base_model}
image_max_pixels: {image_max_pixels}
video_max_pixels: 16384
trust_remote_code: true

### method - full parameter SFT
stage: sft
do_train: true
finetuning_type: full
freeze_vision_tower: true
freeze_multi_modal_projector: false
freeze_language_model: false
deepspeed: {ds_config}

### dataset - GUI-360 balanced
dataset: {train_dataset}
dataset_dir: {dataset_dir}
template: qwen2_vl
cutoff_len: {cutoff_len}
preprocessing_num_workers: 16
dataloader_num_workers: 8

### output
output_dir: {output_dir}
logging_steps: 10
save_strategy: {save_strategy}
save_steps: {save_steps}
save_total_limit: 5
plot_loss: true
overwrite_output_dir: true
save_only_model: false
report_to: {report_to}
run_name: {run_name}

### train
per_device_train_batch_size: 1
gradient_accumulation_steps: {gradient_accumulation_steps}
learning_rate: {learning_rate}
num_train_epochs: {epochs}
lr_scheduler_type: cosine
warmup_ratio: 0.05
weight_decay: 0.1
bf16: true
ddp_timeout: 180000000

### eval
eval_dataset: {val_dataset}
per_device_eval_batch_size: 1
eval_strategy: {eval_strategy}
eval_steps: {eval_steps}
"""
    path.write_text(text, encoding="utf-8")
    return path


def run_command(cmd: Sequence[str], dry_run: bool) -> None:
    print("$ " + " ".join(cmd))
    if not dry_run:
        subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproduce GUI-360 full-parameter SFT data/config/training setup")
    parser.add_argument("--balanced-data-dir", default=DEFAULT_BALANCED_DIR)
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--image-dir", default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--config-out", default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--ds-config", default=DEFAULT_DS_CONFIG)
    parser.add_argument("--train-name", default="gui360_balanced_train")
    parser.add_argument("--val-name", default="gui360_balanced_val")
    parser.add_argument("--train-json", default="gui360_balanced_train.json")
    parser.add_argument("--val-json", default="gui360_balanced_val.json")
    parser.add_argument("--max-train-episodes", type=int, default=-1)
    parser.add_argument("--max-val-episodes", type=int, default=-1)
    parser.add_argument("--max-train-steps", type=int, default=-1)
    parser.add_argument("--max-val-steps", type=int, default=259, help="Default matches eval_samples=259 in trainer_state.json")
    parser.add_argument("--image-max-pixels", type=int, default=1003520)
    parser.add_argument("--cutoff-len", type=int, default=8192)
    parser.add_argument("--epochs", type=float, default=4.0)
    parser.add_argument("--learning-rate", type=float, default=1.0e-5)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--save-strategy", default="steps", choices=["no", "steps", "epoch", "best"])
    parser.add_argument("--save-steps", type=int, default=50)
    parser.add_argument("--eval-strategy", default="steps", choices=["no", "steps", "epoch"])
    parser.add_argument("--eval-steps", type=int, default=50)
    parser.add_argument("--report-to", default="none")
    parser.add_argument("--run-name", default="gui360_balanced_full_sft_repro")
    parser.add_argument("--prepare-data", action="store_true")
    parser.add_argument("--write-config", action="store_true")
    parser.add_argument("--require-images", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    image_dir = Path(args.image_dir)
    balanced_dir = Path(args.balanced_data_dir)
    train_json_path = data_dir / args.train_json
    val_json_path = data_dir / args.val_json

    summary: Dict[str, Any] = {
        "provenance": {
            "base_model": "Qwen/Qwen2.5-VL-7B-Instruct",
            "dataset": "Stevenshuqing/gui360-balanced",
            "method": "LLaMA-Factory full-parameter SFT + DeepSpeed ZeRO-3",
            "target_checkpoint": "checkpoints/gui360-fullparam-sft-step250",
            "target_step": 250,
            "target_epoch": 3.682,
        }
    }

    if args.prepare_data:
        train_summary = convert_balanced_split(
            data_dir=balanced_dir,
            split="train",
            output_json=train_json_path,
            image_root=image_dir,
            max_episodes=args.max_train_episodes,
            max_steps=args.max_train_steps,
            require_images=args.require_images,
        )
        val_summary = convert_balanced_split(
            data_dir=balanced_dir,
            split="test",
            output_json=val_json_path,
            image_root=image_dir,
            max_episodes=args.max_val_episodes,
            max_steps=args.max_val_steps,
            require_images=args.require_images,
        )
        dataset_info = write_dataset_info(data_dir, args.train_name, args.val_name, args.train_json, args.val_json)
        summary["data"] = {"train": train_summary, "val": val_summary, "dataset_info": str(dataset_info)}

    if args.write_config:
        yaml_path = write_train_yaml(
            path=Path(args.config_out),
            base_model=args.base_model,
            dataset_dir=args.data_dir,
            output_dir=args.output_dir,
            ds_config=args.ds_config,
            train_dataset=args.train_name,
            val_dataset=args.val_name,
            image_max_pixels=args.image_max_pixels,
            cutoff_len=args.cutoff_len,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            save_strategy=args.save_strategy,
            save_steps=args.save_steps,
            eval_strategy=args.eval_strategy,
            eval_steps=args.eval_steps,
            report_to=args.report_to,
            run_name=args.run_name,
        )
        summary["config"] = str(yaml_path)

    print(json.dumps(summary, indent=2, ensure_ascii=False))

    if args.run:
        run_command(["llamafactory-cli", "train", args.config_out], dry_run=args.dry_run)
    else:
        print("\nTrain command:")
        print(f"  llamafactory-cli train {args.config_out}")


if __name__ == "__main__":
    main()
