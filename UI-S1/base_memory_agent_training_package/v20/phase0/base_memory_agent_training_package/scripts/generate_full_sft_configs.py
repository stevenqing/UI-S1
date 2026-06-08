#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

AGENTS = [
    "type_recovery",
    "click_recovery",
    "swipe_navigation",
    "minimal_next_step",
    "escape_finish",
    "spreadsheet_formula",
]

TEMPLATE = """### model
model_name_or_path: {model_path}
image_max_pixels: 1003520
video_max_pixels: 16384
trust_remote_code: true

### method
stage: sft
do_train: true
finetuning_type: full
freeze_vision_tower: true
freeze_multi_modal_projector: false
freeze_language_model: false
deepspeed: train_GUI_360/llamafactory/ds_z3_config.json

### dataset
dataset: proposal_{agent}_train
dataset_dir: {dataset_dir}
template: qwen2_vl
cutoff_len: 4096
preprocessing_num_workers: 16
dataloader_num_workers: 8

### output
output_dir: train_GUI_360/llamafactory/output/full_proposal_{agent}
logging_steps: 5
save_steps: 100
save_total_limit: 3
plot_loss: true
overwrite_output_dir: true
save_only_model: false
report_to: none
run_name: full_proposal_{agent}

### train
per_device_train_batch_size: 1
gradient_accumulation_steps: 16
learning_rate: 1.0e-5
num_train_epochs: 2.0
lr_scheduler_type: cosine
warmup_ratio: 0.05
weight_decay: 0.1
bf16: true
ddp_timeout: 180000000

### eval
eval_dataset: proposal_{agent}_dev
per_device_eval_batch_size: 1
eval_strategy: steps
eval_steps: 100
"""


def main() -> None:
    package_dir = Path(__file__).resolve().parents[1]
    config_dir = package_dir / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    model_path = "models/Qwen2.5-VL-7B-Instruct"
    dataset_dir = "train_GUI_360/llamafactory/data"
    for agent in AGENTS:
        path = config_dir / f"qwen25vl_full_proposal_{agent}.yaml"
        path.write_text(TEMPLATE.format(agent=agent, model_path=model_path, dataset_dir=dataset_dir), encoding="utf-8")
        print(path)


if __name__ == "__main__":
    main()
