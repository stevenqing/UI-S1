# GUI-360 SFT Training Guide

A comprehensive guide for using the GUI-360 dataset to perform Supervised Fine-Tuning (SFT) on vision-language models.

---

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Data Preparation Pipeline](#data-preparation-pipeline)
4. [Training Configuration](#training-configuration)
5. [Running SFT Training](#running-sft-training)
6. [Configuration Options](#configuration-options)
7. [Troubleshooting](#troubleshooting)

---

## Overview

This guide explains how to use the GUI-360 dataset for SFT training with the verl framework. The pipeline involves:

1. Converting raw GUI-360 data to parquet format
2. Configuring the SFT trainer
3. Launching distributed training via SLURM

### GUI-360 Dataset Summary

| Feature | Value |
|---------|-------|
| **Total Trajectories** | 79,425 |
| **Action Steps** | 1.2M+ |
| **Applications** | Microsoft Word, Excel, PowerPoint |
| **Processed Datasets** | 4 task-specific (action prediction, grounding, screen parsing) |
| **Image Format** | PNG screenshots |

---

## Prerequisites

### Environment Setup

```bash
# Activate the conda environment
source /home/a5l/shuqing.a5l/miniconda3/bin/activate ui-s1

# Required packages
pip install torch transformers pandas pyarrow hydra-core wandb
```

### Required Files

| File | Location | Purpose |
|------|----------|---------|
| Base Model | `checkpoints/Qwen2.5-VL-7B-Instruct/` | Pre-trained VLM |
| GUI-360 Data | `datasets/GUI-360/` | Raw trajectory data |
| Config | `examples/qwen_gui_sft/config/` | Training configuration |

---

## Data Preparation Pipeline

### Step 1: Understand GUI-360 Data Structure

GUI-360 provides two types of data:

#### Raw Trajectory Data (JSONL format)
```
GUI-360/
├── train/           # Successful training trajectories
│   ├── excel/
│   │   ├── in_app/
│   │   ├── search/
│   │   └── online/
│   ├── word/
│   └── ppt/
├── test/            # Test/benchmark set
└── processed_data/  # Pre-processed SFT-ready data
```

#### Pre-processed Data (Recommended for Quick Start)
```
processed_data/
├── action_prediction_train_resize/    # 101,800 samples
│   ├── training_data.json
│   └── images.tar.gz
├── action_prediction_train_resize_a11y/  # With accessibility info
├── grounding_resize/                  # 79,487 samples
└── screen_parsing_train_resize/       # 97,351 samples
```

### Step 2: Convert to Parquet Format

The SFT trainer requires data in parquet format with multi-turn conversation structure.

#### Using the Preparation Script

```bash
# Convert training data
python scripts/prepare_gui_sft_parquet.py \
    --input datasets/gui360_train.jsonl \
    --output datasets/ui_s1_train_sft.parquet

# Convert evaluation data
python scripts/prepare_gui_sft_parquet.py \
    --input datasets/gui360_eval.jsonl \
    --output datasets/ui_s1_eval_sft.parquet

# Convert with options
python scripts/prepare_gui_sft_parquet.py \
    --input datasets/gui360_train.jsonl \
    --output datasets/ui_s1_train_sft.parquet \
    --max-trajectories 1000 \        # Limit for testing
    --no-terminate \                  # Exclude terminate actions
    --include-unsuccessful            # Include failed trajectories
```

### Step 3: Data Format Details

The preparation script converts trajectories to multi-turn conversations:

#### Input Format (JSONL)
```json
{
  "goal": "Create a pivot table in Excel",
  "is_successful": true,
  "steps": [
    {
      "screenshot": "/path/to/screenshot.png",
      "action_content": {
        "action": "click",
        "coordinate": [150, 65]
      }
    }
  ]
}
```

#### Output Format (Parquet)
```json
{
  "messages": "[{\"role\": \"user\", \"content\": [{\"type\": \"text\", \"text\": \"Task: Create a pivot table...\"}, {\"type\": \"image\", \"image\": \"/path/to/screenshot.png\"}]}, {\"role\": \"assistant\", \"content\": \"{\\\"action\\\": \\\"click\\\", \\\"coordinate\\\": [150, 65]}\"}]"
}
```

### Supported Actions

| Action | Parameters | Example Output |
|--------|------------|----------------|
| `click` | coordinate | `{"action": "click", "coordinate": [150, 65]}` |
| `type` | text | `{"action": "type", "text": "Hello"}` |
| `swipe` | coordinate, coordinate2 | `{"action": "swipe", "coordinate": [100, 500], "coordinate2": [100, 100]}` |
| `long_press` | coordinate | `{"action": "long_press", "coordinate": [150, 65]}` |
| `open` | text | `{"action": "open", "text": "Settings"}` |
| `wait` | time | `{"action": "wait", "time": 2}` |
| `system_button` | button | `{"action": "system_button", "button": "home"}` |
| `terminate` | status | `{"action": "terminate", "status": "success"}` |

---

## Training Configuration

### Configuration File Structure

Create a YAML config at `examples/qwen_gui_sft/config/gui_sft.yaml`:

```yaml
hydra:
  searchpath:
    - file:///path/to/UI-S1/verl/trainer/config

defaults:
  - sft_trainer
  - _self_

# Data Configuration
data:
  train_files: /path/to/ui_s1_train_sft.parquet
  val_files: /path/to/ui_s1_eval_sft.parquet
  train_batch_size: 32        # Global batch size
  micro_batch_size_per_gpu: 1  # Per-GPU batch size
  max_length: 8192            # Max sequence length
  truncation: error           # Error on truncation
  multiturn:
    enable: true
    messages_key: messages
    tools_key: tools
    enable_thinking_key: enable_thinking
  custom_cls:
    path: /path/to/verl/utils/dataset/gui_multiturn_sft_dataset.py
    name: GUIMultiTurnSFTDataset
  trust_remote_code: true

# Model Configuration
model:
  partial_pretrain: /path/to/checkpoints/Qwen2.5-VL-7B-Instruct
  strategy: fsdp2
  fsdp_config:
    model_dtype: fp32
    wrap_policy:
      min_num_params: 0
    cpu_offload: False
    offload_params: False
  enable_gradient_checkpointing: True
  trust_remote_code: True
  lora_rank: 0                # 0 = full fine-tuning
  use_liger: False

# Optimizer Configuration
optim:
  lr: 5e-6                    # Learning rate
  betas: [0.9, 0.95]
  weight_decay: 0.01
  warmup_steps_ratio: 0.1
  clip_grad: 1.0
  lr_scheduler: cosine

# Trainer Configuration
trainer:
  default_local_dir: /path/to/checkpoints/gui_sft
  project_name: gui-sft
  experiment_name: qwenvl_gui360_sft
  total_epochs: 3
  logger: ['console', 'wandb']
  seed: 42
  save_freq: 50
  test_freq: 25
  nnodes: 1
  n_gpus_per_node: 4
  max_ckpt_to_keep: 3
```

### Anti-Overfitting Configuration (v2)

For preventing overfitting, use more aggressive regularization:

```yaml
# Key differences from base config:
optim:
  lr: 1e-6               # Lower LR (was 5e-6)
  betas: [0.9, 0.98]     # Higher beta2
  weight_decay: 0.3      # Higher regularization (was 0.01)
  warmup_steps_ratio: 0.05
  clip_grad: 0.3         # Tighter gradient clipping

trainer:
  total_epochs: 1        # Single epoch
  save_freq: 10          # More frequent saves
  test_freq: 5           # More frequent validation
```

---

## Running SFT Training

### SLURM Job Script

Create `train/train_gui_sft.slurm`:

```bash
#!/bin/bash
#SBATCH --job-name=train_gui_sft
#SBATCH --output=train/logs/train_gui_sft_%j.log
#SBATCH --error=train/logs/train_gui_sft_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --gres=gpu:4

set -e

# Environment setup
PROJECT_DIR="/path/to/UI-S1"
CONFIG_PATH="$PROJECT_DIR/examples/qwen_gui_sft/config"
LOG_DIR="$PROJECT_DIR/train/logs"

# Activate conda environment
source /path/to/miniconda3/bin/activate ui-s1
source $PROJECT_DIR/train/env_config.sh

mkdir -p $LOG_DIR

# Training configuration
EXPERIMENT_NAME="qwenvl_gui360_sft"
CONFIG_NAME="gui_sft"

echo "Starting UI-S1 SFT Training"
echo "Config: $CONFIG_NAME"
echo "Experiment: $EXPERIMENT_NAME"

# Distributed environment
export MASTER_ADDR=$(hostname)
export MASTER_PORT=29500
export NCCL_DEBUG=WARN

cd $PROJECT_DIR

# Launch training
torchrun --nproc_per_node=4 --master_port=$MASTER_PORT \
    -m verl.trainer.fsdp_sft_trainer \
    --config-path=$CONFIG_PATH \
    --config-name=$CONFIG_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME

echo "Training completed"
```

### Submit Training Job

```bash
# Create log directory
mkdir -p train/logs

# Submit job
sbatch train/train_gui_sft.slurm

# Monitor job
squeue -u $USER
tail -f train/logs/train_gui_sft_*.log
```

### Local Training (Non-SLURM)

```bash
# Single GPU
python -m verl.trainer.fsdp_sft_trainer \
    --config-path=examples/qwen_gui_sft/config \
    --config-name=gui_sft

# Multi-GPU
torchrun --nproc_per_node=4 \
    -m verl.trainer.fsdp_sft_trainer \
    --config-path=examples/qwen_gui_sft/config \
    --config-name=gui_sft
```

---

## Configuration Options

### Data Options

| Option | Description | Default |
|--------|-------------|---------|
| `train_batch_size` | Global batch size | 32 |
| `micro_batch_size_per_gpu` | Per-GPU batch size | 1 |
| `max_length` | Maximum sequence length | 8192 |
| `truncation` | Truncation strategy | "error" |
| `multiturn.enable` | Enable multi-turn format | true |

### Model Options

| Option | Description | Default |
|--------|-------------|---------|
| `strategy` | Training strategy | "fsdp2" |
| `enable_gradient_checkpointing` | Memory optimization | True |
| `lora_rank` | LoRA rank (0=full FT) | 0 |
| `model_dtype` | Model data type | "fp32" |

### Optimizer Options

| Option | Description | Default |
|--------|-------------|---------|
| `lr` | Learning rate | 5e-6 |
| `weight_decay` | L2 regularization | 0.01 |
| `clip_grad` | Gradient clipping | 1.0 |
| `lr_scheduler` | LR schedule | "cosine" |
| `warmup_steps_ratio` | Warmup ratio | 0.1 |

### Trainer Options

| Option | Description | Default |
|--------|-------------|---------|
| `total_epochs` | Training epochs | 3 |
| `save_freq` | Checkpoint frequency | 50 |
| `test_freq` | Validation frequency | 25 |
| `logger` | Logging backends | ['console', 'wandb'] |

---

## Troubleshooting

### Common Issues

#### 1. Out of Memory (OOM)

```yaml
# Reduce batch size
data:
  micro_batch_size_per_gpu: 1
  max_length: 4096  # Reduce sequence length

# Enable gradient checkpointing
model:
  enable_gradient_checkpointing: True
```

#### 2. Image Not Found Errors

```bash
# Validate images exist before training
python scripts/prepare_gui_sft_parquet.py \
    --input data.jsonl \
    --output data.parquet
    # Default validates images exist
```

#### 3. JSON Parse Errors

The `GUIMultiTurnSFTDataset` handles JSON serialization automatically. If you see parse errors, verify the parquet file:

```python
import pandas as pd
import json

df = pd.read_parquet("ui_s1_train_sft.parquet")
# Check first sample
sample = json.loads(df['messages'].iloc[0])
print(json.dumps(sample, indent=2))
```

#### 4. NCCL Communication Issues

```bash
# Set in env_config.sh
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=hsn0  # Adjust for your network
export NCCL_P2P_DISABLE=1       # For GH200 systems
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=7200
```

### Monitoring Training

```bash
# Watch training logs
tail -f train/logs/train_gui_sft_*.log

# Monitor GPU usage
watch -n 1 nvidia-smi

# Check WandB dashboard
# https://wandb.ai/YOUR_PROJECT/gui-sft
```

---

## Quick Start Checklist

1. [ ] Download GUI-360 dataset
2. [ ] Extract images: `tar -xzf images.tar.gz`
3. [ ] Prepare parquet data: `python scripts/prepare_gui_sft_parquet.py ...`
4. [ ] Create/modify config: `examples/qwen_gui_sft/config/gui_sft.yaml`
5. [ ] Set up environment: `source train/env_config.sh`
6. [ ] Submit training: `sbatch train/train_gui_sft.slurm`
7. [ ] Monitor: `tail -f train/logs/*.log`

---

## References

- [GUI-360 Dataset Documentation](../datasets/GUI-360/GUI-360_Dataset_Documentation.md)
- [verl SFT Trainer Documentation](https://verl.readthedocs.io/)
- [Qwen2.5-VL Model Card](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct)

---

*Generated: 2026-02-12*
