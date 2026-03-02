# UI-S1 SFT Training Guide

**Date**: 2026-02-10
**Status**: Complete Guide for Setting Up SFT Training

---

## Table of Contents

1. [Overview](#overview)
2. [VERL SFT Infrastructure](#verl-sft-infrastructure)
3. [Data Preparation](#data-preparation)
4. [Configuration](#configuration)
5. [Training Script](#training-script)
6. [Execution](#execution)
7. [Troubleshooting](#troubleshooting)

---

## Overview

This guide explains how to set up **Supervised Fine-Tuning (SFT)** for UI-S1 Android control tasks using the existing VERL framework.

### Key Components

| Component | Location | Purpose |
|-----------|----------|---------|
| SFT Dataset Classes | `verl/utils/dataset/` | Load and preprocess training data |
| SFT Trainer | `verl/trainer/fsdp_sft_trainer.py` | Main training loop with FSDP |
| SFT Config | `verl/trainer/config/sft_trainer.yaml` | Base configuration template |
| Training Scripts | `train/*.slurm` | SLURM job scripts |

### Training Pipeline

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  JSONL Data     │ -> │  Parquet Data   │ -> │  SFT Dataset    │
│  (ui_s1_train)  │    │  (messages)     │    │  (multiturn)     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                       │
                                                       v
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Checkpoints    │ <- │  FSDP Trainer   │ <- │  Config + Model │
│  (saved)        │    │  (training)     │    │  (Qwen2.5-VL)   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

---

## VERL SFT Infrastructure

### 1. Dataset Classes

VERL provides two dataset classes for SFT:

#### **SFTDataset** (`verl/utils/dataset/sft_dataset.py`)
- **Purpose**: Single-turn SFT (prompt → response)
- **Input**: Parquet with `prompt_key` and `response_key` columns
- **Use Case**: Simple Q&A, instruction following

```python
# Data format:
# | prompt | response |
# |--------|----------|
# | "What is 2+2?" | "4" |
```

#### **MultiTurnSFTDataset** (`verl/utils/dataset/multiturn_sft_dataset.py`)
- **Purpose**: Multi-turn conversations
- **Input**: Parquet with `messages` column (list of message dicts)
- **Use Case**: Dialogues, trajectory-based tasks (like UI-S1)

```python
# Data format:
# | messages |
# |----------|
# | [{"role": "user", "content": [...]}, {"role": "assistant", "content": "..."}] |
```

**For UI-S1 SFT, we use MultiTurnSFTDataset** because each trajectory has multiple steps.

### 2. SFT Trainer

**File**: `verl/trainer/fsdp_sft_trainer.py`

**Key Features**:
- FSDP/FSDP2 sharding strategies
- LoRA support (configurable rank/alpha)
- Mixed precision (bfloat16)
- Gradient checkpointing
- Sequence parallelism (Ulysses)
- Automatic checkpointing

**Training Loop**:
```python
def run_sft(config):
    # 1. Initialize distributed process group
    device_mesh = init_device_mesh(...)

    # 2. Load tokenizer
    tokenizer = hf_tokenizer(config.model.partial_pretrain)

    # 3. Create datasets
    train_dataset = create_sft_dataset(config.data.train_files, config.data, tokenizer)
    val_dataset = create_sft_dataset(config.data.val_files, config.data, tokenizer)

    # 4. Initialize trainer
    trainer = FSDPSFTTrainer(config, device_mesh, tokenizer, train_dataset, val_dataset)

    # 5. Train
    trainer.fit()
```

### 3. Dataset Factory Pattern

The `create_sft_dataset()` function automatically selects the dataset class:

```python
# From fsdp_sft_trainer.py
def create_sft_dataset(data_paths, data_config, tokenizer):
    # 1. Check for custom class
    if data_config.custom_cls.path:
        dataset_cls = load_extern_type(data_config.custom_cls.path, data_config.custom_cls.name)

    # 2. Check for multiturn
    elif data_config.multiturn.enable:
        dataset_cls = MultiTurnSFTDataset

    # 3. Default to single-turn
    else:
        dataset_cls = SFTDataset

    return dataset_cls(parquet_files=data_paths, tokenizer=tokenizer, config=data_config)
```

---

## Data Preparation

### Current Data Format

The UI-S1 training data is in JSONL format:

```json
{
  "goal": "Open the Zoho Meet app, view the scheduled meetings.",
  "is_successful": true,
  "steps": [
    {
      "action_content": {
        "action": "open",
        "text": "Zoho Meeting"
      },
      "screenshot": "/datasets/AndroidControl/images/0_0.png"
    },
    {
      "action_content": {
        "action": "click",
        "coordinate": [540.0, 389.8]
      },
      "screenshot": "/datasets/AndroidControl/images/0_1.png"
    }
  ]
}
```

### Target Format (MultiTurnSFTDataset)

Convert to parquet with `messages` column:

```python
# Each trajectory becomes:
{
    "messages": [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Task: Open the Zoho Meet app..."},
                {"type": "image", "image": "/datasets/AndroidControl/images/0_0.png"}
            ]
        },
        {
            "role": "assistant",
            "content": '{"action": "open", "text": "Zoho Meeting"}'
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Task: Open the Zoho Meet app..."},
                {"type": "image", "image": "/datasets/AndroidControl/images/0_1.png"}
            ]
        },
        {
            "role": "assistant",
            "content": '{"action": "click", "coordinate": [540.0, 389.8]}'
        }
    ]
}
```

### Data Conversion Script

Create `scripts/prepare_gui_sft_parquet.py`:

```python
#!/usr/bin/env python3
import argparse
import json
import os
import pandas as pd
from typing import Dict, List, Any

def format_action_to_response(action_content: Dict[str, Any]) -> str:
    """Format action content to JSON response string."""
    action = action_content.get("action", "")
    response = f'{{"action": "{action}"'

    # Add action-specific parameters
    if action == "click":
        coord = action_content.get("coordinate")
        if coord is not None:
            response += f', "coordinate": {list(coord)}'
    elif action == "type":
        text = action_content.get("text", "").replace('"', '\\"')
        response += f', "text": "{text}"'
    # ... handle other action types ...

    response += "}"
    return response

def trajectory_to_messages(trajectory: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Convert a trajectory to multiturn messages format."""
    messages = []
    goal = trajectory.get("goal", "")
    steps = trajectory.get("steps", [])

    for step in steps:
        screenshot_path = step.get("screenshot", "")
        if not os.path.exists(screenshot_path):
            continue

        # User message with image
        user_content = [
            {"type": "text", "text": f"Task: {goal}\\n\\nPlease perform the appropriate action."},
            {"type": "image", "image": screenshot_path}
        ]
        messages.append({"role": "user", "content": user_content})

        # Assistant message with action
        response = format_action_to_response(step["action_content"])
        messages.append({"role": "assistant", "content": response})

    return messages

def convert_jsonl_to_parquet(input_jsonl: str, output_parquet: str):
    """Convert UI-S1 JSONL to parquet format."""
    data_samples = []

    with open(input_jsonl, 'r') as f:
        for line in f:
            trajectory = json.loads(line)
            if not trajectory.get("is_successful", True):
                continue

            messages = trajectory_to_messages(trajectory)
            if messages:
                data_samples.append({"messages": messages})

    df = pd.DataFrame(data_samples)
    df.to_parquet(output_parquet, index=False)
    print(f"Converted {len(data_samples)} trajectories to {output_parquet}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", required=True)
    parser.add_argument("--output", "-o", required=True)
    args = parser.parse_args()

    convert_jsonl_to_parquet(args.input, args.output)
```

### Execute Data Preparation

```bash
# Convert training data
python scripts/prepare_gui_sft_parquet.py \
    --input datasets/ui_s1_train.jsonl \
    --output datasets/ui_s1_train_sft.parquet

# Convert evaluation data
python scripts/prepare_gui_sft_parquet.py \
    --input evaluation/dataset/android_control_evaluation_std.jsonl \
    --output datasets/ui_s1_eval_sft.parquet
```

---

## Configuration

### Base SFT Config

**File**: `verl/trainer/config/sft_trainer.yaml`

```yaml
data:
  train_batch_size: 256
  micro_batch_size_per_gpu: 4
  train_files: ~/data/gsm8k/train.parquet
  val_files: ~/data/gsm8k/test.parquet
  max_length: 1024
  truncation: error
  multiturn:
    enable: false  # Set to true for UI-S1
    messages_key: messages
  custom_cls:
    path: null
    name: null

model:
  partial_pretrain: ~/models/gemma-1.1-7b-it
  strategy: fsdp2
  fsdp_config:
    model_dtype: fp32
    wrap_policy:
      min_num_params: 0
    cpu_offload: False
  enable_gradient_checkpointing: True
  trust_remote_code: False
  lora_rank: 0
  lora_alpha: 16
  target_modules: all-linear

optim:
  lr: 1e-5
  betas: [0.9, 0.95]
  weight_decay: 0.01
  warmup_steps_ratio: 0.1
  clip_grad: 1.0
  lr_scheduler: cosine

trainer:
  project_name: gsm8k-sft
  experiment_name: test
  total_epochs: 4
  logger: ['console', 'wandb']
  save_freq: -1
  test_freq: -1
```

### UI-S1 SFT Config

**Create**: `examples/qwen_gui_sft/config/gui_sft.yaml`

```yaml
hydra:
  searchpath:
    - file:///scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/verl/trainer/config

defaults:
  - sft_trainer
  - _self_

# Data configuration
data:
  train_files: /scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/datasets/ui_s1_train_sft.parquet
  val_files: /scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/datasets/ui_s1_eval_sft.parquet
  train_batch_size: 32  # Adjust based on GPU memory
  micro_batch_size_per_gpu: 2
  max_length: 8192  # Qwen2.5-VL supports long contexts
  truncation: error
  multiturn:
    enable: true  # Use multiturn dataset
    messages_key: messages
  trust_remote_code: true  # For Qwen2.5-VL

# Model configuration
model:
  partial_pretrain: /scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/checkpoints/Qwen2.5-VL-7B-Instruct
  strategy: fsdp2
  fsdp_config:
    model_dtype: fp32
    wrap_policy:
      min_num_params: 0
    cpu_offload: True  # Enable for memory efficiency
    offload_params: True
  enable_gradient_checkpointing: True
  trust_remote_code: True  # Required for Qwen2.5-VL
  lora_rank: 64  # Optional: use LoRA for memory efficiency
  lora_alpha: 128
  target_modules: [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]
  use_liger: False
  external_lib: null

# Optimizer configuration
optim:
  lr: 5e-6  # Lower LR for vision-language models
  betas: [0.9, 0.95]
  weight_decay: 0.01
  warmup_steps_ratio: 0.1
  clip_grad: 1.0
  lr_scheduler: cosine

# Training configuration
trainer:
  default_local_dir: /scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/train/checkpoints/gui_sft
  project_name: gui-sft
  experiment_name: qwenvl_uis1_lora64
  total_epochs: 3
  total_training_steps: null
  logger: ['console', 'wandb']
  seed: 42
  save_freq: 50
  test_freq: 25
  max_ckpt_to_keep: 3

# Sequence parallelism (optional)
ulysses_sequence_parallel_size: 1
use_remove_padding: False
```

---

## Training Script

### SLURM Script

**Create**: `train/train_gui_sft.slurm`

```bash
#!/bin/bash
#SBATCH --job-name=train_gui_sft
#SBATCH --output=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/train/logs/train_gui_sft_%j.log
#SBATCH --error=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/train/logs/train_gui_sft_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --gres=gpu:4

# ============================================
# UI-S1 SFT Training Script
# ============================================

set -e

# Environment
PROJECT_DIR="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1"
CONFIG_PATH="$PROJECT_DIR/examples/qwen_gui_sft/config"
LOG_DIR="$PROJECT_DIR/train/logs"

# Activate conda environment
source /home/a5l/shuqing.a5l/miniconda3/bin/activate ui-s1
source $PROJECT_DIR/train/env_config.sh

# Create log directory
mkdir -p $LOG_DIR

echo "=========================================="
echo "Starting UI-S1 SFT Training"
echo "Project Dir: $PROJECT_DIR"
echo "Config: gui_sft"
echo "Nodes: ${SLURM_NNODES}"
echo "GPUs per node: 4"
echo "Start time: $(date)"
echo "=========================================="

# Set distributed environment variables
export MASTER_ADDR=$(hostname)
export MASTER_PORT=29500
export NCCL_DEBUG=WARN

# Calculate world size
export WORLD_SIZE=${SLURM_NNODES}*4

# Launch training with torchrun
cd $PROJECT_DIR

torchrun --nproc_per_node=4 --master_port=$MASTER_PORT \
    verl/trainer/fsdp_sft_trainer.py \
    --config-path=$CONFIG_PATH \
    --config-name=gui_sft \
    2>&1 | tee $LOG_DIR/train_gui_sft_${SLURM_JOB_ID}.log

TRAIN_EXIT_CODE=${PIPESTATUS[0]}

echo "=========================================="
echo "Training completed at $(date)"
echo "Exit code: $TRAIN_EXIT_CODE"
echo "=========================================="

exit $TRAIN_EXIT_CODE
```

### Multi-Node SLURM Script

For multi-node training (8 nodes x 4 GPUs = 32 GPUs):

```bash
#!/bin/bash
#SBATCH --job-name=train_gui_sft_multinode
#SBATCH --output=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/train/logs/train_gui_sft_mn_%j.log
#SBATCH --error=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/train/logs/train_gui_sft_mn_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=workq
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --gres=gpu:4

set -e

PROJECT_DIR="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1"
CONFIG_PATH="$PROJECT_DIR/examples/qwen_gui_sft/config"

source /home/a5l/shuqing.a5l/miniconda3/bin/activate ui-s1
source $PROJECT_DIR/train/env_config.sh

# Get head node
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address | awk '{print $1}')

export MASTER_ADDR=$head_node_ip
export MASTER_PORT=29500

echo "Head node: $head_node ($head_node_ip)"
echo "Nodes: ${SLURM_NNODES}"

cd $PROJECT_DIR

srun --ntasks-per-node=1 \
    verl/trainer/fsdp_sft_trainer.py \
    --config-path=$CONFIG_PATH \
    --config-name=gui_sft \
    trainer.nnodes=${SLURM_NNODES} \
    trainer.n_gpus_per_node=4
```

---

## Execution

### Step 1: Prepare Data

```bash
cd /scratch/a5l/shuqing.a5l/MobileAgent/UI-S1

# Create conversion script
cat > scripts/prepare_gui_sft_parquet.py << 'EOF'
# (Insert the full conversion script here)
EOF

# Run conversion
python scripts/prepare_gui_sft_parquet.py \
    --input datasets/ui_s1_train.jsonl \
    --output datasets/ui_s1_train_sft.parquet
```

### Step 2: Create Config

```bash
# Create config directory
mkdir -p examples/qwen_gui_sft/config

# Create config file
cat > examples/qwen_gui_sft/config/gui_sft.yaml << 'EOF'
# (Insert the full config here)
EOF
```

### Step 3: Submit Training Job

```bash
# Single node (4 GPUs)
sbatch train/train_gui_sft.slurm

# Multi node (32 GPUs)
sbatch train/train_gui_sft_multinode.slurm
```

### Step 4: Monitor Training

```bash
# Check job status
squeue -u $USER

# View logs
tail -f train/logs/train_gui_sft_<JOB_ID>.log

# Monitor with wandb
# Project: gui-sft
# Experiment: qwenvl_uis1_lora64
```

### Step 5: Check Results

```bash
# Checkpoints saved to:
ls train/checkpoints/gui_sft/qwenvl_uis1_lora64/

# Expected output:
# global_step_50/
# global_step_100/
# global_step_150/
# ...
```

---

## Troubleshooting

### Issue 1: Out of Memory

**Symptoms**: CUDA OOM errors during training

**Solutions**:
```yaml
# Reduce batch sizes
data:
  train_batch_size: 16  # from 32
  micro_batch_size_per_gpu: 1  # from 2

# Enable CPU offloading
model:
  fsdp_config:
    cpu_offload: true
    offload_params: true

# Use LoRA
model:
  lora_rank: 32  # from 64
```

### Issue 2: Slow Training

**Symptoms**: Training takes too long

**Solutions**:
```yaml
# Enable gradient checkpointing (already on)
model:
  enable_gradient_checkpointing: true

# Use more GPUs
#SBATCH --nodes=4  # increase from 1

# Increase micro batch size
data:
  micro_batch_size_per_gpu: 4  # from 2
```

### Issue 3: Data Loading Errors

**Symptoms**: FileNotFoundError for images

**Solutions**:
```python
# In conversion script, validate image paths
if not os.path.exists(screenshot_path):
    print(f"Warning: {screenshot_path} not found")
    continue  # Skip this sample
```

### Issue 4: Tokenizer Errors

**Symptoms**: Chat template errors, tokenization failures

**Solutions**:
```yaml
# Ensure trust_remote_code is set
model:
  trust_remote_code: true

data:
  trust_remote_code: true
```

---

## Summary

### Quick Start Command Sequence

```bash
# 1. Prepare data
python scripts/prepare_gui_sft_parquet.py \
    -i datasets/ui_s1_train.jsonl \
    -o datasets/ui_s1_train_sft.parquet

# 2. Submit training
sbatch train/train_gui_sft.slurm

# 3. Monitor
tail -f train/logs/train_gui_sft_*.log
```

### Key Files Reference

| File | Purpose |
|------|---------|
| `scripts/prepare_gui_sft_parquet.py` | Data conversion |
| `examples/qwen_gui_sft/config/gui_sft.yaml` | Training config |
| `train/train_gui_sft.slurm` | SLURM script |
| `verl/trainer/fsdp_sft_trainer.py` | Trainer (no changes needed) |
| `verl/utils/dataset/multiturn_sft_dataset.py` | Dataset (no changes needed) |

---

*Document created: 2026-02-10*
