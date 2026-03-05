# UI-S1: Advancing GUI Automation via Semi-online Reinforcement Learning

<font size=4><div align='center' > [[Paper](https://arxiv.org/abs/2509.11543)] [[UI-S1-7B](https://huggingface.co/mPLUG/UI-S1-7B)] [[Daily Paper](https://huggingface.co/papers/2509.11543)] [[Dataset](https://huggingface.co/datasets/mPLUG/UI_S1_dataset)]</div></font>

</div>
<div align="center">
  <a href="README.md">English</a> | <a href="README_zh.md">简体中文</a>
<hr>
</div>

## Overview

We present **Semi-online RL**, a novel paradigm that simulates online reinforcement learning using offline trajectories, thereby enabling the efficient training of MLLM-based GUI agents with enhanced multi-turn interaction capabilities.

<div align="center">
  <img src="assets/method_comparison.png" alt="Logo" style="width:80%;">
</div>

Our **UI-S1-7B** achieves SOTA performance on both semi-online metric (SOP) and online metric (AndroidWorld) among open-source 7B models.

<div align="center">
  <img src="assets/metric.png" alt="Logo" style="width:80%;">
</div>

## Detailed Results

<div align="center">
  <img src="assets/result.png" alt="Logo" style="width:80%;">
</div>

---

## Installation

### Quick Install (x86_64 / Standard GPU)

For standard x86_64 machines with NVIDIA GPUs (A100, H100, etc.):

```bash
# 1. Create conda environment
conda create -n ui-s1 python=3.11
conda activate ui-s1

# 2. Install the package
cd UI-S1
pip install -e .

# 3. Install vLLM and Flash Attention
pip install vllm==0.8.2
pip install flash-attn==2.7.4.post1 --no-build-isolation
# Or install from prebuilt wheel:
# pip install flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp311-cp311-linux_x86_64.whl
```

We use [SwanLab](https://swanlab.ai/) for training visualization. Replace your own SwanLab API key and host in `verl/utils/tracking.py`.

### Full Install (GH200 / aarch64 + sm90 HPC)

For NVIDIA GH200 (aarch64, sm90) on HPC systems with SLURM, all components must be built from source. See [docs/installation_guide.md](docs/installation_guide.md) for the full guide.

**Prerequisites:**

| Component | Version |
|-----------|---------|
| Python | 3.11 (via conda) |
| CUDA | 12.6 (system module) |
| GCC | 12.3 (system module) |

**Step 1: Build Triton from source** (required before Flash Attention / vLLM)

```bash
# Prepare offline packages first
mkdir -p ~/.triton/offline

# Triton source with submodules
git clone --recursive https://github.com/triton-lang/triton
cd triton && git checkout v3.2.2 && git submodule update --init --recursive
cd .. && tar -czf triton-3.2-recursive.tar.gz triton
mv triton-3.2-recursive.tar.gz ~/.triton/offline/

# LLVM 17.0.6 source (download from https://github.com/llvm/llvm-project/releases/tag/llvmorg-17.0.6)
# Place as ~/.triton/offline/llvm-project-17.0.6.tar.gz

# Submit build job (~12 hours)
sbatch install_scripts/install_triton.slurm
```

**Step 2: Install PyTorch + Flash Attention + vLLM + UI-S1**

```bash
# All-in-one installation (~12 hours)
sbatch install_scripts/install_all.slurm
```

This installs: PyTorch 2.6.0 (CUDA 12.6) -> Flash Attention 2.7.0.post2 -> vLLM 0.6.4.post1 -> UI-S1.

**Optional: Install newer vLLM (v0.8.5) with FA3 disabled**

```bash
sbatch install_scripts/install_vllm.slurm
```

**Fix version mismatches** (if vLLM downgrades PyTorch):

```bash
sbatch install_scripts/fix_versions.slurm
```

### Verification

```python
import torch
print('PyTorch:', torch.__version__, 'CUDA:', torch.cuda.is_available())

import triton
print('Triton:', triton.__version__)

import flash_attn
print('Flash Attention:', flash_attn.__version__)

import vllm
print('vLLM:', vllm.__version__)
```

---

## Data

### Option A: UI-S1 Dataset (AndroidControl)

For reproducing the original UI-S1 paper results:

1. Download the 1000-example training subset from HuggingFace:

```bash
sbatch install_scripts/download_dataset.slurm
```

This downloads [mPLUG/UI_S1_dataset](https://huggingface.co/datasets/mPLUG/UI_S1_dataset) and saves to `datasets/ui_s1_train.jsonl`.

2. Download AndroidControl images:

```bash
# Training images (from smolagents/android-control, streaming mode)
sbatch install_scripts/download_images.slurm

# Evaluation images (if missing)
sbatch install_scripts/download_eval_images.slurm

# Download any remaining missing episodes
sbatch install_scripts/download_missing.slurm
```

Images are saved to `datasets/AndroidControl/images/` with naming convention `{episode_id}_{step_id}.png`.

### Option B: GUI-360 Dataset

For GUI-360 SFT training (larger-scale, multi-task):

**1. Download the raw dataset:**

Download the GUI-360 dataset from [GUI-360](https://github.com/AkimfromParis/GUI-360) and place it under `datasets/GUI-360/`.

Expected structure:
```
datasets/GUI-360/
  processed_data/
    action_prediction_train_resize/
      training_data.json        # Pre-processed training data
      images/                   # Training screenshots
    action_prediction_test/
      test_data.json
      images/
```

**2. Prepare training data:**

Multiple data formats are supported depending on the training framework and approach:

```bash
# Standard SFT format (parquet, for VeRL)
python train_GUI_360/data_preparation/prepare_gui360_sft_parquet.py

# With reasoning/thinking supervision (recommended)
python train_GUI_360/data_preparation/prepare_gui360_sft_with_thinking.py

# With accessibility (a11y) information
python train_GUI_360/data_preparation/prepare_gui360_sft_parquet_a11y.py
```

Output parquet files are saved to `train_GUI_360/data/`:

| File | Samples | Description |
|------|---------|-------------|
| `gui360_train_sft.parquet` | 81,440 | Standard SFT format |
| `gui360_eval_sft.parquet` | 20,360 | Validation split |
| `gui360_train_sft_with_thinking.parquet` | 105,368 | With reasoning supervision |
| `gui360_test_sft_matched.parquet` | 26,284 | Test set (train format) |
| `gui360_test_sft_eval_format.parquet` | 26,284 | Test set (eval format) |

**3. Prepare data for LlamaFactory** (alternative training framework):

```bash
python train_GUI_360/llamafactory/prepare_data.py \
    --input datasets/GUI-360/processed_data/action_prediction_train_resize/training_data.json \
    --output train_GUI_360/llamafactory/data/gui360_train.json \
    --image-base-dir datasets/GUI-360/processed_data/action_prediction_train_resize \
    --val-size 100
```

---

## Training

### UI-S1 Semi-online RL Training

```bash
bash scripts/train_example.sh
python scripts/model_merger.py merge --local_dir checkpoints/XXX
```

### GUI-360 SFT Training

**With VeRL:**

```bash
# Configure: train_GUI_360/config/gui360_sft.yaml
# Submit SLURM job from train_GUI_360/sft_scripts/
```

**With LlamaFactory:**

```bash
# Configure: train_GUI_360/llamafactory/qwen25vl_gui360_full_sft.yaml
llamafactory-cli train train_GUI_360/llamafactory/qwen25vl_gui360_full_sft.yaml
```

**MoE SFT Training** (Qwen2.5-VL-7B + 6x ExpertLoRA):

```bash
# Configure: train_GUI_360/moe_sft/moe_sft_config.yaml
# Submit: sbatch train_GUI_360/moe_sft/train_moe_sft.slurm

# Merge MoE checkpoint to HuggingFace format
python train_GUI_360/moe_sft/merge_moe_to_hf.py
```

See [train_GUI_360/docs/GUI-360_SFT_Training_Guide.md](train_GUI_360/docs/GUI-360_SFT_Training_Guide.md) for detailed training documentation.

---

## Inference and Evaluation

```bash
# 1. Launch the vLLM server
vllm serve /checkpoints-7B \
    --served-model-name UI-S1-7B \
    --tensor_parallel_size 1 \
    --trust-remote-code \
    --limit-mm-per-prompt image=2

# 2. Evaluate UI-S1-7B's performance on SOP
python evaluation/eval_qwenvl.py --model_name UI-S1-7B

# Evaluate other models
python evaluation/eval_qwenvl.py --model_name Qwen2.5-VL-7B
python evaluation/eval_agentcpm.py --model_name AgentCPM-GUI-8B
python evaluation/eval_os-atlas-7b.py --model_name OS-Atlas-7B
python evaluation/eval_os-genesis-7b.py --model_name OS-Genesis-7B
python evaluation/eval_ui-tars-7b.py --model_name UI-TARS-7B
```

For HPC (SLURM):
```bash
sbatch install_scripts/eval_ui_s1.slurm
```

---

## Project Structure

```
UI-S1/
  setup.py                          # Package definition and dependencies
  requirements.txt                  # Full development dependencies
  install_scripts/                  # SLURM installation & data download scripts
    install_triton.slurm            #   Build Triton 3.2 from source
    install_all.slurm               #   Full installation (PyTorch + FA2 + vLLM + UI-S1)
    install_flash.slurm             #   Flash Attention standalone build
    install_vllm.slurm              #   vLLM v0.8.5 with FA3 disabled
    fix_versions.slurm              #   Fix PyTorch/vLLM version mismatch
    download_dataset.slurm          #   Download UI-S1 dataset from HuggingFace
    download_images.slurm           #   Download AndroidControl training images
    download_eval_images.slurm      #   Download evaluation images
  datasets/                         # Downloaded datasets
  verl/                             # VeRL framework (modified)
    models/moe/                     #   MoE module: router, expert LoRA, wrapper
  train/                            # UI-S1 RL training scripts
  train_GUI_360/                    # GUI-360 SFT training pipeline
    data_preparation/               #   Data conversion scripts
    data/                           #   Processed parquet datasets
    config/                         #   VeRL training configs
    llamafactory/                   #   LlamaFactory training configs
    moe_sft/                        #   MoE SFT training scripts
    docs/                           #   Training documentation
  evaluation/                       # Evaluation scripts and datasets
  scripts/                          # Utility scripts
  docs/                             # Project documentation
    installation_guide.md           #   Detailed installation guide
  Option-incentivized-MoE/          #   Option-Incentivized MoE research
```

---

## News

- **`2025-10-28`**: We release part of our training [dataset](https://huggingface.co/datasets/mPLUG/UI_S1_dataset).
- **`2025-09-17`**: We release the UI-S1 training and evaluation code.
- **`2025-09-16`**: We release the [checkpoints](https://huggingface.co/mPLUG/UI-S1-7B) of UI-S1-7B model.
- **`2025-09-16`**: We release our [paper](https://arxiv.org/abs/2509.11543).

## Citation

If you find this project useful, welcome to cite us.

```bibtex
@article{lu2025ui,
  title={UI-S1: Advancing GUI Automation via Semi-online Reinforcement Learning},
  author={Lu, Zhengxi and Ye, Jiabo and Tang, Fei and Shen, Yongliang and Xu, Haiyang and Zheng, Ziwei and Lu, Weiming and Yan, Ming and Huang, Fei and Xiao, Jun and others},
  journal={arXiv preprint arXiv:2509.11543},
  year={2025}
}
```

## Acknowledgements

We sincerely thank projects [verl](https://github.com/volcengine/verl) and [verl-agent](https://github.com/langfengQ/verl-agent).
