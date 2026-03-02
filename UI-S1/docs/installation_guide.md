# UI-S1 Installation Guide

## Overview

UI-S1 is built on top of **VERL** (Volcano Engine Reinforcement Learning, v0.4.0.dev) and uses **Qwen2.5-VL** as the base model for GUI automation via semi-online reinforcement learning. The installation varies depending on your hardware platform.

---

## Quick Install (x86_64 / Standard GPU)

For standard x86_64 machines with NVIDIA GPUs (A100, H100, etc.):

```bash
# 1. Create conda environment
conda create -n ui-s1 python=3.11
conda activate ui-s1

# 2. Clone and install the package
cd UI-S1
pip install -e .

# 3. Install vLLM and Flash Attention
pip install vllm==0.8.2
pip install flash-attn==2.7.4.post1 --no-build-isolation
# Or install from prebuilt wheel:
# pip install flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp311-cp311-linux_x86_64.whl
```

For training visualization, replace your own SwanLab API key and host in `verl/utils/tracking.py`.

---

## Full Install (GH200 / aarch64 + sm90 HPC)

For NVIDIA GH200 (aarch64 architecture, sm90 GPU) on HPC systems with SLURM, all components must be built from source. The installation order matters due to dependencies.

### Prerequisites

| Component | Version | Notes |
|-----------|---------|-------|
| Python | 3.11 | Via conda |
| CUDA | 12.6 | System module |
| GCC | 12.3 | System module (gcc-native/12.3) |
| Conda | - | Pre-installed |

### Step 0: Create the Conda Environment

```bash
conda create -n ui-s1 python=3.11
conda activate ui-s1
```

### Step 1: Build Triton 3.2 from Source (Prerequisite)

Triton must be built first since Flash Attention and vLLM depend on it.

**Offline packages required** (download before submitting the job):
```bash
mkdir -p ~/.triton/offline

# 1. Triton source with submodules
git clone --recursive https://github.com/triton-lang/triton
cd triton && git checkout v3.2.2 && git submodule update --init --recursive
cd .. && tar -czf triton-3.2-recursive.tar.gz triton
mv triton-3.2-recursive.tar.gz ~/.triton/offline/

# 2. LLVM 17.0.6 source
# Download from: https://github.com/llvm/llvm-project/releases/tag/llvmorg-17.0.6
# Place as ~/.triton/offline/llvm-project-17.0.6.tar.gz (or .tar.xz)
```

**Submit the build job:**
```bash
sbatch install_scripts/install_triton.slurm
```

This script:
1. Extracts LLVM and Triton sources
2. Builds LLVM/MLIR 17 with cmake (targets: NVPTX, AArch64, X86)
3. Builds Triton 3.2 Python package against the custom LLVM
4. Fixes GLIBCXX linking issues
5. Verifies installation

**SLURM resources:** 1 node, 64 CPUs, 100GB RAM, 1 GPU, ~12 hours

### Step 2: Install Everything Else (All-in-One)

```bash
sbatch install_scripts/install_all.slurm
```

This script performs the following in order:

1. **Base build tools**: pip, setuptools, wheel, ninja, cmake
2. **PyTorch 2.6.0** with CUDA 12.6 (`--index-url https://download.pytorch.org/whl/cu126`)
3. **Flash Attention 2** v2.7.0.post2 (built from source)
4. **vLLM** v0.6.4.post1 (built from source with FA2 support, no FA3)
5. **Fix PyTorch version** (vLLM may downgrade it)
6. **UI-S1 package** (`pip install -e .`)

**SLURM resources:** 1 node, 64 CPUs, 200GB RAM, 1 GPU, ~12 hours

### Alternative: Install vLLM v0.8.5.post1

If you need a newer vLLM version with FA3 disabled:

```bash
sbatch install_scripts/install_vllm.slurm
```

This script applies a surgical patch to vLLM source code to disable Flash Attention 3 (which causes ptxas issues on GH200), then builds from source.

**SLURM resources:** 1 node, 288 CPUs, 128GB RAM, 1 GPU, ~24 hours

### Step 3: Fix Version Mismatches (If Needed)

```bash
sbatch install_scripts/fix_versions.slurm
```

Reinstalls PyTorch 2.6.0 and UI-S1 package if versions got out of sync.

---

## Core Dependencies (from `setup.py`)

### Required

| Package | Version Constraint | Purpose |
|---------|-------------------|---------|
| accelerate | latest | Distributed training |
| codetiming | latest | Performance timing |
| datasets | latest | HuggingFace datasets |
| dill | latest | Serialization |
| hydra-core | latest | Configuration management |
| numpy | latest | Numerical computing |
| pandas | latest | Data manipulation |
| peft | latest | Parameter-efficient fine-tuning (LoRA) |
| pyarrow | >= 19.0.0 | Parquet file I/O |
| pybind11 | latest | C++/Python binding |
| pylatexenc | latest | LaTeX encoding |
| ray[default] | >= 2.41.0 | Distributed computing framework |
| torchdata | latest | Data loading |
| tensordict | <= 0.6.2 | Tensor dictionary utilities |
| transformers | == 4.51.1 | Model architectures (Qwen2.5-VL) |
| wandb | latest | Experiment tracking |
| qwen_vl_utils | latest | Qwen VL model utilities |
| packaging | >= 20.0 | Version utilities |
| colorlog | latest | Colored logging |
| squirrel-core | latest | Data pipeline |
| json5 | latest | JSON5 parsing |
| retry | latest | Retry decorators |
| modelscope | latest | Model hub |
| ninja | latest | Build system |
| swanlab | == 0.6.3 | Training visualization |

### Optional Extras

Install with `pip install -e ".[extra_name]"`:

| Extra | Packages | Use Case |
|-------|----------|----------|
| `gpu` | liger-kernel, flash-attn | GPU optimization |
| `vllm` | tensordict<=0.6.2, vllm==0.8.2 | vLLM inference |
| `sglang` | sglang==0.4.6.post5, torch-memory-saver, torch==2.6.0 | SGLang backend |
| `trl` | trl<=0.9.6 | TRL training |
| `test` | pytest, pre-commit, py-spy | Testing |
| `math` | math-verify | Math evaluation |
| `prime` | pyext | PRIME reward |
| `geo` | mathruler | Geometry tasks |

---

## Environment Variables for Training

Source `train/env_config.sh` before running training jobs:

```bash
source train/env_config.sh
```

Key settings configured:

| Category | Variables | Notes |
|----------|----------|-------|
| **CUDA** | `LD_LIBRARY_PATH` with CUDA 12.6 paths | Required for GPU ops |
| **Ray** | `RAY_DISABLE_DOCKER_CPU_WARNING=1`, `RAY_DISABLE_DASHBOARD=1` | Cluster config |
| **NCCL** | `NCCL_SOCKET_IFNAME=hsn0`, `NCCL_NET=Socket`, `NCCL_IB_DISABLE=1` | Multi-node comm via socket |
| **PyTorch** | `TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=10800` | 3hr timeout for checkpointing |
| **vLLM** | `VLLM_USE_V1=1` | Use vLLM V1 engine |
| **Debug** | `NCCL_DEBUG=INFO`, `TORCH_DISTRIBUTED_DEBUG=DETAIL` | Diagnostics |

---

## Data Setup

1. Download AndroidControl dataset:
   - Images into `datasets/AndroidControl/images/`
   - Training data: `datasets/android_control_train_example.jsonl`

2. Or use the 1000-example subset from HuggingFace:
   - [mPLUG/UI_S1_dataset](https://huggingface.co/datasets/mPLUG/UI_S1_dataset)

---

## Training

```bash
bash scripts/train_example.sh
python scripts/model_merger.py merge --local_dir checkpoints/XXX
```

---

## Inference & Evaluation

```bash
# Launch vLLM server
vllm serve /checkpoints-7B \
  --served-model-name UI-S1-7B \
  --tensor_parallel_size 1 \
  --trust-remote-code \
  --limit-mm-per-prompt image=2

# Evaluate
python evaluation/eval_qwenvl.py --model_name UI-S1-7B
```

---

## Optional: Qwen3-VL Evaluation Environment

A separate environment for evaluating Qwen3-VL models:

```bash
sbatch install_scripts/qwen3_eval/install_qwen3_eval.slurm
```

This creates a `qwen3-eval` conda environment with:
- vLLM >= 0.11.0
- transformers >= 4.52.0
- qwen-vl-utils == 0.0.14

---

## Verification

After installation, verify all components:

```python
import torch
print('PyTorch:', torch.__version__, 'CUDA:', torch.cuda.is_available())

import triton
print('Triton:', triton.__version__)

import flash_attn
print('Flash Attention:', flash_attn.__version__)

import vllm
print('vLLM:', vllm.__version__)

import verl
print('VERL:', verl.__version__)
```

Expected output (GH200):
```
PyTorch: 2.6.0 CUDA: True
Triton: 3.2.x
Flash Attention: 2.7.0.post2
vLLM: 0.6.4.post1 (or 0.8.5.post1)
VERL: 0.4.0.dev
```

---

## Known Environment Compatibility Issues (from RL Training)

The following issues were encountered and fixed during RL training (Task 6 in `Option-incentivized-MoE/TASK_LIST.md`). They document real version conflicts between the `ui-s1` and `qwen3-eval` environments and GH200-specific runtime problems.

### Why `qwen3-eval` env cannot be used for RL training

The `qwen3-eval` environment (vLLM 0.15.1, PyTorch 2.9.1, transformers >= 4.52) is **incompatible** with the verl RL training pipeline. Four fatal issues were hit in sequence:

| Error | Root Cause |
|-------|-----------|
| `PermissionError: '/local/user/...'` | Ray cannot create temp dirs on compute nodes. Fix: set `RAY_TMPDIR=/tmp/ray_${USER}_${SLURM_JOB_ID}` in SLURM script |
| `ModuleNotFoundError: 'vllm.lora.models'` | vLLM 0.15.1 renamed `vllm.lora.models` to `vllm.lora.lora_model` |
| `RuntimeError: split_group not supported` | PyTorch 2.9.1 changed `DeviceMesh` API — `split_group` is no longer supported |
| `ModuleNotFoundError: 'vllm.worker'` | vLLM 0.15.1 completely restructured `vllm.worker` module — not patchable |

**Conclusion:** Use the `ui-s1` environment for all RL training.

### Post-install patches needed for `ui-s1` env

After basic installation, two manual patches are required for RL training with Qwen2.5-VL:

#### 1. `KeyError: 'mrope'` in transformers

If you upgrade transformers beyond 4.51.1, Qwen2.5-VL requires `mrope` (multi-resolution RoPE) in `ROPE_INIT_FUNCTIONS`, but some versions don't include it.

**Fix:** Patch the installed transformers:
```python
# In site-packages/transformers/modeling_rope_utils.py
# Add to ROPE_INIT_FUNCTIONS dict:
"mrope": _compute_default_rope_parameters,
```

#### 2. `ImportError: Numba needs NumPy 2.2 or less`

Force-reinstalling transformers can pull in numpy >= 2.4, which breaks numba.

**Fix:**
```bash
pip install "numpy<2.3"  # Downgrades to 2.2.x
```

### GH200-specific NCCL issues during RL training

#### `nccl does not support allgather_into_tensor_coalesced`

During FSDP-to-vLLM weight synchronization, `DTensor.full_tensor()` uses a coalesced allgather op that GH200's NCCL does not support.

**Fix (in `verl/workers/sharding_manager/fsdp_vllm.py`):**
- Detect GH200 GPU at startup
- Skip `full_tensor()` entirely on GH200
- Always use manual `dist.all_gather` with pad-gather-strip for uneven sharding

#### `Detected mismatch between collectives on ranks` (shape [428,1280] vs [424,1280])

After `full_tensor()` fails with try/except, NCCL sequence counters get corrupted. Subsequent `dist.all_gather` calls on different ranks gather different parameters silently.

**Fix:** The same GH200 detection + manual allgather path above. Never let `full_tensor()` fail-and-fallback — always take the manual path on GH200.

### RL training runtime issues

| Issue | Root Cause | Fix |
|-------|-----------|-----|
| `AttributeError: 'NoneType' has no attribute 'keys'` (protocol.py:537) | `main_dapo.py` selected wrong trainer class (`RayPPOTrainer` instead of `RayTrajDAPOTrainer`) based on `project_name` pattern matching | Add `+trainer.trainer_cls=traj_dapo` to config or CLI override |
| `AttributeError: 'NoneType' has no attribute 'enable'` | `FPseudoDAPORewardManager.__call__` didn't check for `overlong_buffer_cfg is None` | Add None guard in `f_pseudo_dapo.py` |
| `OutOfMemoryError` after ~9 steps | Single-node 4-GPU with `batch_size=8, gpu_mem=0.9` exhausts 120GB GH200 | Reduce to `batch_size=4, gpu_mem=0.5` or use multi-node (2 nodes x 4 GPU) |
| Validation OOM | 1,543-sample eval set too large | Use small validation set (50 samples) |
| Multi-node `execve(): No such file or directory` | `srun` doesn't inherit PATH, can't find `hostname`/`bash` | Use absolute paths `/usr/bin/hostname`, `/bin/bash` in SLURM scripts |
| Multi-node RAY_ADDRESS empty | Inner `bash -c` escapes `$head_node_ip`, shell can't find variable | Don't escape — let outer shell expand the variable |

### Verified working environment for RL training

| Package | Version | Notes |
|---------|---------|-------|
| Python | 3.11 | |
| PyTorch | 2.6.0 | CUDA 12.6 |
| vLLM | 0.8.5 | Compatible with verl |
| transformers | 4.57.6 | Needs manual `mrope` patch |
| numpy | 2.2.6 | Must be < 2.3 for numba |
| ray | 2.44.1 | |
| verl | 0.2.0.dev | Local dev version |

---

## Troubleshooting

### Build / Installation Issues

| Issue | Solution |
|-------|----------|
| `GLIBCXX_3.4.30 not found` | Add `/opt/cray/pe/gcc-native/12/lib64` to `LD_LIBRARY_PATH` and `LD_PRELOAD` |
| vLLM downgrades PyTorch | Run `fix_versions.slurm` or reinstall: `pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu126 --no-deps` |
| ptxas crash during vLLM build | Reduce parallelism: `MAX_JOBS=1 CMAKE_BUILD_PARALLEL_LEVEL=1` |
| FA3 build fails on GH200 | Use `install_vllm.slurm` which patches out FA3 |
| Triton import fails | Ensure LLVM/MLIR 17 was built successfully; check `LD_PRELOAD` for libstdc++ |
| `KeyError: 'mrope'` | Patch `modeling_rope_utils.py` to add `"mrope": _compute_default_rope_parameters` |
| `Numba needs NumPy 2.2 or less` | `pip install "numpy<2.3"` |

### Runtime / Training Issues

| Issue | Solution |
|-------|----------|
| Multi-node NCCL hang | Verify `NCCL_SOCKET_IFNAME=hsn0` matches your network interface; check `NCCL_DEBUG=INFO` logs |
| Checkpoint save timeout | Increase `TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC` (default 10800 = 3hr) |
| `nccl does not support allgather_into_tensor_coalesced` | Use manual `dist.all_gather` path in `fsdp_vllm.py` (auto-detected on GH200) |
| Ray `PermissionError` on compute nodes | Set `RAY_TMPDIR=/tmp/ray_${USER}_${SLURM_JOB_ID}` |
| Wrong trainer class selected | Add `+trainer.trainer_cls=traj_dapo` to Hydra CLI overrides |
| OOM during RL training | Reduce `batch_size`, `gpu_memory_utilization`, or go multi-node |
| Validation OOM | Use small validation set (50 samples in `gui360_val_small.jsonl`) |

---

## File Reference

| File | Purpose |
|------|---------|
| `setup.py` | Package definition and dependencies |
| `requirements.txt` | Full development dependencies |
| `install_scripts/install_triton.slurm` | Build Triton 3.2 from source |
| `install_scripts/install_all.slurm` | Full installation orchestrator |
| `install_scripts/install_flash.slurm` | Flash Attention standalone build |
| `install_scripts/install_vllm.slurm` | vLLM v0.8.5 with FA3 disabled |
| `install_scripts/fix_versions.slurm` | Fix PyTorch/vLLM version mismatch |
| `train/env_config.sh` | Runtime environment variables |
