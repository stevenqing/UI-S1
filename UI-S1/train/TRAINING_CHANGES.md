# UI-S1 Training Configuration Changes

## Date: 2026-02-01

## Problem
Training was failing with OOM (Out of Memory) errors on multi-node Ray + FSDP setup with 16 GPUs (4 nodes × 4 GPUs).

## Root Causes
1. **vLLM memory conflict**: vLLM pre-allocated 90% GPU memory, leaving insufficient room for FSDP gradient computation
2. **NCCL communication issues**: Socket errors and timeouts on HPE Slingshot / GH200 systems
3. **Optimizer memory**: Optimizer states consuming too much GPU memory during actor update

---

## Parameter Changes (train_ui_s1.slurm)

| Parameter | Original | Final | Reason |
|-----------|----------|-------|--------|
| `gpu_memory_utilization` | 0.9 | **0.4** | Leave 60% GPU memory for FSDP operations |
| `max_model_len` | 32678 | **16384** | Reduce vLLM KV cache size |
| `optimizer_offload` | False | **True** | Offload optimizer states to CPU RAM |
| `rollout.n` | 8 | **4** | Fewer rollout samples = less memory per batch |

---

## NCCL Configuration (env_config.sh)

```bash
# Use specific high-speed network interface (not exclusion pattern)
export NCCL_SOCKET_IFNAME=hsn0
export GLOO_SOCKET_IFNAME=hsn0

# Force socket transport (bypass AWS-OFI-NCCL plugin issues)
export NCCL_NET=Socket
export NCCL_IB_DISABLE=1

# Disable P2P to avoid memory issues on GH200
export NCCL_P2P_DISABLE=1

# Minimal socket threads to reduce memory overhead
export NCCL_SOCKET_NTHREADS=1
export NCCL_NSOCKS_PERTHREAD=1

# Force IPv4 to avoid binding issues
export NCCL_SOCKET_FAMILY=AF_INET

# Increase timeout for large FSDP all-gather operations (60 min)
export NCCL_TIMEOUT=3600
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=7200
```

---

## NCCL Configuration (verl/__init__.py)

Added environment variables BEFORE torch import to ensure they take effect:

```python
import os

# Set NCCL env vars before any torch imports
os.environ.setdefault("NCCL_SOCKET_IFNAME", "hsn0")
os.environ.setdefault("GLOO_SOCKET_IFNAME", "hsn0")
os.environ.setdefault("NCCL_NET", "Socket")
os.environ.setdefault("NCCL_IB_DISABLE", "1")
os.environ.setdefault("NCCL_P2P_DISABLE", "1")
os.environ.setdefault("NCCL_TIMEOUT", "3600")
# ... etc
```

---

## How to Run Training

### 1. Submit the job
```bash
cd /scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/train
sbatch train_ui_s1.slurm
```

### 2. Monitor job status
```bash
squeue -u shuqing.a5l
```

### 3. Check training progress
```bash
# View latest log
tail -50 /scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/train/logs/train_<JOB_ID>.log

# Check training steps
grep "Training Progress:" /scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/train/logs/train_<JOB_ID>.log

# Check for errors
grep -E "OOM|Error|error" /scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/train/logs/train_<JOB_ID>.log
```

### 4. View metrics on WandB
- Project: https://wandb.ai/k23048755/gui_traj_grpo

---

## Training Configuration Summary

```
Nodes: 4
GPUs per node: 4
Total GPUs: 16
Batch size: 32 (4 nodes × 8)
Rollout samples per prompt: 4
Model: Qwen2.5-VL-7B-Instruct
Total training steps: 93
Epochs: 3
Estimated time: ~13 hours
```

---

## Memory Usage (After Fixes)

- Max GPU memory allocated: **45.7 GB** (was OOM before)
- Max GPU memory reserved: **64.1 GB**
- CPU memory used: **418.3 GB** (optimizer offload)

---

## Files Modified

1. `/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/train/train_ui_s1.slurm`
2. `/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/train/env_config.sh`
3. `/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/verl/__init__.py`
4. `/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/verl/workers/fsdp_workers.py`
