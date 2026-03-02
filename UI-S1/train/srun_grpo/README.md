# srun-based UI-S1 GRPO Training

This folder contains the srun-based implementation of UI-S1 GRPO training that matches the functionality of the Ray-based version.

## Directory Structure

```
srun_grpo/
├── README.md                    # This file
├── CHANGELOG_srun_grpo.md       # Detailed changelog and feature comparison
├── scripts/
│   ├── train_srun_grpo_full.py      # Full GRPO trainer (HF generate)
│   ├── train_srun_grpo_full.slurm   # SLURM script for full version
│   ├── train_srun_grpo_vllm.py      # GRPO trainer with vLLM rollout
│   └── train_srun_grpo_vllm.slurm   # SLURM script for vLLM version
├── logs/                        # Training logs (auto-created)
│   └── train_grpo_*_%j.log      # SLURM job output logs
└── configs/                     # Optional custom configs
```

## Quick Start

### Option 1: vLLM Version (Recommended - Faster)

```bash
sbatch /scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/train/srun_grpo/scripts/train_srun_grpo_vllm.slurm
```

### Option 2: Basic Version (No vLLM dependency)

```bash
sbatch /scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/train/srun_grpo/scripts/train_srun_grpo_full.slurm
```

## Features

| Feature | Full Version | vLLM Version |
|---------|--------------|--------------|
| UI-S1 Advantage | ✅ | ✅ |
| Step Discounted Returns | ✅ | ✅ |
| DAPO Filtering | ✅ | ✅ |
| Reference Policy (KL) | ✅ | ✅ |
| Validation Loop | ✅ | ✅ |
| FSDP Training | ✅ | ✅ |
| WandB Logging | ✅ | ✅ |
| vLLM Rollout | ❌ | ✅ |
| Multi-modal Input | ❌ | ✅ |

## Key Parameters

Edit the SLURM scripts to modify these parameters:

### UI-S1 Algorithm
- `GAMMA=0.5` - Step-level discount factor
- `STEP_ADV_W=1.0` - Step-level advantage weight
- `EPISODE_ADV_W=1.0` - Episode-level advantage weight
- `ADV_MODE="mean_std_norm"` - Normalization mode

### GRPO
- `N_ROLLOUTS=4` - Number of rollouts per prompt
- `CLIP_RANGE=0.2` - PPO clip ratio
- `KL_COEF=0.0001` - KL penalty coefficient

### DAPO
- `DAPO_ENABLED=True` - Enable filtering
- `DAPO_THRESHOLD=0.3` - Std threshold

### vLLM (vllm version only)
- `VLLM_TP_SIZE=1` - Tensor parallel size
- `VLLM_GPU_UTIL=0.7` - GPU memory utilization

## Monitoring

### Check Job Status
```bash
squeue -u $USER
```

### View Logs
```bash
# Real-time log viewing
tail -f /scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/train/srun_grpo/logs/train_grpo_*_<JOB_ID>.log

# Error logs
cat /scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/train/srun_grpo/logs/train_grpo_*_<JOB_ID>.err
```

### WandB
Training metrics are logged to WandB under project `gui_traj_grpo_vllm` or `gui_traj_grpo_srun_full`.

## Comparison with Ray Version

See `CHANGELOG_srun_grpo.md` for detailed feature comparison between srun and Ray implementations.

## Troubleshooting

### NCCL Initialization Timeout
Check that `env_config.sh` has correct network interface settings:
```bash
export NCCL_SOCKET_IFNAME=hsn0
export GLOO_SOCKET_IFNAME=hsn0
```

### vLLM Memory Issues
Reduce `VLLM_GPU_UTIL` (e.g., 0.6) if running out of GPU memory.

### Weight Sync Failures
vLLM weight sync from FSDP may fail on some vLLM versions. The trainer will automatically fall back to HF generate.
