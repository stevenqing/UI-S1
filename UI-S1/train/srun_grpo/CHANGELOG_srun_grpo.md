# Changelog: srun-based UI-S1 GRPO Training

## Overview

This document summarizes the implementation of full-featured srun-based GRPO training scripts that match the functionality of the Ray-based version (`train_ui_s1.slurm`).

---

## New Files Created

### 1. `train_srun_grpo_full.py`
**Basic full-featured GRPO trainer using HuggingFace generate**

| Feature | Status | Description |
|---------|--------|-------------|
| UI-S1 Advantage | ✅ | Episode + step-level advantage normalization |
| Step Discounted Returns | ✅ | Gamma-discounted returns with extract_match breakpoints |
| DAPO Filtering | ✅ | Filter low-variance prompt groups by std threshold |
| Reference Policy | ✅ | KL penalty (kl, abs, mse, low_var_kl) |
| Validation Loop | ✅ | Type match, extract match, reward metrics |
| FSDP | ✅ | Full shard with optional CPU offload |
| Checkpointing | ✅ | FSDP full state dict save |
| WandB Logging | ✅ | Comprehensive metrics tracking |

### 2. `train_srun_grpo_vllm.py`
**Full-featured GRPO trainer with vLLM rollout engine**

| Feature | Status | Description |
|---------|--------|-------------|
| All features from `train_srun_grpo_full.py` | ✅ | Inherited |
| vLLM Rollout Engine | ✅ | High-throughput generation with SamplingParams |
| FSDP-vLLM Weight Sync | ✅ | Sync actor weights to vLLM before rollout |
| Multi-modal Support | ✅ | Support for image inputs in vLLM |
| Automatic Fallback | ✅ | Falls back to HF generate if vLLM unavailable |

### 3. `train_srun_grpo_full.slurm`
SLURM script for running `train_srun_grpo_full.py`

### 4. `train_srun_grpo_vllm.slurm`
SLURM script for running `train_srun_grpo_vllm.py`

---

## Feature Comparison: Ray vs srun Versions

| Feature | Ray Version | srun Original | srun Full | srun vLLM |
|---------|-------------|---------------|-----------|-----------|
| **Rollout Generation** |
| vLLM rollout | ✅ | ❌ | ❌ | ✅ |
| HF generate | ✅ | ❌ | ✅ | ✅ (fallback) |
| Multi-modal input | ✅ | ❌ | ❌ | ✅ |
| **Advantage Estimation** |
| UI-S1 episode-level | ✅ | ✅ (simplified) | ✅ | ✅ |
| UI-S1 step-level | ✅ | ✅ (simplified) | ✅ | ✅ |
| Step discounted returns | ✅ | ✅ | ✅ | ✅ |
| **Policy Optimization** |
| PPO clipped loss | ✅ | ✅ | ✅ | ✅ |
| KL penalty types | ✅ (all) | ❌ | ✅ (all) | ✅ (all) |
| Reference policy | ✅ | ❌ | ✅ | ✅ |
| **Filtering** |
| DAPO filter | ✅ | ❌ | ✅ | ✅ |
| std threshold | ✅ | ❌ | ✅ | ✅ |
| **Training Infrastructure** |
| FSDP | ✅ | ✅ | ✅ | ✅ |
| CPU offload | ✅ | ❌ | ✅ | ✅ |
| Gradient checkpointing | ✅ | ❌ | ✅ | ✅ |
| **Monitoring** |
| WandB logging | ✅ | ✅ | ✅ | ✅ |
| Validation loop | ✅ | ❌ | ✅ | ✅ |
| Checkpointing | ✅ | ✅ | ✅ | ✅ |

---

## Key Implementations

### 1. UI-S1 Advantage Estimation

```python
def compute_uis1_outcome_advantage(
    token_level_rewards,  # (bs, response_length) - for episode normalization
    step_rewards,         # (bs,) - step-level discounted returns
    response_mask,        # (bs, response_length)
    prompt_uids,          # Group by prompt
    traj_uids,            # Group by trajectory
    step_ids,             # Step within trajectory
    step_advantage_w=1.0,
    episode_advantage_w=1.0,
    mode="mean_std_norm",
):
    # Episode-level: normalize by mean (optionally std) across rollouts of same prompt
    episode_adv = episode_norm_reward(token_level_rewards, ...)

    # Step-level: normalize by mean (optionally std) across same step of same prompt
    step_adv = step_norm_reward(step_rewards, ...)

    # Combine with configurable weights
    advantages = episode_advantage_w * episode_adv + step_advantage_w * step_adv
```

### 2. DAPO Filtering

```python
def apply_dapo_filter(batch_data, std_threshold=0.3):
    # Group by prompt_uid
    # Compute std of seq_future_reward per group
    # Keep groups with std > threshold (high variance = informative)
    # Filter batch to keep only these samples
```

### 3. vLLM Rollout Engine

```python
class VLLMRolloutEngine:
    def __init__(self, model_path, config, ...):
        self.inference_engine = LLM(
            model=model_path,
            tensor_parallel_size=config.tensor_parallel_size,
            distributed_executor_backend="external_launcher",
            gpu_memory_utilization=config.gpu_memory_utilization,
            ...
        )

    def sync_weights_from_fsdp(self, fsdp_model):
        # Get full state dict from FSDP
        # Load into vLLM model

    def generate_sequences(self, input_ids, attention_mask, n, temperature):
        # Use vLLM's high-throughput generation
        # Return responses and log_probs
```

### 4. KL Penalty Types

```python
def compute_kl_penalty(log_probs, ref_log_probs, kl_penalty_type):
    if kl_penalty_type == "kl":
        kl = ref_log_probs - log_probs
    elif kl_penalty_type == "abs":
        kl = torch.abs(log_probs - ref_log_probs)
    elif kl_penalty_type == "mse":
        kl = 0.5 * (log_probs - ref_log_probs) ** 2
    elif kl_penalty_type == "low_var_kl":
        log_ratio = log_probs - ref_log_probs
        kl = torch.exp(log_ratio) - 1 - log_ratio  # Low variance estimator
```

---

## Usage

### Option 1: Full version with HF generate (simpler, slower)
```bash
sbatch /scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/train/train_srun_grpo_full.slurm
```

### Option 2: vLLM version (faster, requires vLLM installation)
```bash
sbatch /scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/train/train_srun_grpo_vllm.slurm
```

---

## Configuration Parameters

### UI-S1 Algorithm
| Parameter | Default | Description |
|-----------|---------|-------------|
| `gamma` | 0.5 | Step-level discount factor |
| `step_advantage_w` | 1.0 | Step-level advantage weight |
| `episode_advantage_w` | 1.0 | Episode-level advantage weight |
| `mode` | "mean_std_norm" | Normalization mode (mean_norm or mean_std_norm) |

### GRPO
| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_rollouts` | 4 | Number of rollouts per prompt |
| `clip_range` | 0.2 | PPO clip ratio |
| `kl_coef` | 0.0001 | KL penalty coefficient |
| `kl_loss_type` | "low_var_kl" | KL estimator type |

### DAPO Filtering
| Parameter | Default | Description |
|-----------|---------|-------------|
| `enable` | True | Enable DAPO filtering |
| `std_threshold` | 0.3 | Minimum std to keep prompt group |
| `metric` | "seq_future_reward" | Metric to use for filtering |

### vLLM (vllm version only)
| Parameter | Default | Description |
|-----------|---------|-------------|
| `tensor_parallel_size` | 1 | vLLM tensor parallel size |
| `gpu_memory_utilization` | 0.7 | GPU memory for vLLM |
| `max_model_len` | 8192 | Maximum model context length |
| `enforce_eager` | True | Disable CUDA graphs |

---

## Performance Notes

1. **vLLM vs HF generate**: vLLM can be 2-5x faster for generation, but requires careful memory management when co-existing with FSDP training.

2. **Memory Usage**:
   - With vLLM: Higher peak memory due to separate vLLM engine
   - Recommendation: Use `gpu_memory_utilization=0.7` for vLLM to leave room for FSDP

3. **Weight Sync Overhead**:
   - Syncing FSDP weights to vLLM adds overhead
   - Only sync at the beginning of each rollout phase, not per batch

4. **DAPO Filtering**:
   - Can significantly reduce training time by skipping uninformative samples
   - May reduce effective batch size if threshold is too high

---

## Known Limitations

1. **Multi-modal Support**: Currently simplified; full implementation would need `QwenMessages2Inputs` integration

2. **Multi-Round Generation**: Simplified compared to Ray version's `MultiRoundGenerator`

3. **Async Rollout**: Not implemented; Ray version supports async rollout for better throughput

4. **Hybrid Engine**: Not implemented; Ray version can share weights between FSDP and vLLM more efficiently

---

## Files Modified

None - all new files added to `/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/train/`

---

## Date

2026-02-01
