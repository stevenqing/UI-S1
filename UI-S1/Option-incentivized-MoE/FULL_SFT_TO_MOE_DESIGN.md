# Full-Parameter SFT → MoE Multi-LoRA: Design Document

## Problem Statement

Current MoE pipeline: `Base Model (Qwen2.5-VL-7B) + LoRA SFT v4 → MoE Multi-LoRA Experts`

LoRA SFT v4 在 Action Prediction 上只有 **27.53%**，而 Full-Parameter SFT v2 达到了 **46.90%** (+19.37%)。
LoRA 容量不足以学到复杂的 Action Prediction 能力。

我们需要将 Full-Parameter SFT 的强 AP 能力整合到 MoE Multi-LoRA 框架中。

### 性能对比 (GUI-360 Eval)

| Model | Grounding | Action Pred | A11y |
|-------|----------|-------------|------|
| Base Qwen2.5-VL-7B | 42.47% | 14.53% | 14.53% |
| LoRA v3 | 56.34% | 24.67% | 20.54% |
| LoRA v4 ckpt354 | 64.37% | 27.53% | 20.31% |
| **Full-Param SFT v2** | **70.56%** | **46.90%** | 17.51% |

---

## 方案一: SVD Weight Extraction (Full SFT → LoRA → MoE)

### 核心思想

将全参数 SFT 的知识通过 SVD 分解压缩回 LoRA 格式，然后复用现有的 `convert_sft_lora_to_moe.py` 流程。

```
W_sft (Full SFT weights)  -  W_base (Original base weights)  =  ΔW (Weight delta)
                                                                    │
                                                              SVD decomposition
                                                                    │
                                                            ΔW ≈ B @ A (LoRA format)
                                                                    │
                                                        convert_sft_lora_to_moe.py
                                                                    │
                                                            4 Expert LoRAs + Router
```

### 数学原理

对于每个 target module 的权重差异:

```
ΔW = W_sft - W_base    (shape: [out_features, in_features])

SVD(ΔW) = U @ Σ @ V^T

截取 top-r 个奇异值:
    lora_B = U[:, :r] @ diag(sqrt(Σ[:r]))    (shape: [out_features, r])
    lora_A = diag(sqrt(Σ[:r])) @ V^T[:r, :]  (shape: [r, in_features])

重构误差:
    ||ΔW - lora_B @ lora_A||_F / ||ΔW||_F
```

注意 LoRA 的 scaling factor: `y = Wx + (alpha/r) * B @ A @ x`，所以提取时需要考虑 scaling。

### 实现: `extract_fullsft_to_lora.py`

```python
#!/usr/bin/env python3
"""
Extract LoRA weights from Full-Parameter SFT via SVD decomposition.

Takes a full-param SFT model and the original base model, computes the
weight difference for each target module, and extracts low-rank LoRA
approximations via truncated SVD.

Usage:
    python extract_fullsft_to_lora.py \
        --sft_model train_GUI_360/llamafactory/output/gui360_full_sft_v2 \
        --base_model checkpoints/Qwen2.5-VL-7B-Instruct \
        --output train_GUI_360/moe_rl/extracted_lora_from_fullsft \
        --rank 32 \
        --alpha 64
"""

import argparse
import json
import os

import torch
from safetensors.torch import load_file, save_file


# Qwen2.5-VL-7B target modules (matching LoRA v4)
TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

# Number of transformer layers in Qwen2.5-VL-7B
NUM_LAYERS = 28


def load_model_weights(model_dir: str) -> dict:
    """Load all safetensors from a model directory into a single state dict."""
    index_file = os.path.join(model_dir, "model.safetensors.index.json")

    if os.path.exists(index_file):
        # Sharded model: load from index
        with open(index_file) as f:
            index = json.load(f)
        weight_map = index["weight_map"]

        state_dict = {}
        loaded_files = set()
        for key, filename in weight_map.items():
            if filename not in loaded_files:
                filepath = os.path.join(model_dir, filename)
                shard = load_file(filepath)
                state_dict.update(shard)
                loaded_files.add(filename)
        return state_dict
    else:
        # Single file
        filepath = os.path.join(model_dir, "model.safetensors")
        return load_file(filepath)


def get_target_keys(layer_idx: int, module_name: str) -> str:
    """Get the HuggingFace weight key for a target module."""
    if module_name in ["q_proj", "k_proj", "v_proj", "o_proj"]:
        return f"model.layers.{layer_idx}.self_attn.{module_name}.weight"
    elif module_name in ["gate_proj", "up_proj", "down_proj"]:
        return f"model.layers.{layer_idx}.mlp.{module_name}.weight"
    else:
        raise ValueError(f"Unknown module: {module_name}")


def get_peft_key(layer_idx: int, module_name: str, lora_type: str) -> str:
    """Get the PEFT-format key for LoRA weights."""
    if module_name in ["q_proj", "k_proj", "v_proj", "o_proj"]:
        prefix = f"base_model.model.model.layers.{layer_idx}.self_attn.{module_name}"
    elif module_name in ["gate_proj", "up_proj", "down_proj"]:
        prefix = f"base_model.model.model.layers.{layer_idx}.mlp.{module_name}"
    else:
        raise ValueError(f"Unknown module: {module_name}")
    return f"{prefix}.lora_{lora_type}.weight"


def extract_lora_svd(
    delta_w: torch.Tensor,
    rank: int,
    alpha: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Extract LoRA A and B matrices from weight delta using SVD.

    ΔW ≈ (alpha/r) * B @ A
    So we need: B @ A = (r/alpha) * ΔW_approx

    Args:
        delta_w: [out_features, in_features] weight difference
        rank: LoRA rank
        alpha: LoRA alpha (scaling)

    Returns:
        lora_A: [rank, in_features]
        lora_B: [out_features, rank]
    """
    scaling = alpha / rank

    # Compute SVD in float32 for numerical stability
    delta_float = delta_w.float()
    U, S, Vt = torch.linalg.svd(delta_float, full_matrices=False)

    # Truncate to rank
    U_r = U[:, :rank]       # [out_features, rank]
    S_r = S[:rank]           # [rank]
    Vt_r = Vt[:rank, :]     # [rank, in_features]

    # Compute reconstruction error
    delta_approx = U_r @ torch.diag(S_r) @ Vt_r
    error = torch.norm(delta_float - delta_approx) / torch.norm(delta_float)

    # Distribute singular values between A and B
    # Account for LoRA scaling: output = base + (alpha/r) * B @ A @ x
    # So B @ A should equal (r/alpha) * ΔW_approx = (1/scaling) * ΔW_approx
    sqrt_s = torch.sqrt(S_r / scaling)

    lora_B = U_r * sqrt_s.unsqueeze(0)       # [out_features, rank]
    lora_A = Vt_r * sqrt_s.unsqueeze(1)      # [rank, in_features]

    return lora_A.to(delta_w.dtype), lora_B.to(delta_w.dtype), error.item()


def extract(
    sft_model_dir: str,
    base_model_dir: str,
    output_dir: str,
    rank: int = 32,
    alpha: int = 64,
):
    """Extract LoRA from full-param SFT via SVD."""
    print(f"=== Extracting LoRA from Full-Param SFT ===")
    print(f"  SFT model:  {sft_model_dir}")
    print(f"  Base model: {base_model_dir}")
    print(f"  Output:     {output_dir}")
    print(f"  Rank: {rank}, Alpha: {alpha}")
    print()

    # 1. Load both models
    print("Loading SFT model weights...")
    sft_weights = load_model_weights(sft_model_dir)
    print(f"  Loaded {len(sft_weights)} weight tensors")

    print("Loading base model weights...")
    base_weights = load_model_weights(base_model_dir)
    print(f"  Loaded {len(base_weights)} weight tensors")
    print()

    # 2. Extract LoRA for each target module
    peft_state_dict = {}
    errors = []
    singular_value_stats = []

    for layer_idx in range(NUM_LAYERS):
        for module_name in TARGET_MODULES:
            hf_key = get_target_keys(layer_idx, module_name)

            if hf_key not in sft_weights or hf_key not in base_weights:
                print(f"  WARNING: {hf_key} not found, skipping")
                continue

            w_sft = sft_weights[hf_key]
            w_base = base_weights[hf_key]
            delta_w = w_sft - w_base

            # Check if delta is non-trivial
            delta_norm = torch.norm(delta_w.float()).item()
            if delta_norm < 1e-8:
                print(f"  Layer {layer_idx} {module_name}: delta ~0, skipping")
                continue

            # SVD extraction
            lora_A, lora_B, error = extract_lora_svd(delta_w, rank, alpha)

            # Store in PEFT format
            a_key = get_peft_key(layer_idx, module_name, "A")
            b_key = get_peft_key(layer_idx, module_name, "B")
            peft_state_dict[a_key] = lora_A
            peft_state_dict[b_key] = lora_B

            errors.append(error)

            if layer_idx % 7 == 0 and module_name == "q_proj":
                print(f"  Layer {layer_idx:2d} {module_name}: "
                      f"delta_norm={delta_norm:.4f}, "
                      f"reconstruction_error={error:.4f}, "
                      f"lora_A={lora_A.shape}, lora_B={lora_B.shape}")

    # 3. Save in PEFT format
    os.makedirs(output_dir, exist_ok=True)

    # Save as safetensors
    save_file(peft_state_dict, os.path.join(output_dir, "adapter_model.safetensors"))

    # Save adapter_config.json
    adapter_config = {
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "r": rank,
        "lora_alpha": alpha,
        "target_modules": TARGET_MODULES,
        "lora_dropout": 0.05,
        "bias": "none",
        "base_model_name_or_path": os.path.abspath(base_model_dir),
    }
    with open(os.path.join(output_dir, "adapter_config.json"), "w") as f:
        json.dump(adapter_config, f, indent=2)

    # 4. Report statistics
    print()
    print(f"=== Extraction Statistics ===")
    print(f"  Extracted {len(peft_state_dict)} weight tensors "
          f"({len(peft_state_dict) // 2} LoRA pairs)")
    print(f"  Mean reconstruction error: {sum(errors) / len(errors):.6f}")
    print(f"  Max reconstruction error:  {max(errors):.6f}")
    print(f"  Min reconstruction error:  {min(errors):.6f}")

    total_params = sum(t.numel() for t in peft_state_dict.values())
    print(f"  Total LoRA parameters: {total_params:,} ({total_params / 1e6:.1f}M)")

    # 5. Rank analysis: check how much energy is captured
    print()
    print(f"=== Rank Analysis (suggested rank vs energy captured) ===")
    # Re-do SVD for one representative layer to show energy distribution
    sample_key = get_target_keys(14, "q_proj")  # middle layer
    if sample_key in sft_weights:
        delta_sample = (sft_weights[sample_key] - base_weights[sample_key]).float()
        _, S_full, _ = torch.linalg.svd(delta_sample, full_matrices=False)
        total_energy = (S_full ** 2).sum()
        for r in [16, 32, 64, 128, 256]:
            energy_r = (S_full[:r] ** 2).sum() / total_energy
            print(f"  rank={r:3d}: energy_captured={energy_r:.4f} "
                  f"({energy_r * 100:.1f}%)")

    print()
    print(f"=== Output saved to {output_dir} ===")
    print(f"Next step: python convert_sft_lora_to_moe.py "
          f"--checkpoint {output_dir} --output <moe_init_dir>")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sft_model", type=str, required=True)
    parser.add_argument("--base_model", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--alpha", type=int, default=64)
    args = parser.parse_args()
    extract(args.sft_model, args.base_model, args.output, args.rank, args.alpha)
```

### Pipeline

```bash
# Step 1: Extract LoRA from Full-Param SFT via SVD
cd /scratch/a5l/shuqing.a5l/MobileAgent/UI-S1

python Option-incentivized-MoE/extract_fullsft_to_lora.py \
    --sft_model  train_GUI_360/llamafactory/output/gui360_full_sft_v2 \
    --base_model checkpoints/Qwen2.5-VL-7B-Instruct \
    --output     train_GUI_360/moe_rl/extracted_lora_from_fullsft \
    --rank 64 \
    --alpha 128

# Step 2: Convert to MoE format (existing script)
python train_GUI_360/moe_rl/convert_sft_lora_to_moe.py \
    --checkpoint train_GUI_360/moe_rl/extracted_lora_from_fullsft \
    --output     train_GUI_360/moe_rl/moe_fullsft_svd_init \
    --num_experts 4 \
    --perturbation_std 0.01

# Step 3: MoE RL training (use existing YAML, just change moe_checkpoint)
python -m verl.trainer.main_dapo \
    --config-path=train_GUI_360/moe_rl \
    --config-name=traj_grpo_moe_v4 \
    actor_rollout_ref.model.moe.moe_checkpoint=train_GUI_360/moe_rl/moe_fullsft_svd_init \
    actor_rollout_ref.model.moe.expert_lora_r=64 \
    actor_rollout_ref.model.moe.expert_lora_alpha=128
```

### YAML Config: `traj_grpo_moe_v5_svd.yaml`

基于 `traj_grpo_moe_v4.yaml`，主要改动:

```yaml
actor_rollout_ref:
  model:
    path: /scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/checkpoints/Qwen2.5-VL-7B-Instruct  # 不变
    moe:
      enabled: true
      num_experts: 4
      top_k: 2

      # 提高 rank 以更好捕获 full SFT 的知识
      expert_lora_r: 64        # 从 32 提高到 64
      expert_lora_alpha: 128   # 保持 alpha/r = 2

      target_modules: [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]

      # SVD 提取的 warm-start
      moe_checkpoint: /scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/train_GUI_360/moe_rl/moe_fullsft_svd_init

trainer:
  project_name: gui360_moe_rl_v5_svd
  experiment_name: moe_v5_fullsft_svd_r64
```

### 优缺点

| | 详情 |
|---|------|
| **优点** | 复用现有 `convert_sft_lora_to_moe.py` 和 MoE training pipeline，改动最小 |
| **优点** | Base model 不变 (Qwen2.5-VL-7B)，reference policy 兼容 |
| **优点** | 可以通过调高 rank (64/128) 来减少 SVD 压缩损失 |
| **缺点** | SVD 压缩有信息损失。rank=32 可能只捕获 70-80% 能量 |
| **缺点** | rank 越高，LoRA 参数越多，训练和推理成本增加 |
| **缺点** | 每个 expert 都从同一个 SVD 结果初始化（只有微扰差异），多样性有限 |

### Rank 选择建议

| Rank | 预估能量捕获 | LoRA 参数/Expert | 适用场景 |
|------|------------|-----------------|---------|
| 32 | ~70-80% | ~40M | 快速实验，对精度要求不高 |
| 64 | ~85-90% | ~80M | **推荐**。平衡精度和效率 |
| 128 | ~95%+ | ~160M | 追求最高精度，但参数量大 |

运行 `extract_fullsft_to_lora.py` 时会输出实际的 energy captured 统计，以此选择合适的 rank。

---

## 方案二: Full-Param SFT as New Base Model (SFT Base + MoE LoRA)

### 核心思想

直接将 Full-Param SFT 模型作为新的 Base Model（frozen），MoE Expert LoRAs 在这个更强的 base 上学习 RL 增量。

```
原方案:   Base (Qwen2.5-VL-7B, frozen) + Expert LoRAs (学 SFT + RL)
新方案:   Base (Full-Param SFT v2, frozen) + Expert LoRAs (只学 RL 增量)
```

### 架构对比

```
方案二架构:

    Input: (screenshot, instruction)
              │
              ▼
    ┌─────────────────────────────────────────┐
    │   Full-Param SFT v2 (Frozen)            │  ← 已有 46.9% AP 能力
    │   target Linear 替换为 MoELoRALinear     │
    └─────────────────────────────────────────┘
              │
    Pass 1: LoRA disabled → hidden_states → Router → routing_weights
              │
    Pass 2: LoRA enabled → logits + loss
              │
    LoRA experts 只需学习 RL 增量 (更小的 delta)
```

### 实现步骤

#### Step 1: 确认 Full-Param SFT v2 模型路径

```
/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/train_GUI_360/llamafactory/output/gui360_full_sft_v2/
├── model-00001-of-00004.safetensors    (4.97GB)
├── model-00002-of-00004.safetensors    (4.99GB)
├── model-00003-of-00004.safetensors    (4.93GB)
├── model-00004-of-00004.safetensors    (1.69GB)
├── model.safetensors.index.json
├── config.json
└── ...
```

模型已经是 HuggingFace 格式，可以直接用 `from_pretrained()` 加载。

#### Step 2: Expert LoRA 初始化

Expert LoRAs 从零（或近零）初始化，因为:
- SFT 的知识已经在 frozen base 中
- LoRA 只需学习 RL 探索的增量
- 不需要 SVD 提取

```python
# Expert初始化: 标准 Kaiming init (和现有 LoRALayer.__init__ 一致)
# lora_A: Kaiming uniform
# lora_B: zeros (LoRA 标准初始化，初始 delta = 0)
```

#### Step 3: Reference Policy 处理

**关键问题**: RL 训练中的 KL 散度需要 reference policy。

当前配置中 reference model 和 actor model 使用相同的 `model.path`:

```yaml
# 当前: actor 和 ref 都加载 Qwen2.5-VL-7B
actor_rollout_ref:
  model:
    path: checkpoints/Qwen2.5-VL-7B-Instruct
```

方案二中，**reference policy 也应该是 Full-Param SFT v2**，因为我们希望 KL 约束 actor 不要偏离 SFT 太远:

```yaml
# 方案二: actor 和 ref 都加载 Full-Param SFT v2
actor_rollout_ref:
  model:
    path: train_GUI_360/llamafactory/output/gui360_full_sft_v2
```

#### Step 4: YAML Config: `traj_grpo_moe_v5_sftbase.yaml`

```yaml
# MoE RL Training with Full-Param SFT v2 as Base Model
#
# Key changes from v4:
# - Base model: Full-Param SFT v2 (not original Qwen2.5-VL-7B)
# - Expert LoRAs: initialized from zero (not warm-started from SFT LoRA)
# - No moe_checkpoint needed (experts start fresh)
# - Reference policy: Full-Param SFT v2

hydra:
  searchpath:
    - file:///scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/verl/trainer/config

defaults:
  - ppo_trainer
  - _self_

data:
  max_prompt_length: 8192
  max_response_length: 128
  train_batch_size: 8
  return_raw_chat: False
  filter_overlong_prompts: True
  reward_fn_key: data_source
  custom_cls:
    path: /scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/verl/utils/dataset/rl_dataset.py
    name: TrajDataset
  train_files: /scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/datasets/GUI-360/rl_data/gui360_train.jsonl
  val_files: /scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/datasets/GUI-360/rl_data/gui360_test.jsonl

actor_rollout_ref:
  model:
    # >>> 核心改动: 使用 Full-Param SFT v2 作为 base <<<
    path: /scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/train_GUI_360/llamafactory/output/gui360_full_sft_v2
    trust_remote_code: true
    enable_gradient_checkpointing: true
    use_remove_padding: true

    moe:
      enabled: true
      num_experts: 4
      top_k: 2

      # LoRA rank 可以小一些，因为只学 RL 增量
      expert_lora_r: 32
      expert_lora_alpha: 64
      expert_lora_dropout: 0.05

      target_modules:
        - q_proj
        - k_proj
        - v_proj
        - o_proj
        - gate_proj
        - up_proj
        - down_proj

      router_hidden: 256
      router_dropout: 0.1
      router_temperature: 0.5
      pooling_strategy: mean

      balance_weight: 0.2
      balance_type: mse
      z_loss_weight: 0.01

      use_vectorized_routing: true
      freeze_router_epochs: 0

      # >>> 无 warm-start: experts 从零初始化 <<<
      # moe_checkpoint: null  (不设置即可)

      log_routing_matrix_freq: 100

  actor:
    strategy: fsdp
    ppo_mini_batch_size: 64
    ppo_micro_batch_size_per_gpu: 2
    optim:
      lr: 1e-5
      weight_decay: 0.01
      lr_warmup_steps_ratio: 0.1
    ppo_epochs: 1
    grad_clip: 0.5
    clip_ratio: 0.1
    use_kl_loss: true
    kl_loss_coef: 0.1
    kl_loss_type: low_var_kl
    loss_agg_mode: token-mean
    use_torch_compile: false
    fsdp_config:
      param_offload: false
      optimizer_offload: false
      fsdp_size: -1
    checkpoint:
      contents: [model, hf_model, optimizer, extra]

  rollout:
    name: vllm
    mode: sync
    temperature: 1.0
    top_p: 1.0
    do_sample: true
    n: 4
    tensor_model_parallel_size: 1
    gpu_memory_utilization: 0.7
    max_model_len: 16384
    max_num_batched_tokens: 32768
    enforce_eager: true
    free_cache_engine: true
    load_format: dummy_dtensor
    enable_chunked_prefill: false
    limit_images: 2
    log_prob_micro_batch_size_per_gpu: 1
    val_kwargs:
      temperature: 0
      do_sample: false
      n: 1

  ref:
    strategy: fsdp
    log_prob_micro_batch_size_per_gpu: 1
    fsdp_config:
      param_offload: false

algorithm:
  adv_estimator: uis1
  norm_adv_by_std_in_grpo: true
  gamma: 0.5
  lam: 1.0
  use_kl_in_reward: false
  kl_penalty: kl
  patch_threshold: 2
  hint: false
  actions_only: null
  uis1:
    episode_advantage_w: 1.0
    step_advantage_w: 1.0
    mode: mean_std_norm
  filter_groups:
    enable: false
    max_num_gen_batches: 10
    metric: seq_future_reward
    std_threshold: 0.3

trainer:
  total_epochs: 10
  balance_batch: true
  project_name: gui360_moe_rl_v5_sftbase
  experiment_name: moe_v5_sftbase_r32
  logger: [console, wandb]
  save_freq: 50
  default_local_dir: checkpoints/${trainer.project_name}/${trainer.experiment_name}
  val_before_train: false
  test_freq: 25
  critic_warmup: 0
  nnodes: 8
  n_gpus_per_node: 4

reward_model:
  enable: false
  reward_manager: naive
```

### Pipeline

```bash
# 方案二极其简单——无需转换步骤，直接训练
cd /scratch/a5l/shuqing.a5l/MobileAgent/UI-S1

python -m verl.trainer.main_dapo \
    --config-path=train_GUI_360/moe_rl \
    --config-name=traj_grpo_moe_v5_sftbase
```

### 代码改动

**不需要任何代码改动！**

现有的 `MoEVLMWrapper` 和 `moe_dapo_trainer.py` 已经支持:
- `_freeze_base_model()`: 自动冻结 base model 的所有参数
- `_replace_target_modules()`: 将 Linear 替换为 MoELoRALinear
- 不设置 `moe_checkpoint` 时: expert LoRAs 自动从零初始化 (lora_A: Kaiming, lora_B: zeros)
- Reference policy: 自动从 `model.path` 加载同一模型

唯一需要做的是创建新的 YAML 配置文件，将 `model.path` 指向 SFT 模型。

### 优缺点

| | 详情 |
|---|------|
| **优点** | 零信息损失——SFT 的 46.9% AP 能力完整保留在 frozen base 中 |
| **优点** | 零代码改动——只需改 YAML 中的 `model.path` |
| **优点** | Expert LoRAs 只需学 RL 增量（更小的 delta），收敛可能更快 |
| **优点** | 可以用较小的 rank (32)，因为 RL 调整通常比 SFT 更小 |
| **缺点** | Base model 变大（15.6GB vs 原来的 config 指向同一路径） |
| **缺点** | Rollout/vLLM 需要加载 SFT 模型（但显存不变，因为 base 本来就 frozen） |
| **缺点** | 和其他非 SFT base 的实验不兼容（不同 base 不能直接对比 KL） |

---

## 方案对比

| 维度 | 方案一 (SVD Extraction) | 方案二 (SFT as Base) |
|------|------------------------|---------------------|
| **Base Model** | Qwen2.5-VL-7B (原始) | Full-Param SFT v2 |
| **Expert 初始化** | SVD 提取的 LoRA (warm-start) | 零初始化 (cold-start) |
| **SFT 知识保留** | 部分 (SVD 压缩损失) | 完整 (在 frozen base 中) |
| **代码改动** | 需要新增 `extract_fullsft_to_lora.py` | 无 (只改 YAML) |
| **LoRA Rank** | 建议 64+ (补偿 SVD 损失) | 32 即可 (只学 RL 增量) |
| **参数量/Expert** | ~80M (rank=64) | ~40M (rank=32) |
| **Reference Policy** | 和现有实验一致 (Qwen2.5-VL-7B) | 需要用 SFT 模型做 ref |
| **实现复杂度** | 中等 | 极低 |
| **实验对比** | 和 v4 直接对比 (同 base) | 不同 base，需要单独对比 |
| **Rollout 兼容** | 兼容现有 vLLM (同模型) | 需要加载 SFT 模型到 vLLM |

---

## 推荐策略

**先跑方案二，再用方案一做消融实验。**

理由:
1. 方案二零代码改动，可以立即启动
2. Full SFT 的 AP 能力完整保留，起点最高
3. Expert LoRAs 只学 RL delta，rank=32 够用，训练成本低
4. 方案一可以作为消融实验: 如果方案二效果好，方案一可以帮助理解"LoRA rank 到底需要多大才能近似 full SFT"

### 推荐实验顺序

```
实验 1: 方案二 (rank=32, SFT base)  ← 先跑这个
实验 2: 方案二 (rank=16, SFT base)  ← 如果 rank=32 收敛快，试试更小的 rank
实验 3: 方案一 (rank=64, SVD, original base) ← 对比: SVD 能恢复多少 SFT 知识
实验 4: 方案一 (rank=128, SVD, original base) ← 更高 rank 能否接近方案二
```

---

## 文件索引

| 文件 | 状态 | 说明 |
|------|------|------|
| `Option-incentivized-MoE/FULL_SFT_TO_MOE_DESIGN.md` | 本文档 | 两方案设计 |
| `Option-incentivized-MoE/extract_fullsft_to_lora.py` | **待创建** | 方案一: SVD 提取脚本 |
| `train_GUI_360/moe_rl/traj_grpo_moe_v5_svd.yaml` | **待创建** | 方案一: MoE RL config |
| `train_GUI_360/moe_rl/traj_grpo_moe_v5_sftbase.yaml` | **待创建** | 方案二: MoE RL config |
| `train_GUI_360/moe_rl/convert_sft_lora_to_moe.py` | 已有 | 复用 (方案一 Step 2) |
| `train_GUI_360/moe_rl/traj_grpo_moe_v4.yaml` | 已有 | 当前 baseline |
| `verl/models/moe/moe_wrapper.py` | 已有，无需改动 | MoE wrapper |
| `verl/models/moe/expert_lora.py` | 已有，无需改动 | Expert LoRA 实现 |
| `verl/trainer/ppo/moe_dapo_trainer.py` | 已有，无需改动 | MoE trainer |
| `train_GUI_360/llamafactory/output/gui360_full_sft_v2/` | 已有 | Full-Param SFT v2 模型 |
| `checkpoints/Qwen2.5-VL-7B-Instruct/` | 已有 | 原始 base model |
