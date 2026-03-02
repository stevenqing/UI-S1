# verl 训练框架集成

## 1. 现有训练架构概览

```
verl/trainer/
├── main_dapo.py              # DAPO 训练入口
├── ppo/
│   ├── dapo_ray_trainer.py   # RayTrajDAPOTrainer
│   ├── ray_trainer.py        # RayPPOTrainer (基类)
│   ├── core_algos.py         # PPO/DAPO 核心算法
│   └── reward.py             # 奖励函数
└── config/
    └── ppo_trainer.yaml      # 配置模板
```

### 1.1 关键类继承关系

```
RayPPOTrainer (ray_trainer.py)
        │
        ▼
RayTrajDAPOTrainer (dapo_ray_trainer.py)
        │
        ▼
MoERayTrajDAPOTrainer (新增: moe_dapo_trainer.py)
```

---

## 2. MoE Trainer 实现

### 2.1 新增文件

```python
# verl/trainer/ppo/moe_dapo_trainer.py

import torch
import torch.nn as nn
from typing import Dict, Any, Optional
from omegaconf import DictConfig

from verl.trainer.ppo.dapo_ray_trainer import RayTrajDAPOTrainer
from verl.trainer.ppo.moe_loss import MoELoss
from verl.models.moe.router import TextOnlyRouter, InstructionFeatureExtractor
from verl.models.moe.expert_lora import ExpertLoRACollection, MoEExpertApplier


class MoERayTrajDAPOTrainer(RayTrajDAPOTrainer):
    """
    支持 MoE 的 DAPO Trainer

    扩展点:
    1. 模型初始化: 添加 Router + Expert LoRAs
    2. Forward pass: 集成 MoE routing
    3. Loss 计算: 添加 balance loss
    4. 日志记录: 添加 routing 分析
    """

    def __init__(self, config: DictConfig, **kwargs):
        # 提取 MoE 配置
        self.moe_config = config.model.get('moe', {})
        self.moe_enabled = self.moe_config.get('enabled', False)

        if self.moe_enabled:
            self._validate_moe_config()

        super().__init__(config, **kwargs)

    def _validate_moe_config(self):
        """验证 MoE 配置"""
        required = ['num_experts', 'expert_lora_r', 'balance_weight']
        for key in required:
            if key not in self.moe_config:
                raise ValueError(f"MoE config missing required key: {key}")

    def _init_moe_components(self, actor_module: nn.Module):
        """
        初始化 MoE 组件

        在 actor module 加载后调用
        """
        if not self.moe_enabled:
            return

        hidden_size = actor_module.config.hidden_size
        num_layers = actor_module.config.num_hidden_layers

        # 1. 初始化 Router
        self.router = TextOnlyRouter(
            hidden_size=hidden_size,
            num_experts=self.moe_config['num_experts'],
            router_hidden=self.moe_config.get('router_hidden', 256),
            top_k=self.moe_config.get('top_k', 1),
        )

        # 2. 初始化 Expert LoRAs
        self.expert_collection = ExpertLoRACollection(
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_experts=self.moe_config['num_experts'],
            target_modules=self.moe_config.get('target_modules', ['q_proj', 'v_proj']),
            lora_r=self.moe_config['expert_lora_r'],
            lora_alpha=self.moe_config.get('expert_lora_alpha', 32),
        )

        # 3. 初始化 MoE Expert Applier
        self.expert_applier = MoEExpertApplier(
            expert_collection=self.expert_collection,
            router=self.router,
        )

        # 4. 初始化 Feature Extractor
        self.feature_extractor = InstructionFeatureExtractor(
            pooling_strategy='mean'
        )

        # 5. 初始化 MoE Loss
        self.moe_loss_fn = MoELoss(
            num_experts=self.moe_config['num_experts'],
            balance_weight=self.moe_config['balance_weight'],
            balance_type=self.moe_config.get('balance_type', 'mse'),
        )

        # 6. 注册 hooks 到 actor module
        self._register_expert_hooks(actor_module)

        # 7. 移动到正确的设备
        device = next(actor_module.parameters()).device
        self.router.to(device)
        self.expert_collection.to(device)

        print(f"[MoE] Initialized with {self.moe_config['num_experts']} experts")

    def _register_expert_hooks(self, actor_module: nn.Module):
        """
        注册 forward hooks 来应用 Expert LoRAs

        这允许我们在不修改 base model 代码的情况下注入 LoRA
        """
        target_modules = self.moe_config.get('target_modules', ['q_proj', 'v_proj'])

        for layer_idx, layer in enumerate(actor_module.model.layers):
            for module_name in target_modules:
                # 获取目标模块
                if module_name in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
                    module = getattr(layer.self_attn, module_name)
                elif module_name in ['gate_proj', 'up_proj', 'down_proj']:
                    module = getattr(layer.mlp, module_name)
                else:
                    continue

                # 注册 hook
                hook = self.expert_applier.create_forward_hook(layer_idx, module_name)
                module.register_forward_hook(hook)

    def compute_moe_forward(
        self,
        batch: Dict[str, torch.Tensor],
        actor_module: nn.Module,
    ) -> Dict[str, torch.Tensor]:
        """
        MoE 增强的 forward pass

        Returns:
            Dict containing:
                - logits
                - routing_weights
                - top_k_indices
                - lm_loss
        """
        # 1. 获取 base model 的 hidden states
        with torch.no_grad():
            base_outputs = actor_module(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                pixel_values=batch.get('pixel_values'),
                output_hidden_states=True,
                return_dict=True,
            )

        hidden_states = base_outputs.hidden_states[-1]

        # 2. 提取 instruction features
        instruction_mask = batch.get('instruction_mask')
        if instruction_mask is None:
            # Fallback: 使用整个序列
            instruction_features = hidden_states.mean(dim=1)
        else:
            instruction_features = self.feature_extractor(
                hidden_states, instruction_mask
            )

        # 3. 计算 routing
        router_output = self.router(instruction_features)

        # 4. 设置 routing 到 applier
        self.expert_applier.set_routing(
            router_output.top_k_indices,
            router_output.top_k_weights,
        )

        # 5. 再次 forward (这次会应用 expert LoRAs via hooks)
        outputs = actor_module(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            pixel_values=batch.get('pixel_values'),
            labels=batch.get('labels'),
            return_dict=True,
        )

        # 6. 清除 routing 缓存
        self.expert_applier.clear_routing()

        return {
            'logits': outputs.logits,
            'lm_loss': outputs.loss if hasattr(outputs, 'loss') else None,
            'routing_weights': router_output.routing_weights,
            'top_k_indices': router_output.top_k_indices,
            'router_logits': router_output.router_logits,
        }

    def compute_loss(
        self,
        batch: Dict[str, torch.Tensor],
        moe_outputs: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        计算总 loss (LM loss + MoE balance loss)
        """
        if not self.moe_enabled or moe_outputs.get('lm_loss') is None:
            return {'total_loss': moe_outputs.get('lm_loss', torch.tensor(0.0))}

        loss_dict = self.moe_loss_fn(
            lm_loss=moe_outputs['lm_loss'],
            routing_weights=moe_outputs['routing_weights'],
            router_logits=moe_outputs['router_logits'],
        )

        return loss_dict

    def get_trainable_params(self) -> list:
        """
        获取可训练参数列表

        MoE 模式下只训练:
        - Router
        - Expert LoRAs
        """
        if not self.moe_enabled:
            return super().get_trainable_params()

        params = []

        # Router 参数
        params.extend(list(self.router.parameters()))

        # Expert LoRA 参数
        params.extend(list(self.expert_collection.parameters()))

        return params

    def log_moe_metrics(
        self,
        routing_weights: torch.Tensor,
        instruction_types: Optional[list] = None,
        step: int = 0,
    ):
        """
        记录 MoE 相关的 metrics

        Args:
            routing_weights: [B, num_experts]
            instruction_types: ground truth types for analysis
            step: training step
        """
        with torch.no_grad():
            # Expert 利用率
            expert_usage = (routing_weights.argmax(dim=-1).bincount(
                minlength=self.moe_config['num_experts']
            ).float() / routing_weights.size(0)).cpu().numpy()

            metrics = {
                f'moe/expert_{i}_usage': usage
                for i, usage in enumerate(expert_usage)
            }

            # Routing 熵 (越高越均匀)
            routing_entropy = -(routing_weights * torch.log(routing_weights + 1e-10)).sum(dim=-1).mean()
            metrics['moe/routing_entropy'] = routing_entropy.item()

            # 如果有 instruction types，计算分化指标
            if instruction_types is not None:
                metrics.update(self._compute_specialization_metrics(
                    routing_weights, instruction_types
                ))

            # 记录到 wandb 或其他 logger
            if hasattr(self, 'logger'):
                self.logger.log(metrics, step=step)

            return metrics

    def _compute_specialization_metrics(
        self,
        routing_weights: torch.Tensor,
        instruction_types: list,
    ) -> Dict[str, float]:
        """
        计算 expert 分化指标
        """
        import numpy as np

        types = ['click', 'type', 'navigate', 'scroll']
        num_experts = routing_weights.size(1)

        # 构建 routing matrix
        routing_matrix = np.zeros((len(types), num_experts))

        routing_np = routing_weights.cpu().numpy()
        dominant_experts = routing_np.argmax(axis=1)

        for i, t in enumerate(types):
            mask = np.array([it == t for it in instruction_types])
            if mask.sum() > 0:
                for j in range(num_experts):
                    routing_matrix[i, j] = (dominant_experts[mask] == j).mean()

        # Specialization score
        row_max = routing_matrix.max(axis=1)
        specialization_score = row_max.mean()

        return {
            'moe/specialization_score': specialization_score,
        }

    def save_moe_checkpoint(self, save_dir: str):
        """
        保存 MoE 相关的 checkpoints

        分别保存:
        - router.pt
        - expert_0/, expert_1/, ... (PEFT 格式)
        """
        import os

        os.makedirs(save_dir, exist_ok=True)

        # 保存 Router
        torch.save(self.router.state_dict(), os.path.join(save_dir, 'router.pt'))

        # 保存 Experts (PEFT 格式，便于 vLLM 加载)
        experts_dir = os.path.join(save_dir, 'experts')
        self.expert_collection.save_experts_separately(experts_dir)

        print(f"[MoE] Saved checkpoint to {save_dir}")

    def load_moe_checkpoint(self, load_dir: str):
        """
        加载 MoE checkpoint
        """
        import os

        # 加载 Router
        router_path = os.path.join(load_dir, 'router.pt')
        if os.path.exists(router_path):
            self.router.load_state_dict(torch.load(router_path))

        # 加载 Experts
        experts_dir = os.path.join(load_dir, 'experts')
        if os.path.exists(experts_dir):
            for i in range(self.moe_config['num_experts']):
                expert_path = os.path.join(experts_dir, f'expert_{i}')
                if os.path.exists(expert_path):
                    from verl.models.moe.expert_lora import load_from_peft_checkpoint
                    load_from_peft_checkpoint(
                        self.expert_collection, i, expert_path
                    )

        print(f"[MoE] Loaded checkpoint from {load_dir}")
```

---

## 3. 配置文件模板

### 3.1 MoE GRPO 配置

```yaml
# examples/qwen_gui_moe/config/traj_grpo_moe.yaml

# 继承基础配置
defaults:
  - ../../qwen_gui_static_grpo/config/traj_grpo

# 模型配置
model:
  path: Qwen/Qwen2.5-VL-7B-Instruct

  # MoE 配置
  moe:
    enabled: true
    num_experts: 4
    top_k: 1
    expert_lora_r: 16
    expert_lora_alpha: 32
    target_modules:
      - q_proj
      - v_proj
    router_hidden: 256
    balance_weight: 0.1
    balance_type: mse  # mse, switch, or entropy

# 训练配置
trainer:
  # 使用 MoE trainer
  trainer_class: verl.trainer.ppo.moe_dapo_trainer.MoERayTrajDAPOTrainer

  # 学习率 (router + experts)
  learning_rate: 1e-4

  # 梯度累积
  gradient_accumulation_steps: 4

  # Epochs
  num_epochs: 10

  # 日志
  logging:
    log_moe_metrics: true
    log_routing_matrix_every: 100  # steps

# Rollout 配置 (vLLM)
rollout:
  engine: vllm
  tensor_model_parallel_size: 1

  # LoRA 配置
  enable_lora: true
  max_loras: 4  # 支持 4 个 experts

# 数据配置
data:
  train_path: datasets/ui_s1_train.jsonl
  include_instruction_type: true  # 用于分析
```

---

## 4. 训练启动脚本

### 4.1 SLURM 脚本

```bash
#!/bin/bash
# train/train_moe.slurm

#SBATCH --job-name=moe_train
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --time=48:00:00
#SBATCH --output=logs/moe_train_%j.out
#SBATCH --error=logs/moe_train_%j.err

# 环境设置
source /path/to/conda/etc/profile.d/conda.sh
conda activate verl

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTHONPATH=/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1:$PYTHONPATH

# Ray 配置
export RAY_DEDUP_LOGS=0

# 启动训练
python -m verl.trainer.main_dapo \
    --config examples/qwen_gui_moe/config/traj_grpo_moe.yaml \
    --trainer.output_dir checkpoints/moe_experiment \
    --trainer.wandb_project ui_s1_moe \
    --trainer.wandb_name moe_4experts_r16
```

### 4.2 本地测试脚本

```bash
#!/bin/bash
# scripts/test_moe_local.sh

# 小规模测试
python -m verl.trainer.main_dapo \
    --config examples/qwen_gui_moe/config/traj_grpo_moe.yaml \
    --trainer.num_epochs 1 \
    --trainer.max_steps 10 \
    --data.max_samples 100 \
    --trainer.output_dir checkpoints/moe_test
```

---

## 5. 与 FSDP 集成

### 5.1 修改 fsdp_workers.py

```python
# verl/workers/fsdp_workers.py 中添加

def create_moe_fsdp_model(
    model_config: DictConfig,
    moe_config: DictConfig,
) -> nn.Module:
    """
    创建支持 MoE 的 FSDP 模型

    关键点:
    1. Base model 冻结，不参与 FSDP sharding
    2. Router + Experts 参与 FSDP sharding
    """
    from verl.models.moe.moe_wrapper import MoEVLMWrapper

    # 创建 MoE 包装模型
    moe_model = MoEVLMWrapper(
        base_model_name_or_path=model_config.path,
        num_experts=moe_config.num_experts,
        top_k=moe_config.top_k,
        expert_lora_r=moe_config.expert_lora_r,
        expert_lora_alpha=moe_config.expert_lora_alpha,
        router_hidden=moe_config.router_hidden,
        balance_weight=moe_config.balance_weight,
    )

    # 只对可训练参数应用 FSDP
    # Base model 保持冻结
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp.wrap import lambda_auto_wrap_policy

    def should_wrap(module):
        """只 wrap MoE 组件"""
        return (
            isinstance(module, (TextOnlyRouter, ExpertLoRACollection))
            or 'lora' in module.__class__.__name__.lower()
        )

    fsdp_model = FSDP(
        moe_model,
        auto_wrap_policy=lambda_auto_wrap_policy(should_wrap),
        # ... 其他 FSDP 配置
    )

    return fsdp_model
```

---

## 6. 训练流程图

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         MoE Training Loop                                │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   1. Load Batch                                                          │
│   ┌──────────────────────────────────────────────────────────────┐      │
│   │ {input_ids, attention_mask, pixel_values, labels,            │      │
│   │  instruction_mask, instruction_type}                          │      │
│   └──────────────────────────────────────────────────────────────┘      │
│                              │                                           │
│                              ▼                                           │
│   2. Extract Instruction Features                                        │
│   ┌──────────────────────────────────────────────────────────────┐      │
│   │ hidden_states = base_model(input_ids, ...)                    │      │
│   │ instruction_features = feature_extractor(hidden_states, mask) │      │
│   └──────────────────────────────────────────────────────────────┘      │
│                              │                                           │
│                              ▼                                           │
│   3. Compute Routing                                                     │
│   ┌──────────────────────────────────────────────────────────────┐      │
│   │ router_output = router(instruction_features)                  │      │
│   │ → routing_weights: [B, num_experts]                           │      │
│   │ → top_k_indices: [B, top_k]                                   │      │
│   └──────────────────────────────────────────────────────────────┘      │
│                              │                                           │
│                              ▼                                           │
│   4. Apply Expert LoRAs                                                  │
│   ┌──────────────────────────────────────────────────────────────┐      │
│   │ expert_applier.set_routing(top_k_indices, top_k_weights)      │      │
│   │ outputs = model.forward(...)  # hooks apply LoRAs             │      │
│   └──────────────────────────────────────────────────────────────┘      │
│                              │                                           │
│                              ▼                                           │
│   5. Compute Loss                                                        │
│   ┌──────────────────────────────────────────────────────────────┐      │
│   │ lm_loss = cross_entropy(logits, labels)                       │      │
│   │ balance_loss = compute_load_balance_loss(routing_weights)     │      │
│   │ total_loss = lm_loss + balance_weight * balance_loss          │      │
│   └──────────────────────────────────────────────────────────────┘      │
│                              │                                           │
│                              ▼                                           │
│   6. Backward & Optimize                                                 │
│   ┌──────────────────────────────────────────────────────────────┐      │
│   │ total_loss.backward()                                         │      │
│   │ optimizer.step()  # Only updates Router + Expert LoRAs        │      │
│   └──────────────────────────────────────────────────────────────┘      │
│                              │                                           │
│                              ▼                                           │
│   7. Log Metrics                                                         │
│   ┌──────────────────────────────────────────────────────────────┐      │
│   │ log_moe_metrics(routing_weights, instruction_types)           │      │
│   │ → expert_usage, routing_entropy, specialization_score         │      │
│   └──────────────────────────────────────────────────────────────┘      │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 7. 下一步

完成训练集成后：
- [05_vllm_inference.md](./05_vllm_inference.md) - vLLM 推理集成
