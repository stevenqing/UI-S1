# vLLM 推理引擎集成

## 1. vLLM LoRA 支持现状

### 1.1 当前 repo 中的 vLLM LoRA 实现

```
verl/workers/
├── sharding_manager/
│   └── fsdp_vllm.py          # 现有 LoRA 同步逻辑
└── rollout/
    └── vllm_rollout/
        └── vllm_rollout.py   # 推理实现
```

关键代码位置：`verl/workers/sharding_manager/fsdp_vllm.py:103-146`

```python
# 现有实现：收集 LoRA 参数并同步到 vLLM
def __collect_lora_params() -> OrderedDict:
    from peft.utils.save_and_load import get_peft_model_state_dict
    lora_params = OrderedDict()
    peft_model = getattr(self.module, "_fsdp_wrapped_module", self.module)
    # ...
    return lora_params
```

### 1.2 vLLM Multi-LoRA 能力

| 功能 | 版本要求 | 说明 |
|------|---------|------|
| 单 LoRA 推理 | 0.5.4+ | ✅ 已支持 |
| Multi-LoRA 批处理 | 0.7.0+ | ✅ 已支持 |
| 动态 LoRA 切换 | 0.7.3+ | ✅ 已支持 |
| VLM LoRA | 0.8.0+ | ⚠️ 仅语言层 |

---

## 2. MoE 推理策略

### 2.1 策略选择

| 策略 | 描述 | 适用场景 |
|------|------|---------|
| **Strategy A: 预路由** | 根据 instruction 预先选择 expert，只加载一个 LoRA | 推理速度优先 |
| **Strategy B: 多 LoRA** | 加载所有 experts，推理时动态选择 | 灵活性优先 |
| **Strategy C: 权重合并** | 训练后合并 experts 权重 | 简单部署 |

**推荐: Strategy A (预路由)**
- 原因：MoE 设计本身就是 top-k=1，每个样本只用一个 expert
- vLLM 只需加载对应的 LoRA adapter

### 2.2 Strategy A 实现

```
┌─────────────────────────────────────────────────────────────────────┐
│                    MoE Inference Pipeline                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│   Input: instruction text                                            │
│              │                                                       │
│              ▼                                                       │
│   ┌──────────────────┐                                              │
│   │  Lightweight      │  ← 只用 Router，不需要 GPU                   │
│   │  Router Inference │     (可以用 CPU 或小 GPU)                    │
│   └────────┬─────────┘                                              │
│            │                                                         │
│            ▼                                                         │
│   expert_idx = router(instruction) → 0, 1, 2, or 3                  │
│            │                                                         │
│            ▼                                                         │
│   ┌──────────────────┐                                              │
│   │  vLLM Engine     │                                              │
│   │  with LoRA       │  ← 加载 expert_{idx} 的 LoRA                  │
│   └────────┬─────────┘                                              │
│            │                                                         │
│            ▼                                                         │
│      Action Output                                                   │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 3. 完整实现代码

### 3.1 MoE 推理 Wrapper

```python
# verl/workers/rollout/moe_vllm_rollout.py

import torch
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple
from pathlib import Path

from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

from verl.models.moe.router import TextOnlyRouter
from transformers import AutoTokenizer


class MoEVLLMInference:
    """
    MoE + vLLM 推理引擎

    特点:
    - 使用轻量级 Router 进行预路由
    - 每个请求只加载一个 Expert LoRA
    - 支持批量推理时的混合 experts
    """

    def __init__(
        self,
        base_model_path: str,
        router_checkpoint: str,
        expert_lora_dir: str,
        num_experts: int = 4,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
    ):
        """
        Args:
            base_model_path: Path to base VLM
            router_checkpoint: Path to trained router weights
            expert_lora_dir: Directory containing expert_0/, expert_1/, ...
            num_experts: Number of experts
            tensor_parallel_size: vLLM TP size
            gpu_memory_utilization: GPU memory fraction
        """
        self.num_experts = num_experts
        self.expert_lora_dir = Path(expert_lora_dir)

        # 1. 加载 tokenizer (用于 router 输入处理)
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_path)

        # 2. 加载 Router (轻量级，可以在 CPU 上运行)
        self.router = self._load_router(router_checkpoint, base_model_path)
        self.router.eval()

        # 3. 初始化 vLLM 引擎
        self.llm = LLM(
            model=base_model_path,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            enable_lora=True,
            max_loras=num_experts,
            max_lora_rank=64,  # 根据你的 expert LoRA rank 调整
            trust_remote_code=True,
        )

        # 4. 预加载所有 expert LoRAs
        self.lora_requests = self._prepare_lora_requests()

    def _load_router(self, checkpoint_path: str, model_path: str) -> TextOnlyRouter:
        """加载训练好的 Router"""
        from transformers import AutoConfig

        # 获取 hidden_size
        config = AutoConfig.from_pretrained(model_path)
        hidden_size = config.hidden_size

        # 创建 router
        router = TextOnlyRouter(
            hidden_size=hidden_size,
            num_experts=self.num_experts,
        )

        # 加载权重
        state_dict = torch.load(checkpoint_path, map_location='cpu')
        router.load_state_dict(state_dict)

        return router

    def _prepare_lora_requests(self) -> Dict[int, LoRARequest]:
        """准备所有 expert 的 LoRA requests"""
        lora_requests = {}

        for i in range(self.num_experts):
            expert_path = self.expert_lora_dir / f"expert_{i}"
            if expert_path.exists():
                lora_requests[i] = LoRARequest(
                    lora_name=f"expert_{i}",
                    lora_int_id=i + 1,  # vLLM 要求 ID > 0
                    lora_path=str(expert_path),
                )
            else:
                print(f"Warning: Expert {i} not found at {expert_path}")

        return lora_requests

    def route_instruction(self, instruction: str) -> int:
        """
        使用 Router 确定应该使用哪个 expert

        Args:
            instruction: The instruction text

        Returns:
            expert_idx: Index of the selected expert
        """
        # 1. Tokenize instruction
        inputs = self.tokenizer(
            instruction,
            return_tensors='pt',
            truncation=True,
            max_length=512,
        )

        # 2. 获取简单的 embedding (使用 tokenizer 的 embedding)
        # 注意：这里我们需要一个轻量级的方式获取 instruction features
        # 选项 A: 使用预训练的 sentence encoder
        # 选项 B: 使用简单的 embedding lookup + pooling

        # 简化实现：使用 router 期望的 hidden_size 的随机投影
        # 在实际部署中，应该使用更好的 instruction encoder
        with torch.no_grad():
            # 临时方案：使用 instruction 的 hash 作为 seed
            # 实际应该用轻量级 encoder
            instruction_features = self._get_instruction_features(instruction)
            router_output = self.router(instruction_features)
            expert_idx = router_output.top_k_indices[0, 0].item()

        return expert_idx

    def _get_instruction_features(self, instruction: str) -> torch.Tensor:
        """
        获取 instruction 的特征表示

        注意：这是一个简化实现
        生产环境应该使用:
        1. 轻量级 encoder (如 MiniLM)
        2. 或者缓存的 instruction embeddings
        """
        # 方案 1: 使用 sentence-transformers (推荐)
        try:
            from sentence_transformers import SentenceTransformer
            if not hasattr(self, '_sentence_encoder'):
                self._sentence_encoder = SentenceTransformer('all-MiniLM-L6-v2')
                # 添加投影层匹配 router hidden_size
                self._projection = torch.nn.Linear(384, self.router.hidden_size)

            with torch.no_grad():
                embedding = self._sentence_encoder.encode(
                    instruction, convert_to_tensor=True
                ).unsqueeze(0)
                features = self._projection(embedding)

            return features

        except ImportError:
            # 方案 2: Fallback - 简单的 random projection (仅用于测试)
            print("Warning: sentence-transformers not installed, using fallback")
            torch.manual_seed(hash(instruction) % 2**32)
            return torch.randn(1, self.router.hidden_size)

    def generate(
        self,
        prompts: List[str],
        instructions: List[str],
        sampling_params: Optional[SamplingParams] = None,
    ) -> List[str]:
        """
        批量生成

        Args:
            prompts: 完整的 prompt 列表 (包含 system, image, instruction)
            instructions: instruction 文本列表 (用于 routing)
            sampling_params: vLLM sampling parameters

        Returns:
            生成的文本列表
        """
        if sampling_params is None:
            sampling_params = SamplingParams(
                temperature=0.7,
                top_p=0.9,
                max_tokens=256,
            )

        # 1. 为每个 prompt 确定 expert
        expert_indices = [self.route_instruction(inst) for inst in instructions]

        # 2. 分组：相同 expert 的 prompts 一起处理
        expert_groups: Dict[int, List[Tuple[int, str]]] = {}
        for i, (prompt, expert_idx) in enumerate(zip(prompts, expert_indices)):
            if expert_idx not in expert_groups:
                expert_groups[expert_idx] = []
            expert_groups[expert_idx].append((i, prompt))

        # 3. 按 expert 分批生成
        results = [None] * len(prompts)

        for expert_idx, group in expert_groups.items():
            indices, group_prompts = zip(*group)

            # 使用对应的 LoRA
            lora_request = self.lora_requests.get(expert_idx)

            outputs = self.llm.generate(
                list(group_prompts),
                sampling_params,
                lora_request=lora_request,
            )

            for idx, output in zip(indices, outputs):
                results[idx] = output.outputs[0].text

        return results

    def generate_single(
        self,
        prompt: str,
        instruction: str,
        sampling_params: Optional[SamplingParams] = None,
    ) -> str:
        """单条生成"""
        results = self.generate([prompt], [instruction], sampling_params)
        return results[0]


class MoEVLLMRollout:
    """
    集成到 verl rollout 的 MoE 推理

    继承现有的 VLLMRollout 并添加 MoE routing
    """

    def __init__(
        self,
        config,
        moe_config,
        tokenizer,
        model_hf_config,
        **kwargs,
    ):
        from verl.workers.rollout.vllm_rollout.vllm_rollout import VLLMRollout

        # 初始化基础 VLLMRollout
        self.base_rollout = VLLMRollout(
            actor_module=config.model_path,
            config=config,
            tokenizer=tokenizer,
            model_hf_config=model_hf_config,
            **kwargs,
        )

        # 初始化 MoE 组件
        self.moe_inference = MoEVLLMInference(
            base_model_path=config.model_path,
            router_checkpoint=moe_config.router_checkpoint,
            expert_lora_dir=moe_config.expert_lora_dir,
            num_experts=moe_config.num_experts,
        )

    def generate_sequences(self, prompts, **kwargs):
        """
        MoE 增强的序列生成

        重写基类方法以支持 MoE routing
        """
        # 提取 instructions
        instructions = self._extract_instructions(prompts)

        # 使用 MoE 推理
        # ... 实现细节 ...

        return self.base_rollout.generate_sequences(prompts, **kwargs)

    def _extract_instructions(self, prompts):
        """从 prompts 中提取 instruction 文本"""
        # 根据你的 prompt 格式实现
        # 例如，如果 prompts 是 DataProto:
        if hasattr(prompts, 'non_tensor_batch'):
            return prompts.non_tensor_batch.get('instruction', [])
        return []
```

---

## 4. vLLM Sharding Manager 扩展

### 4.1 支持多 Expert 的权重同步

```python
# verl/workers/sharding_manager/fsdp_vllm_moe.py

from collections import OrderedDict
from typing import Dict
import torch

from verl.workers.sharding_manager.fsdp_vllm import FSDPVLLMShardingManager


class MoEFSDPVLLMShardingManager(FSDPVLLMShardingManager):
    """
    支持 MoE 的 FSDP-vLLM Sharding Manager

    扩展:
    1. 分别同步每个 expert 的 LoRA 参数
    2. 同步 Router 参数
    """

    def __init__(
        self,
        module,
        inference_engine,
        model_config,
        moe_config,
        **kwargs,
    ):
        super().__init__(module, inference_engine, model_config, **kwargs)
        self.moe_config = moe_config
        self.num_experts = moe_config.get('num_experts', 4)

    def __enter__(self):
        """进入推理模式，同步所有 expert 权重"""
        # 1. 收集所有 expert 的 LoRA 参数
        expert_params = self._collect_moe_lora_params()

        # 2. 同步每个 expert 到 vLLM
        for expert_idx, params in expert_params.items():
            self._sync_expert_to_vllm(expert_idx, params)

        # 3. 调用父类方法处理其他同步
        # (如果需要同步 base model 权重)

        return self

    def _collect_moe_lora_params(self) -> Dict[int, OrderedDict]:
        """
        收集所有 expert 的 LoRA 参数

        Returns:
            Dict[expert_idx, OrderedDict[param_name, tensor]]
        """
        expert_params = {}

        # 获取 MoE 模块
        moe_module = self._get_moe_module()
        if moe_module is None:
            return expert_params

        # 收集每个 expert 的参数
        for i in range(self.num_experts):
            expert_state = moe_module.expert_collection.get_expert_state_dict(i)

            # 转换为 vLLM 期望的格式
            converted = self._convert_to_vllm_format(expert_state, i)
            expert_params[i] = converted

        return expert_params

    def _get_moe_module(self):
        """获取 MoE 模块"""
        module = getattr(self.module, "_fsdp_wrapped_module", self.module)

        if hasattr(module, 'expert_collection'):
            return module

        # 遍历查找
        for name, child in module.named_modules():
            if hasattr(child, 'expert_collection'):
                return child

        return None

    def _convert_to_vllm_format(
        self,
        expert_state: OrderedDict,
        expert_idx: int,
    ) -> OrderedDict:
        """
        转换 expert 参数为 vLLM LoRA 格式

        vLLM 期望的格式:
        - base_model.model.layers.{i}.self_attn.{module}.lora_A.weight
        - base_model.model.layers.{i}.self_attn.{module}.lora_B.weight
        """
        converted = OrderedDict()

        for key, value in expert_state.items():
            # 解析 key: layer_{i}_{module}.lora_layers.layer_{i}_{module}.lora_{A/B}
            # 转换为 vLLM 格式
            parts = key.split('.')

            # 提取 layer_idx 和 module_name
            layer_module = parts[0]  # e.g., "layer_0_q_proj"
            layer_parts = layer_module.split('_')
            layer_idx = int(layer_parts[1])
            module_name = '_'.join(layer_parts[2:])

            # 确定是 lora_A 还是 lora_B
            lora_type = parts[-1]  # lora_A 或 lora_B

            # 构建 vLLM key
            if module_name in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
                vllm_key = f"base_model.model.layers.{layer_idx}.self_attn.{module_name}.{lora_type}.weight"
            else:
                vllm_key = f"base_model.model.layers.{layer_idx}.mlp.{module_name}.{lora_type}.weight"

            converted[vllm_key] = value

        return converted

    def _sync_expert_to_vllm(self, expert_idx: int, params: OrderedDict):
        """
        同步单个 expert 的参数到 vLLM

        使用 vLLM 的 LoRA 热更新 API
        """
        # 构建 LoRA adapter 路径 (临时目录)
        import tempfile
        import os

        with tempfile.TemporaryDirectory() as tmpdir:
            expert_dir = os.path.join(tmpdir, f"expert_{expert_idx}")
            os.makedirs(expert_dir)

            # 保存参数
            torch.save(params, os.path.join(expert_dir, "adapter_model.bin"))

            # 保存 config
            self._save_lora_config(expert_dir)

            # 使用 vLLM API 加载
            # 注意：具体 API 取决于 vLLM 版本
            if hasattr(self.inference_engine, 'load_lora'):
                self.inference_engine.load_lora(
                    lora_name=f"expert_{expert_idx}",
                    lora_path=expert_dir,
                )

    def _save_lora_config(self, save_dir: str):
        """保存 LoRA 配置"""
        import json

        config = {
            "peft_type": "LORA",
            "task_type": "CAUSAL_LM",
            "r": self.moe_config.get('expert_lora_r', 16),
            "lora_alpha": self.moe_config.get('expert_lora_alpha', 32),
            "target_modules": self.moe_config.get('target_modules', ['q_proj', 'v_proj']),
            "bias": "none",
        }

        with open(os.path.join(save_dir, "adapter_config.json"), "w") as f:
            json.dump(config, f)
```

---

## 5. 推理配置示例

### 5.1 vLLM 服务器配置

```yaml
# config/moe_inference.yaml

inference:
  engine: vllm
  base_model: Qwen/Qwen2.5-VL-7B-Instruct

  # MoE 配置
  moe:
    enabled: true
    router_checkpoint: checkpoints/moe_experiment/router.pt
    expert_lora_dir: checkpoints/moe_experiment/experts/
    num_experts: 4

  # vLLM 配置
  vllm:
    tensor_parallel_size: 1
    gpu_memory_utilization: 0.9
    enable_lora: true
    max_loras: 4
    max_lora_rank: 64

  # 采样配置
  sampling:
    temperature: 0.7
    top_p: 0.9
    max_tokens: 256
```

### 5.2 推理脚本

```python
# scripts/moe_inference.py

import argparse
from omegaconf import OmegaConf

from verl.workers.rollout.moe_vllm_rollout import MoEVLLMInference


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--instruction", type=str, required=True)
    args = parser.parse_args()

    # 加载配置
    config = OmegaConf.load(args.config)

    # 初始化 MoE 推理引擎
    engine = MoEVLLMInference(
        base_model_path=config.inference.base_model,
        router_checkpoint=config.inference.moe.router_checkpoint,
        expert_lora_dir=config.inference.moe.expert_lora_dir,
        num_experts=config.inference.moe.num_experts,
    )

    # 执行推理
    instruction = args.instruction
    expert_idx = engine.route_instruction(instruction)
    print(f"Routed to Expert {expert_idx}")

    prompt = f"User: {instruction}\nAssistant:"
    result = engine.generate_single(prompt, instruction)
    print(f"Result: {result}")


if __name__ == "__main__":
    main()
```

---

## 6. 下一步

完成推理集成后：
- [06_data_pipeline.md](./06_data_pipeline.md) - 数据管道设计
