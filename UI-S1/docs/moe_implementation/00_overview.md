# MoE Tool Agent 实施概述

## 1. 项目背景

### 1.1 研究问题

**核心假设**: MoE (Mixture of Experts) 结构能否让 expert LoRAs 自动分化，各自专注于不同类型的 GUI instructions？

| 场景 | 预期结果 | 后续行动 |
|------|---------|---------|
| 分化有效 | MoE 比单一 LoRA 更高效（相同参数量，更好性能） | 扩展到完整 Tool Agent |
| 分化无效 | 不需要 MoE | 简化架构，使用单一 LoRA |

### 1.2 GUI Instruction 类型

我们关注四种核心 instruction 类型：

```
┌─────────────────────────────────────────────────────────────┐
│                    GUI Instruction Types                     │
├──────────────┬──────────────────────────────────────────────┤
│    click     │ 点击、轻触、按下、选择特定 UI 元素            │
│    type      │ 输入文本、填写表单、键入内容                  │
│  navigate    │ 导航、跳转、打开页面、访问链接                │
│   scroll     │ 滚动、滑动、翻页浏览                         │
└──────────────┴──────────────────────────────────────────────┘
```

---

## 2. 技术栈选择

基于当前 UI-S1 repo 的技术栈：

| 组件 | 选择 | 版本/路径 |
|------|------|----------|
| Base VLM | Qwen2.5-VL-7B | `verl/models/transformers/qwen2_5_vl.py` |
| 训练框架 | verl (DAPO/GRPO) | `verl/trainer/ppo/dapo_ray_trainer.py` |
| 分布式训练 | FSDP + Ray | `verl/workers/fsdp_workers.py` |
| 推理引擎 | vLLM | `verl/workers/rollout/vllm_rollout/` |
| LoRA 框架 | PEFT + 自定义 | `verl/workers/sharding_manager/fsdp_vllm.py` |

---

## 3. 实施文档结构

```
docs/moe_implementation/
├── 00_overview.md              ← 当前文档
├── 01_architecture.md          # MoE 整体架构设计
├── 02_router_implementation.md # Router 模块实现
├── 03_expert_lora.md           # Expert LoRA 模块实现
├── 04_training_integration.md  # verl 训练框架集成
├── 05_vllm_inference.md        # vLLM 推理引擎集成
├── 06_data_pipeline.md         # 数据管道设计
└── 07_analysis_metrics.md      # 分析与评估指标
```

---

## 4. 实施阶段规划

### Phase 1: 基础设施 (01-03)
- 定义 MoE 架构
- 实现 Router 模块
- 实现 Expert LoRA 模块

### Phase 2: 训练集成 (04-06)
- 集成到 verl DAPO trainer
- 配置 vLLM 推理
- 构建数据管道

### Phase 3: 验证分析 (07)
- 实现分析工具
- 运行实验
- 评估分化效果

---

## 5. 关键文件映射

| 新增模块 | 位置 | 依赖 |
|---------|------|------|
| MoE Router | `verl/models/moe/router.py` | `torch.nn` |
| Expert LoRA | `verl/models/moe/expert_lora.py` | `peft`, `torch.nn` |
| MoE Wrapper | `verl/models/moe/moe_wrapper.py` | Qwen2.5-VL |
| MoE Loss | `verl/trainer/ppo/moe_loss.py` | `verl/trainer/ppo/core_algos.py` |
| MoE Trainer | `verl/trainer/ppo/moe_dapo_trainer.py` | `RayTrajDAPOTrainer` |
| MoE Config | `examples/qwen_gui_moe/config/traj_grpo_moe.yaml` | 现有 config |

---

## 6. 成功标准

| 指标 | 阈值 | 含义 |
|------|------|------|
| Specialization Score | > 0.6 | 明显分化 (random = 0.25) |
| Accuracy (MoE vs Single) | MoE > Single | MoE 更有效 |
| P-value (分化显著性) | < 0.05 | 分化是统计显著的 |
| Expert 利用率 | 每个 > 15% | 没有坍塌到单一 expert |

---

## 7. 下一步

请按顺序阅读后续文档：
1. [01_architecture.md](./01_architecture.md) - 理解整体架构
2. [02_router_implementation.md](./02_router_implementation.md) - 实现 Router
3. [03_expert_lora.md](./03_expert_lora.md) - 实现 Expert LoRAs

---

## 8. 进一步优化方案

在完成基础实施后，可以探索以下进阶方案：

### 8.1 基于KL散度的Expert多样性正则化

**问题**: 当前的balance loss只在routing层面约束，expert输出可能仍然相似。

**解决方案**: 在[07_analysis_metrics.md](./07_analysis_metrics.md#7-进一步方案基于kl散度的expert多样性正则化)中详细介绍了如何通过添加expert输出之间的KL散度来鼓励真正的行为diversity。

**核心思想**:
```python
Loss = LM_loss + balance_weight * routing_balance_loss +
       diversity_weight * expert_diversity_loss
```

其中 `expert_diversity_loss` = -KL(p_i || p_j)

### 8.2 其他备选方案

| 方案 | 适用场景 | 文档位置 |
|------|---------|----------|
| 输出分布KL散度 | 需要expert行为真正diverse | [07_analysis_metrics.md §7.3](./07_analysis_metrics.md#73-实现方案) |
| LoRA参数KL散度 | 计算效率优先 | [07_analysis_metrics.md §7.3](./07_analysis_metrics.md#73-实现方案) |
| 层级化KL散度 | 综合routing + output + parameter | [07_analysis_metrics.md §7.3](./07_analysis_metrics.md#73-实现方案) |
