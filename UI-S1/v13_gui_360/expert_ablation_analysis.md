# V13 Expert Ablation Analysis

## 实验设置

- **Checkpoint**: `v13_gui360_rl_resumed/epoch-1/cooperative` (epoch-5-resumed)
- **测试集**: gui360_test_968.jsonl (968 episodes)
- **模型架构**: Iterative Cooperative LoRA V13, lora_r=128, num_comm_rounds=2
- **4个配置**:
  - Expert 1 + Communication (job 4428278)
  - Expert 2 + Communication (job 4428278)
  - Expert 1 - No Communication (job 4428279)
  - Expert 2 - No Communication (job 4428279)

## 总体结果

| 配置 | TSR | Avg Progress | Step SR |
|------|-----|-------------|---------|
| **Full model (baseline)** | **18.70%** | 31.87% | 63.31% |
| Expert 2 + Comm | **18.70%** | 33.01% | 63.99% |
| Expert 1 + Comm | 17.46% | 31.79% | 62.79% |
| Expert 2 - No Comm | 11.88% | 22.58% | 51.59% |
| Expert 1 - No Comm | 11.98% | 22.18% | 50.89% |

### 通信机制的影响

- 有通信 vs 无通信：TSR 提升 ~50% (11.9% → 18.7%)
- Step SR 提升 ~22% (51.6% → 64.0%)
- **通信是性能的核心来源**

## 核心发现：两个 Expert 没有功能分化

### 1. Action Type 预测 — 完全相同

两个 Expert 都 99.5% 预测 click，无论有无通信：

| 配置 | click | type | swipe | key |
|------|-------|------|-------|-----|
| E1+Comm | 2139 (99.6%) | 4 (0.2%) | 1 (0.0%) | 3 (0.1%) |
| E2+Comm | 2174 (99.5%) | 6 (0.3%) | 4 (0.2%) | 1 (0.0%) |

GT 分布: click 82-84%, type 14-16%, swipe 2%

Type reward 几乎完全一致: E1=0.838 vs E2=0.839 (2028 matched steps 中仅 5 步不同)

### 2. 差异完全在坐标精度 (content_reward)

| GT Type | E1 Content | E2 Content | 差异 |
|---------|-----------|-----------|------|
| click (n=1697) | **0.5626** | 0.5498 | E1 +0.013 |
| type (n=294) | 0.4179 | 0.4161 | E1 +0.002 |
| swipe (n=37) | 0.0270 | **0.0541** | E2 +0.027 |

- E1 坐标更好: **831** steps
- E2 坐标更好: 470 steps
- 相同: 727 steps

E1 在更多步上坐标更准，但 E2 在关键步上表现更好。

### 3. 成败原因分析：98% 由坐标精度决定

分析两个 Expert 在同一步中一个成功另一个失败的情况：

| 情况 | 总步数 | 因坐标更好 | 因类型更好 | 两者都更好 |
|------|--------|-----------|-----------|-----------|
| E1 成功, E2 失败 | 65 | 64 (98%) | 0 (0%) | 0 (0%) |
| E2 成功, E1 失败 | 97 | 95 (98%) | 0 (0%) | 2 (2%) |

**两个 Expert 的差别完全不在"做什么"(what)，而在"点哪里"(where)**。

### 4. Episode 级别对比

| 情况 | Episode 数 |
|------|-----------|
| 两者都成功 | 148 |
| 仅 E1 成功 | 21 |
| 仅 E2 成功 | 33 |
| 两者都失败 | 766 |

E2 赢更多 (33 vs 21)。E2 平均 progress 高 1.22%。

### 5. 输出长度差异

| 指标 | E1 | E2 |
|------|----|----|
| 总输出长度 (mean) | 151.2 chars | 155.5 chars |
| 推理前缀长度 (mean) | 85.1 chars | 89.1 chars |
| 输出更长的步数 | 647 | 929 |

E2 生成更长的推理文本，可能因此产生了稍好的坐标定位。

## 结合权重分析的解释

1. **A1⊥A2 (cosine≈0) 但 routing≈0.5**: 两个 A 矩阵正交但等权混合，两个 expert 的输出被平均，无法形成专业化分工

2. **通信是分化的唯一来源**: 无通信时两个 expert 表现几乎完全相同 (E1: 11.98% vs E2: 11.88%)，通信使它们获得对方的信息后整体精度提升

3. **Expert 2 ≈ Full model**: Expert 2 + Comm 的 TSR (18.70%) 等于 Full model，说明 Expert 2 在通信后已包含了 Expert 1 的信息

## 对后续方法 (V14/V15) 的启示

1. **plan vs exec 分解的局限性**: V15 Factored PG 试图通过 gate variance φ 区分 planning tokens 和 execution tokens。但底层两个 expert 并没有 what/where 的分工，因此 φ 的分解可能效果有限。

2. **通信 > 专业化**: 当前架构的性能提升主要来自通信机制（L27, L10, L18 的 gate 和 projection），而非 expert 的功能分化。

3. **可能的改进方向**:
   - 引入 auxiliary loss 强制 A1/A2 专业化（如 A1 负责 type prediction, A2 负责 coordinate prediction）
   - 改变 routing 机制使其不再平均混合（如 argmax routing 或更大的 routing noise）
   - 通信机制的优化可能比 advantage decomposition 更有效

## 文件路径

- Expert 1 + Comm 结果: `outputs/epoch-5-resumed_comm_expert_1_only/`
- Expert 2 + Comm 结果: `outputs/epoch-5-resumed_comm_expert_2_only/`
- Expert 1 - No Comm 结果: `outputs/epoch-5-resumed_nocomm_expert_1_only/`
- Expert 2 - No Comm 结果: `outputs/epoch-5-resumed_nocomm_expert_2_only/`
- 评估脚本: `scripts/eval_expert_with_comm.slurm`, `scripts/eval_expert_no_comm.slurm`
