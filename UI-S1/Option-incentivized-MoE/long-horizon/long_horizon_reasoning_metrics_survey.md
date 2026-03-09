# GUI Agent 长程推理能力评估指标调研报告

> **背景**：本报告针对 **GUI-360** 项目（1.2M+ 操作步骤、覆盖 Windows 办公应用 Word/Excel/PowerPoint 的大规模 CUA 基准），调研用于评估 GUI/移动端/Web Agent 长程推理能力的指标与基准。

---

## 目录

1. [GUI-360 现有评估体系及不足](#1-gui-360-现有评估体系及不足)
2. [长程推理相关基准综述](#2-长程推理相关基准综述)
3. [长程任务评估指标分类](#3-长程任务评估指标分类)
4. [前沿论文与新评估方法（2024–2026）](#4-前沿论文与新评估方法20242026)
5. [核心挑战](#5-核心挑战)
6. [实践指南](#6-实践指南)
7. [对 GUI-360 的推荐方案](#7-对-gui-360-的推荐方案)
8. [参考文献](#8-参考文献)

---

## 1. GUI-360 现有评估体系及不足

GUI-360 当前在 **三个规范任务** 上评估 Agent：

| 任务 | 指标 | 最优 SFT 结果 |
|------|------|---------------|
| **GUI Grounding（元素定位）** | 成功率（预测点落入真值 bbox） | 82.30% |
| **Screen Parsing（界面解析）** | Recall、Precision、F1、文本相似度、IoU | F1: 40.8% |
| **Action Prediction（动作预测）** | 成功率（function + args + status 三项匹配） | 50.08%（视觉）/ 25.78%（a11y） |

### 核心缺口

Action Prediction 采用 **逐步评估**（next-action accuracy），缺乏对以下能力的度量：

- Agent 能否 **端到端完成完整多步任务**
- 任务完成的 **部分进展**（partial progress）
- **规划质量** vs. 逐步反应式行为
- **轨迹效率**（Agent 步数 vs. 人类参考步数）
- 长动作序列中的 **错误恢复** 能力

---

## 2. 长程推理相关基准综述

### 2.1 Web 类基准

| 基准 | 任务数 | 平均步数 | 核心指标 | 年份 |
|------|--------|----------|----------|------|
| **MiniWoB++** | 100+ 简化 Web 任务 | 3–10 | 任务成功率、平均奖励 | 2018 |
| **Mind2Web** | 2,350 任务 / 137 网站 | 5–15 | 元素准确率、步骤成功率、操作 F1 | 2023 |
| **WebArena** | 812 任务 / 自托管环境 | 10–30 | 功能正确性（终态检验） | 2023 |
| **VisualWebArena** | 910 视觉驱动任务 | 10–30 | 视觉功能正确性 | 2024 |

**WebArena** 是长程 Web 评估的标杆：通过检验 **最终环境状态** 而非匹配动作序列来评分。人类准确率约 89%，最优 Agent 约 57%（Plan-and-Act, 2025）。

### 2.2 移动端/Android 基准

| 基准 | 任务数 | 平均步数 | 核心创新 | 年份 |
|------|--------|----------|----------|------|
| **AITW** | 30K+ 真实 Android 交互 | 5–20 | 大规模离线轨迹 | 2023 |
| **AndroidWorld** | 116 任务 / 20 个 App | ~14 | 动态任务实例化、Pass@k | 2024 |
| **AndroidLab** | 138 任务 / 9 个 App | 不定 | Sub-SR、RRR、ROR | 2024 |
| **AndroidLens** | 571 任务 / 38 领域 | **26+** | 平均任务进展（ATP） | 2025 |
| **MobileWorld** | 201 任务 / 20 个 App | **~28** | 62% 跨应用、Agent-用户交互 | 2025 |

**AndroidLens** 尤为相关：聚焦 **长延迟任务**（平均 26+ 步），引入 **Average Task Progress (ATP)** 指标 —— Agent 在 TSR 仅 12.7% 时，ATP 达到 50.47%，揭示了二元指标掩盖的大量部分进展。

### 2.3 桌面/OS 级基准

| 基准 | 任务数 | 领域 | 核心创新 | 年份 |
|------|--------|------|----------|------|
| **OSWorld** | 369 任务 | Linux 桌面 + Web | 每任务定制评估脚本、15–50 步 | 2024 |
| **macOSWorld** | 202 任务 | macOS / 30 个应用 | 多语言、安全性基准 | 2025 |
| **OdysseyBench** | 602 任务 | 办公应用（Word、Excel、PDF） | 长程办公工作流 | 2025 |

**OdysseyBench**（微软研究院，2025）与 GUI-360 最直接相关 —— 同样针对 **Word/Excel 等办公应用的长程工作流**，要求 Agent 追踪长交互历史并协调多步推理。

### 2.4 多领域综合基准

| 基准 | 任务数 | 核心创新 | 年份 |
|------|--------|----------|------|
| **AgentBoard** | 1,013 任务 / 9 领域 | **Progress Rate** 指标 | 2024 |
| **TheAgentCompany** | 175 职场任务 | **部分完成评分** + 检查点 | 2024 |

---

## 3. 长程任务评估指标分类

### 3.1 步骤级指标（Step-Level）

| 指标 | 描述 | 使用场景 |
|------|------|----------|
| **动作类型准确率** | 是否预测了正确的动作类型（click、type、scroll 等） | Mind2Web、AITW、GUI-360 |
| **元素/目标准确率** | 是否选中了正确的 UI 元素或坐标 | Mind2Web、GUI-360 |
| **步骤成功率** | 动作类型和目标同时正确 | Mind2Web、AndroidLab |
| **合理操作比率 (ROR)** | 产生有意义状态变化的步骤占比 | AndroidLab |

**局限性**：步骤级指标无法捕获 Agent 是否朝任务目标取得了有意义的进展。

### 3.2 轨迹级指标（Trajectory-Level）

| 指标 | 描述 | 使用场景 |
|------|------|----------|
| **任务成功率 (TSR)** | 二元：Agent 是否完成了任务？ | 所有基准 |
| **Pass@k** | k 次独立运行中是否有成功 | AndroidWorld |
| **反向冗余比 (RRR)** | 效率：人类步数 / Agent 步数 | AndroidLab |
| **平均交互长度** | 每个任务的平均步数 | AndroidWorld、OSWorld |
| **归一化编辑距离** | 预测轨迹与参考轨迹的结构距离 | 规划评估 |

### 3.3 部分完成 / 进展度量指标（最具价值）

这是 **指标研发最活跃的方向**，对 GUI-360 最有意义：

#### AgentBoard Progress Rate（NeurIPS 2024 Oral）

```
Progress Rate = max_t [ (1/K) * Σ_{k=1}^{K} check_subgoal_k(state_t) ]
```

- 每个任务有 K 个人工标注的子目标
- 每个子目标达成贡献 1/K
- 追踪整个轨迹上的最大进展
- 即使任务整体失败也能捕获有效推进

#### TheAgentCompany 部分完成评分

```
S_partial = 0.5 × (获得分数 / 总分) + 0.5 × S_full
```

- 每任务设置带权重的检查点
- 强激励完全完成（全部通过额外获得 50% 加分）
- 混合使用确定性评估器（Python 函数、状态检查）和 LLM 评估器

#### AndroidLens 平均任务进展 (ATP)

- 基于 **里程碑** 的中间目标评估
- 使用稳定、可验证的里程碑度量 26+ 步任务的进展
- 揭示了 TSR（12.7%）与 ATP（50.47%）之间的巨大差距
- 证明 Agent 即使在"失败"任务中也取得了实质性进展

#### AndroidLab 子目标成功率 (Sub-SR)

- 将任务分解为子目标
- 无论整体是否成功，度量完成的子目标比例
- 结合 RRR（效率）和 ROR（有效性）实现多维评估

### 3.4 规划质量指标

| 指标 | 描述 | 来源 |
|------|------|------|
| **Plan Match Accuracy** | 与参考规划的结构相似度 | Plan-and-Act |
| **Node F1** | 工具/动作选择的精确率和召回率 | 规划类基准 |
| **Edge F1** | 动作排序和依赖关系的准确度 | 规划类基准 |
| **平均步数（效率）** | 相同结果下步数越少 = 规划越好 | OSWorld、AndroidWorld |

### 3.5 错误恢复指标

| 指标 | 描述 | 来源 |
|------|------|------|
| **自我纠错率** | 错误动作后成功恢复的频率 | 反思类 Agent |
| **回溯频率** | Agent 需要撤销动作的频率 | 轨迹分析 |
| **重试成功率** | 第 2/3 次尝试的成功率 | AndroidWorld Pass@3 |
| **重复性评分** | Agent 是否陷入循环 | AgentRewardBench |
| **副作用评分** | 任务完成是否造成非预期状态变化 | AgentRewardBench |

---

## 4. 前沿论文与新评估方法（2024–2026）

### 4.1 过程奖励模型（Process Reward Models）

| 论文 | 核心思路 | 效果 | 链接 |
|------|----------|------|------|
| **GUI-PRA** (2025.09) | Dynamic Memory + 自适应 UI 感知 PRM | AndroidWorld +14.53% SR | [arXiv:2509.23263](https://arxiv.org/abs/2509.23263) |
| **ProgRM** (2025.05) | 基于 LCS 的自标注进展标签 | 提供密集中间奖励 | [arXiv:2505.18121](https://arxiv.org/abs/2505.18121) |
| **ADMIRE** (2026.02) | 自适应里程碑 + 非对称信用分配 | GRPO/RLOO/DAPO +10% SR | [arXiv:2602.11524](https://arxiv.org/abs/2602.11524) |
| **AgentPRM** (2025.11) | 基于 TD 的 GAE 估计步骤奖励 | 捕获长期 promise | [arXiv:2511.08325](https://arxiv.org/abs/2511.08325) |
| **ToolPRMBench** (2026.01) | 首个工具使用 Agent 的步骤级 PRM 基准 | 离线 + 在线采样 | [arXiv:2601.12294](https://arxiv.org/abs/2601.12294) |

**核心洞察**：PRM 为 RL 训练提供 **密集训练信号**，解决长轨迹中的信用分配根本问题。GUI-360 的 105K+ 训练步骤可用于训练 PRM，为 MoE-SFT 管线提供逐步奖励信号。

### 4.2 轨迹级评估

| 论文 | 核心思路 | 链接 |
|------|----------|------|
| **AgentRewardBench** (2025.04) | LLM 评判 Web Agent 轨迹的基准；规则评估系统性低估成功率 | [arXiv:2504.08942](https://arxiv.org/abs/2504.08942) |
| **WABER** (2025) | 基于网络代理的可靠性 + 效率评估 | [Microsoft Research](https://www.microsoft.com/en-us/research/wp-content/uploads/2025/04/134_WABER_Web_Agent_Benchmarki.pdf) |

### 4.3 规划专项评估

| 论文 | 核心思路 | 链接 |
|------|----------|------|
| **Plan-and-Act** (ICML 2025) | 分离 Planner 与 Executor，独立评估 | [arXiv:2503.09572](https://arxiv.org/abs/2503.09572) |
| **Agent S2** (2025.04) | 组合式框架 + 混合定位 + 主动层级规划 | [arXiv:2504.00906](https://arxiv.org/abs/2504.00906) |

### 4.4 综述论文

| 论文 | 核心贡献 | 链接 |
|------|----------|------|
| **A Hitchhiker's Guide to Agent Evaluation** (ICLR 2026) | 提出 Completion under Policy (CuP) 指标 | [ICLR Blog](https://iclr-blogposts.github.io/2026/blog/2026/agent-evaluation/) |
| **GUI Agents: A Survey** (ACL 2025) | LLM GUI Agent 全面综述 | [ACL Findings](https://aclanthology.org/2025.findings-acl.1158.pdf) |
| **Evaluation and Benchmarking of LLM Agents** (2025) | 评估目标与指标的分类学 | [arXiv:2507.21504](https://arxiv.org/abs/2507.21504) |

---

## 5. 核心挑战

### 5.1 长动作序列中的信用分配

- **稀疏/延迟奖励**：真实任务在 10–50+ 步后才给出终端奖励
- **"中间遗忘"现象**：PRM 在长历史上下文中难以评估当前步骤（GUI-PRA 的 Dynamic Memory 解决此问题）
- **奖励保真度 vs. 密度的权衡**：结果奖励准确但稀疏；过程奖励密集但易产生偏差
- **新兴方案**：ProgRM（LCS 关键步骤标注）、ADMIRE（成功/失败轨迹非对称信用）、AgentPRM（TD + GAE 估计）

### 5.2 部分完成度量

- **二元指标严重不足**：AndroidLens 显示 TSR 12.7% 但 ATP 50.47% —— 二元指标丢失了大量信息
- **子目标标注成本高**：AgentBoard 需要逐任务人工标注
- **多条有效路径**：真实任务通常有多种正确解法，基于单一参考轨迹的度量会错误惩罚创造性但正确的方案
- **规则评估低估成功**：AgentRewardBench 发现规则评估器系统性地低估成功轨迹

### 5.3 规划能力 vs. 反应式行为的区分

- 大多数基准 **混淆了规划和执行**：单一成功率无法区分是规划不佳还是执行不佳
- **Plan-and-Act 架构** 通过分离 Planner 和 Executor 实现独立评估
- **"Why Reasoning Fails to Plan"**（2026.01）证明 Agent 在长程任务中主要问题是规划而非执行
- 规划专用指标（Node F1、Edge F1）已存在但在 GUI 基准中使用不足

### 5.4 多步任务的组合性

- **跨应用工作流**：MobileWorld（62.2% 跨应用）、OSWorld 测试组合性，但多数基准仍聚焦单应用
- **嵌套依赖**：OdysseyBench 揭示 Agent 在"长程上下文依赖和多交互协调"上的失败
- **难度缩放**：随着任务组合的子任务增多，成功率不成比例下降 —— 原子任务与组合任务间的性能差距是一个待探索的关键指标

---

## 6. 实践指南

### 6.1 实现参考

**进展度量（AgentBoard 风格）**：
```python
def compute_progress_rate(trajectory, subgoals):
    K = len(subgoals)
    max_progress = 0
    for state in trajectory:
        achieved = sum(1 for sg in subgoals if sg.check(state))
        max_progress = max(max_progress, achieved / K)
    return max_progress
```

**部分完成评分（TheAgentCompany 风格）**：
```python
def compute_partial_score(trajectory, checkpoints):
    earned = sum(cp.evaluate(trajectory) for cp in checkpoints)
    total = sum(cp.max_points for cp in checkpoints)
    all_passed = (earned == total)
    return 0.5 * (earned / total) + 0.5 * float(all_passed)
```

**过程奖励模型（ProgRM 风格）**：
```python
# 基于 LCS 的自标注
def annotate_key_steps(successful_trajectories):
    # 跨成功轨迹寻找最长公共子序列
    key_steps = lcs_across_trajectories(successful_trajectories)
    # 根据 LCS 位置分配进展标签
    for traj in all_trajectories:
        for step in traj:
            step.progress_label = compute_progress(step, key_steps)
    return annotated_trajectories
```

### 6.2 数据需求

| 指标类别 | 所需数据 | GUI-360 是否具备？ |
|----------|---------|-------------------|
| 步骤级准确率 | 逐步 (action, element) 标注 + 截图 | 是（105K+ 步） |
| 任务成功率 | 完整轨迹结果 | 是（成功/失败标签） |
| 进展度量 | 子目标定义 + 检查函数 | **需要标注** |
| 部分完成评分 | 检查点定义 + 评分规则 | **需要设计** |
| 轨迹效率 | 人类参考轨迹 | 是（专家轨迹） |
| 过程奖励 | 成功/失败轨迹 + 步骤标签 | 部分（有成功标签；步骤标签需标注） |
| 规划质量 | 参考规划或动作依赖图 | **需要标注** |

---

## 7. 对 GUI-360 的推荐方案

### 第一优先级：低成本快速实现

| 指标 | 衡量内容 | 实现难度 |
|------|----------|----------|
| **轨迹级任务成功率** | 端到端完整任务完成情况 | 低 |
| **轨迹效率 (RRR)** | Agent 步数 vs. 参考步数 | 低 |
| **Pass@k** | 多次运行的可靠性 | 低 |

- **任务成功率**：可基于现有数据计算，检查最终步骤 `status: "finish"` 且终态匹配预期
- **轨迹效率**：`RRR = 参考步数 / Agent 步数`，值 > 1 表示 Agent 比参考更高效
- **Pass@k**：每任务独立运行 k 次，报告任意一次成功的比例

### 第二优先级：中等成本、高价值

| 指标 | 衡量内容 | 实现难度 |
|------|----------|----------|
| **子目标进展率** | 部分进展度量 | 中 |
| **里程碑 ATP** | 中间进展追踪 | 中 |

- **子目标进展率**：将 GUI-360 任务分解为有意义的子目标
  - 例如 "在 Excel 中创建数据透视表"（12+ 步）：子目标可包括 (a) 选中数据范围、(b) 插入透视表、(c) 配置行/列、(d) 应用格式
  - 标注 100–200 个测试任务的子目标
  - 计算 `Progress Rate = 达成子目标数 / 总子目标数` 的最大值

### 第三优先级：高成本、战略价值

| 指标 | 衡量内容 | 实现难度 |
|------|----------|----------|
| **过程奖励模型 (PRM)** | 步骤级信用分配 | 高 |
| **规划-执行分离评估** | 独立评估规划质量 | 高 |
| **功能性终态评估** | 真正的任务理解度 | 高 |

- **PRM**：利用 GUI-360 的 105K+ 训练步骤，采用 ProgRM 的 LCS 自标注方法自动发现关键步骤，为 MoE-SFT/RL 管线提供密集奖励
- **规划-执行分离**：实现 Plan-and-Act 架构，分别评估规划是否合理、执行是否准确
- **终态功能评估**（OSWorld/OdysseyBench 风格）：编写确定性评估脚本检查 **最终应用状态**，而非匹配动作序列
  - Excel：检查正确公式是否在正确单元格
  - Word：检查格式是否匹配预期输出
  - PowerPoint：检查幻灯片布局是否匹配预期

### 总结：推荐指标套件

| 指标 | 衡量维度 | 优先级 | 成本 |
|------|----------|--------|------|
| **任务成功率 (TSR)** | 端到端完成 | 高 | 低 |
| **轨迹效率 (RRR)** | 步数效率 | 高 | 低 |
| **Pass@k** | 运行可靠性 | 高 | 低 |
| **子目标进展率** | 部分进展 | 高 | 中 |
| **里程碑 ATP** | 中间进展追踪 | 中 | 中 |
| **过程奖励模型** | 步骤级信用分配 | 高 | 高 |
| **规划-执行分离** | 规划质量 | 中 | 高 |
| **功能性终态评估** | 真实任务理解 | 高 | 高 |

---

## 8. 参考文献

### 基准
- [WebArena](https://webarena.dev/) — Zhou et al., 2023
- [VisualWebArena](https://jykoh.com/vwa) — Koh et al., ACL 2024
- [Mind2Web](https://osu-nlp-group.github.io/Mind2Web/) — Deng et al., NeurIPS 2023
- [AndroidWorld](https://arxiv.org/abs/2405.14573) — Rawles et al., ICLR 2025
- [AndroidLab](https://arxiv.org/abs/2410.24024) — Xu et al., ACL 2025
- [AndroidLens](https://arxiv.org/abs/2512.21302) — 2025.12
- [MobileWorld](https://arxiv.org/abs/2512.19432) — Tongyi-MAI, 2025.12
- [OSWorld](https://os-world.github.io/) — Xie et al., NeurIPS 2024
- [macOSWorld](https://arxiv.org/abs/2506.04135) — Yang et al., NeurIPS 2025
- [OdysseyBench](https://arxiv.org/abs/2508.09124) — Microsoft Research, 2025.08
- [GUI-360](https://arxiv.org/abs/2511.04307) — 2024.11
- [AgentBoard](https://hkust-nlp.github.io/agentboard/) — Ma et al., NeurIPS 2024 Oral
- [TheAgentCompany](https://arxiv.org/abs/2412.14161) — CMU, 2024.12

### 过程奖励模型
- [GUI-PRA](https://arxiv.org/abs/2509.23263) — 2025.09
- [ProgRM](https://arxiv.org/abs/2505.18121) — 2025.05
- [ADMIRE](https://arxiv.org/abs/2602.11524) — 2026.02
- [AgentPRM](https://arxiv.org/abs/2511.08325) — 2025.11
- [ToolPRMBench](https://arxiv.org/abs/2601.12294) — 2026.01

### 轨迹评估
- [AgentRewardBench](https://arxiv.org/abs/2504.08942) — 2025.04
- [WABER](https://www.microsoft.com/en-us/research/wp-content/uploads/2025/04/134_WABER_Web_Agent_Benchmarki.pdf) — 2025

### 规划评估
- [Plan-and-Act](https://arxiv.org/abs/2503.09572) — ICML 2025
- [Agent S2](https://arxiv.org/abs/2504.00906) — 2025.04

### 综述
- [A Hitchhiker's Guide to Agent Evaluation](https://iclr-blogposts.github.io/2026/blog/2026/agent-evaluation/) — ICLR 2026
- [GUI Agents: A Survey](https://aclanthology.org/2025.findings-acl.1158.pdf) — ACL 2025
- [Evaluation and Benchmarking of LLM Agents](https://arxiv.org/abs/2507.21504) — 2025

---

## 9. GUI-360 长程推理评估实现

> 本节记录已实现的长程推理评估模块、文件路径及用法。所有代码位于 `train_GUI_360/GUI-360-eval/` 下。

### 9.1 文件清单

| 文件 | 绝对路径 | 类型 | 说明 |
|------|----------|------|------|
| `evaluator/long_horizon.py` | `.../GUI-360-eval/evaluator/long_horizon.py` | **新建** | 轨迹级后处理分析器（方案 1），计算 TSR、散布 Progress、**顺序 Progress**、分桶、错误级联等指标；支持 `enable_subgoals` 子目标分析 |
| `evaluator/subgoal_progress.py` | `.../GUI-360-eval/evaluator/subgoal_progress.py` | **新建** | 子目标进展框架（P1）：`SubGoalExtractor`（基于 thought 关键词/基于 LCS）+ `SubGoalProgressEvaluator`（二元 + 加权进展） |
| `evaluator/action_prediction_autoregressive.py` | `.../GUI-360-eval/evaluator/action_prediction_autoregressive.py` | **新建** | 自回归评估器（方案 2），支持 **顺序/散布进展度量** 和 **Pass@k 多样本评估** |
| `scripts/analyze_trajectory_metrics.py` | `.../GUI-360-eval/scripts/analyze_trajectory_metrics.py` | **新建** | CLI 脚本，支持单模型分析、多模型对比、TF vs AR 对比，输出 JSON + Markdown；支持 `--enable_subgoals` |
| `config.py` | `.../GUI-360-eval/config.py` | **修改** | `ar_stop_on_error` 默认值改为 `True`；新增 `num_samples`（Pass@k 的 K）、`sampling_temperature` 字段 |
| `evaluation.py` | `.../GUI-360-eval/evaluation.py` | **修改** | 新增 `--ar_no_stop_on_error`、`--num_samples`、`--temperature` 参数；摘要输出支持 AR 和 Pass@k 指标 |

### 9.2 方案 1：轨迹级后处理分析（无需模型）

#### 核心模块 `evaluator/long_horizon.py`

**类：`TrajectoryAnalyzer`**

- **输入**：现有的 `evaluation_results_*.json` 文件（Teacher-Forcing 或 AR 评估结果均可）
- **输出**：`LongHorizonAnalysis` 数据结构，包含所有轨迹级指标

**核心方法**：

| 方法 | 说明 |
|------|------|
| `extract_trajectory_id(sample_id)` | `sample_id.rsplit('_', 1)` 提取轨迹 ID 和步骤号 |
| `group_by_trajectory()` | 按轨迹 ID 分组，步骤按 step_num 排序 |
| `compute_trajectory_metrics(traj_id, steps)` | 计算单条轨迹的 TSR、Progress、首错位置等 |
| `analyze()` | 完整分析：TSR、Progress、分桶、步骤衰减、错误级联、领域分解 |
| `save_analysis(analysis, output_path)` | 保存为 JSON |
| `format_markdown_report(analysis)` | 生成 Markdown 报告 |
| `compare_models(analyses)` | 多模型对比表格 |
| `compare_tf_ar(tf_analysis, ar_analysis)` | TF vs AR gap 分析 |

**计算的指标**：

| 指标 | 描述 |
|------|------|
| Trajectory Success Rate (TSR) | 所有步骤全部正确的轨迹占比 |
| Average Progress Rate | 每条轨迹步骤正确率的均值 |
| Step-Level Success Rate | 所有步骤的整体正确率（与现有评估一致） |
| 分桶指标 | short(1-5) / medium(6-15) / long(16+) 的 TSR 和 Progress |
| 步骤位置准确率 | 第 1/2/3/.../N 步的准确率（衰减曲线） |
| 错误级联分析 | 首次错误位置、错误后剩余步骤准确率、恢复率 |
| 领域分解 | word/excel/ppt 的轨迹级 TSR 和 Progress |
| 轨迹级组件匹配率 | 轨迹维度的 function/args/status 全对率 |

#### CLI 脚本 `scripts/analyze_trajectory_metrics.py`

**用法 1：分析单个模型**

```bash
cd /scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/train_GUI_360/GUI-360-eval

python scripts/analyze_trajectory_metrics.py \
    --results_file results/qwen25vl_sft_v2/action_prediction_20260305_065639/evaluation_results_20260305_081045.json \
    --output_dir results/trajectory_analysis/qwen25vl_sft_v2 \
    --format both
```

输出：
- `trajectory_analysis_evaluation_results_*.json`（完整指标 JSON）
- `trajectory_analysis_evaluation_results_*.md`（Markdown 报告）
- 终端打印摘要

**用法 2：多模型对比**

```bash
python scripts/analyze_trajectory_metrics.py \
    --results_file \
    results/qwen25vl_sft_v2/action_prediction_*/evaluation_results_*.json \
    results/qwen25vl_lora_v3/action_prediction_*/evaluation_results_*.json \
    --output_dir results/trajectory_analysis/comparison \
    --format both
```

输出：
- `model_comparison.json`
- `model_comparison.md`（对比表格）
- 各模型独立分析存于子目录

**用法 3：TF vs AR 对比**

```bash
python scripts/analyze_trajectory_metrics.py \
    --tf_results results/qwen25vl_sft_v2/action_prediction_*/evaluation_results_*.json \
    --ar_results results/qwen25vl_sft_v2/action_prediction_ar_*/ar_evaluation_results_*.json \
    --output_dir results/trajectory_analysis/tf_ar_gap
```

输出：
- `tf_ar_comparison.json`（含 TSR gap、Progress gap、Error Propagation Rate）
- `tf_ar_comparison.md`（含分桶 gap、领域 gap 分析）

### 9.3 方案 2：Autoregressive Rollout 评估（需要模型）

#### 核心模块 `evaluator/action_prediction_autoregressive.py`

**类：`AutoregressiveActionPredictionEvaluator(ActionPredictionEvaluator)`**

继承自现有 TF 评估器，复用 `compare_actions()`、`_compare_regular_args()` 等方法。

**与 Teacher-Forcing 的关键区别**：

| 维度 | Teacher-Forcing (现有) | Autoregressive (新增) |
|------|----------------------|---------------------|
| 历史来源 | GT thought | 模型自身 predicted thoughts |
| 步骤间依赖 | 无（各步独立） | 有（前步输出 → 后步输入） |
| 并行度 | 所有步骤并行 | 轨迹间并行，轨迹内串行 |
| 截图来源 | GT 截图 | GT 截图（semi-online） |
| 数据加载 | `load_data()` yield 单步 | `load_trajectories()` yield 整条轨迹 |
| 评估粒度 | 步骤级 SR | 轨迹级 TSR + Progress + 步骤级 |
| `--max_samples` | 限制步骤数 | 限制轨迹数 |

**核心方法**：

| 方法 | 说明 |
|------|------|
| `load_trajectories(task_type)` | 按轨迹加载数据（一个 JSONL 文件 = 一条轨迹） |
| `evaluate_trajectory(trajectory)` | 自回归评估单条轨迹（核心方法） |
| `evaluate(thread_num, resume_from)` | 主入口：轨迹间 ThreadPoolExecutor 并行 |
| `_calculate_ar_statistics()` | 计算 AR 特有统计量 |
| `_save_ar_results(stats)` | 保存结果（含 `detailed_results` 兼容方案 1 分析器） |

**历史构建方式**：

```python
# Teacher-Forcing (现有):
previous_actions.append(f"Step {j+1}: {gt_thought}")

# Autoregressive (新增):
predicted_history.append(f"Step {i+1}: {model_thoughts}")
# 若 thoughts 解析失败则回退:
predicted_history.append(f"Step {i+1}: Performed {pred_function} action")
```

**stop_on_error 模式**：

- `stop_on_error=True`（**新默认值**）：首次错误后停止轨迹，更贴近真实 Agent 部署场景
- `stop_on_error=False`（通过 `--ar_no_stop_on_error` 切换）：即使步骤失败也继续，可完整对比 TF vs AR

#### 运行 AR 评估

**基本用法**：

```bash
cd /scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/train_GUI_360/GUI-360-eval

# 小样本测试（5 条轨迹）
python evaluation.py \
    --root_dir /path/to/gui360/dataset \
    --type action_prediction_ar \
    --model_type qwen2.5_vl_7b \
    --model_name /path/to/model \
    --api_url http://host:port/v1 \
    --max_samples 5 \
    --threads 5 \
    --output_dir results/ar_test
```

**完整评估（推荐在 Slurm 上运行）**：

```bash
python evaluation.py \
    --root_dir /path/to/gui360/dataset \
    --type action_prediction_ar \
    --model_type qwen2.5_vl_7b \
    --model_name /path/to/model \
    --api_url http://host:port/v1 \
    --threads 8 \
    --output_dir results/qwen25vl_sft_v2/action_prediction_ar
```

**禁用 stop_on_error（对比 TF 时使用）**：

```bash
python evaluation.py \
    --root_dir /path/to/gui360/dataset \
    --type action_prediction_ar \
    --ar_no_stop_on_error \
    --model_type qwen2.5_vl_7b \
    --model_name /path/to/model \
    --api_url http://host:port/v1 \
    --threads 8 \
    --output_dir results/qwen25vl_sft_v2/action_prediction_ar_full
```

#### AR 评估输出格式

输出文件：`ar_evaluation_results_{timestamp}.json`

```json
{
  "config": {"mode": "autoregressive", "stop_on_error": false, ...},
  "statistics": {
    "total_trajectories": 3233,
    "trajectory_success_rate": 0.098,
    "avg_progress_rate": 0.345,
    "step_success_rate": 0.312,
    "bucket_metrics": {"short": {...}, "medium": {...}, "long": {...}},
    "domain_stats": {"word": {...}, "excel": {...}, "ppt": {...}}
  },
  "trajectory_results": [...],
  "detailed_results": [...]  // 兼容方案 1 分析器
}
```

### 9.4 完整工作流示例

```bash
cd /scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/train_GUI_360/GUI-360-eval

# Step 1: 对已有 TF 结果进行轨迹级分析（无需模型，立即可用）
python scripts/analyze_trajectory_metrics.py \
    --results_file results/qwen25vl_sft_v2/action_prediction_*/evaluation_results_*.json \
    --output_dir results/trajectory_analysis/qwen25vl_sft_v2

# Step 2: 运行 AR 评估（需要模型服务）
python evaluation.py \
    --root_dir /path/to/gui360/dataset \
    --type action_prediction_ar \
    --model_type qwen2.5_vl_7b \
    --model_name /path/to/model \
    --api_url http://host:port/v1 \
    --threads 8 \
    --output_dir results/qwen25vl_sft_v2/action_prediction_ar

# Step 3: 对 AR 结果也做轨迹级分析
python scripts/analyze_trajectory_metrics.py \
    --results_file results/qwen25vl_sft_v2/action_prediction_ar/ar_evaluation_results_*.json \
    --output_dir results/trajectory_analysis/qwen25vl_sft_v2_ar

# Step 4: TF vs AR gap 对比
python scripts/analyze_trajectory_metrics.py \
    --tf_results results/qwen25vl_sft_v2/action_prediction_*/evaluation_results_*.json \
    --ar_results results/qwen25vl_sft_v2/action_prediction_ar/ar_evaluation_results_*.json \
    --output_dir results/trajectory_analysis/tf_ar_gap
```

### 9.5 评估 Pipeline 修复（2026-03-06）

在对 Qwen3-VL-8B 的异常低性能（Step SR 2.7%）进行根因分析后，发现评估 pipeline 中存在 **坐标系统不匹配** 问题：

#### 发现的 Bug

| Bug | 描述 | 影响模型 | 修复方式 |
|-----|------|---------|---------|
| **坐标系统** | Qwen3-VL 输出 0-1000 归一化坐标，pipeline 按绝对像素坐标处理 | Qwen3-VL | `_transform_coordinates_to_original()` 增加 `normalized_1000` 分支 |
| **smart_resize factor** | 硬编码 factor=28（Qwen2.5-VL），Qwen3-VL 需要 factor=32 | Qwen3-VL | 增加 `resize_factor` 参数 |
| **ModelFactory** | `_load_qwen_model()` 无法传递坐标系统参数 | Qwen3-VL | 自动检测 model_name 中的 `"qwen3"` |

#### 修改的文件

- `models/qwen2.5_vl_7b.py`：`__init__` 增加 `coordinate_system` 和 `resize_factor` 参数；`_transform_coordinates_to_original()` 增加 `normalized_1000` 模式
- `evaluation.py`：`_load_qwen_model()` 自动检测 Qwen3 并设置 `coordinate_system="normalized_1000"`, `resize_factor=32`

#### 修复效果

| 指标 | Qwen3-VL (旧 pipeline) | Qwen3-VL (修复后) | 提升 |
|------|----------------------|------------------|------|
| **Step SR** | 2.7% | **10.1%** | **+7.4% (3.7×)** |
| PPT | 3.6% | 11.6% | +8.0% |
| Excel | 1.0% | 8.6% | +7.6% |
| Word | 3.3% | 10.0% | +6.7% |

> 修复后 Qwen3-VL (10.1%) 与 Qwen2.5-VL base (18.1%) 的差距从 15.4% 缩小到 8.0%，剩余差距为真实能力差异。

---

### 9.6 全模型轨迹级评估结果（Teacher-Forcing）

> 以下结果基于方案 1 对所有已有 TF 评估结果的轨迹级分析。
> 共 3233 条轨迹，19046 步，平均 5.9 步/轨迹。
> Qwen3-VL 列使用修复后的 pipeline 结果。

#### 结果文件路径

| 模型 | 检查点路径 | 评估结果文件 |
|------|-----------|-------------|
| Qwen2.5-VL-7B base | `.../checkpoints/Qwen2.5-VL-7B-Instruct` | `results/qwen25vl_base/action_prediction_20260228_025613/evaluation_results_20260228_042655.json` |
| Qwen3-VL-8B base (fixed) | `.../checkpoints/Qwen3-VL-8B-Instruct` | `results/qwen3vl_base_fixed/action_prediction_*/evaluation_results_*.json` |
| Qwen2.5-VL SFT v1 | `.../llamafactory/output/gui360_full_sft/checkpoint-398` | `results/qwen25vl_sft/action_prediction_20260301_085102/evaluation_results_20260301_100546.json` |
| Qwen2.5-VL SFT v2 | `.../llamafactory/output/gui360_full_sft_v2` | `results/qwen25vl_sft_v2/action_prediction_20260305_065639/evaluation_results_20260305_081045.json` |
| Qwen2.5-VL LoRA v3 | `.../llamafactory/output/gui360_lora_sft_v3_merged` | `results/qwen25vl_lora_v3/action_prediction_20260306_023010/evaluation_results_20260306_034344.json` |

#### 9.6.1 整体指标对比

| 指标 | Qwen2.5-VL base | Qwen3-VL base (fixed) | SFT v1 | **SFT v2** | LoRA v3 |
|------|-----------------|----------------------|--------|------------|---------|
| Step SR | 0.1805 | 0.1008 | 0.1114 | **0.4690** | 0.2467 |
| **TSR** | 0.0285 | 0.0043 | 0.0309 | **0.1695** | 0.0504 |
| Avg Progress | 0.2032 | 0.0988 | 0.1326 | **0.5060** | 0.2754 |
| Traj Func Match | 0.3727 | 0.4095 | 0.5741 | **0.5753** | 0.4816 |
| Traj Args Match | 0.0288 | 0.0043 | 0.0353 | **0.1791** | 0.0541 |
| Traj Status Match | **0.8494** | 0.7776 | 0.7770 | 0.7742 | 0.7643 |

#### 9.6.2 分桶对比（按轨迹长度）

**TSR by bucket:**

| 桶 | 范围 | #轨迹 | Qwen2.5-VL base | Qwen3-VL (fixed) | SFT v1 | **SFT v2** | LoRA v3 |
|----|------|--------|-----------------|------------------|--------|------------|---------|
| short | 1-5 | 2076 | 0.0434 | 0.0067 | 0.0482 | **0.2471** | 0.0776 |
| medium | 6-15 | 930 | 0.0022 | 0.0000 | 0.0000 | **0.0376** | 0.0022 |
| long | 16+ | 227 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

**Progress by bucket:**

| 桶 | Qwen2.5-VL base | Qwen3-VL (fixed) | SFT v1 | **SFT v2** | LoRA v3 |
|----|-----------------|------------------|--------|------------|---------|
| short | 0.2178 | 0.0940 | 0.1496 | **0.5313** | 0.2959 |
| medium | 0.1922 | 0.1152 | 0.1025 | **0.4810** | 0.2522 |
| long | 0.1140 | 0.0748 | 0.1007 | **0.3777** | 0.1828 |

#### 9.6.3 领域对比

**TSR by domain:**

| Domain | Qwen2.5-VL base | Qwen3-VL (fixed) | SFT v1 | **SFT v2** | LoRA v3 |
|--------|-----------------|------------------|--------|------------|---------|
| excel | 0.0150 | 0.0020 | 0.0470 | **0.1612** | 0.0330 |
| ppt | 0.0474 | 0.0092 | 0.0081 | **0.2081** | 0.0613 |
| word | 0.0263 | 0.0029 | 0.0336 | **0.1512** | 0.0562 |

**Progress by domain:**

| Domain | Qwen2.5-VL base | Qwen3-VL (fixed) | SFT v1 | **SFT v2** | LoRA v3 |
|--------|-----------------|------------------|--------|------------|---------|
| excel | 0.1452 | 0.0854 | 0.1055 | **0.4732** | 0.2116 |
| ppt | 0.3005 | 0.1195 | 0.1371 | **0.5861** | 0.3641 |
| word | 0.1841 | 0.0954 | 0.1496 | **0.4793** | 0.2659 |

#### 9.6.4 错误级联对比

| 指标 | Qwen2.5-VL base | Qwen3-VL (fixed) | SFT v1 | **SFT v2** | LoRA v3 |
|------|-----------------|------------------|--------|------------|---------|
| 首错位置 (normalized) | 0.3806 | 0.3445 | 0.3741 | **0.4369** | 0.3766 |
| 首错步骤 (absolute) | 1.29 | 1.08 | 1.28 | **1.81** | 1.31 |
| 错误后准确率 | 0.1580 | 0.1061 | 0.0751 | **0.4154** | 0.2322 |
| 恢复率 | 0.4285 | 0.3162 | 0.1730 | **0.6596** | 0.5476 |

#### 9.6.5 步骤位置准确率（衰减曲线）

| Step | Qwen2.5-VL base | Qwen3-VL (fixed) | SFT v1 | **SFT v2** | LoRA v3 |
|------|-----------------|------------------|--------|------------|---------|
| 1 | 0.2125 | 0.0786 | 0.2672 | **0.4881** | 0.2413 |
| 2 | 0.2514 | 0.0769 | 0.0969 | **0.5990** | 0.3378 |
| 3 | 0.2278 | 0.1453 | 0.0580 | **0.5412** | 0.3102 |
| 4 | 0.1744 | 0.1396 | 0.0679 | **0.4671** | 0.2341 |
| 5 | 0.1712 | 0.1207 | 0.0653 | **0.4449** | 0.2211 |
| 6 | 0.1625 | 0.1029 | 0.0631 | **0.4235** | 0.2308 |
| 7 | 0.1408 | 0.1037 | 0.0830 | **0.4083** | 0.2074 |
| 8 | 0.1526 | 0.1039 | 0.0816 | **0.4184** | 0.2303 |
| 9 | 0.1008 | 0.0709 | 0.0850 | **0.3764** | 0.1795 |
| 10 | 0.1303 | 0.0917 | 0.0789 | **0.3798** | 0.2073 |

#### 9.6.6 关键发现

1. **SFT v2 全面领先**：在所有指标上均为最优，Step SR (46.9%) → TSR (16.95%)，差距显示轨迹级评估的必要性。
2. **所有模型长轨迹 TSR 为 0**：16+ 步的轨迹无一成功，长程推理是所有模型的共同瓶颈。
3. **Pipeline 修复大幅提升 Qwen3-VL 表现**：修复坐标系统后，Qwen3-VL Step SR 从 2.7% 提升至 10.1%（3.7 倍），但仍低于 Qwen2.5-VL base (18.1%)，剩余差距为真实的 GUI grounding 能力差异。
4. **Qwen3-VL 修复后恢复率显著提升**：从 11.0% → 31.6%，错误后准确率从 2.7% → 10.6%，说明原先大量因坐标错误导致的假阴性被消除。
5. **SFT v1 存在异常**：Step 1 准确率 (26.7%) 高于后续步骤 (6-9%)，存在严重的衰减，说明 v1 的训练可能存在 history-dependent 退化问题。
6. **LoRA v3 vs SFT v2 差距显著**：LoRA v3 Step SR (24.7%) 约为 SFT v2 的一半，TSR (5.0%) 约为三分之一，LoRA 微调在轨迹级任务上效率低于全量 SFT。
7. **SFT v2 错误恢复能力最强**：恢复率 66.0%，错误后准确率 41.5%，首错位置最晚 (归一化 0.44)，说明即使犯错也能继续产出有用的动作。
8. **Progress Rate 比 TSR 更能区分模型能力**：base 模型 TSR 接近 0 但 Progress 仍有 20%+，SFT v2 的 Progress (50.6%) 比 TSR (17.0%) 高出近 3 倍，部分进展信息对模型比较更有信息量。

---

### 9.7 评估改进（2026-03-08）：顺序进展、Pass@k、子目标进展

> 本节记录三项评估改进的实现，对应调研报告中"第一优先级"和"第二优先级"推荐方案的落地。

#### 9.7.1 P0-A：修正 AR 评估默认行为 + 顺序进展度量

##### 背景与动机

原有 GUI-360 的 TSR 是"伪指标"：每一步使用 GT 历史（teacher-forcing）独立评估，然后事后聚合。AR 评估器虽然存在，但默认 `stop_on_error=False`，且使用 **散布进展**（scattered progress = 正确步数 / 总步数），而非 **顺序进展**（sequential progress = 首次错误前的步数 / 总步数）。

顺序进展更符合真实部署场景：一旦出错，后续步骤的价值大幅降低（尤其在有状态的 GUI 操作中）。

##### 变更

| 文件 | 变更 |
|------|------|
| `config.py` | `ar_stop_on_error` 默认值 `False` → `True` |
| `evaluation.py` | 新增 `--ar_no_stop_on_error`（覆盖为 False）；原 `--ar_stop_on_error` 保留但为 no-op |
| `action_prediction_autoregressive.py` | `TrajectoryResult` 新增 `scattered_progress_rate` 字段；`progress_rate` 语义改为顺序进展 |
| `long_horizon.py` | `TrajectoryMetrics`、`BucketMetrics`、`LongHorizonAnalysis` 均新增 `sequential_progress_rate`；Markdown 报告更新 |

##### 两种进展度量的对比

| 度量 | 公式 | 含义 | 适用场景 |
|------|------|------|----------|
| **顺序进展**（`progress_rate`） | `(first_error_step - 1) / total_steps` | Agent 能走多远才犯第一个错 | AR 评估、真实部署、RL 训练奖励 |
| **散布进展**（`scattered_progress_rate`） | `num_correct / total_steps` | 散布在轨迹中的正确步数占比 | TF 评估、模型能力上界估计 |

##### 统计输出示例

```json
{
  "trajectory_success_rate": 0.098,
  "avg_progress_rate": 0.234,
  "avg_scattered_progress_rate": 0.345,
  "bucket_metrics": {
    "short": {"trajectory_success_rate": 0.15, "avg_progress_rate": 0.35, "avg_scattered_progress_rate": 0.42},
    "medium": {"trajectory_success_rate": 0.02, "avg_progress_rate": 0.18, "avg_scattered_progress_rate": 0.30},
    "long": {"trajectory_success_rate": 0.00, "avg_progress_rate": 0.08, "avg_scattered_progress_rate": 0.22}
  }
}
```

##### 验证命令

```bash
cd /scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/train_GUI_360/GUI-360-eval

# 用新的顺序进展度量重新分析已有 TF 结果
python scripts/analyze_trajectory_metrics.py \
    --results_file results/qwen25vl_sft_v2/action_prediction_*/evaluation_results_*.json \
    --output_dir results/trajectory_analysis/test_sequential \
    --format both
# 验证：输出应同时包含 avg_progress_rate 和 avg_sequential_progress_rate
```

---

#### 9.7.2 P0-B：Pass@k 多样本评估支持

##### 背景与动机

AndroidWorld 已实现 Pass@k 评估，但 GUI-360 此前不支持。Pass@k 度量多次独立运行中任意一次成功的概率，反映模型的可靠性和多样性。温度采样 + 多次运行可以揭示模型是否"几乎能做到"（近成功）但被单次采样的方差所掩盖。

参考实现：`evaluation/eval_qwenvl_pass_k.py`（Android Control 评估）。

##### 新增数据结构

```python
@dataclass
class PassKTrajectoryResult:
    trajectory_id: str
    num_steps: int
    num_samples: int         # 实际运行的样本数（可能因提前退出 < K）
    num_successes: int       # 成功的样本数
    pass_k: bool             # 任意样本成功 → True
    best_progress_rate: float       # 所有样本中最高的顺序进展
    best_scattered_progress_rate: float  # 所有样本中最高的散布进展
    domain: str
    category: str
    sample_results: List[TrajectoryResult]  # 各样本的完整结果
```

##### 配置与 CLI 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--num_samples` | int | 1 | K 值（1 = 贪心解码，>1 = Pass@k） |
| `--temperature` | float | 0.7 | 采样温度（仅 K > 1 时生效） |

##### 核心实现

**`evaluate_trajectory_pass_k(trajectory, num_samples, temperature)`**：
- 对同一轨迹运行 K 次独立 rollout，每次使用温度采样
- **提前退出优化**：首次成功即停止采样（参考 `eval_qwenvl_pass_k.py:183`）
- 返回 `PassKTrajectoryResult`

**`_evaluate_pass_k(trajectories, thread_num)`**：
- ThreadPoolExecutor 对轨迹间并行
- 每个线程内对 K 次 rollout 串行执行
- 超时时间随 K 线性缩放：`timeout = 1800 * K`

**`_calculate_pass_k_statistics()`** 计算的指标：

| 指标 | 描述 |
|------|------|
| `pass_k_rate` | K 次中任意一次成功的轨迹占比 |
| `pass_1_rate` | 仅第一次采样的成功率（相当于贪心） |
| `sampling_lift` | `pass_k_rate - pass_1_rate`，温度采样带来的提升 |
| `avg_best_progress` | 每条轨迹中最优样本的顺序进展均值 |
| `avg_best_scattered_progress` | 最优样本的散布进展均值 |

**`_save_pass_k_results(stats)`**：
- 精简存储：每个样本仅保存摘要字段（不存完整 `step_results`），节约内存
- 文件名格式：`pass_k{K}_evaluation_results_{timestamp}.json`

##### 运行示例

```bash
cd /scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/train_GUI_360/GUI-360-eval

# 小样本测试：5 条轨迹、K=3
python evaluation.py \
    --root_dir /path/to/gui360/dataset \
    --type action_prediction_ar \
    --model_type qwen2.5_vl_7b \
    --model_name /path/to/model \
    --api_url http://host:port/v1 \
    --max_samples 5 \
    --num_samples 3 \
    --temperature 0.7 \
    --threads 2 \
    --output_dir results/test_pass_k
# 验证：输出应包含 pass_k_rate, pass_1_rate, sampling_lift

# 完整 Pass@5 评估
python evaluation.py \
    --root_dir /path/to/gui360/dataset \
    --type action_prediction_ar \
    --model_type qwen2.5_vl_7b \
    --model_name /path/to/model \
    --api_url http://host:port/v1 \
    --num_samples 5 \
    --temperature 0.7 \
    --threads 8 \
    --output_dir results/qwen25vl_sft_v2/pass_k5
```

##### 终端输出示例

```
EVALUATION SUMMARY
============================================================
...
Pass@K Metrics (K=5):
  Pass@K Rate: 0.2345
  Pass@1 Rate: 0.1695
  Sampling Lift: 0.0650
  Avg Best Progress: 0.3421
...
============================================================
```

---

#### 9.7.3 P1：子目标进展框架

##### 背景与动机

调研报告第 3.3 节指出，部分完成度量是"指标研发最活跃的方向"。AgentBoard 的 Progress Rate 需要逐任务人工标注子目标，成本高。本实现提供 **自动子目标提取**，无需人工标注，适合 GUI-360 的大规模场景。

##### 新建文件 `evaluator/subgoal_progress.py`

**数据结构**：

```python
@dataclass
class SubGoal:
    id: int
    description: str
    step_range: Tuple[int, int]  # (起始步, 结束步) 包含边界, 1-indexed
    weight: float = 1.0

@dataclass
class SubGoalResult:
    subgoal_id: int
    achieved: bool             # 子目标范围内所有步骤均成功
    steps_correct: int
    steps_total: int
    partial_credit: float      # 范围内正确步骤占比（部分信用）

@dataclass
class TrajectorySubGoalMetrics:
    num_subgoals: int
    num_achieved: int
    subgoal_progress_rate: float            # 达成 / 总数（二元）
    weighted_subgoal_progress_rate: float   # 加权部分信用
    subgoal_results: List[SubGoalResult]
```

**`SubGoalExtractor` — 两种提取方法**：

| 方法 | 输入 | 原理 | 适用场景 |
|------|------|------|----------|
| `extract_from_thoughts(steps)` | 单条轨迹的步骤列表 | 基于关键词的边界检测：动作类型变化、status 变化、thought 中的话题转移关键词 | 通用，无需多条轨迹 |
| `extract_from_lcs(successful_trajectories)` | 同一任务的多条成功轨迹 | ProgRM 风格：跨轨迹计算 LCS，关键位置标记子目标边界 | 有多条成功轨迹时更准确 |

**`extract_from_thoughts` 边界检测规则**：
1. **动作类型变化**：前后步的 `ground_truth_function` 不同 → 边界
2. **状态变化**：`status` 字段变化（如 CONTINUE → FINISH） → 边界
3. **话题转移关键词**：当前步的 thought 包含 `open/navigate/switch/search/type/submit/save/close/finish` 等关键词，而前步不包含 → 边界

**`SubGoalProgressEvaluator`**：
- 子目标"达成"标准：范围内 **所有步骤均成功**
- 二元进展：`num_achieved / num_subgoals`
- 加权进展：`Σ(weight_i × partial_credit_i) / Σ(weight_i)`

##### 与 long_horizon.py 的集成

| 组件 | 变更 |
|------|------|
| `TrajectoryMetrics` | 新增可选字段 `subgoal_progress_rate`、`weighted_subgoal_progress_rate` |
| `LongHorizonAnalysis` | 新增可选字段 `avg_subgoal_progress_rate`、`avg_weighted_subgoal_progress_rate` |
| `analyze(enable_subgoals)` | 当 `enable_subgoals=True` 时自动提取子目标并评估 |
| `format_markdown_report()` | 当有子目标数据时显示在 Overall Metrics 表中 |

##### 验证命令

```bash
cd /scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/train_GUI_360/GUI-360-eval

# 启用子目标分析
python scripts/analyze_trajectory_metrics.py \
    --results_file results/qwen25vl_sft_v2/action_prediction_*/evaluation_results_*.json \
    --output_dir results/trajectory_analysis/test_subgoals \
    --enable_subgoals \
    --format both
# 验证：输出应包含 subgoal_progress_rate, weighted_subgoal_progress_rate
```

##### 示例输出

```
TRAJECTORY ANALYSIS SUMMARY
============================================================
...
Average Progress Rate (scattered): 0.5060
Average Sequential Progress Rate: 0.3421
...
```

Markdown 报告中新增行（当启用子目标时）：

```
| Sub-goal Progress Rate | 0.4200 |
| Weighted Sub-goal Progress Rate | 0.4850 |
```

##### 与调研报告中已有方案的对应关系

| 调研报告推荐 | 本实现 | 状态 |
|--------------|--------|------|
| 第一优先级：TSR | 已有，无变更 | ✅ |
| 第一优先级：Pass@k | P0-B 实现 | ✅ |
| 第二优先级：子目标进展率 | P1 `extract_from_thoughts` | ✅ 自动化版本 |
| 第二优先级：里程碑 ATP | P1 `extract_from_lcs` | ✅ ProgRM 风格 |
| 第三优先级：PRM | 基于 P1 的 LCS 提取可为 PRM 提供标签 | 🔜 后续 |
| 第一优先级：RRR | 需对接参考轨迹步数 | 🔜 后续 |

---

#### 9.7.4 完整更新工作流

```bash
cd /scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/train_GUI_360/GUI-360-eval

# Step 1: 用新指标重新分析已有 TF 结果（含顺序进展 + 子目标）
python scripts/analyze_trajectory_metrics.py \
    --results_file results/qwen25vl_sft_v2/action_prediction_*/evaluation_results_*.json \
    --output_dir results/trajectory_analysis/qwen25vl_sft_v2_v2 \
    --enable_subgoals \
    --format both

# Step 2: AR 评估（stop_on_error=True 为新默认值）
python evaluation.py \
    --root_dir /path/to/gui360/dataset \
    --type action_prediction_ar \
    --model_type qwen2.5_vl_7b \
    --model_name /path/to/model \
    --api_url http://host:port/v1 \
    --threads 8 \
    --output_dir results/qwen25vl_sft_v2/action_prediction_ar

# Step 3: Pass@5 评估
python evaluation.py \
    --root_dir /path/to/gui360/dataset \
    --type action_prediction_ar \
    --model_type qwen2.5_vl_7b \
    --model_name /path/to/model \
    --api_url http://host:port/v1 \
    --num_samples 5 \
    --temperature 0.7 \
    --threads 4 \
    --output_dir results/qwen25vl_sft_v2/pass_k5

# Step 4: TF vs AR gap 对比（含顺序进展）
python scripts/analyze_trajectory_metrics.py \
    --tf_results results/qwen25vl_sft_v2/action_prediction_*/evaluation_results_*.json \
    --ar_results results/qwen25vl_sft_v2/action_prediction_ar/ar_evaluation_results_*.json \
    --output_dir results/trajectory_analysis/tf_ar_gap_v2
```
