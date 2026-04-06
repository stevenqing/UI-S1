# UI-S1 Implementation 全局总结

> 最后更新：2026-03-23
> 项目：GUI Agent Semi-Online RL Training
> 基础模型：Qwen2.5-VL-7B-Instruct

---

## 一、项目全景

UI-S1 是一个 **GUI 自动化 Agent 的 Semi-Online RL 训练框架**。核心挑战：训练多模态 VLM 在视觉界面上执行多步动作序列（点击、滑动、输入文字等），其中长轨迹任务的成功率受限于逐步误差累积（per-step accuracy ~36.5% → TSR ~1%）。

### 核心发现（来自 Exp0-Exp2 诊断实验）

| 发现 | 数据 | 来源 |
|------|------|------|
| 错误根因是 **planning**（选错 UI 元素），非 grounding | 71-98% planning error vs 2-6% grounding error | Exp0, Exp2 |
| Step accuracy 随前置错误 **线性衰减** | -6.6pp/step, R²=0.91 | Exp2 |
| **Failure zone**（pw≥3, pos≥3）精度仅 20.5%，占 30% steps | 不可逆 | Exp2 |
| 所有 **inference-time** 方法天花板 = F4 (+5pp) | verifier/reset/oracle history 均失败 | Exp1, Exp2 |
| **Mismatch tax** = +4.9pp，但非长轨迹失败根因 | GT history vs model history | Exp2 |
| Best-of-K 采样上限：K=10 → 61% step acc | 协议级 agreement ≥0.9 → 83% accuracy | Exp1 |

### 结论：必须在 **training time** 解决，inference-time 不够

---

## 二、系统架构

```
UI-S1/
├── uis1/core_uis1.py              # 核心 RL 算法：GiGPO, UIS1, SP 优势计算
├── verl/                           # 修改后的 VeRL 框架
│   ├── trainer/ppo/
│   │   ├── core_algos.py          # 枚举：GRPO, UIS1, SP_GIGPO, GAE, RLOO...
│   │   ├── ray_trainer.py         # FSDP PPO Trainer (compute_advantage 入口)
│   │   ├── dapo_ray_trainer.py    # 轨迹级 DAPO Trainer (SP/SPWA 集成)
│   │   └── moe_dapo_trainer.py    # MoE-aware Trainer
│   ├── models/moe/                # MoE 模块
│   │   ├── moe_wrapper.py         # Two-pass forward: Router → Expert LoRA
│   │   ├── router.py              # TextOnlyRouter (指令特征 → 专家权重)
│   │   ├── expert_lora.py         # 独立 LoRA per expert per layer
│   │   └── moe_loss.py            # Balance loss + Z-loss
│   └── utils/
│       ├── dataset/rl_dataset.py  # TrajDataset + override_data_source
│       └── reward_score/
│           ├── sp_reward.py       # SP 计算 + SPWA 权重
│           ├── gui_traj.py        # AndroidControl action matching
│           ├── gui360/            # GUI-360 grounding + action scoring
│           └── gui360_pamarl/     # Near-miss credit + φ(t) weighting
├── train/
│   ├── sp_gigpo/                  # SP+GiGPO+SPWA 训练 (AndroidControl)
│   ├── moe_rl_v5/                 # MoE RL 训练 (AndroidControl)
│   └── pamarl_validation/         # PAMARL 验证训练 (GUI-360)
├── train_GUI_360/                 # GUI-360 SFT + eval pipeline
├── scripts/exp0/, exp1/, exp2/    # 诊断实验脚本
├── Option-incentivized-MoE/       # MoE 研究文档
└── Model_Based_Tools/             # 规划与分析
```

---

## 三、算法实现总览

### 3.1 Advantage Estimator 对照表

| Estimator | 信号粒度 | 分组方式 | 用途 |
|-----------|---------|---------|------|
| **UIS1** | episode + step | `(uid)` episode / `(uid, step_id)` step | Baseline：二值 step reward + cross-traj norm |
| **SP_GIGPO** | step-level extract_match | `(uid, step_id)` GiGPO anchor group | **当前主力**：正确/错误步独立获得正/负 advantage |
| **GRPO** | episode-level | `(uid)` | 标准 outcome GRPO |
| **GAE** | token-level | — | 需要 critic，不常用 |

### 3.2 SP+GiGPO+SPWA 三件套（核心创新）

```
┌─────────────────────────────────────────────────────┐
│ 1. Sequential Progress (SP)                         │
│    SP = first_error_step / total_steps ∈ [0, 1]    │
│    用于 SPWA 权重计算（哪些步重要）                    │
├─────────────────────────────────────────────────────┤
│ 2. GiGPO (Cross-Trajectory Step Comparison)         │
│    分组: (uid, step_id) → K rollout 共享 GT screenshot│
│    评分: step-level extract_match (binary 0/1)      │
│    advantage = (score_i - mean) / (std + eps)       │
│    ✅ 正确步 → 正 advantage, 错误步 → 负 advantage   │
├─────────────────────────────────────────────────────┤
│ 3. SPWA (Sequential Progress Weighted Advantage)    │
│    错误前: weight = 1.0（全力奖励）                   │
│    首个错误: weight = 1.0（关键分叉点）               │
│    错误后: weight = decay^(t-first_error), min 0.1   │
│    → 乘到 GiGPO advantage 上                        │
└─────────────────────────────────────────────────────┘
```

### 3.3 数据流

```
dapo_ray_trainer.py:
  rollout concat → compute_sequential_progress(batch)
                    → sp_scores, first_error_steps        ← 用于 SPWA
                    → batch.non_tensor_batch['sp_scores']
                    → batch.non_tensor_batch['first_error_steps']
                 → compute_step_discounted_returns(batch)
                    → batch.batch['step_rewards']

ray_trainer.py compute_advantage():
  SP_GIGPO branch:
    extract_match = batch.non_tensor_batch['extract_match']  ← step-level 0/1
    step_scores = [1.0 if m else 0.0 for m in extract_match]
    → compute_sp_gigpo_advantage(step_scores, ...)           ← GiGPO 分组标准化
    → advantages                                              ← (bs, response_length)

dapo_ray_trainer.py (SPWA 加权):
    spwa_weights = compute_spwa_weights(sp_scores, first_error_steps, ...)
    advantages *= spwa_weights                                ← 按 SP 位置加权
```

---

## 四、实验线 A：SP+GiGPO+SPWA 训练（AndroidControl）

### 训练配置

| 参数 | 值 |
|------|:--:|
| 数据 | `ui_s1_train.jsonl` (1,000 trajectories, 8,444 steps) |
| 模型 | Qwen2.5-VL-7B-Instruct (无 MoE) |
| adv_estimator | `sp_gigpo` |
| K (rollouts/prompt) | 4 |
| SPWA | `use_spwa=true`, `decay=0.5` |
| KL coef | 0.1 |
| clip_ratio | 0.1 |
| Nodes | 4 × 4 GPUs = 16 GPUs |
| Epochs | 5 |

### 版本演进

| 版本 | Job ID | 状态 | 关键变更 | 问题 |
|:----:|:------:|:----:|---------|------|
| v1 | 3267762 | ✅ 完成 | 初始实现：GiGPO 使用 trajectory-level SP | 为 v3 提供 baseline |
| v3 | 3279307 | 🔄 运行中 | KL=0.1, `combine_with_uis1=false` | task_acc 0.120→0.100 ↓, SSR plateau ~0.41 |
| **v4** | 待提交 | 📋 已就绪 | **核心修复**：GiGPO 改用 step-level extract_match | — |

### v3 → v4 核心修复：Credit Assignment 问题

**问题诊断**：

v3 的 `compute_sp_gigpo_advantage()` 使用 **trajectory-level SP** 作为 GiGPO 评分。由于 SP = `first_error_step / total_steps` 是每条轨迹的单一标量，同一轨迹所有步骤得到相同的 advantage 符号：

```
轨迹 A: SP=0.4 → step_0(correct) advantage=+, step_3(wrong) advantage=+  ← 错！
轨迹 B: SP=0.0 → step_0(wrong)   advantage=-, step_3(wrong) advantage=-
```

- 正确步和错误步获得相同的 advantage → 信号混乱
- K=4 rollout 多数 SP=0 → GiGPO group 方差极低 → 零梯度
- SSR 停滞在 ~0.41, task_acc 反而下降

**修复**（v4，已实现）：

GiGPO 评分改用 **step-level extract_match**（每步独立的二值匹配）：

```python
# ray_trainer.py:293-302 (v4)
extract_match = data.non_tensor_batch['extract_match']
step_scores = np.array([1.0 if m else 0.0 for m in extract_match], dtype=np.float32)
sp_gigpo_advantages = core_uis1.compute_sp_gigpo_advantage(
    sp_scores=step_scores,  # ← step-level, NOT trajectory-level
    ...
)
```

| 对比 | v3 (Before) | v4 (After) |
|------|-------------|------------|
| GiGPO 评分 | trajectory SP (同轨迹所有步相同) | step extract_match (每步独立) |
| 正确步 | 可能获得负 advantage | 始终获得正 advantage |
| 错误步 | 可能获得正 advantage | 始终获得负 advantage |
| SP=0 group | 全部相同 → 零梯度 | match/no-match 混合 → 有效梯度 |
| combine_with_uis1 | false | false (GiGPO 已有 step 信号，不需 UIS1 重复) |
| SPWA | 加权统一符号的 advantage | **加权正确符号的 advantage** |

**修改文件**：

| 文件 | 变更 |
|------|------|
| `verl/trainer/ppo/ray_trainer.py:293-302` | GiGPO 评分从 `sp_scores` 改为 `extract_match` |
| `train/sp_gigpo/traj_grpo_sp_gigpo.yaml:144` | `combine_with_uis1: false` |
| `train/sp_gigpo/train_sp_gigpo.slurm:45` | experiment_name → `sp_gigpo_spwa_k4_v4` |

### 验证计划

```bash
# 1. Cancel v3 job
scancel 3279307

# 2. Submit v4
sbatch train/sp_gigpo/train_sp_gigpo.slurm

# 3. 监控
# wandb project: sp_gigpo_ac, experiment: sp_gigpo_spwa_k4_v4
# 关键指标：
#   - SSR 应突破 0.42 而非停滞
#   - task_acc 应单调上升
#   - val-core/sequential_progress/mean 应持续改善
#   - advantage 方差应显著增大（组内有 0/1 差异）
```

---

## 五、实验线 B：PAMARL 验证训练（GUI-360）

### 背景

在 GUI-360（桌面 Office 应用）数据集上验证 **Near-Miss Credit** 和 **φ(t) 时间加权** 的效果。

### 三条件对比设计

| 条件 | Reward 函数 | 关键机制 |
|:----:|:----------:|---------|
| **A** (baseline) | `gui360` | 标准二值 action match |
| **B** (near-miss) | `gui360_pamarl` + `PHI_T=0` | 相似动作获部分分数 (0.08-0.15)，如 click↔type 混淆 |
| **C** (PAMARL) | `gui360_pamarl` + `PHI_T=1` | near-miss + φ(t) 时间加权（长轨迹晚期步权重更大） |

### 实现状态 ✅ Phase 0 完成

| 组件 | 文件 | 状态 |
|------|------|:----:|
| PAMARL reward | `verl/utils/reward_score/gui360_pamarl/reward.py` | ✅ |
| 训练配置 | `train/pamarl_validation/traj_grpo_pamarl_val.yaml` | ✅ |
| SLURM 脚本 | `train/pamarl_validation/train_pamarl_val.slurm` | ✅ |
| 评估脚本 | `scripts/eval/eval_pamarl_validation.py` | ✅ |
| data source 注册 | `verl/utils/reward_score/__init__.py` | ✅ |
| override_data_source | `verl/utils/dataset/rl_dataset.py` | ✅ |

### 训练状态

| 条件 | Job ID | 状态 | 备注 |
|:----:|:------:|:----:|------|
| A | 2848851 | 🔄 运行中 | 4 nodes, 2 epochs, subset 2000 |
| B | 2848852 | 🔄 运行中 | 同上，near-miss credit |
| C | 2848853 | 🔄 运行中 | 同上，near-miss + φ(t) |

> Bug 修复历经 8 轮提交（hostname 路径、bash 路径、Ray 变量转义、OOM、warmup 过长...）

### 成功标准

| 指标 | Baseline (A) | 目标 (C) | 核心 prediction |
|------|:------------:|:--------:|:---------------:|
| func_match | ~88% | ≥91% | near-miss 减少 click↔type 混淆 |
| Long TSR lift / Short TSR lift | — | >2.0 | φ(t) 在长轨迹上更有效 |

---

## 六、实验线 C：MoE RL 训练（AndroidControl）

### 架构设计

```
Qwen2.5-VL-7B-Instruct (frozen base)
    ├── TextOnlyRouter: instruction → softmax weights [B, 4]
    └── 4 Expert LoRAs (random-init, r=32, alpha=64)
        ├── Expert 0: LoRA on 7 target modules
        ├── Expert 1: LoRA on 7 target modules
        ├── Expert 2: LoRA on 7 target modules
        └── Expert 3: LoRA on 7 target modules

Forward: base_output + Σ(weight_k × expert_k_lora_delta)
```

### 配置 (`train/moe_rl_v5/traj_grpo_moe_uis1.yaml`)

| 参数 | 值 | 说明 |
|------|:--:|------|
| num_experts | 4 | — |
| top_k | 4 | soft routing, 所有专家参与 |
| expert_lora_r | 32 | LoRA rank |
| target_modules | 7 | q/k/v/o_proj + gate/up/down_proj |
| balance_weight | 0.2 | 负载均衡损失 |
| z_loss_weight | 0.01 | 防止 logit 饱和 |
| entropy_coeff | 0.01 | **关键**：防止 router collapse |
| adv_estimator | `uis1` | 二值 step reward + episode reward |
| Nodes | 4 × 4 GPUs | 可扩到 8 nodes |

### 状态

- 📋 配置完成，尚未提交（等 SP+GiGPO 结论后决定是否集成）
- 前期 MoE 训练（v1-v3）遭遇 router collapse，已通过 `entropy_coeff=0.01` 修复

---

## 七、Option-Incentivized MoE 研究

### 核心文档

| 文档 | 内容 |
|------|------|
| `FULL_SFT_TO_MOE_DESIGN.md` | SVD 提取 Full-SFT → LoRA：ΔW = W_sft - W_base, SVD(ΔW) ≈ B@A |
| `f_pseudo_tldr.md` | f_pseudo（特征函数 reward）的三个致命问题：无方向性、信号稀疏、长轨迹噪声 |
| `bottleneck_crossing_reward_design.md` | 5,461 对成功/失败轨迹分析：58% 分叉发生在前 20%，66% 为坐标错误 |
| `multi_agent_experiment_design.md` | 双模型协同：Grounding SFT v3 (79.48%) + All-round SFT v2 (46.90%) |
| `multi_agent_stochastic_tool_design.md` | 随机工具选择机制设计 |

### SVD LoRA 提取结果

- Full-SFT v2 action prediction: 46.90%
- LoRA v4 (from-scratch): 27.53% → **-19.37pp 差距**
- SVD 提取理论上可恢复大部分差距
- 脚本：`extract_fullsft_to_lora.py`

### f_pseudo 结论

eigenfunction-based reward **当前不可用**：
1. 54.6% self-loops → f(s_start) ≈ f(s_end), 累积 ≈ 0
2. 72.6% 长轨迹 zero signal
3. 需替换为 Progress Estimator p(s) ∈ [0,1]

---

## 八、GUI-360 SFT Pipeline

### 训练版本

| 版本 | 方法 | Action Prediction | 备注 |
|:----:|------|:-----------------:|------|
| SFT v2 | Full-param, all-round | 46.90% | 最强 baseline |
| SFT v3 | Full-param, grounding-only | 3.07% (79.48% grounding) | 视觉专家 |
| LoRA v4 | LoRA r=128, 2 epochs | 27.53% | 容量不足 |
| Grounding SFT | Full-param, grounding-specific | — | 各 checkpoint 评估完成 |

### 评估结果（`train_GUI_360/GUI_360_all_eval_results.md`）

| 模型 | TSR (stop) | Step SR (stop) |
|------|:----------:|:--------------:|
| SFT v2 (full) | 16.2% | 55.3% |
| Qwen2.5-VL Base | 1.6% | 22.1% |
| OS-Atlas-Pro-7B | 2.1% | 14.5% |

---

## 九、评估框架

### AndroidControl 指标

| 指标 | 含义 |
|------|------|
| **task_acc (TSR)** | 全步正确 = 1，否则 = 0 |
| **sequential_progress (SP)** | first_error_step / total_steps |
| **extract_match** | 逐步 action 完全匹配 (type + args) |
| **type_match** | 仅 action type 匹配 |
| **step_success_rate (SSR)** | 所有步的平均 extract_match |

### 评估数据集

| 数据集 | 规模 | 用途 |
|--------|:----:|------|
| `android_control_evaluation_std.jsonl` | 完整 | 正式评估 |
| `android_control_evaluation_std_50.jsonl` | 50 条 | 快速验证 |
| GUI-360 AR test | 3,233 条 | GUI-360 轨迹评估 |

---

## 十、诊断实验汇总

### Exp0：基础诊断

| 脚本 | 结论 |
|------|------|
| `exp0_1_uncertainty_analysis.py` | 不确定性与失败相关 |
| `exp0_2_zeroshot_verification.py` | Zero-shot verifier 效果有限 |
| `exp0_3_oracle_recovery.py` | Oracle history → +4.9pp (mismatch tax) |
| `exp0_4_domain_fcv.py` | Domain 分布影响训练效果 |
| `exp0_5_retry_with_context.py` | Retry 收益递减 |

### Exp1：多模型 + 采样

| 脚本 | 结论 |
|------|------|
| `exp1_1_sft_v3_multisample.py` | K=5 → 51%, K=10 → 61% (grounding SFT v3) |
| `exp1_3_oracle_coord_replacement.py` | 坐标非主要瓶颈 (Go/No-Go gate) |
| `exp1_5_dual_model_eval.py` | V+H division: +5pp (F4, 当前天花板) |
| `exp1_6_error_diversity.py` | 错误模式多样性分析 |

### Exp2：认知干扰 + 失败区

| 发现 | 数据 |
|------|------|
| Step accuracy 线性衰减 | -6.6pp/step, R²=0.91 |
| Failure zone (pw≥3) | 20.5% accuracy, 占 30% steps |
| 所有 inference-time 方法天花板 | F4 = +5pp，无法触及 failure zone |
| 必须 training-time 解决 | → SP+GiGPO+SPWA motivation |

---

## 十一、当前 TODO / Next Steps

### 立即执行

- [ ] **Cancel v3 (Job 3279307)**, submit v4 — credit assignment 修复
  ```bash
  scancel 3279307
  sbatch train/sp_gigpo/train_sp_gigpo.slurm
  ```
- [ ] 监控 v4 训练：SSR 是否突破 0.42, task_acc 是否单调上升

### 待 v4 结果

- [ ] 如果 v4 有效 → 考虑集成 MoE (v5): SP_GIGPO + MoE soft routing
- [ ] 如果 v4 仍停滞 → 分析 GiGPO group 内方差，检查 extract_match 分布
- [ ] 对比 v4 vs UIS1 baseline 的 delta

### PAMARL 方面

- [ ] 等 A/B/C 三条件训练完成
- [ ] 运行 `scripts/eval/eval_pamarl_validation.py --compare` 进行横向对比
- [ ] 验证核心 prediction: Long lift / Short lift > 2.0

### 研究方向

- [ ] Progress Estimator p(s) ∈ [0,1] 替代 f_pseudo（f_pseudo 已证明不可用）
- [ ] SVD LoRA 提取 → MoE 专家 warm-start（缩小 19.37pp 差距）
- [ ] Multi-agent specialization: π_V (视觉) + π_S (状态) + π_A (动作)

---

## 十二、关键文件速查

| 用途 | 文件路径 |
|------|---------|
| GiGPO advantage 计算 | `uis1/core_uis1.py:67-116` |
| SP_GIGPO advantage 入口 | `verl/trainer/ppo/ray_trainer.py:293-323` |
| SP 计算 + SPWA 权重 | `verl/utils/reward_score/sp_reward.py` |
| SPWA 加权应用 | `verl/trainer/ppo/dapo_ray_trainer.py:519-534` |
| SP 训练 metrics | `verl/trainer/ppo/dapo_ray_trainer.py:600-607` |
| SP 验证 metrics | `verl/trainer/ppo/dapo_ray_trainer.py:206-220` |
| MoE wrapper | `verl/models/moe/moe_wrapper.py` |
| MoE router | `verl/models/moe/router.py` |
| PAMARL reward | `verl/utils/reward_score/gui360_pamarl/reward.py` |
| SP+GiGPO 训练配置 | `train/sp_gigpo/traj_grpo_sp_gigpo.yaml` |
| SP+GiGPO SLURM | `train/sp_gigpo/train_sp_gigpo.slurm` |
| MoE RL 训练配置 | `train/moe_rl_v5/traj_grpo_moe_uis1.yaml` |
| PAMARL 训练配置 | `train/pamarl_validation/traj_grpo_pamarl_val.yaml` |
| 研究全景 | `docs/full_summary_problem_to_framework.md` |

---

## 十三、资源使用

| 实验 | Nodes | GPUs | Walltime | 数据 |
|------|:-----:|:----:|:--------:|:----:|
| SP+GiGPO v4 | 4 | 16 | 24h | 1K traj (AndroidControl) |
| PAMARL A/B/C | 4×3 | 48 | 24h×3 | 2K subset (GUI-360) |
| MoE RL v5 (待) | 4-8 | 16-32 | 24h | 1K traj (AndroidControl) |
| 评估 | 1 | 4 | ~2h | 测试集 |
