# 多 SOTA Agent 错误轨迹 → 全局修订 → 直接噪声训练实验总结

Research framing、Counterfactual Revision Utility、source-conditioned asymmetry 与下一轮因果实验设计见 [Research Contribution](RESEARCH_CONTRIBUTION_ZH.md)。

## 1. 原始要求与完成情况

本实验针对两个要求建立并完整运行了端到端流程：

1. **使用不同 SOTA GUI agent 运行同一批任务，获得多种、彼此不同的错误轨迹。**
2. **让另一个 agent 读取完整轨迹并进行全局修改，然后不保证修改结果正确、也不按 matcher 正确性筛选，直接用结构上可训练的数据训练模型，再通过隔离测试集判断是否有效。**

两个要求均已完成，并先做小规模 pilot，再做全量 train/full-parameter training/全量 test 的正式实验。

> 实验边界：这是离线 teacher-forced GUI-360 实验。Actor 使用数据集给定的截图和历史，而不是在真实环境中闭环 rollout；global corrector 可以看到完整截图序列，包括未来截图。因此，它检验的是“异构错误数据生成 + 离线全局轨迹改写 + 噪声训练”的数据价值，不等价于在线 agent 自主执行或在线纠错。

---

## 2. 为要求 1 做了什么：异构 SOTA actor 生成多样错误轨迹

### 2.1 Pilot：先验证整条链路

Pilot 使用：

- Actor A：Qwen3-VL-8B
- Actor B：Qwen3.5-9B
- 12 个 GUI-360 train episodes / 53 steps
- 两个 actor 都运行全部 12 个 episode，共生成 24 条 actor trajectory
- 24/24 条 trajectory 都至少有一个 matcher 诊断错误

Pilot 多样性：

| 指标 | 结果 |
|---|---:|
| Action disagreement | 79.25% |
| Error-set Jaccard | 72.97% |
| Qwen3-VL step accuracy | 39.62% |
| Qwen3.5 step accuracy | 39.62% |

这一步确认了不同 agent 即使总体准确率相近，也会产生大量不同动作和不同错误位置。

### 2.2 Full：在全部 train 数据上生成错误轨迹

正式运行改用更异构的两个 actor：

- Actor A：Qwen3-VL-8B
- Actor B：InternVL3-8B
- 数据范围：全部 1,573 个 GUI-360 train episodes / 12,574 steps
- 每个 actor 都运行完整 train split
- 总 actor trajectories：3,146（1,573 × 2）
- 含错误 trajectories：3,076

Actor 结果：

| Actor | Episodes | Error episodes | Parse rate | Step accuracy |
|---|---:|---:|---:|---:|
| Qwen3-VL-8B | 1,573 | 1,511 | 92.44% | 27.08% |
| InternVL3-8B | 1,573 | 1,565 | 54.96% | 11.66% |

错误多样性：

| 指标 | 结果 |
|---|---:|
| Action disagreement | **91.68%** |
| Error-set Jaccard | **71.11%** |

解释：

- 91.68% action disagreement 表明两个 actor 在相同步骤上几乎总是给出不同动作，满足“多种不同错误 trajectory”的核心要求。
- 71.11% error-set Jaccard 表明它们也共享一部分困难步骤，但不是简单复制同一种错误。
- 最终保留了 3,076 条含错误的完整 trajectory，作为 global correction 输入。

权威产物：

- [Full actor summary](full_v1/actor_summary.json)
- [Full actor trajectories](full_v1/actor_trajectories.jsonl)
- [Full error trajectories](full_v1/error_trajectories.jsonl)

---

## 3. 为要求 2 做了什么：其他 agent 全局改轨迹并直接训练

### 3.1 Global corrector 设计

Full 实验统一使用独立的 Qwen3.5-9B 作为 global corrector：

- Qwen3-VL actor → Qwen3.5 corrector
- InternVL3 actor → Qwen3.5 corrector

Corrector 不是逐步只看当前截图，而是读取：

- 完整任务描述
- 完整 actor action trajectory
- 完整截图序列
- 后续/未来截图

然后一次性重写整条 trajectory，使动作在全局上尽量一致。

### 3.2 只做结构恢复，不做语义正确性过滤

为了严格满足“修改后的数据不一定正确，但直接训练看看”的要求：

- Matcher 只用于事后诊断，**不用于训练样本选择**。
- 没有用 ground-truth action 替换 matcher-wrong correction。
- 所有结构上可执行的 revision 都保留，包括 matcher 判断错误的 revision。
- 仅进行了格式/结构恢复：JSON 提取、action schema 归一化、坐标有限值检查、`drag`→`swipe`、固定长度对齐。
- Corrector 漏掉某一步时，使用原 actor action 补齐，仅用于保持轨迹长度；不是使用 GT 补标签。
- 超出原轨迹范围的 correction row 被忽略。

最终实际用于训练的 recovery 口径：

| 指标 | 结果 |
|---|---:|
| 输入错误 trajectories | 3,076 |
| 结构可用 trajectories | 2,128 |
| Structurally usable / parse rate | 69.18% |
| Format-recovered trajectories | 2,063 |
| Flattened actions recovered | 9,945 |
| Imputed actor steps | 4,037 |
| Ignored out-of-range rows | 809 |
| Changed-step rate | 50.17% |
| Mean corrector confidence | 95.16% |

Corrector 诊断结果：

| Pair | Trajectories | Parse | Actor step acc | Revised step acc | Revised TSR | Changed steps | Confidence |
|---|---:|---:|---:|---:|---:|---:|---:|
| InternVL3 → Qwen3.5 | 1,565 | 58.72% | 11.59% | 27.62% | 3.48% | 70.90% | 95.23% |
| Qwen3-VL → Qwen3.5 | 1,511 | 80.01% | 26.35% | 25.10% | 0.99% | 37.84% | 95.11% |
| Overall | 3,076 | 69.18% | 18.94% | 26.04% | 2.07% | 50.17% | 95.16% |

关键现象：

- Corrector 对很弱的 InternVL3 trajectory 有明显修复，但修后准确率仍只有 27.62%。
- 对 Qwen3-VL trajectory，修改后反而从 26.35% 降至 25.10%。
- Overall revised label accuracy 只有 26.04%，但 corrector 自报平均 confidence 达 95.16%，出现严重 confident-wrong / confidence miscalibration。

原 schema gate 因 69.18% 低于 80% 阈值而失败。由于原要求明确要求即使数据可能不正确也直接训练，因此记录了显式 override，只要动作结构可执行就进入训练，而不按 matcher 正确性筛选。

权威产物：

- [Raw global corrections](full_v1/global_corrections.jsonl)
- [Recovered corrections](full_v1/global_corrections_recovered.jsonl)
- [Recovered correction summary](full_v1/global_corrections_recovered.summary.json)
- [Explicit low-schema override](full_v1/low_schema_coverage_override.json)

### 3.3 构造直接训练数据

从 2,128 条结构可用 correction trajectory 中展开得到：

| 字段 | 结果 |
|---|---:|
| 唯一 noisy SFT rows | 14,456 |
| 覆盖 train episodes | 1,385 |
| Qwen3-VL → Qwen3.5 rows | 9,065 |
| InternVL3 → Qwen3.5 rows | 5,391 |
| Diagnostic matcher accuracy | 26.04% |
| Semantic quality filter | False |
| Matcher used for selection | False |

转换为 LLaMA-Factory ShareGPT multimodal 数据时，为满足有效 batch 48 的整除要求增加 40 条 padding examples，因此实际 epoch 中处理 14,496 examples；真实唯一训练样本仍是 14,456 条。

- [Noisy SFT data](full_v1/noisy_global_corrections_train.jsonl)
- [Noisy SFT summary](full_v1/noisy_global_corrections_train.summary.json)
- [LLaMA-Factory preparation manifest](full_v1/llamafactory_data/preparation_manifest.json)

---

## 4. 先做的小规模直接噪声训练 Pilot

Pilot 采用 cross-agent correction：

- Qwen3-VL actor → Qwen3.5 corrector
- Qwen3.5 actor → Qwen3-VL corrector
- 20/24 条 correction trajectory 结构可用
- 90 条 noisy SFT rows
- Diagnostic label accuracy：35.56%
- 不使用 matcher 过滤

训练：

- Base：GUI-360 full-parameter SFT step 250
- LoRA-only SFT
- 25 updates
- Effective batch 8
- Final LM loss 0.3645

在隔离的 24 test episodes / 160 steps 上：

| 指标 | Baseline | Pilot post | Delta |
|---|---:|---:|---:|
| TSR | 4.17% | 12.50% | +8.33pp |
| Step accuracy | 56.25% | 55.00% | -1.25pp |

Paired TSR bootstrap 95% interval 为 [0pp, +20.83pp]；区间包含 0，且 step accuracy 下降超过预设 -1pp guardrail。因此 pilot gate 为 **NO CLEAR HELD-OUT SIGNAL**，不能将小样本上的 task-level 上升当作稳定收益。

- [Pilot overall report](pilot_v1/overall_report.md)
- [Pilot paired summary](pilot_v1/training_eval/summary.json)

---

## 5. 全量六卡全参数训练

用户要求使用 6 GPU 且全量参数，因此正式训练配置为：

| 配置 | 值 |
|---|---|
| Initial checkpoint | `checkpoints/gui360-fullparam-sft-step250` |
| Finetuning | Full parameter SFT |
| Trainable parameters | 8.292B / 8.292B |
| Vision tower | Unfrozen |
| Multimodal projector | Unfrozen |
| Language model | Unfrozen |
| Distributed strategy | 6-GPU DeepSpeed ZeRO-3 |
| Precision | BF16 |
| Per-device batch | 1 |
| Gradient accumulation | 8 |
| Effective batch | 48 |
| Learning rate | 6e-6 |
| Scheduler | Cosine, warmup ratio 0.03 |
| Epochs | 1 complete epoch |
| Optimizer steps | 302 |

先执行真实 one-step full-backward smoke gate：

- 8.292B/8.292B 参数可训练
- Loss 2.5561
- Forward/backward/optimizer step 成功

正式训练：

- Runtime：约 1 小时 55 分
- 302 optimizer steps 完成
- Aggregate train loss：0.2131
- Step 300 LM loss：0.1419
- Skipped steps：0
- Checkpoint 四个 safetensors shards 完整

这证明模型能够非常好地拟合 noisy revision 数据，但训练 loss 本身不能说明 revision 是正确监督。

- [Full training config](full_v1/fullparam_6gpu.yaml)
- [Full model checkpoint index](full_v1/fullparam_model/model.safetensors.index.json)
- [Training compatibility summary](full_v1/fullparam_training_compat/summary.json)

---

## 6. 全量隔离测试与最终结果

### 6.1 评估协议

- Test split：全部 1,000 episodes / 7,498 steps
- Test 数据从未用于 actor generation、global correction 或 training
- Train/test 使用不同 source SHA-256 和不同截图 namespace
- Teacher-forced screenshot/history
- Greedy deterministic evaluation
- 8 GPU × 8 deterministic shards，每 shard 125 episodes
- Exact merge 校验：不允许 episode/step 重复或缺失
- Baseline 和 post 使用相同冻结测试网格

### 6.2 结果

| 指标 | Baseline | Noisy-revision fullparam | Delta |
|---|---:|---:|---:|
| Task successes | 187 | 9 | -178 |
| TSR | 18.70% | 0.90% | **-17.80pp** |
| Step accuracy | 57.10% | 35.62% | **-21.47pp** |
| Parse rate | 100.00% | 99.85% | -0.15pp |

Paired TSR delta bootstrap（固定 1,000 episodes，10,000 draws）：

- Mean：-17.80pp
- 95% interval：**[-20.20pp, -15.50pp]**

Paired flips：

| Flip | Count |
|---|---:|
| Task wrong → right | 0 |
| Task right → wrong | 178 |
| Step wrong → right | 631 |
| Step right → wrong | 2,241 |

最终 gate：

**NOISY GLOBAL REVISION HARMS ON HELD-OUT GRID**

- [Full held-out report](full_v1/training_eval/report.md)
- [Full machine-readable summary](full_v1/training_eval/summary.json)
- [Final run status](full_v1/RUN_STATUS.md)

---

## 7. 对两个原始要求的直接回答

### 要求 1：是否获得了不同 SOTA agent 的多种错误 trajectory？

**是，且证据充分。**

- 两个正式 actor 都跑完全部 1,573 train episodes。
- 共生成 3,146 条 actor trajectory，其中 3,076 条含错误。
- Action disagreement 达 91.68%，说明错误动作高度多样。
- 两个模型的 parse/accuracy/error episode 结构明显不同，得到的不是单一模型的重复噪声。

### 要求 2：是否让其他 agent 全局修改，并直接用可能错误的数据训练来验证？

**是，完整执行了 pilot 和 full 两级实验。**

- Qwen3.5 读取完整 trajectory 和完整截图序列进行全局重写。
- 没有使用 matcher 正确性做语义过滤。
- 最终用全部 14,456 条结构可执行 revision rows 做了 6-GPU、8.292B 参数全量训练。
- 再在全部 1,000 个隔离 test episodes 上进行严格配对评估。

实验给出的答案不是“这些数据正确”，而是：

> **当前这批未经语义过滤的 global revisions 不适合作为全量 full-parameter 监督数据。**

Full 实验中，只有 26.04% revision action 通过 diagnostic matcher，但模型可以把训练 loss 降到 0.2131；与此同时 held-out TSR 从 18.70% 降到 0.90%。这说明“可解析、corrector 高置信、训练 loss 能下降”都不等价于“revision 具有正确训练价值”。

---

## 8. 结论边界

本结果否定的是：

- **把当前 Qwen3.5 global revisions 不加语义质量控制地全部用于 full-parameter SFT。**

本结果没有否定：

- 多 agent 错误轨迹具有研究价值；91.68% disagreement 已证明其能提供多样错误模式。
- Global correction 本身一定无效；InternVL3→Qwen3.5 的诊断准确率确有提升。
- 经过 verifier、质量分层、clean/noisy 混合、保守参数更新或 preference/RL 目标后的 revision 数据一定无效。

最重要的经验是：

1. 多 actor 确实能制造丰富、互补的错误轨迹。
2. Global corrector 能改动轨迹，但高 confidence 严重失真。
3. 结构可用性只能保证数据能训练，不能保证监督方向正确。
4. 小 pilot 的正向 TSR 波动不能支持直接放大全量训练。
5. 对 26.04% 准确的 revision 做全参数 SFT 会显著放大错误监督。
6. 下一步若继续，应先解决 revision 质量估计和训练风险控制，而不是继续扩大同一未经筛选的数据。

---

## 9. 工程与资源约束完成情况

同时完成了以下工程工作：

- 支持异构 actor/corrector endpoint、断点续跑和 provenance hash。
- 增加 correction JSON/schema recovery 与 action normalization。
- 增加 train/test source hash、截图 namespace 和固定 evaluation grid。
- 增加 6-GPU ZeRO-3 one-step full-backward smoke gate。
- 增加 8-shard deterministic evaluation 和 exact merge。
- 增加 paired bootstrap、task/step flip 和自动 gate 报告。
- 通过显式 8 GiB KV cache 避免 vLLM 动态显存 profiling 与外部任务竞争。
- 受保护外部 PID 1911 全程未被 signal、kill、重启或要求释放显存，实验结束时仍存活。
- 全部本实验 actor/corrector/training/evaluation 进程已退出，无残留任务。

主要实现：

- [Trajectory generation/revision pipeline](../../scripts/multiagent_trajectory_revision.py)
- [Fullparam data/config preparation](../../scripts/prepare_multiagent_fullparam_llamafactory.py)
- [Held-out evaluator](../../scripts/evaluate_multiagent_revision_pilot.py)
- [Exact shard merger](../../scripts/merge_multiagent_revision_eval.py)
- [Paired report generator](../../scripts/report_multiagent_revision_training.py)

---

## 10. Overnight Research Follow-Up

### Student-relative revision utility

在完全相同的 14,456 个 A4 state/history rows 上，starting student accuracy 为 54.40%，revision accuracy 只有 26.04%。

- Revision rescue student：881 steps（6.09%）。
- Revision regress student：4,981 steps（34.46%）。
- Net student-relative utility：**-28.36pp**。
- Trajectory-cluster 95% interval：**[-29.58pp, -27.18pp]**。

这直接证明：revision 即使相对弱 source actor 有 +4.52pp utility，也相对真正被训练的 student 差 28.36pp。

### History intervention

在 actor×prefix 平衡的 2,048-row 配对网格上，将 revision history 替换为 GT history：

- Frozen-student accuracy：62.11% → 73.29%，**+11.18pp**。
- 按完整 14,456-row 分布标准化后：**+16.47pp**。
- 效果集中在 dirty prefix，约 +22pp；clean prefix 近似为0。

但训练 factorial 显示：GT history 在标签正确时带来 +6.67pp step accuracy，在标签仍为错误 revision 时反而带来 -11.04pp。正确 context 会让错误 label 更直接地被模型拟合，因此只修 history 不能解决监督问题。

### Equal-budget LoRA controls

800 rows / 100 updates / 相同125-episode screen：

| Arm | ΔTSR | Δstep | Gate |
|---|---:|---:|---|
| GT target + GT history | +1.60pp | +0.44pp | HELPS |
| Revision + revision history | -17.60pp | -18.58pp | HARMS |
| Revision + GT history | -18.40pp | -29.62pp | HARMS |
| InternVL3-source only | -16.00pp | -17.27pp | HARMS |
| Qwen3-VL-source only | -16.00pp | -15.30pp | HARMS |
| Clean-prefix revision | -22.40pp | -21.75pp | HARMS |

Positive/random controls验证了trainer与评估链路；source-only、prefix-only和history-only策略均不能恢复训练价值。

### Oracle rescue与clean replay

即使oracle选择“student错误、revision正确”的rows，直接SFT仍会产生分布偏移：

| Rescue / clean replay | ΔTSR | Δstep | Gate |
|---|---:|---:|---|
| 100% / 0% | -10.40pp | -12.57pp | HARMS |
| 50% / 50% | -4.80pp | -4.48pp | HARMS |
| 25% / 75%（125-episode screen） | +1.60pp | -1.75pp | NO CLEAR SIGNAL |
| 10% / 90% | -2.40pp | -2.95pp | HARMS |

Clean replay显著缓解遗忘。25% rescue / 75% clean replay在完整1,000 episodes / 7,498 steps上得到：TSR 18.70% → **21.50%（+2.80pp）**，paired bootstrap区间 **[+1.00pp, +4.60pp]**；step accuracy 57.10% → **57.27%（+0.17pp）**。Task wrong→right/right→wrong为56/28。

这是首个严格full-grid正结果：它证明revision bank中确实存在可学习信号，但必须按student-relative utility选择，并用约75%的clean replay控制遗忘。该结果仍是oracle ceiling，下一步目标是用无GT verifier/ranker逼近这一选择策略。

### Revision verifier

Metadata-only gate在episode-disjoint test上的AP只有6.78%，无法找到正净效用coverage。

多模态candidate-packet verifier：

- Balanced training：`use_revision` precision 10.95%、recall 18.07%，fallback accuracy 54.92% < student baseline 57.20%。
- Dev-only conservative rule search没有找到正净效用rule，因此fail-closed为0% coverage。
- Natural-prior training消除了unsafe overwrite，但`use_revision` recall降为0，fallback accuracy与baseline相同。

这揭示了新的precision–recall难题：balanced training过度接受稀有revision，natural training则完全不接受。下一步应训练可校准的binary/pairwise utility ranker，并采用preference learning或强clean replay/KL，而不是继续plain SFT。

### Binary ranker与transition gate

进一步测试了三个episode-disjoint binary rescue ranker：

| Ranker | Train negatives | Dev AUC / AP | Test AUC / AP | Locked gate |
|---|---|---:|---:|---|
| v1 | regress + neutral | 0.617 / 0.085 | 0.647 / 0.086 | dev无正utility阈值 |
| v2 | regress only，rescue过采样 | 0.538 / 0.070 | 0.589 / 0.078 | dev +0.15pp，test **-0.42pp** |
| v3 | regress only，无重复样本 | 0.466 / 0.054 | 0.523 / 0.056 | dev无正utility阈值 |

同时测试了不使用GT action、只利用前后截图像素变化的transition gate：dev接受54 rows（14 rescue / 8 regress，+0.44pp），但锁定阈值在test接受56 rows（14 rescue / 15 regress，-0.06pp）。

当时的边界是：A15已证明有价值revision真实存在，但confidence、metadata、source、prefix、三分类verifier、binary ranker和简单transition consistency均无法稳定识别。因此后续冻结了新的Pass@8 selector split，并在任何dev/locked标签解封前完成全部预测。

### Pass@8 fixed-choice selector新结果

在962个历史critical steps上，将四个候选源各K=8的exact actions组成匿名blind packet。episode-disjoint split为23 smoke、231 dev、708 locked steps。blind packet不含GT、reward、correctness、source identity或GT-derived diagnostics。

| Selector | Dev utility | Locked utility | Locked cluster CI | Rescue / regress | Oracle capture |
|---|---:|---:|---:|---:|---:|
| Qwen3.5-9B | +6.49pp | **+6.36pp** | **[+4.53,+8.31]pp** | 46 / 1 | **20.93%** |
| Qwen3.5-35B-A3B | +2.60pp | +4.52pp | [+2.93,+6.20]pp | 33 / 1 | 14.88% |
| Exact plurality | +3.03pp | +3.25pp | [+1.87,+4.73]pp | 25 / 2 | 10.70% |
| Cross-source consensus | +5.63pp | +5.37pp | [+3.61,+7.25]pp | 39 / 1 | 17.67% |

Locked packet oracle为30.79%，frozen student仅0.42%，Qwen3.5-9B selected accuracy为6.78%。因此Pass@8方向成立：非oracle selector确实能稳定转化一部分proposal diversity。但“更强模型会更好”不成立：35B相对9B为-1.84pp，CI [-3.69,+0.00]pp；35B changed coverage只有34.18%，9B为60.03%。零GPU cross-source consensus达到+5.37pp且与9B差异不显著，说明独立proposal共识解释了大部分可用信号。

更新后的边界是：已获得正的非oracle hard-step candidate-recovery结果，但这不是arbitrary-state online router，也不是最终TSR。下一步只允许在独立train split生成同构候选，使用冻结9B或cross-source consensus构造25% selected revision + 75% clean replay训练臂；本实验dev/locked rows严禁进入训练，训练后必须重新做完整held-out policy evaluation。

综合自动报告见 [Overnight Final Report](full_v1/overnight/FINAL_REPORT.md)。
