# Not Every Correction Teaches：异构 GUI Agent 的 Counterfactual Revision Utility

## 0. 核心判断

当前实验不是简单的“训练失败”。它完成了一次重要的、可证伪的 intervention：

> 当多个异构 GUI agent 产生高多样性错误轨迹，再由另一个强模型做全局改写时，**actor-relative improvement、trajectory correctness、student-relative supervision value 和 downstream learning utility 并不等价**。

真正的研究问题不应是：

> “Global corrector 能不能改 trajectory？”

而应是：

> “某个 revision 在什么 source actor、什么 student、什么 prefix consistency 条件下，才具有正的训练效用？”

这可以形成一个明确的研究方向：**Counterfactual Revision Utility（CRU）与 Source/Student-Conditioned Revision Gating**。

工作标题建议：

> **Not Every Correction Teaches: Source-Conditioned Revision Utility for Heterogeneous GUI Agents**

中文：

> **不是每个修正都能教会模型：异构 GUI Agent 的源条件轨迹修订效用**

### 0.1 2026-07-12 最新结论

后续实验已经把本文最关键的两个开放问题向前推进：

1. **Oracle training ceiling 已转为严格正结果。** 在完整 1,000-episode held-out grid 上，25% student-rescue revision + 75% clean replay 将 TSR 从 18.70% 提升到 21.50%（**+2.80pp**，paired bootstrap **[+1.00pp,+4.60pp]**），step accuracy 同时 +0.17pp。
2. **Pass@8 non-oracle candidate recovery 首次通过 locked gate。** Qwen3.5-9B fixed-choice selector 在 708 个 locked critical steps 上获得 **+6.36pp** student-relative utility，episode-cluster CI **[+4.53pp,+8.31pp]**，46 rescue / 1 regress。
3. **扩大 corrector 不是答案。** Qwen3.5-35B-A3B 虽然也为正（+4.52pp），但相对 9B 为 -1.84pp，CI [-3.69pp,+0.00pp]，没有 stronger-model win。
4. **Independent proposal consensus 是主要机制。** 零 GPU cross-source consensus 达到 +5.37pp，且与 9B 的直接差异不显著。

因此，研究边界已从“存在 oracle ceiling、但没有非 oracle selector”更新为：

> **Pass@8 proposal diversity 可以被无 GT selector 稳定转化为 hard-step rescue，但这种能力不随 selector scale 单调增长；独立 proposal agreement 比盲目扩大 corrector 更关键。**

这仍不是 arbitrary-state online router，也不是最终 downstream policy gain。下一步必须先完成纯度响应曲线与 train-split aggregate purity 的桥接检验；只有纯度下界跨过经验容忍阈值，才允许用 25% selected revision + 75% clean replay 做完整 held-out policy confirmation。当前 dev/locked rows 严禁进入训练。

---

## 1. 为什么现有结果有 research contribution

### 1.1 Diversity–Utility Decoupling

正式实验让 Qwen3-VL-8B 和 InternVL3-8B 在全部 1,573 个 GUI-360 train episodes / 12,574 steps 上运行：

- Actor trajectories：3,146
- Error trajectories：3,076
- Action disagreement：91.68%
- Error-set Jaccard：71.11%

这证明 heterogeneous actors 可以产生高度多样的失败模式。

但是，将其全局修订数据直接用于 full-parameter SFT 后：

- Held-out TSR：18.70% → 0.90%（-17.80pp）
- Held-out step accuracy：57.10% → 35.62%（-21.47pp）

因此我们得到一个可辩护的经验发现：

> **Error diversity 是数据覆盖属性，不是 supervision utility。高 disagreement 本身不能推出正训练价值。**

这比“multi-agent data 很多样”更有研究意义，因为它给出了一个严格反例。

### 1.2 Counterfactual Revision Utility

对每个 step，定义 frozen matcher $M$ 下的 actor-relative revision utility：

$$
 u_t^{src}=M(a_t^{rev})-M(a_t^{actor})\in\{-1,0,+1\}.
$$

由此把每个 revision 分成四种基本结果：

| 类别 | Actor | Revision | Utility |
|---|---|---|---:|
| rescue | wrong | correct | +1 |
| regress | correct | wrong | -1 |
| preserve-correct | correct | correct | 0 |
| unresolved | wrong | wrong | 0 |

对全部 2,128 条结构可用 correction trajectories / 14,456 steps 的精确统计：

| Outcome | Count | Fraction |
|---|---:|---:|
| Rescue | 1,480 | 10.24% |
| Regress | 827 | 5.72% |
| Preserve-correct | 2,284 | 15.80% |
| Unresolved | 9,865 | 68.24% |

整体：

- Actor accuracy on usable subset：21.52%
- Revised accuracy：26.04%
- Net actor-relative utility：**+4.52pp**
- Trajectory-cluster bootstrap 95% interval：**[+3.61pp, +5.43pp]**
- Actor-wrong 条件下 rescue rate：13.05%
- Actor-correct 条件下 regression rate：26.58%
- 完整 trajectory rescue：44 / 2,128，只有 **2.07%**

这里出现了第一个重要 gap：

> Revision 在 step 平均意义上相对弱 actor 有正效用，但绝大多数错误仍 unresolved，且几乎不能修复完整 trajectory。

### 1.3 Source-Conditioned Correctability

同一个 Qwen3.5 corrector 对两个 actor source 呈现相反效果：

| Source actor | Steps | Actor acc | Revised acc | Net CRU | Rescue / Regress | Trajectory rescue |
|---|---:|---:|---:|---:|---:|---:|
| InternVL3 | 5,391 | 11.57% | 27.62% | **+16.05pp** | 1,075 / 210 | 3.48% |
| Qwen3-VL | 9,065 | 27.44% | 25.10% | **-2.34pp** | 405 / 617 | 0.99% |

两个 source 的 trajectory-cluster bootstrap interval 方向也不同：

- InternVL3：约 [+14.32pp, +17.74pp]
- Qwen3-VL：约 [-3.21pp, -1.49pp]

因此：

> **Correction quality 不是 corrector 自身的固定属性，而是 source-conditioned relation。**

同一个 corrector 可以显著改善弱 actor，同时破坏更强 actor 已经正确的动作。简单地把所有 source 混在一起训练会掩盖这种异质性。

### 1.4 Step–Trajectory Composition Gap

虽然 net step CRU 为 +4.52pp，但完整 trajectory rescue 只有 2.07%。

这是 long-horizon GUI 的重要现象：

- 局部 rescue 不会自动组合成全局成功。
- 任意一个 unresolved/regress step 都可能破坏完整任务。
- “平均 revised step accuracy 上升”并不能推出“可用于 trajectory training”。

因此应显式区分：

1. Step-level revision utility
2. Prefix/transition consistency
3. Complete-trajectory repair utility
4. Student-relative label utility
5. Downstream training utility

这五层构成 **Revision Utility Ladder**。

### 1.5 Prefix-Consistency Gap

Global corrector 在 GT screenshot 序列上改写 action；训练时 screenshot 仍来自 GT trajectory，但 history 使用 revised action prefix。

定义 diagnostic clean prefix：当前 step 之前的 revised actions 全部通过 frozen matcher。统计结果：

| Prefix | Rows | Fraction | Current-label accuracy |
|---|---:|---:|---:|
| Clean prefix | 3,796 | 26.26% | 45.10% |
| Dirty prefix | 10,660 | **73.74%** | 19.25% |

随着之前 matcher-wrong revised action 数增加，当前 label accuracy 单调下降：

| Prior wrong revisions | Rows | Current-label accuracy |
|---:|---:|---:|
| 0 | 3,796 | 45.10% |
| 1 | 2,283 | 25.41% |
| 2 | 1,664 | 19.41% |
| 3 | 1,279 | 17.90% |
| 4+ | 5,434 | 16.93% |

这不是执行环境中的 transition-equivalence 证明，而是一个严格定义的 diagnostic proxy。但它揭示了关键问题：

> **在固定未来 GT screenshots 上“全局改 action”不等于构造出 causally executable revised trajectory。**

如果较早 revised action 不会到达后续 GT screenshot，那么后续样本同时包含 revised history 和不对应的 GT state，形成 sequential contamination。

这使当前工作区别于简单 noisy-label 分析：问题不仅是 target action 错，还可能是 context/history 与 visual state 不一致。

### 1.6 Revision–Student Gap

当前 revisions 相对两个弱 actor 平均提升 +4.52pp，但训练起点不是这两个 actor，而是更强的 GUI-360 fullparam step-250 checkpoint。

Full training 结果：

- Train loss：0.2131
- Held-out step delta：-21.47pp
- Held-out TSR delta：-17.80pp

这说明：

> **Better than the source actor 不等于 good enough to supervise the student。**

当前尚未在完全相同的 14,456 个 train states 上测量 starting student actions，因此不能把 +4.52pp 与 -21.47pp 的差直接解释为因果量。但二者共同证明：actor-relative quality 不是 downstream utility 的充分条件。

### 1.7 Self-Reported Confidence 不是有效 selector

Corrector confidence 的 trajectory-level 诊断：

- 88.11% usable trajectories 的 confidence 都是 0.95，分布高度塌缩。
- Confidence 与 revised step accuracy 的 Spearman：0.009。
- Confidence 与 net CRU 的 Spearman：0.022。
- 对完整 trajectory rescue 的 AUC：0.638。
- Threshold AP：0.041，base rescue rate 为 0.0207。

Confidence 对少数完整 rescue 有有限排序信息，但基本不能排序 step quality 或 net revision utility。

同时必须明确：

- `correction_confidence` 保留在 raw metadata 中。
- 实际 LLaMA-Factory ShareGPT conversations 中没有 confidence。

所以 confidence mismatch 是 selection/calibration diagnosis，**不能声称学生因为看到了 confidence 而学坏**。

### 1.8 Pass@8 Proposal–Selection Gap 与正向非 Oracle Gate

历史 Pass@8 结果表明 proposal bank 中存在长尾正确动作，但“候选里有答案”不等于“无 GT 模型能选中答案”。为隔离这两个问题，新实验在任何 selector inference 之前冻结了 episode-disjoint split：

| Split | Episodes | Steps |
|---|---:|---:|
| Smoke | 12 | 23 |
| Dev | 133 | 231 |
| Locked test | 398 | 708 |

每个 blind packet 固定包含 SFT anchor、Qwen3-VL-8B、Qwen3.5-9B、LLaVA-1.5-7B 四个 source 各 K=8 的 exact actions，但移除 GT action、matcher reward、correctness、source identity 和 GT-derived diagnostics。Dev 与 locked predictions 都在任何 dev/locked label 解封前完成。

定义 student-relative utility：

$$
 u_t^{stu}=M(a_t^{selector})-M(a_t^{student}),\qquad
 U=\frac{N_{rescue}-N_{regress}}{N_{steps}}.
$$

Locked 结果：

| Selector | Selected acc | Net utility | 95% episode-cluster CI | Rescue / regress | Oracle capture |
|---|---:|---:|---:|---:|---:|
| Qwen3.5-9B | **6.78%** | **+6.36pp** | **[+4.53,+8.31]pp** | **46 / 1** | **20.93%** |
| Qwen3.5-35B-A3B | 4.94% | +4.52pp | [+2.93,+6.20]pp | 33 / 1 | 14.88% |
| Exact plurality | 3.67% | +3.25pp | [+1.87,+4.73]pp | 25 / 2 | 10.70% |
| Cross-source consensus | 5.79% | +5.37pp | [+3.61,+7.25]pp | 39 / 1 | 17.67% |

Frozen-student accuracy 只有 0.42%，packet oracle 为 30.79%。因此最佳 9B selector 只捕获了 20.93% oracle headroom，仍有大量未利用空间。

这里得到三个机制结论：

1. **Proposal diversity 是可利用的。** 所有四个 selector 的 locked cluster-bootstrap 下界都大于 0。
2. **Corrector scale 不是单调机制。** 35B 更保守，changed coverage 34.18%，低于 9B 的 60.03%，并且少选中 13 个净正确 step。
3. **Consensus 可以解释大部分增益。** Cross-source consensus 与 9B 的 paired delta 为 -0.99pp，CI [-2.98,+1.12]pp，差异不显著。

这将研究问题从“训练一个更大的 verifier”改写为：

> **如何设计 proposal distribution 与匿名跨模型 agreement，使 selector 在保持低 regression 的同时提高 oracle-headroom capture？**

### 1.9 Selector Utility 与 SFT Purity 之间的桥

Positive utility 不能直接推出训练数据可用。在 student-wrong population 上，selector 再次选错时 utility 为 0，但该 action 作为 SFT target 仍是主动错误标签。

冻结 locked artifacts 的精确离线统计为：

| GT-free 构造 | Changed rows | Correct labels | SFT purity | Wilson 95% |
|---|---:|---:|---:|---:|
| 全部 9B changes | 425 | 46 | 10.82% | [8.21%,14.14%] |
| 全部 consensus changes | 334 | 39 | 11.68% | [8.66%,15.56%] |
| 9B/consensus 同动作交集 | 114 | 13 | 11.40% | [6.79%,18.54%] |

因此三种直接构造都不应进入 SFT；intersection 在当前 split 只降低 coverage，没有提高 purity。正式训练前必须并行完成：

1. P100/P80/P60/P40、固定 25/75 的受控 purity-response LoRA curve；
2. 独立 train split 上冻结 V1 all-9B-change、V2 consensus-change、V3 same-action intersection 的 aggregate purity；
3. 只有 $LB_{95}(p_v)\ge p_{min}^{train}$ 的 GT-free variant 才可训练；
4. 另建 student-correct general-state control 测量真实 regression risk。

9B self-source 诊断也支持 agreement 机制：含自身 exact source 的选择被富集 1.36×，但 self-only purity 只有 6.99%；Qwen3.5 与其他来源共同支持时 purity 为 18.52%。

---

## 2. 当前可以声称什么，不能声称什么

### 2.1 当前已经有证据支持的贡献

1. **大规模 heterogeneous GUI failure bank 与严格 paired protocol**
   两个异构 actor、3,146 trajectories、全 train 生成、全 test 配对评估、固定 hash 和 exact shard merge。

2. **Diversity–utility decoupling 的实证反例**
   91.68% disagreement 并未产生正 downstream utility。

3. **Counterfactual Revision Utility taxonomy 与完整测量**
   Rescue/regress/preserve/unresolved，而不是只报告 revised accuracy。

4. **Source-conditioned correction asymmetry**
   同一个 corrector 对 InternVL3 显著正向、对 Qwen3-VL 显著负向。

5. **Step–trajectory composition gap**
   +4.52pp step utility 只对应 2.07% complete trajectory rescue。

6. **Prefix-consistency/sequential contamination diagnosis**
   73.74% SFT rows 位于 diagnostic dirty prefix，且 label accuracy 随 prefix errors 单调下降。

7. **Naive unfiltered training 的强负向因果结果**
   1,000 held-out episodes 上 TSR -17.80pp，95% interval [-20.20pp, -15.50pp]。

8. **Student-rescue + clean replay 的正向训练 ceiling**
   25% oracle rescue + 75% clean replay 在完整 1,000 episodes 上带来 TSR +2.80pp，区间 [+1.00pp,+4.60pp]，且 step accuracy +0.17pp。

9. **Pass@8 proposal diversity 的正向非 oracle recovery**
   Qwen3.5-9B fixed-choice selector 在 selector-fresh locked split 上达到 +6.36pp utility，CI [+4.53pp,+8.31pp]。

10. **Scale–selection decoupling 与 consensus mechanism**
    Qwen3.5-35B-A3B 没有超过 9B；零 GPU cross-source consensus 接近 9B，说明 independent proposal agreement 是核心信号。

### 2.2 目前不能声称

1. 不能声称 global trajectory correction 普遍无效。
2. 不能声称所有 corrector 都有相同问题；正式 full run 只有一个 corrector family。
3. 不能声称 confidence 导致了训练崩溃；confidence 不在训练 conversation 中。
4. 不能声称 prefix mismatch 已被证明是唯一根因；还需要 history × label factorial ablation。
5. 不能声称 Pass@8 selector 已经提高 downstream policy performance；当前正结果是 hard-step candidate recovery，尚未在独立 train split 上训练 25/75 arm。
6. 不能声称“首次”发现 weak supervision harmful；weak-to-strong、trajectory verifier 和 noisy-label literature 已有大量工作。
7. 不能声称 35B corrector 比 9B 更强；当前 paired locked 结果方向相反。
8. 不能把 GT-conditioned critical-step selector 当成 arbitrary-state online router。

---

## 3. 与相关工作的定位

已有工作已经覆盖“生成或修正 agent trajectory”本身，因此 novelty 不能写成“首次让 agent 改 trajectory”。

### 3.1 GUI trajectory generation / self-improvement

- [UI-Genie](https://arxiv.org/abs/2505.21496)：reward model、controlled corruption、hard-negative mining、reward-guided exploration。
- [AgentTrek](https://arxiv.org/abs/2412.09605)：tutorial-guided replay，并由 VLM evaluator 验证 trajectory。
- [GUI-Reflection](https://arxiv.org/abs/2506.08012)：从成功轨迹构造 reflection/error-correction data，并进行 online reflection tuning。
- [UI-Voyager](https://arxiv.org/abs/2603.24533)：从 group rollouts 的成功轨迹识别 critical fork points。
- [SGCD](https://arxiv.org/abs/2606.18890)：从真实 off-trajectory state 生成 skill-guided successful continuation。

共同点：正向方法通常依赖 successful rollout、environment feedback、reward model 或 verifier，避免把所有 revision 当成同质量数据。

### 3.2 Step verification / verifier-driven GUI learning

- [STEVE](https://arxiv.org/abs/2503.12532)：before/after screenshot step verification，利用正负 action，而非盲目 SFT 全部轨迹。
- [V-Droid](https://arxiv.org/abs/2503.15937)：pairwise progress preference verifier。
- [GAIA](https://arxiv.org/abs/2601.18197)：GUI action critic data flywheel。
- [The Art of Building Verifiers for Computer Use Agents](https://arxiv.org/abs/2604.06240)：区分 process/outcome rewards，并用 divide-and-conquer 管理完整 trajectory context。

我们的区别应写成：

> 我们研究 verifier 之前更基础的对象——**revision 相对不同 source 和不同 student 的条件效用，以及 teacher-forced global rewriting 引入的 prefix consistency gap**。

### 3.3 General agent trajectory refinement

- [STeP](https://arxiv.org/abs/2505.20023)：self-reflected trajectories + partial masking，避免学习错误步骤。
- [Agent-R](https://arxiv.org/abs/2501.11425)：MCTS 找 first error 并拼接相邻 correct path。
- [STeCa](https://arxiv.org/abs/2502.14276)：通过 step-level reward comparison 构造 calibrated trajectories。
- [AgentRefine](https://arxiv.org/abs/2501.01702)：强 LLM 根据环境反馈 refinement tuning。

这些工作进一步说明：完整轨迹改写不是 novelty；**source/student-conditioned utility、prefix consistency、以及从负结果到 conservative gating 的系统测量**才是潜在贡献。

### 3.4 Weak-to-strong supervision

- [Weak-to-Strong Generalization](https://arxiv.org/abs/2312.09390)
- [Reliability-Aware Alignment](https://arxiv.org/abs/2406.19032)
- [Trust Functions](https://arxiv.org/abs/2606.01000)
- [Selective Weak-to-Strong Generalization](https://arxiv.org/abs/2511.14166)

“学习何时信任弱 teacher”不是全新命题。可区分点是：

1. GUI sequential action space；
2. heterogeneous source actors；
3. full-trajectory revision；
4. step utility 与 complete-trajectory utility 的组合问题；
5. revised history 与 teacher-forced screenshot 的 causal consistency；
6. source-relative 与 student-relative utility 的联合 gating。

---

## 4. 建议的核心方法：Source- and Student-Conditioned Revision Utility Gating

工作名可暂定为 **S²-RUG**（Source- and Student-Conditioned Revision Utility Gating）。

### 4.1 两种 utility

Source-relative utility：

$$
 u_t^{src}=M(a_t^{rev})-M(a_t^{actor}).
$$

Student-relative utility：

$$
 u_t^{stu}=M(a_t^{rev})-M(a_t^{student}).
$$

只有 $u_t^{src}>0$ 并不够；如果 starting student 本来已经正确，则 weak revision 仍可能是有害监督。

### 4.2 Prefix consistency

定义：

$$
 c_t=\prod_{j<t} M(a_j^{rev}).
$$

$c_t=1$ 表示 diagnostic clean prefix。真正更严格的版本应使用环境 transition verifier：

$$
 \hat s_{t+1}=T(s_t,a_t^{rev}),\qquad \hat s_{t+1}\sim s_{t+1}^{data}.
$$

只有 revised action 能因果到达后续 state，完整 trajectory 才是 transition-consistent。

### 4.3 Gate 输入与输出

输入 packet：

- Goal、current screenshot、history
- Actor action/reasoning/source identity
- Corrector revision/rationale
- Starting student candidate
- Actor–revision–student disagreement
- Corrector self-consistency
- Prefix consistency / transition evidence
- Optional verifier score

输出：

```text
use_revision | keep_student | keep_actor | reject/replan
```

或输出连续 trust/utility score：

$$
 g_\phi(x_t,a_t^{actor},a_t^{rev},a_t^{student},c_t)\rightarrow \hat u_t.
$$

### 4.4 Conservative learning objective

不要再次把所有 revision 做统一 SFT。可以构造 preference：

- Rescue：revision ≻ actor
- Regress：actor ≻ revision
- Preserve-correct：不强制改写，作为 consistency/replay
- Unresolved：mask/reject/replan

训练可采用：

1. Utility-gated SFT
2. Pairwise DPO/IPO
3. Partial masking
4. Clean-data replay + KL regularization
5. Source-conditioned weights

这与已有 MA-OPD / Het-DPO 工作可以自然连接：heterogeneous agents 负责产生 candidate pair，CRU/verifier 负责决定 preference direction。

---

## 5. 最小可发表实验矩阵

### Phase A：先把根因拆开

所有实验应使用同一批 states、相同 optimizer budget 和相同 1,000-episode held-out grid。

| ID | Training target | History | 目的 |
|---|---|---|---|
| A0 | No update | — | Frozen baseline |
| A1 | GT action | GT history | Positive control |
| A2 | Random/marginal-matched action | GT history | Negative control |
| A3 | Actor action | Actor history | Actor imitation control |
| A4 | Revision action | Revision history | 当前 full experiment |
| A5 | Revision action | GT history | 隔离 label noise，移除 prefix mismatch |
| A6 | GT action | Revision history | 隔离 prefix/context mismatch |
| A7 | Revision action | Clean prefix only | 检验 sequential contamination |
| A8 | Revision action | Dirty prefix only | Harm control |

最关键的是 A4/A5/A6 的 $2\times2$ 分解，它能回答：崩溃主要来自 wrong target，还是 history–screen inconsistency。

### Phase B：source/student utility

1. 在完全相同的 14,456 states 上运行 starting student。
2. 计算 student-relative rescue/regress/preserve/unresolved。
3. 分别训练：
   - InternVL3 revisions only
   - Qwen3-VL revisions only
   - Source-gated mixture
   - Student-agreement subset
   - Oracle positive-CRU ceiling

这一步会把“better than actor”与“good enough for student”严格分开。

### Phase C：训练 revision utility gate

Selector baselines：

1. Corrector confidence
2. Source identity
3. Actor–revision disagreement
4. Corrector self-consistency
5. Student–revision agreement
6. Prefix consistency
7. LLM/verifier over complete candidate packet

Gate metrics 不应只报告 accuracy：

- Rescue AP / recall
- Regression detection AP
- Accepted-set label accuracy
- Accepted coverage
- Net CRU at fixed coverage
- Trajectory-rescue precision
- Cost-adjusted utility

Phase C 的执行结果已经表明：confidence、metadata、balanced/natural verifier、三个 binary ranker 和简单 transition gate 均未稳定泛化；但冻结的 Pass@8 fixed-choice protocol 首次跨过非 oracle hard-step gate。当前最强 selector 是 Qwen3.5-9B，而不是 Qwen3.5-35B-A3B；cross-source consensus 是必须保留的强零 GPU control。

### Phase D：downstream confirmation

建议顺序：

1. 在独立 train split 上生成与 frozen protocol 同构的 Pass@8 packets。
2. 冻结 Qwen3.5-9B 与 cross-source consensus，分别构建 25% selected revision + 75% clean replay arm。
3. 先做 LoRA 等预算筛选，再在全 held-out policy grid 上 paired bootstrap。
4. 只有 LoRA gate 通过后才做 fullparam；再换第二个 benchmark 检验 generalization。

---

## 6. 预注册 Gate

### Selector gate

原 revision-bank selector 至少满足：

- Accepted-set diagnostic label accuracy 显著高于未筛选的 26.04%。
- Regression rate 明显低于当前 26.58% conditional rate。
- 在固定 coverage 下 net CRU 为正，trajectory-cluster CI 不跨 0。
- Confidence-only baseline 被显著超过。

Pass@8 fixed-choice confirmatory gate 预注册为：

1. Locked $U=(N_{rescue}-N_{regress})/N_{steps}>0$；
2. Rescue count 大于 regress count；
3. Episode-cluster bootstrap 95% 下界大于 0；
4. Stronger-model claim 还要求 strong-minus-current 的 paired CI 下界大于 0。

实际结果：四个 selector 都通过前三项；35B 不通过第四项。因此可以声称 candidate recovery 成立，不能声称 larger corrector 更优。

### LoRA downstream gate

- Held-out step accuracy 不低于 baseline -1pp。
- TSR delta 非负，或 Sequential Progress 显著改善。
- 不出现 task right→wrong 远大于 wrong→right。

### Fullparam gate

只有 LoRA gate 通过后才运行：

- Fullparam step accuracy 不低于 baseline -1pp。
- TSR paired CI 不显示 harm。
- Clean benchmark/general capability 不发生明显遗忘。

---

## 7. 论文 contribution 的推荐写法

结合完整 causal matrix、oracle replay 和 Pass@8 locked 结果，最稳妥的是 empirical/diagnostic + proposal-selection method paper：

1. We construct a large-scale heterogeneous GUI trajectory revision stress test with isolated paired evaluation.
2. We formalize Counterfactual Revision Utility and decompose revisions into rescue, regression, preservation, and unresolved outcomes.
3. We discover strong source-conditioned asymmetry: the same global corrector improves a weak actor but degrades a stronger actor.
4. We identify a step-to-trajectory composition gap and a teacher-forced prefix-consistency gap in offline global rewriting.
5. We show that naive unfiltered full-parameter training causes statistically significant negative transfer.
6. We establish a positive oracle training ceiling with student-rescue selection and clean replay.
7. We show that frozen Pass@8 fixed-choice selection yields positive non-oracle hard-step utility, while scaling the selector from 9B to 35B does not improve recovery.
8. We identify cross-source proposal consensus as a competitive zero-training mechanism.

完整 method claim 仍需新增一项：在独立 train split 上用冻结 selector 构造 25/75 arm，并在 full held-out policy grid 上复现正 TSR。

---

## 8. 最重要的 narrative

最强的故事不是：

> “我们让两个 agent 生成错误，第三个 agent 修改，但训练失败了。”

更新后的 narrative 是：

> “Heterogeneous agents provide abundant proposals, but diversity becomes useful only through student-relative selection. Unfiltered revisions cause severe negative transfer, while oracle rescue plus clean replay establishes a positive training ceiling. A frozen Pass@8 selector recovers significant hard-step utility without ground truth, yet a 35B corrector underperforms 9B and simple cross-source consensus is competitive. The key mechanism is therefore proposal–selection alignment and independent agreement, not corrector scale or self-reported confidence.”

这个 narrative 把当前负结果变成：

- 一个反例；
- 一个新的测量对象；
- 一个机制发现；
- 一个已通过 locked hard-step gate 的正向方法入口；
- 一个仍需 full-policy downstream confirmation 的清晰下一步。

---

## 9. 可复现证据

- [完整实验总结](EXPERIMENT_SUMMARY_ZH.md)
- [Counterfactual Revision Utility 报告](full_v1/research_analysis/revision_utility_report.md)
- [机器可读 CRU 统计](full_v1/research_analysis/revision_utility_summary.json)
- [CRU 分析脚本](../../scripts/analyze_multiagent_revision_utility.py)
- [Actor summary](full_v1/actor_summary.json)
- [Recovered corrections](full_v1/global_corrections_recovered.jsonl)
- [Noisy SFT data](full_v1/noisy_global_corrections_train.jsonl)
- [Full held-out evaluation](full_v1/training_eval/report.md)
- [Pass@8 中文总结](../pass8_selector_study/EXPERIMENT_SUMMARY_ZH.md)
- [Pass@8 locked report](../pass8_selector_study/eval/locked_test/report.md)
- [Pass@8 完整研究设计](../../docs/pass8_strong_corrector_study.md)
- [Pass@8 Selector → Training Bridge 预登记](../../docs/pass8_selector_to_training_bridge_zh.md)
- [Pass@8 Bridge 离线诊断](../pass8_selector_study/bridge_diagnostics/locked_v1/report.md)
