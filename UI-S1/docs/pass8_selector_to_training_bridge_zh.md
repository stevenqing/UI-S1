# Pass@8 Selector → Training Bridge：下一步预登记

## 0. 当前决策

这是整条研究线第一个可信的 GT-free selector 正结果，但它是 **candidate-recovery gate**，不是 **SFT-label-quality gate**。

当前立即决策：

- 冻结现有 selector、prompt、candidate schema 与 locked 结果，不再针对该 split 调参。
- 不允许把 current、consensus 或它们的交集直接灌入 25/75 SFT。
- 先并行获得两个桥接数字：
  1. 固定 25% revision + 75% clean replay 时，LoRA 能容忍的最低标签纯度；
  2. 冻结 selector 在独立 train split 上产生的真实 changed-set 纯度及其置信下界。
- 只有第二个数字的保守下界超过第一个数字，训练臂才放行。

这把下一次训练从“看到正 utility 后试一下”改成一个可证伪的预测检验。

---

## 1. 结果的正确读法

### 1.1 被否定的假设：更强 selector 更好

预登记的 model-scaling 假设没有通过：

- Qwen3.5-9B：locked utility +6.36pp，CI [+4.53,+8.31]pp；
- Qwen3.5-35B-A3B：+4.52pp，CI [+2.93,+6.20]pp；
- 35B − 9B：-1.84pp，CI [-3.69,+0.00]pp；
- 35B changed coverage 34.18%，9B 为 60.03%。

因此不能把结果解释为“更大的 corrector 解决了 selection”。35B 更保守，但并没有更准确。

### 1.2 被支持的假设：跨源独立支持是可用信号

两个零训练对照给出直接机制证据：

- Exact plurality：+3.25pp；
- Cross-source consensus：+5.37pp；
- Qwen3.5-9B：+6.36pp；
- Consensus − 9B：-0.99pp，CI [-2.98,+1.12]pp，差异不显著。

不区分来源的重复计票明显弱于要求不同来源一致的 consensus。异构性在这里承担两个角色：

1. **Union 提供 coverage**：不同策略覆盖不同的长尾正确动作；
2. **Agreement 提供 selection signal**：独立策略交集提高候选可信度。

因此论文中的旧表述“选择和生成一样难”应软化为：

> 单输出流中的 correctness verification 仍然困难；但多候选 proposal 加独立支持统计，可以无 GT 地变现约 21% 的 packet-oracle headroom。

剩余约 79% headroom 没有变现，hardness 主张仍然成立。

### 1.3 三层证据结构

现在可以形成完整闭环：

1. **存在性**：oracle student-rescue + 75% clean replay 在完整 held-out grid 上带来 TSR +2.80pp；
2. **部分可达性**：GT-free Qwen3.5-9B selector 捕获 20.93% oracle headroom；
3. **剩余硬度**：packet oracle 30.79%，实际 selected accuracy 只有 6.78%。

---

## 2. 为什么 selector gate 通过仍不能直接训练

### 2.1 Utility 与 SFT purity 是不同代数

Selector utility 定义为：

$$
U=\frac{N_{rescue}-N_{regress}}{N_{steps}}.
$$

在当前 locked hard-step population 中，student accuracy 只有 0.42%。当 student 本来错误、selector 又选错时，该行 utility 为 0；但如果把它作为 SFT target，它是主动错误标签，不是“中性样本”。

因此：

> Positive rescue-minus-regression utility 不等于 selected labels 足够纯，可以直接用于 SFT。

### 2.2 已完成的精确 changed-set 诊断

离线诊断使用冻结 packet、sealed provenance 和已经完成的 selector 输出；matcher 只做事后测量，不参与选择。

| GT-free 构造 | Changed rows | Correct labels | SFT purity | Wilson 95% | Rescue / regress |
|---|---:|---:|---:|---:|---:|
| 全部 9B changes | 425 | 46 | **10.82%** | [8.21%,14.14%] | 46 / 1 |
| 全部 consensus changes | 334 | 39 | **11.68%** | [8.66%,15.56%] | 39 / 1 |
| 9B 与 consensus 同动作交集 | 114 | 13 | **11.40%** | [6.79%,18.54%] | 13 / 0 |

这比近似的“46/425≈11%”更精确，并给出两个新结论：

1. 三个直接构造的纯度都只有约 11%；
2. 9B∩consensus 在当前 locked population 上没有产生预期的纯度递增，只显著降低 coverage。

Oracle 25/75 配方的 revision quarter 是 100% pure rescue；未筛 revision 的整体 accuracy 为 26.04%，并导致 fullparam 崩盘。二者与当前 hard-step changed-set 并非完全同分布，不能直接作数值因果比较，但 10–12% 纯度足以阻断“直接灌入 SFT”。

### 2.3 当前训练授权的准确含义

Pass@8 gate 只授权：

- 构建 train-split candidate packets；
- 测量 GT-free 构造的 aggregate purity；
- 运行 purity-response LoRA screen；
- 在满足桥接 gate 后准备 25/75 arm。

它不授权直接训练 current locked selector 输出，更不授权使用任何 dev/locked row 训练。

---

## 3. Qwen3.5-9B 自选诊断

Qwen3.5-9B 同时是四个 proposal source 之一和 selector。虽然 packet 隐藏 source identity，模型仍可能识别自身输出风格。冻结 provenance 给出以下离线结果：

- 425 个 changed selections 中，197 个 exact selected actions 含 Qwen3.5 source：**46.35%**；
- 全 packet 的 9,350 个非 baseline exact candidates 中，含 Qwen3.5 source 的比例为 34.09%；
- 选择 enrichment：**1.36×**。

按 exact source set 分层：

| Selected source stratum | Rows | Correct | Purity |
|---|---:|---:|---:|
| Qwen3.5 only | 143 | 10 | **6.99%** |
| Qwen3.5 + other source | 54 | 10 | **18.52%** |
| No Qwen3.5 | 228 | 26 | **11.40%** |

解释：

- 9B 确实过度选择了含自身来源的 action；
- 但 self-only 子集纯度最低，不能把自选偏置解释为有效机制；
- Qwen3.5 与其他独立来源共同支持时纯度最高，再次支持“独立 agreement”而非“自身文风识别”。

这仍是 provenance association，不是风格识别的因果证明。论文中应表述为 self-source enrichment diagnostic，而不是 self-recognition mechanism。

---

## 4. Bridge A：受控纯度响应曲线

### 4.1 目标

估计固定 25% revision + 75% clean replay 时，当前 LoRA 训练配置能够容忍的最低 revision-label purity：

$$
p_{min}^{train}=\min\{p:\text{LoRA arm at purity }p\text{ passes all guardrails}\}.
$$

### 4.2 四个预登记 LoRA 臂

| Arm | Revision-quarter purity | Revision / clean | 目的 |
|---|---:|---:|---|
| P100 | 100% | 25 / 75 | 已知 oracle anchor |
| P80 | 80% | 25 / 75 | 轻污染 |
| P60 | 60% | 25 / 75 | 中污染 |
| P40 | 40% | 25 / 75 | 强污染下界 |

固定项：

- 四臂 revision row 数相同；clean replay 严格为 revision 的 3 倍；
- 相同 episode pool、LoRA rank、学习率、epoch/update budget、seed 与 evaluator；
- 不重复 revision rows；共同预算取所有臂可用 unique rows 的最小值；
- 正例来自 oracle student-rescue bank；
- 污染例来自 student-wrong / candidate-wrong 的 selected-like hard negatives，使“utility 为 0、SFT label 为错”的污染结构贴近当前 selector；
- 按 action type、step-position bucket、prefix cleanliness 和 source family 做边际匹配，避免 purity 与 difficulty 混淆。

### 4.3 LoRA purity gate

某个 purity arm 通过，必须同时满足：

1. Held-out step accuracy 不低于 frozen baseline -1pp；
2. TSR delta 非负；
3. task right→wrong 不显著高于 wrong→right；
4. 训练无 collapse/format failure；
5. 结果随 purity 基本单调；若明显非单调，不插值，先重复 seed。

$p_{min}^{train}$ 定义为最低通过 purity。禁止在 40% 以下外推：如果 train-split variant 的纯度低于 40%，必须先提高选择严格度，或追加 P20/P10 calibration arms；不能直接声称曲线支持训练。

---

## 5. Bridge B：Train-split GT-free 构造纯度

### 5.1 冻结协议

在查看任何 train matcher label 之前冻结：

- train episode IDs 与 screenshot namespace；
- GT-free target-selection rule；
- 四个 proposal source、K=8、temperature、action normalization 与 candidate order；
- Qwen3.5-9B prompt/checkpoint；
- cross-source consensus rule；
- 三个主构造及共同预算；
- matcher 只在所有 selector 输出完成后做 aggregate diagnosis。

Train candidate generation 不得使用当前 selector dev/locked rows。

### 5.2 三个预登记构造

| Variant | GT-free 定义 | 预期 |
|---|---|---|
| V1 all-9B-change | 9B 选择与 student action 不同 | 最高 coverage，最低 purity |
| V2 consensus-change | cross-source consensus 与 student action 不同 | 中 coverage，预期更高 purity |
| V3 9B∩consensus | 两者都 change 且选择同一 normalized action | 最低 coverage，预期最高 purity |

当前 locked 诊断没有观察到 V3 purity 提升，因此这只是预登记假设，不能当作既定事实。若 train split 同样不单调，必须如实报告并停止基于 intersection 的训练构造。

每个 variant 报告：

- unique changed rows 与 episode coverage；
- exact selected-label purity；
- Wilson 95% interval；
- rescue / regress / unresolved；
- action type、source support、step position 与 prefix strata；
- 与 P100/P80/P60/P40 的 distribution distance。

### 5.3 训练 eligibility

Variant $v$ 只有满足以下条件才可进入 LoRA 25/75：

$$
LB_{95}(p_v)\ge p_{min}^{train}.
$$

同时要求：

- unique revision rows 达到预登记共同预算；
- 不使用 matcher 做逐行筛选；matcher 只决定整个 GT-free variant 是否放行；
- 多个 variant 通过时，先选 purity 下界最高者；若下界近似，再选 coverage 更高者；
- 所有 arm 使用相同 revision 数与 clean replay 数，禁止通过重复低 coverage rows 补预算。

以当前 locked 数字看，三个 variant 的 purity 上界都低于 19%，显著低于 P40。除非 train distribution 明显不同或 purity curve 证明极低 purity 可安全训练，否则 direct SFT 应 fail closed。

---

## 6. Bridge C：Student-correct 回归控制

当前 locked population 只有 3 / 708 个 student-correct rows，46:1 不能作为 arbitrary-state regression safety 证据。真正起作用的是 hard-step utility CI，而不是 regress count 本身。

需要新增一个独立 general-state control：

1. 从未用于 selector tuning 的 episode pool 做 uniform step sampling；
2. 抽样时不使用 GT 或 student correctness；
3. 先完成 candidate generation 与 blind selection，再用 matcher 分层；
4. 确保事后至少获得 300 个 student-correct rows，否则扩大样本；
5. 报告 conditional regress rate、population utility、changed coverage 与 cluster CI。

安全边界应与最终 policy guardrail 对齐：selector-induced regression 对总体 step accuracy 的保守负贡献不得超过 1pp。未完成此控制前，不能声称 selector 可用于 arbitrary-state online routing。

---

## 7. Bridge D：GT-free 难点检测器缺口

Selector inference 是 GT-free，但当前 962 个 critical steps 的圈定使用了历史 GT diagnostics。换到无 GT 新域时，还缺少 target detector。

下一阶段应把 target detection 与 candidate selection 分开：

- Detector 只能使用 frozen-student sample entropy、parse failure、modal action share、跨源 disagreement、support margin 等 GT-free 特征；
- 在 dev 上冻结 detector threshold；
- 在 locked general-state sample 上报告 hard-step recall、student-correct false-route rate 与最终 routed utility；
- 不得把“selector GT-free”写成“端到端 pipeline GT-free”，直到 detector 也通过。

这是 discussion 中的明确限制，也是跨 benchmark generalization 的必要工作包。

---

## 8. 执行顺序与停止规则

### Phase 0：已完成

- Frozen Pass@8 selector protocol；
- Qwen3.5-9B / 35B paired locked evaluation；
- Exact plurality / cross-source consensus controls；
- Changed-set purity 与 Qwen3.5 self-selection 离线诊断。

### Phase 1：先冻结桥接协议

产物：

- purity-curve manifest；
- train candidate manifest；
- general-state control manifest；
- LoRA configs、共同数据预算、seed、gate 和 artifact hashes。

冻结后禁止根据中间 loss 或单臂结果修改其他 arm。

### Phase 2：并行取得两个桥接数字

Lane A：P100/P80/P60/P40 四个等预算 LoRA arms，得到 $p_{min}^{train}$。

Lane B：生成 train-split packets，运行冻结 9B 与 consensus，得到 V1/V2/V3 purity 与置信下界。

Lane C：完成 student-correct general-state control 和 self-source/provenance strata。

### Phase 3：比较后放行或停止

- 若没有 purity arm 通过：停止 selector-SFT，转向 partial masking、pairwise preference 或环境验证；
- 若所有 train variants 的纯度下界低于 $p_{min}^{train}$：停止 25/75 SFT，先提高 agreement/abstention；
- 只有至少一个 variant 通过 purity eligibility 和 student-correct safety control，才构建正式 25/75 LoRA arm。

### Phase 4：Full-policy confirmation

- 使用新的 policy holdout；若没有 benchmark-fresh episodes，必须标注为 protocol-fresh 而非 benchmark-fresh；
- 先 LoRA，后 fullparam；
- 报告 step accuracy、TSR、wrong→right/right→wrong、paired episode bootstrap 和 general capability；
- 只有 full-policy gate 通过，才能把 hard-step candidate recovery 升级为 downstream method claim。

### 当前算力状态

截至 2026-07-12，外部 PID 2190159 正在 GPU 2–7 上运行训练，GPU 4–7 每卡约占用 27 GiB。所有下一阶段 GPU job 仍只能使用物理 GPU 4–7，并且在该外部任务退出、重新审计显存之前不得启动。

---

## 9. 论文叙事更新

旧结尾是“有用 revision 存在，但钥匙不存在”。新结尾应改为：

> **钥匙来自多样性本身：union 提供长尾覆盖，independent agreement 提供选择信号。单流 verifier、ranker 和 scale-up 都不能稳定找到正确性，但多策略交集可以无 GT 地变现约 21% oracle headroom。这个 selector 仍不足以直接产生干净 SFT labels，因此最后一道科学问题是测量并跨越 purity-to-training threshold。**

最强的三层结构是：

1. Oracle 25/75 证明有用监督存在；
2. GT-free consensus/9B 证明部分可达；
3. 约 79% 未变现 headroom 与约 11% changed-set purity定义剩余硬度。

这比纯负结果更强，也比“35B corrector 更强”更准确。

---

## 10. 预登记产物路径

建议下一阶段固定为：

- `outputs/pass8_training_bridge/frozen_v1/manifest.json`
- `outputs/pass8_training_bridge/purity_curve/`
- `outputs/pass8_training_bridge/train_selector_diagnostics/`
- `outputs/pass8_training_bridge/student_correct_control/`
- `configs/pass8_purity_curve_p100.yaml`
- `configs/pass8_purity_curve_p80.yaml`
- `configs/pass8_purity_curve_p60.yaml`
- `configs/pass8_purity_curve_p40.yaml`
- `scripts/build_pass8_purity_curve.py`
- `scripts/build_pass8_train_candidates.py`
- `scripts/evaluate_pass8_training_bridge.py`

在两个桥接数字出现前，不创建正式 selected-replay training dataset，不启动正式 25/75 policy training。
