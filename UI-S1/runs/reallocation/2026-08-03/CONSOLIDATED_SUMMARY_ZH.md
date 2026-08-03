# Difficulty-Conditioned Reallocation 证据总结

日期：2026-08-03

## 一、实验目的

本轮希望将N5的正面发现推进为一个预算重分配方法：

- 简单样本减少forward；
- 高分歧、困难样本增加forward；
- 在相同或更低平均预算下，提高最终B3准确率。

关键前提是：高分歧样本增加候选后，不仅`pass@N`上升，最终聚合准确率也必须上升。

因此R1先检验“额外候选能否被聚合器兑现”。只有R1通过，才允许执行R2预算重分配和R3候选云条件化新推理。

## 二、最终执行状态

| 项目 | 状态 | 说明 |
|---|---|---|
| R1 分层精度闸门 | FAIL / R-K1 | pass显著上涨，但B3未上涨 |
| R2 难易预算重分配 | CANCELLED_R_K1 | 按预注册取消 |
| R3 条件化第二轮提议 | CANCELLED_R_K1 | 未启动新推理 |
| R4 选择性精度 | PASS | 本轮主要正面结果 |
| R5 72B污染假设 | 原假设FAIL | 发现B3模型来源偏置 |
| 新模型forward | 0 | 全部结果来自已有trace |

## 三、R1：高分歧样本的额外预算能否兑现

### 3.1 分层方式

使用Uniform Mixed N12候选池的SafeGround官方实现迁移分数：

- `patch_size=28`
- `activation_threshold=0.0`
- 分数越高表示候选云分歧越大
- 按`(uncertainty,row_id)`排序
- 使用NumPy `array_split`划分五个等频层

R1只在最高分歧20%的316个样本上做闸门判定。

### 3.2 最高分歧层曲线

| 指标 | N4 | N8 | N12 | N16 | N24 |
|---|---:|---:|---:|---:|---:|
| B3 | 19.62% | 16.77% | 20.25% | 19.62% | 18.99% |
| M1 | 23.10% | 19.62% | 20.25% | 19.30% | 21.84% |
| pass@N | 38.29% | 45.89% | 51.27% | 53.48% | 57.28% |

N24减N4的paired结果：

| 指标 | 差值 | 99% CI | 判定 |
|---|---:|---:|---|
| B3 | -0.63 pp | [-6.27,+5.76] | 不上涨 |
| M1 | -1.27 pp | [-7.89,+5.67] | 不上涨 |
| pass@N | +18.99 pp | [+14.11,+24.91] | 显著上涨 |

冻结MDE为`0.70 pp`。B3差值既未超过MDE，99% CI下界也不为正，因此R1失败。

候选headroom兑现率定义为：

```text
B3增量 / pass@N增量
```

本实验为：

```text
-0.63 / 18.99 = -0.033
```

也就是说，额外候选不仅没有兑现为最终精度，B3点估计还略有下降。

### 3.3 R1结论

> 在最高分歧样本上，将预算从N4增加到N24可以显著提高正确候选出现概率，但现有B3和M1都无法将这部分headroom转化为最终准确率。

这是第五条直接的碰撞墙证据，与以下已有结果一致：

1. CALA-S提高pass@12但降低B3；
2. NOA-static没有改善Uniform Mixed；
3. NOA-stop节省预算但精度下降；
4. 多种selector/reranker独立失败；
5. 当前R1中pass上涨约19 pp而B3不涨。

## 四、R2与R3为何取消

### 4.1 R2预算重分配

R2的核心假设是：高分歧行多给预算后，最终精度能够上涨。

R1已经直接否定该前提。因此按照预注册的`R-K1`：

```text
R2 = CANCELLED_R_K1
```

没有运行：

- S1两档重分配；
- S2三档重分配；
- S3连续预算分配；
- S4随机预算多重集对照。

这不是缺失结果，而是预注册闸门的正式执行结果。不能在R1失败后继续搜索其他阈值或预算映射。

### 4.2 R3条件化第二轮提议

R3同样依赖R1通过。由于R1失败：

```text
R3 = CANCELLED_R_K1
```

没有启动：

- C-cond候选云条件裁剪；
- C-rand随机新裁剪；
- 316行×6次新模型forward。

本轮新增模型推理成本为：

```text
0 forwards
```

## 五、R4：选择性精度正面结果

R4不依赖预算分配算法。它回答另一个部署问题：

> 如果系统可以放弃或转交最不确定的样本，剩余样本的准确率能提高多少？

### 5.1 Uniform Mixed N12

| 保留coverage | 保留行数 | B3 | 相对全覆盖提升 | 随机拒绝均值 | 随机99% CI |
|---|---:|---:|---:|---:|---:|
| 100% | 1,581 | 63.69% | 0.00 pp | 63.69% | [63.69,63.69] |
| 90% | 1,422 | 69.06% | +5.36 pp | 63.69% | [62.73,64.77] |
| 80% | 1,264 | 74.60% | +10.91 pp | 63.69% | [62.10,65.35] |
| 70% | 1,106 | 79.02% | +15.33 pp | 63.69% | [61.66,65.73] |

每个选择性精度点都高于随机拒绝99%区间上界。

在80% coverage下：

- SafeGround选择后B3：`74.60%`
- 随机拒绝均值：`63.69%`
- 超过随机：`+10.91 pp`
- 被拒绝的317行中包含253个B3失败和64个B3成功

这说明SafeGround确实将错误样本集中在高不确定区域，而不是简单依靠降低coverage机械提高准确率。

### 5.2 V-only N12

| 保留coverage | B3 | 相对全覆盖提升 | 随机拒绝均值 | 随机99% CI |
|---|---:|---:|---:|---:|
| 100% | 60.09% | 0.00 pp | 60.09% | [60.09,60.09] |
| 90% | 63.92% | +3.84 pp | 60.08% | [59.07,61.18] |
| 80% | 67.48% | +7.40 pp | 60.09% | [58.47,61.71] |
| 70% | 70.71% | +10.62 pp | 60.08% | [58.05,62.21] |

V-only也具有有效的选择性排序，但整体弱于Mixed。

80% coverage时：

- Mixed：`74.60%`
- V-only：`67.48%`
- Mixed领先：`+7.12 pp`

### 5.3 R4贡献

> 跨谱系分配不仅提高全覆盖grounding准确率，还提高系统识别“自己何时可能出错”的能力。

对应的SafeGround正确性AUROC：

- V-only N12：`0.744`
- Uniform Mixed N12：`0.830`

这支持一个实际部署策略：

- 低不确定样本自动执行；
- 高不确定样本交给人工；
- 或转交给更昂贵的搜索、交互或闭源模型。

需要注意：R4是选择性预测结果，不证明SafeGround适合决定如何增加现有固定视角forward。

## 六、R5：72B聚类崩溃诊断

### 6.1 原假设

原假设认为72B强模型的错误候选更紧，因此密度聚类更容易被错误簇劫持。

该假设没有通过。

| 指标 | 7B Uniform N8 | 72B Uniform N8 |
|---|---:|---:|
| B3 | 61.99% | 41.24% |
| 失败候选平均归一化距离 | 0.1137 | 0.1539 |
| 最大失败簇平均占比 | 0.410 | 0.430 |

72B减7B的失败候选平均距离：

- 差值：`+0.0616`
- 99% CI：`[+0.0368,+0.0848]`

72B失败候选显著更分散，而不是更紧。

因此：

```text
R5 tight-error pollution hypothesis = FAIL
```

### 6.2 新发现：B3模型来源偏置

虽然紧簇假设失败，但B3赢家表现出极强的模型来源偏置。

在72B的929个B3错误样本中，最终被B3选中的候选来源：

| 模型 | 被选中次数 |
|---|---:|
| GTA1-72B | 872 |
| UI-Venus-Ground-72B | 52 |
| Qwen3.5-122B-A10B | 5 |

错误winner group中的模型成员数：

| 模型 | 成员数 |
|---|---:|
| GTA1-72B | 1,374 |
| UI-Venus-Ground-72B | 1,000 |
| Qwen3.5-122B-A10B | 370 |

模型组成均匀性检验：

- $\chi^2=562.97$
- $p=5.66\times10^{-123}$

这表明72B低B3更适合解释为：

> B3的coverage tie-break和候选来源语义使赢家系统性偏向GTA1候选，导致强Qwen3.5裸分无法在混合池中被充分利用。

该诊断与N3一致：

- 三模型裸分均复现anchor；
- 没有全局坐标尺度bug；
- 低B3是聚合器与异构候选池的失配。

## 七、对论文主张的影响

### 可以新增的正面主张

1. SafeGround支持有效的GUI grounding选择性预测。
2. 在Uniform Mixed N12上，保留80%最低不确定样本可将B3从`63.69%`提高到`74.60%`。
3. 该结果显著高于相同coverage下的随机拒绝。
4. 跨谱系分配同时改善准确率与错误风险排序。

### 仍然保留的已有主张

1. 7B等12-forward预算下，Mixed M1提升`+3.42 pp`。
2. 不变B3提升`+3.54 pp`。
3. V-only与Mixed预算斜率符号翻转。
4. 加入较弱模型仍可提升系统。
5. CALA在7B/72B N8均有显著等预算增益。

### 不可以主张

1. 不能声称难行多给现有固定视角预算会提高最终精度。
2. 不能声称R2预算重分配成功；R2未执行。
3. 不能声称R3条件化裁剪有效；R3未执行。
4. 不能省略S4/C-rand然后假设重分配有效。
5. 不能声称72B错误候选更紧。
6. 不能将选择性精度结果描述成全覆盖准确率提升。

## 八、推荐论文表述

英文：

> Candidate disagreement is effective for selective prediction but not for allocating additional fixed-view inference: on the most uncertain examples, pass@N rises by 18.99 points while B3 does not improve, whereas abstaining on the most uncertain 20% raises retained Mixed-N12 B3 from 63.69% to 74.60%.

中文：

> 候选分歧适合用于选择性预测，但不适合直接决定如何追加现有固定视角预算：在最高分歧样本上，pass@N提高18.99个百分点而B3没有改善；相反，放弃最不确定的20%样本，可将Mixed N12剩余样本的B3从63.69%提高到74.60%。

## 九、交付物

- `SPEC.md`
- `configs/strata.yaml`
- `configs/r2_policies.yaml`
- `r1_stratified_accuracy.json`
- `r2_reallocation.json`
- `r3_conditional_proposal.json`
- `r4_risk_coverage.json`
- `r5_72b_pollution.json`
- `fig_stratified_curves.pdf`
- `fig_risk_coverage.pdf`
- `MAIN_TABLE.md`
- `REPORT.md`
- `STATUS.json`

完整状态：

```text
R1 = FAIL_R_K1
R2 = CANCELLED_R_K1
R3 = CANCELLED_R_K1_NO_NEW_INFERENCE
R4 = PASS_SELECTIVE_ACCURACY
R5 = FAIL_TIGHTNESS_HYPOTHESIS_WITH_B3_SOURCE_BIAS
new_model_forwards = 0
```