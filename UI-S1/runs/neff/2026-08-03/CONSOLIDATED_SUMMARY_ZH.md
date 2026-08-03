# Effective-Sample-Size 证据总结

日期：2026-08-03

## 一、最终结论

强形式的 Effective-Sample-Size Law **未成立**。

我们原本希望使用

```text
N_eff(K, rho) = K / (1 + (K - 1) * rho)
```

将不同候选池的准确率统一解释为有效样本量的函数。但三种预注册的相关性口径均未达到collapse判据：

| rho口径 | 单因子残差SD | R² | collapse判定 |
|---|---:|---:|---|
| failure kappa | 7.30 pp | 0.324 | FAIL |
| rho_geom | 8.02 pp | 0.184 | FAIL |
| rho_cond | 8.87 pp | 0.002 | FAIL |

预注册要求残差SD不超过`1.40 pp`，并且严格优于直接使用原始forward数$K$的拟合。实际结果远未达到这一标准。

加入proposal质量后的二因子模型有所改善：

- 最佳口径：`rho_cond + proposal quality`
- 调整后$R^2=0.616$
- 残差SD：`5.41 pp`

因此最终框架应降级为：

> 错误相关性与proposal质量共同影响测试时扩展收益，但它们目前只能提供定性两因子解释，不能构成统一、可预测的准确率定律。

## 二、为什么单一N_eff不足

逐池测量的相关性方向基本符合预期：

| 7B N12池 | failure kappa | N_eff | B3 |
|---|---:|---:|---:|
| V-only | 0.689 | 1.40 | 60.09% |
| Uniform Mixed | 0.594 | 1.59 | 63.69% |
| CALA-S | 0.556 | 1.69 | 62.18% |

CALA-S进一步降低了相关性，并将$N_\mathrm{eff}$从`1.59`提高到`1.69`，但B3反而从`63.69%`下降到`62.18%`。

这给出一个直接反例：

> 降低候选相关性、提高N_eff或提高oracle coverage，并不保证最终聚合器能够选中正确候选。

原因是最终准确率还取决于：

- 候选本身的质量；
- 错误候选是否形成更大的错误簇；
- 聚合器能否兑现新增的正确候选；
- 新候选是否污染已有的正确mode。

## 三、仍然成立的主结果

### 3.1 7B等算力跨谱系分配

最稳健的主结果仍然是相同12-forward预算下的跨谱系分配：

| 配置 | B3 | M1 | pass@12 |
|---|---:|---:|---:|
| GTA1-only N12 | 60.09% | 60.40% | 72.80% |
| Uniform Mixed N12 | 63.63% / 63.69% | 63.82% | 79.19% |

M1的paired结果：

- 提升：`+3.42 pp`
- 99% CI：`[+1.41,+5.67]`
- 单边`p=1/10001`
- 冻结MDE：`0.70 pp`

不修改B3规则的drop-in比较：

- `60.09% -> 63.63%`
- 提升`+3.54 pp`
- 99% CI：`[+1.27,+6.15]`

因此可以继续支持：

> 在固定测试时forward预算下，将计算从高度相关的同模型视角重新分配到多个模型谱系，可以显著提升GUI grounding性能。

### 3.2 预算斜率符号翻转

X3的结果保持成立：

- V-only M1斜率：`-0.002467`
- 99% CI：`[-0.004908,-0.000124]`
- Mixed M1斜率：`+0.003052`
- 99% CI：`[+0.001082,+0.005053]`

这说明增加forward是否有效取决于新增候选的类型，而不只是数量。

### 3.3 Proposal质量衰减

L4测得共享GTA1 proposer的完整bbox containment随rank下降：

- rank 0：`99.94%`
- rank 11：`61.04%`

这支持定性的两因子解释：

- 同模型加视角时，相关性下降有限；
- 同时后排proposal质量持续下降；
- 因此净收益可能为负。

## 四、Coverage与最终准确率的分离

CALA-S N12是最直接的负结果：

| 方法 | B3 | pass@12 |
|---|---:|---:|
| Uniform Mixed | 63.69% | 79.19% |
| CALA-S | 62.18% | 80.01% |

CALA-S提高了候选并集覆盖，但降低了最终准确率。

因此：

> pass@N衡量“候选池中是否至少有一个正确答案”，而最终准确率还要求聚合规则正确识别这个候选。二者不能互换。

这也解释了为什么此前多次selector改进没有稳定解决问题：瓶颈不是单纯“有没有正确候选”，而是候选结构、错误簇和聚合器之间的联合失配。

## 五、N2单模型上界

N2状态为：

```text
BLOCKED_N1_COLLAPSE
```

因为N1的一维collapse没有通过，所以不能从$1/\rho_\mathrm{view}$推出可信的单模型准确率上界。

failure-kappa拟合在$N_\mathrm{eff}=1/0.895$处会给出约`65.65%`的诊断性外推，但该拟合：

- 残差SD为`7.30 pp`；
- 斜率为负；
- 不满足预注册collapse条件。

因此不能将该数字写成理论天花板或不可能性定理。

## 六、72B Lane审计

N3否定了“全局坐标解析bug”猜测。

| 模型 | 本地全图分数 | 纸面anchor | 差异 | Anchor判定 |
|---|---:|---:|---:|---|
| GTA1-72B | 58.51% | 58.4% | +0.11 pp | PASS |
| UI-Venus-72B | 60.53% | 61.9% | -1.37 pp | PASS |
| Qwen3.5-122B-A10B | 71.41% | 70.4% | +1.01 pp | PASS |

坐标范围审计：

- GTA1：0个全图点越界；
- Qwen3.5：0个全图点越界；
- UI-Venus：5/1,581个点越界；
- 三个模型裸分均在±2 pp容差内。

最终N3状态：

```text
PASS_NO_GLOBAL_COORDINATE_BUG
```

因此没有修改parser或坐标缩放。72B低B3被保留为真实的聚类/候选污染边界。

## 七、NOA结果

### 7.1 NOA-static

NOA-static按照开发集failure相关矩阵，贪心最大化generalized N_eff：

```text
N_eff(S) = |S|² / (1ᵀ R_S 1)
```

N12结果：

| 方法 | B3 | M1 | pass@12 |
|---|---:|---:|---:|
| Uniform Mixed | 63.69% | 63.82% | 79.19% |
| NOA-static | 62.24% | 63.06% | 79.51% |

B3比较：

- 差值：`-1.45 pp`
- 99% CI：`[-3.03,+0.06]`
- 最低成功判据：FAIL

NOA没有修复CALA-S，进一步说明仅优化相关矩阵下的有效样本量仍不足以优化最终聚合准确率。

### 7.2 NOA-stop

NOA-stop使用已经观察到的候选坐标计算实现的边际N_eff，并由开发fold选择停止阈值。

结果：

- 平均forward：`6.19`
- 中位数forward：`5`
- B3：`61.10%`
- Uniform Mixed N12 B3：`63.69%`
- 差值：`-2.59 pp`
- 99% CI：`[-3.70,-1.40]`

它显著节省计算，但精度下降超过冻结MDE，因此不能声称“同等精度下节流成功”。

## 八、停止收益闸门

N5本身强通过。

在SafeGround最高分歧20%的样本中：

- pass@4：`38.29%`
- pass@8：`45.89%`
- pass@12：`51.27%`
- pass@12 − pass@4：`+12.97 pp`
- 99% CI：`[+8.33,+18.12]`
- `p=1/10001`

这说明高分歧行仍有可被额外预算救回的候选headroom。

所以NOA-stop失败不能解释为“困难行已经无药可救”，而是：

> 当前停止信号和allocation目标没有把可用headroom转化为最终准确率。

## 九、CALA低预算结果仍成立

尽管N12的CALA/NOA目标失败，N8的预算特定正结果仍然成立。

### 7B N8

- Uniform Mixed：`61.99%`
- CALA-A：`63.06%`
- 提升：`+1.08 pp`
- 99% CI：`[+0.29,+2.10]`

### 72B N8

- Uniform Mixed：`41.24%`
- CALA-S：`45.41%`
- 提升：`+4.17 pp`
- 99% CI：`[+1.56,+7.06]`

这些结果支持低预算调度，但不能扩展为“所有预算下都有效”的普适方法。

## 十、论文应如何表述

### 可以主张

1. 在固定12-forward预算下，跨谱系分配显著优于同模型多视角。
2. 同模型视角具有较高失败相关性，后排proposal质量显著下降。
3. 相关性和proposal质量共同解释预算扩展为何可能饱和或反转。
4. pass@N、N_eff和最终聚合准确率是不同层次的量，不能互相替代。
5. CALA在N8低预算下获得7B/72B方向一致的等算力提升。
6. 高分歧样本仍存在额外预算headroom。

### 不可以主张

1. 不存在已验证的统一N_eff准确率collapse。
2. 不能提出单模型$1/\rho$准确率上界或不可能性定理。
3. NOA-static没有优于Uniform Mixed。
4. NOA-stop没有达到同等精度节流。
5. 72B低B3不是已证实的坐标解析bug。
6. 不能宣称绝对开源SOTA。

## 十一、推荐的一句话总结

> Equal-compute cross-lineage allocation robustly improves GUI grounding, but neither candidate coverage nor effective sample size alone predicts final accuracy; error correlation, proposal quality, candidate pollution, and aggregator realizability jointly determine test-time scaling behavior.

中文表述：

> 等算力跨谱系分配能够稳定提升GUI grounding，但候选覆盖率或有效样本量都不能单独预测最终准确率；错误相关性、proposal质量、候选污染和聚合器兑现能力共同决定测试时扩展的收益。

## 十二、执行状态

| 项目 | 状态 |
|---|---|
| N1 collapse | FAIL |
| N2 upper bound | BLOCKED |
| N3 72B repair | PASS_NO_GLOBAL_COORDINATE_BUG；不修复 |
| N4 NOA-static | FAIL |
| N4 NOA-stop | FAIL |
| N5 stopping gate | PASS |
| N6 action-space extension | 不运行，N4闸门失败 |

完整机器可读结果、主表和图见：

- `n1_collapse.json`
- `n2_upper_bound.json`
- `n3_72b_repair.json`
- `n4_noa.json`
- `n5_stopping_gate.json`
- `MAIN_TABLE.md`
- `REPORT.md`
- `STATUS.json`
- `fig2_neff_collapse.pdf`