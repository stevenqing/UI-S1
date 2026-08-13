# ScreenSpot-Pro Consolidation 总结

日期：2026-08-06

状态：`COMPLETE`

性质：单 benchmark 诊断论文加固；S 系列为零 GPU 重分析，Q1/Q2b 为冻结设计后的少量新推理。

## 0. 一页结论

本轮同时修正了旧主张并得到一条新的正面结果。

1. **跨谱系 action pool 优势不是普遍现象。** 在 2,160 个同预算 2/3-forward 池中，双谱系 B3 没有一个超过最优同预算单谱系池，三谱系只有 0.75% 超过。因此 S-K1 触发，主张必须限定为“少数特定跨谱系配置有效”。
2. **UI-TARS 在 N12 的边际贡献为零。** 去掉 UI-TARS 后 B3/M1 从 63.69/63.82% 变为 63.76/63.88%。S-K2 触发，弱模型表述改为“谱系多样性有效，但第三谱系已经饱和”。
3. **预算曲线应以配对端点差而非线性斜率作为主统计。** Dense V-only slope CI 跨零，但 N4 到 N16 的配对差稳定：V-only B3/M1 下降 2.91/3.16 pp，Mixed 上升 1.90/3.86 pp。S-K3 触发。
4. **V-only 下降主要由 proposer rank decay 驱动。** 打乱 GTA1 view 顺序后平均 slope 从 -0.001590 反转到 +0.002466，仅 21.2% 的随机顺序仍为负。
5. **无标注池选择器只能做 enrichment。** Held-out Spearman 为 0.638，低于 0.7 门槛；但 top-10% 池平均 B3 比随机高 4.34 pp，99% CI [+2.27,+6.95]。
6. **Q1 sequential cross-lineage consensus RoI 成功。** C-cond B3 为 65.91%，相对 Uniform N12 提升 2.21 pp，99% CI [+0.50,+4.16]；同时显著优于随机新裁剪和 GTA 自共识两个强制对照。
7. **Q2a 元素空间失败。** Combined-25 比 Combined-24 低 0.19 pp。
8. **Q2b 验证通道本身可判别，但作为过滤器失败。** Binary accuracy 为 73.68%，但 verified B3 只有 59.08%，比 Uniform N12 低 4.62 pp。

最终最强新结果是：

> 由跨谱系第一阶段共识几何生成第二阶段 RoI，在相同 12-forward 预算下，显著优于固定 Uniform views、随机新裁剪和 GTA 单谱系自共识。

## 1. 实验与数据契约

### 1.1 现有 bank

- ScreenSpot-Pro：1,581 行；
- 三条谱系：GTA1-7B、Qwen3-VL-8B-Instruct、UI-TARS-7B-SFT；
- 公共 action bank：三模型各 views 0-11；
- GTA1 额外 views 12-15 用于 S4/S5；
- 五个 application-group outer folds；
- B3 使用冻结 MVP grouping 与 coverage tie-break；
- M1 使用每折开发集拟合的 CCM。

### 1.2 Q1 模型 provenance

| 模型 | Revision | Model-index SHA-256 |
|---|---|---|
| GTA1-7B | `701bedc80b447863bd60e3318ae44f6cbbfafd78` | `3067e9b0f35596ff3426a0d0ec8c982a51fa1e110c4fc30dcf3be9ea37409df6` |
| Qwen3-VL-8B-Instruct | `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b` | `520b2e05079402e9468a8701d03d1154d14b2599593afb6effa7fb60c1bff070` |
| UI-TARS-7B-SFT | `3434901a9dd04dd3625617d839a5724fe5e2db20` | `25b162a0f0f47af097d6a49b7da3d5c7d9c2b352490131c8cde5ca59d285f18b` |

Q1 与 Q2b 每个模型均完成 1,581 行，共验证：

- Q1 raw rows：4,743；
- Q2b raw rows：4,743；
- Q2b binary checks：9,486。

所有 raw rows 均重新验证：

- ID 唯一；
- stable index 与 modulo shard 一致；
- model ID/revision/index hash 一致；
- prepared region/crop hash 一致；
- prediction/output hash 可重算；
- Q2b candidate model/view/crop 与 verifier mapping 逐项匹配。

## 2. S1：池分布

D1 的 2,160 个池是 action-level 2/3-forward 池，不是 12-forward allocation configuration。N12 的 63.69% 不能在该分布中计算 percentile。

### 2.1 B3 分布

| Pool size | 池数 | Positive share | Median delta | IQR | 最小 / 最大 |
|---|---:|---:|---:|---:|---:|
| 双谱系 | 432 | 0.00% | -6.99 pp | [-10.37,-4.05] pp | -13.47 / 0.00 pp |
| 三谱系 | 1,728 | 0.75% | -6.39 pp | [-9.23,-3.35] pp | -12.33 / +0.57 pp |

### 2.2 极值组合

三谱系 B3 最优池为：

- GTA1 view1；
- Qwen3 view1；
- UI-TARS view5；
- B3 61.67%；
- 相对同预算最优单谱系 +0.57 pp。

双谱系没有正 delta。S-K1 明确触发。

### 2.3 正确表述

不得再写“跨谱系分配普遍优于单谱系”。正确表述为：

> 少数经过结构化选择的跨谱系配置可以超过同预算单谱系配置，但任意混合通常更差。

产物：

- [s1_pool_distribution.json](s1_pool_distribution.json)
- [fig_pool_distribution.pdf](fig_pool_distribution.pdf)

## 3. S2：无标注池选择器

设计采用 5 个 application outer folds 和 3 个 action folds。训练与测试 pool 不共享 action。

| Selector | Held-out Spearman mean | Median | Top-10% delta | 99% CI |
|---|---:|---:|---:|---:|
| Geometry only | 0.361 | 0.371 | +3.78 pp | [+2.31,+5.43] pp |
| Primary：geometry + composition + dev reliability | 0.638 | 0.689 | +4.34 pp | [+2.27,+6.95] pp |
| Quality only | 0.509 | 0.580 | +3.89 pp | [+2.13,+6.07] pp |

Primary Spearman 没超过预注册的 0.7，因此不是可靠 ranking method。但 top-decile enrichment 稳定为正，说明可以作为候选池粗筛工具。

结论：`POOL_SELECTOR_NOT_SUPPORTED`，但保留 enrichment diagnostic。

产物：[s2_pool_selector.json](s2_pool_selector.json)

## 4. S3：留一谱系消融

所有池固定 12 forwards，使用同一 bank、同一 folds。

| Pool | B3 | M1 | pass@N |
|---|---:|---:|---:|
| Full 4x3 | 63.69% | 63.82% | 79.19% |
| Leave UI-TARS，6x2 | 63.76% | 63.88% | 78.56% |
| Leave Qwen3，6x2 | 62.05% | 62.05% | 75.65% |
| Leave GTA1，6x2 | 59.27% | 60.40% | 76.72% |

去掉 UI-TARS 后 B3/M1 各提高约 0.06 pp，说明其边际贡献不为正。GTA1 与 Qwen3 的边际贡献显著为正。

S-K2 触发，claim 为：

`LINEAGE_DIVERSITY_WITH_THIRD_LINEAGE_SATURATION`

产物：[s3_leave_one_lineage.json](s3_leave_one_lineage.json)

## 5. S4：预算曲线加固

### 5.1 Dense slope

N=2 到 16 的全部整数预算：

| Pool / metric | Slope per forward | 99% CI |
|---|---:|---:|
| V-only B3 | -0.001590 | [-0.003933,+0.000640] |
| V-only M1 | +0.000926 | [-0.001678,+0.003390] |
| Mixed B3 | +0.007240 | [+0.005252,+0.009309] |
| Mixed M1 | +0.005480 | [+0.003184,+0.008081] |

V-only slope CI 跨零，因此 S-K3 触发。Slope 降为补充统计。

### 5.2 主统计：N16 minus N4

| Pool / metric | N4 | N16 | Delta | 99% CI |
|---|---:|---:|---:|---:|
| V-only B3 | 61.23% | 58.32% | -2.91 pp | [-5.58,-0.36] pp |
| V-only M1 | 61.42% | 58.25% | -3.16 pp | [-6.23,-0.25] pp |
| Mixed B3 | 61.86% | 63.76% | +1.90 pp | [+0.42,+3.47] pp |
| Mixed M1 | 59.90% | 63.76% | +3.86 pp | [+1.51,+6.28] pp |

主张的第一句应使用这一配对差：V-only 随预算下降，Mixed 随预算上升。

注意：历史 `+5.44/+5.50 pp` 是 N16 时 Mixed minus V-only，不是 N16 minus N4。

产物：[s4_slope_hardening.json](s4_slope_hardening.json)

## 6. S5：下降归因

GTA1 views 0-15 原始顺序：

- slope：-0.001590；
- N16 minus N4：-2.91 pp。

对 view 顺序做 1,000 次随机 permutation：

- mean slope：+0.002466；
- median slope：+0.002334；
- negative slope share：21.20%；
- mean N16 minus N4：+1.93 pp。

随机顺序后下降消失并平均反转为上升。因此主要机制是 rank decay，而不是高 failure correlation 本身必然导致随预算下降。

结论：`RANK_DECAY_DOMINANT`

产物：[s5_decline_attribution.json](s5_decline_attribution.json)

## 7. S6：SafeGround 与 action-cluster bootstrap

### 7.1 SafeGround

- 官方 GTA1-7B anchor：0.6344；
- 本地值：0.6278；
- delta：-0.00660；
- K、temperature、patch size、thresholding 不同；
- protocol 不匹配。

结论：`NUMERICAL_ANCHOR_NOT_PASSED_ALGORITHM_LEVEL_PORT`

只能写算法级几何移植，不能写数值复现。

### 7.2 D1 action-cluster bootstrap

2,160 个池共享 36 个 actions，因此改为 action-cluster bootstrap。

| Outcome | Raw rho 99% CI | Partial rho 99% CI | Negative share |
|---|---:|---:|---:|
| B3 minus best | [-0.659,-0.124] | [-0.619,-0.105] | 99.98% / 99.97% |
| M1 minus best | [-0.608,-0.314] | [-0.602,-0.316] | 100% / 100% |

方向在处理 action 依赖后仍稳定为负。M1 partial rho-squared 为 0.249，但它只是秩关联强度代理，不是因果方差分解。

定位：`MECHANISM_EVIDENCE_NOT_LAW`

产物：[s6_anchors.json](s6_anchors.json)

## 8. Q1：Sequential Cross-Lineage Consensus RoI

### 8.1 冻结设计

第一阶段固定六次：

- GTA1/Qwen3/UI-TARS view0；
- GTA1/Qwen3/UI-TARS view1。

第二阶段固定六次：两个 RoI × 三模型。

四个 arm：

- C-uni：原 Uniform views 2/3；
- C-cond：第一阶段最大与次大跨谱系共识簇中心；
- C-rand：seeded 随机裁剪；
- C-self：GTA1 view0/view1 中心裁剪。

所有 arm 总预算均为 12 forwards。

### 8.2 准确率

| Arm | B3 | M1 | pass@N |
|---|---:|---:|---:|
| C-uni | 63.69% | 63.82% | 79.19% |
| C-cond | **65.91%** | **66.60%** | **81.15%** |
| C-rand | 60.53% | 60.85% | 78.68% |
| C-self | 64.58% | 64.77% | 79.25% |

### 8.3 强制对照

| Comparison，B3 | Delta | 99% CI | p，plus-one |
|---|---:|---:|---:|
| C-cond minus C-uni | +2.21 pp | [+0.50,+4.16] pp | 0.0008 |
| C-cond minus C-rand | +5.38 pp | [+2.94,+8.08] pp | 0.0001 |
| C-cond minus C-self | +1.33 pp | [+0.06,+2.75] pp | 0.0040 |

Primary delta 超过 MDE 0.70 pp 且 CI 下界为正；C-cond 同时优于随机新鲜 crop 与 GTA 自共识。

Kill conditions：

- Q-K1：`false`；
- Q-K2：`false`。

结论：`CROSS_LINEAGE_CONSENSUS_ROI_SUPPORTED`

这表明收益既不是只来自“换一批新 crop”，也不是任意单谱系共识就能解释；跨谱系共识几何本身提供了额外价值。

产物：

- [configs/q1_arms.yaml](configs/q1_arms.yaml)
- [q1_sequential.json](q1_sequential.json)

## 9. Q2a：元素空间聚类

在原 combined-24 后加入 patch-28 element-cell mode，形成 combined-25：

- Combined-24 nested：63.69%；
- Combined-25 nested：63.50%；
- delta：-0.19 pp；
- 99% CI：[-0.34,0.00] pp；
- E1 在一个 outer fold 被选中。

结论：`ELEMENT_SPACE_NOT_SUPPORTED`

原 combined-24 保留，不被 combined-25 替换。

产物：

- [configs/q2a_variant.yaml](configs/q2a_variant.yaml)
- [q2a_element_space.json](q2a_element_space.json)

## 10. Q2b：跨谱系二元验证

### 10.1 冻结设计

- 第一阶段：六个生成候选；
- 第二阶段：每个候选由另一谱系做一次非自验证，共六次；
- Prompt 为单一 user message，不附加额外 system prompt；
- YES 候选送入冻结 B3；
- 全 NO 时回退到六个 stage-1 候选的冻结 B3；
- 标签：target-bbox center 是否位于 verification crop；
- parse failure 视为 NO；
- 50% seeded random filter 为强制对照。

### 10.2 验证通道本身

总体 9,486 个 binary checks：

| Metric | Value |
|---|---:|
| Accuracy | 73.68% |
| YES precision | 87.23% |
| YES recall | 74.39% |
| Label positive rate | 72.13% |
| Predicted positive rate | 61.51% |

分 verifier：

| Verifier | Accuracy | YES precision | YES recall |
|---|---:|---:|---:|
| GTA1-7B | 78.46% | 82.98% | 86.66% |
| Qwen3-VL-8B | 77.58% | 88.57% | 78.70% |
| UI-TARS-7B | 64.99% | 91.79% | 59.15% |

Q-K3 为 `false`：验证判别显著高于随机。

### 10.3 作为聚合过滤器

| Comparison | Verified B3 | Reference | Delta | 99% CI |
|---|---:|---:|---:|---:|
| Verified vs Uniform N12 | 59.08% | 63.69% | -4.62 pp | [-7.26,-2.14] pp |
| Verified vs random 50% filter | 59.08% | 55.91% | +3.16 pp | [+0.91,+5.48] pp |

验证通道确实比随机筛选更好，但它删除了太多对 B3 有用的候选，最终明显低于 Uniform N12。

结论：`CROSS_LINEAGE_VERIFICATION_NOT_SUPPORTED`

这是一条有机制信息的负结果：二元 crop containment 可以学会，但“先独立过滤再聚合”并不等价于更好的最终选择。

产物：

- [configs/q2b_verification.yaml](configs/q2b_verification.yaml)
- [q2b_verification.json](q2b_verification.json)

## 11. Kill conditions 总表

| Kill condition | 状态 | 后果 |
|---|---|---|
| S-K1：正池占比低于 60% | **触发** | 主张弱化为少数配置有效 |
| S-K2：三谱系不优于双谱系 | **触发** | 弱模型改为第三谱系饱和 |
| S-K3：dense V-only slope CI 跨零 | **触发** | 配对端点差成为主统计 |
| Q-K1：C-cond 不优于 C-rand | 未触发 | 条件化有效 |
| Q-K2：C-cond 不优于 C-self | 未触发 | 跨谱系共识有额外价值 |
| Q-K3：验证准确率不高于随机 | 未触发 | 通道可判别，但最终过滤策略失败 |

## 12. 论文主张更新

### 12.1 可以主张

1. ScreenSpot-Pro 上 V-only 与 Mixed 的 N4→N16 配对趋势方向相反。
2. V-only 下降主要由 proposer rank decay 驱动。
3. 两尺度都存在严重来源偏置。
4. D1 的 dominance-gap 负相关在 action-cluster bootstrap 下仍稳健，可作为机制证据。
5. 跨谱系共识驱动的 sequential RoI 在相同预算下显著优于三个冻结对照。
6. 无标注 pool selector 可以做 top-decile enrichment，但不是可靠全排序方法。

### 12.2 必须弱化或删除

1. 删除“跨谱系池普遍优于单谱系池”。
2. 删除“UI-TARS 弱模型带来正边际贡献”。
3. Dense slope 不再作为主证据。
4. SafeGround 只称算法级移植。
5. Q2a 不支持元素空间聚类。
6. Q2b 不支持验证过滤器带来最终 B3 提升。

### 12.3 最终正面结果表述

> Fixed late-rank proposals suffer from strong rank decay. Replacing them with equal-budget second-stage crops derived from cross-lineage consensus improves B3 by 2.21 percentage points over Uniform N12, with a positive 99% paired interval, and also outperforms random fresh crops and same-lineage self-consensus.

## 13. 复现与安全

- 三模型 Q1：每模型 1,581 行；
- 三模型 Q2b：每模型 1,581 行；
- 所有 shard 支持 resume 与逐行 fsync；
- 所有结果逐条通过 ID、shard、model revision、model-index、region/crop、output hash 检查；
- Q1/Q2b 使用模型各自兼容环境，避免 Transformers overlay 污染；
- 不同模型从未并发加载；
- 所有 pipeline logs 无 fatal signature；
- 外部 PID `2274` 全程存活且未被操作。

## 14. 交付物

- [SPEC.md](SPEC.md)：冻结协议
- [MAIN_TABLE.md](MAIN_TABLE.md)：最终主表
- [REPORT.md](REPORT.md)：英文报告
- [STATUS.json](STATUS.json)：机器可读状态与 artifact hashes
- [s1_pool_distribution.json](s1_pool_distribution.json)
- [s2_pool_selector.json](s2_pool_selector.json)
- [s3_leave_one_lineage.json](s3_leave_one_lineage.json)
- [s4_slope_hardening.json](s4_slope_hardening.json)
- [s5_decline_attribution.json](s5_decline_attribution.json)
- [s6_anchors.json](s6_anchors.json)
- [q1_sequential.json](q1_sequential.json)
- [q2a_element_space.json](q2a_element_space.json)
- [q2b_verification.json](q2b_verification.json)
- [fig_pool_distribution.pdf](fig_pool_distribution.pdf)
