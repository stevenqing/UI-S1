# 多模型 GUI Grounding、72B Scale-Up 与来源偏置完整总结

更新日期：2026-08-06
最终状态：`COMPLETE_WITH_RECOVERY_DRIFT`
数据集：ScreenSpot-Pro，共 1,581 条样本

## 阅读说明

本文件是本轮工作的单一 Markdown 总入口，整合研究 idea、恢复环境、三模型推理、Scale-Up、M0、B1、B4、B2、可复现性边界、最终 gate 和交付物路径。

正文使用 2026-08-06 恢复 bank 重新计算的 combined-24 结果。文末保留 2026-08-03 的 frozen 21-method 历史总结作为附录，用于解释协议演进；两套数字属于不同方法集合，不能当作同一次运行的重复结果。

## 一、研究目标

本轮不是简单比较三个大模型，而是在验证：

> 多个 GUI grounding 模型能否提供互补候选，以及一个消除同源重复票、尽量与候选来源解耦的聚合器，能否把候选池的高 oracle coverage 转化为稳定优于单模型的最终点击准确率。

系统分为两个阶段：

1. **候选生成**：GTA1、UI-Venus、Qwen3.5 等模型从全图和 crop 提出点击候选；
2. **聚合选择**：B3、M1 或 lineage-normalized router 根据候选几何、来源和开发集可靠性输出最终点击点。

需要区分四个问题：

| 层次 | 研究问题 | 主要证据 |
|---|---|---|
| 互补性 | 多模型是否覆盖更多正确区域 | `pass@N`、proposal sensitivity |
| 可选择性 | 聚合器能否识别正确候选 | B3、M1、nested LN |
| 机制 | 失败是否来自来源偏置、计数不平衡或重复票 | M0、B1、B4 |
| 可扩展性 | 修复是否同时适用于 7B 与 72B/122B | B2 双尺度 gate |

## 二、一页结论

1. 三个强模型具有明显候选互补性。72B mixed N12 的 `pass@N` 为 **84.63%**。
2. 原始聚合没有兑现候选覆盖。mixed N12 的 M1 为 **49.15%**，B3 为 **32.45%**。
3. B3 存在极强的模型来源偏置。72B N8 的 929 个错误样本中，GTA1 成为赢家 **871 次**，期望只有 **348.38 次**。
4. Combined-24 lineage normalization 在 72B 上达到 **70.52%**，比 B3 高 **29.29 pp**，比恢复 M1 高 **17.33 pp**。
5. 72B nested LN 仍比 reported Qwen3.5 best-single 低 **0.89 pp**，没有超过最强单模型。
6. 7B nested LN 为 **63.69%**，与 B3 相同，因此双尺度主标准失败。
7. B4 不支持“共享 GTA proposer 单独导致偏置”的强解释；正式解释是 `heterogeneous_pool_aggregation_effect`。
8. 恢复 bank 结构完整且内部哈希一致，但不与 historical frozen bank 字节一致，因此状态是 `COMPLETE_WITH_RECOVERY_DRIFT`，不是 exact reproduction。

## 三、模型、环境与安全约束

### 3.1 模型

| 模型 | revision | 角色 |
|---|---|---|
| GTA1-72B | `674ce162e90c5b335ad5d1abc08ca7bfc3f42558` | attention proposer 与 scorer |
| UI-Venus-Ground-72B | `e9d2aa95593df7dc029d7717d59a2abebbea987a` | scorer |
| Qwen3.5-122B-A10B | `dc4d348443bc740c68e2d77492492c11606384d5` | scorer / best-single 对照 |

7B 对照池由 GTA1-7B、Qwen3-VL-8B-Instruct 和 UI-TARS-7B-SFT 构成。

### 3.2 运行环境

- `uv 0.11.28`；
- `.venv-scaleup`：Python 3.12、Torch 2.13.0+cu132、Transformers 5.14.1、vLLM 0.26.1 开发版；
- 三个大模型均使用 TP=8，严格顺序运行；
- `gpu_memory_utilization=0.58`；
- 外部 AzureML 进程 PID `2274` 全程受保护，没有被 signal、kill、pause 或 reprioritize。

### 3.3 恢复 bank

| Score bank | 行数 | SHA-256 |
|---|---:|---|
| GTA1-72B | 1,581 | `59bf61d6446cf8169411e05bf6b9c72de0aef944d4a1a3087a372a1113ac64ae` |
| UI-Venus-Ground-72B | 1,581 | `2cc070d49dc6ae17d9700e9bea23aeb15cf670e6cde5d343e47faa20fc89b018` |
| Qwen3.5-122B-A10B | 1,581 | `4bcaf70ab385c20d2d91c1251f69c6641fc23fa19570cfddd5096cc8c2dcb553` |

每个 bank 均通过模型 revision、1,581 个 ID、region manifest hash、prediction hash、region coverage 和候选顺序检查。

## 四、恢复执行过程

### 4.1 P1 N8 fallback

P1 原计划使用 12 个 GTA1 forward，但 `stata_windows_27` 只能产生 7 个唯一 crop。因此在任何 G2 scoring accuracy 产生前启用全局 fallback：

- P1：GTA1 full image + 7 个唯一 crop，共 8 forwards；
- P2：三个模型各使用 full image + 3 个共享 crop，共 12 forwards；
- P2 与 P1 只能作 unequal-budget context，不能表述为 equal-budget allocation；
- 73.1% 绝对门槛不变。

GTA1 crop 在 smart resize 前放大 2 倍并映射回原坐标；Venus 与 Qwen3.5 使用原 crop。

### 4.2 三模型评分

1. GTA1-72B smoke 通过，随后完成 `1581/1581`；
2. UI-Venus-Ground-72B smoke 通过，随后完成 `1581/1581`；
3. Qwen3.5 在 `batch-size=4` 时触发 vLLM scheduler `KeyError`；
4. `batch-size=2` smoke 通过，但全量首行触发 `Encoder cache miss`；
5. `batch-size=1` 稳定完成 `1581/1581` 并输出 `PASS`。

Qwen3.5 的故障来自开发版 vLLM 多模态 batch queue / encoder cache，不是 OOM、权重缺失或数据损坏。任务结束后的 NCCL abort、TCPStore closed 和 resource-tracker 警告发生在正常 teardown 阶段，不影响已经逐行 flush 和 fsync 的结果。

### 4.3 分析顺序

最终顺序为：重建 mixed 72B -> B1 -> B4 -> combined-24 与 R0-only B2 -> 根据 gate 决定 B3x。

## 五、Scale-Up 结果

| Pool | Budget | pass@N | M1 | B3 |
|---|---:|---:|---:|---:|
| P1 GTA1-72B fallback | 8 | 69.39% | 25.62% | 23.59% |
| P2 mixed 72B | 12 | 84.63% | 49.15% | 32.45% |

P2 相对 P1 的 M1 提升是 **23.53 pp**，但预算不同。Proposal-sensitivity MDE 为 **1.83 pp**。

| Gate | 结果 |
|---|---|
| P2 高于 P1 unequal-budget context | PASS |
| 73.1% effective threshold | FAIL |
| 73.1% system-SOTA gate | FAIL |
| outcome | `BELOW_PAPER_MODEL_REFERENCE` |

`pass@N=84.63%` 表明候选池经常已经包含正确答案；M1/B3 与该上限之间的巨大差距说明主要瓶颈是候选选择，而不是候选生成。

P2 mixed N12 与后续 Source-Bias N8 是不同候选池，因此 P2 B3 `32.45%` 与 N8 B3 `41.24%` 不能视为同一指标。

## 六、M0：净 +1 背后的五个翻转

M0 对齐 H3 model-major 与 L1 view-major 的 B3 语义。Aggregate 只显示净 `+1`，但逐样本有：

- 3 个 rescue；
- 2 个 regression；
- 合计 5 个 correctness flip；
- 净结果 `+1`。

这说明候选顺序、MVP group 顺序和 tie-break 可以产生相反方向的个体变化。Headline parity 与 ordering sensitivity 必须分开报告。

## 七、B1：来源偏置

B1 比较实际赢家来源与按候选占比推导的期望分布，报告卡方检验、Cramer's V 和 standardized residual。Gate 要求 7B N12 与 72B N8 的错误行中 GTA residual 均显著为正且 $p<0.001$。

| Pool | 错误行 | GTA 实际赢家 | GTA 期望赢家 | GTA residual | $p$ | Cramer's V |
|---|---:|---:|---:|---:|---:|---:|
| 7B Uniform Mixed N12 | 574 | 489 | 191.33 | +26.36 | $4.12\times10^{-152}$ | 0.779 |
| 72B Uniform Mixed N8 recovery | 929 | 871 | 348.38 | +35.42 | $1.21\times10^{-273}$ | 0.822 |

72B 错误行完整分布：

| 来源 | 候选成员 | 期望赢家 | 实际赢家 | residual |
|---|---:|---:|---:|---:|
| GTA1-72B | 2,787 | 348.38 | 871 | +35.42 |
| UI-Venus-Ground-72B | 2,787 | 348.38 | 53 | -20.02 |
| Qwen3.5-122B-A10B | 1,858 | 232.25 | 5 | -17.22 |

B1 在两个尺度上均通过。错误赢家严重集中于 GTA，远超 nominal candidate share 能解释的程度。

## 八、B4：机制归因与候选计数

### 8.1 Proposer 强归因

| 尺度 | GTA view-0 residual | GTA crop residual | view-0 更弱 | GTA 几何显著低于两者 |
|---|---:|---:|---|---|
| 7B | +37.16 | +24.68 | 否 | 是 |
| 72B | +31.56 | +40.32 | 是 | 否 |

两个尺度没有同时满足强归因条件，因此不能声称共享 GTA proposer 单独造成偏置。正式结论是：

> `heterogeneous_pool_aggregation_effect`

### 8.2 Count balancing

72B N8 的 lineage slot 为 `3/3/2`。按最早 view 确定性平衡到 `2/2/2` 后：

- B3：41.24% -> 49.72%；
- delta：**+8.48 pp**；
- 候选数：8 -> 6。

但平衡后的 795 个错误样本中 GTA 仍成为赢家 744 次，residual 为 `+36.04`。计数不平衡是问题的一部分，但不是全部机制。

10,000 次随机全局 action-subset 平衡得到：

| 指标 | 数值 |
|---|---:|
| mean accuracy | 41.03% |
| median | 37.51% |
| standard deviation | 11.36 pp |
| 99% interval | [23.34%, 55.22%] |
| min / max | 23.34% / 55.22% |

不同被保留 view 会造成巨大差异，因此确定性 `+8.48 pp` 不能解释为“平衡数量必然提升”；更准确的机制是 candidate count 与 view composition 共同作用。

## 九、B2：Combined-24 Lineage Normalization

最终 amendment 冻结 24 个方法：R0a/R0b/R0c 加 R1--R7 与 D1--D3 的 21 个组合。每个模型谱系先压缩为一个代表，再由三个谱系代表进行决策。

采用严格五折 nested selection：inner validation 只选方法，outer development 重拟合，outer test 每行只产生一个 claim-bearing prediction。完整 grid 最大值只能作描述性分析。

### 9.1 Recovery 主结果

| 比较 | Nested LN | Reference | Delta | 99% CI | 单边 $p$ |
|---|---:|---:|---:|---:|---:|
| 7B LN vs B3 | 63.69% | 63.69% | 0.00 pp | [-0.54, +0.65] | 0.5626 |
| 7B LN vs M1 | 63.69% | 63.82% | -0.13 pp | [-1.01, +0.76] | 0.6731 |
| 72B LN vs B3 | 70.52% | 41.24% | +29.29 pp | [+21.21, +35.88] | $1/10001$ |
| 72B LN vs recovered M1 | 70.52% | 53.19% | +17.33 pp | [+11.96, +21.99] | $1/10001$ |
| 72B LN vs matched Qwen3.5 view-0 | 70.52% | 71.28% | -0.76 pp | [-2.92, +1.73] | 0.8081 |
| 72B LN vs reported best-single | 70.52% | 71.41% | -0.89 pp | 独立口径 | n/a |

72B 五折选择为 R5_D2 三次、R5_D3 两次，说明有效修复依赖“每个谱系选择开发集最强 view，再按谱系可靠性处理支持或完全分歧”，而不是简单平均。

### 9.2 Gate

- Combined-24：72B 成功，7B 失败；
- R0-only：两个尺度均失败；
- 双尺度 primary success：`false`；
- B-K4：`true`，72B LN 仍低于 best-single；
- B-K5：`false`；
- B3x：`CANCEL`。

Lineage normalization 是 72B 聚合污染的强修复，但不是已验证的跨尺度通用方法。

## 十、Frozen 与 Recovery 边界

恢复 bank 的 ID、行数、模型 revision、region manifest 与内部 prediction hash 均一致，但 raw SHA 与 historical frozen bank 不同。小量输出变化传播到 tie-break 和 fold-local reliability。

### 10.1 B1 anchor drift

| Anchor | Frozen GTA/Venus/Qwen3.5 | Recovery GTA/Venus/Qwen3.5 |
|---|---|---|
| winning-set members | 1374 / 1000 / 370 | 1370 / 1003 / 369 |
| final winners | 872 / 52 / 5 | 871 / 53 / 5 |

### 10.2 B2 baseline drift

| Baseline | Frozen | Recovery | Delta |
|---|---:|---:|---:|
| 7B B3 | 63.69% | 63.69% | 0.00 pp |
| 7B M1 | 63.82% | 63.82% | 0.00 pp |
| 72B B3 | 41.24% | 41.24% | 0.00 pp |
| 72B M1 | 52.12% | 53.19% | +1.08 pp |

默认 frozen runner 仍会在 anchor 或 baseline 不一致时失败。只有显式 recovery 参数才允许继续，并在独立 JSON 中记录 expected、actual、delta 和 `RECOVERY_DRIFT_ACCEPTED`。历史 B1/B2/B4 结果没有被覆盖。

原始 21-method frozen study 得到 7B 61.99%、72B 70.59%。后续 amendment 冻结 R0a/R0b/R0c 并形成 combined-24。本报告正文使用最终 combined-24 recovery 数字：7B 63.69%、72B 70.52%。

## 十一、最终 Gate 总表

| Gate | 结果 | 含义 |
|---|---|---|
| 三模型 score bank | PASS | 各 1,581 行，hash/coverage/order 一致 |
| Mixed Scale-Up structure | PASS | source SHA 与 bank 匹配 |
| 73.1% threshold | FAIL | P2 M1 为 49.15% |
| B1 source bias | PASS BOTH SCALES | GTA 错误赢家显著过度代表 |
| B4 proposer-specific attribution | NOT SUPPORTED | 条件未跨尺度同时满足 |
| B2 72B correction | PASS | LN 显著高于 B3/M1 |
| B2 7B correction | FAIL | LN 与 B3 相同 |
| B2 cross-scale primary | FAIL | 要求两个尺度同时成功 |
| B-K4 | TRIGGERED | 72B LN 仍低于 best-single |
| B3x | CANCELLED | B2 双尺度 gate 失败 |
| Exact frozen reproduction | FAIL | raw SHA 不同 |
| Recovery analysis | COMPLETE | `COMPLETE_WITH_RECOVERY_DRIFT` |

## 十二、Claim 边界

### 支持

1. 多模型候选具有显著互补性；
2. B3 在两个尺度上都存在强模型来源偏置；
3. 同源重复票与候选 composition 会严重影响聚合结果；
4. Lineage normalization 能在 72B 污染场景恢复大部分 latent best-single headroom；
5. 候选覆盖、相关性与聚合可兑现性必须分开分析。

### 不支持

1. 偏置已被证明由共享 GTA proposer 单独造成；
2. Lineage normalization 在所有尺度都优于 B3/M1；
3. 72B nested LN 超过最强单模型；
4. 描述性 grid 最大值可以作为部署 headline；
5. B3x 已验证统一修复；
6. Recovery 是 frozen raw bank 的字节级精确复现。

推荐总表述：

> 多模型 GUI grounding 的价值主要来自候选互补性，但 flat aggregation 会因来源敏感的重复投票而丢失大量可兑现性能。谱系归一化在 72B/122B 异质候选池中能恢复大部分 best-single headroom，但该收益没有跨尺度泛化，也未超过最强单模型。因此当前证据支持“72B 聚合偏置修复”，而不是“通用多模型聚合方法”。

## 十三、验证与安全

- Source-bias contracts：`7/7 PASS`；
- 7 个核心 recovery artifact 路径与 SHA-256 全部重新验证；
- Python、JSON、Markdown 无 VS Code diagnostics；
- `git diff --check` 通过；
- 无残留评分进程；
- 外部 PID `2274` 在最终检查时正常存活，且从未被操作。

Scale-Up venv 未安装 pytest，因此没有通过 pytest runner 执行 `test_scaleup.py`；但重建脚本自身的 1,581 行 identity、source hash、prediction hash、coverage 和 target-leak 门禁全部通过。Source-bias tests 使用 `.venv-scaleup` 的标准库 `unittest` 执行。

## 十四、交付物索引

### 总状态

- 机器可读状态：[RECOVERY_STATUS.json](RECOVERY_STATUS.json)；
- 精简 recovery 报告：[RECOVERY_REPORT.md](RECOVERY_REPORT.md)。

### Scale-Up

- Mixed 72B：[g2_mixed_72b.json](../../scaleup/2026-08-02/g2_mixed_72b.json)；
- GTA1 bank：[g2-score-gta1.jsonl](../../scaleup/2026-08-02/raw/g2-score-gta1.jsonl)；
- Venus bank：[g2-score-venus.jsonl](../../scaleup/2026-08-02/raw/g2-score-venus.jsonl)；
- Qwen3.5 bank：[g2-score-qwen35.jsonl](../../scaleup/2026-08-02/raw/g2-score-qwen35.jsonl)；
- Regions：[g2-regions.jsonl](../../scaleup/2026-08-02/raw/g2-regions.jsonl)；
- P1 fallback：[AMENDMENT_001_G2_P1_N8_FALLBACK.md](../../scaleup/2026-08-02/AMENDMENT_001_G2_P1_N8_FALLBACK.md)。

### B1 / B4 / B2

- Recovery B1：[recovery_b1_source_bias.json](results/recovery_b1_source_bias.json)；
- Recovery B1 figure：[recovery_b1_source_bias.pdf](figures/recovery_b1_source_bias.pdf)；
- Recovery B4：[recovery_b4_attribution.json](results/recovery_b4_attribution.json)；
- Recovery B2：[recovery_b2_lineage_normalized.json](results/recovery_b2_lineage_normalized.json)；
- 预注册协议：[SPEC.md](SPEC.md)；
- Full-spec amendment：[AMENDMENT_001_FULL_SPEC_EXTENSION.md](AMENDMENT_001_FULL_SPEC_EXTENSION.md)；
- B1 config：[b1_pools.yaml](configs/b1_pools.yaml)；
- B2 combined-24 config：[b2_variants.yaml](configs/b2_variants.yaml)。

### 历史与诊断

- M0：[m0_manifest_diff.json](../../final/2026-08-04/m0_manifest_diff.json)；
- Frozen B1：[b1_source_bias.json](results/b1_source_bias.json)；
- Frozen B4：[b4_attribution.json](results/b4_attribution.json)；
- Frozen B2：[b2_lineage_normalized.json](results/b2_lineage_normalized.json)。

## 十五、最终定位

1. **有没有正确候选？** 有，mixed N12 oracle coverage 为 84.63%；
2. **原始聚合能否选出来？** 不能稳定做到，M1/B3 与 oracle coverage 差距很大；
3. **能否修复？** 72B 修复到 70.52%，7B 没有改善，且仍未超过 Qwen3.5 best-single。

最稳健结论是：

> 来源偏置和同源重复投票是强模型异质候选池中的关键聚合失败机制。Lineage normalization 能显著修复 72B 的聚合崩溃，但修复具有尺度依赖性；下一步不应只是继续堆候选，而应设计保留谱系内有效几何、同时控制谱系间重复票的自适应 router。

---

## 附录 A：2026-08-03 Frozen 21-Method 历史总结

以下内容保留原始 frozen study 的方法、数字与表格，不代表 2026-08-06 combined-24 recovery headline。

### A.1 研究问题

此前的实验已经表明：候选池的oracle headroom、有效样本量和最终聚合准确率并不是同一个量。尤其在72B候选池中，单个Qwen3.5候选本身达到`71.41%`，但B3聚合后仅有`41.24%`。本实验研究两个问题：

1. B3是否存在系统性的模型来源偏置，即某一模型谱系是否因候选数量、组内一致性或投票结构而过度成为赢家；
2. 将每个模型谱系先压缩为一个代表，再让三个谱系各投一票，是否可以兑现候选池中已经存在的best-single headroom。

整个实验不进行任何新模型推理，只复用已有候选trace。协议在结果产生前提交，固定了候选池、来源归因、21个lineage-normalized变体、五折嵌套选择和成功标准。

### A.2 实验协议

### 2.1 B1来源偏置检验

对预注册候选池中的B3、M1和graph centroid分别恢复赢家来源，并在`all`、`correct`和`incorrect`三个分层中比较：

- 观测到的赢家来源分布；
- 根据候选池实际模型占比得到的期望分布。

统计量包括卡方拟合优度检验、Cramer's V和标准化残差。

来源归因规则为：

- B3：使用官方MVP分组与coverage tie-break最终选中的真实候选；
- M1：使用`ccm_select()`返回的真实候选索引；
- graph centroid：质心本身不是候选，因此归因到距离生成质心最近的真实候选，同时单独报告赢家component的来源构成。

B1通过条件是：7B Uniform Mixed N12和72B Uniform Mixed N8的B3错误行中，GTA来源均呈显著正残差，且$p<0.001$。

### 2.2 B4机制归因

B4使用三组零推理诊断区分两种解释：

- 强解释：共享GTA proposer直接造成来源偏置；
- 弱解释：异质候选池与聚合规则共同形成来源投票偏置。

具体检查：

1. 比较view 0全图池与views 1--3 crop池的GTA来源残差；
2. 比较各谱系四视图内的归一化pairwise distance；
3. 对候选数不平衡的池进行确定性下采样，再重算B3。

只有当两个尺度上GTA在view 0中的过度代表都弱于crop views，且GTA组内距离都显著低于另外两条谱系时，才支持共享proposer强归因。

### 2.3 B2谱系归一化聚合

Lineage normalization分两步：

1. 每个模型谱系内部的多个候选压缩为一个代表；
2. 三个谱系代表进行等谱系投票或可靠性决策。

预注册了$7\times3=21$个组合：

- R1：最大official complete-link group的质心；
- R2：几何中位数；
- R3：medoid；
- R4：最大14px图连通分量质心；
- R5：开发集最强view候选；
- R6：view-0候选；
- R7：最大official group内按开发集action accuracy加权的质心；
- D1：三个谱系代表的等权最大连通分量质心；
- D2：开发集lineage reliability加权的14px支持代表；
- D3：完全分歧时回退到开发集最可靠谱系，否则使用D1。

采用严格五折嵌套选择。对outer fold $f$：

- inner validation固定为$(f+1)\bmod5$；
- 其余三折用于拟合开发统计；
- 21个组合仅根据inner-validation B3准确率选择；
- 选定组合在完整outer-dev四折上重拟合；
- 最后只在outer-test上评估一次。

因此headline结果由1,581条完全held-out预测组成。完整21-grid仅作描述性敏感性分析，不使用其最大值作为主结果。

### A.3 B1：来源偏置在两个尺度上均成立

| Pool/stratum | 错误行 | GTA观测赢家 | 按候选比例期望 | GTA标准残差 | 卡方$p$ | Cramer's V |
|---|---:|---:|---:|---:|---:|---:|
| 7B Uniform Mixed N12，B3错误行 | 574 | 489 | 191.33 | +26.36 | $4.12\times10^{-152}$ | 0.779 |
| 72B Uniform Mixed N8，B3错误行 | 929 | 872 | 348.38 | +35.49 | $1.15\times10^{-274}$ | 0.824 |

B1在两个尺度上均通过。偏置不仅统计显著，而且效应量极大：

- 7B错误行中，GTA成为赢家的比例为`85.19%`；
- 72B错误行中，GTA成为赢家的比例为`93.86%`；
- 这些比例远高于候选池中GTA候选本身所占比例。

这说明B3的错误并非来源中性。某一谱系可以依靠候选数量和组内投票结构形成占优势的错误簇，从而压制更强的单候选来源。

### A.4 B4：不支持共享proposer强归因

### 4.1 View-0与crop来源偏置

| 尺度 | GTA view-0残差 | GTA views 1--3残差 | view-0是否更弱 |
|---|---:|---:|---|
| 7B | +37.16 | +24.68 | 否 |
| 72B | +31.60 | +40.08 | 是 |

7B方向与共享proposer强归因的要求相反：GTA在全图view 0中的过度代表反而强于crop views。

### 4.2 谱系内几何

7B中，GTA四视图内距离显著低于Qwen3和UI-TARS：

- GTA减Qwen3：point `-0.01561`，99% CI `[-0.02814,-0.00367]`；
- GTA减UI-TARS：point `-0.03147`，99% CI `[-0.04269,-0.02128]`。

但72B不满足同一条件：

- GTA减Qwen3.5：point `+0.00122`，99% CI `[-0.00653,+0.00901]`；
- GTA减Venus：point `+0.00027`，99% CI `[-0.00509,+0.00533]`。

因此不能把两个尺度上的来源偏置统一归因于“GTA proposer产生了更紧的同源候选”。

### 4.3 候选数平衡敏感性

72B Uniform Mixed N8原始谱系候选数为`3/3/2`。将其确定性平衡为`2/2/2`后：

- B3从`41.24%`提高到`49.84%`；
- 提升`+8.60 pp`；
- 候选数从8降至6。

这是描述性敏感性结果，不是预注册方法结果，但它说明候选数量不平衡确实是72B B3污染的重要组成部分。

### 4.4 B4结论

两个尺度不共同满足proposer强归因条件。因此最稳健的机制表述是：

> 观测到的是异质候选池中的聚合投票偏置，而不是已经被证明由共享proposer单独造成的偏置。

### A.5 B2：72B强修复，7B失败

### 5.1 主结果

| 比较 | Nested LN | Reference | Delta | 99% CI | 单边$p$ |
|---|---:|---:|---:|---:|---:|
| 7B LN vs B3 | 61.99% | 63.69% | -1.71 pp | [-3.09,-0.21] | 0.9990 |
| 7B LN vs M1 | 61.99% | 63.82% | -1.83 pp | [-3.28,-0.19] | 0.9985 |
| 7B LN vs reported best-single | 61.99% | 54.65% | +7.34 pp | 独立trace，无paired CI | n/a |
| 72B LN vs B3 | 70.59% | 41.24% | +29.35 pp | [+21.57,+35.78] | $1/10001$ |
| 72B LN vs M1 | 70.59% | 52.12% | +18.47 pp | [+12.95,+23.44] | $1/10001$ |
| 72B LN vs best-single | 70.59% | 71.41% | -0.82 pp | [-3.01,+1.72] | 0.8222 |

### 5.2 72B结果解释

72B nested LN将B3从`41.24%`提高到`70.59%`，恢复了`29.35 pp`，也比M1高`18.47 pp`。这说明72B候选池中大部分性能损失确实来自聚合偏置，而不是候选本身缺乏正确答案。

五个outer fold的选择高度稳定：

- fold 0--2选择R5_D2；
- fold 3--4选择R5_D3。

R5直接使用每个谱系在开发集上最强的view，D2/D3再利用谱系可靠性处理代表之间的支持或完全分歧。这个稳定选择与72B的机制一致：弱谱系不应通过候选数量获得额外票数，开发集更可靠的谱系应在完全分歧时占优。

但是nested LN仍比Qwen3.5 best-single低`0.82 pp`，且99% CI跨0。因此它接近兑现best-single headroom，但没有显著超过或完全达到best-single。

### 5.3 7B结果解释

7B nested LN为`61.99%`，显著低于B3和M1。五折选择也不稳定：

- R1_D2；
- R4_D1；
- R1_D1；
- 两次R3_D1。

这说明7B没有单一稳定的谱系归一化决策结构。原始B3/M1在7B上已经能够利用同谱系内部的多候选几何；过早将每条谱系压缩到一个代表会丢失有效信息。

因此lineage normalization不是跨尺度普适改进，而是72B污染场景中的强修复。

### 5.4 Best-single口径说明

7B预注册best-single `54.65%`来自独立H3 bare trace。该trace在仓库中仅保留汇总及source hash，没有保留逐行预测，因此不能构造paired CI。当前36-action反事实bank中的Qwen3 view-0为`54.52%`，仅用于matched-bank诊断，不替代预注册best-single。

72B Qwen3.5 bare trace与当前bank一致，均为`71.41%`，因此可以进行paired bootstrap。

### A.6 门控判定

| Gate | 结果 | 原因 |
|---|---|---|
| B1 source-bias gate | PASS，两个尺度 | GTA错误赢家残差均显著为正 |
| B2 72B bias correction | PASS | 对B3和M1的99% CI下界均为正 |
| B2跨尺度主成功 | FAIL | 7B nested LN显著低于B3 |
| B-K4 | TRIGGERED | 72B nested LN仍低于71.41% best-single |
| B3x | NOT RUN | B2要求两个尺度同时成功 |
| 共享proposer强归因 | NOT SUPPORTED | B4条件未在两个尺度同时满足 |

### A.7 完整21变体敏感性结果

下表是cross-fitted描述性敏感性分析，不是headline nested结果。

| Variant | 7B | 72B |
|---|---:|---:|
| R1_D1 | 62.62% | 51.36% |
| R1_D2 | 61.99% | 64.58% |
| R1_D3 | 61.80% | 64.77% |
| R2_D1 | 60.40% | 23.53% |
| R2_D2 | 59.52% | 22.83% |
| R2_D3 | 59.71% | 22.83% |
| R3_D1 | 62.49% | 25.81% |
| R3_D2 | 61.23% | 57.37% |
| R3_D3 | 61.61% | 57.50% |
| R4_D1 | 62.87% | 51.49% |
| R4_D2 | 61.92% | 64.83% |
| R4_D3 | 61.99% | 65.02% |
| R5_D1 | 61.16% | 62.81% |
| R5_D2 | 60.09% | 70.46% |
| R5_D3 | 60.28% | 70.59% |
| R6_D1 | 51.04% | 62.81% |
| R6_D2 | 51.11% | 70.46% |
| R6_D3 | 51.17% | 70.59% |
| R7_D1 | 2.28% | 2.66% |
| R7_D2 | 2.15% | 2.40% |
| R7_D3 | 2.21% | 2.34% |

7B描述性最佳为R4_D1，准确率`62.87%`，仍低于原始B3。72B描述性最佳为R5_D3，准确率`70.59%`。这些结果只能用于解释敏感性，不能替代嵌套选择结果。

### A.8 论文贡献与正确表述

本实验支持三项结论：

1. **B3存在强模型来源偏置。** 错误赢家高度集中于GTA，且远超候选比例能够解释的水平；
2. **候选计数和谱系内重复票会造成严重聚合污染。** 72B计数平衡和lineage normalization均带来大幅恢复；
3. **Lineage normalization能够在污染严重的72B池中近似兑现best-single headroom。** 它将B3从`41.24%`恢复到`70.59%`。

推荐论文表述：

> Shared-proposal candidate pools can exhibit severe model-source voting bias: repeated candidates from one lineage dominate erroneous consensus far beyond their nominal pool share. A nested lineage-normalized aggregator removes duplicate lineage votes and recovers most of the latent best-single headroom at 72B, but the effect does not transfer to 7B, where within-lineage geometry remains useful.

不能声称：

- 来源偏置已被证明由共享proposer单独造成；
- lineage normalization在所有模型尺度上都优于B3或M1；
- 72B nested LN已经超过best-single；
- 21-grid中的事后最大值是可部署headline结果；
- B3x已经验证统一修复CALA-S、NOA和高分歧预算轴。

### A.9 最终定位

最稳健的总判断是：

> 来源偏置是B3在72B异质候选池中性能崩溃的主要机制之一。谱系归一化能够修复大部分崩溃，但该修复具有尺度依赖性；它是72B聚合偏置修复，而不是已经成立的通用聚合方法。

这与此前Effective-Sample-Size结果一致：候选覆盖、相关性和聚合可兑现性必须分开讨论。即使候选池中存在高质量候选，重复来源形成的错误簇仍可能使最终规则远离best-single上限。

### A.10 交付物索引

- `SPEC.md`：结果盲预注册协议；
- `configs/b1_pools.yaml`：B1候选池、规则、分层和来源归因；
- `configs/b2_variants.yaml`：完整21变体和嵌套选择协议；
- `results/b1_source_bias.json`：全部B1来源偏置统计；
- `results/b2_lineage_normalized.json`：B2 nested输出、fold选择和paired统计；
- `results/b4_attribution.json`：B4 proposer/几何/计数平衡诊断；
- `MAIN_TABLE.md`：核心结果表；
- `B2_VARIANT_GRID.md`：完整21-grid；
- `REPORT.md`：精简英文报告；
- `STATUS.json`：最终门控状态和artifact hashes；
- `figures/b1_source_bias.pdf`：B1来源赢家观测与期望图。

协议commit：`18e0267`。

最终结果commit：`248f336`。