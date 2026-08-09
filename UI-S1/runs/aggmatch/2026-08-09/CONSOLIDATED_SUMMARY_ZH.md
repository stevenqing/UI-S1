# 聚合器与动作空间匹配：统一总结

日期：2026-08-09

上游：`runs/close/2026-08-08/`

状态：`COMPLETE_F1_PRIMARY_F3_APPENDIX`

## 一句话结论

同一个多候选 GUI agent 不应跨动作空间机械复用聚合器：在 Mind2Web 的积动作空间（动作类型 × 坐标 × 参数）中，majority 显著优于 sequential density；在 ScreenSpot-Pro 的纯坐标空间中，离散密度族显著优于 majority。这个反向结果通过了预冻结的配对 CI 判据，成为论文新主线。

## 1. 为什么更换主线

上游 E1 已经否定了“C-cond 候选池在聚合器无关意义下更优”的主张。在预注册 majority 下，C-cond 相对 C-uni 的差异为：

- Mind2Web：+0.29 pp，99% CI 跨零；
- ScreenSpot-Pro：+1.27 pp，99% CI 跨零。

但同一个 C-uni 池在不同聚合器下差异很大：Mind2Web 的 majority 比原 sequential aggregator 高约 5.34 pp，而 ScreenSpot-Pro 的密度聚合器仍然更强。这表明候选池效果与聚合器强耦合，不能只优化候选生成而固定一个不匹配的聚合规则。

因此论文主线从“跨谱系共识 RoI 普遍改善候选池”改为：

> 聚合器的有效性依赖输出动作空间与候选错误结构；纯坐标空间和积动作空间需要不同的聚合归纳偏置。

## 2. F1：新主结果

效应统一定义为 `majority - density`，单位为百分点。所有区间使用 10,000 次配对分层 bootstrap 的 99% percentile CI；主判据只使用最接近文献默认池的 C-uni。

| Benchmark | 动作空间 | 密度对照 | Majority − Density | 99% CI | 结果 |
| --- | --- | --- | ---: | ---: | --- |
| Mind2Web | 动作类型 × 坐标 × 参数 | Sequential | **+5.34 pp** | **[+2.50,+8.04]** | Majority 显著更优，且超过 MDE 0.61 pp |
| Mind2Web | 动作类型 × 坐标 × 参数 | A1 geometric median | +8.80 pp | [+5.99,+11.57] | Majority 更优 |
| Mind2Web | 动作类型 × 坐标 × 参数 | A2 density medoid | +7.84 pp | [+5.02,+10.72] | Majority 更优 |
| Mind2Web | 动作类型 × 坐标 × 参数 | A3 joint PKA medoid | +7.88 pp | [+5.06,+10.74] | Majority 更优 |
| Mind2Web | 动作类型 × 坐标 × 参数 | A4 continuous PKA | +18.70 pp | [+16.17,+21.31] | Majority 更优 |
| ScreenSpot-Pro | 隐含单一动作 × 二维坐标 | Official B3 | **−3.86 pp** | **[−5.84,−1.92]** | B3 更优 |
| ScreenSpot-Pro | 隐含单一动作 × 二维坐标 | A1 geometric median | −0.51 pp | [−2.83,+1.84] | 不可区分 |
| ScreenSpot-Pro | 隐含单一动作 × 二维坐标 | A2 density medoid | **−4.05 pp** | **[−6.11,−2.08]** | 预冻结反向主对照通过 |
| ScreenSpot-Pro | 隐含单一动作 × 二维坐标 | A3 joint PKA medoid | −4.05 pp | [−6.11,−2.08] | A3 更优 |
| ScreenSpot-Pro | 隐含单一动作 × 二维坐标 | A4 continuous PKA | −4.11 pp | [−6.11,−2.15] | A4 更优 |

Kill 状态：`F-K1=false`，`F-K2=false`。两个 benchmark 的预冻结主对照显著反向，新主线成立。

ScreenSpot 的 official B3 在上游 E1 文件中存为 `ours`。B3 与 A2 属于同一密度家族，但不是实现等价。

## 3. Mind2Web 动作类型分层

| GT 动作 | Rows | Majority − Sequential | 99% CI |
| --- | ---: | ---: | ---: |
| CLICK | 1,774 | **+6.26 pp** | **[+2.86,+9.55]** |
| TYPE | 227 | +0.44 pp | [−2.48,+3.64] |
| SELECT | 79 | −1.27 pp | [−5.88,0.00] |

差距主要来自 CLICK，而不是 TYPE/SELECT。因此本轮支持的是 benchmark-level 的“动作空间/错误结构—聚合器匹配”现象，不支持“参数动作导致密度聚合失效”的因果解释。可能机制包括跨类型 plurality 先过滤错误、CLICK 候选的多簇错误结构，以及 sequential complete-link 的次序敏感性；当前实验不能区分这些解释。

## 4. F2：跨聚合器 arm 一致性

F2 是明确标注的事后探索，不参与 F1 主判据。每次 bootstrap 先计算七个聚合器下的 C-cond−C-uni，再跨聚合器取平均，从而保留共享候选造成的相关性。

| Benchmark | C-cond 点估计最优 | 平均 C-cond − C-uni | 99% CI | 聚合器间效应 SD |
| --- | ---: | ---: | ---: | ---: |
| Mind2Web | 4 / 7 | +2.05 pp | [+1.23,+2.95] | 1.78 pp |
| ScreenSpot-Pro | 7 / 7 | +1.92 pp | [+0.84,+3.23] | 0.60 pp |

`F-K4=false`。允许的表述是：“arm 排序在跨聚合器的事后合并分析中一致，但单个预注册 majority 判据未通过。”该结果不能恢复四臂主张。Mind2Web 的实际重算结果是 4/7，而不是分析动机中的 5/7。

## 5. F3：AndroidControl 部分行

AndroidControl 四臂协议保持取消。F3 只使用已完成行做同池换聚合器，不计算 C-cond，也不产生任何四臂结论。

逐行完整交集为：

- Low：1,096 行；
- High：1,056 行。

现有 checkpoint 只有 stage1 单视角，因此池是 `3 models × 1 view = 3 forwards`，不是 6-forward，也不是 Mind2Web 的 12-forward 池。

| Setting | Majority | Sequential | Majority − Sequential | 99% CI | 与 Mind2Web 同向 |
| --- | ---: | ---: | ---: | ---: | --- |
| Low | 77.55% | 76.82% | +0.73 pp | [−0.64,+2.09] | 是 |
| High | 60.32% | 59.38% | +0.95 pp | [−0.10,+2.06] | 是 |

两个 setting 的方向与 Mind2Web 一致，但 CI 均跨零。此外，单模型子集分数相对历史 7,708 行全量分数的最大偏差为：

- Low：2.22 pp（UI-R1-E）；
- High：2.91 pp（UI-AGILE）。

偏差超过 2 pp，触发 `F-K3=true`。因此 AndroidControl 只能作为附录中的有限方向性证据，不能写成新主线的第三个数据点。

## 6. F0：trace 保留

在 F1–F3 前，所有已完成 AndroidControl trace 已归档。共保留 8 个 JSONL 分片、6 个 lane、10,768 条 lane 记录：

| Lane | Low | High |
| --- | ---: | ---: |
| UI-AGILE | 2,000 | 2,000 |
| GUI-R1 | 1,096 | 1,056 |
| UI-R1-E | 1,824 | 1,792 |

归档执行了逐行 JSON 解析、唯一 ID 检查、字段完整性检查、shard/lane/row-ID SHA-256、原子复制、文件和目录 fsync。独立备份位置：

`/scratch/workspaceblobstore/aggmatch-traces/2026-08-09/`

Retention manifest 共验证 25 项，包括原始分片、冻结配置、脚本、逐行派生缓存、结果 JSON、PDF、主表、报告和状态文件。`raw/`、`predictions*.jsonl` 和现有 JSONL 禁止递归清理。

## 7. 论文最终结构

### 主结果

聚合器与输出动作空间/错误结构存在经验匹配：Mind2Web 的积动作空间中 majority 更优；ScreenSpot-Pro 的纯坐标空间中离散密度族更优。主证据来自两个 benchmark 的预冻结反向配对 CI。

### 次级结果

跨谱系共识 RoI 在原密度聚合器下显著改善：

- ScreenSpot-Pro：+2.21 pp，99% CI [+0.50,+4.16]；
- Mind2Web：+4.90 pp，99% CI [+2.94,+6.86]。

但在 majority 下 C-cond 与 C-uni 不可区分。该结果只能说明候选池与聚合器存在交互，不能说明 C-cond 聚合器无关地优越。

### 机制结果

E3 的 high-start 条件保留为两 benchmark 定性机制：

- ScreenSpot-Pro：rank0 containment 99.94%，到 rank11 下降 38.90 pp；
- Mind2Web：rank0 containment 40.38%，到 rank11 仅下降 9.23 pp。

几何 rank decay 要转化为明显性能下降，需要高起点提议器。该结论只有两个数据点，不是普遍预算定律。

### 统一负结果解释

CALA 的覆盖率上升、NOA 的有效样本量上升，以及 C-cond 在 majority 下被吸收，都是同一问题的不同表现：候选池、错误结构与聚合器三者耦合，只优化其中一条边不足以保证最终精度提升。

## 8. 明确不主张

- 不主张绝对分数或 SOTA，native anchors 未重跑；
- 不主张 C-cond 的聚合器无关优越性；
- 不主张参数动作是 F1 反转的原因；
- 不主张 rank decay 是普遍预算定律；
- 不主张任意跨谱系池优于单谱系池；
- 不主张 AndroidControl 四臂或 C-cond 结果；
- 不把 F2 事后结果写成预注册主张；
- 不把 F3 有偏部分子集写成第三个主证据点。

## 9. Kill 状态

| ID | 状态 | 后果 |
| --- | --- | --- |
| F-K1 | false | 新论文主线保留 |
| F-K2 | false | 两 benchmark 反向结论保留 |
| F-K3 | **true** | AndroidControl 降为附录 |
| F-K4 | false | 仅允许跨聚合器事后一致性表述 |

## 10. 关键文件与复现

- 主表：`MAIN_TABLE.md`
- 完整报告：`REPORT.md`
- 最终状态：`STATUS.json`
- F0：`f0_ac_archive.json`
- F1：`f1_aggregator_matching.json`
- F2：`f2_arm_consistency.json`
- F3：`f3_androidcontrol_aggregator.json`
- 图：`fig_aggregator_matching.pdf`

```bash
.venv-scaleup/bin/python runs/aggmatch/2026-08-09/f0_ac_archive.py
.venv-scaleup/bin/python runs/aggmatch/2026-08-09/f1_aggregator_matching.py
.venv-scaleup/bin/python runs/aggmatch/2026-08-09/f2_arm_consistency.py
.venv-scaleup/bin/python runs/aggmatch/2026-08-09/f3_androidcontrol_aggregator.py
```

全程未启动新推理、未恢复 AndroidControl worker、未触碰外部 PID 2274。

冻结配置 commits：`4827afc`、`d98c0ae`。最终 aggmatch 发布 commit：`105b7abe419501f84243d7fa1fbc429df3d6fab5`。