# Aggregator–Action-Space Matching 主表

效应统一定义为 `majority - density`，单位为百分点（pp）；区间为 10,000 次配对分层 bootstrap 的 99% percentile CI。主判据只使用 C-uni。

## F1 主结果

| Benchmark | 动作空间 | 对照聚合器 | Majority − Density | 99% CI | 判定 |
| --- | --- | --- | ---: | ---: | --- |
| Mind2Web | 动作类型 × 坐标 × 参数 | Sequential | **+5.34** | **[+2.50, +8.04]** | 高于 0 且超过 MDE 0.61 pp |
| Mind2Web | 动作类型 × 坐标 × 参数 | A1 geometric median | +8.80 | [+5.99, +11.57] | 正向 |
| Mind2Web | 动作类型 × 坐标 × 参数 | A2 density medoid | +7.84 | [+5.02, +10.72] | 正向 |
| Mind2Web | 动作类型 × 坐标 × 参数 | A3 joint PKA medoid | +7.88 | [+5.06, +10.74] | 正向 |
| Mind2Web | 动作类型 × 坐标 × 参数 | A4 continuous PKA | +18.70 | [+16.17, +21.31] | 正向 |
| ScreenSpot-Pro | 隐含单一动作 × 二维坐标 | Official B3 | **−3.86** | **[−5.84, −1.92]** | 密度族更优 |
| ScreenSpot-Pro | 隐含单一动作 × 二维坐标 | A1 geometric median | −0.51 | [−2.83, +1.84] | 不可区分 |
| ScreenSpot-Pro | 隐含单一动作 × 二维坐标 | A2 density medoid | **−4.05** | **[−6.11, −2.08]** | 预冻结反向主对照通过 |
| ScreenSpot-Pro | 隐含单一动作 × 二维坐标 | A3 joint PKA medoid | −4.05 | [−6.11, −2.08] | 密度族更优 |
| ScreenSpot-Pro | 隐含单一动作 × 二维坐标 | A4 continuous PKA | −4.11 | [−6.11, −2.15] | 密度族更优 |

`F-K1=false`，`F-K2=false`。新主结果通过：两个 benchmark 的预冻结主对照显著反向。B3 在 E1 文件中存为 `ours`，与 A2 仅属同一家族，不是实现等价。

## Mind2Web 动作分层

| GT 动作 | Rows | Majority − Sequential | 99% CI |
| --- | ---: | ---: | ---: |
| CLICK | 1,774 | **+6.26** | **[+2.86, +9.55]** |
| TYPE | 227 | +0.44 | [−2.48, +3.64] |
| SELECT | 79 | −1.27 | [−5.88, 0.00] |

差距主要来自 CLICK，而不是带参数动作。因此主表支持 benchmark-level 的聚合器匹配现象，但不支持“参数维度导致密度聚合失效”的因果解释。

## F2 事后分析

| Benchmark | C-cond 最优聚合器数 | 跨七聚合器平均 C-cond − C-uni | 99% CI | 聚合器间效应 SD |
| --- | ---: | ---: | ---: | ---: |
| Mind2Web | 4 / 7 | +2.05 | [+1.23, +2.95] | 1.78 |
| ScreenSpot-Pro | 7 / 7 | +1.92 | [+0.84, +3.23] | 0.60 |

`F-K4=false`。本节是事后探索，不用于恢复预注册的四臂主张；Mind2Web 的重算结果是 4/7，而不是分析动机中的 5/7。

## F3 AndroidControl 附录结果

池均为 `3 models × 1 view × stage1 = 3 forwards`，不是 Mind2Web 的 12-forward 池。

| Setting | 完整交集 | Majority | Sequential | Majority − Sequential | 99% CI | 与 Mind2Web 同向 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Low | 1,096 | 77.55 | 76.82 | +0.73 | [−0.64, +2.09] | 是 |
| High | 1,056 | 60.32 | 59.38 | +0.95 | [−0.10, +2.06] | 是 |

两个 setting 的方向一致但 CI 均跨零。单模子集相对历史全量的最大偏差为 Low 2.22 pp、High 2.91 pp，触发 `F-K3`，因此 F3 仅进入附录，不构成第三个主证据点。AndroidControl 四臂状态保持 `CANCELLED`，本节没有任何 C-cond 结论。

## Kill 状态

| ID | 状态 | 后果 |
| --- | --- | --- |
| F-K1 | false | 新主线保留 |
| F-K2 | false | 两 benchmark 反向结论保留 |
| F-K3 | **true** | AndroidControl 降为附录 |
| F-K4 | false | 仅允许“跨聚合器事后一致性”表述 |
