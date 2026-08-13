# Post-B2 支配差、R7 排错与写作冻结总报告

日期：2026-08-06
状态：`COMPLETE_PATH_B_WITH_D2_BLOCKED`
性质：零 GPU、不开新方法、路径 B 诊断论文

## 0. 最终结论

本轮完成了三件事：

1. **D0 排错**：确认 R7 的历史 2% 准确率来自实现故障；修复后 R7 恢复正常，但不改变 B2 nested selection 与最终 gate。
2. **D1 支配差检验**：ScreenSpot-Pro 上存在稳定负相关方向，但强度没有达到预注册的 `rho < -0.6`；Mind2Web 与 AndroidControl 又缺逐行 trace，因此不能把它写成“支配差定律”。
3. **D2 跨 benchmark 迁移**：冻结的 `rows.parquet` 与源 predictions 缺失，只能保留成员质量 anchor，不能计算 mixed accuracy、failure kappa 或迁移方向。

最终论文形状保持为：

> **ScreenSpot-Pro 路径 B 诊断论文。** 跨谱系分配改善预算曲线，但异质候选池会产生严重的来源敏感聚合失败；lineage normalization 能修复 72B 崩塌，却不超过最强单模型，也不改善 7B，因此不是跨尺度通用方法。

## 1. 上游 B2 状态

| 项目 | 结果 |
|---|---:|
| 7B nested LN | 63.69% |
| 7B B3 | 63.69% |
| 7B delta | 0.00 pp |
| 7B 99% CI | [-0.54, +0.65] pp |
| 7B p | 0.5626 |
| 72B nested LN | 70.52% |
| 72B B3 | 41.24% |
| 72B LN minus B3 | +29.29 pp |
| Qwen3.5 reported best-single | 71.41% |
| 72B LN minus best-single | -0.89 pp |

两个限定必须保留：

- 72B 的 `+29.29 pp` 是恢复 B3 在异质池中的聚合崩塌，不是超过最强单模型的新收益。
- B2 跨尺度失败。7B 完全没有改善，因此不能声称存在通用聚合修法。

B4 也没有支持共享 proposer 强归因。正式机制标签是：

`heterogeneous_pool_aggregation_effect`

## 2. D0：R7 实现故障

裁决：`IMPLEMENTATION_FAULT_CONFIRMED`

### 2.1 根因

R7 的 weighted centroid 历史实现计算了：

$$
\sum_i w_i x_i
$$

但遗漏了标准化分母：

$$
\frac{\sum_i w_i x_i}{\sum_i w_i}
$$

对于点 `(0,0)`、`(10,0)` 和权重 `(2,1)`：

- 错误输出：`(10,0)`；
- 正确输出：`(3.33,0)`。

这会按总权重整体放大坐标，解释了 7B 与 72B 同时落到约 2% 的异常形状。

### 2.2 修复前后

| 尺度 | 历史 R7 D1/D2/D3 | 修复后 R7 D1/D2/D3 |
|---|---|---|
| 7B | 2.28 / 2.15 / 2.21% | 62.62 / 61.99 / 61.80% |
| 72B | 2.66 / 2.40 / 2.34% | 51.61 / 64.64 / 64.83% |

### 2.3 是否污染 B2 headline

在同一 recovered bank 上进行了受控比较：

- 修复后的 frozen 21-method grid；
- 相同 grid，但移除 R7，剩余 18 methods。

结果：

- 两个尺度的 nested selections 完全相同；
- 两个尺度的 nested predictions 完全相同；
- R7 从未被选中；
- `D-K1=false`；
- B2 gate 不改变。

必须并列保留以下来源：

| 实验 | 7B | 72B |
|---|---:|---:|
| 历史坏掉的 21-method | 61.99% | 70.59% |
| 修复后的 21-method recovery | 61.99% | 70.52% |
| Combined-24 recovery | 63.69% | 70.52% |

它们不是同一方法集、同一字节 bank 下的重复实验。

产物：

- [d0_r7_audit.py](d0_r7_audit.py)
- [d0_r7_audit.json](d0_r7_audit.json)

## 3. D1：支配差检验

### 3.1 假设

假设为：

> 池内最强模型与次强模型的支配差越大，混合池相对 best-single 的收益越低。

为避免把不可交换指标混在一起，统计分析没有把“7B B3 63.69%”与“72B nested LN 70.52%”直接放进一条相关曲线。

### 3.2 ScreenSpot 池枚举

冻结 3 个模型、每个模型 views 0--11，共 36 actions。每个池对每条保留谱系只取一个 action：

| Pool 类型 | 数量 |
|---|---:|
| 双谱系 | 432 |
| 三谱系 | 1,728 |
| 总计 | 2,160 |

每个池均在 1,581 个 ScreenSpot-Pro 样本上计算：

- best member；
- second-best member；
- dominance gap；
- mean member quality；
- mean pairwise failure kappa；
- frozen B3；
- fold-local M1；
- B3/M1 minus best member。

### 3.3 统计结果

| Outcome | Raw Spearman | Raw 99% CI | 控制后 rho | 控制后 99% CI |
|---|---:|---:|---:|---:|
| B3 minus best | -0.388 | [-0.430, -0.347] | -0.367 | [-0.410, -0.323] |
| M1 minus best | -0.482 | [-0.530, -0.434] | -0.499 | [-0.547, -0.450] |

控制变量：

- 池内平均成员质量；
- 池内平均 failure kappa。

置信区间使用 10,000 次按 pool size 分层的 pool bootstrap，seed `20260806`。

### 3.4 判定

方向稳定为负，控制混淆后仍为负，但：

- B3 raw rho 没达到 `-0.6`；
- M1 raw rho 没达到 `-0.6`；
- Mind2Web 与 AndroidControl 的 mixed 指标缺失；
- 三 benchmark 方向一致性无法判定。

所以最终状态为：

`INCONCLUSIVE_BLOCKED_CROSS_BENCHMARK_ROWS`

论文动作：

- 不称为“支配差定律”；
- 不把 72B 失败改写成定律确认；
- 7B/72B 分裂保留为未解释的尺度依赖；
- 可以在附录报告 ScreenSpot 的方向性负相关。

产物：

- [d1_dominance_law.py](d1_dominance_law.py)
- [d1_dominance_law.json](d1_dominance_law.json)
- [fig_dominance.pdf](fig_dominance.pdf)

## 4. D2：跨 Benchmark 迁移

### 4.1 阻塞原因

冻结 runner 需要：

`runs/complementarity/2026-07-30/rows.parquet`

Manifest 记录该文件应包含 102,054 个 tidy rows，但文件不存在。尝试从锁定 summary 重建时，第一处缺失源文件为：

`runs/androidcontrol-rft/2026-07-29/artifacts/ui-agile-3b/low/predictions.jsonl`

Workspace-wide 搜索确认：

- `rows.parquet` 不存在；
- AndroidControl lane predictions 不存在；
- Mind2Web lane predictions 不存在；
- lane 目录只保留 aggregate `score.json` 与 `audit.json`。

Aggregate 文件不能恢复：

- joint correctness；
- candidate coordinates；
- mixed-pool predictions；
- pairwise failure kappa；
- paired transfer statistics。

### 4.2 可保留的质量 Anchor

| Pool | Dominance gap | Mean member quality |
|---|---:|---:|
| Mind2Web M-cross-3 | 2.79 pp | 45.58% |
| Mind2Web M-same-3 | 0.91 pp | 51.31% |
| AndroidControl Low A-cross-2 | 19.34 pp | 67.88% |
| A-same-2-agile | 1.53 pp | 78.32% |
| A-same-2-gui | 1.32 pp | 57.55% |

这些只能作为 preflight 质量 anchor，不能附加 mixed accuracy 或迁移方向。

### 4.3 判定

- D2：`BLOCKED_MISSING_ROW_LEVEL_TRACES`；
- D-K3：`NOT_ADJUDICATED`；
- 不主张 Mind2Web/AndroidControl transfer；
- 分配与支配差结论限定在 ScreenSpot-Pro。

产物：

- [d2_cross_benchmark_audit.py](d2_cross_benchmark_audit.py)
- [d2_cross_benchmark_status.json](d2_cross_benchmark_status.json)

## 5. 冻结的三条主张

### 5.1 预算曲线符号翻转

- 单谱系 slope：`-0.002467/forward`；
- 99% CI：`[-0.004908, -0.000124]`；
- 跨谱系 slope：`+0.003052/forward`；
- 99% CI：`[+0.001082, +0.005053]`；
- 两个区间不重叠，且分居零两侧；
- N=16 时 Mixed 相对 V-only：B3 `+5.44 pp`，M1 `+5.50 pp`。

该主张限定为 ScreenSpot-Pro。

### 5.2 弱模型结果

- UI-TARS-7B-SFT bare：`33.46%`；
- 比 GTA1 低 `15.94 pp`；
- 准入在结果前冻结；
- Mixed N12 达到 B3/M1 `63.69% / 63.82%`。

正确表述是弱 lineage 可以与更强 mixed pool 共存，不是任意弱模型都会带来提升。

### 5.3 双尺度来源偏置

| 尺度 | 错误行 | GTA observed | Expected | Residual | p | Cramer's V |
|---|---:|---:|---:|---:|---:|---:|
| 7B | 574 | 489 | 191.33 | +26.36 | `4.12e-152` | 0.779 |
| 72B recovery | 929 | 871 | 348.38 | +35.42 | `1.21e-273` | 0.822 |

B1 双尺度通过，但 B4 不支持共享 proposer 强归因。

## 6. 明确不主张

不得主张：

1. **绝对分数优势**：Qwen3.5 bare `71.41%` 高于 72B LN `70.52%`。
2. **选择规则优势**：CCM 在相关预算仅增加约 `0.13--0.19 pp`，M1-minus-B3 最大不超过 `1.51 pp`。
3. **跨尺度通用聚合修法**：B2 cross-scale FAIL。
4. **共享 proposer 因果归因**：B4 `NOT_SUPPORTED`。
5. **计数平衡是通用有效方法**：确定性 balanced accuracy `49.72%` 位于随机平衡 99% accuracy 区间 `[23.34%, 55.22%]` 内；随机均值 `41.03%`，SD `11.36 pp`。
6. **支配差定律**：D1 未达到冻结门槛。
7. **跨 benchmark transfer**：D2 被逐行资产阻塞。
8. **UI-Zoomer / GUI-RC 头对头优势**：这些本地 runs 不存在。

注意单位：确定性 `+8.48 pp` 是 delta，不能直接与 accuracy 区间比较。相对原 B3 `41.24%`，随机 balanced-accuracy 区间约对应 delta `[-17.90, +13.98] pp`。

## 7. 可复现性声明

论文必须明确：

1. recovery bank 与 frozen bank 非字节一致，状态为 `COMPLETE_WITH_RECOVERY_DRIFT`；
2. 72B M1 从 `52.12%` 漂到 `53.19%`；
3. B1 winning-set 从 `1374/1000/370` 变为 `1370/1003/369`；
4. `stata_windows_27` 只产生 7 个唯一 crop，P1 退到 N8，因此 P1/P2 只能作不等预算参照；
5. historical 21-method 与 combined-24 recovery 使用不同方法集；
6. R7 历史实现有故障，但修复不改变 recovered-bank nested selection。

## 8. 其它冻结口径

- X2 从结果章节删除，只在 limitation 留一句“未能复现”。
- AndroidControl 主 MDE 使用 v1-only 的 `0.09--1.16 pp`；五视角版进入附录并标注不可交换。
- M0：净 `+1`，实际包含 5 次翻转，即 3 rescue、2 regression。
- M0 canonical drop-in：`+3.60 pp`，99% CI `[+1.31, +6.22]`；CCM attribution `+0.13 pp`。
- R4 只写 cross-lineage strengthening：AUROC `0.744 -> 0.830`，matched 80% coverage 下 Mixed B3 领先 `7.12 pp`。
- R4 deterministic N12 不继承 SafeGround K=10 随机协议和 FDR guarantee。
- 纸面数字 `62.8`、`70.4`、独立口径 `71.41`、`73.1`、`+13.4%`、`+5.38` 不参与本地差值计算。

## 9. Kill Conditions

| Kill condition | 状态 | 后果 |
|---|---|---|
| D-K1：R7 修复改变 B2 结论 | FALSE | 保留 B2 gate，并列报告旧/新 R7 |
| D-K2：控制后支配差不显著 | ScreenSpot 未触发；combined blocked | 不写定律，因为强度和 benchmark coverage 均未过门槛 |
| D-K3：Mind2Web 方向与 ScreenSpot 相反 | NOT ADJUDICATED | 主张限定 ScreenSpot-Pro |

## 10. 最终论文定位

正文结构：

1. 预算曲线符号翻转；
2. 弱模型冻结准入；
3. 七条带机制的负结果；
4. B1 双尺度来源偏置；
5. 72B 强修复、7B 无改善的 B2 定位；
6. D1 方向性结果进入附录，不称定律；
7. D2 缺失逐行资产进入 limitation。

最终表述：

> 跨谱系分配改善了 ScreenSpot-Pro 的预算扩展趋势，但异质候选池暴露出严重的来源敏感聚合失败。Lineage normalization 能恢复 72B 的聚合崩塌，却没有超过最强单模型，也没有改善 7B，因此该修复具有尺度依赖性，而非通用规律。

## 11. 交付物

- [SPEC.md](SPEC.md)：Post-B2 冻结协议
- [FREEZE.md](FREEZE.md)：写作阶段唯一依据
- [REPORT.md](REPORT.md)：英文执行报告
- [STATUS.json](STATUS.json)：机器可读状态
- [d0_r7_audit.json](d0_r7_audit.json)：R7 排错结果
- [d1_dominance_law.json](d1_dominance_law.json)：2,160 池完整统计
- [fig_dominance.pdf](fig_dominance.pdf)：支配差图
- [d2_cross_benchmark_status.json](d2_cross_benchmark_status.json)：D2 阻塞状态

## 12. 验证

- 三个 Python 脚本通过 `py_compile`；
- D0/D1/D2 与 `STATUS.json` 交叉校验通过；
- `git diff --check` 通过；
- VS Code diagnostics 无错误；
- PDF 为有效单页文档；
- 全部任务为 CPU-only；
- 无残留 dominance 或 GPU scoring 进程；
- 外部 PID `2274` 正常且未被操作。
