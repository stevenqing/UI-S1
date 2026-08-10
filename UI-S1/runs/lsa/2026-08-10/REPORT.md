# Learned Structural Aggregator Report

日期：2026-08-10

上游：`runs/cev/2026-08-09/`

计算约束：CPU、零 GPU、零新模型推理；只使用冻结 C-uni 12-forward candidate banks。

## 1. 结论

主 LSA-pooled-safe 是安全但未显著提升的 learned aggregator。它把 Mind2Web 从 CEV-A 32.02% 提到 33.03%（+1.01 pp），但 99% CI [−0.81,+2.83] 跨零；ScreenSpot-Pro 从 63.88% 降到 63.63%（−0.25 pp），损失小于 MDE 0.70 pp。L1 通过，L2/L3 未通过，论文定位为 `SAFE_BUT_NO_SIGNIFICANT_GAIN`。

风险门控是必要的。直接选择 learned top-1 在 Mind2Web 只有 28.41%，相对 CEV-A −3.61 pp，CI [−6.21,−0.88]；ScreenSpot 为 62.68%。安全回退相对 direct 在 Mind2Web 显著恢复 +4.62 pp。

一个预注册消融产生了明确的新假设：去掉 action 特征后，Mind2Web 达 33.75%，相对 CEV-A +1.73 pp，CI [+0.19,+3.32]；ScreenSpot 63.82%，仅 −0.06 pp且不显著。它相对 nested dev-selection 的 benchmark-balanced 标准化增益 CI 也为正。但 `no-action` 不是预注册主模型，不能事后升级为确认性“最好聚合器”；它需要在未用于发现的候选池上冻结确认。

## 2. 精确 headroom

| Benchmark | Rows | Mixed-label rows | All-negative rows | Oracle pass@12 | CEV-A |
| --- | ---: | ---: | ---: | ---: | ---: |
| Mind2Web | 2,080 | 1,168 | 891 | 57.16% | 32.02% |
| ScreenSpot-Pro | 1,581 | 928 | 329 | 79.19% | 63.88% |

候选池仍有明显 oracle headroom，因此失败不来自“池内没有正确答案”。

## 3. 方法

主模型是 sklearn `HistGradientBoostingClassifier`。它对每个真实候选预测 evaluator success 概率。训练只使用同时包含正确/错误候选的行，每行正负候选各占 0.5 权重，两 benchmark 总训练质量相等。

主特征不包含 benchmark ID、model/source ID、slot index、instruction/raw response、截图 embedding 或 test GT。允许特征包括 parse/action/parameter/stage、cross-fitted source reliability，以及候选在集合内的动作支持、坐标距离、多尺度邻域、谱系支持、参数一致性和行级 dispersion。

最终输出不是 learned top-1，而是安全 override：只有 learned winner 相对冻结 CEV-A winner 的概率差超过 inner-OOF 选择的统一阈值时才覆盖，否则回退 CEV-A。

每个 outer fold 同时封存两个 benchmark 的同编号 fold。在剩余四折上再做四折 OOF，选择 H1–H4 与单一全局 threshold，之后用全部 outer-dev 重训并只评一次 outer test。

## 4. Main results

### 4.1 Safety and significance

- L1：两 benchmark 非劣均通过。
- L2：失败；没有 benchmark 的主模型 99% CI 下界高于零。
- L3 strong：失败；相对 dev-selection 均未显著。
- L3 safe balanced：失败；标准化平均增益点估计 0.86 MDE，CI [−0.88,+2.54]。
- L4：通过；safe point estimate 在两端均不低于 direct。

五折都选择有限阈值，LSA-K3=false。Benchmark-specific 模型在 Mind2Web 完全回退 CEV-A，在 ScreenSpot 略差，因此 pooled 信号不是简单的 benchmark 内拟合优势；LSA-K4=false。

### 4.2 Override behavior

Mind2Web safe 相对 CEV-A 有 94 wins / 73 losses，净增 21 行。ScreenSpot 只有 1 win / 5 losses，净损 4 行。阈值跨折变化较大，且最后一折在 test 上没有 override，说明 override 信号仍不稳定。

Candidate AUROC fold 均值约为 Mind2Web 0.745、ScreenSpot 0.828，但 direct selection 更差。这说明候选级校准质量不能替代最终 decision-level paired evaluation。

## 5. Feature evidence

主模型的 permutation candidate-AUROC drop 中，geometry 最大：Mind2Web 约 0.210，ScreenSpot 约 0.317；action 次之，reliability 较小，parameter 接近零。

- No geometry：Mind2Web 31.88%，增益完全消失。
- Reliability only：Mind2Web 32.12%，仅 +0.10 pp。
- No parameter：与主模型接近。
- No action：Mind2Web 显著提升且 ScreenSpot 基本持平。

为什么 no-action 可能更好：ScreenSpot 全部候选是 POINT，Mind2Web 是 CLICK/TYPE/SELECT，动作 one-hot 实际上是强 benchmark proxy。主 pooled 模型可能先按 benchmark 分流，而不是学习跨 benchmark 的共同集合结构。去掉 action 后，模型被迫依靠 geometry/support/reliability，反而更接近“统一结构聚合器”的目标。

该解释是结果后机制假设，不是预注册主张。

## 6. Kill conditions

- LSA-K1=false：两端都有混合标签行，bank 均为 12 candidates。
- LSA-K2=false：主模型相对 CEV-A 安全非劣。
- LSA-K3=false：0/5 折选择 infinity threshold。
- LSA-K4=false：within-safe 没有优于 pooled，且 pooled 通过 L1。
- LSA-K5 不适用：主模型禁止 source ID，且没有确认性显著增益需要做 source-ID 依赖审计。

## 7. 论文与下一步

当前论文不应把主 LSA 写成新方法贡献。F1 与 CEV-A 定位不变。

唯一值得继续的学习方向是 `LSA-no-action-safe`，并且下一轮不能复用 C-uni 来确认。应冻结主模型/特征/阈值选择流程，然后训练仅基于 C-uni outer-dev，迁移评估到尚未跑 learned selector 的 C-cond、C-rand、C-self 候选池。三臂全部报告，禁止根据结果挑 arm。

如果跨臂确认失败，则 no-action 只保留为探索性消融，停止 learned aggregator 线。若确认成功，再讨论独立新 benchmark 或新 trace 的最终验证。

外部 PID 2274 未触碰，无 GPU/model inference worker 启动。
