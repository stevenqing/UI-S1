# LSA No-Action Cross-Arm Confirmation Report

日期：2026-08-10

发现轮：`runs/lsa/2026-08-10/`

计算约束：CPU、零 GPU、零新模型推理。

## 1. 结论

No-action learned aggregator 在未用于发现的三个候选池上实现了**安全但不显著的部分迁移**。

六个 arm×benchmark 单元全部通过非劣安全门。Mind2Web 的 C-cond/C-rand/C-self 分别提高 +0.96、+0.34、+0.67 pp，但各自 CI 均跨零；三臂平均 +0.66 pp，99% CI [−0.02,+1.32]，下界略低于零，因此 T2 失败、LT-K3 触发。ScreenSpot 三臂平均 +0.02 pp，CI [−0.09,+0.14]，保持中性。

相对 nested dev-selection，双 benchmark × 三 arm 的标准化平均增益 CI [+0.57,+2.58] 为正，T4 通过。但 CEV-A 是更强且更直接的基线；相对 CEV-A 的 Mind2Web transfer 未达到 99% 显著，所以不能声称 learned aggregator 是新的最好规则。

最终定位：`PARTIAL_TRANSFER`。当前最强可辩护聚合方案仍是 CEV-A/nested endpoint selection。LSA-no-action 是安全 learned wrapper 和未来数据上的候选方法，不进入当前论文主方法。

## 2. Confirmatory separation

发现轮只在 C-uni 上观察 no-action 消融。本轮在结果前以 commit `cdb7f26` 冻结：

- H3 每折固定；
- 五个 C-uni OOF threshold 固定；
- learned weights 只训练于 C-uni outer-dev；
- C-cond/C-rand/C-self 标签不训练、不调 threshold；
- learned reliability 对未见 source 只使用 C-uni train lineage-average；
- 每臂冻结 CEV-A fallback 按原协议重建并逐行断言。

三臂全部报告，没有根据结果挑选候选池。

## 3. Results

Mind2Web safe accuracy：C-cond 33.41%、C-rand 32.16%、C-self 32.07%。对应 frozen CEV-A 为 32.45%、31.83%、31.39%。改进方向三臂一致，但统计证据不足。

ScreenSpot safe accuracy：C-cond 66.60%、C-rand 61.10%、C-self 65.09%。与 CEV-A 几乎相同。C-cond 仅增加两个净成功左右，C-rand 完全相同，C-self 净损一行。

这说明 no-action 模型的安全阈值能跨候选池保持风险，但在 C-uni 上的 +1.73 pp 发现效应没有以同等强度复现。

## 4. Interpretation

训练确实学到了一些可迁移结构信号：三个 Mind2Web arm 点估计都为正，并且相对较弱的 dev-selection 基线整体显著。但强 CEV-A 已经吸收大部分可利用的候选集合结构，剩余 override headroom 小、wins/losses 方差大。

直接 learned top-1 在所有池上仍明显低于 safe wrapper，进一步说明训练模型适合做保守 override，而不是替代强聚合器。

## 5. Decision

停止在当前数据上继续搜索 learned architecture 或 feature subset。继续调参会重复使用同一 3,661 行和多 arm 共享候选，无法提供独立确认。

只有获得新 benchmark、新模型 bank 或未使用的新 trace 后，才值得预注册验证 LSA-no-action。当前论文保留：

1. F1 主结果；
2. CEV-A 统一解释；
3. LSA/no-action 作为附录中的安全但未确认 learned attempt。

外部 PID 2274 未触碰，无 inference worker 启动。
