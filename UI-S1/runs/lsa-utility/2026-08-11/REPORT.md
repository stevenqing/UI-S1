# Utility-Aligned Aggregator Training Report

日期：2026-08-11

上游：CEV、LSA、LSA cross-arm confirmation。

计算约束：CPU、零 GPU、零新 VLM inference。

## 1. 为什么重做训练信号

上一轮 correctness-LSA 的 candidate AUROC 很高，但 direct top-1 比 CEV-A 差。这说明训练“候选是否正确”没有对齐部署问题：是否值得推翻一个已经很强的 CEV-A 决策。

Utility-LSA 改为训练候选相对 exact cross-fitted CEV-A 的净效用：修复 baseline 为 +1，不改变为 0，破坏 baseline 为 −1。安全策略只在 learned score 为正且相对 fallback score margin 超过 OOF threshold 时 override。

## 2. 从 GTA1 和 MVP 借鉴了什么

GTA1 官方 grounding GRPO 使用 8 个 generation、click-in-bbox binary reward、无 KL 的公开 recipe，以及 sample-standardized group advantage。Utility-LSA 使用同样的组内相对思想，但它是固定 heterogeneous candidate bank 上的 offline utility regression，不是 on-policy GRPO，也不更新 VLM。

MVP 主方法是 training-free complete-link clustering；其 appendix 还训练 Qwen3VL-4B numbered-point GRPO selector。该 selector 在 GTA1 候选上 60.5→62.8，超过 clustering 61.7；在 Qwen3VL-8B 候选上 62.7→65.3，略低于 clustering 65.5，因此不稳定。Utility-LSA 针对这一问题，不再优化 absolute correct label，而优化相对强 clustering/CEV fallback 的净收益。

MVP 的结构支持进入特征：多尺度坐标支持、medoid 距离、lineage support 与 dispersion。AGVP coverage 在 Mind2Web 没有等价量，因此没有伪造。

## 3. 防泄漏训练协议

Utility-LSA 外层同时封存两 benchmark 的同一 fold，并将四 arms 的同一 underlying row 保持在同一 fold。

第二层 OOF 还嵌套拟合 CEV behavior policy：每个 inner holdout 的 CEV 配置与 reliability 只由另外三折产生。最终 outer test 使用冻结上游 outer-fold CEV global config，并在其余四折 refit reliability；八个 cell 均逐行匹配冻结 CEV correctness。

每个 underlying row 具有固定总权重，按 active nonconstant-utility arms 平分，再按 12 candidates 平分。常量 utility 组没有相对 advantage，不训练但保留评测。

## 4. Reward signal audit

每个池有约 56%–75% 行具备非零组内 utility 方差。Mind2Web 的 baseline-correct-at-risk / repairable 比约 1.1；ScreenSpot-Pro 约 3.0–3.3。后者意味着 ScreenSpot 上错误 override 的机会约为可修复机会的三倍，统一策略必须显式建模 downside 并使用安全阈值。

## 5. Main results

五折选择 U-GRPO 三次、U-HYBRID 两次，全部选择 H3 和有限阈值。GTA-style relative target 因而不是名义装饰；它在 inner OOF 最终效用目标下被实际选择。UR-K3/UR-K4 不触发。

Utility-LSA 在八个 cell 均安全，UR1 通过。ScreenSpot 四臂平均相对 CEV-A +0.25 pp，99% CI [+0.04,+0.47]，主要来自 C-rand +0.95 pp。Mind2Web 四臂平均 −0.02 pp，CI [−0.28,+0.23]，没有 robust gain，UR2 失败。

相对 nested dev-selection 的 equal-benchmark/equal-arm standardized CI [+0.22,+1.70] 为正，UR4 通过。但相对 correctness-LSA 的 standardized CI [−1.12,−0.07] 为负，UR5 失败：utility target 把收益从 Mind2Web 移向更高风险的 ScreenSpot，却没有得到更好的跨 benchmark 总体策略。

## 6. Interpretation

训练信号确实被改善了，但“改善”是目标对齐而非全面提分：

- GTA-style group advantage 被 3/5 折选中；
- 八个 cell 安全；
- ScreenSpot aggregate 得到小但显著增益；
- Mind2Web 不退化但也不增益；
- 相对 correctness-LSA 的总体表现更差。

这表明一个统一 utility reward 仍受 benchmark 风险分布影响。ScreenSpot 的 downside/repairable 比约三倍，需要比 Mind2Web 更保守的 override；单一 global threshold 只能折中。

## 7. Current decision

当前结果为 `SAFE_EXPLORATORY_OVERRIDE`，不是新的最好聚合器。CEV-A 仍是论文中最强可辩护的统一规则。Utility-LSA 可以作为 learned training-signal 研究：它验证了 GTA-style relative advantage 和 MVP structure 能构造安全 override，但没有通过 UR2/UR5。

## 8. Fixed ablations and UR-K5

两个 fully nested 消融已完成：

- no-MVP-structure：相对主方法，Mind2Web `main−ablation` −0.11 pp，99% CI [−0.38,+0.17]；ScreenSpot +0.25 pp，[+0.05,+0.48]。no-MVP 在 ScreenSpot 显著更差，因此 UR-K5 的 “no-MVP not worse” 前提为 false，UR-K5=false。该 gate 是合取条件，不再运行无必要的 structure permutation importance。
- absolute-only：Mind2Web `main−ablation` −0.61 pp，[−1.33,+0.10]；ScreenSpot −0.16 pp，[−0.41,+0.08]。equal-benchmark standardized CI [−1.25,−0.003] MDE，absolute-only 整体优于 pair transform。强 fallback-pair 扩展并非主增益来源，可能对当前小 HGB 增加方差。

消融不改变主 UR2/UR5 失败与 `SAFE_EXPLORATORY_OVERRIDE` 结论。后续 VUS-SR 使用视觉 listwise set encoder 和显式 downside state，已显著超过 Utility-LSA。
