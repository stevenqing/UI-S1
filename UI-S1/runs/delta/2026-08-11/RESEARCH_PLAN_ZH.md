# DELTA：分解证据的决策层后融合

## 为什么继续

RAVEL 并没有证明局部像素无用：local 相对 random-center 的 utility AUROC 在两个 benchmark 都高约 +0.046。它证明的是把 global/fine/context 放入一个固定-token prompt 会互相竞争，尤其破坏 Mind2Web 全局语义。

现有 blind channels 已经完整且独立锁定，不需要新 VLM inference。DELTA 先回答一个最小问题：在候选决策层融合这些 logits，能否同时保留 VUS 的 binding、global-only 的语义和 local channels 的局部外观？

## 必须区分的贡献

- 如果 full late fusion 不优于同容量 VUS-only，增益只是模型容量，停止。
- 如果 random channel 同样有效，增益只是多通道正则化，停止。
- 只有真实 global/fine/context channels 提供额外 held-out utility，才能称 evidence complementarity。

## 部署边界

DELTA 首先是四次独立视觉编码的 research oracle，不是部署方法。它通过后还必须蒸馏成一次 selector invocation，并在 GUI-Odyssey app-split 一次性确认。当前 GUI-Odyssey 数据未挂载，因此不能提前做最终确认。

## 正式结果与停止决定

DELTA 已按冻结协议完成五折。FULL 相对 VUS-SR 在 Mind2Web 为 -0.41 pp，99% CI `[-1.20,+0.40]`；ScreenSpot-Pro 为 +0.11 pp，`[-0.20,+0.41]`。DELTA-1/3/4/5 失败，DELTA-2/6 通过，结论为 `DELTA_NOT_SUPPORTED`。

强制对照定位了失败机制：FULL 无法显著超过 VUS_ONLY 或 RANDOM_PLACEBO，并显著差于 VUS_GLOBAL；冻结模型中移除 fine/context 会使 Mind2Web 分别上升 +0.83/+0.56 pp。当前 shared simplex gate 没有学会保留 global/binding utility 并拒绝有害 local evidence。

因此不运行 distillation、GUI-Odyssey confirmation 或任何 post-result mask/loss/threshold 调整。VUS_GLOBAL 仅是诊断 control，不事后晋升为方法。
