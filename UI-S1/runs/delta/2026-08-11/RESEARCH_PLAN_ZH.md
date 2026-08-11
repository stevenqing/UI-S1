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
