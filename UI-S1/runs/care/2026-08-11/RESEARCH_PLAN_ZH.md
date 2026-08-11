# CARE 下一代聚合方法：研究计划

## 结论先行

下一步不应继续扩大现有 set-ranker，也不应重复普通 verifier。最有研究价值的方向是把聚合改写成**固定预算下的序贯证据决策**：

1. 前 6 个共享 candidates 后，学习剩余 6 forwards 应采用哪种 acquisition policy；
2. 用全局截图 + 候选局部多尺度 mosaics 恢复被全图缩放丢失的细粒度证据；
3. 不做独立 YES/NO 过滤，而直接学习 challenger 相对 CEV-A 的 REPAIR/SAME/BREAK；
4. 用 one-sided utility risk control 决定是否 override；
5. 最后在第三个 untouched benchmark 一次性确认。

## 为什么这是第一性原理方案

最终成功率由三个不同问题相乘/串联决定：

- acquisition 是否产生了正确候选；
- evidence 是否从候选集中识别出正确者；
- override 是否值得推翻强 fallback。

VUS-SR 主要改善第三项和一部分第二项，但没有改变候选获取，也没有真正查看每个候选附近的高分辨率元素证据。当前剩余 gap 明确集中在第二项，四臂 counterfactual bank 又给第一项提供了可学习的完整监督。

## 最关键的实证

- Mind2Web：safe 34.92%，pass@12 59.21%；18.52 pp 是 candidate-ranking gap。
- ScreenSpot-Pro：safe 64.26%，pass@12 79.57%；14.60 pp 是 candidate-ranking gap。
- ScreenSpot 只有一个正确 candidate 时，现有 direct recall 仅 8.37%。
- 最小 target quartile 的 ranking failure 比最大 quartile高约 19 pp，两个 benchmark 都成立。
- 四臂前 6 candidates 100% 相同；oracle stage-2 routing coverage gain 为 +6.06/+3.67 pp。
- 旧 Q2b verifier 虽有 73.68% binary accuracy，却使最终 B3 −4.62 pp，因此“verifier accuracy 高”不是方法目标。

## 最小可执行顺序

### 1. 先跑 A1 router

已完成。corrected cross-fitted reliability run 相对 static C-cond 的 pass@12 为 −1.01 pp `[−2.10,0.00]` / +0.06 pp `[−0.81,+1.05]`；routing 已关闭。

### 2. 再跑 E0 local-evidence anchor

用 GPU 0--7 生成 fine/context candidate mosaics，每行仍只做一次 selector VLM forward。先冻结 Qwen3-VL，只看 logits 是否提高小 target 和 unique-correct recall。

### 3. E0 通过才训练 relational ranker

以 REPAIR/SAME/BREAK 和 pairwise utility 训练，不删除候选；保持 fold-sealed labels 与 pretest seals。

### 4. 最后加 risk control

阈值选择目标是 net utility lower bound，不是 AUROC、accuracy 或 raw confidence。

## 论文贡献可能性

如果 A1、E0、R1、C1 均通过，CARE 的贡献不再是“一个更好的聚合器网络”，而是：

> GUI test-time scaling should be formulated as budgeted counterfactual evidence acquisition followed by relational, risk-controlled selection.

这比继续调 MLP/Transformer 层数更有方法性，也直接解释已有 F1 的 pool × aggregator interaction。

## 必须保持的边界

- 当前两 benchmark 已被用于诊断和设计，只能算 discovery。
- 第三个 benchmark 必须在 CARE 完全冻结后一次性使用。
- transition verifier 不适合作为当前静态单步主方法；它只能在有真实前后帧的独立长时程 benchmark 中作为辅助。
- full Qwen LoRA 不是第一步；必须先证明局部 evidence 本身带来可测增量。
