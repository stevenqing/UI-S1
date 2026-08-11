# RAVEL 研究计划

## 现在最值得做什么

固定使用 C-cond 候选池，不再学习 stage-2 路由。把研究资源集中到当前最大、最稳定的缺口：**正确候选已经存在，但模型没有识别出来**。

RAVEL 的三个模块：

1. **Local evidence**：全局截图之外，为 12 个 candidates 提供 fine/context 两级局部 mosaics；
2. **Relational utility**：直接判断 candidate 相对 CEV-A 是 REPAIR、SAME 还是 BREAK；
3. **Lower-bound safety**：只有净效用 99% lower bound 为正才 override。

## 为什么不再做 routing

CARE A1 已经做了严格五折测试。结构 router 相对 static C-cond：

- Mind2Web pass@12 −1.01 pp，CI [−2.10,0.00]；
- ScreenSpot pass@12 +0.06 pp，CI [−0.81,+1.05]；
- ScreenSpot final safe −1.27 pp，超过 MDE。

oracle routing 存在，但 stage1 structural state 无法预测。继续增加结构 router 容量不是 research，属于对负结果调参。

初版实现遗漏了 frozen cross-fitted reliability，已作废并按 Correction 002 原设置重跑；上面数字来自 corrected run。

## 为什么 local evidence 最合理

- candidate-ranking gap 为 18.52/14.60 pp；
- small-target ranking failure 显著更高；
- ScreenSpot 唯一正确候选时 direct recall 只有 8.37%；
- 原 full-screen overlay 会缩小目标、标签重叠，并让多数几何簇压过 minority truth。

## 为什么不是普通 verifier

旧 Q2b binary accuracy 73.68%，但最终指标下降 4.62 pp。RAVEL 不删除候选，也不优化 YES/NO accuracy；它优化 candidate-vs-fallback 的净 utility，并在最终 Step-SR 上选择阈值。

## 下一步执行

先只跑 E0 frozen anchor：生成 token-matched multi-scale evidence logits，比较 AUROC、unique-correct recall、small-target recall 和 safe Step-SR。E0 不通过就停止；通过后才训练 relational model。
