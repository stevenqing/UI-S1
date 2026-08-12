# CIVA：条件增量价值准入

## 核心变化

CARE、RAVEL、DELTA 说明“更多 acquisition / pixel / channel”都不会自动增加最终效用。DELTA 尤其表明 shared simplex gate 会给有害 local evidence 稳定分配质量，却没有学会拒绝它。

CIVA 不再融合所有证据，而是把每个额外 channel 看成一个 treatment。对每行分别预测它相对 VUS-binding direct baseline 的 `RESCUE` 与 `HARM` 概率，仅在预测净增量通过 development threshold 时 hard switch，否则保持 baseline。

## A0 最小实验

- 不做新 VLM inference，只使用五个已经 blind-locked 的 logits 文件。
- admission 特征只能来自 instruction、VUS-binding uncertainty、公开候选结构和图像尺寸。
- global/fine/context/random logits 在做决定前全部不可见。
- website/application 只用于 held-out grouping，不进入模型。
- target bbox/area 只属于已知诊断，不进入模型。
- 五折 outer label 物理封存，pretest fsync 后才打开。

主要对照是 no-text、text-only、random-channel placebo 与 matched-random switching。只有同时超过四类对照并满足所有 cell noninferiority，才说明 task-conditioned admission 可学习。

## 后续边界

A0 通过后也不能直接称方法。下一轮才允许把 frozen VUS-SR 作为 default policy，训练安全 policy-level switch；之后才可能测试 selective contrastive verifier。A0 失败则整条 admission 分支停止。