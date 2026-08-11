# Visual Utility Selector Report

日期：2026-08-11

最终状态：`VUS_SET_RANKER_METHOD_CANDIDATE`

## 1. 为什么原来的提升太小

Utility-LSA 只看约 49 个手工结构统计，把 12 个候选独立回归，并用一个全局阈值覆盖两种风险结构。它看不到截图与任务语义。与此同时，候选 oracle 相对 CEV-A 仍有 Mind2Web 25.1--28.8 pp、ScreenSpot-Pro 14.1--17.6 pp 的行级 headroom。

ScreenSpot-Pro 中 baseline-correct-at-risk / repairable 约为 3.0--3.3，Mind2Web 约为 1.1。因此统一模型既需要视觉语义，也需要 benchmark 条件化的 downside gate；单纯加大 HGB 容量不是根因修复。

## 2. Blind visual evidence

VUS 将每行 12 个候选以 A--L 编号覆盖到原截图，并提供任务、最近动作、动作类型、坐标和参数。Qwen3-VL-8B-Instruct 在 GPU 0--7 上只输出 A--L 的单 token logits：

- 14,644 个 row-arm records，八个 shards 完整；
- public manifest 不含 target bbox、正 DOM、candidate success 或 evaluator 字段；
- blind predictions 在打开 private labels 前锁定 SHA-256 `b9b12b9cfc75ba7d797f7711bd56accdfb34d3dd81c9792069de2f6d8459ef40`；
- 模型 index SHA-256 为 `520b2e05079402e9468a8701d03d1154d14b2599593afb6effa7fb60c1bff070`。

第一次 anchor 在 10,278/14,644 行时主动停止：prompt 中的 row-cross-fitted fallback 对其他 outer fold 的 inner development 构成 second-level stacking leakage。该批次完整隔离在 `INVALID_SECOND_LEVEL_LEAKAGE/`，禁止判定。修正后视觉 prompt 完全 fallback-agnostic，CEV 只在 CPU safe gate 中按 nested context 注入。

clean zero-shot anchor 已有实质信号：Mind2Web equal-arm +1.65 pp，ScreenSpot-Pro −0.03 pp；utility-positive candidate AUROC 分别为 0.639 与 0.560，A1/A2 均通过。

## 3. Learned set ranker

VUS-SR 使用冻结 visual logits 和结构特征，不再次运行 VLM。每个候选输入包含视觉 logit/probability/rank/entropy、相对 exact fallback 的差值、LSA fallback-pair 结构特征、动作类别、benchmark/arm 风险状态及 fallback flag，共 119 维。

模型是 permutation-equivariant set ranker：共享 candidate encoder、两层无位置编码 Transformer、一个从 fallback state 初始化的 KEEP token，以及 candidate/KEEP utility head。S2/S3 另有 fallback-correct auxiliary head；部署使用 `1-sigmoid(correct_logit)` 作为 downside score。S1 未训练 auxiliary，因此该 gate 恒为 1。

冻结目标：

- S1：soft listwise repair-or-KEEP CE；
- S2：S1 + 0.5 fallback-correct BCE；
- S3：S2 + 0.25 expected U-GRPO utility。

S1 选择一次，S2 选择四次，S3 未选择。显式 downside head 的跨折选择频率为 4/5。

每个 epoch 将 256-row micro-batches 的加权梯度按该 split 全部 active weight mass 归一后累积，再执行一次 AdamW step。这样 benchmark/underlying-row/active-arm 的预设总质量不随 batch 边界变化；五个 final fits 均运行 30 个 optimizer steps。

## 4. Triple-nested protocol

对每个 outer fold 和每个 OOF holdout：

1. 两折拟合模型、CEV behavior policy 和结构 reliability；
2. 第三折只选择 checkpoint epoch；
3. 第四折只产生 OOF 配置与 safe-threshold 选择预测；
4. outer-test 完全封存；
5. final model 在四个 outer-development folds 训练四个 inner selected epoch 的 half-up median；
6. outer-test 只评估一次。

五个 outer artifacts 的 train/checkpoint/OOF/test 集合互斥；73,220 个 exact nested fallback contexts 唯一；所有 outer-test fallback 与冻结 CEV-A correctness 逐行一致，mismatch 为 0。

首次 formal 实现虽按 fold 正确索引 labels，但进程在 selection 前 eager parse 了五折 private-label 文件。按 V-K5 最严格解释，该执行边界被主动 invalidated。Correction 006 将 labels 物理拆成五个文件：每个 outer 进程只打开四个 dev files，完成 selection/final fit 并 fsync `outer-k.pretest.json` 后，guard 才允许首次打开 outer-test file。hardened rerun 的五个完整 outer JSON SHA、最终 adjudication 和描述性 controls 与 invalidated run 逐位相同。invalid artifacts 保留在独立 blobstore，仅 hardened outputs 用于结论。

## 5. Main result

Mind2Web 四臂 safe 增益均独立显著：

- C-uni +2.79 pp，99% CI [+0.94,+4.59]；
- C-cond +2.16 pp，[+0.57,+3.80]；
- C-rand +3.65 pp，[+1.75,+5.69]；
- C-self +3.37 pp，[+1.59,+5.22]；
- equal-arm +2.99 pp，[+2.10,+3.91]。

ScreenSpot-Pro equal-arm +0.11 pp，CI [−0.17,+0.37]；四个 cells 均安全。结果不是以牺牲 ScreenSpot 换取 Mind2Web。

相对 Utility-LSA，Mind2Web +3.02 pp，[+2.09,+3.92]；ScreenSpot −0.14 pp，[−0.48,+0.20]。equal-benchmark/equal-arm standardized 99% CI [+1.57,+3.17] MDE，SR4 通过。

## 6. Why training helped

相对同一 blind visual anchor，set ranker 在 Mind2Web 再提升 +1.35 pp，[+0.50,+2.21]；ScreenSpot +0.14 pp，[−0.13,+0.40]。因此主增益并非只来自额外 Qwen3-VL forward，listwise utility/downside training 提供了显著附加价值。

部署行为符合风险差异：

- Mind2Web equal-arm override rate 30.9%，总计 573 wins / 324 losses；
- ScreenSpot-Pro override rate 2.23%，21 wins / 14 losses。

模型不是统一提高 override 数，而是对 ScreenSpot 的约三倍 downside/repairable 比保持保守。

## 7. Decision and boundary

VUS-SR 通过全部四个 promotion gates，成为当前最强经验统一聚合器。CEV-A 仍是最强 training-free 统一规则。

按结果前冻结的 Amendment 003，VUS-SR 晋级后不继续 full Qwen3-VL LoRA；这是预设停止规则，不是算力不足。GPU 0--7 已用于完整 blind visual evidence 提取，GPU 0--4 用于五个 formal outer fits，protected PID 2274 全程未被 signal、暂停、kill 或改优先级。

VUS-SR 仍是已知两 benchmark 上的 nested discovery。所有训练设计发生在 CEV/LSA 结果已知之后；没有第三个独立 benchmark。论文可称 learned method candidate 或 strongest observed unified aggregator，不得称跨 benchmark 最终确认或 absolute SOTA。
