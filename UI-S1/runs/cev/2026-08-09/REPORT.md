# CEV / CEV-A Reconstructed Study Report

日期：2026-08-10

上游：`runs/aggmatch/2026-08-09/`、`runs/eqv/2026-08-09/`

计算约束：零 GPU、零新推理，只复用 E1 四臂候选 bank。

## 1. 结论

CEV-A 通过 V1–V3，并在两个 benchmark 上恢复各自的强极点：Mind2Web 精确恢复 majority 32.0192%，ScreenSpot-Pro 精确恢复 A2 aggregate 63.8836%。相对错误极点，Mind2Web 提升 5.34 pp，99% CI [+2.49,+8.21]；ScreenSpot-Pro 提升 4.05 pp，CI [+2.14,+5.99]。

但 CEV-A 与强制 nested dev-selection 打平：Mind2Web +0.19 pp，CI [−0.58,+0.96]；ScreenSpot-Pro +0.06 pp，CI [−0.50,+0.71]。因此 V4 为 `EXPLANATORY_CONTRIBUTION`，不是方法贡献。

论文定位：F1 继续是主结果。CEV-A 紧随其后，说明 majority 与 coordinate density 可以写成同一个 complete-link candidate-voting 过程的两个 inner-dev 极点。它没有证明优于“每个 benchmark 在 dev 上选聚合器”的朴素方案。

## 2. 预注册与泄漏边界

原 CEV 正本在 repo 与会话索引中均不存在。本轮以 `post-leakage reconstructed preregistration` 形式重建：

- `d873c41`：重建 spec、Amendment 011、机器可读 prereg；
- `de5b125`：在任何 CEV 结果前澄清 V1 的“逐位复现”是冻结 aggregate 数字精确一致，并报告逐行一致率。

重建前已知五个 ScreenSpot-Pro C-uni 格子：63.8836 / 63.8836 / 63.0614 / 62.5553 / 63.2511。它们没有参与阈值或粒度选择，只用于实现锚与污染披露。P-A/P-B 不作为新确认性证据。

## 3. 方法

CEV 在固定候选顺序上做 deterministic complete-link。主方法按候选计票，类平票依次使用 reliability 总和、最高 reliability 和冻结候选顺序。输出胜出类中的真实候选，不生成连续新点。

固定粒度阶梯为 G0 动作端点、G1 动作+坐标、G2 动作+参数、G3 完整积空间和 G4 坐标端点。ScreenSpot-Pro 固定 14 px。Mind2Web 坐标基尺度只从 inner-training GT bbox 尺寸估计，测试 GT 禁用；倍率与参数阈值只在 inner validation 选择。

CEV-A 比较 global 与 action-conditional 选择。外折 $k$ 为 test，$(k+1) \bmod 5$ 为 inner validation，其余三折为 inner training。所有 reliability、尺度、粒度和阈值选择都发生在 outer test 之外。

Mind2Web 五折 global 粒度为 G0/G0/G2/G0/G0，五折都选择 global。CLICK 每折都选 G0。TYPE/SELECT 的预测 plurality 行每折只有 0–3 行，低于 30 行门槛，全部回退 global；不能独立支持稀疏动作粒度结论。ScreenSpot-Pro 五折固定 G4。

## 4. V1–V4

V1：G4 complete-link candidate votes 与 A2 aggregate 都是 0.6388361796331435，精确通过。逐行 correctness 一致率 97.72%，有 36 行互换成功/失败。

V2：ScreenSpot-Pro CEV-A−A2 = 0.00 pp，CI [−0.57,+0.62]；Mind2Web CEV-A−majority = 0.00 pp，CI [0.00,0.00]。两端通过。

V3：ScreenSpot-Pro CEV-A−majority = +4.05 pp，CI [+2.14,+5.99]；Mind2Web CEV-A−sequential = +5.34 pp，CI [+2.49,+8.21]。两端通过。

V4：nested dev-selection 每折从七个 E1 聚合器中选择一个。CEV-A 与该对照在两端都不可区分且点差小于 MDE，所以定位为解释贡献。

## 5. P-A–P-G

P-A 受泄漏污染，只作诊断。7B ScreenSpot-Pro unlimited 为 63.8836%，cap 1 为 63.0614%，cap 2 为 64.0101%；简单 cap 没有消除 GTA1 胜出集中。历史 72B lineage normalization 从 B3 41.24% 提升到 70.59%，但与 best-single 71.41% 不可区分。去重与密度信号存在尺度依赖张力。

P-B 成立但属于污染锚：ScreenSpot 五折 G4。

P-C 成立：Mind2Web 五折中四折 G0、一折 G2，最终 correctness 与 majority 完全一致。

P-D 仅 CLICK 可判：五折均 G0。TYPE/SELECT 全部因样本过少回退，无法独立判定。

P-E 成立：sequential 下 C-cond−C-uni = +4.90 pp，CI [+2.97,+6.87]；CEV-A 下为 +0.43 pp，CI [−1.57,+2.57]；difference-in-differences = **−4.47 pp**，CI **[−7.34,−1.68]**。这是最重要的机制结果：恢复合适 G0 后，原显著 pool effect 被吸收。

P-F 端点选择基本稳定，但 C-K5 触发。G1/G3 的中央 0.75x/1.0x/1.25x 容差排名跨折翻转，只能说端点稳定，不能说容差层面存在普适规则。

P-G 部分支持。ScreenSpot 高 coordinate-support margin（≥2）组 1,104 行，CEV-A−majority +4.44 pp，CI [+2.41,+6.34]；低 margin 组 CI 跨零。Mind2Web 2,050/2,080 行 action margin ≥2，低 margin 组太小，不能建立可靠单调关系。Support/error structure 比“坐标 vs 积动作”标签更贴近现象，但不是因果定律。

## 6. 四臂与消融

Mind2Web 的 C-cond、C-rand、C-self 相对 C-uni 在 CEV-A 下均不可区分。ScreenSpot-Pro 在 G4 下 C-cond +2.59 pp 且显著，C-rand −2.78 pp 且显著，C-self +1.27 pp 但不可区分。

Mind2Web 从 G0、固定阈值 global、action conditional、参数阈值选择到完整 CEV-A 均为 32.0192%，额外复杂度没有 held-out 收益。Cap 1/2 均略降至 31.9712%；single-link 不改变 aggregate。ScreenSpot cap 1 与 single-link 更差；cap 2 点估计 64.0101%，但该轴受泄漏污染且没有预注册升级路径，不替换主方法。

## 7. 最终论文定位

1. F1 仍是主结果：不同错误结构对应不同有效聚合极点。
2. CEV-A 是紧随 F1 的统一解释：一个 nested complete-link voting 过程在两端恢复 G0/G4。
3. CEV-A 不优于 nested dev-selection，不能作为方法优势或额外 SOTA。
4. P-E 提供 pool × aggregator 交互的直接机制证据。
5. C-K5 与 post-leakage reconstruction 必须在 limitation 中明确披露。

外部 PID 2274 未触碰，无 GPU/model worker 启动。
