# HANDOFF — GUI Test-Time Scaling 论文项目

日期：2026-08-11

读者：接手实验执行或论文写作的人。

本文自包含，所有数字使用冻结口径。逐轮完整数据以各 `runs/*/REPORT.md` 为准。

## 0. Repo 审计与当前状态

截至 commit `02b8a575db4c7c2c8ec151568bdc1dcb04f84cb6`：

- `MASTER_RESULTS.md` 已于 2026-08-10 创建并覆盖 consolidation、xfer、close、aggmatch、eqv、cev。
- 原 CEV 正本未找到；已透明重建为 `post-leakage reconstructed preregistration`，正本为 `cev-spec-2026-08-09.md`、`runs/cev/2026-08-09/amendment_011.md` 与 `configs/cev_prereg.yaml`。
- CEV/CEV-A 已执行完成：V1–V3 通过，V4 为 `EXPLANATORY_CONTRIBUTION`，C-K5 触发。
- 08-03 之后已完成的是五轮：consolidation、xfer、close、aggmatch、eqv，而不是“六轮”。
- 原 handoff 中的 N12 manifest SHA 有转写错误；repo 正本为 `2a7233cf0fbab109ea481ba891d2d7e65a481c48cd73c623d7d76fc61c06ae17`。

除以上 repo 审计校正外，下文保留项目冻结结论与执行纪律。

## 1. 一句话状态

本项目研究多模型 GUI grounding 的 test-time scaling。论文主结果为**聚合器与动作空间/错误结构的匹配**（F1，双向显著）；次级结果为跨谱系共识 RoI 两阶段候选生成（Q1，仅在匹配聚合器下成立）。CEV-A 作为统一解释恢复 Mind2Web G0 与 ScreenSpot G4，但与 nested dev-selection 打平。VUS-SR 进一步加入 fallback-agnostic blind visual evidence 与 triple-nested listwise utility/downside training，在 Mind2Web 四臂平均相对 CEV-A +2.99 pp `[+2.10,+3.91]`，并保持 ScreenSpot-Pro +0.11 pp `[−0.17,+0.37]`。

学习聚合器路线现已晋级：主 LSA 安全但无显著增益，Utility-LSA 只在 ScreenSpot 获得 +0.25 pp；VUS-SR 的八个 cells 全部安全，相对 Utility-LSA 的平衡 standardized CI `[+1.57,+3.17]` MDE。当前最强经验聚合器为 VUS-SR，最强 training-free 规则仍为 CEV-A。VUS-SR 仍需新的独立 benchmark/trace 才能从 method candidate 升级为最终确认。

## 2. 论文骨架（冻结）

### 2.1 主结果：F1

同一 C-uni 候选池上，majority 与密度族在两个 benchmark 显著反向：

- Mind2Web（积动作空间）：majority 相对 sequential 为 **+5.34 pp**，99% CI **[+2.50,+8.04]**；相对 A1–A4 为 +7.8 到 +18.7 pp。
- ScreenSpot-Pro（纯坐标空间）：密度族相对 majority 优势为 3.86 到 4.11 pp；A1 不可区分。

文献常用的正是密度族，例如 MVP B3、ReGUIDE KDE 和本项目 A1–A4。

必须使用的机制限定：

- Mind2Web 的 E1 majority 是**动作类型 plurality + dev-priority 真实候选**，不是完整候选 exact majority。
- ScreenSpot-Pro 只有一个隐含动作，因此 majority 退化为 dev-best slot，没有使用连续坐标邻近性。
- CLICK 分层也是坐标动作，却与 ScreenSpot-Pro 方向相反，所以“动作空间维度”不是已证明的因果自变量。更稳妥的解释是元素离散性、候选错误结构或提议器质量。

### 2.2 次级结果：Q1

两阶段 12-forward 设计在密度族聚合器下显著优于三个对照：

- ScreenSpot-Pro：C-cond−C-uni **+2.21 pp**，CI **[+0.50,+4.16]**；对 C-rand +5.38；对 C-self +1.33。
- Mind2Web：C-cond−C-uni **+4.90 pp**，CI **[+2.94,+6.86]**；对 C-rand +3.27；对 C-self +2.31。
- 两个 benchmark 上六个强制对照 CI 下界全为正。

必须并列写 E1 限定：该优势在 majority 下不可区分，Mind2Web 为 +0.29 pp，ScreenSpot-Pro 为 +1.27 pp，两个 CI 下界均为负。机制解释是更合适的聚合器吸收了候选池差异。

### 2.3 机制结果：E3 high-start condition

几何 rank decay 两端都存在，但转化为可见性能下降需要高起点提议器：

- ScreenSpot-Pro：rank0 containment 99.94%，到 rank11 下降 38.90 pp；V-only N4→N16 为 −2.91 pp。
- Mind2Web：rank0 containment 40.38%，到 rank11 下降 9.23 pp；V-only 曲线统计上平坦。

只有两个数据点，仅作定性机制，不拟合普遍定律。

### 2.4 统一负结果链

候选池、错误结构与聚合器三者耦合，单边优化不足以保证最终精度：

- CALA：覆盖率上升但精度下降；
- NOA：$N_{eff}$ 上升但精度下降；
- R1：困难行预算增加使 pass@N 上升 18.99 pp，但 B3 不变；
- 四个选择规则独立失败；
- C-cond 候选池改进被 majority 吸收。

### 2.5 R4 选择性预测

跨谱系池增强 SafeGround 信号：AUROC 0.744→0.830，80% coverage 下 Mixed B3 领先 7.12 pp。

必须说明：信号本身来自 SafeGround（arXiv 2602.02419）；本项目确定性 N12 不继承其 K=10 随机协议、Learn-Then-Test/FDR 保证。定位为算法级迁移与信号增强。

### 2.6 EQV 与 CEV/CEV-A

EQV 在第一顺序 ABL-4 自检触发 U-K4：

- A2：63.8836%；
- complete-link + candidate votes：63.8836%；
- complete-link + lineage dedup：63.0614%，相对 A2 −0.8223 pp；
- single-link + lineage dedup：62.5553%；
- single-link + candidate votes：63.2511%。

这说明 complete-link 本身逐位复现 A2，但谱系去重损伤 ScreenSpot-Pro 的密度信号。U1–U3、dev-selection 和后续诊断均未运行，不能读作 null。

CEV/CEV-A 已完成。Mind2Web 选择 G0 主导并精确匹配 majority；ScreenSpot-Pro 选择 G4 并精确匹配 A2 aggregate。相对 nested dev-selection 两端 CI 均跨零，因此定位为解释贡献。C-K5 因中央容差排名跨折翻转而触发，禁止“处处匹配”或普适无容差规则表述。

### 2.7 VUS-SR learned aggregator

VUS 先在 GPU 0--7 对 14,644 个 row-arm records 提取 fallback-agnostic Qwen3-VL-8B A--L logits，再用无位置编码 set ranker 联合视觉证据、fallback-pair 结构特征与 benchmark/arm downside state。五个 outer folds 各自进行两折 model fit、一折 checkpoint selection、一折 OOF selection，outer-test 只访问一次。

- Mind2Web C-uni/C-cond/C-rand/C-self：+2.79/+2.16/+3.65/+3.37 pp，四个 99% CI 下界均为正；
- Mind2Web equal-arm：+2.99 pp `[+2.10,+3.91]`；
- ScreenSpot-Pro equal-arm：+0.11 pp `[−0.17,+0.37]`；
- 相对 blind visual anchor：Mind2Web 再增 +1.35 pp `[+0.50,+2.21]`；
- 五折选择 S1/S2/S2/S2/S2，显示 downside BCE 在 4/5 folds 被选择。

一次含 fallback prompt 的 early anchor 在 10,278/14,644 时因 second-level stacking leakage 主动停止并隔离，未进入判定。clean visual prompt 不含 CEV 或 GT；73,220 nested fallback contexts 与 outer CEV-A mismatch 为 0。

另一次 post-result 代码审计发现 formal process 在 selection 前 eager parse 了五折 label 文件，虽未索引 test labels，仍按 V-K5 严格作废。Correction 006 把 labels 物理拆成五折文件；每折先只打开四个 dev files，fsync pretest selection 后才由 guard 打开 test file。hardened 五折 JSON、adjudication 与 controls 均 bit-identical，只有 hardened outputs 用于主表。

### 2.8 CARE, RAVEL, and DELTA negative sequence

VUS-SR 后的 error decomposition 显示主要 headroom 是 candidate identification：Mind2Web/ScreenSpot candidate-ranking gap 为 18.52/14.60 pp，pairwise safe-gate gap 只有 5.77/0.71 pp。最小目标 quartile 与 unique-correct rows 尤其困难。

四臂共享前六 candidates，因此 CARE A1 严格测试了固定 12-forward 预算下的 stage-2 acquisition routing。虽然 oracle routing coverage gain 为 +6.06/+3.67 pp，corrected structural router 相对 nested static C-cond 的 pass@12 为 −1.01 pp `[−2.10,0.00]` / +0.06 pp `[−0.81,+1.05]`，ScreenSpot final safe 还下降 1.27 pp。A1 三项 gates 全失败，`CLOSE_ROUTING`；禁止增加结构容量或使用 stage-2 信息搜救。初版遗漏 cross-fitted reliability 的实现已按 Correction 002 作废并重跑。

RAVEL E0 已完成并触发 RAVEL-K4。local 相对 random-center 的 utility AUROC 两端均高约 +0.046，说明 candidate-centered pixels 有信息；但相对 VUS，Mind2Web AUROC −0.043，ScreenSpot +0.015。unchanged nested VUS-SR 的最终差值为 Mind2Web −2.19 pp `[−2.98,−1.41]`、ScreenSpot −0.03 pp `[−0.24,+0.16]`。因此 relational/lower-bound/LoRA stages 均取消。该结果支持 evidence competition，而不支持 pixel-level early fusion。

DELTA 随后按独立 result-free commit `de6a716` 测试 locked channels 的 decision-level late fusion。FULL 相对 VUS-SR 为 Mind2Web −0.41 pp `[−1.20,+0.40]`、ScreenSpot +0.11 pp `[−0.20,+0.41]`；DELTA-1/3/4/5 失败。FULL 无法超过同容量 VUS_ONLY 或 RANDOM_PLACEBO，并显著差于 VUS_GLOBAL。dropout 显示移除 fine/context 使 Mind2Web +0.83/+0.56 pp，说明当前 objective 仍让 local evidence 稀释 global/binding utility。结论为 `DELTA_NOT_SUPPORTED`，不运行 distillation 或 GUI-Odyssey confirmation。

## 3. 立即要做的事

### 3.1 论文写作

1. 以 F1 为主结果，CEV-A 为紧随其后的统一解释。
2. 把 P-E difference-in-differences `−4.47 pp [−7.34,−1.68]` 写入机制段，解释 pool effect 被匹配聚合器吸收。
3. 明确 CEV-A 与 nested dev-selection 打平，不写方法优势。
4. 将 VUS-SR 写为 learned method candidate / strongest observed unified aggregator，不写独立跨 benchmark confirmation 或 absolute SOTA。
5. 在 limitation 中披露 post-leakage reconstruction、五个泄漏格子、C-K5 容差翻转，以及 VUS-SR 仍缺第三 benchmark。
6. 使用 `MASTER_RESULTS.md` 冻结全部论文主表。

### 3.2 实验停止边界

- CARE routing、RAVEL early fusion 与 DELTA late fusion 均已关闭，不再调参。
- DELTA 未通过 same-capacity/placebo controls；其单调用 distillation 与第三 benchmark confirmation 取消。
- VUS_GLOBAL 是事后可见的 diagnostic control，不允许晋升为 selected method。
- 新研究必须另立 result-free preregistration，且不能在当前两 benchmark 上调整 DELTA channel masks、loss、gate regularization 或 threshold 后再称确认。

已泄漏的五个 ScreenSpot-Pro 格子不得作为优化目标：

```text
63.8836 / 63.8836 / 63.0614 / 62.5553 / 63.2511
```

### 3.2 可选项

- F3 按动作类型重加权：将 AndroidControl 子集动作分布配平到全量 7,650 行。方案必须在看结果前冻结；若单模偏差降到 2 pp 内，才可能把 AC 从附录救回第三数据点。

### 3.4 永久不做

- E2 原生 prompt 重跑与 AndroidControl 四臂：E-K1 取消。
- X2 自适应 zoom：整节删除，仅在 limitation 中一句话说明，不给数字。
- R2/R3、B3x、N6：各自 kill condition 连坐取消。

## 4. 数据资产与丢失清单

### 4.1 可用资产

- ScreenSpot-Pro 7B 36-action bank：GTA1-7B、Qwen3-VL-8B、UI-TARS-7B-SFT × 12 视角 × 1,581 行。所有聚合器可零 GPU 重算。
- ScreenSpot-Pro 72B N8/N12 bank：GTA1-72B、UI-Venus-72B、Qwen3.5-122B-A10B。存在 recovery drift，见 §7。
- Mind2Web 统一 prompt 四臂 bank：TongUI-7B、CogAgent-18B、UI-TARS-7B × 2,080 行 × 12 forwards × 4 arms，含 2026-08-07 新逐行 trace。
- AndroidControl stage1 部分 checkpoint：UI-AGILE 2,000/2,000，GUI-R1 1,096/1,056，UI-R1-E 1,824/1,792。
- AndroidControl 完整三模型交集：Low 1,096，High 1,056；仅单视角 3-forward 池。
- AC 归档：`/scratch/workspaceblobstore/aggmatch-traces/2026-08-09/`。
- EQV 归档：`/scratch/workspaceblobstore/eqv-traces/2026-08-09/`。

### 4.2 永久丢失

2026-08-06 已确认：历史 AndroidControl 与 Mind2Web lane 的逐行 `predictions.jsonl` 和 `rows.parquet` 已丢失；旧 manifest 曾记录 102,054 行。只剩 aggregate `score.json` / `audit.json`。

后果：

- 历史数字只能引用，不能重算；
- 历史 aggregate 不能与新数据做配对统计；
- 新 Mind2Web lane 使用统一 prompt，与历史数字相差约 20 pp，两套口径不可混用。

完整声明见 `runs/xfer/2026-08-07/LOST_TRACES.md`。

### 4.3 强制保留策略

- 逐行 JSONL 写入后立即 flush + fsync；
- lane 完成即计算 SHA-256 并写入 manifest；
- 使用独立于 git 的备份；
- 备份路径写入对应 `STATUS.json`；
- 清理脚本禁止递归删除 `raw/` 与 `predictions*.jsonl`。

## 5. 名词速查

### 5.1 Arms

- C-uni：三谱系 × proposer 视角 0–3，即 Uniform Mixed N12。
- C-cond：stage1 六前向（三谱系 × 视角 0/1）+ 跨谱系共识两裁剪 × 三谱系。
- C-rand：随机裁剪强制对照。
- C-self：单谱系自共识强制对照。

### 5.2 聚合器

- Majority：Mind2Web 为动作类型 plurality 后取 dev-priority 真实候选；ScreenSpot-Pro 因单一隐含动作退化为 dev-best slot。
- Sequential：type-first complete-link 密度聚合。
- B3：官方 MVP 最密 complete-link 簇。
- A0：fold-held-out best slot。
- A1：几何中位。
- A2：密度 medoid。
- A3：joint PKA medoid。
- A4：连续 PKA；Mind2Web 降到 13.46%，不作为推荐规则。
- M1：CCM，即 A5d-risk 族，使用校准 LR 与可靠性先验。

### 5.3 池与分配

V-only 为单谱系多视角；Uniform_Mixed、Quality_Only、CALA-S/A、NOA 为既有分配策略，均已判定。

### 5.4 切分与统计

- ScreenSpot-Pro：按 application GroupKFold(5)。
- Mind2Web：按 website-fold 内 episode 重采样。
- 一律 paired bootstrap 10,000 次，99% percentile CI。
- MDE：ScreenSpot-Pro 0.70 pp；Mind2Web 0.61 pp。
- AndroidControl 主口径 v1-only MDE 约 0.09–1.16 pp；五视角 30–42 版本禁用，v4 是部署偏移而非噪声。

## 6. 冻结关键数字

| 项 | 数字 | 备注 |
| --- | --- | --- |
| C-uni ScreenSpot-Pro 基线 | B3 63.69% | canonical；不用 63.63% |
| 63.63% 差异 | Amendment 005 前后 B3 实现一行之差 | 仅实现敏感性 |
| Drop-in 池增益 | +3.60 pp，CI [+1.31,+6.22] | 63.69−60.09 |
| M1 池增益 | +3.42 pp，CI [+1.41,+5.67] | 冻结口径 |
| CCM 归因 | +0.13 pp | 增益来自分配而非选择规则 |
| 端点差 | V-only N4→16 −2.91/−3.16；Mixed +1.90/+3.86 | 四个 CI 排除零；斜率降为附录 |
| 弱模型表述 | `LINEAGE_DIVERSITY_WITH_THIRD_LINEAGE_SATURATION` | leave-UI-TARS 63.76/63.88 不低于 full |
| Kappa | 视角轴 0.895；同族跨规模 0.618；跨族 0.398 | 冻结锚 |
| Containment | SSPro 99.94→61.04（rank0→11）；M2W 40.38→31.15（rank0→11） | E3 同范围两点 |
| 来源偏置 | 7B p=4.12e-152，V=0.779；72B p=1.21e-273，V=0.822 | B1 双尺度 |
| B2 三组数 | 61.99/70.59；61.99/70.52；63.69/70.52 | 历史 21 法、修复 21 法、combined-24 |
| F1 主对照 | M2W +5.34 [+2.50,+8.04]；SSPro −3.86 [−5.84,−1.92] | C-uni |
| CLICK 分层 | +6.26 [+2.86,+9.55] | TYPE +0.44、SELECT −1.27 均 n.s. |
| Q1 三对照 | SSPro +2.21/+5.38/+1.33；M2W +4.90/+3.27/+2.31 | 密度族下 |
| E1 限定 | majority 下 M2W +0.29；SSPro +1.27 | CI 下界为负 |
| 去重两面 | 7B SSPro −0.82；72B LN +29.29 | 72B 仍近 best-single 70.52 vs 71.41 |

说明：原 handoff 给出 Mind2Web containment rank15 27.60，但 E3 的冻结同范围比较使用 rank11 31.15；若写 rank15，必须明确它不是 E3 两端同 rank 的主表口径。

## 7. 论文必须自我披露

1. 统一 prompt 使 Mind2Web 各模型全图分数低于已发表数字约 20 pp，例如 CogAgent 30.87 vs 历史 50.1。四臂共享 prompt，内部效度成立；外部效度与 SOTA 保持开放。原生锚因 E-K1 未跑。
2. F2 动机写 5/7，实际重算为 4/7；必须主动披露，且全程标记 post-hoc。
3. E1 majority 两端定义不同，不得都写成 exact-candidate majority。
4. Evaluator equivalence 无法在 Mind2Web/ScreenSpot-Pro 上无 GT 严格实现，因为 point-in-bbox 需要目标 bbox；只能写 GT-free tolerance proxy。
5. EQV 已看过五个 ScreenSpot-Pro 格子；U1–U3 未运行，不能读作 null。
6. Recovery bank 非字节一致：72B M1 52.12→53.19；B1 winning-set 1374/1000/370→1370/1003/369。P1 因 `stata_windows_27` 只有 7 个唯一 crop 退到 N8。
7. Mind2Web 无可审计修正标签确认集，结果停留在 discovery 阶段。
8. AndroidControl Curated 排名倒置，单模最大摆动 21.6 pp；半径敏感性是 AC 不承载方法主张的理由。
9. 数据丢失清单必须进入可复现性声明。
10. F1 自变量存在张力：Mind2Web CLICK 同样是坐标动作却与 ScreenSpot-Pro 反向。讨论节不得把差异简单归因于动作空间维度。
11. 纸面数字 62.8、70.4、73.1、+13.4、+5.38，以及 Qwen3.5 71.41 独立口径，必须标注非同环境，不进入差值计算。

## 8. 项目纪律

### 8.1 预注册证据链

已核验以下 commits 存在：

```text
89f492c
9279fab
00aa688
18e0267
248f336
4827afc
d98c0ae
105b7ab
```

EQV 额外 commits：

```text
35cb9b6  initial EQV configs（含一处 YAML 语法错误）
4660661  syntax-only fix，先于任何结果
02b8a57  fail-closed EQV self-check
```

N12 manifest SHA-256：

```text
2a7233cf0fbab109ea481ba891d2d7e65a481c48cd73c623d7d76fc61c06ae17
```

### 8.2 Kill conditions

Kill condition 触发即按预注册处理，不搜救、不换判据。已触发的包括 K3、K4、H-K1、H-K4、L-K2、L-K3、R-K1、S-K1/2/3、B-K4、E-K1、F-K3、U-K4。

### 8.3 强制对照与选择

- 多方法比较必须 nested dev selection；held-out 每折只产生一个选择结果。
- 强制对照不得省略：C-rand、C-self、随机拒绝区间、best-single、dev-selection。
- 失败后禁止换定义再报告。EQV 自检后没有移除谱系去重，是应延续的范例。

## 9. Spec 与 run 索引

| 时间/目录 | 状态 |
| --- | --- |
| `runs/complementarity/2026-07-30/`、`runs/collision-law/2026-07-30/` | 完成，K3/K4 触发 |
| ccm-headtohead（07-31） | 完成，H-K1 触发 |
| diversity-axis、closing、scaleup-gate（08-02） | 完成，X3/X6/X7 通过，X2 删除 |
| neff-law、reallocation、source-bias（08-03） | 完成，N1/R1 失败，R4 通过，B2 路径 B |
| cross-benchmark、execution-plan（08-04） | 被数据丢失阻断，仅作历史参考 |
| `runs/dominance/2026-08-06/` | 完成，D0 修复 R7，D1 方向性 |
| `runs/consolidate/2026-08-06/` | 完成，Q1 通过，S5 定位 rank decay |
| `runs/xfer/2026-08-07/` | 完成，Mind2Web 迁移通过，含 trace 保留 |
| `runs/close/2026-08-08/` | 完成，E-K1 触发，E3 通过 |
| `runs/aggmatch/2026-08-09/` | 完成，F1 成为主结果 |
| `runs/eqv/2026-08-09/` | U-K4 停止 |
| `runs/cev/2026-08-09/` | 完成；V4 解释贡献，C-K5 触发 |
| `runs/lsa/2026-08-10/` | 完成；主模型安全但不显著 |
| `runs/lsa-confirm/2026-08-10/` | 完成；partial transfer，停止当前数据上的 learned 搜索 |
| `runs/lsa-utility/2026-08-11/` | 完成；safe exploratory，UR2/UR5 失败，UR-K5 false |
| `runs/visual-utility-selector/2026-08-11/` | 完成；VUS-SR method candidate，SR1--SR4 通过 |
| `runs/care/2026-08-11/` | A1 完成；structural routing 失败，`CLOSE_ROUTING` |
| `runs/ravel/2026-08-11/` | E0 完成；RAVEL-K4，停止 relational/LoRA |
| `runs/delta/2026-08-11/` | 协议冻结；locked-channel late fusion 尚未运行 |

## 10. 已知未完事项

1. SafeGround 锚是 `ALGORITHM_LEVEL_PORT`：本地 0.6278 vs 官方 0.6344，差 0.0066 且协议不同。
2. X1 sampling 轴始终未补，每行仅 5 采样；标题只能写固定视角轴 + 跨谱系轴。
3. Thesis 层面对应 GUI agent 章节，与 Cooperation through Diversity 主线如何衔接由作者决定，不在本 handoff 范围。

## 11. 接手者启动检查表

- [ ] 阅读本文件与 `runs/aggmatch/2026-08-09/CONSOLIDATED_SUMMARY_ZH.md`。
- [ ] 阅读 `runs/eqv/2026-08-09/REPORT.md`，确认理解 U-K4 与五个已泄漏格子。
- [x] 重建 CEV spec、Amendment 011、`cev_prereg.yaml`，并披露 post-leakage 状态。
- [x] 在跑数前提交 CEV 配置：`d873c41`、`de5b125`。
- [x] 验证 complete-link + candidate votes 精确复现 A2 aggregate 63.8836%。
- [x] VUS 获授权使用 GPU 0--7；blind visual inference 与 formal set-ranker 已完成。
- [x] CARE A1 完成并按 gate 关闭 routing。
- [x] RAVEL E0 完成并按 RAVEL-K4 停止。
- [ ] DELTA：先跑多通道 late-fusion viability，未通过不得蒸馏。
- [ ] 保持 PID 2274 不被 signal、暂停、kill 或改优先级。
- [ ] 新产物逐行 fsync、SHA manifest、独立备份。
- [ ] 按 kill condition 停止，不以已泄漏格子调参。
- [x] CEV 完成并刷新主表与 `MASTER_RESULTS.md`。