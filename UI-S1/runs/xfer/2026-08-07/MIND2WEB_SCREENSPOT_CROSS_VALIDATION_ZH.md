# Mind2Web × ScreenSpot-Pro 交叉验证总结

日期：2026-08-08

状态：`MIND2WEB_PRIMARY_TRANSFER_PASS_E_K1_CLOSED`

## 0. Closure update（2026-08-08）

后续四臂 × 聚合器矩阵已经完成，详见 [Aggregator Closure Report](../../close/2026-08-08/REPORT.md)。结论是 E-K1 触发：C-cond 在 majority 下仍是两个 benchmark 的点估计最优 arm，但 C-cond − C-uni 的 99% CI 均跨0，且三个强制对照没有重新全部通过。因此候选生成改进必须限定在原冻结 sequential/density 聚合器下；E2 原生 prompt 重跑与 AndroidControl 均按预注册取消，SOTA 线保持开放。

## 1. 结论

Mind2Web 对 ScreenSpot-Pro 的核心 Q1 结论形成了明确的跨 benchmark 正向验证：

> 先用多谱系模型在全图与首个注意力视角上形成第一阶段预测，再根据跨谱系共识几何生成第二阶段 RoI，在相同 12-forward 预算下，稳定优于固定注意力视角、随机新裁剪和提议器单谱系自共识。

这个结论同时在纯坐标 grounding benchmark（ScreenSpot-Pro）和积动作 benchmark（Mind2Web，动作类型 × 坐标 × 参数）上成立。三个强制对照的 99% CI 下界在两个 benchmark 上都为正，因此收益不能只解释为“增加了新裁剪”“任意自共识有效”或“预算更多”。

但机制层面不是所有现象都迁移：ScreenSpot-Pro 上的 V-only 随预算下降、Mixed 随预算上升的符号翻转没有在 Mind2Web 上复现。Mind2Web 的提议器几何覆盖率随 rank 下降，但该几何衰减没有转化为显著的 Step SR 预算曲线下降。因此：

- **Q1 方法效果得到交叉验证；**
- **rank-decay 的几何趋势部分复现；**
- **预算曲线符号翻转没有交叉验证。**
- **同池A1–A4强聚合baseline已击败，但纯majority与fold-held-out最佳slot尚未击败。**

## 2. 两个 benchmark 的实验对应关系

| 项目 | ScreenSpot-Pro | Mind2Web |
|---|---|---|
| 任务空间 | 坐标 grounding | 动作类型 × 坐标 × 参数 |
| 行数 | 1,581 | 2,080 steps / 252 episodes |
| 第一阶段 | 三谱系 × views 0/1，共6次 | 三谱系 × 全图/view1，共6次 |
| 第二阶段 | 两个 RoI × 三谱系，共6次 | 两个 RoI × 三谱系，共6次 |
| 总预算 | 每行12 forwards | 每行12 forwards |
| 主指标 | B3 grounding accuracy | Micro Step SR |
| 附加指标 | M1、pass@N | Episode-macro Step SR |
| 主统计 | application-group bootstrap，99% CI | website-fold内episode bootstrap，99% CI |
| MDE | 0.70 pp | 0.61 pp |

Mind2Web 使用 TongUI-7B、CogAgent-18B、UI-TARS-7B 三条谱系；TongUI-7B 为共享注意力提议器。所有四个 arm 的平均前向数均为12，最大预算差为0，不需要补预算匹配对照。第二阶段触发率为100%，因此主结果不是仅在触发子集上成立。

## 3. 核心结果对照

### 3.1 四个 arm

| Benchmark / arm | C-uni | C-cond | C-rand | C-self |
|---|---:|---:|---:|---:|
| ScreenSpot-Pro B3 | 63.69% | **65.91%** | 60.53% | 64.58% |
| Mind2Web Micro Step SR | 26.68% | **31.59%** | 28.32% | 29.28% |
| Mind2Web Episode macro | 32.26% | **37.37%** | 33.28% | 34.49% |

C-cond 在两个 benchmark 上均为四臂最优。

### 3.2 三个强制配对比较

| Comparison | ScreenSpot-Pro delta，99% CI | Mind2Web delta，99% CI |
|---|---:|---:|
| C-cond − C-uni | +2.21 pp，[+0.50,+4.16] | **+4.90 pp，[+2.94,+6.86]** |
| C-cond − C-rand | +5.38 pp，[+2.94,+8.08] | **+3.27 pp，[+1.26,+5.32]** |
| C-cond − C-self | +1.33 pp，[+0.06,+2.75] | **+2.31 pp，[+0.95,+3.68]** |

三个比较在两个 benchmark 上方向完全一致，且六个 99% CI 下界全部大于0。

Mind2Web 的 C-cond − C-uni 为 +4.90 pp，明显超过其独立估计的 MDE 0.61 pp。对应预注册判据：

- XF1：`true`；
- XF2：`true`；
- XF-K1：`false`；
- XF-K2：`false`；
- XF-K3：`false`。

因此 Mind2Web 不仅没有触发停止条件，反而比 ScreenSpot-Pro 给出了更大的主效应点估计。

## 4. 三个对照分别排除了什么

### 4.1 相对 C-uni

C-uni 使用固定的提议器 views 2/3。C-cond 在两个 benchmark 上均显著优于 C-uni，说明性能提升来自根据当前样本第一阶段预测动态构造 RoI，而不是固定增加六次后续推理。

### 4.2 相对 C-rand

C-rand 与 C-cond 都使用新的第二阶段裁剪，预算相同。C-cond 在两个 benchmark 上均显著优于 C-rand，说明收益不是“新鲜裁剪”或随机增加图像覆盖带来的。

### 4.3 相对 C-self

C-self 使用提议器自身 view0/view1 预测点构造裁剪。C-cond 在两个 benchmark 上均显著优于 C-self，说明任意单谱系自共识不足以解释结果；跨谱系预测几何提供了额外信息。

这三个对照共同支持的最小充分主张是：

> 跨谱系第一阶段共识能够提供比固定视角、随机裁剪和单谱系自共识更有效的第二阶段 RoI。

## 5. Mind2Web 上额外确认了什么

### 5.1 方法可以扩展到积动作空间

ScreenSpot-Pro 只需要选坐标。Mind2Web 必须先在 CLICK/TYPE/SELECT 上做动作类型 plurality，再只对 winning type 的坐标点聚类，同时用 token-set F1 处理参数。Q1 在这个更复杂的动作空间中仍然成立，说明方法不局限于纯坐标聚合。

### 5.2 Micro 与 episode macro 同向

Mind2Web C-cond：

- Micro Step SR：31.59%；
- Episode-macro Step SR：37.37%。

相对三个对照，两种汇总口径的点估计均同向。主判据仍按预注册使用2,080步 micro；252-episode macro 作为附加稳健性证据。

### 5.3 聚合略高于最佳全图单模型

Mind2Web 最佳全图单模型是 CogAgent-18B，Micro Step SR 为30.87%；C-cond 为31.59%，高0.72 pp。这个差值不是本轮预注册的主配对判据，不能替代 C-cond 对 C-uni 的结论，但它说明最终聚合结果没有通过牺牲最强单模性能来获得对照优势。

### 5.4 几何退化不是主结果来源

- Stage-2 trigger rate：100%；
- 单簇时使用冻结最远点 fallback：37/2,080行，约1.78%；
- 四臂平均预算：全部12 forwards。

因此结果不是由大量跳过第二阶段、不同平均预算或大规模几何fallback驱动。

## 6. Majority voting 与强聚合 baseline

为避免把RoI构造收益和聚合器收益混在一起，我们在完全相同的 C-cond 12候选池上补做了零GPU、五折、逐行配对比较。所有方法读取同一批模型输出；差异只在最终聚合算子。priority与grounding weight只由outer-fold开发集估计。

### 6.1 同池结果

| Aggregator | Micro Step SR | Ours − baseline | 99% CI | 是否显著击败 |
|---|---:|---:|---:|---:|
| Ours：sequential complete-link cluster | 31.59% | — | — | — |
| Majority：动作plurality + dev-priority exact candidate | **32.31%** | -0.72 pp | [-3.24,+1.78] pp | **否** |
| A0：fold-held-out最佳candidate slot | 31.88% | -0.29 pp | [-2.93,+2.49] pp | **否** |
| A1：plurality + geometric median | 25.63% | +5.96 pp | [+4.01,+8.02] pp | **是** |
| A2：plurality + density medoid | 27.45% | +4.13 pp | [+2.25,+6.05] pp | **是** |
| A3：joint PKA medoid | 27.45% | +4.13 pp | [+2.26,+6.03] pp | **是** |
| A4：continuous PKA | 13.46% | +18.13 pp | [+15.75,+20.58] pp | **是** |

我们的聚合器显著优于Collision-Law的A1–A4，但当前最强同池baseline是纯majority voting，32.31%，比我们的31.59%高0.72 pp。该差值的99% CI跨0，因此不能写“majority显著更好”，也不能写“我们已经击败majority”。A0同样与我们统计不可区分。

最佳全图单模型CogAgent-18B为30.87%，我们的C-cond为31.59%，点估计高0.72 pp；但它不是本轮新增baseline表中的预注册主比较，不能用来替代majority结论。

### 6.2 当前可以和不可以声称什么

可以写：

> 在同一C-cond候选池上，sequential cluster显著优于geometric-median、density-medoid和joint/continuous PKA聚合器。

不可以写：

- “最终系统已经击败majority voting”；
- “最终系统已经达到或超过Mind2Web SOTA”；
- “31.59%优于所有同池聚合baseline”。

### 6.3 SOTA比较为什么仍未闭合

本轮transfer使用的是在推理前冻结的新统一product-action prompt。历史Mind2Web数字（例如TongUI-7B约52.9%、CogAgent约50.1%）来自旧的模型原生adapter与prompt；旧逐行trace已经丢失，不能与本轮31.59%做配对统计，也不能把prompt差异忽略后直接声称SOTA。

要闭合SOTA主张，必须新增一个单独的、结果前冻结的baseline阶段：

1. 在同一2,080行、同一官方评分器上精确重跑公开SOTA模型的原生prompt/adapter；
2. 报告模型checkpoint、prompt hash、parser和逐行prediction trace；
3. 同时报单模型、majority、A0、A1–A4和我们的系统；
4. 对“ours − majority”和“ours − best published deployable baseline”做episode分层99% CI；
5. 若新方法或router是在看到本轮test结果后提出，必须在独立确认集或新benchmark上验证，不能把当前2,080行继续当未见test集。

因此，当前论文最准确的状态是：**RoI方法的跨benchmark主张已闭合，最终聚合器相对majority/SOTA的主张仍开放。**

## 7. 哪些 ScreenSpot 机制没有在 Mind2Web 复现

### 7.1 ScreenSpot-Pro

ScreenSpot-Pro 的主配对端点统计为：

- V-only N16 − N4：-2.91 pp，99% CI [-5.58,-0.36]；
- Mixed N16 − N4：+1.90 pp，99% CI [+0.42,+3.47]。

随机打乱view顺序后，V-only下降平均反转为上升，因此 ScreenSpot 上得到 `RANK_DECAY_DOMINANT`。

### 7.2 Mind2Web

Mind2Web 的对应结果为：

- V-only N16 − N4：+0.34 pp，99% CI [-1.22,+1.98]；
- Mixed N16 − N4：-0.63 pp，99% CI [-2.53,+1.30]。

两个区间都跨0，方向也不满足“V-only负、Mixed正”，所以 XF4 为 `false`。

同时，Mind2Web 的 proposer full-bbox containment 确实随 rank 下降：

- rank0：40.38%；
- rank15：27.60%；
- rank0–11均值：35.12%。

这说明几何 rank decay 存在，但它并不足以推出最终 Step SR 必然随预算下降。可能原因包括：

1. Mind2Web 的成功还受动作类型和参数正确性控制；
2. 后续低rank视角即使几何更弱，也可能补充动作语义；
3. 产品动作聚合的 plurality 与参数medoid会改变“更多坐标候选”的作用；
4. 两个 benchmark 的提议器、模型谱系和错误结构不同。

因此不能把 ScreenSpot 的预算符号翻转写成跨 benchmark 定律。

## 8. 交叉验证裁决

| 主张 | 裁决 | 证据 |
|---|---|---|
| C-cond 优于固定 Uniform views | **跨 benchmark 支持** | 两边99% CI下界均为正 |
| C-cond 优于随机新裁剪 | **跨 benchmark 支持** | 两边99% CI下界均为正 |
| C-cond 优于单谱系自共识 | **跨 benchmark 支持** | 两边99% CI下界均为正 |
| 方法可扩展到积动作空间 | **支持** | Mind2Web XF1/XF2通过 |
| Stage-2收益依赖低触发子集 | **不支持** | Mind2Web触发率100% |
| Proposer几何覆盖随rank下降 | **两个benchmark均观察到** | ScreenSpot归因；Mind2Web rank containment下降 |
| V-only下降、Mixed上升的符号翻转 | **未跨 benchmark 支持** | Mind2Web XF4失败，两个CI均跨0 |
| Rank decay决定最终预算曲线 | **只能限定在ScreenSpot-Pro** | Mind2Web几何衰减未转化为Step SR下降 |
| Ours优于A1–A4聚合器 | **Mind2Web支持** | 四个99% CI下界均为正 |
| Ours优于majority voting | **未支持** | Ours低0.72 pp，99% CI跨0 |
| Ours达到Mind2Web SOTA | **未判定** | 新旧prompt/adapter与trace协议不可直接混合 |

## 9. 推荐论文表述

### 9.1 可以写

> Across ScreenSpot-Pro and Mind2Web, sequential RoIs derived from cross-lineage first-stage consensus consistently outperform fixed uniform views, random fresh crops, and proposer-only self-consensus under the same 12-forward budget. The effect transfers from coordinate-only grounding to product-action prediction.

中文：

> 在 ScreenSpot-Pro 与 Mind2Web 上，由跨谱系第一阶段共识生成的顺序 RoI，在相同12次前向预算下，均显著优于固定视角、随机新裁剪和提议器单谱系自共识。该收益从纯坐标 grounding 迁移到了积动作预测。

### 9.2 必须限定

> The budget-sign reversal and rank-decay performance law are benchmark-dependent. Mind2Web reproduces geometric rank decay but not the ScreenSpot-Pro V-only-decline/Mixed-improvement pattern.

中文：

> 预算曲线符号翻转及其性能层面的 rank-decay 规律具有 benchmark 依赖性。Mind2Web 复现了几何覆盖率随rank下降，但没有复现 ScreenSpot-Pro 的 V-only下降/Mixed上升模式。

### 9.3 不应写

- “任意跨谱系池都优于单谱系池”；
- “增加模型或视角必然提升性能”；
- “rank decay 是所有GUI benchmark预算下降的统一定律”；
- “Mind2Web 已验证所有 ScreenSpot 机制”。
- “当前最终系统已经击败majority voting或Mind2Web SOTA”。

## 10. 最终判断

这次 Mind2Web 迁移对最重要的方法主张给出了强交叉证据：Q1 的效果不是 ScreenSpot-Pro 特例，并且在更复杂的积动作空间中效应更大。与此同时，Mind2Web 否定了把 ScreenSpot 的预算曲线机制过度泛化为普遍规律。

因此当前最稳健的论文结构应是：

1. **主方法结论：跨谱系共识 RoI 具有跨 benchmark 可迁移性；**
2. **机制结论：几何 rank decay 可观察，但其对最终性能曲线的影响依赖 benchmark；**
3. **边界结论：结构化顺序共识有效，任意混合和普遍预算定律不成立。**
4. **开放问题：当前聚合器尚未击败纯majority；SOTA比较需要原生adapter的独立重跑。**

## 11. 结果与复现文件

- ScreenSpot-Pro 总结：[../../consolidate/2026-08-06/CONSOLIDATED_SUMMARY_ZH.md](../../consolidate/2026-08-06/CONSOLIDATED_SUMMARY_ZH.md)
- ScreenSpot-Pro Q1：[../../consolidate/2026-08-06/q1_sequential.json](../../consolidate/2026-08-06/q1_sequential.json)
- Mind2Web XF结果：[xf_mind2web.json](xf_mind2web.json)
- Mind2Web MDE：[mde_mind2web.json](mde_mind2web.json)
- Mind2Web同池baseline：[baseline_mind2web.json](baseline_mind2web.json)
- Mind2Web简版总结：[CONSOLIDATED_SUMMARY_ZH.md](CONSOLIDATED_SUMMARY_ZH.md)
- 发布trace清单：[PUBLICATION_MANIFEST.json](PUBLICATION_MANIFEST.json)
- 冻结迁移协议：[SPEC.md](SPEC.md)
