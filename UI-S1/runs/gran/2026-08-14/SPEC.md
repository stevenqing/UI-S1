# GRAN — 聚合粒度轴 预注册 spec

Round: `gran`
Run dir: `runs/gran/2026-08-14/`
Date: 2026-08-14
Status: DRAFT，未 commit。本文件必须先于任何 τ sweep 结果进 git，时间戳即证据链。
GPU: 零。全部基于现存 SSPro 7B 36-action bank 与 M2W 统一 prompt 四臂 bank 重算。

## 0. 本轮的性质（先定死，防止重演 TriVUS）

本轮产出的是 explanatory result，不是 method。核心量 $p$ 与 $q_{\max}$ 是 label-dependent 的，只能作为 evaluation-side 分层变量，地位等同 TriVUS 报告里的 hit@k 与 first-success rank。本 spec 不承诺、也禁止在本轮内提出任何 GT-free 的 τ 选择器。τ 一律是 per-benchmark 或 per-action-type 的常数，由 nested inner dev 选出，held-out 只出一个数。

本轮要回答的唯一问题是 F1 的自变量是什么。§7 记录的张力是 M2W 的 CLICK 子集同样是坐标却与 SSPro 反向，因此“动作空间维度”不是自变量。本 spec 给出一个替代候选并把它写成可证伪的形式。

## 1. 形式化

### 1.1 统一分解

对一行 $r$，候选池 $C={c_1,\dots,c_N}$，$c_i$ 由 source $a(i)$ 产生，source 沿 lineage 与 view 两轴取值。存在不可观测的行为等价关系 $\sim_B$，$c\sim_B c'$ 当且仅当二者触发同一 GUI 行为。正确性即落入目标等价类 $X_{gt}$。

任何 aggregator 都可写成三元组 $(\mathcal{P}_\tau, s, \pi)$。$\mathcal{P}_\tau$ 是候选集上的代理划分，由粒度参数 $\tau$ 索引；$s$ 是块打分函数；$\pi$ 是块内选取规则。

| 方法 | $\mathcal{P}_\tau$ | $s$ | $\pi$ |
| --- | --- | --- | --- |
| B3 / MVP | 空间聚类，半径 $\tau$ | 块大小 | 质心 |
| A2 | 同上 | 块大小 | medoid |
| A1 | 单块（$\tau\to\infty$） | 平凡 | 几何中位 |
| sequential | type-first 后 complete-link $\tau$ | 块大小 | 密度 |
| majority (M2W) | 动作类型 | 块大小 | dev-priority 先验 |
| majority (SSPro) | 精确重合，即 $\tau\to 0$ | 恒为 1 | dev-best slot 先验 |

关键观察是最后两行。SSPro 上 majority 之所以退化成先验选取，不是因为它是另一种方法，而是因为坐标连续使 $\tau\to 0$，所有块变成单点，$s$ 恒等于 1，选择权全部落给 $\pi$。M2W 上 majority 则是 $\tau$ 取到类型级的粗端。也就是说 F1 比较的两个方法是同一族在粒度轴两端的取值，不是两个独立方法。这一点是本 spec 的出发点，也是 CEV 打平分支所需的结构基础。

### 1.2 生成模型（Assumption A1）

对每行假设如下。目标区域 $X_{gt}$ 是轴对齐矩形，特征半径 $R$ 取其短边之半。存在 $K$ 个 distractor 模式，错误质量分布为 $q_1,\dots,q_K$，记 $q_{\max}=\max_j q_j$。每个候选以概率 $p$ 正确，正确候选在 $X_{gt}$ 内以尺度 $\sigma_c$ 散布；以概率 $1-p$ 错误，按 $q$ 落到某个 distractor 上。目标与最近 distractor 的间距记 $d_{\min}$。

同源候选共享潜变量，产生组内相关。用 §6 的 kappa 锚作为相关系数刻度，视角轴 $\rho_v=0.895$，同族跨规模 $0.618$，跨族 $\rho_\ell=0.398$。

## 2. 推导

### 2.1 引理 1（纯度充分性）

若块 $B\subseteq X_{gt}$，则 $\mathrm{centroid}(B)\in X_{gt}$，因为 $X_{gt}$ 是凸集；同时 $\mathrm{medoid}(B)\in B\subseteq X_{gt}$。故在纯块上，密度族的一切块内规则给出相同的正确性判定。

推论 1。A1、A2、B3 三者的性能差异整体支撑在受污染块上，其可能差距的上界是污染率 $1-\alpha(\tau)$，其中 $\alpha(\tau)=P(c\sim_B c'\mid \text{同块})$。这解释了为什么 SSPro 上 A1 到 A4 彼此相差不到半个点，也给出一条无需新实验即可自检的不等式。

### 2.2 命题 1（粒度窗口）

记 $\beta_c(\tau)=P(\text{两正确候选同块})$，随 $\tau$ 单调增，在 $\tau\gtrsim 2\sigma_c$ 后趋于 1。记正确块被污染的概率随 $\tau$ 在 $\tau\gtrsim d_{\min}$ 后急增。因此密度信息可用的窗口是

$$
2\sigma_c \lesssim \tau \lesssim d_{\min}.
$$

窗口非空的条件是 $\gamma := d_{\min}/\sigma_c \gtrsim 2$。$\gamma$ 是尺度可分性，与动作是否离散无关。离散动作空间之所以看起来特殊，是因为元素身份给出的划分对应 $d_{\min}$ 极大而 $\sigma_c$ 为零，即 $\gamma=\infty$，窗口无限宽。这说明离散只是 $\gamma$ 轴的一个极限点，不是一个独立的类别。

### 2.3 命题 2（计数何时优于先验）

在窗口内 $\beta_c\approx 1$，正确块期望计数 $n_c \approx Np$，最大错误块期望计数 $n_w\approx N(1-p)q_{\max}$。密度族在期望意义上选对块的条件为 $n_c>n_w$，即

$$
\frac{p}{1-p} > q_{\max} \quad\Longleftrightarrow\quad p > \frac{q_{\max}}{1+q_{\max}}.
$$

错误完全集中于单一 distractor 时 $q_{\max}=1$，要求 $p>1/2$；错误弥散时门槛显著放低。

先验选取（$\pi$ 单独工作，即 majority 在 SSPro 的退化形态，或 dev-best slot）的准确率是最优单源正确率 $p_{a^*}$。因此

$$
\textbf{密度族优于先验选取} \iff P(n_c>n_w) > p_{a^*}.
$$

这条不等式是 F1 的机制形式。它预测的自变量是 $p$ 与 $q_{\max}$，不是动作空间维度。两个 benchmark 的 containment 数字已经把方向摆在那里，SSPro 从 99.94 降到 61.04，M2W 从 40.38 降到 27.60，前者的 $p$ 上界远高于后者。CLICK 子集与 SSPro 反向因此不再是反例，CLICK 继承的是 M2W 的低 $p$，不是 SSPro 的高 $p$。

### 2.4 命题 3（有效票数）

计数量的方差按 design effect 膨胀。四视角单谱系有 $\mathrm{deff}_v = 1+3\rho_v = 3.685$，三谱系跨族有 $\mathrm{deff}_\ell = 1+2\rho_\ell = 1.796$。于是

$$
N_{\text{eff}}(\text{C-uni}) \approx \frac{12}{3.685\times 1.796} \approx 1.81,
\qquad
N_{\text{eff}}(\text{V-only},N{=}12) \approx \frac{12}{1+11\rho_v} \approx 1.11.
$$

十二次前向只买到不到两票有效独立证据。这直接解释三条既有结果，第三谱系饱和（跨族相关已经吃掉边际收益），端点差的符号（V-only $N4\to16$ 为 $-2.91$ 而 Mixed 为 $+1.90$），以及 NOA 的 $N_{\text{eff}}$ 涨而精度不涨（涨的是名义有效数，不是命题 2 里的 $p$）。定量预测为 margin 的 z 统计量之比应约为 $\sqrt{1.81/1.11}=1.28$。

Assumption A2。以上把 agreement kappa 直接当作计数统计的 intraclass correlation，这一步需要单独验证，见 G-P6。

### 2.5 命题 4（两极点，对 CEV 叙事的修正）

$\tau\to 0$ 时所有块为单点，$s$ 恒为 1，聚合器等于 $\pi$。$\tau\to\infty$ 时只剩一块，聚合器等于 $\pi$ 作用于全池。两极点重合当且仅当 $\pi$ 是先验选取。若 $\pi$ 是 medoid，$\tau\to\infty$ 给出全局 medoid 而非先验，两极点不重合。

因此“一个过程、两个极点、处处匹配”这句话只对 count-then-prior 型聚合器成立，对 medoid 型不成立。CEV 若要走解释贡献分支，必须把 $\pi$ 固定为先验选取，否则叙事在数学上是错的。这是本推导对 CEV 定位的一个硬约束。

## 3. 可证伪预测

全部在现有 bank 上零 GPU 可算。$\hat p$ 定义为池中落入 gt 区域的候选比例，$\hat q_{\max}$ 定义为错误候选中落入最大错误簇的比例，二者均在 $\tau^*$ 下计算，且均只作分层用。

| ID | 预测 | 判定口径 |
| --- | --- | --- |
| G-P1 | SSPro 上密度族减先验选取的 margin 随 $\hat p$ 单调增 | Spearman $\rho$ 的 99% CI 下界为正 |
| G-P2 | M2W CLICK 子集内，高 $\hat p$ 层的 margin 符号相对该 benchmark 聚合方向反转 | 最高层 margin 的 99% CI 下界为正 |
| G-P3 | 两 benchmark 的 margin 对 $\hat p$ 的曲线在共同坐标下重合 | 层间配对差的 99% CI 含零 |
| G-P4 | 零点位置落在 $\hat p/(1-\hat p)=\hat q_{\max}$ 附近 | 预测零点与观测零点差不超过一个分层宽度 |
| G-P5 | 准确率对 $\tau$ 的曲线在 $[2\hat\sigma_c, \hat d_{\min}]$ 上有平台，平台宽度与 margin 正相关 | 平台由二阶差分定义，相关系数 CI 下界为正 |
| G-P6 | 两极点 $\tau\to 0$ 与 $\tau\to\infty$ 在 $\pi$ 为先验时收敛到同一值 | 两端差的 99% CI 落在 MDE 内 |
| G-P7 | A1/A2/B3 三者的实测差距不超过实测污染率 $1-\hat\alpha(\tau^*)$ | 推论 1 的不等式在每折成立 |
| G-P8 | Mixed 与 V-only 的 margin z 统计量之比接近 1.28 | 观测比落在 $[1.0,1.6]$ |

主检验是 G-P2。它是唯一能把 §7 那条张力从开放陈述变成机制结论的检验，其余七条为次级，报告时明确标注。

## 4. Kill conditions

| ID | 触发条件 | 处理 |
| --- | --- | --- |
| G-K1 | G-P1 的 Spearman CI 含零 | $p$ 阈值模型不成立，本轮整体报为失败尝试，F1 自变量在讨论节维持开放 |
| G-K2 | G-P2 不成立 | 主检验失败，$p$ 不是自变量，转而如实写“候选解释为提议器质量”，不搜救 |
| G-K3 | G-P6 两极点差超出 MDE | 粒度轴统一叙事不成立，CEV 的解释贡献分支不可用，须改写 |
| G-K4 | G-P7 不等式在任一折被违反 | 引理 1 的实现与理论不符，先查实现，实现无误则 Assumption A1 被证伪 |
| G-K5 | 任一层的行数低于 400 | 该层不报，且不得通过合并层来救回 |
| G-K6 | $\tau$ sweep 触及预注册网格边界 | 视为网格设定失败，按 VUS-SR 的 epoch 触顶同类问题处理，明写而非当作选中值 |

触发即按上述处理，不改判据，记录进 §8 的触发清单。

## 5. 实验设计

数据。SSPro 7B 36-action bank，1,581 行，三谱系 × 12 视角。M2W 统一 prompt 四臂 bank，2,080 行 × 12 forwards，含逐行 trace。AndroidControl 不参与，维持其不承载方法主张的既有决定，且每行仅三候选无法支撑 $\tau$ sweep。

切分。SSPro 按 application GroupKFold(5)。M2W 按 website-fold 内 episode 重采样。$\tau$ 在 inner dev 上选，held-out 每格只出一个数。

$\tau$ 网格（冻结）。以图像对角线归一，对数等距 16 点覆盖 $[10^{-3}, 10^{-0.3}]$，两端各加一个极点配置（精确重合与单块）。同时记录以 $\hat\sigma_c$ 为单位的换算值，$\hat\sigma_c$ 由同源候选散布估计，label-free。若最优点落在 16 点的首末两点，触发 G-K6。

分层。$\hat p$ 分四层，边界取全池分位数 0.25/0.5/0.75，先于看 margin 定死。SSPro 每层约 395 行，MDE 由 0.70 膨胀到约 1.40；M2W CLICK 子集行数在执行第一步时先统计，若任一层不足 400 行则降为三层，此规则先于结果生效。

统计。paired bootstrap 10,000 次 99% percentile CI，与既有口径一致。多重性处理为主检验 G-P2 单独判定，其余七条整体标注为次级探索。

禁止事项。已泄漏的五个 SSPro 格子（63.8836 / 63.8836 / 63.0614 / 62.5553 / 63.2511）不得作为任何优化目标或分层边界依据。$\hat p$、$\hat q_{\max}$、$\hat\alpha$、$\hat d_{\min}$ 一律不得进入任何 runtime 决策路径。

## 6. 可行性风险（如实登记）

第一，$\hat p$ 与 $\hat q_{\max}$ 需要标签，本轮结论因此只能是机制说明。任何把它包装成方法的写法都会重复 TriVUS 的错误，spec 层面直接禁止。

第二，功效偏紧。SSPro 分四层后 MDE 约 1.40，而模型预测的层间 margin 量级在 4 到 8 个点之间，可检出但余量不大。G-K5 就是为此设的。

第三，M2W 上的 $\tau$ 只在 CLICK 子集有定义，非坐标动作的划分必须是类型级嵌套空间级的两层结构，实现上容易写错，A2 那条逐位复现的锚（63.8836）在本轮同样适用于 sequential 与 B3 的重算。

第四，Assumption A2 把 agreement kappa 当 ICC 用，若 G-P8 落在区间外，命题 2.4 的定量部分作废而定性部分（有效票数远小于名义票数）仍成立，报告时分开处理。

第五，本轮不改变 TRIVUS_NOT_PROMOTED，也不改变 VUS-SR 的状态，与那两条线无授权耦合。

## 7. 执行顺序

先 commit 本 spec 与 `gran_prereg.yaml`。然后统计 M2W CLICK 子集行数并据 G-K5 定死层数。然后跑实现锚，sequential 与 B3 在 C-uni 上必须逐位复现既有数字。然后 $\tau$ sweep 与 inner dev 选择。然后按 G-P1 到 G-P8 顺序判定，主检验 G-P2 单独出结论。最后按结果决定讨论节那段的写法，通过则 F1 的自变量改写为池正确率与错误集中度，不通过则维持开放并如实登记本轮为失败尝试。
