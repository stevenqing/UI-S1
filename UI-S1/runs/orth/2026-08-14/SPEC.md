# ORTH — 正交证据通道 探索性摸底 spec

Round: `orth`
Run dir: `runs/orth/2026-08-14/`
Date: 2026-08-14
Status: FROZEN BEFORE ANY ORTH RESULT
GPU: 零；OCR 只允许 CPU engine，不引入 VLM 调用

## 0. 本轮性质

本轮是 exploratory scoping，不是 confirmatory。所有结果仅用于设计后续轮和判断是否值得设计，不可提升为论文结果或方法主张。进入论文必须由后续 confirmatory round重新产生。

网格可宽开；同一量允许多种估计并列为range；除 §7 两条硬规则外无自动 kill/promotion condition。结果无论正负都不改变 F1、Q1、`TRIVUS_NOT_PROMOTED`、VUS-SR、SPLIT、MASK、CEIL 的状态。

## 1. 背景更正

MASK M-G1 的同池4,095子集 calibration 约束固定候选池后的聚合，不识别改变输入后候选如何生成。天花板链 $Y\to A\to c_{1:N}$ 排除固定 $A$ 的重复读数，不自动排除改变输入或引入DOM/OCR等观测通道。MASK 的 pre-GPU stop 保持有效，但不得把它外推成“所有proposal通道已否定”。

项目已有提议侧drop-in pool gain +3.60 pp，而同池有限三票isotonic gain更小；本轮只摸底自带定位证据的新proposal通道。

## 2. 行分类与问题

按冻结原池的最终选择与candidate labels，把行分为：

1. `selected_correct`：原聚合器已选对；
2. `recoverable`：原聚合器错，但候选池至少有一个正确候选；
3. `zero_coverage`：候选池无正确候选。

所有分类依赖GT，只能evaluation-side使用。不同benchmark/arm分别报告，不跨benchmark合并。

目标问题：OCR或结构化元素通道是否同时满足：能覆盖recoverable/zero-coverage rows；自身定位有用；误差与原池充分去相关。项目刻度0.895/0.618/0.398只作描述性参照，不是本轮pass/fail阈值。

## 3. Arm 0 — CEIL口径清理

### 3.1 事实更正

CEIL `recoverable_samples` 已是unique `sample_key`，不是candidate count。M2W `family_samples=8320` 是2,080 base rows × 4 arms；`recoverable_samples=2021`分布于arm-expanded sample keys。SSPro同理有4 arms，`recoverable_samples=968`。因此“2,021/968超过对应行数”不是CEIL实现错误，而是把base row和`benchmark/arm/row_id` sample混为一谈。

CEIL主CI已经是fold-stratified group-clustered bootstrap：M2W group=episode，SSPro group=application。Arm 0不把它描述成未聚类CI。

### 3.2 并列重算口径

对M2W、SSPro分别报告：

- arm-expanded unique sample keys及recoverable counts，逐arm分解；
- unique base `row_id` union counts，说明同一row跨arm重复；
- candidate-context AUROC点估计；
- `group_clustered`：严格复现CEIL的fold内episode/application grouped bootstrap；
- `row_clustered`：fold内按base `row_id`整块重采样，一个row的全部arms、4 OOF contexts和candidates不可拆；
- `unclustered_context_candidate`：把每个context×candidate pair视为IID，10,000次bootstrap，仅作anti-conservative sensitivity。

三种CI都用99% percentile，seed分别为CEIL seed、CEIL seed+1000、CEIL seed+2000。Arm 0还必须逐位复现CEIL point AUROC与group-clustered CI，否则停止Arm 0并记录实现不一致。

C-D2原判定不自动改变。报告M2W三种CI下界相对0.65的margin，由人在scoping报告中记录“维持/不维持为后续设计线索”及理由；该人工记录不改变CEIL status，也不授权实验。

## 4. Arm 1 — OCR通道摸底（SSPro）

### 4.1 输入与engine

输入为冻结SSPro 1,581 screenshots、instruction、target bbox、`ui_type`（977 text / 604 icon）及C-uni pool outputs/labels。至少执行两个互相独立的CPU OCR实现；engine/version/model hash、安装receipt、语言、检测/识别参数和raw output schema在OCR forward前写入并commit preflight。禁止VLM、GPU OCR和结果后换engine隐藏失败。

本轮不预先择engine；所有engine分别报告，另报union/intersection，不跨engine挑best作为结论。

### 4.2 raw retention

每个engine×row原始文本、box/polygon、confidence、orientation及engine metadata逐行JSONL write/flush/fsync。原图不复制。所有匹配与指标只从锁定raw OCR重算。

### 4.3 匹配family

把每个OCR box的识别string与完整instruction并列计算，禁止用GT bbox选text：

- exact substring：trim后OCR string逐字出现在instruction；
- normalized substring：Unicode NFKC、casefold、连续空白折叠、去首尾标点后substring；
- normalized edit similarity：$1-d(s,t)/\max(|s|,|t|)$，其中$t$遍历instruction中长度为$|s|\pm20\%$的token-contiguous spans，取最大。

为避免单字符噪声，字符长度网格 `[1,2,3,4,5]` 全部报告；edit threshold网格 `[0.5,0.6,0.7,0.8,0.9]` 全部报告。exact/normalized substring得分依次为matched字符数、normalized matched字符数；edit得分为similarity后以字符数打tie。最终box tie依次按OCR confidence、较长文本、top、left、raw engine order。

这些网格是scoping surface，不允许从中择一写论文结论。后续confirmatory spec必须在看到本轮结果后另行冻结一个setting并重新产生数据。

### 4.4 四组量

每个engine×matcher×grid point报告：

1. 覆盖：instruction存在至少一个可匹配OCR box的row比例、每行候选box数分布；
2. overlap：`selected_correct/recoverable/zero_coverage × matched/unmatched`交叉表和各类条件match rate；
3. channel accuracy：最高分box中心是否命中target bbox；按`ui_type=text/icon`分别报告；未匹配row计错，同时另报matched-only accuracy；
4. error kappa：OCR row error与原C-uni density error的Cohen $\kappa$，全体及三类内分别报告；常量变量导致不可定义时写`UNDEFINED_DEGENERATE`。

另报OCR预测与原pool top-1位置是否同一target outcome的error overlap。所有label-dependent量evaluation-side。

## 5. Arm 2 — DOM/AX通道摸底（M2W）

先做数据可得性，不做下载或重新抓网页。

冻结统一prompt task JSONL的top-level/step schema不含完整DOM、AX tree或全量候选元素；但`step.pos_candidates`和history中的`pos_candidates`保留GT正候选的tag/attributes/choice片段。这些片段是label-selected positives，不能作为“从全树选最优元素”的可部署channel输入。

在preflight中继续扫描已锁XFER publication artifacts和workspace内原始M2W数据：

- 若找到每行完整候选元素集合或完整DOM/AX snapshot，锁path/hash/schema后执行与Arm 1对应的coverage/overlap/accuracy/kappa；
- 若只有`pos_candidates`或GT bbox，Arm 2状态为`FULL_DOM_AX_UNAVAILABLE_POSITIVE_SNIPPETS_ONLY`并停止，不用GT正候选伪造预测器；
- 若完整tree只覆盖部分rows，报告覆盖并把可用subset限定为描述性，不补抓网页。

使用DOM/AX会改变现有纯截图设定。即使正结果也必须另立setting、对照和主表，不得混入现有视觉主结果。

## 6. Arm 3 — 融合权重可行性草算

不做新模型实验。用SSPro C-uni冻结row outcomes及synthetic binary channel模拟log-odds evidence fusion。

新通道假设网格：accuracy `[0.50,0.55,...,0.95]`，error kappa `[-0.2,-0.1,0,0.1,...,0.8]`。对每个格点用两种construction并列：

- empirical constrained coupling：在原pool error strata内解2×2 joint probabilities，使marginal accuracy/kappa尽可能接近目标；不可行时投影到最近可行joint并报告实际值；
- Gaussian-copula Bernoulli simulation：100个固定seeds，每seed 1,581 rows，校准latent correlation到目标error kappa。

比较三种视觉权重：raw $N=12$、MASK full-pool generalized $N_{\mathrm{eff}}=1.5936767669403409$、unit visual channel=1。先验为冻结full-set base rate；channel likelihood ratio由模拟joint的sensitivity/specificity确定。报告synthetic fused accuracy、相对原pool变化和两construction range。该表只是设计headroom，不是性能主张。

任何使用GT构造的coupling都不得成为runtime方法。

## 7. 硬规则

1. ORTH任何结果不可提升为论文结果或方法主张；后续confirmatory round必须重新产生。
2. 行分类、channel accuracy、$\kappa$、交叉表及synthetic coupling均依赖labels，只能evaluation-side，不得导出runtime rule。

沿用约束：五个已泄漏SSPro cells不得作为优化目标；后续confirmatory对照必须包含majority加dev-selection，不得只报best-single。

## 8. 留存与执行顺序

所有输入先hash-lock。派生row JSONL逐行write/flush/fsync并写SHA manifest；OCR raw完整保存；独立备份根 `/scratch/workspaceblobstore/orth/2026-08-14`；最终`STATUS.json`记录manifest。禁止递归删除raw OCR或predictions JSONL。

顺序：

1. commit本SPEC；
2. 输入/engine/DOM availability preflight并commit；
3. implement/commit/执行Arm 0；
4. 安装并锁至少两个CPU OCR engines，commit raw writer后执行Arm 1 raw OCR；
5. 锁raw manifests，再实现/执行Arm 1派生分析；
6. 执行Arm 2 availability branch；
7. implement/execute Arm 3；
8. 生成scoping report，人工记录CEIL C-D2设计解释但不改status；
9. retention、STATUS、push。

最终产出至少包括：Arm 0三种CI表；Arm 1 class×match交叉表、text/icon accuracy与kappa；Arm 2 availability表（若可用再加对应指标）；Arm 3二维headroom表；以及一个只面向后续confirmatory设计的结论。