# CEIL — 收口诊断 预注册 spec

Round: `ceil`
Run dir: `runs/ceil/2026-08-14/`
Date: 2026-08-14
Status: FROZEN BEFORE ANY CEIL RESULT
GPU: 零；两臂均为冻结产物上的确定性重算

## 0. 范围

本轮用于论文收口，不产生方法，不改变 F1、Q1、`TRIVUS_NOT_PROMOTED`、VUS-SR、`SPLIT_STOPPED_PRE_GPU_Z_K6_GEOMETRY_AND_Z_K7_LOW_N` 或 `MASK_STOPPED_M_K1_IDEAL_NEFF_GAIN_BELOW_MDE`。两臂独立判定，互不作为前置。

Arm A 把 MASK gate 扩为编号 post-hoc 结果。Arm B 是唯一可能授权另立新 spec 的诊断；其结果不授权本轮内训练、重加权或 runtime selector。

## 1. Arm A 的证据地位

MASK 已观察到 SSPro full-pool generalized $N_{\mathrm{eff}}=1.5937$ 与理想三票预测 +0.538 pp。Arm A 不可能因本 spec 变成 confirmatory，全程标注 post-hoc，沿用 F2 先例。本 spec 只冻结从已见 gate 到论文图表的报告协议。

既有 NEFF round 已否定跨池一维强律。Arm A 只给 benchmark/arm-specific 描述，不恢复普适 $N_{\mathrm{eff}}\to$ accuracy 定律。

## 2. Arm A 冻结协议

### 2.1 数据与 panels

SSPro panel 是 7B 36-action bank 中 C-uni 的 12 个固定 source slots，枚举全部 $2^{12}-1=4095$ 个非空子集。

M2W 使用统一 prompt bank 的四臂 `C_uni/C_cond/C_rand/C_self`，每臂 12 个冻结 slots，逐臂枚举 4,095 子集。四臂分别拟合、分别报告；C-uni 是正文 panel，其余三臂为附录稳健性。不得跨臂或跨 benchmark 合并。M2W 的 GRAN G-K6 是 $\tau$ 不可识别，不影响本臂，因为 Arm A 不做模式提取。

AndroidControl 不纳入，维持附录决定。五个已泄漏 SSPro cells 不作为阈值或拟合选择目标。

### 2.2 $N_{\mathrm{eff}}$：严格复用 MASK

草案中的 equicorrelation 简式由下列 MASK 实际 estimator 取代，以保证逐位一致。对 panel、outer fold $f$ 和子集 $S$，在 outer-development rows 上计算 source failure indicators 的 pairwise Cohen $\kappa$ matrix $R_{f,S}$；对不可定义 pair 置 0 并计数。对角线为 1：

$$
N_{\mathrm{eff},f}(S)=\frac{|S|^2}{\mathbf 1^\top R_{f,S}\mathbf 1}.
$$

分母非正则该子集该折无效。cross-fitted $N_{\mathrm{eff}}(S)$ 按各 outer-heldout fold 行数加权平均。accuracy 同样拼接 outer-heldout outputs。

SSPro full-pool 值冻结复现锚为 1.5936767669403409。注意它不是支撑上界；已见 MASK 4,095 子集的真实最大支撑为 1.7073149168564605。每个 M2W panel 的 full 值与最大支撑待算并报告。

### 2.3 聚合器

每个 panel 分别报告两种聚合器：

- SSPro `density_B3` 严格复用 `sourcebias_common.b3_select_index`；`F1_majority` 严格复用 outer-development source-reliability priority。
- M2W `density` 严格复用 E1 的 `ours` unified product-action sequential aggregator；`majority` 严格复用 action plurality 后 outer-development source priority。

任意子集保持原 12-slot 相对顺序。reliability/priority 仅用 outer-development rows。候选 error indicator 使用 benchmark 原生 success 定义，与 E1/MASK 完全一致。

### 2.4 曲线与有限外推

每个 panel×aggregator 把相同 cross-fitted $N_{\mathrm{eff}}$ 的 accuracy 先取均值，unique x 等权。

主支撑内拟合为 `IsotonicRegression(increasing=True, out_of_bounds="clip")`。图中画出 full-pool x 和观测最大 x；最大 x 外一律标注 extrapolation。

有限外推目标固定为

$$
x_1=x_{\mathrm{full}}+\frac{3}{1+2\times0.6}=x_{\mathrm{full}}+1.363636\ldots.
$$

isotonic 在最大支撑外只沿最后两个不同 isotonic x-threshold 的割线延伸到 $x_1$；斜率为非负，预测裁剪到 `[0,1]`。该有限-x 数字在饱和假设下作为 optimistic upper bound 报告。若不足两个不同 threshold，则斜率置 0 并标记。禁止把线性延伸到无穷。

### 2.5 饱和参数曲线与 $\Delta_\infty$

唯一 sensitivity family 为

$$
f(x)=a-b\exp(-cx),\qquad 0\le a\le1,\quad 0\le b\le1,\quad 10^{-8}\le c\le100.
$$

对 unique-x mean accuracy 等权最小二乘。使用 `scipy.optimize.least_squares`，容差 `xtol=ftol=gtol=1e-12`，`max_nfev=100000`。deterministic multistart 为

- $a_0\in\{\max(y),\min(1,\max(y)+0.05),1\}$；
- $b_0\in\{0.01,0.1,0.5\}$；
- $c_0\in\{0.01,0.1,1,10\}$。

越界初值裁剪入 bounds 内。取 SSE 最小解；SSE 在 $10^{-12}$ 内并列时按 `(a,b,c)` 字典序最小。全部 fit 失败则该 panel×aggregator 的 $\Delta_\infty$ 为 `NA_FIT_FAILURE`，不更换曲线。

主报告量只由参数曲线定义：

$$
\Delta_\infty=a-\operatorname{acc}_{\mathrm{observed}}(S_{\mathrm{full}}).
$$

isotonic 不定义 $\Delta_\infty$。次级量为 full $N_{\mathrm{eff}}$、真实支撑、有限 $x_1$ isotonic upper bound、参数曲线在 $x_1$ 的预测、两者差。

### 2.6 raw 非单调与报告

不以 noisy 4,095-point scatter 的任一局部下降取消参数 fit。描述性报告：按 unique x mean accuracy 排序后的相邻下降比例、最大相邻下降和 Spearman $\rho$。参数 family 本身由 $b,c\ge0$ 强制单调。不得事后平滑、删点或换 family。

### 2.7 Bootstrap

10,000 replicates，99% percentile CI。每个 replicate 在每个 outer fold 内按 group 有放回抽样，抽中 group 时保留其全部 rows：SSPro group 为 application，M2W group 为 episode。每次 replicate 从 group sufficient statistics 重新计算 fold-level $R$、4,095 个子集 accuracy、isotonic 和参数 fit；不得固定原始 $N_{\mathrm{eff}}$ 或曲线。

seed 为 `20260814 + panel_index*100 + aggregator_index`；panel order 为 `SSPro/C_uni, M2W/C_uni, C_cond, C_rand, C_self`，aggregator order 为 density、majority。

Arm A 无 kill condition，始终描述性报告。

## 3. Arm B 冻结协议

### 3.1 输入与完整性

主输入是第三 nonce 原子发布目录 `runs/trivus/2026-08-13/sequential_exploratory/` 的 240 个 artifacts 及 `MANIFEST.json`。先验证 manifest status、artifact count、每文件 bytes/SHA-256，禁止读取未列入 manifest 的 cheap/verifier产物。

候选 labels 从该发布所锁定的 VUS/Android private manifests 读取；public rows、candidate order、fold/group、blind visual score 与 strongest fallback 从 TriVUS assembly dependencies及 frozen baseline loader读取，所有路径和 hash 在 CEIL preflight 再锁一次。

blind visual score 是 blind prediction 的 `label_probabilities`，按 `display_to_candidate` 映回 candidate index，严格复用 `trivus_data.restore_visual_values`；不从 115 维 feature array 猜列。

### 3.2 可回收子集

unique sample 满足：frozen strongest fallback 错，且 candidate labels 中至少一个为正。strongest 定义逐 benchmark由 `finalize_trivus.frozen_baselines` 恢复，不重新指定。

该子集依赖 labels，只是 evaluation-side 分析，不得导出 runtime rule。SSPro 与 M2W 为 decision benchmarks；AndroidControl只描述。M2W unique recoverable samples 少于 100 时降为描述，不进入 overall branch。SSPro 若少于100同样降为描述，尽管预期不会发生。

### 3.3 四个 OOF contexts

每个 unique sample 必须恰有四个 OOF contexts，对应四个 `outer != heldout fold` 模型。主条件 AUROC 保留全部四 contexts，与历史边缘 AUROC 0.831/0.844/0.813 同口径；bootstrap 时以 unique sample/group 为不可拆分块，选中 sample 后纳入其四个 contexts及全部候选。

top-1 与强制改判先对四 contexts的 candidate probabilities逐候选取算术均值，stable descending order，tie 取 candidate index 较小者，形成 unique-sample prediction。verifier同理。

### 3.4 度量

主度量：可回收 subset 中 cheap `candidate_probabilities` 的候选级 AUROC，正类为正确候选，负类为同行其余候选。所有四 contexts pooled，但 bootstrap按 sample/group整块。

对照：

1. 同一 subset/context/candidate pairs 上的 blind visual probability AUROC；
2. 理论 random 0.5；
3. second verifier probability AUROC，仅作完整性记录。

次级量：四-context mean cheap score 的 unique-sample top-1 hit rate；verifier/visual同口径top-1；以及 evaluation-oracle policy“可回收 subset 上强制使用cheap top-1、其余行保持 strongest”的全 benchmark accuracy change。后者报告点估计与CI，不作判据，且不得描述为 runtime policy。

### 3.5 CI 与判定

10,000-replicate 99% percentile CI。按 fold 分层、fold内 group有放回抽样；SSPro group=application，M2W group=episode，Android group=episode/trajectory的冻结 public `group`。每个抽中 group保留其全部 unique samples、每个 sample的四 contexts与全部 candidates。seed：SSPro `20261014`、M2W `20261114`、Android `20261214`。

每个 decision-eligible benchmark独立判定：

| ID | 条件 | 结论 |
| --- | --- | --- |
| C-D1 | cheap条件AUROC 99% CI上界 `<0.60` | 该 benchmark 无可用翻转信号 |
| C-D2 | 99% CI下界 `>0.65` | 该 benchmark有信号，授权另立全候选重加权spec |
| C-D3 | 其余 | 不定，两个方向都不主张 |

overall branch：任一 decision-eligible benchmark触发 C-D2，则 overall=`OPEN_NEW_SPEC_C_D2`；否则若所有 eligible benchmarks均C-D1，则 overall=`CLOSE_C_D1`；其余为 `FREEZE_C_D3_INDETERMINATE`。若无 eligible benchmark，则 overall为C-D3。C-D2 只授权写新 spec，不授权本轮训练或实验。

## 4. 通用约束与执行顺序

两臂均零 GPU、确定性重算。逐行衍生 JSONL write/flush/fsync；manifest记录实现 commit、seed、输入/输出SHA-256。独立备份根 `/scratch/workspaceblobstore/ceil/2026-08-14`，最终写 `STATUS.json`。禁止结果后更换 estimator、curve、subset、重复context口径或阈值。

模型角色核对：`Qwen2.5-VL-7B-Instruct` 是 SPLIT 中缺失的 deferred checkpoint；`UI-TARS-7B-SFT` 是 SSPro bank lineage。二者不冲突，也不进入 CEIL forward，因为本轮无 forward。

顺序：

1. commit 本 spec 与 `configs/ceil_prereg.yaml`；
2. 锁所有输入、240-artifact manifest、labels/public/blind predictions/strongest dependencies，并记录模型命名核对；
3. implement并commit Arm B；
4. 执行 Arm B，按 C-D1/D2/D3冻结 overall branch；
5. implement并commit Arm A；
6. 执行 Arm A，生成五个panels及表；
7. 合并报告：C-D1或C-D3冻结主表；C-D2只另立新spec；
8. retention、STATUS、push。