# MASK — 共识遮挡作为提议器 预注册 spec

Round: `mask`
Run dir: `runs/mask/2026-08-14/`
Date: 2026-08-14
Status: FROZEN BEFORE ANY MASK RESULT OR FORWARD
GPU: 第二阶段需要，第一阶段零 GPU

## 0. 范围、授权与命名

本轮是 exploratory pilot。不改变 F1、Q1、`TRIVUS_NOT_PROMOTED`、VUS-SR 状态，不产生可部署方法，不定义确认集。

本轮是新一轮，不是 SPLIT 的修订。SPLIT 的观测量是裁剪下的置信度，用途是 verifier，终点是稀有事件检测的判别力。本轮的观测量是遮挡下的坐标预测本身，用途是 proposer，终点是与原池的误差相关。三者全变，因此挂在 SPLIT 下面会构成失败后换定义。SPLIT 的 `SPLIT_STOPPED_PRE_GPU_Z_K6_GEOMETRY_AND_Z_K7_LOW_N` 结论保持不变，其 $\Delta_2=6.45$ pp、门内正例率 8.59% 与 geometry failure 26.79% 照常写进 limitation。

X2 是最大化 containment 的确认式 zoom，已永久取消。RegionFocus 是执行出错触发的 zoom。本轮不裁剪、不改变图像尺寸与分辨率，因此与二者形式不同。遮挡本身是旧技术；允许的主张只限于把共识模式遮挡用作额外提议来源，不得声称发明遮挡归因。

### 0.1 模型命名核对

三个名称承担不同角色，不得互换：

- `GTA1-7B`：本轮唯一 proposer，checkpoint READY；
- `UI-TARS-7B-SFT`：原 36-action bank 的三个候选谱系之一，不是本轮 forward 模型；
- `Qwen2.5-VL-7B-Instruct`：SPLIT 中缺失的 deferred probe checkpoint，不进入 MASK。

SPLIT 报告与 preflight 的命名并不矛盾。第二阶段只允许加载已锁定的 GTA1 checkpoint，禁止用 UI-TARS 或 Qwen2.5 替换。

## 1. 假设与机制主张

天花板定理给出 $Y\to A\to c_{1:N}$，$I(Y;c_{1:N})\le I(Y;A)$，与 $N$ 无关。遮挡把驱动 $A$ 收敛到 $M_1$ 的像素证据物理移除，产生感知状态 $A'$；本轮把它视作近似 $do(\text{mask }M_1)$。

机制主张是：遮挡 proposer 的误差与原池误差相关低于项目跨模型族刻度 $\kappa=0.398$。视角轴 0.895、同族跨规模 0.618 与跨族 0.398 只作项目自有诊断刻度，不是理论常数。

## 2. Proposer 而非 verifier

SPLIT 门内负正比为 $1085/102=10.64$。任何先承诺两个模式再判断是否翻转的 verifier 都承受该基率。本轮不判断哪一行应翻转，而把遮挡前向作为新票加入池中。

有效票数记账仅作本 benchmark 的 exploratory compute gate：

$$
N_{\mathrm{eff}}' = N_{\mathrm{eff}} +
\frac{m}{1+(m-1)\kappa_{\mathrm{in}}}(1-\kappa_{\mathrm{new}}).
$$

固定 $m=3$、$\kappa_{\mathrm{in}}=0.6$。M-G1 使用理想上界 $\kappa_{\mathrm{new}}=0$。既有 NEFF round 已否定跨池一维 $N_{\mathrm{eff}}\to$ accuracy 强律（最佳 residual SD 7.30 pp）；因此本轮只在同一个 SSPro C-uni pool 的冻结子池 panel 内做 monotone calibration，不得把它解释为普适规律或恢复 NEFF/N2。

## 3. 核心风险

遮挡 $M_1$ 在 $M_1$ 正确行上可能造成结构性损害，因此低 $\kappa$ 不充分证明新信息。主终点是池错层的 M-P1；$\kappa$ 降为分层诊断。

池侧改进可能被强聚合器吸收。本轮同时报告 B3 density 与 F1 `majority`。后者严格指 outer-development source-reliability priority，不是新定义的坐标多数。

## 4. 冻结方法

### 4.1 输入、fold 与模式

输入是 ScreenSpot-Pro 7B 36-action bank，共 1,581 行、三个谱系、每谱系 12 views。MASK 的原始 C-uni pool 严格为三个谱系各前四个 views，共 12 票。GRAN 的 SSPro $\tau^*$ 虽在完整 36-action bank 上 nested 选出，但按 outer fold原样应用到 C-uni 12 票：

- fold 0: `0.0022908676527677724`
- fold 1: `0.0015135612484362087`
- fold 2: `0.012022644346174132`
- fold 3: `0.0034673685045253167`
- fold 4: `0.0034673685045253167`

$\tau^*$ 的单位是候选坐标除以图像对角线后的距离。模式用 GRAN deterministic complete-link 提取；候选票数、source-reliability 总和、最大 reliability、最早冻结 candidate order 依次打破 mode 并列。source reliability 只在对应 outer-development rows 拟合。$M_1$ 为最大模式；若无可解析候选则该行实现失败。

$M_1$ representative 是块内 outer-development source reliability 最高的候选，tie 按 source 名再按冻结 candidate order。$M_2$ 是第二大模式，使用相同 mode tie 规则；`M2_correct` 表示块内至少一个候选命中 target bbox。若只有一个 mode，则保留该行、令 `M2_correct=false`，不得更换 $\tau$ 或合成第二模式。

### 4.2 信息性遮挡

$M_1$ 质心是在原像素坐标中对块内候选求算术均值。像素半径为

$$
r=2\tau^*\sqrt{W^2+H^2}.
$$

保持原图 $W\times H$、长宽比及 processor resize 参数不变。信息 mask 包含所有像素中心 $(x+0.5,y+0.5)$ 满足到 $M_1$ 质心距离不超过 $r$ 的像素；超出图像的圆自然截断。填充值为原图逐 RGB channel 的全局像素均值，按 float 计算后以 round-half-to-even 转为 `uint8`。alpha channel 若存在先转 RGB。遮挡图像不落盘。

逐行记录质心、归一化 $\tau^*$、像素半径、信息 mask 像素数、RGB fill、图像 SHA-256、原始与 processor resize size。

### 4.3 空遮挡

空遮挡必须与信息 mask 有完全相同的离散像素数 $A$ 和相同 RGB fill。设图心 $o=((W-1)/2,(H-1)/2)$，$d=\|M_1-o\|$。候选控制中心按角度 $0,1,\ldots,359$ 度搜索：

$$
q_\theta=o+d(\cos\theta,\sin\theta).
$$

只保留位于图像内的中心。对每个候选中心，取全图中到中心距离最小的 $A$ 个像素中心作为空 mask；距离并列按 row-major pixel order。该定义保证面积精确匹配，即使信息圆在边界被截断。空 mask 不得包含任何 C-uni complete-link mode 的质心。首个满足条件的角度获选；若 360 个角度均不可行，该行标记 `empty_mask_infeasible` 并从第二阶段两层同时排除，不补抽。第一阶段必须先报告其预期可行率；若不可行率超过 15%，视为实现/对照失败并在 GPU 前停止。

### 4.4 GTA1 前向与行级 proposer

唯一模型为 GTA1-7B。prompt 为 SPLIT 冻结的 pixel-point prompt。信息遮挡采 $k=3$，空遮挡采 $k=1$；temperature `0.9`、top-p `1.0`、max new tokens `32`。

seed 为 SHA-256(`row_id|GTA1-7B|mask_kind|sample_index|20260814`) 前八字节模 $2^{31}-1$。输出使用首个可解析坐标对，并按 processor resize 映射回原图坐标。任一 response、token IDs、decoded tokens、逐 token logprob、解析坐标、seed、mask 参数与 resize size完整保留。

三次信息遮挡点用该行 outer-fold $\tau^*$ 做 GRAN deterministic complete-link；最大模式的代表点按 source 顺序从块中取最早 sample。mode tie 依次按票数、最早 sample。该代表点是 M-P1/M-P2/M-P3 的唯一行级信息 proposer。空遮挡单票直接作为行级 control proposer。任一必要 sample 无法解析时，该模型×行无效；无效率超过 15% 时停止并修实现，不进入 endpoint。

### 4.5 池聚合

三个信息遮挡 sample 作为三个等权候选加入原 C-uni 12 票，形成 15 票池。报告两种冻结聚合器：

- `density_B3`：严格复用 `sourcebias_common.b3_select_index`；
- `F1_majority`：严格复用 F1 的 outer-development source-reliability priority。三个 sample 是三个新 source slots，其 reliability 只在 outer-development sampled rows 上估计；tie 按原 12 slots 的冻结顺序后接 mask sample 0、1、2。

空遮挡票不加入主池，只用于 M-P2 control。本轮不定义 runtime gate；所有入选行无条件遮挡。

## 5. 第一阶段：零 GPU

第一阶段必须在任何 subset manifest、GPU authorization 或模型 forward 前完成。

### 5.1 SPLIT verifier 收口曲线

在 SPLIT 已提交 held-out row outputs 上，对 $g\in[0.25,0.40,0.55,0.70,0.85]$ 分别计算 gate rows $n$、M2-only positives $P$、其余 gate rows $N=n-P$、$\pi_0=P/n$ 与 $N/P$。不重新选择 $g$。

假设 AUROC 网格冻结为 `[0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.88,0.90,0.95]`。在 equal-variance Gaussian model 下，正类 score $N(d',1)$、负类 $N(0,1)$，$d'=\sqrt2\Phi^{-1}(\mathrm{AUROC})$。对 threshold 网格 `[-8,8]` 步长 `0.001` 最大化全 1,581 行净收益

$$
G=(P\cdot\mathrm{TPR}-N\cdot\mathrm{FPR})/1581.
$$

报告最佳 threshold、TPR、FPR 与 $G$；等值取更高 threshold。此项只关闭 verifier 路线，不参与 MASK authorization。

### 5.2 benchmark 内 $N_{\mathrm{eff}}$ calibration

对 C-uni 12 个固定 source slots 枚举所有 $2^{12}-1=4095$ 个非空子集。每个 outer fold：

1. 用 outer-development rows 计算该子集的 failure-kappa 矩阵 $R$；对不可定义的 pair kappa 置 0，并显式计数；
2. 计算 generalized $N_{\mathrm{eff}}=K^2/(\mathbf1^TR\mathbf1)$；若分母非正则该子集无效；
3. 在 outer-heldout rows 上分别计算 `density_B3` 与 `F1_majority` accuracy；所有 reliability 只用 outer-development；
4. 五折拼接后，每个子集得到 cross-fitted accuracy 与按 test-row 数加权的 $N_{\mathrm{eff}}$。

对两种聚合器分别拟合 `sklearn.isotonic.IsotonicRegression(increasing=True, out_of_bounds="clip")`。相同 $N_{\mathrm{eff}}$ 先取 accuracy 均值，每个唯一 x 等权。基线 $x_0$ 是完整 12 票子集的 cross-fitted $N_{\mathrm{eff}}$；理想 $x_1=x_0+3/(1+2\times0.6)$。预测 gain 为 $f(x_1)-f(x_0)$。同时报告 leave-one-fold-out sensitivity，但不据此择曲线。

这只是同 benchmark 内 empirical calibration，不声称因果或跨池外推。M-G1 取两种聚合器预测 gain 的最大值，作为最有利的理想上界。

### 5.3 基率与 mask 可行性

按 outer-heldout C-uni 12 票计算：`density_B3` 错误率；在 density 错误行中 $M_2$ 正确率；pool-correct/pool-wrong 两层行数；信息 mask 与空 mask 参数可行率。模式正确定义为块内至少一个候选命中 target bbox。

### 5.4 M-G1

MDE 固定为 0.007。若 `max(predicted_gain_density_B3, predicted_gain_F1_majority) < 0.007`，触发 M-K1 并在 GPU 前终止。等于 0.007 视为通过。若空 mask 不可行率超过 15%，同样在 GPU 前停止，但记为 control implementation failure，不改 M-G1 数值。

## 6. 第二阶段：GPU pilot

仅在 M-G1 通过且空 mask 可行率不超过 15% 后执行。按 outer-heldout `density_B3` 结果分层：

- pool-wrong 目标 300 行；
- pool-correct 目标 200 行。

每层按 `(outer_fold, application)` 分层，以 SHA-256(`stratum|row_id|20260814`) 排序做 proportional largest-remainder allocation；每个非空 cell 至少一行，超过目标时按 cell size 比例分配，cell 内取 hash 最小 rows。若层总数低于目标则纳入全层。抽样概率与 inverse-probability weight 写入 manifest。任一层可行 rows 少于 150，触发 M-K7，降级为观察性报告。

row ID manifest 必须在 GPU runner 和 authorization 前 commit。GPU runner implementation commit 必须先于一次性 nonce authorization commit。最多使用 8 张 GPU，禁止 signal、暂停或 reprioritize 外部 protected process。

## 7. 终点与统计

统计使用 application group、outer-fold stratified paired bootstrap，10,000 次，99% percentile CI，seed `20260814 + endpoint_index`。全分布估计使用冻结 inverse-probability weights。endpoint 顺序固定为 M-P2、M-P1、M-P3、M-P4、M-P5；M-P2 不过即停。

### M-P1（主）

pool-wrong 层每行差值为 `information_proposer_correct - M2_correct`。无 $M_2$ 时按 §4.1 令 baseline 为 false。报告加权均值及 99% CI。CI 下界严格大于 0 才通过。

### M-P2

pool-correct 层比较行级 proposer error：`empty_proposer_wrong - information_proposer_wrong`。差值 99% CI 下界严格大于 0，表示信息遮挡优于同面积空遮挡并通过 control。CI 含零或为负触发 M-K2。

### M-P3

分别在 pool-wrong 与 pool-correct 层计算 Cohen's $\kappa(\text{information proposer error},\text{original M1 representative error})$，其中 representative 按 §4.1 冻结。原请求的层内 `original density pool error` 在分层后是常量，数学上不可定义；只报告其退化状态，不生成数值。另在两层合并样本上报告 proposer error 对 original density pool error 的加权 $\kappa$ 作为描述性统计。pool-wrong 层 M1-error kappa 必须小于 0.398 才可称新通道。若期望概率为 1 或任一变量无方差导致 kappa 不可定义，则 endpoint 不通过。

### M-P4

三个信息票加入原 12 票后，分别计算 `density_B3` 与 `F1_majority` 相对原池的全分布重加权 accuracy change 与 99% CI。若 density CI 下界大于 0 而 majority CI 包含 0，触发 M-K5，结论为 absorbed。任一聚合器重加权 change 的 CI 上界小于 0，触发 M-K6。

### M-P5

在 sampled rows 上用与 §5.2 相同 generalized failure-kappa 定义计算实测 $N_{\mathrm{eff}}'$。报告

$$
\frac{N_{\mathrm{eff}}'-N_{\mathrm{eff}}}{3(1-\hat\kappa_{\mathrm{new}})/(1+2\hat\kappa_{\mathrm{in}})}.
$$

$\hat\kappa_{\mathrm{new}}$ 是三个新票与原 12 票 pairwise failure kappa 的均值，$\hat\kappa_{\mathrm{in}}$ 是三个新票内部 pairwise failure kappa 的均值。比值落在 `[0.7,1.4]` 记为一致；分母非正或 kappa 不可定义则不通过。

## 8. Kill conditions

| ID | 触发条件 | 处理 |
| --- | --- | --- |
| M-K1 | M-G1 理想预测 gain `<0.70` pp | GPU 前终止 |
| M-K2 | M-P2 CI 含零或为负 | 分布外效应无法剥离，终止 |
| M-K3 | M-P1 CI 含零或为负 | 遮挡 proposer 无增量，失败 |
| M-K4 | pool-wrong M1-error $\kappa\ge0.398$ 或不可定义 | 不得称新通道 |
| M-K5 | density 正而 F1 majority 不可区分 | 判为 absorbed，不得只报 density |
| M-K6 | 任一主聚合器全分布 change CI 上界 `<0` | 损害超过收益，终止，不加 gate |
| M-K7 | 任一层可行 rows `<150` | 观察性报告，不出判定 |
| M-K8 | empty-mask infeasible 或 generation invalid rate `>15%` | 修实现/对照或 pre-endpoint stop |

失败后禁止更换遮挡方式、fill、半径、proposer、三票合成、空遮挡搜索或 aggregator 再报。替代设定属于新一轮。

## 9. 数据、留存与执行顺序

M2W 不进入，因其 GRAN $\tau$ grid 触发 G-K6。AndroidControl 不进入。五个已泄漏 SSPro cells 不得作为优化目标或阈值依据。

逐行 JSONL 每条 write/flush/fsync；lane 完成写 SHA-256 manifest。原始 response、token IDs、decoded tokens、logprobs、坐标、mask 参数、resize size完整保留。遮挡图像不落盘。独立备份根为 `/scratch/workspaceblobstore/mask/2026-08-14`。清理脚本禁止递归删除 `raw/` 与 `predictions*.jsonl`。

顺序冻结为：

1. commit 本 SPEC 与 `configs/mask_prereg.yaml`；
2. 锁输入与 GTA1 preflight，完成 UI-TARS/Qwen2.5 角色核对；
3. 实现并 commit 第一阶段；
4. 执行三项零 GPU分析与 M-G1；
5. 若通过，冻结并 commit分层 subset manifest；
6. commit GPU runner；
7. 单独 commit 一次性 nonce authorization；
8. 执行 GTA1 lanes；
9. 按 M-P2、M-P1、M-P3、M-P4、M-P5 adjudicate；
10. 无论结果如何，将第一阶段三项与最终结论写入 limitation，并完成独立 retention。