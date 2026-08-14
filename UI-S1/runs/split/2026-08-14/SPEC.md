# SPLIT — 证伪式裁剪探针 预注册 spec

Round: `split`
Run dir: `runs/split/2026-08-14/`
Date: 2026-08-14
Status: DRAFT，未 commit。必须先于任何探针结果进 git。
GPU: 需要。本轮是本项目第一条非零 GPU 线，授权范围见 §0。

## 0. 范围与授权（先定死）

本轮是 exploratory pilot，不进当前论文的任何主表。当前论文的状态不变，F1 仍为主结果，Q1 限定不变，`TRIVUS_NOT_PROMOTED` 与 VUS-SR 状态不变。本轮不产生可部署方法，不产生 confirmation 结论，未触碰确认集不在本轮定义范围内。

与已取消项的关系必须写明。X2 自适应 zoom 已永久取消，其形式是最大化 containment 的确认式裁剪。本轮是相反方向，构造刻意排除部分后验质量的裁剪，目的是最小化 containment。二者的目标函数符号相反，因此本轮不是 X2 的复活。若审稿或内部质疑提出此点，以 §2 的不等式作答。

与文献的关系必须写明。以 token perplexity 或置信度作为观测量并非新事，AutoFocus 用逐 token 困惑度构造各向异性空间概率场，UI-Zoomer 融合空间共识与 token 置信度做可靠性门控。本轮的差别只在于观测量是在证伪式裁剪下读取的，而非确认式裁剪下。贡献主张必须限定在这一点上。

### 0.1 模型多重性与本地可用性

三模型分别执行、分别报告，禁止跨模型择优或合并后宣称通过。

- 主模型：Qwen3-VL-8B-Instruct。Z-P1 至 Z-P5 的正式 pilot 判定只由该模型决定。
- secondary replication：GTA1-7B。只报告同方向性与效应量，不改变主判定。
- deferred secondary replication：Qwen2.5-VL-7B-Instruct。旧 UI-Zoomer 本地 checkpoint 当前缺失；只有在任何 probe 前恢复 checkpoint、锁定 index SHA-256 并提交 preflight 后才可执行。缺失不阻塞主模型，也不得用其后续结果替换 Qwen3 主判定。

三个模型共用同一冻结 balanced subset、窗口几何、$g/t$ 网格、seed 与统计代码。若三模型全部可执行，目标前向预算约为 $3\times 3\times 2n_+\times k\approx 9{,}450$；当前可执行两模型约为 6,300。

## 1. 动机（上一轮推导的直接推论）

设 $Y$ 为目标元素，$A$ 为这一行的感知状态，池子满足 $Y\to A\to c_{1:N}$。由数据处理不等式 $I(Y;c_{1:N})\le I(Y;A)$，且右端与 $N$ 无关。因此任何只读取池子的聚合器有一个与预算无关的上界。$N_{\text{eff}}\approx 1.81$、第三谱系饱和、NOA 的有效样本数涨而精度不涨、四个选择规则独立失败，都是这条上界的投影。

要突破上界必须引入通道 $V$ 满足 $P(V\mid Y,A)=P(V\mid Y)$。本轮检验的假设是，在排除当前主模式的裁剪下读取模型自身的置信度，可以构成这样一个近似正交的通道。

## 2. 推导

### 2.1 命题 1（确认式裁剪的信息上界）

把探针看作对模式集合的一个二分 $S$ 与 $S^c$，记 $\pi=P(Y\in S\mid c_{1:N})$。任何以该二分为唯一区分对象的观测 $Z$ 满足

$$
I(Y;Z)\le H(\pi),
$$

在 $\pi=1/2$ 取最大，在 $\pi\to 1$ 时趋于零。

现有 zoom 方法的裁剪选择准则都是提高 $\pi$。SSPro 的 containment 从 99.94 起步意味着 $\pi\approx 1$，对应 $H(\pi)\approx 0$，这与视角轴 $\kappa_v=0.895$ 是同一件事的两面。所以“裁得准”和“裁得有信息”在数学上是对立目标，不是可以同时优化的两个指标。

### 2.2 命题 2（噪声信道下的可用区间）

设读取 $Z$ 后对二分的判决错误率为 $e$，则

$$
I(Y;Z)\le\min\big(H(\pi),\,1-H_b(e)\big).
$$

以等方差高斯近似，AUROC $=0.65$ 对应 $d'\approx 0.55$、$e\approx 0.39$、容量约 $0.03$ bit；AUROC $=0.75$ 对应 $d'\approx 0.95$、$e\approx 0.32$、容量约 $0.10$ bit。这说明弱观测量几乎不可能推翻悬殊的先验。

### 2.3 推论（触发门必须存在）

探针只有在似然比超过先验优势比时才改变判决。记 top-2 模式权重为 $w_1\ge w_2$，先验优势比为 $w_1/w_2$。当 $w_1/w_2$ 很大时，任何 AUROC 在 0.7 附近的观测量都不足以翻转。因此探针不得对全部行施加，必须限制在 $w_2/w_1\ge g$ 的模糊行上。这同时使 $\pi$ 接近 $1/2$，即命题 1 的信息上界最大处。$g$ 由 nested inner dev 从冻结网格选出。

触发门只用池内可算量 $w_2/w_1$，不需要 GRAN 的 $\hat p$，不需要标签。

### 2.4 预算记账

一次模型探针含 $W_1,W_2,W_0$ 三个窗口，每个窗口采 $k=3$ 次，共九次前向/模型/行。主模型目标预算约 $9\times2n_+\approx3{,}150$ 次。三模型全部执行时约 9,450 次；当前两模型约 6,300 次。本轮不主张 $\log_2 3$ 的理想数字。

## 3. 方法定义（冻结）

### 3.1 模式提取

在 SSPro C-uni 完整 36-action bank 上，逐 outer fold 使用 GRAN 已提交的 ScreenSpot-Pro $\tau^*$ 聚类，取候选数最大的两个模式 $M_1,M_2$，权重为块内 candidate votes。块计数并列时按 source reliability 总和、最大 source reliability、最早冻结 candidate order 依次打破。source reliability 只在对应 inner-training 或 outer-development rows 上拟合。

$g$ 网格冻结为 `[0.25, 0.40, 0.55, 0.70, 0.85]`。每个 outer fold 的 inner validation fold 为 $(k+1)\bmod5$，其余三折为 inner training。$g$ 在 inner validation 上最大化两模式可达头顶空间，tie 取更大的 $g$（更少 GPU rows）。

### 3.2 裁剪构造

对模式 $M_j$，中心为块内候选坐标均值，RMS 尺度为候选到中心欧氏距离的均方根。共享方形边长

$$
L=\min\big(\min(W,H),\max(512,2.5\max(\sigma_1,\sigma_2))\big).
$$

$W_1$ 与 $W_2$ 使用同一 $L$。为包含目标模式中心并排除另一中心，选两中心差绝对值最大的坐标轴，把目标中心放在窗口远离另一中心的一侧边缘内 1 pixel；另一轴以目标中心居中。窗口随后在保持 $L$ 不变的条件下平移入图像。若平移后无法同时包含目标中心并排除另一中心，则该行几何匹配失败。

$W_0$ 使用同一 $L$，在四个图像角的合法窗口中选择使其中心到 $M_1,M_2$ 两中心最小距离最大的窗口；tie 按左上、右上、左下、右下。$W_0$ 必须同时排除两个模式中心，否则该行失败。

三窗口面积与长宽比理论上完全相同。审计容差仍为面积比 `[0.9,1.1]`、长宽比差 $\le0.05$、模型预处理后的 resize 边长完全相同。失败行整行丢弃，不补救。

### 3.3 生成与观测

每窗口 $k=3$，temperature `0.9`，top-p `1.0`，max new tokens `32`。每次采样 seed 为 SHA-256(`row_id|model_id|window|sample_index|20260814`) 的前八字节模 $2^{31}-1$。

主观测量 $s(W)$ 为三次采样中“首个可解析坐标对所包含的数值 token”的平均 logprob，再对三次取均值。token 归属按 tokenizer decode 后首个坐标正则匹配的字符跨度与逐 token 字符跨度相交确定。实现禁止读取生成坐标位置作为判决特征；位置仅用于次级空间离散度。

次级观测为同一坐标 token 的 perplexity，以及三次解析坐标在 crop-normalized 空间的 RMS 离散度。不得用次级观测替换主观测。

任一窗口的三次采样中若有一次无法解析坐标对或无法定位数值 token，则该“模型×行”无效并整行丢弃。无效率超过 15% 时与几何失败同样视为实现失败，须先修实现，不进入终点判定。

判决统计量 $\Lambda=s(W_1)-s(W_2)$。默认选择 $M_1$；当 $\Lambda<t$ 时改选 $M_2$。$t$ 网格为 inner-dev $\Lambda$ 分位点 `[0.10,0.25,0.50,0.75,0.90]`，另含 $-\infty,+\infty$；按 inner-validation accuracy 选择，tie 依次偏好 $t=-\infty$、由低到高的有限分位点、$+\infty$。

## 4. 分阶段执行与门（GPU 前必须先过零 GPU 门）

### 第一阶段：零 GPU

在现存 SSPro 7B 36-action bank 上 nested 计算每折 $g$、触发门占比，以及门内目标仅在 $M_2$ 而不在 $M_1$ 的比例。两模式可达头顶空间

$$
\Delta_2=P(\text{gate}\land M_2\text{ correct}\land M_1\text{ wrong}).
$$

Z-G1。若 held-out 聚合 $\Delta_2<3.0$ pp，本轮在 GPU 前终止，如实记为不值得执行。

### 第二阶段：GPU pilot

仅在 Z-G1 通过后执行。正例为 gate 内 $M_2$ correct 且 $M_1$ wrong 的 held-out rows。负例为 gate 内 $M_1$ correct 且 $M_2$ wrong 的 rows。纳入全部正例，按 application 与 outer fold 分层，从负例中等量确定性抽样；seed `20260814`。行 ID 清单必须先提交 manifest 再申请一次性 GPU authorization。

若 $n_+<120$，降级为观察性报告，不作 Z-P 判定。

### 第三阶段：判定

顺序固定为 Z-P3、Z-P1、Z-P2、Z-P4、Z-P5。Z-P3 不过即停。

## 5. 主次终点

| ID | 终点 | 判据 |
| --- | --- | --- |
| Z-P1（主） | Qwen3 主模型的 $\Lambda$ 区分“目标在 $M_1$”与“目标在 $M_2$” | 99% CI 下界 $>0.60$ |
| Z-P2 | $\Lambda$ 相对单用 $w_2/w_1$ 的增量 AUC | nested paired bootstrap 99% CI 下界为正 |
| Z-P3 | $W_0$ 与 wrong-mode 窗口均不含目标，应不可分 | AUC 99% CI 包含 0.5，且 $|\mathrm{AUC}-0.5|$ 的 99% 上界 $\le0.10$ |
| Z-P4 | 探针误差与池误差的 $\kappa$ | $<0.398$ 方可称近似正交通道 |
| Z-P5 | 端到端准确率变化 | 仅报点估计与 CI，不作判定依据 |

Qwen3 决定主判定。GTA1 与可选 Qwen2.5 分别复现同一五项终点，只报告模型异质性，不改变主结论。

## 6. Kill conditions

| ID | 触发条件 | 处理 |
| --- | --- | --- |
| Z-K1 | $\Delta_2<3.0$ pp | GPU 前终止，写入 limitation |
| Z-K2 | Z-P3 超出预注册不可分界 | 观测量受内容/几何混淆，整轮终止，不换观测量重试 |
| Z-K3 | Z-P1 AUROC 99% CI 上界 $<0.60$ | 假设否证，写为失败尝试 |
| Z-K4 | Z-P2 增量 CI 含零 | 探针与池冗余，未构成新通道，终止 |
| Z-K5 | Z-P4 的 $\kappa\ge0.398$ | 不满足近似正交条件，不得称新通道 |
| Z-K6 | 几何匹配丢弃率或生成无效率任一超过 15% | 先修实现再重跑，不带病判定 |
| Z-K7 | $n_+<120$ | 降级为观察性报告，不出判定结论 |

失败后禁止换观测量、换门限、换 benchmark、换主模型再报。次级观测量只在主观测量已判定后作为附录描述。

## 7. 必须控制的混淆

第一，裁剪几何影响置信度且与目标是否存在无关。靠 matched windows 与 Z-K6 控制。

第二，图像内容复杂度影响置信度。因为 wrong-mode 与 $W_0$ 都不含目标，二者应不可分；显著可分表示内容/几何混淆，触发 Z-K2。

第三，模型在任何窗口内都会输出坐标，因此位置不可用作主观测量。本 spec 只读置信类量，位置仅用于预注册的次级离散度。

第四，GRAN 的 M2W 网格触发 G-K6，因此 SPLIT 只在 SSPro 上执行。

第五，模型多重性不得用于择优。Qwen3 是唯一主模型；GTA1/Qwen2.5 为 secondary replication。

## 8. 数据与留存

输入为 SSPro 7B 36-action bank，切分沿用 application GroupKFold(5)。$g$ 与 $t$ 在 nested inner dev 上选择，held-out 每格只出一个数。统计沿用 application-group paired bootstrap 10,000 次 99% percentile CI。

已泄漏的五个 SSPro 格子不得作为任何优化目标或门限依据。GRAN 的 SSPro 逐折 $\tau^*$ 只作为已提交上游选择使用。

逐行 JSONL 必须每 lane fsync；原始 response、逐 token id、逐 token decoded text、逐 token logprob、coordinate-token mask、解析坐标、resize size 与窗口几何必须完整保存。lane 完成后写 SHA-256 manifest。备份路径独立于 git并写入 STATUS。清理脚本禁止递归删除 raw/ 与 `predictions*.jsonl`。

## 9. 执行顺序

1. commit 本 spec 与 `split_prereg.yaml`；
2. 锁定输入与三模型 preflight；
3. 零 GPU nested 计算 $\Delta_2$ 与 $g$，按 Z-G1 判定；
4. 若通过，构造平衡子集并先提交行 ID manifest；
5. 提交 GPU runner 与一次性 authorization；
6. 执行 Qwen3 主模型和 GTA1 secondary；Qwen2.5 仅在 checkpoint 恢复且预先锁 hash 后执行；
7. 按 Z-P3、Z-P1、Z-P2、Z-P4、Z-P5 顺序判定；
8. 无论结果如何，把 $\Delta_2$ 与主模型 Z-P1 写入当前论文 limitation。
