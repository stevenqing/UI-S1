# Source Bias 与 Lineage-Normalized Aggregation 完整总结

日期：2026-08-03

## 一、研究问题

此前的实验已经表明：候选池的oracle headroom、有效样本量和最终聚合准确率并不是同一个量。尤其在72B候选池中，单个Qwen3.5候选本身达到`71.41%`，但B3聚合后仅有`41.24%`。本实验研究两个问题：

1. B3是否存在系统性的模型来源偏置，即某一模型谱系是否因候选数量、组内一致性或投票结构而过度成为赢家；
2. 将每个模型谱系先压缩为一个代表，再让三个谱系各投一票，是否可以兑现候选池中已经存在的best-single headroom。

整个实验不进行任何新模型推理，只复用已有候选trace。协议在结果产生前提交，固定了候选池、来源归因、21个lineage-normalized变体、五折嵌套选择和成功标准。

## 二、实验协议

### 2.1 B1来源偏置检验

对预注册候选池中的B3、M1和graph centroid分别恢复赢家来源，并在`all`、`correct`和`incorrect`三个分层中比较：

- 观测到的赢家来源分布；
- 根据候选池实际模型占比得到的期望分布。

统计量包括卡方拟合优度检验、Cramer's V和标准化残差。

来源归因规则为：

- B3：使用官方MVP分组与coverage tie-break最终选中的真实候选；
- M1：使用`ccm_select()`返回的真实候选索引；
- graph centroid：质心本身不是候选，因此归因到距离生成质心最近的真实候选，同时单独报告赢家component的来源构成。

B1通过条件是：7B Uniform Mixed N12和72B Uniform Mixed N8的B3错误行中，GTA来源均呈显著正残差，且$p<0.001$。

### 2.2 B4机制归因

B4使用三组零推理诊断区分两种解释：

- 强解释：共享GTA proposer直接造成来源偏置；
- 弱解释：异质候选池与聚合规则共同形成来源投票偏置。

具体检查：

1. 比较view 0全图池与views 1--3 crop池的GTA来源残差；
2. 比较各谱系四视图内的归一化pairwise distance；
3. 对候选数不平衡的池进行确定性下采样，再重算B3。

只有当两个尺度上GTA在view 0中的过度代表都弱于crop views，且GTA组内距离都显著低于另外两条谱系时，才支持共享proposer强归因。

### 2.3 B2谱系归一化聚合

Lineage normalization分两步：

1. 每个模型谱系内部的多个候选压缩为一个代表；
2. 三个谱系代表进行等谱系投票或可靠性决策。

预注册了$7\times3=21$个组合：

- R1：最大official complete-link group的质心；
- R2：几何中位数；
- R3：medoid；
- R4：最大14px图连通分量质心；
- R5：开发集最强view候选；
- R6：view-0候选；
- R7：最大official group内按开发集action accuracy加权的质心；
- D1：三个谱系代表的等权最大连通分量质心；
- D2：开发集lineage reliability加权的14px支持代表；
- D3：完全分歧时回退到开发集最可靠谱系，否则使用D1。

采用严格五折嵌套选择。对outer fold $f$：

- inner validation固定为$(f+1)\bmod5$；
- 其余三折用于拟合开发统计；
- 21个组合仅根据inner-validation B3准确率选择；
- 选定组合在完整outer-dev四折上重拟合；
- 最后只在outer-test上评估一次。

因此headline结果由1,581条完全held-out预测组成。完整21-grid仅作描述性敏感性分析，不使用其最大值作为主结果。

## 三、B1：来源偏置在两个尺度上均成立

| Pool/stratum | 错误行 | GTA观测赢家 | 按候选比例期望 | GTA标准残差 | 卡方$p$ | Cramer's V |
|---|---:|---:|---:|---:|---:|---:|
| 7B Uniform Mixed N12，B3错误行 | 574 | 489 | 191.33 | +26.36 | $4.12\times10^{-152}$ | 0.779 |
| 72B Uniform Mixed N8，B3错误行 | 929 | 872 | 348.38 | +35.49 | $1.15\times10^{-274}$ | 0.824 |

B1在两个尺度上均通过。偏置不仅统计显著，而且效应量极大：

- 7B错误行中，GTA成为赢家的比例为`85.19%`；
- 72B错误行中，GTA成为赢家的比例为`93.86%`；
- 这些比例远高于候选池中GTA候选本身所占比例。

这说明B3的错误并非来源中性。某一谱系可以依靠候选数量和组内投票结构形成占优势的错误簇，从而压制更强的单候选来源。

## 四、B4：不支持共享proposer强归因

### 4.1 View-0与crop来源偏置

| 尺度 | GTA view-0残差 | GTA views 1--3残差 | view-0是否更弱 |
|---|---:|---:|---|
| 7B | +37.16 | +24.68 | 否 |
| 72B | +31.60 | +40.08 | 是 |

7B方向与共享proposer强归因的要求相反：GTA在全图view 0中的过度代表反而强于crop views。

### 4.2 谱系内几何

7B中，GTA四视图内距离显著低于Qwen3和UI-TARS：

- GTA减Qwen3：point `-0.01561`，99% CI `[-0.02814,-0.00367]`；
- GTA减UI-TARS：point `-0.03147`，99% CI `[-0.04269,-0.02128]`。

但72B不满足同一条件：

- GTA减Qwen3.5：point `+0.00122`，99% CI `[-0.00653,+0.00901]`；
- GTA减Venus：point `+0.00027`，99% CI `[-0.00509,+0.00533]`。

因此不能把两个尺度上的来源偏置统一归因于“GTA proposer产生了更紧的同源候选”。

### 4.3 候选数平衡敏感性

72B Uniform Mixed N8原始谱系候选数为`3/3/2`。将其确定性平衡为`2/2/2`后：

- B3从`41.24%`提高到`49.84%`；
- 提升`+8.60 pp`；
- 候选数从8降至6。

这是描述性敏感性结果，不是预注册方法结果，但它说明候选数量不平衡确实是72B B3污染的重要组成部分。

### 4.4 B4结论

两个尺度不共同满足proposer强归因条件。因此最稳健的机制表述是：

> 观测到的是异质候选池中的聚合投票偏置，而不是已经被证明由共享proposer单独造成的偏置。

## 五、B2：72B强修复，7B失败

### 5.1 主结果

| 比较 | Nested LN | Reference | Delta | 99% CI | 单边$p$ |
|---|---:|---:|---:|---:|---:|
| 7B LN vs B3 | 61.99% | 63.69% | -1.71 pp | [-3.09,-0.21] | 0.9990 |
| 7B LN vs M1 | 61.99% | 63.82% | -1.83 pp | [-3.28,-0.19] | 0.9985 |
| 7B LN vs reported best-single | 61.99% | 54.65% | +7.34 pp | 独立trace，无paired CI | n/a |
| 72B LN vs B3 | 70.59% | 41.24% | +29.35 pp | [+21.57,+35.78] | $1/10001$ |
| 72B LN vs M1 | 70.59% | 52.12% | +18.47 pp | [+12.95,+23.44] | $1/10001$ |
| 72B LN vs best-single | 70.59% | 71.41% | -0.82 pp | [-3.01,+1.72] | 0.8222 |

### 5.2 72B结果解释

72B nested LN将B3从`41.24%`提高到`70.59%`，恢复了`29.35 pp`，也比M1高`18.47 pp`。这说明72B候选池中大部分性能损失确实来自聚合偏置，而不是候选本身缺乏正确答案。

五个outer fold的选择高度稳定：

- fold 0--2选择R5_D2；
- fold 3--4选择R5_D3。

R5直接使用每个谱系在开发集上最强的view，D2/D3再利用谱系可靠性处理代表之间的支持或完全分歧。这个稳定选择与72B的机制一致：弱谱系不应通过候选数量获得额外票数，开发集更可靠的谱系应在完全分歧时占优。

但是nested LN仍比Qwen3.5 best-single低`0.82 pp`，且99% CI跨0。因此它接近兑现best-single headroom，但没有显著超过或完全达到best-single。

### 5.3 7B结果解释

7B nested LN为`61.99%`，显著低于B3和M1。五折选择也不稳定：

- R1_D2；
- R4_D1；
- R1_D1；
- 两次R3_D1。

这说明7B没有单一稳定的谱系归一化决策结构。原始B3/M1在7B上已经能够利用同谱系内部的多候选几何；过早将每条谱系压缩到一个代表会丢失有效信息。

因此lineage normalization不是跨尺度普适改进，而是72B污染场景中的强修复。

### 5.4 Best-single口径说明

7B预注册best-single `54.65%`来自独立H3 bare trace。该trace在仓库中仅保留汇总及source hash，没有保留逐行预测，因此不能构造paired CI。当前36-action反事实bank中的Qwen3 view-0为`54.52%`，仅用于matched-bank诊断，不替代预注册best-single。

72B Qwen3.5 bare trace与当前bank一致，均为`71.41%`，因此可以进行paired bootstrap。

## 六、门控判定

| Gate | 结果 | 原因 |
|---|---|---|
| B1 source-bias gate | PASS，两个尺度 | GTA错误赢家残差均显著为正 |
| B2 72B bias correction | PASS | 对B3和M1的99% CI下界均为正 |
| B2跨尺度主成功 | FAIL | 7B nested LN显著低于B3 |
| B-K4 | TRIGGERED | 72B nested LN仍低于71.41% best-single |
| B3x | NOT RUN | B2要求两个尺度同时成功 |
| 共享proposer强归因 | NOT SUPPORTED | B4条件未在两个尺度同时满足 |

## 七、完整21变体敏感性结果

下表是cross-fitted描述性敏感性分析，不是headline nested结果。

| Variant | 7B | 72B |
|---|---:|---:|
| R1_D1 | 62.62% | 51.36% |
| R1_D2 | 61.99% | 64.58% |
| R1_D3 | 61.80% | 64.77% |
| R2_D1 | 60.40% | 23.53% |
| R2_D2 | 59.52% | 22.83% |
| R2_D3 | 59.71% | 22.83% |
| R3_D1 | 62.49% | 25.81% |
| R3_D2 | 61.23% | 57.37% |
| R3_D3 | 61.61% | 57.50% |
| R4_D1 | 62.87% | 51.49% |
| R4_D2 | 61.92% | 64.83% |
| R4_D3 | 61.99% | 65.02% |
| R5_D1 | 61.16% | 62.81% |
| R5_D2 | 60.09% | 70.46% |
| R5_D3 | 60.28% | 70.59% |
| R6_D1 | 51.04% | 62.81% |
| R6_D2 | 51.11% | 70.46% |
| R6_D3 | 51.17% | 70.59% |
| R7_D1 | 2.28% | 2.66% |
| R7_D2 | 2.15% | 2.40% |
| R7_D3 | 2.21% | 2.34% |

7B描述性最佳为R4_D1，准确率`62.87%`，仍低于原始B3。72B描述性最佳为R5_D3，准确率`70.59%`。这些结果只能用于解释敏感性，不能替代嵌套选择结果。

## 八、论文贡献与正确表述

本实验支持三项结论：

1. **B3存在强模型来源偏置。** 错误赢家高度集中于GTA，且远超候选比例能够解释的水平；
2. **候选计数和谱系内重复票会造成严重聚合污染。** 72B计数平衡和lineage normalization均带来大幅恢复；
3. **Lineage normalization能够在污染严重的72B池中近似兑现best-single headroom。** 它将B3从`41.24%`恢复到`70.59%`。

推荐论文表述：

> Shared-proposal candidate pools can exhibit severe model-source voting bias: repeated candidates from one lineage dominate erroneous consensus far beyond their nominal pool share. A nested lineage-normalized aggregator removes duplicate lineage votes and recovers most of the latent best-single headroom at 72B, but the effect does not transfer to 7B, where within-lineage geometry remains useful.

不能声称：

- 来源偏置已被证明由共享proposer单独造成；
- lineage normalization在所有模型尺度上都优于B3或M1；
- 72B nested LN已经超过best-single；
- 21-grid中的事后最大值是可部署headline结果；
- B3x已经验证统一修复CALA-S、NOA和高分歧预算轴。

## 九、最终定位

最稳健的总判断是：

> 来源偏置是B3在72B异质候选池中性能崩溃的主要机制之一。谱系归一化能够修复大部分崩溃，但该修复具有尺度依赖性；它是72B聚合偏置修复，而不是已经成立的通用聚合方法。

这与此前Effective-Sample-Size结果一致：候选覆盖、相关性和聚合可兑现性必须分开讨论。即使候选池中存在高质量候选，重复来源形成的错误簇仍可能使最终规则远离best-single上限。

## 十、交付物索引

- `SPEC.md`：结果盲预注册协议；
- `configs/b1_pools.yaml`：B1候选池、规则、分层和来源归因；
- `configs/b2_variants.yaml`：完整21变体和嵌套选择协议；
- `results/b1_source_bias.json`：全部B1来源偏置统计；
- `results/b2_lineage_normalized.json`：B2 nested输出、fold选择和paired统计；
- `results/b4_attribution.json`：B4 proposer/几何/计数平衡诊断；
- `MAIN_TABLE.md`：核心结果表；
- `B2_VARIANT_GRID.md`：完整21-grid；
- `REPORT.md`：精简英文报告；
- `STATUS.json`：最终门控状态和artifact hashes；
- `figures/b1_source_bias.pdf`：B1来源赢家观测与期望图。

协议commit：`18e0267`。

最终结果commit：`248f336`。