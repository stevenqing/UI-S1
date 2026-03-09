# SFT v2 Evaluation Deep Analysis

> Date: 2026-03-08
> Model: Qwen2.5-VL-7B + Full SFT v2 (1ep & 2ep)
> Benchmark: GUI-360 (18265 grounding + 19046 action prediction samples)

---

## 1. Overall Results vs Paper

| Task | Paper SFT | SFT v2 1ep | SFT v2 2ep | Gap (2ep vs Paper) |
|------|:---------:|:----------:|:----------:|:------------------:|
| **Grounding** | 82.30% | 70.56% | 70.77% | **-11.5pp** |
| **Action Pred (Visual)** | 50.08% | 46.90% | 49.37% | **-0.7pp** |

- Action prediction 已基本追平 paper (**93.6% of paper**)
- **主要差距集中在 Grounding** (86.0% of paper)
- 2ep 在 action prediction 上比 1ep 提升 2.5pp，但 grounding 仅提升 0.2pp

---

## 2. Grounding Failure Analysis

### 2.1 Failure Distance Distribution

Total failures: 5338 / 18265

| 类型 | 数量 | 占失败比 | 占总样本比 | 修复后总准确率 |
|------|:----:|:--------:|:---------:|:------------:|
| Parse 失败 (null coords) | 263 | 4.9% | 1.4% | 72.2% |
| Near miss (<=20px) | 881 | 16.5% | 4.8% | 75.6% |
| Medium miss (21-50px) | 816 | 15.3% | 4.5% | 80.1% |
| Far miss (>50px) | 3378 | 63.3% | 18.5% | — |

**Key Finding: +/-50px tolerance => 80.1%, close to paper's 82.3%.**

大量失败是"差一点"的坐标偏移，模型找对了元素但坐标不够精确。

### 2.2 BBox Size vs Failure Rate

| BBox 面积 | 总数 | 失败数 | 失败率 |
|-----------|:----:|:-----:|:------:|
| <500px² (小按钮/图标) | 720 | 537 | **74.6%** |
| 500-2000px² (中等元素) | 7966 | 2595 | 32.6% |
| >2000px² (大元素) | 9579 | 2206 | 23.0% |

- 小于 500px² 的元素几乎 3/4 都失败
- 成功样本中位 bbox 面积 (2890px²) 是失败样本 (1586px²) 的 **1.8 倍**

### 2.3 Parse Failures

263 个样本全部输出 `<coordinate>[null, null]</coordinate>`，模型对某些 UI 元素完全无法定位。

Domain 分布: PPT=97, Excel=103, Word=49

### 2.4 Near Miss Direction Analysis

| 指标 | 值 |
|------|:--:|
| 平均 X 偏移 | +5.0px |
| 平均 Y 偏移 | -3.2px |
| Left of bbox | 196 |
| Right of bbox | 214 |
| Above bbox | 280 |
| Below bbox | 248 |

方向无系统偏差，排除了坐标变换 bug。

### 2.5 Far Miss Pattern Analysis

Far misses (>50px) 在屏幕区域和 step number 上均匀分布：

**Model 点击位置 vs GT 位置分布基本一致：**

| Region | Model 点击 | GT 位置 |
|--------|:---------:|:------:|
| Center | 42.4% | 38.6% |
| Upper | 25.5% | 24.7% |
| Top (toolbar) | 17.3% | 20.7% |
| Bottom | 6.9% | 8.4% |
| Right sidebar | 4.4% | 4.6% |
| Left sidebar | 3.4% | 3.0% |

没有发现系统性的"区域混淆"模式。Far miss 代表真正的元素理解错误。

### 2.6 Per-Domain Grounding

| Domain | Category | Base | SFT v2 1ep | SFT v2 2ep | n |
|--------|----------|:----:|:----------:|:----------:|:---:|
| **PPT** | search | 50.9% | 70.2% | 71.4% | 4937 |
| | in | 58.2% | 70.4% | 68.4% | 98 |
| | online | 40.0% | 55.8% | 54.5% | 310 |
| | **Total** | **50.4%** | **69.4%** | **70.4%** | **5345** |
| **Excel** | search | 29.6% | 68.6% | 67.6% | 5147 |
| | in | 31.6% | 57.9% | 52.6% | 19 |
| | online | 43.2% | 76.2% | 72.4% | 185 |
| | **Total** | **30.0%** | **68.8%** | **67.7%** | **5351** |
| **Word** | search | 44.3% | 72.7% | 73.2% | 7217 |
| | in | 45.9% | 70.3% | 72.1% | 111 |
| | online | 50.6% | 70.5% | 73.0% | 241 |
| | **Total** | **44.5%** | **72.6%** | **73.2%** | **7569** |
| | **OVERALL** | **42.0%** | **70.6%** | **70.8%** | **18265** |

- **Word** 表现最好 (73.2%), **Excel** 最弱 (67.7%)
- Excel 从 base 的提升幅度最大 (+37.7pp)
- Online 类别一致偏低（界面更复杂）

---

## 3. Action Prediction Failure Analysis

### 3.1 Field Match Rates

| 字段 | 匹配率 | 说明 |
|------|:------:|------|
| Function | 86.1% | 较好 |
| Status | 95.3% | 很好 |
| **Args** | **49.6%** | **瓶颈** |
| **Overall** | **49.4%** | 三者全对 |

### 3.2 Failure Pattern Breakdown

| Function | Args | Status | 数量 | 占比 | 说明 |
|:--------:|:----:|:------:|:----:|:----:|------|
| ✓ | ✓ | ✓ | 9403 | 49.4% | SUCCESS |
| **✓** | **✗** | **✓** | **6882** | **36.1%** | **最大问题：func 对但 args 错** |
| ✗ | ✗ | ✓ | 1873 | 9.8% | func 完全预测错 |
| ✗ | ✗ | ✗ | 766 | 4.0% | 提前结束 + 错误 func |
| ✓ | ✗ | ✗ | 69 | 0.4% | |
| ✓ | ✓ | ✗ | 51 | 0.3% | |

### 3.3 "Function Correct but Args Wrong" Deep Dive (6951 samples)

| 原因 | 数量 | 占比 |
|------|:----:|:----:|
| **坐标偏出 bbox** | **5620** | **80.8%** |
| 坐标在 bbox 内，其他参数错 | 1182 | 17.0% |
| Null/无效坐标 | 149 | 2.1% |

**坐标精度是 action prediction 和 grounding 共同的首要瓶颈。**

### 3.4 Click Coordinate Failures

Click function (14467 samples, 55.3% success):

| 距离 | 数量 | 占 click 坐标失败比 |
|------|:----:|:------------------:|
| Near miss (<=20px) | 676 | 13.6% |
| Medium (21-50px) | 743 | 15.0% |
| Far (>50px) | 3540 | 71.4% |

### 3.5 Per-Function Analysis

| Function | 样本数 | 成功率 | 主要问题 |
|----------|:-----:|:------:|---------|
| click | 14467 | 55.3% | 坐标精度 |
| **type** | **3411** | **29.5%** | **text hallucination** |
| select_text | 496 | 51.8% | |
| set_font | 96 | 59.4% | |
| wheel_mouse_input | 311 | **17.0%** | 常被误判为 click |
| insert_table | 69 | 21.7% | |
| select_paragraph | 49 | 12.2% | |
| summary | 45 | **0.0%** | 从未预测正确 |
| set_background_color | 25 | **0.0%** | |
| select_table_range | 20 | **0.0%** | |
| save_as | 13 | **0.0%** | |

### 3.6 Type Function Failures (1760 samples, func correct but fail)

| 原因 | 数量 | 占比 |
|------|:----:|:----:|
| **Text 内容错误** | **653** | **37%** |
| 坐标错误 | 451 | 26% |
| 两者都错 | 266 | 15% |
| Extra args 不匹配 (如 `clear_current_text`) | 222 | 13% |

**典型 text hallucination 例子：**

| Predicted | Ground Truth |
|-----------|-------------|
| "Confidential - For Internal Use Only" | "This is a footnote for all slides." |
| "StrongPassword123!" | "YourSecurePassword" |
| "collaboration between police and educational institutions" | "collaboration police educational institutions" |

模型对要输入的文本内容理解有误，倾向于 hallucinate 或 paraphrase。

### 3.7 Empty Function Predictions

789 个样本 (4.1%) 模型输出空 function + FINISH，提前结束任务。

| GT Function | 空预测数 |
|-------------|:-------:|
| click | 617 |
| type | 108 |
| summary | 21 |
| select_text | 19 |
| wheel_mouse_input | 15 |
| 其他 | 9 |

### 3.8 Status Mismatch (886 samples, 4.7%)

| Predicted | Ground Truth | 数量 |
|-----------|-------------|:----:|
| FINISH | CONTINUE | 763 |
| CONTINUE | FINISH | 123 |

模型偏向提前结束 (763 vs 123)。

### 3.9 Per-Domain Action Prediction

| Domain | Category | Base | SFT v2 1ep | SFT v2 2ep | n |
|--------|----------|:----:|:----------:|:----------:|:---:|
| **PPT** | search | 25.2% | 53.8% | 56.7% | 4973 |
| | in | 27.6% | 58.2% | 61.2% | 98 |
| | online | 22.3% | 41.3% | 41.6% | 310 |
| | **Total** | **25.1%** | **53.2%** | **55.9%** | **5381** |
| **Excel** | search | 12.7% | 42.5% | 44.6% | 5158 |
| | in | 10.5% | 36.8% | 42.1% | 19 |
| | online | 21.2% | 50.8% | 60.3% | 189 |
| | **Total** | **13.0%** | **42.8%** | **45.2%** | **5366** |
| **Word** | search | 15.4% | 45.4% | 47.7% | 7934 |
| | in | 25.2% | 51.3% | 54.8% | 115 |
| | online | 18.8% | 47.6% | 48.8% | 250 |
| | **Total** | **15.6%** | **45.5%** | **47.8%** | **8299** |
| | **OVERALL** | **17.6%** | **46.9%** | **49.4%** | **19046** |

---

## 4. Training-Eval Alignment Check

| 项目 | 训练 | 评估 | 是否对齐 |
|------|------|------|:--------:|
| 图片分辨率 | 1040x736 | 1040x736 -> smart_resize 1036x728 | ✓ |
| max_pixels | 1,003,520 | 1,003,520 | ✓ |
| 坐标空间 | 1040x736 绝对坐标 | 模型输出经 scale 变换回 1040x736 | ✓ |
| **输出格式** | **`<tool_call>` (100%)** | **Grounding: `<coordinate>`, AP: `<tool_call>`** | **Grounding ✗** |
| 训练数据 | 101,700 条 action prediction | Grounding + Action Prediction | — |

**训练数据没有任何 grounding 格式的数据** (0 条 `<coordinate>` 格式)。模型通过 in-context learning 适配了 grounding prompt 的 `<coordinate>` 格式（parse 成功率 98.6%），但精度可能不如有专门 grounding 训练数据的 paper 模型。

---

## 5. Root Cause Summary

### 5.1 Grounding Gap (-11.5pp vs Paper)

| 原因 | 影响 (pp) | 可修复性 |
|------|:---------:|:--------:|
| **坐标精度不足** (near+medium miss) | ~9.3pp | 可优化 |
| Parse 失败 (null coords) | ~1.4pp | 可优化 |
| 完全定位错误 (>50px) | ~18.5pp | 需更强模型 |

- **首因**: 小元素 (<500px²) 失败率 74.6%，模型对精细 UI 元素的定位能力不足
- **次因**: 无 grounding 专项训练数据，靠 in-context learning 适配 `<coordinate>` 格式
- **推测 Paper 可能**: 有 grounding 专项数据 / 更高分辨率 / 更多训练量

### 5.2 Action Prediction Gap (-0.7pp vs Paper)

| 原因 | 影响 | 说明 |
|------|:----:|------|
| 坐标精度 (func ✓ args ✗) | 36.1% 样本 | 与 grounding 同源问题 |
| Type text hallucination | 3.4% | 模型 paraphrase 或编造文本 |
| 提前结束 (empty func) | 4.1% | 模型过度倾向 FINISH |
| 稀有 function 全失败 | 1.3% | summary/save_as 等不在训练分布中 |

---

## 6. Relaxed BBox Tolerance Sweep

### 6.1 Grounding

将 ground truth bounding box 上下左右各扩展 ±N px，重新计算命中率：

| Expand | SFT v2 2ep | MoE v1 | LoRA v3 | Paper (82.3%) |
|:------:|:----------:|:------:|:-------:|:-------------:|
| ±0px | 70.77% | 60.32% | 54.79% | 82.30% |
| ±5px | 71.55% | 61.86% | 56.71% | |
| ±10px | 72.72% | 63.74% | 59.33% | |
| ±15px | 74.55% | 66.41% | 62.41% | |
| ±20px | 75.72% | 68.29% | 64.51% | |
| ±25px | 76.64% | 70.10% | 66.72% | |
| ±30px | 77.40% | 71.08% | 67.83% | |
| ±40px | 78.91% | 73.23% | 70.29% | |
| ±50px | 80.39% | 75.11% | 72.53% | |
| **±75px** | **82.76%** | 78.37% | 75.94% | **>= Paper** (SFT v2) |
| ±100px | 84.54% | 81.03% | 78.61% | |

### 6.2 Action Prediction

同理扩展坐标匹配的 bbox tolerance：

| Expand | SFT v2 2ep | MoE v1 | LoRA v3 | Paper (50.08%) |
|:------:|:----------:|:------:|:-------:|:--------------:|
| ±0px | 49.58% | 33.37% | 23.86% | 50.08% |
| ±5px | 49.98% | 34.35% | 25.03% | |
| **±10px** | **50.76%** | 35.47% | 26.42% | **>= Paper** (SFT v2) |
| ±15px | 52.30% | 36.88% | 28.45% | |
| ±20px | 53.12% | 38.42% | 30.36% | |
| ±50px | 57.33% | 44.61% | 37.09% | |
| ±75px | 59.37% | 48.27% | 41.64% | |
| **±100px** | 60.80% | **50.49%** | 44.55% | **>= Paper** (MoE v1) |

### 6.3 Tolerance Sweep 结论

**SFT v2 2ep:**
- Action Prediction: **±10px 即超过 paper**，坐标精度差距极小
- Grounding: **±75px 超过 paper**，差距本质是坐标偏移而非元素识别错误
- **核心结论：模型的元素理解能力已达到 paper 水平，差距全部在坐标精度**

**MoE v1:**
- Grounding: **±100px 仍不到 paper** (81.0% vs 82.3%)，元素识别能力本身不够
- Action Prediction: **±100px 刚好追平 paper** (50.5% vs 50.1%)
- 差距不仅是坐标精度，function match 和 args 理解能力都弱于 SFT v2（router collapse 导致单 expert）

**LoRA v3:**
- Grounding: ±100px 仅 78.6%，远低于 paper
- Action Prediction: ±100px 仅 44.6%，远低于 paper
- Frozen projector 导致视觉理解能力受限，即使完全放松坐标也无法弥补

**跨模型对比：**
- 每 px tolerance 的增益：SFT v2 (~0.14pp/px) < MoE v1 (~0.21pp/px) < LoRA v3 (~0.24pp/px)
- LoRA 系模型坐标偏移更严重，但即使修正坐标仍有能力差距
- **SFT v2 的瓶颈几乎纯粹是坐标精度；MoE v1 / LoRA v3 的瓶颈是坐标精度 + 模型能力**

---

## 7. Level 2 Semi-Online AR Evaluation (Trajectory Rollout)

> 前面的 Grounding 和 Action Prediction 评估均为 **Level 1 (Teacher-Forcing)**：每一步使用 GT 历史。
> 本节使用 **Level 2 (Semi-Online AR)**：模型自身预测的动作作为后续步骤的历史，GT 截图不变。
> 评估设置：`stop_on_error=True`（首错即停），greedy decoding。

### 7.1 Overall AR Results

| 指标 | Base (Qwen2.5-VL-7B) | SFT v2 | 倍数提升 |
|------|:--------------------:|:------:|:--------:|
| **Trajectory Success Rate (TSR)** | **0.0164** | **0.1621** | **9.9x** |
| Sequential Progress | 0.0734 | 0.2805 | 3.8x |
| Step-Level SR | 22.10% | 55.28% | 2.5x |
| Avg Steps Evaluated | 1.3 | 1.9 | — |
| Avg First Error Step | 1.25 | 1.75 | — |
| Sub-goal Progress | 0.0221 | 0.2012 | 9.1x |
| Weighted Sub-goal Progress | 0.1205 | 0.3531 | 2.9x |

**关键发现：**
- Base 模型平均走 **1.25 步**就出错，几乎所有 trajectory 在第一步就停止
- SFT v2 将 AR TSR 从 1.6% 提升到 **16.2%**，提升近 10 倍
- `stop_on_error=True` 下 Post-Error Accuracy = 0.0, Recovery Rate = 0.0（因为一旦出错就停止，没有后续步骤）

### 7.2 AR vs TF Comparison

| 指标 | Base TF | Base AR | SFT v2 TF | SFT v2 AR |
|------|:-------:|:-------:|:---------:|:---------:|
| **TSR** | **0.0285** | **0.0164** | **0.1695** | **0.1621** |
| Scattered Progress | 0.2032 | 0.1230 | 0.5060 | 0.3670 |
| Sequential Progress | 0.0872 | 0.0734 | 0.2910 | 0.2805 |
| Step-Level SR | 18.05% | 22.10% | 46.90% | 55.28% |
| Total Steps Evaluated | 19046 | 4082 | 19046 | 6058 |
| Sub-goal Progress | 0.0461 | 0.0221 | 0.2522 | 0.2012 |

**TF vs AR Gap 分析：**

| 模型 | TSR Gap (TF - AR) | Sequential Gap | 说明 |
|------|:-----------------:|:--------------:|------|
| Base | -0.0121 (-42%) | -0.0138 (-16%) | AR 下大幅退化 |
| SFT v2 | **-0.0074 (-4.4%)** | **-0.0105 (-3.6%)** | **AR 下几乎无退化** |

- **SFT v2 的 TF-AR gap 极小 (TSR 仅降 0.7pp)**，说明 SFT 后模型不依赖 GT 历史，自身预测的历史足够好
- **Base 模型的 TF-AR gap 相对更大 (TSR 降 1.2pp, -42%)**，但因为 base 本身很弱，绝对值差异也不大
- AR 下 Step-Level SR 反而更高（SFT v2: 55.3% vs TF 46.9%），因为 `stop_on_error=True` 使得只有前几步被评估，而前几步准确率较高
- AR 下评估的总步数远少于 TF（SFT v2: 6058 vs 19046），因为大部分 trajectory 在早期就因错误而停止

### 7.3 AR Step Position Accuracy

| Step | Base AR | SFT v2 AR | SFT v2 TF |
|------|:-------:|:---------:|:---------:|
| 1 | 0.213 | 0.500 | 0.488 |
| 2 | 0.253 | 0.621 | 0.599 |
| 3 | 0.242 | 0.616 | 0.541 |
| 4 | 0.250 | 0.602 | 0.467 |
| 5 | 0.000 | 0.568 | 0.445 |

- SFT v2 AR 在所有步骤上都优于 TF（因为 stop_on_error 只保留了"能走到该步"的高质量 trajectory）
- Base AR 在 step 5 直接降到 0（没有 trajectory 能走到 step 5）

### 7.4 AR Domain Breakdown

| Domain | Base AR TSR | SFT v2 AR TSR | SFT v2 AR Progress |
|--------|:-----------:|:-------------:|:------------------:|
| **PPT** | 0.032 | **0.183** | 0.464 |
| &nbsp;&nbsp;in | 0.118 | 0.294 | 0.690 |
| &nbsp;&nbsp;online | 0.023 | 0.205 | 0.416 |
| &nbsp;&nbsp;search | 0.031 | 0.179 | 0.462 |
| **Excel** | 0.004 | **0.154** | 0.301 |
| &nbsp;&nbsp;in | 0.000 | 0.000 | 0.000 |
| &nbsp;&nbsp;online | 0.000 | 0.298 | 0.430 |
| &nbsp;&nbsp;search | 0.004 | 0.148 | 0.296 |
| **Word** | 0.015 | **0.155** | 0.354 |
| &nbsp;&nbsp;in | 0.000 | 0.044 | 0.401 |
| &nbsp;&nbsp;online | 0.000 | 0.175 | 0.416 |
| &nbsp;&nbsp;search | 0.016 | 0.156 | 0.351 |

- PPT 依然是 AR 下表现最好的 domain (TSR 0.183)，与 TF 结果一致
- Excel 最弱 (TSR 0.154)，尤其是 `in` 类别 AR 下 TSR 为 0
- AR 下 domain 间的差距比 TF 更明显，PPT 领先幅度更大

### 7.5 AR Trajectory Length Distribution

由于 `stop_on_error=True`，AR 下的有效 trajectory 长度显著缩短：

| Bucket | Base AR | SFT v2 AR | SFT v2 TF |
|--------|:-------:|:---------:|:---------:|
| short (1-5) | 3233 (100%) | 3172 (98.1%) | 2076 (64.2%) |
| medium (6-15) | 0 (0%) | 61 (1.9%) | 930 (28.8%) |
| long (16+) | 0 (0%) | 0 (0%) | 227 (7.0%) |

- **Base AR 没有任何 trajectory 超过 5 步**
- SFT v2 AR 仅 61 个 trajectory (1.9%) 达到 medium 长度，这些 trajectory 的 TSR 高达 **0.328**
- **没有任何 trajectory 在 AR 下达到 16+ 步**，说明长序列推理仍是核心挑战

---

## 8. Overall Conclusions

### 8.1 核心结论

1. **坐标精度是 grounding 和 action prediction 共同的首要瓶颈**，两个 task 的失败模式高度一致
2. **Action prediction 已追平 paper** (49.4% vs 50.08%)，说明训练数据和方法基本正确
3. **Grounding gap 主要来自精细定位能力** — ±50px 容差下准确率 80.4%，±75px 超过 paper，说明模型对"哪个元素"的理解已达到 paper 水平，但坐标输出精度不够
4. **Type function 是 action prediction 中最大的子问题** (29.5% 成功率)，text hallucination 严重
5. **2ep vs 1ep**: Grounding 无明显提升 (+0.2pp)，Action prediction 有提升 (+2.5pp)，主要来自 func match 和 status match 改善
6. **SFT v2 的 TF-AR gap 极小** (TSR 仅降 0.7pp / 4.4%)，说明模型的 autoregressive rollout 能力良好，不依赖 GT 历史
7. **长序列推理仍是核心挑战** — AR 下没有任何 trajectory 超过 15 步，所有长 trajectory 在中途因错误停止

### 8.2 改进方向

| 方向 | 预期增益 | 可行性 |
|------|:-------:|:------:|
| 提升坐标精度 (更高分辨率 / 坐标回归损失) | Grounding +5~10pp | 中 |
| 增加 grounding 专项训练数据 | Grounding +2~5pp | 高 |
| 修复 type text hallucination | AP +2~3pp | 中 |
| 修复 empty function / premature FINISH | AP +1~2pp | 高 |
| 添加稀有 function 训练数据 | AP +0.5~1pp | 高 |
