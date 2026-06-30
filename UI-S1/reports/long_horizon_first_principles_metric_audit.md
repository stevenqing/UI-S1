# Long-Horizon Evaluation: First-Principles Metric Audit

日期：2026-06-30

## 核心结论

当前项目里，“long-horizon 问题”不能用 overall TSR、overall step accuracy 或 always-history 提升来证明。

真正能 reveal long-horizon 的证据必须满足一个因果条件：

```text
current state alone insufficient
true history changes the correct action
wrong/shuffled history does not produce the same gain
failure localizes to the history-consuming step/value
```

换句话说，long-horizon 不是“episode 很长”，也不是“给 history 后整体准确率变高”。它是：

```text
P(a_t | current screen, instruction, true history)
  != P(a_t | current screen, instruction)
```

行为上最干净的证据是：

```text
no_history wrong
true_history correct
wrong_history wrong
error is a carried-value / state-dependency error, not grounding / format / prior-chain error
```

## 第一性原理：什么能 reveal long-horizon 问题

### 1. Dataset Prevalence Gate

先问数据里有没有战场。

候选 `(i, value, j)` 必须满足：

```text
produced earlier
consumed later
not available at consumption
not given by goal/template/world prior
not routine/default/forced/clipboard/resurfaced
distance and interference enough to matter
```

这个 gate 的作用不是训练，而是决定有没有必要训练 history model。

GUI-360-balanced 结果：

```text
OCR-backed train: 78 / 1578 = 4.94%
OCR-backed test:   9 / 541  = 1.66%
data-only upper train: 116 / 1578 = 7.35%
data-only upper test:   19 / 541  = 3.51%
```

结论：GUI-360-balanced 不是大规模 cross-step dependency battlefield。

### 2. Counterfactual History Utility

对每个 step 构造：

```text
x_t     = current screen + instruction
m_pos   = true relevant history / segment memory
m_neg   = wrong or unrelated history
```

有效 long-horizon 样本是：

```text
no_history wrong, true_history correct, wrong_history wrong
```

负样本也同样重要：

```text
no_history correct, true_history wrong -> stale-history regression
all contexts wrong -> not a memory problem
wrong_history correct -> history signal non-specific
```

这比“long_horizon=true”标签更干净，因为它直接测 memory 的因果 utility。

### 3. Current-State Equivalence Pairs

最强的 reveal 方式是找两个样本：

```text
same/similar current screen
same/similar instruction surface
different prior state/history
different correct action/value
```

如果 current screen 足以决定动作，就不会出现这种 pair。

如果出现并且模型只在 true history 条件下做对，这就是强 long-horizon 证据。

### 4. First-Error Localization

trajectory failure 只有在以下条件下才算 memory defect：

```text
produce step correct
teacher-forced to consumption step
first error exactly at consumption step
wrong value is a carried-value mixup or history-dependent wrong value
```

否则可能只是：

```text
grounding error
format error
prior-step chain error
planning error
action-type bias
```

## 当前 Metrics 审计

| Metric / Probe | 能证明什么 | 不能证明什么 | 风险 |
| --- | --- | --- | --- |
| TSR | 全轨迹是否成功 | 是否用了 history | 被早期 grounding/action-type 错误主导 |
| progress | 失败走到多远 | 哪一步需要 memory | 长任务 late progress 不等于 long-horizon reasoning |
| step_sr / step_correct | teacher-forced step 是否动作正确 | history 是否必要 | 容易被 current-screen routine steps 稀释 |
| dense reward | format/type/content/coordinate 分解 | memory utility | reward 高不代表依赖 history |
| first-error analysis | 错误位置和粗类型 | causal memory defect | 需要 teacher-forced produce/consume check |
| V1 OOD repair | history-format arm 是否不崩 | history utilization | 代码里也明确禁止用 V1 claim utilization |
| V2 Condition-C drift | 注入错误 history 是否扰动 | 真 memory benefit | drift 可能只是 prompt sensitivity |
| V3 longdep pairs | 近/远依赖差异 | 依赖是否真实存在 | pair 构造必须先过 dependency gate；否则是 pseudo-dep |
| V4 oracle plan | plan 是否补足 | memory 是否补足 | planning 和 memory 是不同缺陷 |
| dependency gate | 数据中 battlefield 大小上界/估计 | 模型是否会失败 | Q3 需要 teacher-forced predictions |
| counterfactual utility | memory 是否 causally changes behavior | 数据规模本身 | 需要 no/true/wrong history 三条件 |

## 发现并修正的 Metric 问题

### Type 文本内容不能被忽略

原来 `gui360_long_horizon` 的部分 correctness 路径主要看：

```text
function match + coordinate hit
```

这会把一种关键 long-horizon failure 洗掉：

```text
type 框点对了，但 carried value 文本错了
```

这正是 memory failure 最常见的表现之一。

已修复：

- `gui360_long_horizon/harness/correctness.py`
- `gui360_long_horizon/experiments/capstone_runtime.py`

现在 `type/input/paste` 类动作如果 GT 有文本，必须文本匹配才算正确。

新增测试覆盖：

- `tests/gui360_long_horizon/test_harness.py`
- `tests/gui360_long_horizon/test_capstone_history_ab.py`

### Missing controls 不能等于 forced action

`gui360-balanced` 没有 a11y/control 信息。`controls=[]` 只能表示未知，不能表示 legal action space cardinality 为 0。

已修复：

- `gui360_long_horizon/data/pseudo_consumption.py`

### 快捷键不是 carried value

`{ENTER}`、`{VK_CONTROL}a`、`+{RIGHT}` 这类键盘命令不是语义值。

已修复：

- `gui360_long_horizon/data/carried_value.py`

规则：

```text
pure shortcut -> given
shortcut prefix + semantic text -> strip prefix, keep semantic text
```

## 对当前 GUI-360-balanced 结论的解释

GUI-360-balanced 的 dependency gate 不是说“绝对没有任何 long-horizon 样本”。

更准确地说：

```text
没有足够大的、足够干净的 cross-step dependency battlefield 来支持 whole-dataset history training。
```

原因：

1. OCR-backed survivor share 低。
2. 不用 OCR 的 data-only upper bound 仍低。
3. survivors 被少数模板/长文本/目录类操作集中贡献。
4. overall step distribution 仍然由 current-screen actions 主导。

因此 GUI-360-balanced 更适合作为：

```text
negative/control dataset
current-screen-driven confirmation dataset
```

不适合作为：

```text
primary history-utilization training battlefield
```

## 如何真正 reveal 当前 long-horizon 问题

### A. 先换 evaluation unit

不要用全量 overall accuracy 作为主 metric。

主 metric 应该是条件化的：

```text
memory_positive_accuracy
  = accuracy on no_history_wrong & true_history_correct & wrong_history_wrong

stale_history_regression_rate
  = no_history_correct & true/wrong_history_wrong

specificity_gap
  = true_history_accuracy - wrong_history_accuracy

value_accuracy_on_consumption_steps
  = exact/normalized carried-value content accuracy
```

### B. 构造四条件 probe

每个候选 step 跑四个条件：

```text
O: current screen only
G: true GT history / true segment memory
S: shuffled/wrong history
P: oracle plan without carried value
```

解释：

- `G > O` 才说明 history 有用。
- `G > S` 才说明不是 generic prompt/history effect。
- `P` 能区分 planning deficit 和 memory deficit。
- consumption-step value accuracy 是主指标，不是整体 step accuracy。

### C. 只在通过 dependency gate 的 subset 上看效果

如果一个 dataset 的 true dependency share 很低，whole-task SFT 会被 routine steps 淹没。

应只看：

```text
survivor dependency candidates
teacher-forced consumption steps
hard no-history-wrong subset
```

### D. 需要跨数据源找 battlefield

GUI-360-balanced 给出的是 negative result。要 reveal long-horizon 问题，应优先扫描：

- GUI-Odyssey hard cases
- AndroidControl / AndroidWorld long tasks
- 自己构造的 controlled current-state-equivalence tasks
- segment-summary memory utility 数据
- raw GUI-360 而非 balanced parquet，如果 raw 包含更完整 a11y/control/referee 信息

## 推荐下一步

### 1. 把 GUI-360-balanced 固定为 negative control

保留当前报告和 JSON：

- `reports/gui360_balanced_dependency_diagnostic.md`
- `reports/dependency_verdict.json`
- `reports/dependency_verdict_balanced_train_data_only_upper.json`
- `reports/dependency_verdict_balanced_test_data_only_upper.json`

### 2. 在 GUI-Odyssey / AndroidControl 上跑同一套 prevalence gate

目标不是直接训练，而是先问：

```text
data-only upper bound 是否明显 > 10%?
OCR/a11y-backed survivor 是否 >= 15%?
distance>=3 subset 是否 n>=30?
```

### 3. 对 survivor subset 跑 behavior interventions

必须记录：

```text
no_history prediction
true_history prediction
wrong_history prediction
oracle_plan prediction
```

并输出：

```text
memory_positive_n
memory_positive_precision
wrong_history_specificity
stale_regression_n
value_accuracy_delta
first_error_at_consumption_rate
```

### 4. 更新所有 history-utilization probe 的主 metric

主指标应从：

```text
overall step_correct
```

改为：

```text
value-aware consumption-step accuracy
true-vs-wrong-history specificity
first-error localization
```

## Practical Decision Rule

只有同时满足下面条件，才值得训练 history / memory model：

```text
dependency prevalence >= 15%
distance>=3 survivor n >= 30
no_history wrong & true_history correct subset is non-trivial
wrong_history does not recover the same subset
memory failures localize to consumption value, not grounding/format/planning
history benefit exceeds stale-history regressions
```

否则：

```text
do not train always-history or whole-task history SFT
use selective verifier/router or treat dataset as current-screen negative control
```

## Validation

Metric fixes and diagnostic tests pass:

```bash
python -m pytest tests/gui360_long_horizon -q
```

Result:

```text
68 passed
```