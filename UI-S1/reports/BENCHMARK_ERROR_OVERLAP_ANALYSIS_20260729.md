# Benchmark Trace Error Overlap Analysis

Date: 2026-07-29

## Executive summary

所有已完成的 AndroidControl 与 Mind2Web baseline 都保留了可逐 step 回溯的 trace。本报告没有修改原始 trace，而是按各 lane 的原始 parser/evaluator 重新计算互斥错误标签，并与已有 score/audit 逐项互证。

核心结论：

1. AndroidControl 的共同错误高度稳定。Low 有 965/7,708 steps 被全部 5 个模型做错，High 增至 1,758/7,708；其中分别有 829 和 1,007 steps 连错误类型都五模一致。
2. Mind2Web 的视觉错误更互补。11 个视觉模型共同失败 379/2,080 steps，但只有 7 steps 的 11 个模型错误类型完全一致。视觉 oracle 可解 1,701 steps，比最佳单模 TongUI-7B 多 600 steps。
3. Grounding/element selection 是最强的共享瓶颈。Mind2Web 共同硬核中 281/379 以 element miss 为主导；AndroidControl High 共同硬核中 657/1,758 以 grounding miss 为主导。
4. Action-space coverage 是第二个强信号。Mind2Web SELECT 有 53/79 steps 被全部视觉模型做错；AndroidControl 的 `wait`、`press_back`、`long_press` 在至少一个 setting 中具有异常高的共同失败率。
5. 模型规模不是单调信号。Mind2Web TongUI-7B 到 32B 净回退 19 steps，而 UI-TARS-2B 到 72B 净修正 439 steps。AndroidControl UI-AGILE-3B 到 7B 在 Low 反而净回退 118 steps。
6. HTML 与视觉模态有真实但有限的互补。MindAct 在 379 个视觉共同硬核中救回 109 个，跨模态 oracle 达到 1,810/2,080，仍剩 270 个共同失败。

## Trace inventory and comparability

| Benchmark lane | Complete traces | Rows per trace | Cross-model alignment |
|---|---:|---:|---|
| AndroidControl unified Low/High | 10 | 7,708 | setting 内按 ordered index；跨 setting 按 image SHA256 + GT action + GT bbox |
| AndroidControl original UI-R1 | 1 | 7,868 | 单独 evaluator lane，不进入 10-lane overlap |
| Mind2Web visual | 11 | 2,080 | `(annot_id, action_uid)` 完全一致 |
| Mind2Web MindAct HTML | 1 | 2,094 | 与 visual 共享 2,080 identities，另有 14 个 non-bbox actions |

AndroidControl Low/High 的强 identity multiset 完全一致，但有 58 steps 的 released GT parameter 不一致。跨 setting transition 只使用其余 7,650 个 exact-parameter matches；58 个冲突保留在 JSON 中单独审计。

## Error contract

AndroidControl 严格复用统一 evaluator：

- action/type：`pred_action == gt_action`；
- grounding：归一化欧氏距离小于 0.14；
- text/parameter：token-set F1 大于等于 0.5；
- Step SR：action 正确且对应 grounding/parameter 正确；
- exclusive taxonomy precedence：`parse_failure -> action_mismatch -> grounding_miss/parameter_miss -> success`。

Mind2Web 对每个模型使用其自己的 released local parser 和 scorer semantics，不用宽松 parser 修复输出：

- element：预测点在 GT bbox 内；
- operation：action ID 与 TYPE/SELECT value 的 token-set F1；
- Step SR：element 为 1 且 operation F1 为 1；
- exclusive taxonomy：parse、unsupported action、missing position/parameter、action mismatch、parameter miss、element miss、success。

这些标签是互斥归因，适合比较“主要失败原因”；原 evaluator 的 element、operation、action 子指标仍保留在 summary JSON 中。

## Real task and trajectory case studies

下面五例从锁定的逐 step trace 中按预先声明的机制类型选择，用于说明聚合统计背后的具体行为，不作为额外的总体效果估计。AndroidControl 坐标和距离均归一化到屏幕宽高；Mind2Web 坐标为归一化页面坐标。案例字段来自 `runs/collision-law/2026-07-30/rows.parquet`，并回查各模型原始 `predictions.jsonl`；错误标签仍由本报告的锁定 evaluator 重算。展示前排除了含邮箱、密码、电话号码或长数字标识符的任务文本。

### Case 1: instruction says click, released trajectory expects wait

- Identity: AndroidControl Low, `row_id=1`, episode `ac_goal_5f2021f390f831ec2f3d`.
- Task: `Click on the first result podcast`.
- History: empty.
- Released GT: `wait`.

| Model | Predicted action | Predicted coordinate | Audited label |
|---|---|---:|---|
| GUI-R1-3B | `click` | (0.211, 0.307) | action mismatch |
| GUI-R1-7B | `click` | (0.209, 0.437) | action mismatch |
| UI-AGILE-3B | `click` | (0.209, 0.336) | action mismatch |
| UI-AGILE-7B | `click` | (0.209, 0.349) | action mismatch |
| UI-R1-E-3B | `click` | (0.186, 0.307) | action mismatch |

五个模型都遵循自然语言指令并点击首个 podcast，但 released step label 要求 `wait`。这是真实的五模同类错误，同时也是一个重要的反例：高错误碰撞不必然表示五个模型共享同一种能力缺陷，也可能暴露 trajectory state、异步加载或 step annotation 与指令之间的错位。因此 consensus hard core 进入训练前必须先做状态/标注审计。

### Case 2: action consensus is correct, grounding consensus is wrong

- Identity: AndroidControl High, `row_id=1`, episode `ac_goal_d160a956920ac0ce98bc`.
- Task: `Browse Leonardo Da Vinci Mona lisa's painting for me on the Artsy app.`
- History: open Artsy -> open search -> type the query -> submit search.
- Released GT: `click` at (0.559, 0.321).

| Model | Predicted action | Predicted coordinate | Distance to GT | Audited label |
|---|---|---:|---:|---|
| GUI-R1-3B | `click` | (0.304, 0.300) | 0.256 | grounding miss |
| GUI-R1-7B | `click` | (0.314, 0.300) | 0.246 | grounding miss |
| UI-AGILE-3B | `click` | (0.302, 0.308) | 0.257 | grounding miss |
| UI-AGILE-7B | `click` | (0.301, 0.300) | 0.259 | grounding miss |
| UI-R1-E-3B | `click` | (0.903, 0.108) | 0.404 | grounding miss |

四个模型形成紧密但错误的坐标簇，距离约 0.25，已经明显超过 0.14 成功半径；第五个模型错得更远。这一例直接说明为什么观测 density 不能自动等同于 truth density：语义上合理的错误目标也会形成高密簇，且普通坐标投票会强化该错误。

### Case 3: larger model regresses on the action contract

- Identity: AndroidControl Low, `row_id=29`, episode `ac_goal_ea2651953f230f821987`.
- Task: `Open Etsy app.`
- History: empty.
- Released GT: `open_app("Etsy")`.

| Model | Prediction | Audited label |
|---|---|---|
| UI-AGILE-3B | `open_app("Etsy")` | success |
| UI-AGILE-7B | `click` at (0.164, 0.194) | action mismatch |
| GUI-R1-3B | `click` at (0.163, 0.179) | action mismatch |
| GUI-R1-7B | `click` at (0.163, 0.179) | action mismatch |
| UI-R1-E-3B | `click` at (0.162, 0.179) | action mismatch |

除 UI-AGILE-3B 外，模型都正确识别了 Etsy 图标，却输出通用 `click` 而不是 lane 要求的 `open_app` contract。UI-AGILE 从 3B 扩大到 7B 在这一行发生真实回退，说明规模提升不是成功集合包含关系，也说明 action-contract 错误与视觉定位错误必须分开处理。

### Case 4: seven cross-lineage models collide on the same wrong element

- Identity: Mind2Web `(annot_id, action_uid) = (8f6261cf-d665-4e61-93af-f50f0d366245, 0b5f2213-5eef-4cb0-b67b-16bb398e5285)`.
- Screenshot: [New York events step](../runs/mind2web/2026-07-27/data/ming2web_images/8f6261cf-d665-4e61-93af-f50f0d366245-0b5f2213-5eef-4cb0-b67b-16bb398e5285.jpg).
- Task: find all New York City events during September.
- History: change location -> type New York -> choose New York, NY -> open `Filter by`.
- GT: click `Next month`, bbox $x\in[0.639,0.658]$, $y\in[0.070,0.103]$.

| Models | Shared or representative prediction | Audited outcome |
|---|---|---|
| TongUI-3B/7B/32B | click `Filter by Date` near (0.22, 0.98-0.99) | element miss |
| CogAgent-18B | click `Filter by Date` at (0.221, 0.985) | element miss |
| SeeClick-9.6B | click at (0.220, 0.980) | element miss |
| UI-TARS-7B/72B | click near (0.21, 0.99) | element miss |
| ShowUI-2B | click `9/1` at (0.220, 0.090) | element miss |
| Qwen2.5-VL-7B | click `New York, NY` at (0.180, 0.470) | element miss |
| Qwen2.5-VL-3B / UI-TARS-2B | unsupported lowercase action / parse failure | failure |

全部 11 个视觉模型失败，其中 9 个是 element miss。更重要的是，TongUI、CogAgent、SeeClick 和 UI-TARS 共七个跨谱系模型集中到页面底部几乎相同的错误区域，而 GT 在页面顶部右侧。这是“错误质量也会集中”的直接实例：family 去重可以减少重复计票，但跨 family 的共享语义误读仍可能制造错误 mode。截图同时暴露了测量侧风险：底部 `Filter by Date` 清晰可见，而 GT `Next month` 在当前渲染中并不明显，因此该碰撞也可能混合 partial render/state mismatch，不能只归因于模型能力。

### Case 5: HTML grounding rescues an all-visual failure

- Identity: Mind2Web `(annot_id, action_uid) = (1d73ad40-f7f8-435e-a83d-8b38534427fd, 5a014915-42f1-4c39-a61e-447b5480e8c4)`.
- Screenshot: [brown loungewear step](../runs/mind2web/2026-07-27/data/ming2web_images/1d73ad40-f7f8-435e-a83d-8b38534427fd-5a014915-42f1-4c39-a61e-447b5480e8c4.jpg).
- Task: find the cheapest women's plus-size brown loungewear in size 3XL.
- History: open Women/Loungewear -> choose 3XL -> sort Price Low-High.
- GT: click the `Color` filter, bbox $x\in[0.008,0.168]$, $y\in[0.783,0.825]$.

| Lane/model | Prediction | Audited outcome |
|---|---|---|
| TongUI-7B | click at (0.210, 0.090), reasoning refers to brown color | element miss |
| TongUI-32B | click `Brown` at (0.340, 0.090) | element miss |
| SeeClick-9.6B | click at (0.190, 0.690) | element miss, bbox distance 0.096 |
| UI-TARS-7B | click at (0.195, 0.694) | element miss, bbox distance 0.093 |
| Other visual models | seven element/action/parse failures | failure |
| MindAct HTML | select the positive DOM candidate and emit `CLICK` | success |

视觉模型通常理解下一步与颜色筛选有关，但无法把这个意图落到正确的折叠式 `Color` 控件上；两个最接近的视觉点仍落在 bbox 外约 0.09。截图可见顶部颜色圆点，但 GT `Color` bbox 位于页面左下区域且在当前渲染中不醒目，说明这里同样含有可观测性因素。MindAct 使用 HTML candidate grounding 后同时命中元素和操作，成为 109 个 `MindAct-only` steps 之一。这一例支持“HTML 提供有限但真实的候选元素互补”，而不是“HTML 可以替代视觉”：完整 contingency 中 visual-only 仍有 804 行。

### Trace provenance

- AndroidControl 原始输出：`runs/androidcontrol-rft/2026-07-29/artifacts/<model>/<setting>/predictions.jsonl`。
- Mind2Web 视觉输出：各 lane 在 `runs/mind2web-*/2026-07-28/artifacts/**/predictions.jsonl` 下的锁定 merged/full trace。
- MindAct 输出：`runs/mindact/2026-07-29/artifacts/full/test_task_predictions_top50.json`。
- 案例 identity、标签和统计可以分别由 `analyze_androidcontrol.py` 与 `analyze_mind2web.py` 的 reproduction 命令重算；MD 中没有手工改写 evaluator 结果。

## AndroidControl findings

### Per-model results and exclusive errors

| Model | Low Step | High Step | Low main errors: action / ground / param / parse | High main errors: action / ground / param / parse |
|---|---:|---:|---:|---:|
| UI-AGILE-3B | 6,096 (79.1%) | 4,517 (58.6%) | 811 / 524 / 277 / 0 | 1,776 / 1,122 / 293 / 0 |
| UI-AGILE-7B | 5,978 (77.6%) | 4,666 (60.5%) | 952 / 536 / 242 / 0 | 1,544 / 1,237 / 261 / 0 |
| UI-R1-E-3B | 3,739 (48.5%) | 1,764 (22.9%) | 1,027 / 949 / 1,993 / 0 | 2,567 / 2,326 / 1,051 / 0 |
| GUI-R1-3B | 4,385 (56.9%) | 2,962 (38.4%) | 1,194 / 885 / 1,240 / 4 | 2,745 / 1,444 / 544 / 13 |
| GUI-R1-7B | 4,487 (58.2%) | 3,466 (45.0%) | 1,261 / 739 / 1,221 / 0 | 2,205 / 1,335 / 699 / 3 |

### Shared hardness and oracle headroom

| Setting | All 5 fail | At least 1 succeeds | All 5 succeed | Mixed/disagreement |
|---|---:|---:|---:|---:|
| Low | 965 (12.5%) | 6,743 (87.5%) | 3,266 (42.4%) | 3,477 (45.1%) |
| High | 1,758 (22.8%) | 5,950 (77.2%) | 937 (12.2%) | 5,013 (65.0%) |

相对最佳单模，Low oracle 多解 647 steps（6,743 vs 6,096），High oracle 多解 1,284 steps（5,950 vs 4,666）。High 不仅整体更难，模型分歧也显著更多，因此 High 是更有价值的 router/verifier 训练池。

共同硬核的主导错误：

| Setting | Action mismatch dominant | Grounding dominant | Parameter dominant | 五模同类错误总计 |
|---|---:|---:|---:|---:|
| Low, hard core = 965 | 527 | 273 | 165 | 829 (85.9%) |
| High, hard core = 1,758 | 932 | 657 | 169 | 1,007 (57.3%) |

Low 的共同硬核更像稳定、可复现的专项缺陷；High 的共同失败中混合错误更多，但仍以 action 与 grounding 为主。

### Action-conditioned hard core

| GT action | Low all-fail | High all-fail |
|---|---:|---:|
| click | 287/4,598 (6.2%) | 907/4,598 (19.7%) |
| type | 30/569 (5.3%) | 93/569 (16.3%) |
| open_app | 54/554 (9.7%) | 97/554 (17.5%) |
| scroll | 157/1,138 (13.8%) | 315/1,138 (27.7%) |
| press_back | 93/315 (29.5%) | 165/315 (52.4%) |
| wait | 341/527 (64.7%) | 176/527 (33.4%) |
| long_press | 3/7 (42.9%) | 5/7 (71.4%) |

`long_press` 只有 7 个样本，不能据此做稳定排序。后续 D1 显示所有模型在 High 都更频繁预测 `wait`；例如 GUI-R1-3B 的 `pred=wait` 基率从 0.84% 升到 8.42%，precision 从 82.8% 降到 25.9%。因此 `wait` 的 Low/High 反转至少部分来自 hedging，不能作为状态/历史理解改善的证据。

### Cross-setting and scale transitions

Low 到 High 的 exact-parameter transition，denominator = 7,650：

| Model | Both success | Low only | High only | Net High gain |
|---|---:|---:|---:|---:|
| UI-AGILE-3B | 4,090 | 1,962 | 411 | -1,551 |
| UI-AGILE-7B | 4,156 | 1,778 | 492 | -1,286 |
| UI-R1-E-3B | 1,574 | 2,165 | 190 | -1,975 |
| GUI-R1-3B | 2,655 | 1,690 | 301 | -1,389 |
| GUI-R1-7B | 3,127 | 1,320 | 332 | -988 |

所有模型在 High 都净退化，但每个模型仍有 190--492 个 High-only successes。后续 D5 显示这些样本的 Low grounding distance 相对全部 Low failures 没有显著集中（五模型 Mann-Whitney `p=0.309--0.930`），且 High-only 数量只有 matched-marginal independence 期望的 21.1%--47.2%。当前证据既不支持“主要是阈值抖动”，也不足以证明 history-conditioned correction。

3B 到 7B：

| Family/setting | 3B only | 7B only | Net 7B gain |
|---|---:|---:|---:|
| UI-AGILE Low | 339 | 221 | -118 (-1.5 pp) |
| UI-AGILE High | 624 | 773 | +149 (+1.9 pp) |
| GUI-R1 Low | 376 | 478 | +102 (+1.3 pp) |
| GUI-R1 High | 482 | 986 | +504 (+6.5 pp) |

规模增大有大量双向迁移，而不是简单包含关系。训练时应保留 3B-only 样本作为 regression set。

### Grounding distance signal

对 exclusive grounding misses，near miss 定义为距离落在 `[0.14, 0.28)`，即刚好错过阈值但仍在两倍半径内。

- Low：各模型有 39.9%--46.6% grounding misses 属于 near miss。
- High：各模型只有 22.2%--29.8% 属于 near miss，其余是大于等于 0.28 的 gross miss。

因此 High 的 grounding 退化主要来自更大的定位偏差，而不是阈值边缘抖动。Low 的 near misses 更适合坐标 refinement 或局部 crop 二阶段训练。

后续 D3 用 Cohen kappa 和 1,000 次 matched-marginal permutation 替代受失败率抬升的 Jaccard。High 中 UI-AGILE-3B/7B kappa 为 0.621，GUI-R1-3B/7B 为 0.609；Low 中分别为 0.784 和 0.774，均有 permutation `p<0.001`。这支持 family 内稳定错误结构，但不再用原始 Jaccard 量化该结论。

## Mind2Web findings

### Per-model visual results

| Model | Step success | Main exclusive errors |
|---|---:|---|
| TongUI-7B | 1,101 (52.9%) | element 714, action 220, parameter 44, parse 1 |
| TongUI-32B | 1,082 (52.0%) | element 758, action 205, parameter 34, parse 1 |
| CogAgent-18B | 1,043 (50.1%) | element 678, action 181, parameter 56, parse 122 |
| TongUI-3B | 1,019 (49.0%) | element 803, action 209, parameter 49 |
| UI-TARS-72B | 831 (40.0%) | element 826, action 291, parse 118, missing-position 12 |
| UI-TARS-7B | 700 (33.7%) | element 1,019, action 288, missing-position 34, parse 31 |
| SeeClick-9.6B | 456 (21.9%) | element 1,340, action 252, parameter 32 |
| UI-TARS-2B | 392 (18.8%) | element 963, parse 460, action 265 |
| ShowUI-2B | 377 (18.1%) | element 1,346, action 258, unsupported 98 |
| Qwen2.5-VL-7B | 114 (5.5%) | element 1,358, action 333, parse 235 |
| Qwen2.5-VL-3B | 19 (0.9%) | unsupported 829, action 508, parse 459, element 206 |

原始 base-model lanes 的低分很大一部分与 output/action contract 不匹配有关，不能直接解释为纯视觉能力不足。

### Shared errors and complementarity

- all 11 fail：379/2,080 (18.2%)；
- visual oracle：1,701/2,080 (81.8%)；
- exactly one model succeeds：269；
- model-disagreement pool：1,701，因为所有 oracle-success rows 都至少有一个模型失败。

最佳单模 TongUI-7B 成功 1,101，11 模 visual oracle 多 600 steps，相当于总集的 28.8 percentage points。但 deployable subset（parse failure rate <5%、Step micro >30%）只包含 TongUI-3B/7B/32B 与 UI-TARS-7B，其 oracle 为 1,475/2,080 (70.91%)。因此 11 模 oracle 只作为附录上界，正文使用 deployable oracle；oracle headroom 本身不能证明 learned router 可以兑现。

Mind2Web 聚合口径显式区分为 2,080-step micro 与 252-episode macro。11 模 oracle 分别为 81.78% 和 83.97%；不得与各模型已有 episode-macro headline 混用。

共同硬核的 dominant error 为：element miss 281、action mismatch 84、parameter miss 5、tie 9。只有 7/379 steps 的 11 模型错误类型完全一致，其中 element miss 5、action mismatch 2。与 AndroidControl 相比，Mind2Web 的“共同失败”通常不是“共同错法”，模型多样性明显更高。

按 GT action：

| Action | Rows | All 11 fail | Visual oracle |
|---|---:|---:|---:|
| CLICK | 1,774 | 287 (16.2%) | 1,487 (83.8%) |
| TYPE | 227 | 39 (17.2%) | 188 (82.8%) |
| SELECT | 79 | 53 (67.1%) | 26 (32.9%) |

SELECT 是最明确的 action-space curriculum 候选，但样本仅 79 个，应先检查 parser/action support 与 annotation，再扩充训练数据。

### Family and scale structure

- TongUI family oracle：1,406/2,080；比 family 最佳单模多 305。
- UI-TARS family oracle：1,057/2,080；比 UI-TARS-72B 多 226。
- Qwen2.5-VL 3B/7B 的原 failure Jaccard = 0.946 主要由高失败率边际决定，不再作为主统计。chance-corrected D3 显示 Qwen2.5-VL-3B 与 ShowUI-2B kappa 为 0.003 (`p=0.458`)，与 UI-TARS-2B 为 -0.008 (`p=0.894`)，没有超出机会水平的共享失败证据。
- TongUI-3B/7B 的 chance-corrected failure kappa 为 0.587，matched-marginal permutation `p<0.001`，支持 family 内稳定错误结构。

| Transition | Left only | Right only | Net right gain |
|---|---:|---:|---:|
| Qwen2.5-VL 3B -> 7B | 8 | 103 | +95 (+4.6 pp) |
| TongUI 3B -> 7B | 174 | 256 | +82 (+3.9 pp) |
| TongUI 7B -> 32B | 257 | 238 | -19 (-0.9 pp) |
| UI-TARS 2B -> 7B | 96 | 404 | +308 (+14.8 pp) |
| UI-TARS 7B -> 72B | 176 | 307 | +131 (+6.3 pp) |

TongUI-7B 是当前 sweet spot，但“32B 回退来自过拟合”只是待验证假设；不同 checkpoint 的训练数据与优化过程可能是混杂因素。

TongUI-7B/32B 的 element misses 中，距离 bbox 不超过 0.05 的比例约 17%--18%；CogAgent 与 UI-TARS-72B 约 10%--12%。其余多数 element miss 不是简单边界误差，需要重新选择候选元素。

### Visual versus HTML

MindAct full 2,094-step audit 重新计算完全一致。在共享的 2,080 steps 上：

| Outcome | Count | Rate |
|---|---:|---:|
| Visual oracle and MindAct both succeed | 897 | 43.1% |
| Visual only | 804 | 38.7% |
| MindAct only | 109 | 5.2% |
| Both fail | 270 | 13.0% |

MindAct 救回 109/379 (28.8%) visual-hard steps，使 combined oracle 达到 1,810/2,080 (87.0%)。HTML routing/fusion 有价值，但 visual-only 是 MindAct-only 的 7.4 倍，当前数据不支持用 HTML 替代视觉。

## Learnable signals

### 1. Ensemble and router feasibility

后续 E1 在五个 shared grouped folds 上测试了 plurality、geometric median、weighted geometric median 与 text medoid。AndroidControl Low/High 的 dev selection 都只保留最佳单模，test delta 均为 0.00 pp；Mind2Web weighted composite 从 51.78% 提升到 54.32%，delta +2.55 pp，negative dispersion AUROC 为 0.660。是否保留 Mind2Web ensemble 线需等待 E5 MDE；AndroidControl ensemble 是负结果。

Routing 从默认方案降级为按pool判定的假设。E2 使用严格 nested grouped split，仅在 disagreement pool 上测试 GT-free 特征。T2 pooled headroom capture 为 AndroidControl Low 29.92%、High 5.69%、Mind2Web 14.65%。因此 High 低于10%门槛并关闭routing线；Low与Mind2Web只在加入所有模型输出后通过，属于多模型reranking，必须与等算力E1比较。T2一致性特征在AC没有新增信息，在Mind2Web还低于T1的15.75%。

### 2. Error-aware verifier

AndroidControl E1 中因每fold只选择单模，vote margin 与 dispersion 都是常数，AUROC 0.5，没有产生可用 verifier 信号。Mind2Web 的 negative geometric dispersion AUROC 为 0.660、vote margin 为 0.548、两者 logistic 组合为 0.623；这只支持继续评估几何分歧信号，不足以宣称 verifier 已可用。

### 3. Curriculum and sampling weights

建议分四层：

1. contract errors：parse/unsupported/missing field，先用 constrained decoding 与格式 SFT 修复；
2. action-space errors：SELECT、wait、press_back、long_press，按 action 重采样；
3. near grounding misses：做 local crop、coordinate refinement、hard-negative element contrast；
4. consensus hard core：低权重起步，先做 annotation/state audit，再进入后期 curriculum。

58 个 AndroidControl GT parameter conflicts 必须 quarantine，不能直接作为 Low/High preference pair。

### 4. Regression-preserving distillation

规模增大存在大量 left-only successes。蒸馏或 merge 时应把 UI-AGILE Low 的 339 个 3B-only、TongUI 7B 的 257 个 7B-only、UI-TARS 7B 的 176 个 7B-only steps 固定为 regression suites，而不是只优化总均值。

### 5. Cross-modal hard-example mining

109 个 MindAct-only steps 是 DOM/HTML signal 的直接正样本；804 个 visual-only steps 是视觉不可替代的反例；270 个双模态共同失败应优先做 annotation、candidate recall、state observability 审计。

## What is supported and what remains a hypothesis

直接由 trace 支持：错误重合计数、transition、oracle headroom、action-conditioned failure、near/gross miss、跨模态 contingency。

尚不能由当前 trace 单独证明：

- High 退化究竟由 history、instruction abstraction、screen complexity 中哪一项导致；
- 32B 回退是否由 overfitting 导致；
- parser-invalid outputs 是否代表底层能力不足；
- 共同 hard core 是否包含 annotation error 或 partial observability；
- 哪个 routing signal 在 held-out task/domain 上能泛化。

当前 trace 没有统一可比的 token log-probability，因此不能声称已经发现可靠的置信度校准信号。下一步应在相同 identity split 上训练 router/verifier，并严格按 task/domain 做 held-out evaluation。

## Reproduction

```bash
.venv-ac-vllm/bin/python \
  runs/error-overlap-analysis/2026-07-29/analyze_androidcontrol.py \
  --output runs/error-overlap-analysis/2026-07-29/androidcontrol_summary.json

runs/mindact/2026-07-29/run_python.sh \
  runs/error-overlap-analysis/2026-07-29/analyze_mind2web.py \
  --output runs/error-overlap-analysis/2026-07-29/mind2web_summary.json
```

Both analyzers fail closed on incomplete identities or any mismatch with the audited evaluator metrics.