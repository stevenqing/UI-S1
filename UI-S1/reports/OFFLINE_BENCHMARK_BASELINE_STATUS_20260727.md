# Mind2Web + AndroidControl Baseline 总览

更新时间：2026-07-29

## 1. 当前结论

目前有两个 benchmark、二十五个已完成且可审计的模型/设置结果：

1. Mind2Web Cross-Task / SeeClick：论文 anchor 复现通过。
2. Mind2Web Cross-Task / ShowUI-2B 公共 checkpoint：完整受控结果，但未精确复现论文 ShowUI-ZS anchor。
3. Mind2Web Cross-Task / TongUI-3B：完整受控结果，但与论文主表 anchor 存在超过 1 pp 的差异。
4. Mind2Web Cross-Task / TongUI-7B：完整受控结果，但与论文主表 anchor 存在超过 1 pp 的差异。
5. Mind2Web Cross-Task / Qwen2.5-VL-3B：lower-bound 复现通过。
6. Mind2Web Cross-Task / Qwen2.5-VL-7B：完整受控 lower-bound 结果。
7. AndroidControl-Low / OS-Atlas-Pro-7B：完整 successor baseline。
8. AndroidControl-High / OS-Atlas-Pro-7B：完整 successor baseline。
9. AndroidControl-High / Qwen2.5-VL-3B：完整 base lower bound，严格格式下为 0 分。
10. AndroidControl-Low / Qwen2.5-VL-3B：完整 base lower bound，严格格式下为 0 分。
11. AndroidControl-High / Qwen2.5-VL-7B：完整 base lower bound，严格格式下为 0 分。
12. AndroidControl-Low / Qwen2.5-VL-7B：完整 base lower bound，严格格式下为 0 分。
13. AndroidControl-Low / InfiGUI-R1-3B：官方 8,444-step lane 复现通过。
14. AndroidControl-High / InfiGUI-R1-3B：官方 8,444-step lane 复现通过。
15. Mind2Web Cross-Task / CogAgent public checkpoint：完整受控迁移结果，不是论文 anchor 复现。
16. Mind2Web Cross-Task / UI-TARS-2B-SFT：完整公共 checkpoint 受控迁移结果。
17. Mind2Web Cross-Task / UI-TARS-7B-SFT：完整公共 checkpoint 受控迁移结果。
18. AndroidControl-Low / UI-TARS-2B-SFT：完整公共 checkpoint 受控迁移结果。
19. AndroidControl-High / UI-TARS-2B-SFT：完整公共 checkpoint 受控迁移结果。
20. AndroidControl-Low / UI-TARS-7B-SFT：完整公共 checkpoint 受控迁移结果。
21. AndroidControl-High / UI-TARS-7B-SFT：完整公共 checkpoint 受控迁移结果。
22. Mind2Web Cross-Task / TongUI-32B：完整受控结果，独立审计通过。
23. Mind2Web Cross-Task / UI-TARS-72B-SFT：完整公共 checkpoint 受控迁移结果，独立审计通过。
24. MindAct HTML / Mind2Web Cross-Task：官方 2,094-action top-50 tournament 完整结果，独立审计通过。
25. AndroidControl-Low / UI-AGILE-3B：官方 7,708-step 完整结果，独立审计通过。

AndroidControl 论文 Table 5 的 OS-Atlas-7B zero-shot OOD checkpoint 未公开，因此该行精确复现仍为 `BLOCKED`。InfiGUI-R1-3B Low/High 均已完成官方 8,444 行并通过独立严格审计；该 evaluator lane 与 OS-Atlas 7,708-step lane 分开报告。TongUI-3B/7B 和 Qwen2.5-VL-3B/7B 均已完成全部 2,080 行、独立计分和审计。Qwen2.5-VL-7B 两项 anchor delta 略超过 1 pp，因此作为完整受控结果而不是严格 anchor PASS。

| Benchmark | 完成的受控设置 | 精确论文复现 | 当前阻断/运行项 |
| --- | ---: | ---: | --- |
| Mind2Web Cross-Task | 12 | 2 | UI-TARS-72B 与 MindAct 均已完成并通过独立审计 |
| AndroidControl 7,708-step | 11 | 0 | UI-AGILE-3B Low 已完成；其余 9 条 UI-AGILE/UI-R1-E/GUI-R1 lane 正在执行 |
| AndroidControl InfiGUI official 8,444-step | 2 | 2 | Low/High 均完成并通过严格审计 |

## 2. 统一口径

### 2.1 Mind2Web

- Split：Cross-Task / `test_task`。
- 原始动作：2,094。
- 可评分 bbox 动作：2,080。
- Episode：252。
- Element Accuracy：预测归一化坐标是否落在 GT bbox 内，边界包含。
- Operation F1：动作类型以及 `TYPE`/`SELECT` value 的 token-set F1。
- Step SR：Element 正确且 Operation F1 严格等于 1。
- 主结果：先在每个 episode 内求均值，再对 252 个 episode 做 macro mean。
- Parse rate 单独报告，不用 parser 放宽来改善主指标。

### 2.2 AndroidControl

- 固定 `ac_idx.txt`：7,708 个 step identity。
- Type Accuracy：动作类型完全匹配。
- Grounding Accuracy：在 click-type-match 样本中，预测点击位置满足 OS-Atlas 距离规则。
- Step SR：动作类型和对应参数均成功。
- Low：提供当前 low-level instruction。
- High：只提供 high-level goal 和历史，不提供当前 low-level instruction。
- AndroidControl 的 Type/Grounding/SR 与仓库历史的 1,543-task Pass@K 或 task-success 不是同一口径，不能直接比较。

## 3. 已完成结果

### 3.1 Mind2Web Cross-Task

所有数值均为 episode-macro 百分比。

| Model / 设置 | Steps | Element Acc | Operation F1 | Step SR | Parse | 论文 anchor | 结论 |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| SeeClick corrected full coverage | 2,080 | **28.5204** | **86.9429** | **25.9765** | 100% | 28.3 / 87.0 / 25.5 | `PASS`，三项均在 1 pp 内 |
| ShowUI-2B public checkpoint / ZS | 2,080 | **23.3366** | **81.9835** | **19.9609** | 100% | 21.4 / 85.2 / 18.6 | 完整公共 checkpoint 结果；不是精确 anchor 复现 |
| TongUI-3B (GUI-Net-1M) | 2,080 | **56.2867** | **89.2036** | **51.3824** | 100% | 53.4 / 89.0 / 48.8 | 完整受控结果；不是精确 anchor 复现 |
| TongUI-7B (GUI-Net-1M) | 2,080 | **60.8320** | **89.1854** | **55.6095** | 99.9519% | 58.1 / 88.7 / 53.4 | 完整受控结果；不是精确 anchor 复现 |
| Qwen2.5-VL-3B-Instruct | 2,080 | **2.0448** | **14.4735** | **0.7949** | 77.9327% | 2.5 / 14.5 / 0.4 | `PASS`，三项均在 1 pp 内 |
| Qwen2.5-VL-7B-Instruct | 2,080 | **7.2573** | **71.6565** | **5.8983** | 88.7019% | 6.2 / 72.8 / 5.0 | 完整受控结果；两项略超 1 pp |
| CogAgent public transfer | 2,080 | **58.9280** | **85.4882** | **56.0392** | 94.1346% | 22.4 / 53.0 / 17.6 | 完整公共 checkpoint 结果；不是精确 anchor 复现 |
| UI-TARS-2B-SFT public transfer | 2,080 | **28.3011** | **65.1850** | **22.6630** | 77.8846% | 62.3 / 90.0 / 56.3 | 完整公共 checkpoint 下界；不是精确 anchor 复现 |
| UI-TARS-7B-SFT public transfer | 2,080 | **44.1924** | **84.7497** | **37.8746** | 98.5096% | 73.1 / 92.2 / 67.1 | 完整公共 checkpoint 下界；不是精确 anchor 复现 |
| UI-TARS-72B-SFT public transfer | 2,080 | **52.3932** | **80.3124** | **43.4052** | 94.3269% | 74.7 / 92.5 / 68.6 | TP=4 完整公共 checkpoint 迁移；audit PASS |
| MindAct official HTML tournament | 2,094 | **54.3660** | **74.3162** | **50.0012** | 100% | 55.1 / 75.7 / 52.0 | 独立 HTML lane；252 episodes、top-50、seed 123、audit PASS |

#### SeeClick 说明

- Anchor delta：Element `+0.2204 pp`，Operation F1 `-0.0571 pp`，Step SR `+0.4765 pp`。
- 上游 evaluator 因 `32" curved monitor` 未转义而静默跳过一个 `TYPE` step；修复 GT 构造后补跑该 step，达到 2,080/2,080 覆盖。
- 独立审计：0 duplicate、0 score mismatch、parse 100%。
- 报告：[Gate 1](../runs/mind2web/2026-07-27/GATE_1_REPORT.md)
- Audit：[gate1_audit.json](../runs/mind2web/2026-07-27/artifacts/gate1_audit.json)

#### ShowUI 说明

- Anchor delta：Element `+1.9366 pp`，Operation F1 `-3.2165 pp`，Step SR `+1.3609 pp`。
- 公共 `showlab/ShowUI-2B` 是通用 checkpoint，不是论文中 Mind2Web downstream-finetuned `ShowUI` 行。
- 2,080/2,080 响应可解析；1,982/2,080 使用 benchmark 支持的 `CLICK/TYPE/SELECT` action。
- 模型输出计数：`CLICK=1,974`、`SELECT=8`、`INPUT=77`、`SCROLL=14`、`ENTER=7`、`TYPE=0`。
- 上游 dataset 会把 `anno_id` 覆盖成 step index，使其所谓 Macro 退化为 micro；正式 scorer 保留原始 `annot_id` 并重算真实 episode-macro。
- 报告：[ShowUI final report](../runs/mind2web-showui/2026-07-28/FINAL_REPORT.md)
- Score：[score.json](../runs/mind2web-showui/2026-07-28/artifacts/merged/score.json)
- Audit：[audit.json](../runs/mind2web-showui/2026-07-28/artifacts/merged/audit.json)

### 3.2 AndroidControl OS-Atlas-Pro-7B

这两行是公开 successor model 的结果，不是 OS-Atlas Table 5 zero-shot OOD 结果。

| 设置 | Steps | Type Acc | Grounding Acc | Step SR | Parse |
| --- | ---: | ---: | ---: | ---: | ---: |
| Low | 7,708 | **93.4743%** | **86.7576%** | **83.9647%** | 100% |
| High corrected prompt | 7,708 | **86.3129%** | **77.9466%** | **71.3285%** | 100% |
| Low - High | - | **+7.1614 pp** | **+8.8110 pp** | **+12.6362 pp** | 0 pp |

范围边界：

- Model：`OS-Copilot/OS-Atlas-Pro-7B` revision `6c0135de0627db98533ac4b47ae71fa17cf21c48`。
- Pro 模型训练使用全部七个 agent datasets，因此不能作为 Table 5 OOD checkpoint。
- High 已按官方 multi-step prompt 修正非空 history 后的换行并完整重跑。
- Low/High 均为 7,708 unique identities、0 missing、0 duplicate、0 runtime error。
- High 报告：[FINAL_REPORT.md](../runs/androidcontrol-pro/2026-07-27/FINAL_REPORT.md)
- Low 报告：[LOW_REPORT.md](../runs/androidcontrol-pro/2026-07-27/LOW_REPORT.md)
- 对比：[LOW_HIGH_COMPARISON.md](../runs/androidcontrol-pro/2026-07-27/LOW_HIGH_COMPARISON.md)

### 3.3 AndroidControl Qwen2.5-VL-3B High

| 设置 | Steps | Type Acc | Grounding Acc | Step SR | Parse |
| --- | ---: | ---: | ---: | ---: | ---: |
| High / strict OS-Atlas parser | 7,708 | **0.0000%** | **0.0000%** | **0.0000%** | 0.0000% |
| High / flexible diagnostic | 7,708 | 51.6087% | 2.3207% | 11.5205% | 94.6290% |

- 正式主结果使用上游 exact parser；flexible 行仅用于诊断，不进入 benchmark 主指标。
- 7,294 条输出使用大写 `Actions:`，414 条使用单数 `Action:`，0 条命中严格小写 `actions:\n`。
- 不做 lowercase alias、delimiter repair 或 GT 依赖修复。
- 7,708 unique identities、0 missing、0 duplicate、0 extra，vLLM-aware audit PASS。
- 报告：[QWEN3_HIGH_REPORT.md](../runs/androidcontrol-pro/2026-07-27/QWEN3_HIGH_REPORT.md)
- Score：[score.json](../runs/androidcontrol-pro/2026-07-27/artifacts/qwen-3b-high/vllm-merged/score.json)
- Audit：[audit.json](../runs/androidcontrol-pro/2026-07-27/artifacts/qwen-3b-high/vllm-merged/audit.json)

### 3.4 AndroidControl Qwen2.5-VL-3B Low

| 设置 | Steps | Type Acc | Grounding Acc | Step SR | Parse |
| --- | ---: | ---: | ---: | ---: | ---: |
| Low / strict OS-Atlas parser | 7,708 | **0.0000%** | **0.0000%** | **0.0000%** | 0.0000% |
| Low / flexible diagnostic | 7,708 | 70.0701% | 3.5733% | 15.7499% | 98.6767% |

- 7,606 条输出使用大写 `Actions:`，102 条使用单数 `Action:`，0 条命中严格小写 `actions:\n`。
- Flexible diagnostic 相对 High 为 Type `+18.4613 pp`、Grounding `+1.2526 pp`、Step SR `+4.2294 pp`。
- 正式主结果不做大小写或 delimiter 修复；flexible 数字不进入 benchmark 主指标。
- 7,708 unique identities、0 missing、0 duplicate、0 extra，vLLM-aware audit PASS。
- 报告：[QWEN3_LOW_REPORT.md](../runs/androidcontrol-pro/2026-07-27/QWEN3_LOW_REPORT.md)
- Score：[score.json](../runs/androidcontrol-pro/2026-07-27/artifacts/qwen-3b-low/vllm-merged/score.json)
- Audit：[audit.json](../runs/androidcontrol-pro/2026-07-27/artifacts/qwen-3b-low/vllm-merged/audit.json)

### 3.5 AndroidControl Qwen2.5-VL-7B High

| 设置 | Steps | Type Acc | Grounding Acc | Step SR | Parse |
| --- | ---: | ---: | ---: | ---: | ---: |
| High / strict OS-Atlas parser | 7,708 | **0.0000%** | **0.0000%** | **0.0000%** | 0.0000% |
| High / flexible diagnostic | 7,708 | 66.8656% | 3.7829% | 13.7779% | 95.5890% |

- 0 条命中严格小写 `actions:\n`；不做 delimiter repair。
- 7,708 unique identities、0 missing、0 duplicate、0 extra，vLLM-aware audit PASS。
- 报告：[QWEN7_HIGH_REPORT.md](../runs/androidcontrol-pro/2026-07-27/QWEN7_HIGH_REPORT.md)
- Score：[score.json](../runs/androidcontrol-pro/2026-07-27/artifacts/qwen-7b-high/vllm-merged/score.json)
- Audit：[audit.json](../runs/androidcontrol-pro/2026-07-27/artifacts/qwen-7b-high/vllm-merged/audit.json)

### 3.6 AndroidControl Qwen2.5-VL-7B Low

| 设置 | Steps | Type Acc | Grounding Acc | Step SR | Parse |
| --- | ---: | ---: | ---: | ---: | ---: |
| Low / strict OS-Atlas parser | 7,708 | **0.0000%** | **0.0000%** | **0.0000%** | 0.0000% |
| Low / flexible diagnostic | 7,708 | 78.5807% | 5.3366% | 17.9165% | 99.5459% |

- 0 条命中严格小写 `actions:\n`；不做 delimiter repair。
- Flexible diagnostic 相对 High 为 Type `+11.7151 pp`、Grounding `+1.5537 pp`、Step SR `+4.1386 pp`。
- 7,708 unique identities、0 missing、0 duplicate、0 extra，vLLM-aware audit PASS。
- 报告：[QWEN7_LOW_REPORT.md](../runs/androidcontrol-pro/2026-07-27/QWEN7_LOW_REPORT.md)
- Score：[score.json](../runs/androidcontrol-pro/2026-07-27/artifacts/qwen-7b-low/vllm-merged/score.json)
- Audit：[audit.json](../runs/androidcontrol-pro/2026-07-27/artifacts/qwen-7b-low/vllm-merged/audit.json)

#### TongUI-3B 说明

- Anchor delta：Element `+2.8867 pp`，Operation F1 `+0.2036 pp`，Step SR `+2.5824 pp`。
- 2,080/2,080 响应可解析且使用支持的 `CLICK/TYPE/SELECT` action。
- 输出计数：`CLICK=1,791`、`TYPE=245`、`SELECT=44`。
- 四个 shard 各 520 行；ordered merge、scorer 字节级复算和 2,080-identity audit 均通过。
- 报告：[TongUI-3B report](../runs/mind2web-tongui/2026-07-28/TONGUI_3B_REPORT.md)
- Score：[score.json](../runs/mind2web-tongui/2026-07-28/artifacts/tongui-3b/merged/score.json)
- Audit：[audit.json](../runs/mind2web-tongui/2026-07-28/artifacts/tongui-3b/merged/audit.json)

#### TongUI-7B 说明

- Anchor delta：Element `+2.7320 pp`，Operation F1 `+0.4854 pp`，Step SR `+2.2095 pp`。
- 2,079/2,080 响应可解析；唯一失败输出非法位置 `[_from]`，严格计零且不修复。
- 输出计数：`CLICK=1,779`、`TYPE=248`、`SELECT=52`。
- 四个 shard 各 520 行；ordered merge、scorer 字节级复算和 2,080-identity audit 均通过。
- 报告：[TongUI-7B report](../runs/mind2web-tongui/2026-07-28/TONGUI_7B_REPORT.md)
- Score：[score.json](../runs/mind2web-tongui/2026-07-28/artifacts/tongui-7b/merged/score.json)
- Audit：[audit.json](../runs/mind2web-tongui/2026-07-28/artifacts/tongui-7b/merged/audit.json)

#### Qwen2.5-VL-3B 说明

- Anchor delta：Element `-0.4552 pp`，Operation F1 `-0.0265 pp`，Step SR `+0.3949 pp`。
- Parse 1,621/2,080；严格支持的 uppercase action 792/2,080。
- 790 条 lowercase alias、890 条超出 0-1 的坐标均不做修复。
- TongUI-3B 相对同架构 base 增益：Element `+54.2418 pp`、Operation F1 `+74.7302 pp`、Step SR `+50.5874 pp`。
- 报告：[Qwen base report](../runs/mind2web-tongui/2026-07-28/QWEN_BASE_REPORT.md)
- Score：[score.json](../runs/mind2web-tongui/2026-07-28/artifacts/qwen-3b/merged/score.json)
- Audit：[audit.json](../runs/mind2web-tongui/2026-07-28/artifacts/qwen-3b/merged/audit.json)

#### Qwen2.5-VL-7B 说明

- Anchor delta：Element `+1.0573 pp`，Operation F1 `-1.1435 pp`，Step SR `+0.8983 pp`。
- Parse 1,845/2,080；严格支持的 uppercase action 1,844/2,080。
- 260 条解析坐标超出 0-1，不做修复；235 条不可解析。
- TongUI-7B 相对同架构 base 增益：Element `+53.5747 pp`、Operation F1 `+17.5289 pp`、Step SR `+49.7111 pp`。
- 报告：[Qwen base report](../runs/mind2web-tongui/2026-07-28/QWEN_BASE_REPORT.md)
- Score：[score.json](../runs/mind2web-tongui/2026-07-28/artifacts/qwen-7b/merged/score.json)
- Audit：[audit.json](../runs/mind2web-tongui/2026-07-28/artifacts/qwen-7b/merged/audit.json)

## 4. 当前未完成运行

### 4.1 Qwen2.5-VL Base Mind2Web

状态：`QWEN_3B_7B_COMPLETE`。

- Qwen2.5-VL-3B revision `66285546d2b821cf421d4f5eb2576359d3770cd3` 已下载并通过 hash/index 校验。
- Qwen2.5-VL-7B revision `cc594898137f460bfe9f0759e9844b3ce807cfb5` 已下载并通过 hash/index 校验。
- 两者复用 TongUI 的 exact data、`v2/vtvt` prompt、greedy generation 和 independent scorer。
- Qwen2.5-VL-3B 完成 2,080 行并 audit PASS：`2.0448 / 14.4735 / 0.7949`。
- Qwen2.5-VL-7B 完成 2,080 行并 audit PASS：`7.2573 / 71.6565 / 5.8983`。

## 5. 精确复现阻断

### AndroidControl OS-Atlas-7B Table 5

状态：`BLOCKED_UNPUBLISHED_CHECKPOINT`。

- 目标：High zero-shot OOD，Type/Grounding/SR `57.44 / 54.90 / 29.83`。
- 7,708 行 `ac_idx` 已固定，但论文 Table 5 action checkpoint 未公开。
- `OS-Atlas-Base-7B` 是 grounding model；`OS-Atlas-Pro-7B` 是 all-dataset successor，二者都不是目标 checkpoint。
- 精确 Gate 2 只有在获得 Table 5 action checkpoint 及完整转换/推理 contract 后才能恢复。
- 阻断报告：[GATE_2_PREFLIGHT_REPORT.md](../runs/androidcontrol/2026-07-27/GATE_2_PREFLIGHT_REPORT.md)

## 6. 全部 baseline 矩阵

这里的“待跑”表示尚无当前受控 pipeline 结果；不能用仓库中的其他数据切分、Pass@K、task-success 或历史实验数字代替。

### 6.1 Mind2Web

| Baseline | Cross-Task paper anchor（Elem / Op F1 / Step） | 当前状态 | 下一动作 |
| --- | --- | --- | --- |
| SeeClick-9.6B | 28.3 / 87.0 / 25.5 | **完成，PASS** | 保持冻结 |
| ShowUI-2B ZS | 21.4 / 85.2 / 18.6 | **完成公共 checkpoint 结果** | 保持冻结 |
| ShowUI-2B downstream FT | 39.9 / 88.6 / 37.2 | **BLOCKED：downstream state dict 未公开** | 公共 `showlab/ShowUI-2B` 是已完成的通用 ZS checkpoint，不是该行 |
| TongUI-3B (1M) | 53.4 / 89.0 / 48.8 | **完成受控结果：56.29 / 89.20 / 51.38** | 保持冻结 |
| TongUI-7B (1M) | 58.1 / 88.7 / 53.4 | **完成受控结果：60.83 / 89.19 / 55.61** | 保持冻结 |
| TongUI-32B | 57.2 / 88.1 / 52.4 | **完成受控结果：58.92 / 89.74 / 54.33** | 2,080 rows、TP=4 native LoRA、audit PASS；三项 delta 约 1.6-1.9 pp |
| Qwen2.5-VL-3B | 2.5 / 14.5 / 0.4 | **完成，PASS：2.04 / 14.47 / 0.79** | 保持冻结 |
| Qwen2.5-VL-7B | 6.2 / 72.8 / 5.0 | **完成受控结果：7.26 / 71.66 / 5.90** | audit PASS；两项 delta 略超 1 pp |
| Qwen2.5-VL-3B-ShowUI | 43.2 / 88.7 / 39.7 | **BLOCKED：无官方公开 model ID/权重** | showlab、TongUI 官方 release 均无对应 checkpoint |
| UI-TARS-2B | 62.3 / 90.0 / 56.3 | **完成公共 checkpoint 受控迁移：28.30 / 65.19 / 22.66** | audit PASS；论文 Mind2Web contract 未公开 |
| UI-TARS-7B | 73.1 / 92.2 / 67.1 | **完成公共 checkpoint 受控迁移：44.19 / 84.75 / 37.87** | audit PASS；论文 Mind2Web contract 未公开 |
| UI-TARS-72B | 74.7 / 92.5 / 68.6 | **完成公共 checkpoint 受控迁移：52.39 / 80.31 / 43.41** | TP=4、2,080 rows、273GB/64 shards、audit PASS |
| CogAgent | 22.4 / 53.0 / 17.6 | **完成公共 checkpoint 受控迁移：58.93 / 85.49 / 56.04** | audit PASS；论文 split contract 未公开，不是 anchor 复现 |
| AgentTrek Qwen2-VL + AT | 45.5 / 84.9 / 40.9 | **BLOCKED：视觉 checkpoint 未公开** | 公开 32B 是 Qwen2.5 文本 WebArena agent，不可替代 |
| OS-Atlas-4B/7B | - | **不属于论文 Mind2Web 评测范围** | Mind2Web 是训练数据；论文评测为 GUI-Act/OmniAct/AndroidControl/GUI-Odyssey |
| OmniParser | 42.4 / 87.6 / 39.4 | **BLOCKED：无闭源 API 凭证/日期版本 contract** | detector + GPT-4V 组合需额外固定与成本授权 |
| MindAct | 55.1 / 75.7 / 52.0 | **完成：54.37 / 74.32 / 50.00** | 独立 2,094-action HTML lane；252 episodes、官方 top-50 tournament、audit PASS |
| SeeAct | 待固定同 split anchor | **BLOCKED：闭源 API/版本/隐私 contract 缺失** | 当前环境无 API credentials |
| GPT-4o + UGround/Aria-UI/Aguvis | 待固定论文行 | **BLOCKED：闭源 API/版本/预算 contract 缺失** | 当前环境无 API credentials |

### 6.2 AndroidControl

| Baseline | Low | High | 当前状态 | 备注 |
| --- | --- | --- | --- | --- |
| OS-Atlas-7B Table 5 | - | 57.44 / 54.90 / 29.83 | **BLOCKED** | exact action checkpoint 未公开 |
| OS-Atlas-Pro-7B | **93.47 / 86.76 / 83.96** | **86.31 / 77.95 / 71.33** | **完成** | successor，不是 OOD Table 5 |
| SeeClick | 93.0 / 73.4 / 75.0 | 82.9 / 62.9 / 59.1 | preflight blocked | 仅公开 general/AITW checkpoint；缺 AndroidControl conversion |
| UI-TARS-2B | **79.81 / 80.77 / 63.53** | **69.65 / 64.47 / 48.14** | **完成公共 checkpoint 受控迁移，Low/High audit PASS** | 论文 anchor 为 98.1/87.3/89.3 与 81.2/78.4/68.9；公开 split contract 未发布 |
| UI-TARS-7B | **89.53 / 85.42 / 73.85** | **79.58 / 73.46 / 61.01** | **完成公共 checkpoint 受控迁移，Low/High audit PASS** | 论文 anchor 为 98.0/89.3/90.8 与 83.7/80.5/72.5；公开 split contract 未发布 |
| Qwen2.5-VL-3B/7B | **0.00 / 0.00 / 0.00** (Qwen3/7) | **0.00 / 0.00 / 0.00** (Qwen3/7) | Qwen3 [Low](../runs/androidcontrol-pro/2026-07-27/QWEN3_LOW_REPORT.md)/[High](../runs/androidcontrol-pro/2026-07-27/QWEN3_HIGH_REPORT.md)、Qwen7 [Low](../runs/androidcontrol-pro/2026-07-27/QWEN7_LOW_REPORT.md)/[High](../runs/androidcontrol-pro/2026-07-27/QWEN7_HIGH_REPORT.md) 完成 | strict parser；flexible diagnostic 不混报 |
| TongUI-3B/7B | - | - | **不属于论文 AndroidControl 评测；完整 transfer blocked** | AITW checkpoint 缺 `LONG_PRESS/OPEN_APP/WAIT`；`mobile_use` utility 未接入模型 |
| UI-AGILE-3B/7B | **89.48/87.58/79.09**（3B 完成）；98.0/92.2/91.0（7B anchor） | 88.8/85.8/78.7；91.0/87.2/81.4 | **UI-AGILE-3B Low 完成并 audit PASS；其余 lane 正在执行** | 3B Low 论文 anchor 98.1/91.8/90.8；官方 parquet 7,708 steps |
| GUI-R1-3B/7B | 96.9/89.9/87.3；97.5/91.7/89.7 | 58.0/56.2/46.6；71.6/65.6/51.7 | **公开 checkpoint 已固定，Low/High 待统一复测** | `ritzzai/GUI-R1@e74baccc...`，官方 `gui_r1` prompt |
| UI-R1-E-3B | 97.7/91.2/89.4 | 83.5/78.9/69.8 | **公开 checkpoint 已固定，Low/High 待统一复测** | `LZXzju/Qwen2.5-VL-3B-UI-R1-E@91c3e5f...` |
| UI-R1-3B v1 original lane | **94.3 / 82.6 / 88.5** | - | **7,868-step selected Low 数据/模型完整，full 待跑** | 数字为 Type / click Grounding / 两者算术平均，绝不是 Step SR；与 7,708-step lane 分开 |
| InfiGUI-R1-3B | **95.97 / 93.87 / 92.09** | **82.75 / 74.44 / 71.28** | [**完成，PASS**](../runs/androidcontrol-infigui/2026-07-28/FINAL_REPORT.md) | 官方 8,444-step split；Low/High `--require-complete` 独立审计均 PASS；论文 anchor 为 96.0/93.2/92.1 与 82.7/74.4/71.1 |

AndroidControl 7,708-step 与 InfiGUI 8,444-step 表中数字顺序为
`Type / Grounding / Step SR`。原始 UI-R1 的 `94.3 / 82.6 / 88.5` 例外：
发布 evaluator 定义为 `Type / click Grounding / 两者算术平均`，没有 Step SR。

## 7. 不可混用的历史结果

以下结果不进入本报告主表：

- `evaluation/results_comparison.md` 及类似历史 AndroidControl task-success/Pass@K 结果。
- 1,543-task episode success 与本报告 7,708-step Type/Grounding/SR。
- UI-S1、Qwen 或其他模型在 385/1,000/1,543 episode 子集上的结果。
- ShowUI downstream-finetuned paper row与公共通用 `ShowUI-2B` checkpoint。
- OS-Atlas-Pro successor 与未公开的 OS-Atlas Table 5 zero-shot OOD checkpoint。
- TongUI 官方 vLLM 脚本的 best-of-3 + GT action repair 结果。

## 8. 推荐执行顺序

1. 完成 UI-AGILE-3B/7B、UI-R1-E-3B、GUI-R1-3B/7B 的 10 条官方 7,708-step Low/High lane。
2. 完成原始 UI-R1-3B v1 的独立 7,868-step selected-Low Type/Grounding/Average lane。
3. 其余行等待外部条件：未发布 checkpoint、闭源 API 凭证/日期版本/预算，或官方 action/evaluator contract。

UI-TARS checkpoint 边界：原论文说明离线 benchmark 主表使用 annealing
阶段的 SFT 模型。公开原始 checkpoint 已固定为
`ByteDance-Seed/UI-TARS-2B-SFT@f366a1db...` 和
`ByteDance-Seed/UI-TARS-7B-SFT@3434901a...`；两者 Mind2Web 与
AndroidControl Low/High controlled-transfer 均已完成并通过审计。
`UI-TARS-72B-SFT@8e7a031...` 的 64 shards/273GB 也已通过完整 SHA/index
校验，TP=4 完整运行与审计已通过。不得用 7B-DPO revision
`727b0df3...` 或后续 `UI-TARS-1.5-7B` 替代原论文行；进入推理前还需固定
其 prompt、0-1000 坐标转换和 action parser。官方仓库未发布 split-specific
Mind2Web evaluator，因此所有公共 checkpoint 结果均与论文 anchor 分开标为
受控迁移。

## 9. 关键 artifact 索引

| 内容 | 路径 |
| --- | --- |
| SeeClick final report | `runs/mind2web/2026-07-27/GATE_1_REPORT.md` |
| SeeClick audit | `runs/mind2web/2026-07-27/artifacts/gate1_audit.json` |
| ShowUI final report | `runs/mind2web-showui/2026-07-28/FINAL_REPORT.md` |
| ShowUI merged score/audit | `runs/mind2web-showui/2026-07-28/artifacts/merged/` |
| TongUI-3B final report | `runs/mind2web-tongui/2026-07-28/TONGUI_3B_REPORT.md` |
| TongUI-3B merged score/audit | `runs/mind2web-tongui/2026-07-28/artifacts/tongui-3b/merged/` |
| TongUI-7B final report | `runs/mind2web-tongui/2026-07-28/TONGUI_7B_REPORT.md` |
| TongUI-7B merged score/audit | `runs/mind2web-tongui/2026-07-28/artifacts/tongui-7b/merged/` |
| TongUI-32B final report | `runs/mind2web-tongui/2026-07-28/TONGUI_32B_REPORT.md` |
| TongUI-32B score/audit | `runs/mind2web-tongui/2026-07-28/artifacts/tongui-32b/full/` |
| Qwen base report | `runs/mind2web-tongui/2026-07-28/QWEN_BASE_REPORT.md` |
| Qwen2.5-VL-3B merged score/audit | `runs/mind2web-tongui/2026-07-28/artifacts/qwen-3b/merged/` |
| TongUI run tree | `runs/mind2web-tongui/2026-07-28/` |
| UI-TARS checkpoint/evaluator preflight | `runs/mind2web-uitars/2026-07-28/PREFLIGHT.md` |
| UI-TARS public-transfer final report | `runs/mind2web-uitars/2026-07-28/FINAL_REPORT.md` |
| UI-TARS 2B/7B merged score/audit | `runs/mind2web-uitars/2026-07-28/artifacts/{2b,7b}/merged/` |
| UI-TARS-72B checkpoint manifest | `runs/mind2web-uitars/2026-07-28/uitars72_checkpoint_manifest.json` |
| UI-TARS-72B score/audit | `runs/mind2web-uitars/2026-07-28/artifacts/72b/full/` |
| UI-TARS AndroidControl preflight | `runs/androidcontrol-uitars/2026-07-29/PREFLIGHT.md` |
| UI-TARS-2B AndroidControl score/audit | `runs/androidcontrol-uitars/2026-07-29/artifacts/2b-{low,high}/merged/` |
| UI-TARS AndroidControl final report | `runs/androidcontrol-uitars/2026-07-29/FINAL_REPORT.md` |
| UI-TARS-7B AndroidControl score/audit | `runs/androidcontrol-uitars/2026-07-29/artifacts/7b-{low,high}/merged/` |
| CogAgent public-transfer final report | `runs/mind2web-cogagent/2026-07-28/FINAL_REPORT.md` |
| CogAgent merged score/audit | `runs/mind2web-cogagent/2026-07-28/artifacts/merged/` |
| AgentTrek visual-checkpoint blocker | `runs/mind2web-agenttrek/2026-07-29/PREFLIGHT.md` |
| OS-Atlas Mind2Web scope check | `runs/mind2web-osatlas/2026-07-29/PREFLIGHT.md` |
| MindAct HTML-lane preflight | `runs/mindact/2026-07-29/PREFLIGHT.md` |
| MindAct data/model manifest | `runs/mindact/2026-07-29/artifact_manifest.json` |
| MindAct merged result/audit | `runs/mindact/2026-07-29/artifacts/full/` |
| Closed API baseline blockers | `runs/mind2web-closed-api/2026-07-29/PREFLIGHT.md` |
| Unreleased ShowUI checkpoint blockers | `runs/mind2web-showui-unreleased/2026-07-29/PREFLIGHT.md` |
| UI-AGILE/UI-R1-E/GUI-R1 unified runner | `runs/androidcontrol-rft/2026-07-29/` |
| RFT checkpoint/data/source manifest | `runs/androidcontrol-rft/2026-07-29/artifact_manifest.json` |
| Original UI-R1 selected-Low lane | `runs/androidcontrol-rft/2026-07-29/original-ui-r1/` |
| Official AndroidControl GCS manifest | `runs/androidcontrol-rft/2026-07-29/data/official-gcs/official_gcs_manifest.json` |
| TongUI AndroidControl scope/action check | `runs/androidcontrol-tongui/2026-07-29/PREFLIGHT.md` |
| AndroidControl SeeClick preflight | `runs/androidcontrol-seeclick/2026-07-28/PREFLIGHT.md` |
| AndroidControl exact blocker | `runs/androidcontrol/2026-07-27/GATE_2_PREFLIGHT_REPORT.md` |
| OS-Atlas-Pro High report | `runs/androidcontrol-pro/2026-07-27/FINAL_REPORT.md` |
| OS-Atlas-Pro Low report | `runs/androidcontrol-pro/2026-07-27/LOW_REPORT.md` |
| Low/High comparison | `runs/androidcontrol-pro/2026-07-27/LOW_HIGH_COMPARISON.md` |
| Qwen3 AndroidControl-High report | `runs/androidcontrol-pro/2026-07-27/QWEN3_HIGH_REPORT.md` |
| Qwen3 AndroidControl-High score/audit | `runs/androidcontrol-pro/2026-07-27/artifacts/qwen-3b-high/vllm-merged/` |
| Qwen3 AndroidControl-Low report | `runs/androidcontrol-pro/2026-07-27/QWEN3_LOW_REPORT.md` |
| Qwen3 AndroidControl-Low score/audit | `runs/androidcontrol-pro/2026-07-27/artifacts/qwen-3b-low/vllm-merged/` |
| Qwen7 AndroidControl-High report | `runs/androidcontrol-pro/2026-07-27/QWEN7_HIGH_REPORT.md` |
| Qwen7 AndroidControl-High score/audit | `runs/androidcontrol-pro/2026-07-27/artifacts/qwen-7b-high/vllm-merged/` |
| Qwen7 AndroidControl-Low report | `runs/androidcontrol-pro/2026-07-27/QWEN7_LOW_REPORT.md` |
| Qwen7 AndroidControl-Low score/audit | `runs/androidcontrol-pro/2026-07-27/artifacts/qwen-7b-low/vllm-merged/` |
| InfiGUI-R1 AndroidControl preflight | `runs/androidcontrol-infigui/2026-07-28/PREFLIGHT.md` |
| InfiGUI-R1 independent scorer/auditor | `runs/androidcontrol-infigui/2026-07-28/audit.py` |
| InfiGUI-R1 AndroidControl final report | `runs/androidcontrol-infigui/2026-07-28/FINAL_REPORT.md` |
| InfiGUI-R1 Low/High final audits | `runs/androidcontrol-infigui/2026-07-28/artifacts/{low,high}/final_audit.json` |
