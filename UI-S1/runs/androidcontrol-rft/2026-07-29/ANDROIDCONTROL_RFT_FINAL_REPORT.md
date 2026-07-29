# AndroidControl RFT Baseline Results

更新时间：2026-07-29

## 1. 统一 7,708-step Low/High lane

指标顺序为 Type Accuracy / Grounding Accuracy / Step Success Rate。所有结果使用固定官方 parquet、官方 prompt/parser/evaluator、temperature 0、max tokens 256，四卡独立分片、有序合并和独立全量审计；不做 GT output repair。

| Model | Setting | Type | Grounding | Step SR | Paper anchor | Audit |
| --- | --- | ---: | ---: | ---: | --- | --- |
| UI-AGILE-3B | Low | **89.4785** | **87.5787** | **79.0867** | 98.1 / 91.8 / 90.8 | PASS |
| UI-AGILE-3B | High | **76.9590** | **55.8523** | **58.6015** | 88.8 / 85.8 / 78.7 | PASS |
| UI-AGILE-7B | Low | **87.6492** | **87.9479** | **77.5558** | 98.0 / 92.2 / 91.0 | PASS |
| UI-AGILE-7B | High | **79.9689** | **61.9761** | **60.5345** | 91.0 / 87.2 / 81.4 | PASS |
| UI-R1-E-3B | Low | **86.6762** | **78.5451** | **48.5080** | 97.7 / 91.2 / 89.4 | PASS |
| UI-R1-E-3B | High | **66.6969** | **38.9359** | **22.8853** | 83.5 / 78.9 / 69.8 | PASS |
| GUI-R1-3B | Low | **84.4577** | **80.0434** | **56.8889** | 96.9 / 89.9 / 87.3 | PASS |
| GUI-R1-3B | High | **64.2190** | **56.9598** | **38.4276** | 58.0 / 56.2 / 46.6 | PASS |
| GUI-R1-7B | Low | **83.6404** | **83.5396** | **58.2122** | 97.5 / 91.7 / 89.7 | PASS |
| GUI-R1-7B | High | **71.3544** | **64.7774** | **44.9663** | 71.6 / 65.6 / 51.7 | PASS |

## 2. 原始 UI-R1-3B v1 selected-Low lane

该 lane 使用发布的 7,868-step `ac_test.json`，与上面的 7,708-step lane 不是同一 split。指标为 Type、click Grounding，以及两者算术平均，不含 Step SR。

| Rows | Episodes | Type | Click Grounding | Reported Average | Paper anchor | Audit |
| ---: | ---: | ---: | ---: | ---: | --- | --- |
| 7868 | 1543 | **94.9161** | **76.8033** | **85.8597** | 94.3 / 82.6 / 88.5 | PASS |

发布代码存在可复核的坐标矛盾：7,744张标准截图的 slow-processor grid 为 672x1484，另外124张为 672x1456 或 700x1400，但 `eval_ac.py` 对全部样本统一按 644x1484 缩放。这里按发布 evaluator 复测并在逐行 provenance 中记录、审计实际 grid，因此该行标为 released-code controlled reproduction，而不是无保留的 strict paper reproduction。

## 3. Provenance

- Unified artifact manifest: `runs/androidcontrol-rft/2026-07-29/artifact_manifest.json`
- Official GCS source manifest: `runs/androidcontrol-rft/2026-07-29/data/official-gcs/official_gcs_manifest.json`
- Original UI-R1 image manifest: `runs/androidcontrol-rft/2026-07-29/original-ui-r1/image_manifest.json`
- Unified checkpoint revisions:
  - `KDEGroup/UI-AGILE-3B@84c28b06a7bda29a741139d64e227d176c0fb1c0`
  - `KDEGroup/UI-AGILE@de01366937b3c921f49ae1abe3b2c4a39b40ce8d`
  - `LZXzju/Qwen2.5-VL-3B-UI-R1-E@91c3e5f213ab3f42931e6398174f470c8500167f`
  - `ritzzai/GUI-R1:GUI-R1-3B@e74baccc4cfa77074e2d53e99a8244ab9fc2ca10`
  - `ritzzai/GUI-R1:GUI-R1-7B@e74baccc4cfa77074e2d53e99a8244ab9fc2ca10`
- Original UI-R1: `LZXzju/Qwen2.5-VL-3B-UI-R1@9cc2fbb7d99ffe90c21f9cd0eb19c45380f8bb0f`
