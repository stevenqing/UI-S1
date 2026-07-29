# AndroidControl RFT Baseline Progress

更新时间：2026-07-29

## 当前结果

统一复测使用 `KDEGroup/UI-AGILE-Data@6b4f69d9...` 的 Low/High parquet，各 7,708 steps。指标顺序为 Type Accuracy / Grounding Accuracy / Step Success Rate。

| Model | Setting | Current result | Paper anchor | Delta | Coverage | Audit |
| --- | --- | --- | --- | --- | ---: | --- |
| UI-AGILE-3B | Low | **89.4785 / 87.5787 / 79.0867** | 98.1 / 91.8 / 90.8 | -8.6215 / -4.2213 / -11.7133 pp | 7,708 / 7,708 | PASS |
| UI-AGILE-3B | High | **76.9590 / 55.8523 / 58.6015** | 88.8 / 85.8 / 78.7 | -11.8410 / -29.9477 / -20.0985 pp | 7,708 / 7,708 | PASS |

该结果来自固定 checkpoint `KDEGroup/UI-AGILE-3B@84c28b06a7bda29a741139d64e227d176c0fb1c0`。预测按四个 GPU shard 生成后严格按全局 index 合并；独立 audit 从官方 parquet、原始模型输出和固定 parser 逐行重算指标。未使用 GT output repair。

## 正在运行

严格串行队列继续执行：

1. UI-AGILE-7B Low / High
2. UI-R1-E-3B Low / High
3. GUI-R1-3B Low / High
4. GUI-R1-7B Low / High
5. 原始 UI-R1-3B v1 selected-Low 7,868-step 独立 lane

原始 UI-R1 lane 不与统一 7,708-step lane 混报。其发布指标 `94.3 / 82.6 / 88.5` 表示 Type / click Grounding / 两者算术平均，不是 Step SR。发布代码还存在可复核的坐标矛盾：实际 slow-processor grid 为 672x1484，但 `eval_ac.py` 按 644x1484 缩放；本地 runner 同时记录两者并按发布 evaluator 计分。

## Provenance

- Unified source commit: `KDEGroup/UI-AGILE@3a397b078d6c14338f0646070212f8c3eb837881`
- Low parquet SHA256: `ffb8e19f5091c339aea4060e062cc47405f57da3f35c22699a82390d5769cf47`
- High parquet SHA256: `ec70c99046aa4fb1557c61bf2c5d1266f87d4b9dc879128b2de20bf0aca7c72f`
- UI-AGILE-3B Low predictions SHA256: `73a515adb79b6182f414f24ab2a6e225495abf27b9eaf4531a1ba2a6e203b5a0`
- UI-AGILE-3B Low score SHA256: `b829c64c11dfa0ebcaac539819ec9f97f28ef8ba52e69060318cc78d5fc783f1`
- UI-AGILE-3B High predictions SHA256: `9e96878fc3638ac49039a400ad15177bb5c318479247719bdaa2cd7f26b8d812`
- UI-AGILE-3B High score SHA256: `29cc90b4ecf09bc47569e0ff0cb16030ae6c8aee5632cffc409b472630432e6e`
- Official AndroidControl source: 20 TFRecord shards plus two split files, 49,930,349,992 bytes, all public GCS MD5 checks PASS
- Checkpoint/data/source manifest: `artifact_manifest.json`
- Completed score/audit: `artifacts/ui-agile-3b/low/{score,audit}.json`
