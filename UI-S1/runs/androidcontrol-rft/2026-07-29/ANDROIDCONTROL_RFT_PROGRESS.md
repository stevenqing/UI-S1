# AndroidControl RFT Baseline Progress

更新时间：2026-07-29

## 当前结果

统一复测使用 `KDEGroup/UI-AGILE-Data@6b4f69d9...` 的 Low/High parquet，各 7,708 steps。指标顺序为 Type Accuracy / Grounding Accuracy / Step Success Rate。

| Model | Setting | Current result | Paper anchor | Delta | Coverage | Audit |
| --- | --- | --- | --- | --- | ---: | --- |
| UI-AGILE-3B | Low | **89.4785 / 87.5787 / 79.0867** | 98.1 / 91.8 / 90.8 | -8.6215 / -4.2213 / -11.7133 pp | 7,708 / 7,708 | PASS |
| UI-AGILE-3B | High | **76.9590 / 55.8523 / 58.6015** | 88.8 / 85.8 / 78.7 | -11.8410 / -29.9477 / -20.0985 pp | 7,708 / 7,708 | PASS |
| UI-AGILE-7B | Low | **87.6492 / 87.9479 / 77.5558** | 98.0 / 92.2 / 91.0 | -10.3508 / -4.2521 / -13.4442 pp | 7,708 / 7,708 | PASS |
| UI-AGILE-7B | High | **79.9689 / 61.9761 / 60.5345** | 91.0 / 87.2 / 81.4 | -11.0311 / -25.2239 / -20.8655 pp | 7,708 / 7,708 | PASS |
| UI-R1-E-3B | Low | **86.6762 / 78.5451 / 48.5080** | 97.7 / 91.2 / 89.4 | -11.0238 / -12.6549 / -40.8920 pp | 7,708 / 7,708 | PASS |
| UI-R1-E-3B | High | **66.6969 / 38.9359 / 22.8853** | 83.5 / 78.9 / 69.8 | -16.8031 / -39.9641 / -46.9147 pp | 7,708 / 7,708 | PASS |
| GUI-R1-3B | Low | **84.4577 / 80.0434 / 56.8889** | 96.9 / 89.9 / 87.3 | -12.4423 / -9.8566 / -30.4111 pp | 7,708 / 7,708 | PASS |

结果来自固定 checkpoint：`KDEGroup/UI-AGILE-3B@84c28b06...`、`KDEGroup/UI-AGILE@de013669...`、`LZXzju/Qwen2.5-VL-3B-UI-R1-E@91c3e5f...` 和 `ritzzai/GUI-R1@e74baccc...`。预测按四个 GPU shard 生成后严格按全局 index 合并；独立 audit 从官方 parquet、原始模型输出和固定 parser 逐行重算指标。未使用 GT output repair。

## 正在运行

严格串行队列继续执行：

1. GUI-R1-3B High
2. GUI-R1-7B Low / High
3. 原始 UI-R1-3B v1 selected-Low 7,868-step 独立 lane

原始 UI-R1 lane 不与统一 7,708-step lane 混报。其发布指标 `94.3 / 82.6 / 88.5` 表示 Type / click Grounding / 两者算术平均，不是 Step SR。发布代码还存在可复核的坐标矛盾：实际 slow-processor grid 为 672x1484，但 `eval_ac.py` 按 644x1484 缩放；本地 runner 同时记录两者并按发布 evaluator 计分。

## Provenance

- Unified source commit: `KDEGroup/UI-AGILE@3a397b078d6c14338f0646070212f8c3eb837881`
- Low parquet SHA256: `ffb8e19f5091c339aea4060e062cc47405f57da3f35c22699a82390d5769cf47`
- High parquet SHA256: `ec70c99046aa4fb1557c61bf2c5d1266f87d4b9dc879128b2de20bf0aca7c72f`
- UI-AGILE-3B Low predictions SHA256: `73a515adb79b6182f414f24ab2a6e225495abf27b9eaf4531a1ba2a6e203b5a0`
- UI-AGILE-3B Low score SHA256: `b829c64c11dfa0ebcaac539819ec9f97f28ef8ba52e69060318cc78d5fc783f1`
- UI-AGILE-3B High predictions SHA256: `9e96878fc3638ac49039a400ad15177bb5c318479247719bdaa2cd7f26b8d812`
- UI-AGILE-3B High score SHA256: `29cc90b4ecf09bc47569e0ff0cb16030ae6c8aee5632cffc409b472630432e6e`
- UI-AGILE-7B Low/High score SHA256: `e64ce51215be564f59d922a081a9aa409d8bb49ff2005bd34fb7fc82c7cae08f` / `d5999f117e4530945b77e69af6d90fd88e6bcaa025e1830e880ee7989a5c358b`
- UI-R1-E-3B Low/High score SHA256: `61e93dc644e98aeb2e519a45a382f751942659521f9a693447e60ea69858247a` / `78fec3850aebdf4d3ecdac0836e2f82b4329978de2b4df727c8e3a17097da476`
- GUI-R1-3B Low score SHA256: `9d4368656c70b1f811ec60cd4964cc0c14b42ca157031964c8286c05680f744e`
- Official AndroidControl source: 20 TFRecord shards plus two split files, 49,930,349,992 bytes, all public GCS MD5 checks PASS
- Checkpoint/data/source manifest: `artifact_manifest.json`
- Completed score/audit: `artifacts/ui-agile-3b/low/{score,audit}.json`
