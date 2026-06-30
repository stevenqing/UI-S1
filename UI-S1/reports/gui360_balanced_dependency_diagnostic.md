# GUI-360-balanced Cross-Step Dependency Diagnostic

日期：2026-06-30

## 结论

`gui360-balanced` 没有可支撑 multi-turn / history training 的大规模 cross-step dependency battlefield。

最终判定：`NO_BATTLEFIELD`

原因：OCR-backed train split 的 survivor share 是 `4.94%`，低于预注册 no-battlefield 区间；不用 OCR/a11y 的 data-only upper bound 也只有 `7.35%`，仍达不到 battlefield 门槛。

## 为什么不能只靠 OCR

OCR 是离线 referee 的近似工具，不是完美屏幕可见性判定。它可能：

- 漏掉画面上的小字、模糊字、遮挡文本。
- 把格式相近的文字识别错。
- 对长文本只能提供不完整覆盖。

所以单个候选是否满足“消费时屏幕不可见”，OCR-backed 结果仍然有局部不确定性。

但这个任务的 gate 问的不是“每个 survivor 都绝对真实”，而是“是否存在大规模 battlefield”。为此我们额外做了 data-only upper bound。

## Data-only Upper Bound

data-only upper bound 完全不使用 OCR/a11y availability exclusion。

含义：

- 不排除任何“可能屏幕可见”的候选。
- 只使用 `goal + action labels + step sequence + value semantics + routine entropy + distance`。
- survivor 是偏高估的上界，不是 confirmed true dependency。

结果：

| Split | Candidate Total | Survivors | Share | Verdict |
| --- | ---: | ---: | ---: | --- |
| train data-only upper | 1578 | 116 | 7.35% | NO_BATTLEFIELD |
| test data-only upper | 541 | 19 | 3.51% | NO_BATTLEFIELD |

对应文件：

- `reports/dependency_verdict_balanced_train_data_only_upper.json`
- `reports/dependency_verdict_balanced_test_data_only_upper.json`

解释：即使完全不相信 OCR，最大可能 battlefield 规模也没有接近 `15-20%` 的 BATTLEFIELD 门槛。

## OCR-backed Results

OCR cache 是从 `datasets/gui360-balanced/data/*.parquet` 的 screenshot bytes 直接构建的，不依赖原始图片路径存在。

uv 环境：

```bash
/tmp/gui360-depdiag-uv
```

OCR cache：

- `reports/ocr_cache_gui360_balanced_train.jsonl`
- `reports/ocr_cache_gui360_balanced_test.jsonl`

train OCR target 覆盖：

```text
dependency target steps: 994
covered: 994/994
OCR errors: 0
```

OCR-backed verdict：

| Split | Candidate Total | Survivors | Share | Verdict |
| --- | ---: | ---: | ---: | --- |
| train OCR-backed | 1578 | 78 | 4.94% | NO_BATTLEFIELD |
| test OCR-backed | 541 | 9 | 1.66% | NO_BATTLEFIELD |

对应文件：

- `reports/dependency_verdict.json`
- `reports/dependency_verdict_balanced_test.json`

## Train Bucket Accounting

OCR-backed train split bucket 分布：

```text
given          939
noise          193
onscreen_ocr    24
resurfaced      44
routine        251
adjacent        49
survivor        78
```

合计：`1578`

Q2 距离分布：

```text
distance_ge3_n = 78
distance_ge3_share = 1.0
```

distance histogram：

```text
3: 17
4: 9
5: 10
6: 15
7: 7
8: 7
9: 3
10: 1
11: 1
12: 1
13: 1
14: 2
15: 1
17: 1
21: 1
23: 1
```

## Test Bucket Accounting

OCR-backed test split bucket 分布：

```text
given          172
noise           15
onscreen_ocr     4
resurfaced       7
routine        325
adjacent         9
survivor         9
```

合计：`541`

Q2 距离分布：

```text
distance_ge3_n = 9
distance_ge3_share = 1.0
```

## 修正过的诊断问题

这次仔细检查发现并修正了两个会影响结论的 bug / pseudo-class 问题。

### 1. Missing controls 不能判 forced

`gui360-balanced` 没有 a11y/control 信息。之前代码把 `controls=[]` 当成合法动作空间大小为 `0`，导致大量候选被错误归为 `forced`。

修正后：

- `controls=[]` 表示动作空间未知。
- 未知动作空间不能判 `forced`。

修改文件：

- `gui360_long_horizon/data/pseudo_consumption.py`

### 2. 快捷键命令不是 carried value

很多重复值其实是 `{ENTER}`、`{VK_CONTROL}a`、`+{RIGHT}`、`^a{BACKSPACE}` 这类键盘命令。它们不是语义 carried value，不满足 Layer 0 的 first-principles 定义。

修正后：

- 纯快捷键命令归入 `given`。
- `^a{BACKSPACE}正文`、`{VK_CONTROL}a{DEL}正文` 只保留真正输入的正文。

修改文件：

- `gui360_long_horizon/data/carried_value.py`

## Detector Controls

做过的正负控：

| Case | Expected | Result |
| --- | --- | --- |
| 前面输入、后面离屏再用 | survivor | pass |
| 消费屏 OCR 可见 | onscreen_ocr | pass |
| 中间屏重新出现 | resurfaced | pass |
| 纯快捷键 `{ENTER}` | given | pass |
| 缺 OCR cache | onscreen_ocr conservative exclusion | pass |

这说明诊断器有基本鉴别力：能检出人工真依赖，也能排除主要伪依赖。

## OCR Caveat Scan

对 OCR-backed train survivors 做了 token overlap 检查，发现少数 survivor 与消费屏 OCR 有较高 token overlap，说明 exact OCR match 仍可能漏掉一些“部分可见”长文本。

这不会推翻结论，原因是：

- OCR-backed survivor share 已低至 `4.94%`。
- data-only upper bound 不使用 OCR，也只有 `7.35%`。
- OCR 漏检只会让 OCR-backed survivor 偏高；即便按上界算，也不是 battlefield。

## Commands

OCR-backed train gate：

```bash
PYTHONPATH=$PWD /tmp/gui360-depdiag-uv/bin/python -m gui360_long_horizon.analysis.dependency_diag \
  --balanced-data-dir datasets/gui360-balanced/data \
  --split train \
  --ocr-cache reports/ocr_cache_gui360_balanced_train.jsonl \
  --out reports/dependency_verdict.json
```

Data-only upper bound：

```bash
PYTHONPATH=$PWD /tmp/gui360-depdiag-uv/bin/python - <<'PY'
import json
from pathlib import Path
from gui360_long_horizon.analysis.dependency_diag import load_balanced_parquet, run_dependency_diagnostic, write_dependency_verdict
from gui360_long_horizon.data.availability import OcrReferee

for split, out in [
    ('train', 'reports/dependency_verdict_balanced_train_data_only_upper.json'),
    ('test', 'reports/dependency_verdict_balanced_test_data_only_upper.json'),
]:
    episodes = load_balanced_parquet('datasets/gui360-balanced/data', split)
    payload = run_dependency_diagnostic(episodes, ocr=OcrReferee(cache={}, missing_is_available=False))
    payload['analysis_note'] = 'Data-only upper bound: no OCR/a11y availability exclusion; missing OCR is treated as not visible, so survivors are an upper bound, not confirmed true dependencies.'
    write_dependency_verdict(payload, out)
PY
```

## Validation

测试命令：

```bash
python -m pytest tests/gui360_long_horizon -q
```

结果：

```text
68 passed
```

## Final Interpretation

严格表述：

```text
GUI-360-balanced does not contain a large-scale cross-step dependency battlefield under this gate.
```

中文表述：

```text
GUI-360-balanced 没有可支撑 multi-turn/history training 的大规模跨步骤依赖战场。
```

限制：

- 纯数据不能证明单个 survivor 一定是“消费时不可见”。
- OCR-backed 单候选标签仍可能有局部误差。
- 但 data-only upper bound 已经低于 battlefield 门槛，所以 OCR 不精确不会改变最终 gate 结论。

执行建议：

```text
STOP. Do not proceed to multi-turn/history training for GUI-360-balanced on this basis.
```