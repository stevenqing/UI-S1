# Validation Loss上升问题分析与解决

## 问题描述

在GUI-360 SFT训练过程中，观察到validation loss持续上升，而training loss正常下降：

```
step:49  → val/loss: 0.404  (基线)
step:98  → val/loss: 0.517  (+28%)
step:147 → val/loss: 0.552  (+37%)
step:196 → val/loss: 0.570  (+41%)
step:245 → val/loss: 0.556
step:294 → val/loss: 0.567
```

Train loss: 1.64 → 0.13 (正常下降)

## 根本原因

### 1. 训练集和验证集来自不同数据源

| 数据集 | 来源 | 脚本 |
|--------|------|------|
| **训练集** | `processed_data/action_prediction_train_resize/training_data.json` | `prepare_gui360_sft_parquet.py` |
| **验证集** | `test/data/*.jsonl` (原始测试数据) | `prepare_gui360_eval_parquet.py` |

- 训练使用的是GUI-360的**预处理训练数据**
- 验证使用的是GUI-360的**原始测试数据**
- 这两个是完全不同的数据分布！

### 2. 数据格式不一致

**训练数据格式 (预处理后)**:
```json
{
  "id": "sample_id",
  "images": ["path/to/image.png"],
  "conversation": [
    {"from": "human", "value": "<image>\nYou are a helpful assistant..."},
    {"from": "gpt", "value": "<tool_call>{\"function\": \"click\", \"args\": {...}}</tool_call>"}
  ]
}
```

**验证数据格式 (原始测试数据)**:
```json
{
  "execution_id": "excel_1_101",
  "request": "...",
  "step": {
    "action": {"function": "click", "coordinate_x": 39, "coordinate_y": 71}
  }
}
```

`prepare_gui360_eval_parquet.py` 使用了不同的prompt模板和响应格式，导致：
- Prompt格式不同
- 响应格式不同（`<tool_call>...` vs 简化JSON）

### 3. 任务指令分布差异

| 指标 | 训练集 | 验证集 (旧) |
|------|--------|-------------|
| 独特指令数 | 12,980 | 3,371 |
| 指令重叠率 | - | 仅7.1% |
| 内容长度标准差 | 1,581 | 100 |

**92.9%的验证指令在训练集中从未出现过！**

## 解决方案

### 从预处理训练数据中划分80/20

```bash
python scripts/GUI_360/prepare_gui360_sft_parquet.py \
    --input datasets/GUI-360/processed_data/action_prediction_train_resize/training_data.json \
    --output train_GUI_360/data/gui360_train_sft.parquet \
    --image-base-dir datasets/GUI-360/processed_data/action_prediction_train_resize \
    --train-split 0.8 \
    --eval-output train_GUI_360/data/gui360_eval_sft.parquet
```

### 修复后的数据统计

| 指标 | 训练集 | 验证集 |
|------|--------|--------|
| 样本数 | 81,440 | 20,360 |
| 划分比例 | 80% | 20% |
| 内容长度均值 | 7,667 | 7,054 |
| 内容长度标准差 | 1,646 | 1,166 |

Action类型分布对比:
- click: 训练64.6% vs 验证60.8%
- type: 训练16.0% vs 验证17.8%
- 分布基本一致

## 参考资料

### GUI-360论文数据划分

根据[GUI-360论文](https://arxiv.org/html/2511.04307v1)：

> "all three tasks—GUI Grounding, Screen Parsing, and Action Prediction—share the same data partition to maintain consistency across evaluations."

| 数据集 | Trajectories | Steps |
|--------|--------------|-------|
| GUI-360-Train | 13,750 | 105,368 |
| GUI-360-Bench | 3,439 | 26,284 |

训练和测试数据应使用**相同格式**，只是80/20划分。

### 数据目录说明

```
GUI-360/
├── processed_data/           # 预处理数据 (只有训练集)
│   └── action_prediction_train_resize/
│       └── training_data.json    # ← SFT训练使用这个
├── train/                    # 原始训练数据
├── test/                     # 原始测试数据 (不要直接用于validation)
└── fail/                     # 失败轨迹
```

**注意**: `processed_data/` 只包含训练数据的预处理版本，没有测试数据的预处理版本。

## 完整数据配置（对齐论文）

### 当前数据文件

```
train_GUI_360/data/
├── gui360_train_sft.parquet      # 训练集 (81,440 samples)
├── gui360_eval_sft.parquet       # 验证集 (20,360 samples) - 用于训练监控
└── gui360_test_sft_matched.parquet  # 测试集 (26,284 samples) - 用于最终评估
```

### 数据用途说明

| 数据集 | 样本数 | 用途 | 与论文对应 |
|--------|--------|------|-----------|
| `gui360_train_sft.parquet` | 81,440 | SFT训练 | GUI-360-Train的80% |
| `gui360_eval_sft.parquet` | 20,360 | 训练过程监控、early stopping | GUI-360-Train的20% |
| `gui360_test_sft_matched.parquet` | 26,284 | 最终模型评估 | GUI-360-Bench (100%) |

### 数据准备脚本

所有脚本位于 `train_GUI_360/data_preparation/`:

1. **训练/验证集**（从预处理数据划分）:
```bash
python train_GUI_360/data_preparation/prepare_gui360_sft_parquet.py \
    --input datasets/GUI-360/processed_data/action_prediction_train_resize/training_data.json \
    --output train_GUI_360/data/gui360_train_sft.parquet \
    --image-base-dir datasets/GUI-360/processed_data/action_prediction_train_resize \
    --train-split 0.8 \
    --eval-output train_GUI_360/data/gui360_eval_sft.parquet
```

2. **测试集**（从原始测试数据转换，格式对齐训练数据）:
```bash
python train_GUI_360/data_preparation/prepare_gui360_test_parquet_matched.py \
    --test-dir datasets/GUI-360/test/data \
    --image-dir datasets/GUI-360/test/image \
    --output train_GUI_360/data/gui360_test_sft_matched.parquet
```

### 脚本说明

| 脚本 | 用途 |
|------|------|
| `prepare_gui360_sft_parquet.py` | 从预处理数据生成训练/验证集 |
| `prepare_gui360_sft_parquet_a11y.py` | a11y版本数据准备 |
| `prepare_gui360_test_parquet_matched.py` | 从原始测试数据生成格式对齐的测试集 |
| `prepare_gui360_eval_parquet.py` | ⚠️ 旧脚本，格式不一致，不推荐使用 |

### 格式一致性验证

所有三个数据集现在使用相同格式:
- Message结构: `[{role: "user", content: [{type: "text"}, {type: "image"}]}, {role: "assistant", content: "<tool_call>..."}]`
- Response格式: `<tool_call>{"function": "...", "args": {...}, "status": "..."}</tool_call>`

## 总结

| 问题类型 | 描述 |
|----------|------|
| 根本原因 | 训练/验证数据来自不同源，格式不一致 |
| 表现 | validation loss持续上升，与training loss趋势相反 |
| 解决方案 | 从同一数据源划分train/eval，确保格式一致 |

---

*文档创建时间: 2026-02-14*
*最后更新: 2026-02-14 (添加测试集准备)*
*问题发现于: gui360_sft_v2_2311691.log*
