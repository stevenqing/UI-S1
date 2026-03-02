# Task 5: Tool-Use 宏动作 SFT 训练 — 任务清单

## 任务列表

### Task 5.1: 探索现有代码结构和数据格式
- **状态**: ✅ 完成
- **说明**: 阅读现有脚本/配置，理解数据格式和代码约定

### Task 5.2: 创建 `scripts/prepare_macro_sft_data.py`
- **状态**: ✅ 完成
- **说明**: 数据准备脚本 — 识别跨越轨迹 → 分类宏类型 → 构造增强 SFT 样本 → 输出 parquet
- **依赖**: Task 5.1

### Task 5.3: 创建 `train_GUI_360/config/gui360_sft_macro.yaml`
- **状态**: ✅ 完成
- **说明**: 宏增强 SFT 训练配置，继承现有配置
- **依赖**: Task 5.1

### Task 5.4: 验证输出和正确性
- **状态**: ✅ 完成
- **说明**: 运行脚本，检查统计数据，验证 parquet 输出格式
- **依赖**: Task 5.2, Task 5.3

## 运行结果

### 跨越轨迹统计

| App | 跨越轨迹数 | 宏样本数 | navigate_to_dialog | navigate_and_return | switch_ui_mode |
|-----|-----------|---------|-------------------|-------------------|----------------|
| Excel | 89 | 89 | 87 | 2 | 0 |
| Word | 69 | 69 | 59 | 10 | 0 |
| PPT | 201 | 201 | 192 | 9 | 0 |
| **Total** | **359** | **359** | **338** | **21** | **0** |

### 混合数据集

| 数据源 | 样本数 | 上采样后 | 占比 |
|--------|--------|---------|------|
| 原始 SFT | 105,340 | 105,340 | 96.7% |
| 宏增强 | 359 | 3,590 | 3.3% |
| **混合总计** | - | **108,930** | 100% |

### 评估数据集
- 基线 eval: 26,273 + 宏样本 35 = 26,308

## 输入数据

| 文件 | 来源 |
|------|------|
| `outputs/fnet/gui360/{app}/f_values.npz` | Task 3 — 每个状态的 f-value |
| `outputs/transitions/gui360_full/transitions.jsonl` | Task 2 — 转移记录 |
| `outputs/transitions/gui360_full/state_registry.json` | Task 2 — 状态元数据 |
| `datasets/GUI-360/train/data/{app}/**/*.jsonl` | 原始轨迹 |
| `train_GUI_360/data/gui360_train_sft_a11y_thinking.parquet` | 已有基线 SFT 数据 |

## 输出目录

```
outputs/macro_sft/
├── crossing_trajectories/
│   ├── excel_crossings.json       (31K, 89 crossings)
│   ├── word_crossings.json        (26K, 69 crossings)
│   └── ppt_crossings.json         (70K, 201 crossings)
├── macro_playbook.json            (291K, 359 entries)
├── macro_augmented_train.parquet  (5.2M, 359 samples)
├── macro_mixed_train.parquet      (354M, 108,930 samples)
├── macro_mixed_eval.parquet       (53M, 26,308 samples)
├── macro_tool_definitions.json    (2.4K, 3 tools)
└── statistics.json                (1.1K)
```

## 新建文件

| 文件 | 说明 |
|------|------|
| `scripts/prepare_macro_sft_data.py` | 数据准备脚本 |
| `train_GUI_360/config/gui360_sft_macro.yaml` | 训练配置 |

## CLI 命令

```bash
# 数据准备（CPU，~15 秒）
python scripts/prepare_macro_sft_data.py

# 训练
python -m verl.trainer.fsdp_sft_trainer --config-path train_GUI_360/config --config-name gui360_sft_macro
```
