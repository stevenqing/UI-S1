# GUI-360 SFT Training Guide

本文档总结了 GUI-360 论文中三种任务的 SFT（Supervised Fine-Tuning）训练方法。

---

## 1. 数据集概览

| 数据集 | 轨迹数 | 总步数 | 平均步数 | UI元素数 | 截图数 |
|--------|--------|--------|----------|----------|--------|
| **GUI-360-Train** | 13,750 | 105,368 | 7.66 | 17,668,694 | 210,736 |
| **GUI-360-Bench** | 3,439 | 26,284 | 7.64 | 4,324,617 | 52,568 |

### 应用分布
- Word: 41.0%
- Excel: 31.6%
- PowerPoint: 27.4%

---

## 2. 三种 Canonical Tasks

### 2.1 GUI Grounding (元素定位)

**目标**: 给定截图和 Agent 的思考，预测目标元素的操作坐标。

**数据量**: 79,487 样本

**数据格式**:
```json
{
  "id": "excel_4s_1_2",
  "images": ["screenshot.png"],
  "conversation": [
    {
      "from": "human",
      "value": "<image>\nYou are a helpful assistant. Given a screenshot of the current screen and user instruction, you need to output the position of the element you will operate.\n\nThe instruction is:\n[Agent's thought about the current step]\n\nOutput the coordinate of the element you will operate within <coordinate></coordinate> tag:\n<coordinate> [x, y] </coordinate>"
    },
    {
      "from": "gpt",
      "value": "<coordinate> [630, 241] </coordinate>"
    }
  ]
}
```

**评估指标**:
- Accuracy: 预测坐标落在目标元素 bounding box 内的比例

### 2.2 Screen Parsing (屏幕解析)

**目标**: 给定截图，输出所有可交互 UI 元素的列表。

**数据量**: 97,351 样本

**数据格式**:
```json
{
  "id": "excel_bing_search_excel_4s_1_2",
  "images": ["screenshot.png"],
  "conversation": [
    {
      "from": "human",
      "value": "<image>\n\nYou are an expert in screen parsing and GUI element extraction.\n\nYou will be provided with a screenshot of a desktop application interface. Your task is to find all interactive controls on the screen (anywhere that can be clicked, typed into, etc.), with a maximum number not to exceed 500.\n\nFor each control element you identify, you need to provide:\n1. **control_text**: The visible text displayed on or within the control.\n2. **control_rect**: The bounding rectangle coordinates [left, top, right, bottom].\n\nOutput your response in JSON format:\n```json\n[{\"control_text\": \"File\", \"control_rect\": [x1,y1,x2,y2]}, ...]\n```"
    },
    {
      "from": "gpt",
      "value": "[{\"control_type\": \"Button\", \"control_rect\": [62, 699, 94, 720], \"control_text\": \"Macro Recording\", \"source\": \"uia\", \"label\": 1}, ...]"
    }
  ]
}
```

**评估指标**:
- Precision / Recall / F1 (元素检测)
- Mean IoU (定位质量)
- Text Similarity (语义名称准确性)

### 2.3 Action Prediction (动作预测) - Long Horizon Task

**目标**: 给定用户指令、截图（和可选的 accessibility 信息），预测下一步动作。

**数据量**: 101,800 样本

**两种输入模态**:

| 模态 | 输入 | 坐标表示 |
|------|------|----------|
| **Visual-only** | 仅截图 | 绝对坐标 `[x, y]` |
| **Visual+A11y** | 截图 + accessibility 元素列表 | 元素 ID + 名称 |

#### Visual-only 格式:
```json
{
  "id": "excel_bing_search_excel_4s_1_2",
  "images": ["screenshot.png"],
  "conversation": [
    {
      "from": "human",
      "value": "<image>\nYou are a helpful assistant. Given a screenshot of the current screen, user instruction and history of actions, you need to decide the next action to take.\n\nThe instruction is:\n[User's natural language request]\n\nThe history of actions are:\n[Past actions]\n\nThe actions supported are:\n<action>\n- click: click(coordinate=[x, y], button='left', double=False)\n- type: type(coordinate=[x, y], keys='Hello')\n- drag: drag(start_coordinate=[x1,y1], end_coordinate=[x2,y2])\n- wheel_mouse_input: wheel_mouse_input(coordinate=[x,y], wheel_dist=1)\n..."
    },
    {
      "from": "gpt",
      "value": "click(coordinate=[100, 200], button='left')"
    }
  ]
}
```

#### Visual+A11y 格式:
- 截图上会标注 Set-of-Mark (SoM) 标记
- 模型同时接收元素列表
- 坐标用元素 ID 代替：`click(element_id=5, element_name="Insert")`

**评估指标**:
- Function Accuracy: 动作类型是否正确
- Argument Accuracy: 参数（坐标/值）是否正确
- Status Accuracy: 是否正确预测继续/结束
- Step Success Rate: 三者全部正确的比例

---

## 3. SFT 训练配置

### 3.1 硬件配置 (论文 Appendix D)

```yaml
硬件: 4x NVIDIA A100 GPUs (40GB each)
精度: FP16 mixed precision
分布式: FSDP (Fully Sharded Data Parallel)
```

### 3.2 超参数

```yaml
# 学习率网格搜索
learning_rate: [1e-5, 5e-6, 1e-6]  # 论文推荐

# 优化器
optimizer: AdamW
weight_decay: 0.01
warmup_steps_ratio: 0.1

# 训练
epochs: 3
gradient_accumulation: enabled
gradient_checkpointing: enabled

# 序列长度
max_length: 8192
```

### 3.3 推荐配置 (基于本项目)

```yaml
# 参考 examples/qwen_gui_sft/config/gui_sft.yaml

data:
  train_files: /path/to/gui360_train.parquet
  val_files: /path/to/gui360_eval.parquet
  train_batch_size: 32
  micro_batch_size_per_gpu: 1
  max_length: 8192

model:
  partial_pretrain: Qwen/Qwen2.5-VL-7B-Instruct
  strategy: fsdp2
  enable_gradient_checkpointing: True

optim:
  lr: 5e-6
  weight_decay: 0.01
  clip_grad: 1.0
  lr_scheduler: cosine
```

---

## 4. 训练效果 (论文 Table 7, 9)

### 4.1 GUI Grounding

| 模型 | Word | Excel | PPT | Overall |
|------|------|-------|-----|---------|
| Qwen-2.5-VL-7B (Zero-shot) | 38.09% | 26.76% | 41.55% | 35.78% |
| **Qwen-2.5-VL-7B-SFT** | **84.11%** | **79.20%** | **82.84%** | **82.30%** |

### 4.2 Action Prediction (Visual-only)

| 模型 | Word | Excel | PPT | Overall |
|------|------|-------|-----|---------|
| Qwen-2.5-VL-7B (Zero-shot) | 15.70% | 12.75% | 25.09% | 17.52% |
| **Qwen-2.5-VL-7B-SFT** | **49.10%** | **45.12%** | **56.53%** | **50.08%** |

### 4.3 Action Prediction (Visual+A11y)

| 模型 | Word | Excel | PPT | Overall |
|------|------|-------|-----|---------|
| Qwen-2.5-VL-7B (Zero-shot) | 15.64% | 3.56% | 22.51% | 14.18% |
| Qwen-2.5-VL-7B-SFT | 31.68% | 7.44% | 34.99% | 25.78% |

---

## 5. 关键发现

1. **SFT 对 Visual-only 设置帮助最大**
   - Action Prediction: 17.52% → 50.08% (近 3 倍提升)
   - GUI Grounding: 35.78% → 82.30%

2. **A11y 信息减少了对 SFT 的依赖**
   - 有 A11y 时，SFT 提升幅度较小
   - 说明结构化信息本身提供了大量对齐信号

3. **Argument 预测是主要瓶颈**
   - Function 准确率较高 (>80%)
   - 坐标/参数预测仍是挑战 (Args Mismatch Error > 50%)

4. **Coord. OOB (Out-of-Bounds) 是主要错误来源**
   - 视觉定位困难
   - A11y 信息可显著减少此类错误

---

## 6. 数据处理流程

```
原始轨迹 (JSONL)
    │
    ├──> GUI Grounding 数据
    │    └── 每步提取: 截图 + thought → 坐标
    │
    ├──> Screen Parsing 数据
    │    └── 每步提取: 截图 → 所有可交互元素列表
    │
    └──> Action Prediction 数据
         ├── Visual-only: 截图 + instruction + history → action
         └── Visual+A11y: 截图 + a11y元素 + instruction → action
```

---

## 7. Action Space

### 7.1 GUI Actions (通用)

| Action | 参数 | 示例 |
|--------|------|------|
| `click` | coordinate, button, double, pressed | `click(coordinate=[100, 100], button='left')` |
| `type` | coordinate, keys, clear_current_text | `type(coordinate=[100, 100], keys='Hello')` |
| `drag` | start_coordinate, end_coordinate, button, duration | `drag(start=[0,0], end=[100,100])` |
| `wheel_mouse_input` | coordinate, wheel_dist | `wheel_mouse_input(coordinate=[100,100], wheel_dist=1)` |

### 7.2 API Actions (应用特定)

**Word API**: `insert_table`, `select_text`, `select_table`, `save_as`, `set_font`...

**Excel API**: `table2markdown`, `insert_excel_table`, `set_cell_value`, `auto_fill`, `reorder_columns`...

**PowerPoint API**: `set_background_color`, `save_as`...

---

## 8. 参考文献

- Paper: [GUI-360: A Comprehensive Dataset and Benchmark for Computer-Using Agents](https://arxiv.org/abs/2511.04307)
- GitHub: https://github.com/2020-qqtcg/GUI-360
- HuggingFace: https://huggingface.co/datasets/vyokky/GUI-360

---

## 9. 附录：完整数据 Schema

```json
{
  "execution_id": "excel_search_00001",
  "app_domain": "excel",
  "request": "Create a pivot table...",
  "template": "template9.xlsx",
  "step_id": 3,
  "total_steps": 12,
  "evaluation": {
    "reason": "Task completed successfully...",
    "complete": "yes",
    "sub_scores": {"correctness": 1.0, "format": 1.0, "completeness": 1.0}
  },
  "step": {
    "screenshot_clean": "path/to/clean.png",
    "screenshot_desktop": "path/to/desktop.png",
    "screenshot_annotated": "path/to/annotated.png",
    "ui_tree": {...},
    "control_infos": {
      "application_windows_info": {...},
      "uia_controls_info": [...],
      "grounding_controls_info": [...],
      "merged_controls_info": [...]
    },
    "subtask": "Click on the Insert menu",
    "observation": "I see the Excel window with...",
    "thought": "To create a pivot table, I need to...",
    "action": {
      "action_type": "GUI",
      "control_text": "Insert",
      "control_label": "Menu Item",
      "function": "click",
      "args": {},
      "rectangle": [100, 50, 200, 80],
      "coordinate_x": 150,
      "coordinate_y": 65
    },
    "status": "CONTINUE",
    "tags": ["grounding", "screen_parsing", "action_prediction"]
  }
}
```

---

*文档生成时间: 2025-02-13*
*基于 GUI-360 论文 (arXiv:2511.04307)*
