# UI-S1 Datasets Analysis Report

## 1. 数据集概览

| 数据集 | 记录数 | 总步数 | 平均轨迹长度 | 最短 | 最长 | 文件大小 |
|--------|--------|--------|--------------|------|------|----------|
| android_control_train_example.jsonl | 32 | 206 | 6.44 | 2 | 13 | 35KB |
| android_control_evaluation_std.jsonl | 1,543 | 8,444 | 5.47 | 1 | 24 | 2.6MB |
| ui_s1_train.jsonl | 1,000 | 6,536 | 6.54 | 1 | 34 | 2.1MB |
| ui_s1_train_filtered.jsonl | 1,000 | 6,536 | 6.54 | 1 | 34 | 1.6MB |
| ui_s1_dataset (Arrow) | 1,000 | - | - | - | - | 719KB |

**AndroidControl/images**: 16,523 张截图

---

## 2. 数据格式详解

### 2.1 android_control_train_example.jsonl

**顶层字段**:
- `goal`: 任务目标描述 (string)
- `is_successful`: 是否成功 (bool)
- `steps`: 操作步骤列表 (array)

**Step结构**:
```json
{
  "action_content": {
    "action": "click",
    "coordinate": [540.0, 389.8],
    "bbox": [360, 327, 695, 466],
    "text": "Zoho Meeting"
  },
  "screenshot": "/datasets/AndroidControl/images/0_0.png"
}
```

### 2.2 android_control_evaluation_std.jsonl

**顶层字段**:
- `goal`: 任务目标描述 (string)
- `is_successful`: 是否成功 (bool)
- `episode_id`: episode唯一标识 (int)
- `steps`: 操作步骤列表 (array)

**Step结构** (包含额外评估字段):
```json
{
  "action_content": {
    "action": "click",
    "coordinate": [540, 2273]
  },
  "screenshot": "/datasets/AndroidControl/images/8193_1.png",
  "step_instruction": "click on create tab",
  "check_options": {
    "action": "click",
    "candidate_bbox": [[456, 2211, 624, 2337], [0, 473, 1080, 2337]],
    "coordinate": [540, 2273]
  }
}
```

**特点**:
- 包含 `step_instruction` (步骤说明)
- 包含 `check_options` (评估选项/候选框)
- **不包含 terminate action** (用于评估，非完整轨迹)

### 2.3 ui_s1_train.jsonl

**顶层字段**:
- `goal`: 任务目标描述 (string)
- `is_successful`: 是否成功 (bool)
- `steps`: 操作步骤列表 (array)

**Step结构** (标准化字段):
```json
{
  "action_content": {
    "action": "click",
    "coordinate": [540.0, 389.8],
    "bbox": [360, 327, 695, 466],
    "text": null,
    "status": null,
    "button": null,
    "coordinate2": null,
    "time": null
  },
  "screenshot": "/lus/lfs1aip2/.../images/0_0.png"
}
```

**特点**:
- action_content 字段标准化（所有字段都存在，缺失为null）
- screenshot 使用完整绝对路径
- **包含 terminate action** (完整轨迹)

### 2.4 ui_s1_dataset (Arrow格式)

HuggingFace Datasets 格式，来源于 `mPLUG/UI_S1_dataset`。

**Schema**:
- `goal`: string
- `is_successful`: bool
- `steps`: List of step objects

---

## 3. Action类型分布

### 3.1 android_control_train_example (32条轨迹)

| Action | 数量 | 占比 |
|--------|------|------|
| click | 102 | 49.5% |
| terminate | 32 | 15.5% |
| swipe | 19 | 9.2% |
| type | 16 | 7.8% |
| wait | 15 | 7.3% |
| open | 13 | 6.3% |
| system_button | 9 | 4.4% |

### 3.2 android_control_evaluation_std (1,543条轨迹)

| Action | 数量 | 占比 |
|--------|------|------|
| click | 5,074 | 60.1% |
| swipe | 1,211 | 14.3% |
| type | 632 | 7.5% |
| open | 608 | 7.2% |
| wait | 567 | 6.7% |
| system_button | 343 | 4.1% |
| long_press | 9 | 0.1% |

### 3.3 ui_s1_train.jsonl (1,000条轨迹)

| Action | 数量 | 占比 |
|--------|------|------|
| click | 3,380 | 51.7% |
| terminate | 1,000 | 15.3% |
| swipe | 782 | 12.0% |
| type | 391 | 6.0% |
| open | 376 | 5.8% |
| wait | 372 | 5.7% |
| system_button | 227 | 3.5% |
| long_press | 8 | 0.1% |

---

## 4. 轨迹长度分布

### 4.1 百分位数统计

| 数据集 | P25 | P50(中位数) | P75 | P90 | 平均 |
|--------|-----|-------------|-----|-----|------|
| android_control_train_example | 4 | 7 | 8 | 10 | 6.44 |
| android_control_evaluation_std | 3 | 5 | 7 | 9 | 5.47 |
| ui_s1_train | 4 | 6 | 8 | 11 | 6.54 |

### 4.2 详细分布 (android_control_evaluation_std)

| 长度 | 轨迹数 | 累计占比 |
|------|--------|----------|
| 1 | 116 | 7.5% |
| 2 | 122 | 15.4% |
| 3 | 200 | 28.4% |
| 4 | 221 | 42.7% |
| 5 | 256 | 59.3% |
| 6 | 173 | 70.5% |
| 7 | 138 | 79.5% |
| 8 | 102 | 86.1% |
| 9 | 65 | 90.3% |
| 10+ | 150 | 100% |

### 4.3 详细分布 (ui_s1_train)

| 长度 | 轨迹数 | 累计占比 |
|------|--------|----------|
| 1 | 8 | 0.8% |
| 2 | 72 | 8.0% |
| 3 | 108 | 18.8% |
| 4 | 139 | 32.7% |
| 5 | 156 | 48.3% |
| 6 | 143 | 62.6% |
| 7 | 114 | 74.0% |
| 8 | 70 | 81.0% |
| 9 | 45 | 85.5% |
| 10+ | 145 | 100% |

---

## 5. 数据集差异对比

| 特性 | train_example | evaluation_std | ui_s1_train |
|------|---------------|----------------|-------------|
| terminate action | 有 | 无 | 有 |
| episode_id | 无 | 有 | 无 |
| step_instruction | 无 | 有 | 无 |
| check_options | 无 | 有 | 无 |
| 字段标准化 | 否 | 否 | 是 |
| 路径格式 | 相对路径 | 相对路径 | 绝对路径 |
| 用途 | 训练示例 | 评估 | 训练 |

---

## 6. Action类型说明

| Action | 参数 | 说明 |
|--------|------|------|
| click | coordinate, bbox | 点击指定坐标位置 |
| type | text | 输入文本 |
| swipe | coordinate, coordinate2 | 从起点滑动到终点 |
| open | text | 打开指定应用 |
| wait | time | 等待指定时间(秒) |
| system_button | button | 按系统按钮(Back/Home等) |
| long_press | coordinate | 长按指定位置 |
| terminate | status | 结束任务(success/fail) |

---

## 7. 总结

1. **数据规模**: 共有约2,575条轨迹，15,186个操作步骤
2. **Action分布**: click操作占主导(50-60%)，其次是swipe(9-14%)
3. **轨迹长度**: 平均5-7步，90%的轨迹在11步以内
4. **数据质量**: 所有轨迹标记为成功(is_successful=true)
5. **格式差异**: evaluation_std包含评估相关字段，ui_s1_train字段标准化

---

*Generated: 2026-02-06*
