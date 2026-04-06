# GUI-Odyssey 评测方案设计

基于现有 GUI-360 和 Android Control 的评测体系，为 GUI-Odyssey 设计统一的评测框架。

---

## 1. 数据集概览

| 属性 | 值 |
|------|-----|
| 总 episodes | 8,334 |
| 平均步数 | 15.3 |
| 步数范围 | 2–59 |
| 截图总数 | 127,893 |
| 设备 | 6 种 (Pixel 7/8 Pro, Fold, Tablet, Small/Medium Phone) |
| 类别 | 6 种 (见下) |
| 坐标范围 | [0, 1000] 归一化 |

**类别分布:**

| 类别 | 数量 |
|------|------|
| General_Tool | 1,821 |
| Multi_Apps | 1,736 |
| Information_Management | 1,423 |
| Social_Sharing | 1,316 |
| Media_Entertainment | 1,080 |
| Web_Shopping | 958 |

**数据划分:**

| Split | Train | Test | 聚合方式 |
|-------|-------|------|----------|
| random_split | 6,668 | 1,666 | micro (按类别平均) |
| app_split | 7,165 | 1,169 | macro (全局) |
| device_split | 6,953 | 1,381 | macro |
| task_split | 7,199 | 1,135 | macro |

**Action 类型分布 (采样 200 episodes):**

| Action | 数量 | 对应我们格式 |
|--------|------|-------------|
| CLICK | 2,238 | click |
| TEXT | 332 | type |
| SCROLL | 275 | swipe |
| COMPLETE | 187 | terminate (status=FINISH) |
| LONG_PRESS | 10 | long_press |
| INCOMPLETE | 13 | terminate (status=IMPOSSIBLE) |

---

## 2. 与现有评测体系的对齐

### 2.1 Action 类型映射

| GUI-Odyssey 原始 | 转换后命令格式 | 对应 AC 类型 | 对应 GUI-360 类型 |
|-----------------|---------------|-------------|------------------|
| `CLICK` | `click(coordinate=[x, y])` | `click` | `click` |
| `LONG_PRESS` | `long_press(coordinate=[x, y])` | `long_press` | — |
| `SCROLL` | `swipe(direction=UP/DOWN/LEFT/RIGHT)` | `swipe` | `scroll` |
| `TEXT` | `type(content="...")` | `type` | `type` |
| `COMPLETE` | `terminate(status="success")` | — | `status=FINISH` |
| `INCOMPLETE` | `terminate(status="impossible")` | — | — |
| CLICK+KEY_HOME | `system_button(button="home")` | `system_button` | — |
| CLICK+KEY_BACK | `system_button(button="back")` | `system_button` | — |
| CLICK+KEY_APPSELECT | `system_button(button="recent")` | `system_button` | — |

### 2.2 坐标系统对齐

| 系统 | 坐标格式 | 范围 |
|------|----------|------|
| GUI-Odyssey 原始 | `[0, 1000]` 归一化 | 宽高各 1000 |
| Android Control | 绝对像素坐标 | 设备分辨率 |
| GUI-360 | 绝对像素坐标 | 截图分辨率 |
| **Qwen2.5-VL 输出** | `[0, 1000]` 归一化 | 宽高各 1000 |

GUI-Odyssey 的坐标系与 Qwen2.5-VL 的输出格式天然对齐（都是 [0,1000]），无需额外转换。

### 2.3 Prompt 格式对齐

沿用 Android Control 的 `JsonFormat`（`x` 库）生成 prompt，输出格式:

```xml
<think>reasoning about the current step</think>
<action>
action_type(param1=value1, param2=value2)
</action>
```

**与 GUI-Odyssey 原始 prompt 的区别:**
- 原始: `"I'm looking for guidance on how to {instruction}\nProvide the command-style action directly."`
- 我们: 使用标准化的 system prompt + 多轮对话格式 + `<think>/<action>` 结构

---

## 3. 评测流程

### 3.1 数据预处理

将 GUI-Odyssey 标注转换为 AC 兼容的 JSONL 格式:

```python
# 输入: GUI-Odyssey annotation JSON
# 输出: JSONL, 每行一个 episode
{
    "episode_id": "0000628297785099",
    "instruction": "Use Threads and Agoda to ...",
    "category": "Web_Shopping",
    "device_name": "Pixel 7 Pro",
    "resolution": {"width": 1440, "height": 3120},
    "steps": [
        {
            "screenshot": "/path/to/screenshots/0000628297785099_0.png",
            "action_content": {
                "action": "click",
                "coordinate": [144, 383]  # [0, 1000] 归一化
            },
            "check_options": {
                "candidate_bbox": [113, 359, 189, 400]  # sam2_bbox
            },
            "thought": "Open the Threads app.",  # low_level_instruction
            "description": "Android home screen with app icons..."
        },
        ...
    ]
}
```

### 3.2 推理流程 (与 AC eval_a_ar_trajectory.py 对齐)

```
┌─────────────────────────────────────────────────┐
│  1. 启动 vLLM server (4 GPU tensor parallel)     │
│  2. 加载 test episodes                           │
│  3. 对每个 episode:                               │
│     a. 构造初始 prompt (instruction + screenshot) │
│     b. 模型推理 → 解析 <think>/<action>           │
│     c. 与 GT 比较 (action matching)               │
│     d. 用模型预测构造下一步 history (AR模式)       │
│     e. stop_on_error: 首个错误即停               │
│  4. 汇总指标                                      │
└─────────────────────────────────────────────────┘
```

### 3.3 评测模式

| 模式 | 说明 | 对应 GUI-360 | 对应 AC |
|------|------|-------------|---------|
| **Teacher-Forced** | 每步用 GT screenshot，单步评测 | `action_prediction` | — |
| **AR Trajectory** | 用模型自身预测构造 history，stop on error | `action_prediction_ar` | `eval_a_ar_trajectory` |
| **AR Non-Stop** | AR 但不 stop on error，评测所有步 | `action_prediction_ar` (stop_on_error=False) | — |

---

## 4. 评测指标

### 4.1 Step-Level 指标

#### Action Type Match
```
type_match = (pred_action_type == gt_action_type)
```

#### Action Extract Match (完整匹配)
基于 action type 的细分匹配:

| Action | 匹配规则 | 来源 |
|--------|----------|------|
| **click / long_press** | 坐标在 `sam2_bbox` 内，或归一化欧氏距离 ≤ 0.14 | GUI-Odyssey 官方 |
| **swipe (scroll)** | 方向完全匹配 (UP/DOWN/LEFT/RIGHT) | GUI-Odyssey 官方 |
| **type** | ANLS ≥ 0.5 (Levenshtein)，或互为子串 | GUI-Odyssey 官方 |
| **system_button** | button 名称匹配 | AC |
| **terminate** | status 匹配 | GUI-360 |

#### Action Matching Score (AMS)
- **macro**: `AMS = correct_steps / total_steps * 100`
- **micro**: 按 6 类别分别算 AMS，取平均

### 4.2 Trajectory-Level 指标

| 指标 | 公式 | 说明 |
|------|------|------|
| **TSR** (Trajectory Success Rate) | `success_episodes / total_episodes` | 所有 step 都正确 |
| **Avg Progress** | `mean(final_correct_step / total_steps)` | 平均进度 (stop on error) |
| **Scattered Progress** | `total_correct_steps / total_steps` | 分散正确率 |

### 4.3 维度分解 (与 GUI-360 对齐)

| 维度 | 切分 |
|------|------|
| **按类别** | 6 categories |
| **按 trajectory 长度** | short(1-5), medium(6-15), long(16+) |
| **按设备** | 6 devices |
| **按 action type** | click, scroll, type, long_press, system_button, terminate |
| **按 step position** | 每个 step index 的准确率 |

---

## 5. 实现计划

### 5.1 需要编写的文件

```
gui_odyssey_eval/
├── GUIOdyssey_action_matching.py   # ✅ 已有 (官方)
├── evaluate_GUIOdyssey.py          # ✅ 已有 (官方, 仅参考)
├── format_converter.py             # ✅ 已有 (官方)
├── preprocessing.py                # ✅ 已有 (官方)
├── eval.sh                         # ✅ 已有 (官方)
├── README.md                       # ✅ 已有
├── EVALUATION_DESIGN.md            # ✅ 本文件
├── convert_to_ac_format.py         # 🔲 新建: 标注转 AC JSONL
├── eval_ar_trajectory.py           # 🔲 新建: AR 评测 (基于 AC eval_a)
├── eval_teacher_forced.py          # 🔲 新建: Teacher-forced 评测
├── odyssey_utils.py                # 🔲 新建: 数据加载/指标计算
├── eval_odyssey.slurm              # 🔲 新建: SLURM 脚本
└── results/                        # 🔲 评测结果输出
```

### 5.2 SLURM 脚本模板 (沿用 GUI-360 模式)

```bash
#!/bin/bash
#SBATCH --job-name=eval_odyssey
#SBATCH --time=08:00:00
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --gres=gpu:4

# 1. 启动 vLLM
python -m vllm.entrypoints.openai.api_server \
    --model $CHECKPOINT \
    --tensor-parallel-size 4 \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.9 &

# 2. 等待就绪
wait_for_server

# 3. 运行评测
python eval_ar_trajectory.py \
    --data /datasets/GUI-Odyssey \
    --split random_split \
    --level high \
    --model-name $MODEL_NAME \
    --n-history-image-limit 4 \
    --stop-on-error \
    --output results/

# 4. 清理
kill $VLLM_PID
```

### 5.3 与 RL Reward 的关系

如需将 GUI-Odyssey 加入 RL 训练，reward 函数设计:

| 组件 | 权重 | 说明 |
|------|------|------|
| format_score | 0.1 | `<think>` 非空 |
| action_score | 0.9 | soft coordinate score (连续, 非二元) |

沿用 `verl/utils/reward_score/gui360/reward.py` 的 soft coordinate scoring:
- 距离 ≤ 0.02 → 1.0
- 距离 ≤ 0.05 → 线性插值至 0.3
- 距离 ≤ 0.10 → 线性插值至 0.1
- 距离 ≤ 0.15 → 线性插值至 0.0

注意: GUI-Odyssey 坐标已经是 [0,1000]，需除以 1000 归一化到 [0,1] 后再算距离。

---

## 6. 关键差异与注意事项

| 维度 | GUI-360 | Android Control | GUI-Odyssey |
|------|---------|----------------|-------------|
| 平台 | Desktop (Word/Excel/PPT) | Android 单 app | Android 跨 app |
| 坐标 | 绝对像素 | 绝对像素 | [0,1000] 归一化 |
| GT 匹配 | point-in-rectangle (25px tol) | candidate_bbox | sam2_bbox + 距离阈值 0.14 |
| Trajectory 长度 | ~3-20 步 | ~1-15 步 | **2-59 步 (更长)** |
| 特有 action | hotkey, drag, select_text | open, wait | LONG_PRESS, KEY_HOME/BACK |
| 历史 | 文本 history | 图像+文本 history | 图像+文本 history |

**关键注意:**
1. GUI-Odyssey 的 trajectory 显著更长（平均 15.3 步，最长 59 步），TSR 会非常低
2. 跨 app 任务涉及 app 切换，对 history 管理要求更高
3. `sam2_bbox` 可能为 `null`（TYPE/SCROLL 等非坐标 action），需兼容处理
4. 6 类别的 micro 聚合要求按类别分别计算后取平均
