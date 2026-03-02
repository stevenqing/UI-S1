# UI-S1 训练指南

## 1. 训练脚本

### 位置
```
scripts/train_example.sh      # 7B 模型
scripts/train_example_32b.sh  # 32B 模型
```

### 启动命令
```bash
bash scripts/train_example.sh
```

### 核心配置参数
| 参数 | 值 | 说明 |
|------|-----|------|
| 模型 | Qwen2.5-VL-7B-Instruct | 基座模型 |
| Batch size | 8 × 节点数 | 每批次样本数 |
| Rollout n | 8 | 每个 prompt 生成 8 个 rollout |
| Epochs | 3 | 训练轮数 |
| Learning rate | 1e-6 | 学习率 |
| max_prompt_length | 12288 | 最大输入长度 |
| max_response_length | 512 | 最大输出长度 |

---

## 2. 训练数据统计

| 数据集 | 样本数 | 总步数 | 平均步数 |
|--------|--------|--------|----------|
| `ui_s1_train.jsonl` | 1,000 trajectories | 6,536 steps | 6.5 步/样本 |
| 图片数量 | 16,523 张 | - | - |

### 动作分布
```
click:         3,380 (51.7%)
terminate:     1,000 (15.3%)
swipe:           782 (12.0%)
type:            391 (6.0%)
open:            376 (5.8%)
wait:            372 (5.7%)
system_button:   227 (3.5%)
long_press:        8 (0.1%)
```

---

## 3. 训练时间估算 (7B 模型)

| 节点配置 | 估算时间 |
|----------|----------|
| 1 节点 (8 GPU) | ~15-16 小时 |
| 2 节点 (16 GPU) | ~8-9 小时 |
| 4 节点 (32 GPU) | ~4-5 小时 |

---

## 4. 数据格式

### JSONL 结构
```json
{
  "goal": "Open the Zoho Meet app, view the scheduled meetings.",
  "is_successful": true,
  "steps": [
    {
      "action_content": {
        "action": "open",
        "text": "Zoho Meeting"
      },
      "screenshot": "/images/0_0.png"
    },
    {
      "action_content": {
        "action": "wait",
        "time": 2
      },
      "screenshot": "/images/0_1.png"
    },
    {
      "action_content": {
        "action": "click",
        "coordinate": [540, 390]
      },
      "screenshot": "/images/0_2.png"
    },
    {
      "action_content": {
        "action": "terminate",
        "status": "success"
      },
      "screenshot": "/images/0_3.png"
    }
  ]
}
```

---

## 5. 动作空间 (Action Space)

| 动作 | 参数 | 示例 |
|------|------|------|
| `click` | coordinate | `{"action": "click", "coordinate": [540, 390]}` |
| `swipe` | coordinate, coordinate2 | `{"action": "swipe", "coordinate": [540, 600], "coordinate2": [540, 1800]}` |
| `type` | text | `{"action": "type", "text": "Hello"}` |
| `open` | text | `{"action": "open", "text": "Zoho Meeting"}` |
| `wait` | time | `{"action": "wait", "time": 2}` |
| `system_button` | button | `{"action": "system_button", "button": "Back"}` |
| `long_press` | coordinate, time | `{"action": "long_press", "coordinate": [540, 390], "time": 2}` |
| `terminate` | status | `{"action": "terminate", "status": "success"}` |

---

## 6. 多轮对话结构

### 第 1 轮
```
[SYSTEM] You are a GUI agent...

[USER]
User Instruction: Open the Zoho Meet app, view the scheduled meetings.
Output Format: <think>...</think><action>...</action>
[Screenshot 0_0.png - 日历界面]

[ASSISTANT - 训练标签]
<think>...</think>
<action>{"action": "open", "text": "Zoho Meeting"}</action>
```

### 第 2 轮
```
[SYSTEM] You are a GUI agent...

[USER]
User Instruction: Open the Zoho Meet app, view the scheduled meetings.
[Screenshot 0_0.png]

[ASSISTANT - 历史]
<action>{"action": "open", "text": "Zoho Meeting"}</action>

[USER]
Output Format: <think>...</think><action>...</action>
[Screenshot 0_1.png - 蓝色加载页]

[ASSISTANT - 训练标签]
<think>...</think>
<action>{"action": "wait", "time": 2}</action>
```

### 第 3 轮
```
[SYSTEM] ...
[USER] ... [Screenshot 0_0.png]
[ASSISTANT] {"action": "open", ...}
[USER] [Screenshot 0_1.png]
[ASSISTANT] {"action": "wait", ...}
[USER] [Screenshot 0_2.png - UPCOMING 页面]

[ASSISTANT - 训练标签]
<think>...</think>
<action>{"action": "click", "coordinate": [540, 390]}</action>
```

### 第 4 轮
```
[SYSTEM] ...
[USER] ... (完整历史)
[USER] [Screenshot 0_3.png - PAST 页面，显示会议记录]

[ASSISTANT - 训练标签]
<think>...</think>
<action>{"action": "terminate", "status": "success"}</action>
```

---

## 7. 完整样例截图序列

**Goal:** "Open the Zoho Meet app, view the scheduled meetings."

| 步骤 | 截图 | 动作 | 说明 |
|------|------|------|------|
| 1 | 0_0.png | `open "Zoho Meeting"` | 初始日历界面 → 打开应用 |
| 2 | 0_1.png | `wait 2` | 蓝色加载页 → 等待加载 |
| 3 | 0_2.png | `click [540, 390]` | UPCOMING 页面 → 点击切换标签 |
| 4 | 0_3.png | `terminate success` | PAST 页面显示会议 → 任务完成 |

### 步骤必要性分析

| 步骤 | 必要性 | 原因 |
|------|--------|------|
| `open` | ✅ 必要 | 必须打开目标应用 |
| `wait` | ✅ 合理 | 等待应用完全加载，避免操作失败 |
| `click` | ✅ 合理 | 切换到 PAST 标签查看历史会议记录 |
| `terminate` | ✅ 必要 | 标记任务完成 |

---

## 8. 关于 `<think>` 标签

**重要：数据集中没有 thought/subgoal 字段。**

- 训练时使用 `force_add_thought=True`
- 模型需要**自己学习生成**推理过程
- 这是 UI-S1 的核心设计：通过 RL 训练学会 self-reasoning

### 模型输出格式
```
<think>
I can see the Zoho Meeting app is now open and showing the UPCOMING tab.
To view scheduled meetings, I should check if there are any meetings listed.
The screen shows "No Upcoming Meetings". Let me check the PAST tab.
</think>
<action>
{"action": "click", "coordinate": [540, 390]}
</action>
```

---

## 9. 训练后处理

### 合并 Checkpoint
```bash
python scripts/model_merger.py merge \
    --backend fsdp \
    --local_dir checkpoints/XXX \
    --target_dir /path/to/merged_model
```

### 启动推理
```bash
vllm serve /path/to/merged_model \
    --served-model-name UI-S1-7B \
    --tensor_parallel_size 1 \
    --limit-mm-per-prompt image=2
```

---

## 10. 关键文件路径

| 文件 | 路径 |
|------|------|
| 训练脚本 | `scripts/train_example.sh` |
| 配置文件 | `examples/qwen_gui_static_grpo/config/traj_grpo.yaml` |
| 数据格式化 | `x/data/agent/json.py` |
| 动作空间定义 | `x/data/agent/space/std_space.py` |
| 训练数据 | `datasets/ui_s1_train.jsonl` |
| 图片目录 | `datasets/AndroidControl/images/` |
