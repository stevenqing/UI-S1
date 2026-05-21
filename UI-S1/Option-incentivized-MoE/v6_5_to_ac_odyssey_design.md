# v6.5 → AndroidControl & GUI-Odyssey 迁移设计

> 目标：把 cooperative LoRA v6.5 配方（GUI-360 SOTA 50.06%）分别迁移到 AndroidControl 和 GUI-Odyssey 两个 mobile benchmark，**为每个 benchmark 单独训练一个 v6.5-style 模型**，验证 cooperative LoRA 配方在跨域上的可迁移性。
>
> **重要约定**：AC 和 Odyssey 不做联合训练。两个 benchmark 各自训练一份独立的 cooperative LoRA ckpt，互不混合。

---

## 0. 快速上下文

| 项 | 现状 |
|---|---|
| v6.5 训练域 | GUI-360 桌面（Word/Excel/PPT），单步、单图 |
| v6.5 配方 | tanh gate, gate_init=0, gate_lr_mult=100, lora_r=256, lora_alpha=512, target_modules=q/k/v/o/gate/up/down, eff_bs=128, lr=1e-5, 4 epoch |
| v6.5 ckpt | `train_GUI_360/llamafactory/output/cooperative_v6_5_comm_thought/epoch-{1..4}` |
| v6.5 ep4 性能 | GUI-360 action_prediction = **50.06%**（cooperative LoRA SOTA） |
| 训练入口 | `train_cooperative.py` + `scripts/exp_cooperative/train_v6_5_comm_thought.slurm` |
| 评测入口 | `evaluation/eval_cooperative_batch.py` + `scripts/exp_cooperative/eval_v6_5_thought_epoch4_ap.slurm` |

---

## 1. 关键技术事实（决定一切方案的硬约束）

### 1.1 CooperativeVLMWrapper 是 prompt-agnostic 的

**`verl/models/cooperative/cooperative_wrapper.py:generate()`** 注册一个 forward_pre_hook：

```python
VISION_START_ID = 151652
VISION_END_ID   = 151653
IMAGE_PAD_ID    = 151655   # ←  V/A token 路由的唯一依据

mask = (input_ids == IMAGE_PAD_ID)   # True = visual agent, False = action agent
self._set_token_mask(mask)
```

**含义**：路由仅依赖 image_pad token 的位置，**与 prompt 模板、动作 schema 完全无关**。任意 mobile / desktop / 任意 chat template 都能跑通。

### 1.2 vLLM 不能复用

vLLM 的 `Qwen2_5_VLForConditionalGeneration` 优化路径绕过 `forward_pre_hook`，cooperative wrapper 的 V/A 路由会失效。AC/Odyssey 现有评测全部基于 vLLM（`call_mobile_agent_vllm`），**必须重写 HF 版本**。

### 1.3 prompt 模板 / 动作 schema 不一致

| 维度 | GUI-360（v6.5 训练） | AC / Odyssey |
|---|---|---|
| 响应格式 | `<thought>...</thought>\n<tool_call>{"function":..,"args":{},"status":"CONTINUE"}</tool_call>` | `<think>...</think>\n<action>{...}</action>` |
| 动作动词 | click/type/drag/wheel_mouse_input/table2markdown/insert_excel_table/select_table_range/set_cell_value/auto_fill/reorder_columns | click/long_press/swipe/type/key/system_button/open/wait/answer/terminate |
| 坐标空间 | 像素 | AC：像素；Odyssey：GT 在 `[0,1000]`，模型输出像素，scorer 内部 `pred_coord_to_1k` |
| 历史 | 单图单步 | AC：adapter-specific；Odyssey：默认 2 图 |
| Prompt 构造 | 硬编码 `ACTION_PREDICTION_USER_PROMPT_QWEN` | `JsonFormat(RAW_SPACE, add_thought=True, force_add_thought=True)` |

**结论**：v6.5 ckpt 直接零样本扔到 AC/Odyssey 上几乎肯定 < 5%（输出格式都不对），必须**重训 v6.5-mobile**。但 wrapper 本身能跑，所以管线是通的。

---

## 2. 整体方案：评测管线 + 两份独立训练

```
路径 A：评测管线（两份共用代码）          路径 B：两份独立训练
─────────────────────────              ─────────────────────────
M1: AC HF 评测器  (1d)         ┌─►   M3a: AC 数据准备 (0.5d)
M2: Odyssey HF 评测器 (0.5d)   │         ↓
       │                       │     M4a: train v6.5-AC (5h)
       │                       │         │
       │                       │         ↓
       │                       │     M5a: AC scaling 表
       │                       │
       │                       └─►   M3b: Odyssey 数据准备 (0.5d)
       │                                 ↓
       │                             M4b: train v6.5-Odyssey (~6h)
       │                                 │
       │                                 ↓
       └──────────────────────────►  M5b: Odyssey scaling 表
```

**核心约定**：
- **不做联合训练**。AC 训出来的 ckpt 只在 AC 上评测；Odyssey 训出来的 ckpt 只在 Odyssey 上评测。
- 两份训练共用 v6.5 配方（lora_r=256, gate_type=tanh, lr_mult=100, eff_bs=128, 4 epoch），唯一差异是训练数据。
- 评测代码两份共用（`cooperative_trajectory_common.py`），数据准备代码两份独立。

**先做路径 A**：即使是零样本（垃圾结果），也能验证全链路 + 建立 baseline floor + 评测器以后还要长期使用。

---

## 3. 路径 A —— HF 版评测器

### 3.1 文件结构

```
evaluation/
├── eval_cooperative_batch.py             # 已有，GUI-360 单步
├── eval_cooperative_ac_trajectory.py     # 新增，AC 多步轨迹
└── eval_cooperative_odyssey_trajectory.py # 新增，Odyssey 多步轨迹

scripts/exp_cooperative/
├── eval_v6_5_thought_epoch4_ap.slurm     # 已有，GUI-360
├── eval_v6_5_ac_epoch4.slurm             # 新增
└── eval_v6_5_odyssey_epoch4.slurm        # 新增
```

### 3.2 `eval_cooperative_ac_trajectory.py` 设计

**参考代码**：
- 模型加载样板：`evaluation/eval_cooperative_batch.py`（cooperative wrapper + ckpt 加载）
- 轨迹外壳：`scripts/eval/eval_ar_trajectory_generic.py`（episode 循环 + 指标累加）
- Message 构造：`x/data/agent/json.py:JsonFormat`（`gen_next_round`）
- 评分函数：`evaluation/qwenvl_utils.py:evaluate_android_control_action`
- 工具：`evaluation/qwenvl_utils.py:find_last_image_ele` + `slim_messages`

**核心循环（伪代码）**：

```python
# 1. Load model (复用 eval_cooperative_batch.py 的 setup_model_and_processor)
model, processor = setup_model_and_processor(args.base_model, args.coop_checkpoint, device)
processor.tokenizer.padding_side = "left"

# 2. Load AC trajectories (从 datasets/android_control_evaluation_std.jsonl)
episodes = load_ac_trajectories(args.eval_data, shard_id=args.shard_id, num_shards=args.num_shards)

# 3. Initialize JsonFormat
fm = JsonFormat(RAW_SPACE_PATH, add_thought=True, force_add_thought=True)

results = []
for ep in episodes:
    state = {"messages": [], "step_idx": 0, "goal": ep["goal"]}
    model_response = None
    step_results = []
    final_step_id = 0

    for step in ep["steps"]:
        # 3.1 推进对话状态（追加图像 + 用户指令）
        state = fm.gen_next_round(ep, state, previous_model_response=model_response)
        messages = slim_messages(state["messages"], num_image_limit=args.n_history_image_limit)
        _, w, h, rw, rh = find_last_image_ele(messages)

        # 3.2 HF generate (替代 call_mobile_agent_vllm)
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        images = process_vision_info(messages)[0]
        inputs = processor(text=[text], images=images, return_tensors="pt", padding=True).to(device)

        with torch.no_grad():
            output_ids = model.generate(**inputs, max_new_tokens=512, do_sample=False)
        input_len = inputs["input_ids"].shape[1]
        model_response = processor.decode(output_ids[0][input_len:], skip_special_tokens=True)

        # 3.3 Parse + score
        parsed = fm.parse_response(model_response)  # → {"think", "action", "action_content"}
        type_match, extract_match = evaluate_android_control_action(
            parsed["action_content"], step["check_options"], w, h, rw, rh,
        )
        step_results.append({"step_num": step["step_num"], "type_match": type_match,
                             "extract_match": extract_match, "pred_action": parsed["action_content"]})
        if extract_match:
            final_step_id += 1
        if not extract_match and not args.no_stop:
            break

    results.append({"episode_id": ep["episode_id"], "goal": ep["goal"],
                    "num_steps": len(ep["steps"]), "task_success": final_step_id == len(ep["steps"]),
                    "final_step_id": final_step_id, "step_results": step_results})

# 4. Save shard results + summary (per-shard)
save_jsonl(f"{args.output_dir}/trajectory_results_shard{args.shard_id}.jsonl", results)
save_json(f"{args.output_dir}/summary_shard{args.shard_id}.json", compute_metrics(results))
```

**CLI 参数**：

```bash
python evaluation/eval_cooperative_ac_trajectory.py \
    --base_model checkpoints/Qwen2.5-VL-7B-Instruct \
    --coop_checkpoint train_GUI_360/llamafactory/output/cooperative_v6_5_comm_thought/epoch-4 \
    --eval_data datasets/android_control_evaluation_std.jsonl \
    --output_dir train_GUI_360/GUI-360-eval/results/cooperative_v6_5_ac_epoch4 \
    --gpu_id 0 --shard_id 0 --num_shards 4 \
    --n_history_image_limit 2 --max_new_tokens 512 \
    --no_stop  # 可选：不在第一个错误处停止
```

### 3.3 `eval_cooperative_odyssey_trajectory.py` 设计

**只与 AC 版有 3 处差异**：

1. **数据源**：`datasets/GUI-Odyssey/gui_odyssey_random_split_test.jsonl`（已经是 AC 兼容格式，由 `gui_odyssey_eval/convert_to_eval_format.py` 生成）
2. **评分函数**：`gui_odyssey_eval/odyssey_action_matching.py:evaluate_odyssey_action`
   - 调用方式：`evaluate_odyssey_action(parsed["action_content"], step["check_options"], rw, rh)`（注意只传 resized 不传 orig，因为 GT 已经在 [0,1000]）
3. **指标 breakdown**：额外按 Odyssey 的 `category` / `device_name` 分桶（参考 `gui_odyssey_eval/eval_ar_trajectory.py` 末尾的统计逻辑）

其他**完全相同**（连 message 构造都一样，因为 `JsonFormat` 是统一的）。建议把公共部分抽到 `evaluation/cooperative_trajectory_common.py`，两个评测器都 import。

### 3.4 Slurm 模板（基于 `eval_v6_5_thought_epoch4_ap.slurm`）

`scripts/exp_cooperative/eval_v6_5_ac_epoch4.slurm`：

```bash
#!/bin/bash
#SBATCH --job-name=v65_ac_e4
#SBATCH --output=/scratch/.../logs/eval_v6_5_ac_epoch4_%j.log
#SBATCH --error=/scratch/.../logs/eval_v6_5_ac_epoch4_%j.err
#SBATCH --time=06:00:00
#SBATCH --nodes=1 --gres=gpu:4 --cpus-per-task=72 --mem=0

PROJECT_DIR="/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1"
BASE_MODEL="checkpoints/Qwen2.5-VL-7B-Instruct"
COOP_CKPT="train_GUI_360/llamafactory/output/cooperative_v6_5_comm_thought/epoch-4"
OUT_DIR="$PROJECT_DIR/train_GUI_360/GUI-360-eval/results/cooperative_v6_5_ac_epoch4"
NUM_SHARDS=4

cd "$PROJECT_DIR"
mkdir -p "$OUT_DIR"

PIDS=()
for shard in $(seq 0 $((NUM_SHARDS-1))); do
    LOG="scripts/exp_cooperative/logs/v6_5_ac_epoch4_shard${shard}_${SLURM_JOB_ID}.log"
    python -u evaluation/eval_cooperative_ac_trajectory.py \
        --base_model "$BASE_MODEL" \
        --coop_checkpoint "$COOP_CKPT" \
        --eval_data datasets/android_control_evaluation_std.jsonl \
        --output_dir "$OUT_DIR" \
        --gpu_id $shard --shard_id $shard --num_shards $NUM_SHARDS \
        --n_history_image_limit 2 \
        > "$LOG" 2>&1 &
    PIDS+=($!)
done
for pid in "${PIDS[@]}"; do wait $pid; done

# Aggregate
python - << PYEOF
import json, glob, os
out_dir = "$OUT_DIR"
shards = sorted(glob.glob(f"{out_dir}/summary_shard*.json"))
total_eps, total_succ = 0, 0
total_steps, total_correct = 0, 0
for s in shards:
    d = json.load(open(s))
    total_eps += d["num_episodes"]
    total_succ += d["num_success"]
    total_steps += d["total_steps"]
    total_correct += d["total_correct_steps"]
print(f"=== AC ep4 AGGREGATE: TSR={100*total_succ/total_eps:.2f}%  step_acc={100*total_correct/total_steps:.2f}% ===")
PYEOF
```

Odyssey 版几乎一字不差，只换 `--eval_data` 和 `--output_dir`。

### 3.5 路径 A 验收标准

- [ ] AC 零样本 v6.5 ep4 跑完 1 个 shard，无 OOM/格式异常
- [ ] Odyssey 零样本 v6.5 ep4 跑完 1 个 shard，无异常
- [ ] aggregate 脚本能算出 TSR + step_accuracy
- [ ] **零样本数字记录到 baseline floor**，预期 < 5%

---

## 4. 路径 B —— 两份独立训练

> **核心原则**：AC 和 Odyssey 各训一份，**不混合、不联合**。两份训练完全独立，使用相同的 v6.5 配方。

### 4.1 训练数据格式（共用）

两份数据都转成 v6.5 训练所需的 `gui360_train_thought.jsonl` 格式：

```json
{
  "conversations": [
    {"from": "human", "value": "<image>\n{system_prompt}\n\nUser Instruction: ..."},
    {"from": "gpt",   "value": "<think>...</think>\n<action>{\"action\":\"click\",\"coordinate\":[x,y]}</action>"}
  ],
  "images": ["/abs/path/to/screenshot.png"],
  "has_thought": true,
  "gt_coords": [x, y]
}
```

### 4.2 AC 数据准备 — `datasets/cooperative_thought_ac/prepare_ac_thought.py`

- **输入**：`datasets/ui_s1_dataset/ui_s1_train_with_desc.jsonl`（1k 轨迹，带 `desc_t1` 屏幕描述可作为 thought 来源）
- **prompt 渲染**：用 `JsonFormat(RAW_SPACE_PATH, add_thought=True, force_add_thought=True).gen_next_round` 渲染每个 step
- **thought 来源**：优先用 `desc_t1`；缺失时用启发式 `f"I need to {step_instruction}"`
- **拆分**：每个 step 一个独立样本（与 v6.5 单步训练一致），**不做多轮历史**
- **坐标**：原始像素，无转换
- **输出**：
  - `datasets/cooperative_thought_ac/ac_train_thought.jsonl`
  - `datasets/cooperative_thought_ac/ac_val_thought.jsonl`（90/10 split）
- **估计样本量**：1k 轨迹 × 平均 6 step ≈ **6k 训练步样本**

### 4.3 Odyssey 数据准备 — `datasets/cooperative_thought_odyssey/prepare_odyssey_thought.py`

- **输入**：`datasets/GUI-Odyssey/all_annot.json` + `datasets/GUI-Odyssey/screenshots/`
- **schema 转换**：复用 `gui_odyssey_eval/convert_to_eval_format.py:convert_action()` 把原始 Odyssey schema (CLICK/SCROLL/TEXT/...) 转成 `RAW_SPACE` 动作 (click/swipe/type/...)
- **prompt 渲染**：与 AC 相同，用 `JsonFormat.gen_next_round`
- **thought 来源**：用每步的 `intention` 字段
- **坐标**：**保留原始截图分辨率的像素**（评测时模型输出像素，scorer 内部做 `pred_coord_to_1k` → [0,1000] 与 GT 比对）。**不要**提前转成 [0,1000]
- **训练/验证切分**：用 `datasets/GUI-Odyssey/splits/random_split_train.json` / `random_split_val.json`
- **输出**：
  - `datasets/cooperative_thought_odyssey/odyssey_train_thought.jsonl`
  - `datasets/cooperative_thought_odyssey/odyssey_val_thought.jsonl`
- **估计样本量**：~7.7k 轨迹 × 平均 ~9 step ≈ **70k 训练步样本**

### 4.4 AC 训练 slurm — `scripts/exp_cooperative/train_v6_5_ac_comm_thought.slurm`

**完全照搬** `train_v6_5_comm_thought.slurm` 的所有超参，仅改动：

```diff
- #SBATCH --job-name=coop_v65t
+ #SBATCH --job-name=coop_v65ac

- OUTPUT_DIR="train_GUI_360/llamafactory/output/cooperative_v6_5_comm_thought"
+ OUTPUT_DIR="train_GUI_360/llamafactory/output/cooperative_v6_5_ac_comm_thought"

-     --train_data datasets/cooperative_thought/gui360_train_thought.jsonl \
-     --val_data datasets/cooperative_thought/gui360_val_thought.jsonl \
+     --train_data datasets/cooperative_thought_ac/ac_train_thought.jsonl \
+     --val_data datasets/cooperative_thought_ac/ac_val_thought.jsonl \
```

**估计训练时长**：6k 样本 × 4 epoch / eff_bs 128 = ~190 步，约 **40 分钟 wall**（远短于 GUI-360 的 ~3050 步）。
**风险**：训练步数太少可能不足以让 cooperative gate 充分学习。如果 gate 还在 < 0.05 量级（v6.5 GUI-360 ep4 是 0.1-0.13），考虑：
- 选项 1：增加到 8 epoch（仍然只有 ~80min）
- 选项 2：把 `gate_lr_multiplier` 从 100 提高到 200

### 4.5 Odyssey 训练 slurm — `scripts/exp_cooperative/train_v6_5_odyssey_comm_thought.slurm`

同样照搬 `train_v6_5_comm_thought.slurm`，仅改动：

```diff
- #SBATCH --job-name=coop_v65t
+ #SBATCH --job-name=coop_v65od

- OUTPUT_DIR="train_GUI_360/llamafactory/output/cooperative_v6_5_comm_thought"
+ OUTPUT_DIR="train_GUI_360/llamafactory/output/cooperative_v6_5_odyssey_comm_thought"

-     --train_data datasets/cooperative_thought/gui360_train_thought.jsonl \
-     --val_data datasets/cooperative_thought/gui360_val_thought.jsonl \
+     --train_data datasets/cooperative_thought_odyssey/odyssey_train_thought.jsonl \
+     --val_data datasets/cooperative_thought_odyssey/odyssey_val_thought.jsonl \
```

**估计训练时长**：70k 样本 × 4 epoch / eff_bs 128 = ~2200 步，约 **3.5h wall**（与 v6.5 GUI-360 ~3050 步、~5h 同量级）。

### 4.6 配方一致性

**两份训练共用 100% 相同的 cooperative LoRA 超参**：
- `lora_r=256, lora_alpha=512, lora_dropout=0.05`
- `target_modules=q_proj k_proj v_proj o_proj gate_proj up_proj down_proj`
- `num_agents=2, cooperative_comm`
- `gate_type=tanh, gate_init=0.0, gate_lr_multiplier=100.0, gate_weight_decay=0.0`
- `bind_weight=0.0, per_device_batch_size=1, gradient_accumulation_steps=4`
- `learning_rate=1e-5, num_epochs=4.0, warmup_ratio=0.03, max_length=4096`

**只有训练数据不同**。这样得到的是"v6.5 配方在 AC 域 / Odyssey 域上的复刻"，三个数字（GUI-360 v6.5 / AC v6.5-AC / Odyssey v6.5-Odyssey）可以横向对比验证配方的跨域可迁移性。

### 4.7 验收标准

**AC 训练**：
- [ ] 4 个 epoch 全部产出 ckpt
- [ ] 每个 epoch ckpt 用 §3 AC 评测器跑出 TSR + step_acc
- [ ] 至少有 1 个 epoch TSR > 30%（对照 OS-Atlas/UI-TARS baseline 水平）
- [ ] gate 幅度 (`|tanh(gate)|`) 与 v6.5 GUI-360 同量级（0.05-0.5），未饱和（< 0.8）

**Odyssey 训练**：
- [ ] 4 个 epoch 全部产出 ckpt
- [ ] 每个 epoch ckpt 用 §3 Odyssey 评测器跑出 TSR + step_acc
- [ ] 至少有 1 个 epoch TSR > 25%（Odyssey 比 AC 难）
- [ ] gate 幅度同上

---

## 5. 训练数据格式细节（关键陷阱）

### 5.1 v6.5 训练数据是单步、单图

`gui360_train_thought.jsonl` 一个样本一张图片，模型只见过单步推理。**不要尝试在训练数据里塞多图历史**——会破坏 cooperative wrapper 的 token 路由假设（多图情况下 V-stream 太长会压垮 A-stream），同时会让 prompt 长度爆炸。

如果担心多步评测时的分布偏移，可以在评测器里默认 `n_history_image_limit=1`（只看当前图），与训练对齐；测得 baseline 后再实验放宽到 2。

### 5.2 prompt 模板必须用 `JsonFormat` 生成

数据准备脚本里**不要**手写 prompt 字符串，必须调用：

```python
from x.data.agent.json import JsonFormat
from x.data.agent.space.std_space import RAW_SPACE_PATH

fm = JsonFormat(RAW_SPACE_PATH, add_thought=True, force_add_thought=True)
state = {"messages": [], "step_idx": 0, "goal": ep["goal"]}
state = fm.gen_next_round(ep, state, previous_model_response=None)
# state["messages"] 就是标准的 (system, user, image) 三段
```

然后从 `state["messages"]` 里抽出 `system + user_text + image_path` 拼成 `conversations` 字段。**评测时的 message 构造也是同一份代码**，保证训练/评测分布一致。

### 5.3 坐标空间

- **AC**：原始像素，**不需要任何转换**（训练用什么像素，评测就用什么像素）
- **Odyssey**：训练数据用**原始截图分辨率的像素**（不是 [0,1000]！）。原因：模型在 Qwen2.5-VL 处理器内部会被 resize，实际看到的是 resize 后的图，输出是 resize 后像素空间的坐标。评测时 scorer 会用 `pred_coord_to_1k(coord, resized_w, resized_h)` 把模型输出转回 [0,1000] 与 GT 比对。如果训练时把坐标提前转成 [0,1000]，那么模型学到的输出空间和评测时调用 `pred_coord_to_1k` 的语义就对不上了。

### 5.4 thought 字段的来源策略

| 数据集 | 推荐 thought 来源 | 备选 |
|---|---|---|
| AC | `ui_s1_train_with_desc.jsonl` 的 `desc_t1` | 启发式 `f"I need to {step_instruction}"` |
| Odyssey | annotation 里的 `intention` 字段 | 启发式 `f"To {goal}, I should {action_verb} ..."` |

**不建议**用 LLM 重新蒸馏 thought——成本高，而且 v6.5 GUI-360 训练数据的 thought 也不是高质量蒸馏出来的，保持一致即可。

---

## 6. 实施时间线

| 里程碑 | 工作量 | 产出 | 依赖 |
|---|---|---|---|
| **M1** AC HF 评测器 | 1d | `eval_cooperative_ac_trajectory.py`, slurm, 零样本 baseline | — |
| **M2** Odyssey HF 评测器 | 0.5d | `eval_cooperative_odyssey_trajectory.py`, slurm, 零样本 baseline | M1（共用代码） |
| **M3a** AC 数据准备 | 0.5d | `prepare_ac_thought.py` + `ac_train/val_thought.jsonl` | — |
| **M3b** Odyssey 数据准备 | 0.5d | `prepare_odyssey_thought.py` + `odyssey_train/val_thought.jsonl` | — |
| **M4a** v6.5-AC 训练 | ~40min wall（8 nodes × 4 GPU） | AC ckpt epoch 1-4 | M3a |
| **M4b** v6.5-Odyssey 训练 | ~3.5h wall（8 nodes × 4 GPU） | Odyssey ckpt epoch 1-4 | M3b |
| **M5a** AC scaling 评测 | 0.5d | v6.5-AC epoch 1-4 在 AC 上的完整表 | M1, M4a |
| **M5b** Odyssey scaling 评测 | 0.5d | v6.5-Odyssey epoch 1-4 在 Odyssey 上的完整表 | M2, M4b |

**并行依赖**：
- M1, M2, M3a, M3b 完全独立，全部并行。
- M4a 等 M3a；M4b 等 M3b。两份训练可以同时跑（如果 slurm 配额够），也可以串行。
- M5a 等 M1+M4a；M5b 等 M2+M4b。

---

## 7. 风险与回退

| 风险 | 影响 | 缓解 |
|---|---|---|
| 零样本评测器跑通但数字 = 0% | 无法验证管线是否真的工作 | M1 同时跑一遍 base Qwen2.5-VL（不加 cooperative wrapper）作为 sanity check，预期数字接近 OS-Atlas baseline |
| AC 训练步数太少（~190 步），cooperative gate 学不充分 | AC 上 cooperative wrapper 退化为普通 LoRA，效果与 vanilla LoRA 持平 | 检查训完 ckpt 的 `\|tanh(gate)\|`，如果 < 0.05 就把 epoch 数提高到 8 或把 `gate_lr_multiplier` 提高到 200 |
| Odyssey 训练数据规模 70k 但视觉域单一（Pixel Tablet），泛化差 | Odyssey TSR 不稳定 | 配方上没有干预空间，如果出问题就承认 Odyssey 是更难的 benchmark |
| `JsonFormat` 在训练数据里渲染出的 prompt 与评测时不一致 | 训练分布偏移 | 训练数据准备脚本和评测器**共用**同一个 `JsonFormat` 实例，从源头消除偏移 |
| v6.5 配方（gate_lr_mult=100）在 mobile 域上 gate 饱和 | 主 loss 退化 | M4 训完检查 `\|tanh(gate)\|` mean，如果 > 0.6 就把 lr_mult 降回 50（v6.4 的值）重训 |
| 多图历史导致 cooperative wrapper 路由失效 | 评测崩盘 | 评测时默认 `n_history_image_limit=1`，与训练对齐；2 图作为 ablation |

---

## 8. 关键文件索引

### 现有文件（参考）

| 文件 | 用途 |
|---|---|
| `Option-incentivized-MoE/cooperative_lora_v6_per_layer_communication.md` | v6.5 完整设计与实验记录 |
| `verl/models/cooperative/cooperative_wrapper.py` | CooperativeVLMWrapper, generate(), forward_pre_hook |
| `verl/models/cooperative/cooperative_lora.py` | CooperativeLoRALinear |
| `train_cooperative.py` | v6.5 训练脚本 |
| `evaluation/eval_cooperative_batch.py` | GUI-360 HF 评测器（模型加载样板） |
| `scripts/exp_cooperative/train_v6_5_comm_thought.slurm` | v6.5 训练 slurm |
| `scripts/exp_cooperative/eval_v6_5_thought_epoch4_ap.slurm` | v6.5 GUI-360 评测 slurm |
| `x/data/agent/json.py` | `JsonFormat`, `MOBILE_USE` prompt, `gen_next_round`, `parse_response` |
| `x/data/agent/space/std_space.py` | `RAW_SPACE` 动作 schema |
| `evaluation/qwenvl_utils.py` | `evaluate_android_control_action`, `find_last_image_ele`, `slim_messages`, `call_mobile_agent_vllm` |
| `gui_odyssey_eval/odyssey_action_matching.py` | `evaluate_odyssey_action`, `pred_coord_to_1k` |
| `gui_odyssey_eval/eval_ar_trajectory.py` | Odyssey vLLM 评测器（轨迹外壳参考） |
| `gui_odyssey_eval/convert_to_eval_format.py` | `convert_action()` Odyssey → AC schema |
| `scripts/eval/eval_ar_trajectory_generic.py` | AC vLLM 评测器（轨迹外壳参考） |
| `datasets/android_control_evaluation_std.jsonl` | AC 评测数据 |
| `datasets/GUI-Odyssey/gui_odyssey_random_split_test.jsonl` | Odyssey 评测数据 |
| `datasets/ui_s1_dataset/ui_s1_train_with_desc.jsonl` | 1k AC 训练轨迹（带 desc_t1） |
| `datasets/GUI-Odyssey/all_annot.json` | Odyssey 全量 annotation |
| `datasets/cooperative_thought/gui360_train_thought.jsonl` | v6.5 训练数据（格式参考） |

### 新增文件

| 文件 | 阶段 |
|---|---|
| `evaluation/cooperative_trajectory_common.py` | M1（AC/Odyssey 共用：模型加载、message 构造、generate） |
| `evaluation/eval_cooperative_ac_trajectory.py` | M1 |
| `evaluation/eval_cooperative_odyssey_trajectory.py` | M2 |
| `scripts/exp_cooperative/eval_v6_5_ac_epoch4.slurm` | M1（零样本 baseline） |
| `scripts/exp_cooperative/eval_v6_5_odyssey_epoch4.slurm` | M2（零样本 baseline） |
| `datasets/cooperative_thought_ac/prepare_ac_thought.py` | M3a |
| `datasets/cooperative_thought_ac/ac_train_thought.jsonl` | M3a 产物 |
| `datasets/cooperative_thought_ac/ac_val_thought.jsonl` | M3a 产物 |
| `datasets/cooperative_thought_odyssey/prepare_odyssey_thought.py` | M3b |
| `datasets/cooperative_thought_odyssey/odyssey_train_thought.jsonl` | M3b 产物 |
| `datasets/cooperative_thought_odyssey/odyssey_val_thought.jsonl` | M3b 产物 |
| `scripts/exp_cooperative/train_v6_5_ac_comm_thought.slurm` | M4a |
| `scripts/exp_cooperative/train_v6_5_odyssey_comm_thought.slurm` | M4b |
| `scripts/exp_cooperative/eval_v6_5_ac_epoch{1..4}.slurm` | M5a |
| `scripts/exp_cooperative/eval_v6_5_odyssey_epoch{1..4}.slurm` | M5b |

---

## 9. 待确认事项

1. **thought 来源**：AC 是否直接用 `desc_t1`？Odyssey 是否直接用 `intention`？还是要重新蒸馏？
   - 推荐：直接用，与 v6.5 GUI-360 thought 质量保持一致
2. **多图历史**：训练保持单步单图（与 v6.5 一致），评测的 `n_history_image_limit` 默认设 1 还是 2？
   - 推荐：默认 1（与训练对齐），2 作为 ablation
3. **AC 训练步数太短**（仅 ~190 步）：要不要默认就开 8 epoch？
   - 推荐：先 4 epoch 看 gate 学习情况，不够再加
4. **是否要先做 zero-shot 测试**？
   - 强烈推荐：M1/M2 完成后立刻零样本跑 v6.5 GUI-360 ep4，建立 floor + 验证管线

---

## 10. TL;DR

1. **wrapper 是 prompt-agnostic 的**，v6.5 ckpt 技术上能跑 AC/Odyssey，但因为分布完全不同，零样本几乎肯定崩盘。
2. **vLLM 不能用，必须写 HF 版评测器**——AC/Odyssey 共用一份 `cooperative_trajectory_common.py`。
3. **AC 和 Odyssey 各自独立训练 v6.5-style 模型**，**不联合、不混合**：
   - `cooperative_v6_5_ac_comm_thought` ← `ac_train_thought.jsonl`（~6k 步样本，~40min 训练）
   - `cooperative_v6_5_odyssey_comm_thought` ← `odyssey_train_thought.jsonl`（~70k 步样本，~3.5h 训练）
4. **两份训练共用 100% 相同的 v6.5 配方**（lora_r=256, gate_type=tanh, lr_mult=100, eff_bs=128, 4 epoch），唯一差异是训练数据。这样产出的 (GUI-360 v6.5 / AC v6.5-AC / Odyssey v6.5-Odyssey) 三个数字可以横向对比验证配方的跨域可迁移性。
5. **数据准备的关键**：用 `JsonFormat` 渲染 prompt（与评测时同源），坐标保持原始像素空间（不要提前 to_1k）。
6. **分 7 个里程碑推进**：M1+M2+M3a+M3b 完全并行，M4a/M4b 独立训练，M5a/M5b 独立评测。
