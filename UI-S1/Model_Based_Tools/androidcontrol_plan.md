# AndroidControl 评估全流程计划

> 将 GUI-360 的完整诊断 pipeline（Eval A-E, Pre-tests）迁移到 AndroidControl 数据集。
> 与 GUI-360 plan 的核心区别：**单模型**（非 V2+V3 双模型），**移动端 action space**（click/swipe/type/open/system_button/wait/long_press），**无 domain 划分**。

---

## 第一部分：环境与数据

### 1.1 数据集

| 项目 | 值 |
|------|-----|
| 数据集 | `datasets/android_control_evaluation_std.jsonl` |
| Episodes | 1,543 |
| Steps | 8,444 |
| 图片根目录 | `datasets/AndroidControl/images/` |
| 图片路径前缀 | `/datasets/` → 需替换为绝对路径 |

数据格式（每行一个 episode）：
```json
{
  "goal": "I'd like to publish my sculpture art from the gallery.",
  "is_successful": true,
  "episode_id": 8193,
  "steps": [
    {
      "action_content": {"action": "system_button", "button": "Back"},
      "screenshot": "/datasets/AndroidControl/images/8193_0.png",
      "step_instruction": "go back",
      "check_options": {"action": "system_button", "button": "Back"}
    },
    {
      "action_content": {"action": "click", "coordinate": [540, 2273]},
      "screenshot": "/datasets/AndroidControl/images/8193_1.png",
      "check_options": {"action": "click", "candidate_bbox": [[456,2211,624,2337]], "coordinate": [540,2273]}
    }
  ]
}
```

### 1.2 Action Space

| Action | Arguments | 类型 | 评估方式 |
|--------|-----------|------|---------|
| click | coordinate | coord | bbox+point distance |
| long_press | coordinate, time | coord | bbox+point distance |
| swipe | coordinate, coordinate2 | coord | 方向匹配（当前 always True） |
| type | text | non-coord | 文本子串匹配 |
| open | text | non-coord | 文本匹配 / bbox fallback |
| system_button | button | non-coord | 精确匹配 (Back/Home/Menu/Enter) |
| wait | time | non-coord | 类型匹配即可 |

与 GUI-360 的区别：
- 无 `wheel_mouse_input`、`drag`、`select_text`、`select_table_range` 等 Office 特有 action
- 无 domain 划分（Word/Excel/PPT）→ 按 **action type** 和 **trajectory length** 做 breakdown
- 单模型评估，无 V2+V3 双模型对比

### 1.3 复用组件

| 组件 | 来源 | 用途 |
|------|------|------|
| `evaluate_android_control_action()` | `evaluation/qwenvl_utils.py` | Step-level 评估（bbox 坐标匹配） |
| `JsonFormat` | `x/data/agent/json.py` | Prompt 构造 + response 解析 |
| `RAW_SPACE` | `x/data/agent/space/std_space.py` | Action space 定义 |
| `slim_messages` | `x/qwen/data_format.py` | History 图片截断 |
| `smart_resize` | `x/qwen/image.py` | 图片预处理 |
| `call_mobile_agent_vllm` | `evaluation/qwenvl_utils.py` | vLLM API 调用 |
| SLURM 模板 | `evaluation/slurm_scripts/eval_qwen25vl.slurm` | Job 配置 |

### 1.4 可用模型

- Qwen2.5-VL-7B（默认）：`checkpoints/Qwen2.5-VL-7B-Instruct`
- UI-TARS-7B、OS-Atlas-7B 等（可配置）

---

## 第二部分：脚本总览

所有脚本位于 `scripts/eval/ac/`，共 22 个文件（18 Python + 4 SLURM + logs 目录）。

```
scripts/eval/ac/
├── ac_utils.py                      # 共享工具函数
├── eval_a_ar_trajectory.py          # [GPU] Core AR trajectory eval
├── eval_a_ar_trajectory.slurm
├── eval_b_step_analysis.py          # [Offline] Step-position accuracy
├── eval_c_hard_cases.py             # [Offline] Hard cases identification
├── eval_c2_action_type_analysis.py  # [Offline] Per-action-type accuracy
├── eval_c4c7_multisample.py         # [GPU] Multi-sample (K=10)
├── eval_c4c7_multisample.slurm
├── eval_c4c7_analysis.py            # [Offline] Multi-sample analysis
├── eval_c8_verifier_trigger.py      # [Offline] Silent failure / verification
├── eval_d0_error_ceiling.py         # [Offline] Error type ceiling
├── eval_d1_observer_ar.py           # [GPU] Observer AR evaluation
├── eval_d1_observer_ar.slurm
├── eval_d2_prompted_observer.py     # [GPU] Prompted Observer
├── eval_d2_prompted_observer.slurm
├── eval_d4_d6_d7_offline.py         # [Offline] Planner ceiling + failure types + length
├── eval_d8_info_transfer.py         # [GPU] Observer info transfer ablation
├── eval_d8_info_transfer.slurm
├── eval_d9_critic_zeroshot.py       # [GPU] Critic zero-shot
├── eval_d9_critic_zeroshot.slurm
├── eval_d10_action_diagnosis.py     # [Offline] Action confusion matrix
├── eval_e2_observer_length.py       # [Offline] Observer value by length
├── eval_pretest1_shapley.py         # [Offline] Credit analysis
├── eval_pretest3_counterfactual.py  # [Offline] Counterfactual oracle fix
└── logs/
```

### 共享工具：`ac_utils.py`

核心函数：

| 函数 | 说明 |
|------|------|
| `load_ac_trajectories(jsonl_path, image_root, max_episodes)` | 加载数据、修复图片路径、确保 check_options |
| `fix_line(line)` | 确保每步有 check_options（兼容 eval_qwenvl.py） |
| `categorize_action(action_type)` | 分类为 coord（click/long_press）vs non_coord |
| `length_bucket(n)` | 长度分桶：short(1-3), medium(4-7), long(8-15), vlong(16+) |
| `init_format()` | 初始化 JsonFormat（RAW_SPACE, add_thought=True） |
| `compute_trajectory_metrics(results)` | 计算 TSR, avg_progress, scattered_progress |
| `save_jsonl / load_jsonl / save_json` | IO 工具 |

---

## 第三部分：实验详述

### Eval A：AR Trajectory Evaluation [GPU]

**脚本**：`eval_a_ar_trajectory.py` + `.slurm`
**输出**：`outputs/eval_a_ac/{MODEL_NAME}/trajectory_results.jsonl` + `summary.json`

**流程**：
1. 逐 episode 自动回归评估（stop-on-error）
2. 每步使用 `JsonFormat.gen_next_round()` 构建 messages
3. `slim_messages()` 限制历史图片数量
4. `call_mobile_agent_vllm()` 调用 vLLM 推理
5. `evaluate_android_control_action()` 评估 type_match / extract_match
6. 保存详细 per-step 结果（pred_action, gt_action, step_num 等）

**与 GUI-360 Eval A 的区别**：
- **单模型**，无 V2/V3 双模型对比（无 condition A/B）
- 使用 `evaluate_android_control_action()` 而非 `compare_actions()`
- Breakdown 按 action type（7 类）和 trajectory length bucket，无 domain

**输出 schema**：
```json
{
  "episode_id": 8193,
  "goal": "...",
  "num_steps": 11,
  "task_success": false,
  "final_step_id": 5,
  "length_bucket": "long(8-15)",
  "step_results": [
    {
      "step_num": 0,
      "type_match": true,
      "extract_match": true,
      "pred_action": {"action": "system_button", "button": "Back"},
      "gt_action": {"action": "system_button", "button": "Back"},
      "gt_action_type": "system_button"
    }
  ]
}
```

**summary.json 包含**：TSR, avg_progress, scattered_progress, action_type_stats, length_bucket_stats

**SLURM**：1 node, 4 GPU, TP=4, vLLM + 4 worker threads, ~24h walltime

---

### Eval B：Step-Position Accuracy [Offline]

**脚本**：`eval_b_step_analysis.py`
**输入**：Eval A `trajectory_results.jsonl`
**输出**：`outputs/eval_b_ac/eval_b_step_analysis.json`

**分析内容**：
- Per-step-position accuracy（step 0, 1, 2, ..., 10+）
- type_match 和 extract_match 分别按 step position
- 准确率曲线：是否 step 0 最简单（GUI-360 结论）？
- Length-conditioned step accuracy（每个 length bucket 内的 step curve）

**对应 GUI-360 Eval B 的问题**：
- GUI-360 发现 step 0 accuracy = 86.6%（最高），late steps 降至 ~70%
- AndroidControl 是否有同样趋势？→ 决定 RL 训练的 step amplifier 设计

---

### Eval C：Hard Cases [Offline]

**脚本**：`eval_c_hard_cases.py`
**输入**：Eval A `trajectory_results.jsonl`
**输出**：`outputs/eval_c_ac/eval_c_hard_cases.json`

**分析内容**：
- 失败步骤的 step position 分布
- 失败步骤的 action type 分布
- 整体 fail rate
- 每个 failed episode 的平均失败步数

**与 GUI-360 Eval C 的区别**：
- 无 V2/V3 "both-wrong" 概念（单模型）
- 无 domain 分析，改为 action type 分析
- 无 K-sample agreement 分析（需 C4+C7 数据）

---

### Eval C2：Per-Action-Type Analysis [Offline]

**脚本**：`eval_c2_action_type_analysis.py`
**输入**：Eval A `trajectory_results.jsonl`
**输出**：`outputs/eval_c2_ac/eval_c2_action_type.json`

**分析内容**（替代 GUI-360 的 C2+C1 和 domain 分析）：
- Per-action-type: type_match rate, extract_match rate
- 7×7 action confusion matrix（GT vs Predicted）
- Cross: action_type × step_position
- Cross: action_type × trajectory_length

**关键问题**：
- click/type/swipe 的准确率差异？
- 最常见的 action 混淆对是什么？（GUI-360 是 click↔type 55%）

---

### Eval C4+C7：Multi-Sample Grounding [GPU + Offline]

**GPU 脚本**：`eval_c4c7_multisample.py` + `.slurm`
**Offline 脚本**：`eval_c4c7_analysis.py`
**输出**：`outputs/eval_c4c7_ac/{MODEL_NAME}/`

**GPU 阶段**（数据收集）：
- 每步采样 K=10 次（第一次 greedy，其余 temperature=1.0）
- 解析每次的 pred_action，记录 type_match / extract_match
- 保存 `multisample_results.jsonl`

**Offline 阶段**：
- **Agreement rate 分析**：K 个样本中最多数 action type 的占比
- **Oracle accuracy**：best-of-K（至少一个正确）
- **Coordinate clustering**：coord-based actions 的 DBSCAN 聚类 / spread 统计
- **Adaptive K 策略**：不同 agreement threshold 下的 K=1/K=5 切换效果
- **Selector ceiling**：理想 selector 能达到的上限

**对应 GUI-360 C4+C7 结论**：
- GUI-360: Adaptive v1 (K=1 if agree≥0.9, else K=5) 以 Avg K=2.0 达到 oracle 82%
- AndroidControl 的 oracle headroom 有多大？

---

### Eval C8：Verifier Trigger Analysis [Offline]

**脚本**：`eval_c8_verifier_trigger.py`
**输入**：Eval A `trajectory_results.jsonl`
**输出**：`outputs/eval_c8_ac/eval_c8_verifier.json`

**分析内容**：
- **Silent failure rate**：步骤"正确"但轨迹失败的比例
- **First error step distribution**：失败轨迹的首个错误在哪一步
- **Near-miss analysis**：progress ≥ 50% 的失败轨迹（差一点就成功）
- **Perfect verification ceiling**：理想 verify + recovery 能达到的 TSR
- **Per-length bucket 分析**

**对应 GUI-360 C8 结论**：
- GUI-360: silent failure 36.1%, perfect verify ceiling = 43.9% (+15.1pp)
- AndroidControl 的 verification 价值多大？

---

### Eval D0：Error Type Ceiling [Offline]

**脚本**：`eval_d0_error_ceiling.py`
**输入**：Eval A `trajectory_results.jsonl`
**输出**：`outputs/eval_d0_ac/eval_d0_error_ceiling.json`

**错误类型定义**（单模型版本）：
- **Type A: Grounding error** — type_match=True, extract_match=False（对的 action，错的目标）
- **Type B: Action error** — type_match=False（预测了错误的 action type）

**分析内容**：
- 错误类型分布
- 错误类型 × step position
- 错误类型 × trajectory length
- Repeated action detection（连续相同 action）
- Observer ceiling estimation（conservative: fix 50% grounding, optimistic: fix 100%）

**对应 GUI-360 D0 结论**：
- GUI-360: state confusion = 38.9%, observer ceiling +8.8pp (conservative)
- AndroidControl 中 grounding vs action 错误的比例？

---

### Eval D1：Zero-shot Observer AR [GPU]

**脚本**：`eval_d1_observer_ar.py` + `.slurm`
**输出**：`outputs/eval_d1_ac/{MODEL_NAME}/observer_results.jsonl` + `summary.json`

**架构**（与 GUI-360 相同概念）：
1. **Observer**（同一模型，额外推理调用）读截图 → 生成状态描述（2-4 句）
2. **State document** 累积每步的 observation（保留最近 5 步）
3. **Executor** 读 state document + 截图 → 预测 action
4. 与 Eval A baseline（无 observer）对比

**Observer Prompt**：
```
You are a mobile screen observer. Look at the current screenshot and describe what you see.
Focus on:
1. What app is currently open
2. What screen/page is shown
3. Key UI elements visible
4. Any changes from the previous state
5. Current progress toward the task goal
```

**对应 GUI-360 D1 结论**：
- GUI-360: zero-shot Observer +1.34pp TSR, win ratio 2.3:1
- AndroidControl 的 Observer 效果？

---

### Eval D2：Prompted Observer [GPU]

**脚本**：`eval_d2_prompted_observer.py` + `.slurm`
**输出**：`outputs/eval_d2_ac/{MODEL_NAME}/`

**与 D1 的区别**：结构化 observer prompt，mobile-specific：
```
APP: [Name of the currently open app]
SCREEN: [Current screen/page description]
UI_ELEMENTS: [List of key interactive elements visible]
STATE_CHANGE: [What changed from the previous step]
PROGRESS: [Assessment of progress toward the task goal]
```

**对应 GUI-360 D2 结论**：
- GUI-360: prompted observer -0.9pp vs D1 → prompt engineering 天花板已到
- AndroidControl 是否同样结论？

---

### Eval D4/D6/D7：Offline Failure Analysis [Offline]

**脚本**：`eval_d4_d6_d7_offline.py`
**输入**：Eval A (required) + D1 (optional)
**输出**：`outputs/eval_d4d6d7_ac/eval_d4_d6_d7.json`

**D4 (Planner ceiling)**：如果有 D1 数据，分析 Observer win vs loss
- Observer 成功但 baseline 失败 → Observer 价值
- Observer 失败但 baseline 成功 → Observer 损害
- 净增益估计

**D6 (Failure type classification)**：
- 所有失败轨迹按 first error 分类：A_grounding / B_action
- 占比分析

**D7 (Length × failure)**：
- 交叉表：trajectory length × failure type
- 关键问题：grounding error 是否随长度递增？（GUI-360: 33%→49%）

---

### Eval D8：Observer Info Transfer Ablation [GPU]

**脚本**：`eval_d8_info_transfer.py` + `.slurm`
**输出**：`outputs/eval_d8_ac/{MODEL_NAME}/`

**三个条件**：
- **C**: No observer（= Eval A baseline）
- **B**: Current-step observer only（无历史累积）
- **A**: Full state document（= D1，历史累积）

**对应 GUI-360 D8 结论**：
- GUI-360: **History = 75% of Observer value**，current-only 几乎无用
- AndroidControl 是否同样——历史累积才是有效部分？

---

### Eval D9：Critic Zero-shot [GPU]

**脚本**：`eval_d9_critic_zeroshot.py` + `.slurm`
**输出**：`outputs/eval_d9_ac/{MODEL_NAME}/`

**方法**：每步 action 后，额外调用模型评估"该步骤是否正确"
- Critic prompt：看截图 + action → 输出 PASS 或 FAIL
- 用 GT extract_match 作为真实标签
- 计算 FAIL detection 的 precision / recall / F1
- Per-action-type critic 性能
- PASS bias 分析

**对应 GUI-360 D9 结论**：
- GUI-360: zero-shot Critic 完全无用（precision 9.8%, recall 4.8%, PASS bias 94.9%）
- AndroidControl 是否同样？

---

### Eval D10：Action Confusion Matrix [Offline]

**脚本**：`eval_d10_action_diagnosis.py`
**输入**：Eval A `trajectory_results.jsonl`
**输出**：`outputs/eval_d10_ac/eval_d10_diagnosis.json`

**分析内容**：
- 完整 7×7 action type confusion matrix（click/type/swipe/open/system_button/wait/long_press）
- Top confusion pairs 排序
- Near-miss vs complete-miss 分类
  - Near-miss: type_match=True, extract_match=False（对的 action 类型，错的参数）
  - Complete-miss: type_match=False（连 action 类型都错）
- Error distribution by step position
- Per-action-type accuracy

**对应 GUI-360 D10 结论**：
- GUI-360: click↔type = 55% of action errors, step 1 = 51.6%
- AndroidControl 的主要混淆对是什么？

---

### Eval E2：Observer Value by Length [Offline]

**脚本**：`eval_e2_observer_length.py`
**输入**：Eval A + D1 结果
**输出**：`outputs/eval_e2_ac/eval_e2_observer_length.json`

**分析内容**：Observer 贡献（TSR delta, progress delta）按 trajectory length bucket 分解

**对应 GUI-360 E2 结论**：
- GUI-360: Observer 贡献份额随长度单调递增（6%→11%→24%）
- AndroidControl 是否有同样趋势？→ 支持 Shapley value 假设

---

### Pre-test 1：Credit Analysis [Offline]

**脚本**：`eval_pretest1_shapley.py`
**输入**：Eval A (required) + D1 (optional)
**输出**：`outputs/eval_pretest1_ac/eval_pretest1_shapley.json`

**分析内容**（单模型版本）：
- Per-step accuracy 曲线
- Cumulative success probability（连乘 step accuracy）
- 如果有 D1 数据：Observer credit by step position（observer vs baseline 的 per-step delta）

**与 GUI-360 Pre-test 1 的区别**：
- 无 V2/V3 split，无法计算 φ_V3(t)
- 但可以计算 Observer credit by step（如果有 D1 数据）

---

### Pre-test 3：Counterfactual Oracle Fix [Offline]

**脚本**：`eval_pretest3_counterfactual.py`
**输入**：Eval A `trajectory_results.jsonl`
**输出**：`outputs/eval_pretest3_ac/eval_pretest3_counterfactual.json`

**方法**：
- Oracle-fix grounding errors（type_match=True, extract_match=False → 强制 extract_match=True）
- Oracle-fix action errors（type_match=False → 强制正确）
- 分别计算修复后的 TSR
- Per-length bucket 分析

**核心问题**：
- 短轨迹：action fix > grounding fix?
- 长轨迹：grounding fix > action fix?
- 交叉点是否存在？

**对应 GUI-360 Pre-test 3 结论**：
- GUI-360: 完美交叉——短轨迹 Fix Action > Fix Ground (ratio 1.24), 长轨迹 Fix Ground > Fix Action (ratio 0.67)
- AndroidControl 是否有同样的 crossover？

---

## 第四部分：执行顺序与依赖

```
Phase 1 [GPU]: Eval A (core AR trajectory)
    ↓ unlocks all offline analyses
    ├── Eval B (step-position accuracy)
    ├── Eval C (hard cases)
    ├── Eval C2 (action-type analysis)
    ├── Eval C8 (verifier trigger)
    ├── Eval D0 (error type ceiling)
    ├── Eval D10 (action confusion matrix)
    └── Pre-test 3 (counterfactual oracle fix)

Phase 2 [GPU]: Eval C4+C7 multisample (K=10 per step)
    ↓
    └── Eval C4+C7 analysis (offline)

Phase 3 [GPU]: Eval D1 observer AR
    ↓ unlocks
    ├── Eval D4/D6/D7 (planner ceiling + failure types + length)
    ├── Eval E2 (observer value by length)
    └── Pre-test 1 (credit analysis with observer)

Phase 4 [GPU, can run in parallel]:
    ├── Eval D2 (prompted observer)
    ├── Eval D8 (info transfer ablation) — 3× cost vs D1
    └── Eval D9 (critic zero-shot) — needs extra inference per step
```

**建议执行路线**：
1. 先用 `--max_episodes 50` 在 Eval A 上快速验证 pipeline
2. 跑完整 Eval A（~24h）
3. 立即跑所有 Phase 1 offline analyses
4. 根据 Eval A 结果决定是否跑 Phase 2-4

---

## 第五部分：SLURM 模板

所有 GPU 实验共用同一模板：

```bash
#!/bin/bash
#SBATCH --job-name=<name>
#SBATCH --output=.../scripts/eval/ac/logs/<name>_%j.log
#SBATCH --error=.../scripts/eval/ac/logs/<name>_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=400G
#SBATCH --gres=gpu:4

source /home/a5l/shuqing.a5l/miniconda3/bin/activate ui-s1
export LD_LIBRARY_PATH=/opt/nvidia/hpc_sdk/Linux_aarch64/24.11/cuda/12.6/targets/sbsa-linux/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/opt/nvidia/hpc_sdk/Linux_aarch64/24.11/cuda/12.6/lib64:$LD_LIBRARY_PATH

# 启动 vLLM → 等待就绪 → 执行评估 → 清理
```

模型路径通过环境变量配置：
```bash
MODEL_PATH=/path/to/model MODEL_NAME=MyModel sbatch eval_a_ar_trajectory.slurm
```

---

## 第六部分：验证结果

### 6.1 基础设施验证 ✅

| 测试 | 状态 | 说明 |
|------|:----:|------|
| `ac_utils.py` imports | ✅ | 所有依赖加载成功（ui-s1 conda env） |
| 数据加载 + 路径修复 | ✅ | 1,543 episodes, 图片路径正确、文件存在 |
| `fix_line()` | ✅ | check_options 正确生成 |
| `length_bucket()` | ✅ | short/medium/long/vlong 分桶正确 |
| `categorize_action()` | ✅ | coord/non_coord 分类正确 |
| `init_format()` | ✅ | JsonFormat 初始化成功 |
| `gen_next_round()` | ✅ | Messages 构建正确（system + user with image） |
| `slim_messages()` | ✅ | 图片截断工作正常 |
| `find_last_image_ele()` | ✅ | 返回正确的 w/h/rw/rh (1080×2400) |
| `parse_response()` | ✅ | `<think>/<action>` 标签解析正确 |
| `evaluate_android_control_action()` | ✅ | GT action 评估为 True, wrong type 评估为 False |
| Python 编译检查 | ✅ | 全部 18 个 .py 文件通过 py_compile |

### 6.2 Offline Pipeline 验证 ✅

用 50 条 synthetic 结果（70% per-step accuracy, seed=42）测试所有 offline 脚本：

| 脚本 | 状态 | 关键输出 |
|------|:----:|---------|
| `eval_b_step_analysis.py` | ✅ | Step 0: 0.660, Step 2: 0.944, length-conditioned 输出正确 |
| `eval_c_hard_cases.py` | ✅ | 37/124 failed, fail_rate=0.298, by action type 和 step position |
| `eval_c2_action_type_analysis.py` | ✅ | Confusion matrix 生成, top pair: open→type (6) |
| `eval_c8_verifier_trigger.py` | ✅ | Silent fail rate=0.655, near-miss=11, conservative ceiling=0.360 |
| `eval_d0_error_ceiling.py` | ✅ | A_grounding=14, B_action=23, repeated_action_rate=0.095 |
| `eval_d4_d6_d7_offline.py` | ✅ | D6: A_grounding=0.378, B_action=0.622; D7: length×failure 正确 |
| `eval_d10_action_diagnosis.py` | ✅ | 7×7 matrix, near_miss_fraction=0.378, top confusion pairs |
| `eval_e2_observer_length.py` | ✅ | Delta=0 (same data both sides, 符合预期) |
| `eval_pretest1_shapley.py` | ✅ | Step accuracy curve + cumulative probability |
| `eval_pretest3_counterfactual.py` | ✅ | Fix grounding +0.040, fix action +0.060, crossover analysis |

### 6.3 GPU 脚本验证

| 脚本 | 状态 | 说明 |
|------|:----:|------|
| `eval_a_ar_trajectory.py` | ✅ 运行 | Job 2848495 + 2848748 (recovery), 1543/1543 完成 |
| `eval_c4c7_multisample.py` | ✅ 编译 | 需 vLLM server |
| `eval_d1_observer_ar.py` | ✅ 编译 | 需 vLLM server |
| `eval_d2_prompted_observer.py` | ✅ 编译 | 需 vLLM server |
| `eval_d8_info_transfer.py` | ✅ 编译 | 需 vLLM server, 3× 推理 cost |
| `eval_d9_critic_zeroshot.py` | ✅ 编译 | 需 vLLM server, 2× 推理 cost |

### 6.4 Bug 修复记录

| Bug | 影响 | 修复 |
|-----|------|------|
| `numpy.bool_` 不可 JSON 序列化 | 296/1543 episodes 丢失 | `ac_utils.py` 中 wrap `evaluate_android_control_action()` 返回值为 `bool()`；所有 GPU 脚本添加 `_json_default` handler |
| 模型输出双花括号 `}}` | JSON 解析失败 | `safe_parse_response()` 函数：regex 修复 `}}` → `}`，fallback 手动提取 `<action>` 标签 |
| `check_click()` 返回 `np.linalg.norm() <= threshold` | 返回 `numpy.bool_` 而非 Python `bool` | 在 `ac_utils.py` 中增加 wrapper，`_json_default` 作为 safety net |

---

## 第七部分：Phase 1 实验结果（Qwen2.5-VL-7B）

> **完成时间**: 2026-03-14
> **SLURM Jobs**: 2848495 (Eval A, 1247 eps) + 2848748 (Recovery, 296 eps) + auto offline
> **结果目录**: `outputs/eval_a_ac/Qwen2.5-VL-7B/`

### 7.1 Eval A: Core Metrics

| 指标 | 值 |
|------|-----|
| **TSR** | **16.07%** (248 / 1,543) |
| **Avg Progress** | 26.4% |
| **Scattered Progress** | 19.1% |
| **Total Steps Evaluated** | 2,905 |
| **Step-level Accuracy** | 55.5% (1,613 / 2,905) |

#### 按 Trajectory Length

| Bucket | Episodes | TSR | Avg Progress | Scattered Progress |
|--------|:--------:|:---:|:-----------:|:------------------:|
| short(1-3) | 438 | **34.9%** | 43.8% | 40.8% |
| medium(4-7) | 788 | 11.8% | 23.3% | 22.0% |
| long(8-15) | 289 | 0.7% | 10.3% | 9.8% |
| vlong(16+) | 28 | 0.0% | 5.6% | 5.8% |

#### 按 Action Type (extract_match)

| Action | Total | Type Match | Extract Match |
|--------|:-----:|:----------:|:------------:|
| type | 116 | 97.4% | **85.3%** |
| click | 1,569 | 88.1% | **73.2%** |
| swipe | 263 | 52.5% | **52.5%** |
| system_button | 238 | 50.4% | **45.4%** |
| wait | 111 | 31.5% | **31.5%** |
| open | 603 | 14.3% | **13.9%** |
| long_press | 5 | 0.0% | **0.0%** |

### 7.2 Eval B: Step-Position Accuracy

| Step | Type Match | Extract Match | Count |
|:----:|:----------:|:------------:|:-----:|
| 0 | 50.3% | **41.8%** | 1,540 |
| 1 | 80.7% | 71.6% | 580 |
| 2 | 79.9% | 72.5% | 374 |
| 3 | 82.5% | 67.7% | 223 |
| 4 | 87.4% | 75.7% | 103 |
| 5 | 76.9% | 69.2% | 52 |
| 6+ | ~80% | ~60-70% | <30 |

**关键发现**：Step 0 是最弱的步骤（41.8%），而非最强。这与 **GUI-360 完全相反**（GUI-360 step 0 = 86.6% 最高）。原因：AndroidControl 的第一步经常是 `open` action（打开 app），而模型几乎不会使用 `open` action。

#### Cumulative Success Probability

| Through Step | Cumulative |
|:-----------:|:----------:|
| 0 | 41.8% |
| 1 | 29.9% |
| 2 | 21.7% |
| 3 | 14.7% |
| 4 | 11.1% |
| 5 | 7.7% |
| 10 | 1.1% |

### 7.3 Eval C + C2: Hard Cases & Action Type Analysis

#### 失败率 by Step Position
- **Step 0: 58.2% failure rate** — 占 all errors 的 **69.3%**（896 / 1,292）
- Steps 1-4: 24-32% failure rate
- Steps 6-7: 50-57% failure rate（小样本）

#### 失败率 by Action Type
| Action | Fail Rate | Count |
|--------|:---------:|:-----:|
| long_press | 100% | 5/5 |
| open | **86.1%** | 519/603 |
| wait | 68.5% | 76/111 |
| system_button | 54.6% | 130/238 |
| swipe | 47.5% | 125/263 |
| click | 26.8% | 420/1,569 |
| type | 14.7% | 17/116 |

#### Top Confusion Pairs (GT → Predicted)
| Rank | GT → Pred | Count |
|:----:|-----------|:-----:|
| 1 | open → click | **242** |
| 2 | open → system_button | **205** |
| 3 | open → swipe | **137** |
| 4 | system_button → click | 110 |
| 5 | swipe → click | 78 |

**核心发现**：`open` action 是灾难性弱点。模型基本不会输出 `open` action，而是替代为 click/system_button/swipe。前 3 大混淆对全部涉及 `open`。

### 7.4 Eval C8: Verifier Analysis

| 指标 | 值 |
|------|-----|
| Silent fail steps | 863 (53.5% of correct steps followed by failure) |
| Near-miss episodes | 155 (12.0% of failed episodes, progress ≥ 50%) |
| Perfect verifier ceiling | 100% (theoretical) |
| **Conservative verifier ceiling** | **25.3%** (+9.2pp over baseline) |

### 7.5 Eval D0: Error Type Taxonomy

| 指标 | 值 |
|------|-----|
| Total errors | 1,292 |
| **A_grounding** (right action, wrong target) | 262 (**20.3%**) |
| **B_action** (wrong action type) | 1,030 (**79.7%**) |
| Repeated action rate | 9.4% (128 actions) |

**Observer ceiling estimation**:
- Conservative (+50% grounding fix): step accuracy 55.5% → 60.0%
- Optimistic (+100% grounding fix): step accuracy 55.5% → 64.5%

### 7.6 Eval D4/D6/D7: Failure Type by Length

#### D6: Overall Failure Types (1,295 failed episodes)
| Error Type | Count | Rate |
|-----------|:-----:|:----:|
| B_action | 1,030 | 79.5% |
| A_grounding | 262 | 20.2% |
| D_unknown | 3 | 0.2% |

#### D7: Length × Failure Type
| Length Bucket | A_grounding | B_action |
|:-------------|:-----------:|:--------:|
| short(1-3) | 22.1% | 77.9% |
| medium(4-7) | 19.9% | 80.0% |
| long(8-15) | 19.5% | 79.8% |
| vlong(16+) | 17.9% | 82.1% |

**关键发现**：Action error 占比在所有 trajectory 长度上**恒定 ~80%**。与 GUI-360 不同（GUI-360 grounding error 从 33%→49% 随长度递增）——AndroidControl 无 crossover。

### 7.7 Eval D10: Action Diagnosis

**Near-miss vs Complete-miss**:
- Near-miss (type_match=True, extract_match=False): 262 (20.3%)
- Complete-miss (type_match=False): 1,030 (79.7%)

**Step 0 Error Breakdown**:
- Near-miss at step 0: 8.5%
- Complete-miss at step 0: 49.7%

### 7.8 Pre-test 1: Step Accuracy Curve

Step accuracy 在 step 0 最低（41.8%），step 1+ 稳定在 67-76%。Cumulative success probability 呈指数衰减——即使 per-step 70%，到 step 10 cumulative 仅 1.1%。

### 7.9 Pre-test 3: Counterfactual Oracle Fix

#### Overall
| Scenario | TSR | Δ TSR | Avg Progress |
|----------|:---:|:-----:|:------------:|
| Baseline | 16.1% | — | 26.4% |
| **Fix all grounding** | 18.0% | +1.9pp | 30.6% |
| **Fix all action** | 23.3% | **+7.3pp** | 42.2% |

Action fix 收益 = **3.7× grounding fix 收益**

#### Per-Length Counterfactual
| Length | Baseline TSR | Fix Grounding | Fix Action | Dominant |
|--------|:-----------:|:------------:|:----------:|:--------:|
| short(1-3) | 34.9% | 39.0% (+4.1pp) | **55.0% (+20.1pp)** | action |
| medium(4-7) | 11.8% | 13.3% (+1.5pp) | 14.5% (+2.7pp) | action |
| long(8-15) | 0.7% | 0.7% (+0.0pp) | 1.7% (+1.0pp) | action |
| vlong(16+) | 0.0% | 0.0% (+0.0pp) | 0.0% (+0.0pp) | — |

**关键发现**：Action error 在所有长度上都是 dominant error type。**不存在 crossover**——与 GUI-360 的"完美交叉"结论不同。短任务 fix action 效果最大（+20.1pp）。

---

### 7.10 核心发现汇总

1. **Step 0 是瓶颈**：58.2% failure rate，占 all errors 的 69.3%。与 GUI-360（step 0 最强）完全相反
2. **`open` action 灾难性弱**：13.9% accuracy (603 instances)——前 3 大 confusion pairs 全部是 open→其他
3. **Action errors (80%) >> Grounding errors (20%)**：比例在所有 trajectory 长度上恒定
4. **Compounding 使长任务不可能**：per-step 70% → step 10 cumulative 1.1%，vlong TSR=0%
5. **Fix action 收益 3.7× fix grounding**：短任务 fix action → TSR 34.9%→55.0%
6. **无 crossover**：与 GUI-360 不同，action error 在所有长度上 dominant

### 7.11 与 GUI-360 的关键差异

| 发现 | GUI-360 | AndroidControl | 差异分析 |
|------|---------|---------------|---------|
| Step 0 accuracy | **86.6%**（最高） | **41.8%**（最低） | AC 第一步常是 `open` app，模型不会 |
| Error type | grounding 38.9% | grounding **20.3%** | AC 的 action error 更严重 |
| Crossover | 短=action, 长=grounding | 全长度=action | AC 无 crossover |
| Fix action vs ground | 短: 1.24× 长: 0.67× | 全长度: **3.7×** | AC 的 action fix 价值远超 grounding |
| TSR | 28.8% (V2+V3) | **16.1%** (单模型) | 单模型 + 更难的 action space |

### 7.12 对 PAMARL 框架的启示（Phase 1）

1. **Step amplifier 设计需调整**：GUI-360 的"step 0 最简单"假设在 AC 上不成立 → step amplifier 不应 downweight step 0
2. **优先修 action selection**：AC 上 grounding fix 收益很小，应优先训练 action type 选择（尤其是 `open`）
3. **Observer 的预期价值有限**：grounding error 仅 20%，Observer ceiling 有限（+4.5-9pp step accuracy）
4. **Verifier 有中等价值**：conservative ceiling +9.2pp，near-miss 12%

---

## 第 7.5 部分：Phase 3 实验结果 — D1 Observer AR

> **完成时间**: 2026-03-14 09:35 UTC
> **SLURM Job**: 2848840 (nid011238)
> **结果目录**: `outputs/eval_d1_ac/Qwen2.5-VL-7B/`

### D1 核心结果：Observer 有害

| 指标 | Baseline (Eval A) | Observer (D1) | Delta |
|------|:-----------------:|:-------------:|:-----:|
| **TSR** | **16.1%** (248) | **12.1%** (186) | **-4.0pp** |
| Avg Progress | 26.4% | 22.7% | -3.7pp |
| Scattered Progress | 19.1% | 16.3% | -2.8pp |

**Win/Loss 分析（D4 Planner Ceiling）**：
| 指标 | 值 |
|------|-----|
| Observer wins (obs 成功, baseline 失败) | 37 |
| Observer losses (obs 失败, baseline 成功) | **99** |
| Both success | 149 |
| Both fail | 1,258 |
| **Win/Loss ratio** | **0.37:1** |

Observer losses 是 wins 的 **2.7 倍**——Observer 不仅无用，而且**积极有害**。

### E2: Observer TSR Delta by Length

| Length Bucket | Baseline TSR | Observer TSR | Delta |
|:-------------|:-----------:|:----------:|:-----:|
| short(1-3) | 34.9% | 29.0% | **-5.9pp** |
| medium(4-7) | 11.8% | 7.2% | **-4.6pp** |
| long(8-15) | 0.7% | 0.7% | 0.0pp |
| vlong(16+) | 0.0% | 0.0% | 0.0pp |
| **Overall** | **16.1%** | **12.1%** | **-4.0pp** |

**关键发现**：Observer 在短/中长度任务上造成显著 TSR 下降，长任务上无效果（本身已近 0%）。与 GUI-360（Observer value 6%→24% 随长度递增）**完全相反**——AC 上 Observer 在所有长度上为负。

### 对比 GUI-360 D1

| 指标 | GUI-360 | AndroidControl |
|------|:-------:|:-------------:|
| Observer TSR delta | **+1.34pp** | **-4.0pp** |
| Win/Loss ratio | **2.3:1** | **0.37:1** |
| Observer 结论 | 有用（小幅正面） | **有害（显著负面）** |

---

## 第 7.6 部分：Phase 4a 实验结果 — D8 Info Transfer

> **完成时间**: 2026-03-14 11:30 UTC
> **SLURM Job**: 2848841 (nid011276)
> **结果目录**: `outputs/eval_d8_ac/Qwen2.5-VL-7B/`

### D8 三条件对比

| Condition | 说明 | TSR | Avg Progress | Delta vs C |
|:---------:|------|:---:|:-----------:|:----------:|
| **C** | No observer | **16.1%** (248) | **26.4%** | — (最好) |
| **B** | Current-step only | 13.5% (208) | 24.6% | **-2.6pp** |
| **A** | Full state doc | 13.6% (210) | 24.4% | **-2.5pp** |

### D8 核心发现

1. **无 Observer 是最优的**：C > A ≈ B，注入任何 observer context 都会降低性能
2. **History 无额外价值**：A (full history) ≈ B (current only)，差异仅 0.1pp → history 积累在 AC 上没有帮助
3. **Observer context 是有害噪声**：模型在移动端任务上的 action 选择被 observer 描述干扰

### 对比 GUI-360 D8

| 指标 | GUI-360 | AndroidControl |
|------|:-------:|:-------------:|
| 最优 condition | **A** (Full state doc) | **C** (No observer) |
| History value | History = **75%** of Observer value | History = **无额外价值** (A ≈ B) |
| Observer 整体效果 | 正面 | **负面** |
| 结论 | History 累积是核心 | Observer context 整体有害 |

---

## 第 7.7 部分：Phase 3 + 4a 更新后的启示

### 7.7.1 更新后的 PAMARL 验证表

5/6 个 GUI-360 核心发现已完成验证：

| 发现 | GUI-360 | AC 结果 | 验证 |
|------|:-------:|:------:|:----:|
| Step accuracy 单调递减 | ✅ step 0 最高 | ❌ step 0 最低 | ❌ |
| Grounding error 随长度递增 | ✅ 33%→49% | ❌ 恒定 ~20% | ❌ |
| Oracle fix 收益交叉 | ✅ 完美交叉 | ❌ 无 crossover | ❌ |
| **Observer value 随长度递增** | ✅ 6%→24% | ❌ **全长度为负** (-5.9pp→0pp) | ❌ |
| **History > Current for Observer** | ✅ 75% | ❌ **A ≈ B，history 无价值** | ❌ |
| Zero-shot Critic 无用 | ✅ PASS bias 94.9% | ⬜ 待验证 (D9 未提交) | ⬜ |

**5/5 已验证的发现全部不成立。**

### 7.7.2 更新后的框架启示

1. **Observer 在移动端不仅无用，而且有害**：TSR -4.0pp，win/loss 0.37:1
2. **History 积累无价值**：full state doc ≈ current-only，都比 no observer 差
3. **PAMARL 的 Observer 模块在移动端需要根本性重设计**（或直接去除）
4. **根因**：移动端 action error (80%) 远超 grounding error (20%)，Observer 针对的是 grounding/state confusion，但 AC 的瓶颈是 action selection
5. **建议优先级**：(1) 修 `open` action selection (2) Multi-sample/Verifier → action diversity (3) Observer 最低优先级

---

## 第八部分：与 GUI-360 实验的对照表

| Eval | GUI-360 结论 | AndroidControl 结果 | 验证状态 |
|------|-------------|---------------------|:--------:|
| **A** | V2+V3 TSR=28.8%, coord_match=81.6% | TSR=**16.1%**, step accuracy=55.5% | ✅ |
| **B** | Step 0 accuracy=86.6%（最高），late steps ~70% | Step 0=**41.8%**（最低），step 1+=71-76% → **趋势相反** | ✅ ❌ |
| **C** | Hard case rate step 1=11.8% → step 14=26.6% | Step 0 fail=**58.2%**（最高），hard cases 集中在 step 0 | ✅ ❌ |
| **C2** | click↔type = 主要混淆；Excel type=71% | open→click=**242×** 为主要混淆；type=85.3%, click=73.2% | ✅ |
| **C4+C7** | Oracle best-of-K=86.8%, Adaptive K Avg=2.0 | Oracle=81.0%, gain=+19.0pp, open oracle仅29.9% | ✅ |
| **C8** | Silent failure=36.1%, verify ceiling=43.9% | Silent fail=53.5%, conservative ceiling=**25.3%** (+9.2pp) | ✅ |
| **D0** | State confusion=38.9%, observer ceiling +8.8pp | Grounding=**20.3%**, action=**79.7%** → action error 主导 | ✅ |
| **D1** | Zero-shot Observer +1.34pp, win ratio 2.3:1 | Observer **-4.0pp**, win/loss **0.37:1** → **有害** | ✅ ❌ |
| **D2** | Prompted Observer -0.9pp vs D1 | 未提交（D1 已负，D2 预期更差） | ⬜ |
| **D8** | History=75% of Observer value | C>A≈B，**history 无价值，observer 整体有害** | ✅ ❌ |
| **D9** | Critic precision=9.8%, PASS bias=94.9% | 未提交 | ⬜ |
| **D10** | click↔type=55% of action errors, step1=51.6% | open→click/sys/swipe=**584×** 占 45% of all errors | ✅ |
| **E2** | Observer share 6%→24% with length | **全长度为负** (short -5.9pp, medium -4.6pp) | ✅ ❌ |
| **Pre-test 1** | φ_V3: +16.5pp → +38.9pp (2.4× 增长) | Step accuracy: 41.8%→71.6%→72.5%→67.7%... | ✅ |
| **Pre-test 3** | Fix Action/Ground 收益完美交叉 | **无 crossover**，action fix 全长度 dominant (3.7×) | ✅ ❌ |

---

## 第九部分：Phase 1 关键指标对照

| 指标 | GUI-360 值 | AndroidControl 实测 | 对比 |
|------|:----------:|:-------------------:|:----:|
| Baseline TSR | 28.8% (V2+V3) | **16.1%** (单模型) | 更低 |
| Step accuracy | 81.6% | **55.5%** | 更低 |
| Step 0 accuracy | 86.6% (最高) | **41.8%** (最低) | **相反** |
| Grounding error ratio | 38.9% | **20.3%** | 更低 |
| Action error ratio | 61.1% | **79.7%** | 更高 |
| Verify ceiling delta | +15.1pp | **+9.2pp** | 更低 |
| Fix action/ground ratio | 短 1.24×, 长 0.67× | **全长度 3.7×** | 更极端 |
| 主要混淆 | click↔type (55%) | **open→click/sys/swipe** | 不同 |
| Observer TSR delta | +1.34pp | **-4.0pp** | **相反** |
| Observer win/loss | 2.3:1 | **0.37:1** | **相反** |
| History value | 75% of observer | **无额外价值** (A≈B) | **不同** |

---

## 第十部分：下一步计划

### SLURM Job 跟踪

#### 已完成 Jobs

| Job ID | 名称 | Phase | 脚本 | 状态 | 结果 |
|:------:|------|:-----:|------|:----:|------|
| 2848413 | eval_a_test50 | 验证 | `eval_a_test50.slurm` | ✅ 完成 | 50 eps 验证，TSR=9.5%，发现 JSON 双括号 bug |
| 2848495 | eval_a_ac | Phase 1 | `eval_a_ar_trajectory.slurm` | ✅ 完成 | 1,247/1,543 eps（296 eps 因 numpy.bool_ bug 丢失） |
| 2848748 | eval_a_recovery | Phase 1 | `eval_a_recovery.slurm` | ✅ 完成 | 补跑 296 eps + 全部 9 个 offline 分析，总计 1,543/1,543 |
| 2848840 | eval_d1_ac | Phase 3 | `eval_d1_observer_ar.slurm` | ✅ 完成 | TSR=12.1% (**-4.0pp** vs baseline)，Observer **有害** |
| 2848841 | eval_d8_ac | Phase 4a | `eval_d8_info_transfer.slurm` | ✅ 完成 | C=16.1% > A=13.6% ≈ B=13.5%，**无 observer 最优** |
| 2848839 | eval_c4c7_ac | Phase 2 | `eval_c4c7_multisample.slurm` | ✅ 完成 | Oracle=81.0%, gain=+19.0pp, open oracle仅29.9% |
| 2861568 | eval_m1_ac | Phase 5 | `eval_m1_decompose.slurm` | ✅ 完成 | TSR=13.35% (**-2.7pp**)，decomposition **有害** |
| 2861569 | eval_m2_ac | Phase 5 | `eval_m2_plan_exec.slurm` | ✅ 完成 | TSR=16.27% (+0.2pp)，planner type acc=72.1%，**中性** |
| 2861570 | eval_m3_ac | Phase 5 | `eval_m3_router.slurm` | ✅ 完成 | TSR=**18.54%** (**+2.5pp**)，**唯一正面结果** |

#### 未提交 Jobs

| 名称 | Phase | 脚本 | 优先级 | 备注 |
|------|:-----:|------|:------:|------|
| eval_d2_ac | Phase 4b | `eval_d2_prompted_observer.slurm` | P3 低 | D1 已负面，D2 预期更差，**建议不提交** |
| eval_d9_ac | Phase 4b | `eval_d9_critic_zeroshot.slurm` | P3 低 | 预期同样无效（GUI-360: PASS bias 94.9%） |

### Phase 完成状态

#### Phase 1：✅ 完成

- [x] Eval A: AR Trajectory (1,543 episodes, TSR=16.1%)
- [x] Eval B: Step-position accuracy
- [x] Eval C: Hard cases
- [x] Eval C2: Action type analysis
- [x] Eval C8: Verifier trigger
- [x] Eval D0: Error type ceiling
- [x] Eval D10: Action diagnosis
- [x] Pre-test 1: Credit analysis
- [x] Pre-test 3: Counterfactual oracle fix
- [x] D4/D6/D7: Failure type by length

**结果目录**: `outputs/eval_a_ac/Qwen2.5-VL-7B/` (trajectory_results.jsonl + summary.json + offline/)

#### Phase 2：✅ 完成 (Job 2848839)

- [x] C4+C7 Multi-sample GPU (K=10 per step, temperature=1.0) — 1,543/1,543
- [x] C4+C7 Offline analysis (agreement, oracle, adaptive K)

**结果目录**: `outputs/eval_c4c7_ac/Qwen2.5-VL-7B/` (multisample_results.jsonl + eval_c4c7_analysis.json)

#### Phase 3：✅ 完成 (Job 2848840)

- [x] D1 Observer AR GPU → TSR=12.1% (**-4.0pp**, Observer 有害)
- [x] E2 Observer value by length → 全长度为负 (short -5.9pp, medium -4.6pp)
- [x] D4 Planner ceiling → wins=37, losses=99, win/loss=0.37:1
- [x] Pre-test 1 with observer → observer credit by step

**结论**：Observer 在 AndroidControl 上**有害**，与 GUI-360 完全相反
**结果目录**: `outputs/eval_d1_ac/Qwen2.5-VL-7B/` (observer_results.jsonl + offline/)

#### Phase 4a：✅ 完成 (Job 2848841)

- [x] D8 Info Transfer → C=16.1% > A=13.6% ≈ B=13.5%

**结论**：无 observer 最优，history 累积无额外价值，与 GUI-360 (history=75%) 完全相反
**结果目录**: `outputs/eval_d8_ac/Qwen2.5-VL-7B/` (info_transfer_results.jsonl + summary.json)

#### Phase 4b：⬜ 未提交（建议不提交）

| 实验 | 预期价值 | 理由 |
|------|---------|------|
| **D2 Prompted Observer** | 极低 | D1 已经 -4.0pp，structured prompt 不可能扭转 |
| **D9 Critic** | 低 | 预期同样无效（GUI-360: PASS bias 94.9%） |

**提交命令**（如确实需要运行）：
```bash
sbatch /scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/scripts/eval/ac/eval_d2_prompted_observer.slurm
sbatch /scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/scripts/eval/ac/eval_d9_critic_zeroshot.slurm
```

### 输出目录总览

```
outputs/
├── eval_a_ac/Qwen2.5-VL-7B/           # ✅ Phase 1 完成
│   ├── trajectory_results.jsonl        # 1,543 episodes
│   ├── summary.json
│   └── offline/                        # 9 个 offline 分析结果
├── eval_c4c7_ac/Qwen2.5-VL-7B/        # ✅ Phase 2 完成
│   ├── multisample_results.jsonl       # 1,543 episodes, K=10
│   └── eval_c4c7_analysis.json         # Oracle=81.0%, gain=+19.0pp
├── eval_d1_ac/Qwen2.5-VL-7B/          # ✅ Phase 3 完成
│   ├── observer_results.jsonl          # TSR=12.1%
│   ├── summary.json
│   └── offline/
├── eval_d8_ac/Qwen2.5-VL-7B/          # ✅ Phase 4a 完成
│   ├── info_transfer_results.jsonl
│   └── summary.json                    # C>A≈B
├── eval_m1_ac/Qwen2.5-VL-7B/          # ✅ Phase 5 完成
│   ├── decompose_results.jsonl         # TSR=13.35%
│   └── summary.json
├── eval_m2_ac/Qwen2.5-VL-7B/          # ✅ Phase 5 完成
│   ├── planner_executor_results.jsonl  # TSR=16.27%
│   └── summary.json
└── eval_m3_ac/Qwen2.5-VL-7B/          # ✅ Phase 5 完成
    ├── router_results.jsonl            # TSR=18.54%
    └── summary.json
```

### 与 PAMARL 的关系

AndroidControl 实验的核心价值是**跨数据集验证 GUI-360 的核心发现**：

| 发现 | GUI-360 | AC 结果 | 验证状态 |
|------|:-------:|:------:|:--------:|
| Step accuracy 单调递减 | ✅ step 0 最高 | ❌ **step 0 最低** (41.8%) | ❌ 不同 |
| Grounding error 随长度递增 | ✅ 33%→49% | ❌ **恒定 ~20%** 无递增 | ❌ 不同 |
| Oracle fix 收益交叉 | ✅ 短=action, 长=grounding | ❌ **全长度 action dominant，无 crossover** | ❌ 不同 |
| Observer value 随长度递增 | ✅ 6%→24% | ❌ **全长度为负** (-5.9pp→0pp) | ❌ 不同 |
| History > Current for Observer | ✅ 75% | ❌ **A≈B，history 无价值** | ❌ 不同 |
| Zero-shot Critic 无用 | ✅ PASS bias 94.9% | ⬜ 未提交 (D9) | ⬜ |

**结论：GUI-360 的 5/5 个已验证核心发现在 AndroidControl 上全部不成立。**

这意味着：
1. **PAMARL 框架的核心发现是 dataset-specific 而非 universal**——桌面 Office vs 移动端有本质差异
2. **根因**：AndroidControl 的 `open` action 导致 step 0 失败率极高 (58.2%)，action error 占 80%，grounding error 仅 20%。GUI-360 假设的"grounding/state confusion 是核心瓶颈"在移动端不成立
3. **Observer 在移动端有害**：TSR -4.0pp，win/loss 0.37:1。原因是 action error dominant，Observer 针对的 grounding/state confusion 不是瓶颈，反而引入噪声
4. **移动端优化路径**：(1) 修 `open` action selection (2) Multi-sample → action diversity (3) Verifier → +9.2pp ceiling (4) Observer **最低优先级**或直接去除

---

## 第十一部分：Multi-Agent Reasoning 实验（Phase 5）

> **动机**：Observer 模块在 AC 上有害 (-4.0pp)，需要转向直接攻击 **action selection** 瓶颈。
> 核心发现：action error 80%, step 0 fail 58%, open 86% fail rate。
> 新方向：Multi-agent reasoning，通过多 agent 协作改善 action type 选择。

### 瓶颈分析

| 瓶颈 | 数据支撑 | 应对思路 |
|------|---------|---------|
| Action type 选错 (80%) | open→click 242×, open→sys_button 205× | 多 agent 商议 action type |
| Step 0 失败 (58%) | 第一步常需 open app，模型不会 | 先 plan 再 execute |
| Compounding (step 10 cumulative 1.1%) | 70% per-step → 指数衰减 | Self-correction / retry |

### Exp M1: Task Decomposition (Hierarchical Planning) [P1]

**脚本**: `eval_m1_decompose.py` + `.slurm`
**成本**: 1 extra inference (decompose) + 1× per step
**输出**: `outputs/eval_m1_ac/{MODEL_NAME}/`

**架构**：
```
Decomposer Agent (第一步，一次性调用):
  输入: Goal + 第一张截图
  输出: Sub-goal sequence (numbered list)

  Prompt: "You are a mobile task planner. Given a high-level goal and the
  current screenshot, decompose the goal into a sequence of atomic sub-steps.
  Each sub-step should correspond to exactly one UI action.
  Output format: numbered list of sub-steps."

  Example output:
  1. Open the Settings app
  2. Tap on "Location"
  3. Toggle on "Use location"
  4. Go back
  5. Tap on "App permissions"
  6. Find and tap "Google"
  7. Select "Allow"

Executor Agent (每步):
  输入: 当前 sub-goal + 截图 + action history
  Prompt: 在标准 system prompt 后注入:
    "## Current Sub-goal\n{sub_goal}\nFocus on completing this specific sub-step."
  输出: action JSON (标准格式)
```

**关键问题**：
- Decompose 一次还是每步 re-plan？→ 先测一次性 decompose，再测 adaptive re-plan
- Sub-goal 和 GT step 的对齐方式？→ 仍用原始 GT check_options 评估，只改 prompt

### Exp M2: Planner-Executor Decomposition [P1]

**脚本**: `eval_m2_plan_exec.py` + `.slurm`
**成本**: 2× inference per step
**输出**: `outputs/eval_m2_ac/{MODEL_NAME}/`

**架构**：
```
Step-level Planner Agent (每步):
  输入: Goal + 截图 + action history
  Prompt: "You are a mobile action planner. Look at the screenshot and decide:
  1. What action TYPE should be taken? (click/open/swipe/type/system_button/wait)
  2. What is the TARGET of this action? (element description or app name)
  Output JSON: {"action_type": "...", "target": "..."}"

  Example output: {"action_type": "open", "target": "Settings"}

Executor Agent (每步):
  输入: Planner 输出 + 截图
  Prompt: 在标准 system prompt 后注入:
    "## Action Plan\nAction type: {action_type}\nTarget: {target}\n
    Execute this specific action. Output the action JSON."
  输出: action JSON (标准格式)
```

**与 D1 Observer 的区别**：
- Observer 提供 state description（被证明有害）
- Planner 提供 **action-level guidance**（直接攻击 action selection 瓶颈）

### Exp M3: Step-0 Specialist (Router) [P1]

**脚本**: `eval_m3_router.py` + `.slurm`
**成本**: step 0 = 2× inference, 后续 = 1× (bypass router)
**输出**: `outputs/eval_m3_ac/{MODEL_NAME}/`

**架构**：
```
Router Agent (仅 step 0):
  输入: Goal + 截图
  Prompt: "Look at the current screenshot. Is the app needed for this task
  already open? Answer YES or NO.
  If NO, what app should be opened? Output JSON:
  {"app_open": true/false, "app_name": "..." (if false)}"

  → If app_open=false:
      直接生成 {"action": "open", "text": "{app_name}"}
  → If app_open=true:
      走标准 pipeline (gen_next_round + vLLM)

Standard Agent (step 1+):
  标准 Eval A pipeline，无额外处理
```

**为什么精准有效**：
- 69.3% of all errors 在 step 0
- Step 0 error 的主因是 open→click 混淆
- Router 只在 step 0 介入，零额外 overhead on steps 1+

### Exp M4: Self-Reflection + Retry [P2]

**脚本**: `eval_m4_reflect.py` + `.slurm`
**成本**: 2-3× inference per step
**输出**: `outputs/eval_m4_ac/{MODEL_NAME}/`

**架构**：
```
Round 1: Standard agent → action JSON

Reflection Agent:
  输入: Goal + 截图 + proposed action
  Prompt: "You proposed this action: {action}
  Look at the screenshot and the task goal.
  Is this the right action TYPE? Consider:
  - If the goal requires opening an app and you're on home screen, use 'open'
  - If you need to go back, use 'system_button' with 'Back'
  - If you need to type text, make sure to use 'type' not 'click'
  Output: KEEP or REVISE with explanation."

  → If KEEP: 使用 Round 1 的 action
  → If REVISE: Round 2 重新生成（with reflection 注入 prompt）
```

### Exp M5: Action-Type Debate (Multi-Agent Voting) [P2]

**脚本**: `eval_m5_debate.py` + `.slurm`
**成本**: K+1 inference per step (K=3 proposals + 1 judge)
**输出**: `outputs/eval_m5_ac/{MODEL_NAME}/`

**架构**：
```
K=3 Proposer Agents (并行, temperature=0.7):
  各自生成 action JSON (不同 random seed → diversity)

Judge Agent:
  输入: Goal + 截图 + K 个 proposals
  Prompt: "Here are 3 proposed actions for this task step:
  1. {action_1}
  2. {action_2}
  3. {action_3}
  Which action is most appropriate? Consider the action TYPE first,
  then the specific parameters. Output the best action."
  输出: 选择最佳 action JSON
```

### Exp M6: Counterfactual Action Enumeration [P3]

**脚本**: `eval_m6_enumerate.py` + `.slurm`
**成本**: 2× inference per step
**输出**: `outputs/eval_m6_ac/{MODEL_NAME}/`

**架构**：先让 agent 列出可行的 action types，再从中选择。

### 优先级与执行计划

| 实验 | 预期 Impact | 成本 | 优先级 | 脚本 |
|------|:----------:|:----:|:------:|------|
| **M1: Task Decomposition** | 高 | 低 (1+1×) | **P1** | `eval_m1_decompose.py` |
| **M2: Planner-Executor** | 高 | 中 (2×) | **P1** | `eval_m2_plan_exec.py` |
| **M3: Step-0 Router** | 中-高 | 低 (step0 2×) | **P1** | `eval_m3_router.py` |
| M4: Self-Reflection | 中 | 中 (2-3×) | P2 | `eval_m4_reflect.py` |
| M5: Debate/Voting | 中 | 高 (K+1×) | P2 | `eval_m5_debate.py` |
| M6: Action Enumeration | 低-中 | 低 (2×) | P3 | `eval_m6_enumerate.py` |

**Phase 5 执行顺序**：
1. 先实现并提交 M1 + M2 + M3（P1，互相独立，可并行 3 nodes）
2. 根据 P1 结果决定是否跑 M4 + M5（P2）
3. M6 最低优先级

**脚本目录**: `scripts/eval/ac/` (与现有脚本同目录)

### Phase 5：✅ 完成 (Jobs 2861568/2861569/2861570)

- [x] M1 Task Decomposition → TSR=13.35% (-2.7pp) — **有害**
- [x] M2 Planner-Executor → TSR=16.27% (+0.2pp) — 中性
- [x] M3 Step-0 Router → TSR=**18.54%** (+2.5pp) — **唯一正面**

**结果目录**: `outputs/eval_m{1,2,3}_ac/Qwen2.5-VL-7B/`

---

## 第 7.13 部分：Phase 2 实验结果 — C4+C7 Multi-Sample

> **完成时间**: 2026-03-14 21:44 UTC
> **SLURM Job**: 2848839 (nid010225)
> **结果目录**: `outputs/eval_c4c7_ac/Qwen2.5-VL-7B/`

### C4+C7 核心结果

| 指标 | 值 |
|------|-----|
| Greedy step accuracy | 62.0% |
| **Oracle (best-of-K=10)** | **81.0%** |
| **Oracle gain** | **+19.0pp** |
| Mean agreement | 84.7% |
| All-K-correct rate | 12.7% |

### Per-Action-Type Oracle Rate

| Action Type | Oracle Rate | Total Steps | 备注 |
|-------------|:----------:|:-----------:|------|
| type | **92.2%** | 632 | 最高，text 输入比较确定 |
| swipe | 89.1% | 1,211 | 方向匹配容易 |
| click | 86.4% | 5,074 | 主要 action，多数可修复 |
| system_button | 84.0% | 343 | |
| wait | 56.3% | 567 | 较低 |
| **open** | **29.9%** | 608 | **最低** — 即使 K=10 也修不好 |
| long_press | 22.2% | 9 | 样本太少 |

### Adaptive K 策略

| Agreement Threshold | Step Accuracy | High-agree Fraction |
|:-------------------:|:------------:|:-------------------:|
| 0.6 | 65.3% | 89.5% |
| 0.7 | 67.5% | 81.1% |
| 0.8 | 69.6% | 70.5% |
| 0.9 | 71.6% | 55.3% |

### C4+C7 关键发现

1. **Oracle gap 巨大 (+19.0pp)**：模型有潜力但 greedy decoding 选错。Multi-sample + selection 策略空间大
2. **`open` action 是 sampling 也解决不了的问题**：oracle 仅 29.9%，10 个 sample 里 7 个都是错的 → 这是 **model capability gap** 而非 sampling diversity 问题
3. **Agreement 可作为 confidence signal**：high-agree (>0.9) 的步骤准确率 74.9%，low-agree 的 67.6%
4. **与 GUI-360 的对比**：GUI-360 Oracle=86.8%, AC Oracle=81.0%，差距不大。但 AC 的 `open` action oracle 极低 (29.9% vs GUI-360 无此 action type)

### 对比 GUI-360 C4+C7

| 指标 | GUI-360 | AndroidControl |
|------|:-------:|:-------------:|
| Greedy accuracy | 81.6% | **62.0%** |
| Oracle accuracy | 86.8% | **81.0%** |
| Oracle gain | +5.2pp | **+19.0pp** (3.7×) |
| 最弱 action type | click (domain 相关) | **open** (29.9%) |
| 启示 | 模型已接近上限 | 模型有大量 headroom，问题在 selection |

---

## 第 7.14 部分：Phase 5 实验结果 — Multi-Agent Reasoning

> **完成时间**: 2026-03-15 02:00 UTC
> **SLURM Jobs**: 2861568 (M1), 2861569 (M2), 2861570 (M3)

### Phase 5 总览

| 实验 | TSR | Delta vs Baseline | Avg Progress | 关键指标 |
|------|:---:|:-----------------:|:------------:|---------|
| **Baseline (Eval A)** | 16.07% | — | 0.246 | — |
| **M1: Task Decompose** | 13.35% | **-2.7pp** | 0.232 | avg 6.6 substeps vs 5.5 GT |
| **M2: Planner-Executor** | 16.27% | +0.2pp | 0.274 | planner type acc: 72.1% |
| **M3: Step-0 Router** | **18.54%** | **+2.5pp** | **0.306** | router triggered: 63% |

### M1 Task Decomposition: 有害 (-2.7pp)

| 指标 | 值 |
|------|-----|
| TSR | 13.35% (206/1543) |
| Avg substeps | 6.6 (vs GT avg 5.5) |
| Exact match (substeps=GT steps) | 286/1543 (18.5%) |

**失败原因分析**：
- Decomposer **过度分解**：平均多 1.1 步，引入不必要的中间步骤
- Sub-goal 和 GT step 不对齐：executor 试图执行 decomposer 的第 N 步，但实际屏幕状态对应 GT 的第 M 步
- **一次性 decompose 不可靠**：模型在仅看第一张截图的情况下无法准确预测完整路径

### M2 Planner-Executor: 中性 (+0.2pp)

| 指标 | 值 |
|------|-----|
| TSR | 16.27% (251/1543) |
| Planner type accuracy | 72.1% (2181/3023) |
| Avg Progress | 0.274 (vs baseline 0.246) |

**分析**：
- Planner 72.1% type accuracy vs executor 自身 ~62% → planner **确实更准** (+10pp)
- 但 planner 的错误会误导 executor：planner 错了 → executor 被迫按错误 plan 执行
- 净效果几乎为零 — planner 的增益被其错误的副作用抵消

### M3 Step-0 Router: 唯一正面 (+2.5pp) ⭐

| 指标 | 值 |
|------|-----|
| TSR | **18.54%** (286/1543) |
| Router triggered | 967/1543 (62.7%) |
| Step-0 acc (with router) | 58.3% |
| Step-0 acc (without router) | 62.2% |
| Avg Progress | **0.306** (vs baseline 0.246) |

**分析**：
- Router 被触发了 63% 的 episodes → 直接生成 `open` action
- Step-0 acc with router (58.3%) 看似低于 without (62.2%)，但 **without 组大多是 app 已经打开的 easy cases**
- 关键：router 将原本 baseline 会因 open→click 混淆而失败的 episodes 拉回正轨
- TSR +2.5pp 和 avg progress +0.06 是 **所有 multi-agent 实验中最大的改善**
- **成本最低**：仅 step 0 有 2× inference，steps 1+ 完全零开销

### Phase 5 关键洞察

1. **精准干预 > 全局改造**：M3 只改 step 0 效果最好，M1/M2 改每步反而有害/中性
2. **`open` action 是 model capability gap**：C4+C7 显示 oracle 仅 29.9%，M3 router 通过绕过标准 pipeline 直接生成 open，部分解决了这个问题
3. **信息注入 = 噪声**：M1 (sub-goal) 和 D1 (observer context) 都是往 prompt 注入额外信息，都有害。模型不缺信息，缺的是正确的 action mapping
4. **Planner 72% 不够**：需要更高精度的 action type 决策才能作为可靠的 guidance

---

## 第十二部分：全实验汇总与下一步方向

### 所有实验结果一览

| 实验 | Phase | TSR | Delta | 成本 | 结论 |
|------|:-----:|:---:|:-----:|:----:|------|
| **Eval A Baseline** | 1 | 16.07% | — | 1× | 基线 |
| **D1 Observer** | 3 | 12.1% | **-4.0pp** | 2× | ❌ 有害 |
| **D8 Info Transfer (A)** | 4a | 13.6% | -2.5pp | 2× | ❌ 有害 |
| **D8 Info Transfer (B)** | 4a | 13.5% | -2.6pp | 2× | ❌ 有害 |
| **M1 Task Decompose** | 5 | 13.35% | **-2.7pp** | 1+1× | ❌ 有害 |
| **M2 Planner-Executor** | 5 | 16.27% | +0.2pp | 2× | ➖ 中性 |
| **M3 Step-0 Router** | 5 | **18.54%** | **+2.5pp** | step0:2× | ✅ **最佳** |
| C4+C7 Oracle ceiling | 2 | — | +19.0pp | 10× | 巨大 headroom |

### 核心教训

1. **Prompt 注入信息 = 有害噪声**：D1/D8/M1 都往 executor prompt 注入额外信息（state description, sub-goals），全部有害或中性。模型的瓶颈不是缺信息
2. **精准干预 > 全局干预**：M3 仅改 step 0 得 +2.5pp，M2 改每步仅 +0.2pp
3. **`open` action 是不可逾越的 capability gap**：C4+C7 oracle 仅 29.9%，10 个 sample 也解决不了
4. **Oracle headroom 巨大 (+19pp)**：模型有潜力，问题在 selection 而非 generation

### 下一步可能方向

#### 方向 A: M3 Router 增强 (基于 Phase 5 最佳结果)

M3 Router 是唯一有效的干预。可以进一步增强：

1. **M3+: 扩展 Router 到 steps 1+**
   - 不仅在 step 0 router，在每一步都检查"当前 action 是否应该是 open"
   - 针对 open→click 在所有 step 上的混淆（不仅 step 0）
   - 预期成本：每步 2× inference

2. **M3+M2: Router + Planner 组合**
   - Step 0 用 M3 Router，steps 1+ 用 M2 Planner
   - 预期：M3 的 step 0 修复 + M2 的 per-step guidance
   - 成本：2× per step

#### 方向 B: Multi-Sample Selection (基于 C4+C7 Oracle Gap)

Oracle gain +19pp 是最大的 untapped potential：

3. **M5: Action-Type Debate (P2)**
   - K=3 proposers + 1 judge，利用 sample diversity
   - C4+C7 显示 agreement >0.9 时准确率 74.9% → judge 可利用 consensus signal
   - 但 `open` 的 oracle 仅 29.9%，debate 对此无效

4. **M7: Majority Vote (简单 baseline)**
   - K=5 samples，majority vote 选 action type，greedy 选 parameters
   - 最简单的 multi-sample selection 策略
   - 预期：介于 greedy (62%) 和 oracle (81%) 之间

#### 方向 C: RL/SFT 训练 (跳出 inference-time 限制)

所有 inference-time 方法的上限受限于 model capability（尤其 `open` oracle 29.9%）：

5. **在 AndroidControl 上做 SFT/RL 训练**
   - 直接用 AndroidControl 训练数据 fine-tune `open` action mapping
   - 预期：从根本上解决 open→click 混淆
   - 与当前 GUI-360 的 SFT/LoRA/MoE pipeline 对齐

6. **M3 Router 作为 RL Reward Signal**
   - Router 的正确性可以作为 step 0 的 reward signal
   - 与 PAMARL 的 reward 设计整合

#### 方向 D: 写论文/Report

7. **整理全部实验数据为论文格式**
   - 9 个 Phase 1 分析 + 3 个 Phase 3-4 实验 + 3 个 Phase 5 实验 + C4+C7
   - 核心 story：GUI agent 的跨数据集泛化——desktop vs mobile 的本质差异
   - PAMARL 的 5 个核心 claims 全部不泛化的 negative result

### 推荐优先级

| 优先级 | 方向 | 理由 |
|:------:|------|------|
| **P1** | M3+ Router 增强 / M3+M2 组合 | 基于已验证有效的 M3，低风险高回报 |
| **P1** | M7 Majority Vote | 最简单的 baseline，利用 C4+C7 已有数据 |
| P2 | M5 Debate/Voting | 更复杂的 multi-sample selection |
| P2 | RL/SFT on AndroidControl | 从根本解决 capability gap |
| P3 | M4 Self-Reflection | 类似 M2 的全局干预，可能中性 |
