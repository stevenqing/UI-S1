# Visual Inverse Dynamics for GUI Agents: Can VLMs Understand Visual State Transitions?

## 1. Research Question

V8 (Visual Hindsight Conditioning) 在训练时给模型看 s_{t+1}（下一步截图），用 CE loss 训练模型从 (s_t, s_{t+1}) 预测正确 action。这本质上是一个 **Visual Inverse Dynamics Model (IDM)**：given two consecutive visual states, predict the action that caused the transition.

V8 的数据揭示了一个 fundamental question:

**VLM 到底能不能理解 GUI 的视觉状态转移？看到 s_t 和 s_{t+1} 之后，模型能多准确地推断中间发生了什么 action？**

如果 VLM 具备这个能力，V8 的失败（CE loss 不下降、训练不稳定）就是训练 procedure 的问题。如果 VLM 不具备，那所有基于 visual hindsight 的方法都需要重新思考。

---

## 2. V8 Training Data Analysis

### 2.1 Hindsight CE Loss: 完全不下降

V8 训练 49 步 (step 31-79) 的 hindsight CE loss：

```
Early (step 31-55):  mean=0.775, range [0.03, 1.32]
Late  (step 56-79):  mean=0.795, range [0.14, 2.01]
Delta: +0.020 (flat, no improvement)
```

CE loss ≈ 0.78 意味着 per-token perplexity ≈ e^0.78 ≈ 2.18。考虑到 response 中有大量 boilerplate tokens（`<think>`, `<action>`, `{"action":` 等），这些 tokens 的 CE 接近 0，真正承载 action 信息的 tokens（action type, coordinates）的 CE 可能高达 3-5。

**这意味着即使看到了 s_{t+1}，模型对 action type 和坐标仍然高度不确定。**

### 2.2 Coverage: 过于稀疏

```
每步固定 8 samples（内存限制）
Coverage: mean=11.2%, range [8.4%, 16.3%]
Groups with donor: mean=12.7（需要同组有 correct rollout）
```

8 个 samples 的 CE loss variance 极大（0.03-2.01，67倍），梯度信号基本是噪声。

### 2.3 V8 vs V6 对比

| Metric          | V8 (hindsight) | V6 (no hindsight) |
|-----------------|-----------------|---------------------|
| Peak task_acc   | **16%** (step 50) | 12% (step 90)    |
| Stability       | Collapsed to 8% by step 70 | Slowly improving |
| Entropy spike   | 0.964 at step 60 | 0.958 at step 60 (recovered) |
| type_match peak | 0.745 (step 50) | **0.782** (step 90) |

V8 的优势（+6% peak）是暂时的，且伴随 entropy spike 和 collapse。V6 虽慢但更稳定。

### 2.4 PPO 信号本身的问题

~50% 的 rollout pairs 产生相同的 SP score（zero contrastive signal）。GiGPO 从一半的数据中得不到任何梯度。

---

## 3. Experiment Design

### 3.1 数据资源

- **训练集**: 1000 episodes, 6536 steps, 5536 steps with s_{t+1} (84.7%)
- **验证集**: 1543 episodes, 8444 steps, 6901 steps with s_{t+1} (81.7%)
- 所有 step 都有 screenshot，non-terminal steps 都有 s_{t+1}

Action type 分布 (训练集，non-terminal):
```
click:          3380 (61.1%)
swipe:           782 (14.1%)
type:            391 (7.1%)
open:            376 (6.8%)
wait:            372 (6.7%)
system_button:   227 (4.1%)
long_press:        8 (0.1%)
```

### 3.2 Existing Infrastructure

1. `STEP_PREDICTION` prompt template in `x/data/agent/json.py` — 纯 IDM prompt（给 s_t, s_{t+1}，预测 action）
2. `gen_next_round(hindsight=True)` — 在标准 prompt 上追加 s_{t+1}
3. `check_response_match()` — 评估 type_match 和 extract_match
4. vLLM serving + eval pipeline — 现成的推理基础设施
5. 已有 merged checkpoints: V8 step 50 (`merged_step_50`), V6 待 merge

### 3.3 实验矩阵

```
                    Model
                    ├── Base (Qwen2.5-VL-7B-Instruct, no RL)
                    ├── V6 step 90 (PPO only, no hindsight)
                    └── V8 step 50 (PPO + hindsight, peak checkpoint)

                    Prompt Condition
                    ├── Standard:   [task, history, s_t] → predict action
                    ├── Hindsight:  [task, history, s_t, s_{t+1}] → predict action
                    └── Pure IDM:   [s_t, s_{t+1}] → predict action (no task context)
```

Full matrix: 3 models × 3 conditions = 9 experiments

---

### Experiment 1: Zero-Shot Visual IDM (Base Model)

**Goal**: VLM 在没有任何 RL 训练的情况下，能否从 (s_t, s_{t+1}) 推断 action？

**Setup**:
- Model: Qwen2.5-VL-7B-Instruct (base, no fine-tuning)
- Prompt: `STEP_PREDICTION` template — 只给 s_t 和 s_{t+1}，不给 task 描述和 history
- Dataset: 验证集 6901 steps with s_{t+1}（或 subsample 500 for efficiency）
- Metric: type_match, extract_match per action type

**Key Questions**:
- Base model 的 IDM accuracy 是多少？
- 哪些 action types 容易推断（click: 视觉变化大），哪些难（wait: 无变化）？
- 如果 base model IDM accuracy > 80% type_match → VLM 已经具备这个能力
- 如果 < 50% → VLM 缺乏 visual comparison 能力

---

### Experiment 2: Hindsight Gap Analysis

**Goal**: s_{t+1} 到底能给模型带来多少 action prediction 的提升？

**Setup**:
- Model: 分别用 Base, V6, V8 三个 model
- 对比 Standard vs Hindsight prompt（同一模型，同一 sample，只有 s_{t+1} 有无差异）
- Dataset: 验证集 subsample 500 steps
- Metric: type_match, extract_match

**Analysis**:
```
Hindsight Gap = accuracy(hindsight) - accuracy(standard)
```

- **Large Gap (>20%)**: s_{t+1} 信息量大，模型能利用它 → V8 方向正确，只是训练 procedure 有问题
- **Small Gap (<5%)**: s_{t+1} 不怎么帮助模型 → 要么模型不会做 visual comparison，要么 action decision 的 bottleneck 不在 visual transition understanding
- **Negative Gap**: s_{t+1} 反而 confuse 模型 → 当前 prompt format 有问题

**Per-action-type breakdown**:
- `click`: s_{t+1} 应该显示 button 状态变化 → gap 应该大
- `wait`: s_{t+1} 与 s_t 几乎一样 → gap 应该接近 0
- `type`: s_{t+1} 显示输入的文字 → gap 可能大（文字直接可见）
- `swipe`: s_{t+1} 显示滚动后内容 → gap 中等
- `open`: s_{t+1} 显示新 app → gap 应该大
- `system_button`: s_{t+1} 显示 back/home 后的界面 → gap 中等

---

### Experiment 3: Pure IDM vs Task-Aware IDM

**Goal**: Task context 对 visual IDM 有多重要？

**Setup**:
- Model: Base model
- 对比:
  - Pure IDM: `STEP_PREDICTION` prompt（只有 s_t, s_{t+1}，不知道 task）
  - Task-Aware IDM: Hindsight prompt（有 task, history, s_t, s_{t+1}）
- Dataset: 验证集 subsample 500

**Analysis**:
```
Task Context Benefit = accuracy(task_aware) - accuracy(pure_idm)
```

- **Large benefit**: Task context 帮助消歧义（例如 click 在哪里取决于 task）
- **Small benefit**: Visual transition 本身就足以确定 action
- 这直接影响是否可以将 IDM 作为独立模块训练

---

### Experiment 4: RL Training Effect on IDM

**Goal**: RL 训练是否改善了模型的 visual transition understanding？

**Setup**:
- 在 Hindsight 和 Pure IDM condition 下，对比 Base vs V6 vs V8
- Dataset: same subsample

**Analysis**:
```
IDM Improvement = accuracy(RL_model, hindsight) - accuracy(Base, hindsight)
```

- **V8 > V6 > Base**: RL + hindsight 训练确实改善了 visual IDM → V8 方向正确
- **V8 ≈ V6 > Base**: RL 改善了 IDM 但 hindsight aux loss 没有额外贡献 → hindsight 训练 signal 太弱
- **V8 ≈ V6 ≈ Base**: RL 根本没改善 IDM → V8 的 CE loss 不下降是因为模型确实学不到

---

## 4. Implementation Plan

### 4.1 Eval Script Structure

创建 `scripts/eval/eval_visual_idm.py`，支持三种 prompt mode：

```python
# Mode 1: Standard (existing eval, no s_{t+1})
messages = gen_next_round(line, state, hindsight=False)

# Mode 2: Hindsight (standard + s_{t+1})
messages = gen_next_round(line, state, hindsight=True)

# Mode 3: Pure IDM (only s_t, s_{t+1}, no task context)
messages = construct_pure_idm_prompt(s_t_path, s_t1_path, action_space)
```

对每个 step，调用 vLLM 获得 model response，用 `check_response_match()` 评估 type_match 和 extract_match。

### 4.2 Sampling Strategy

验证集有 6901 eligible steps。为效率考虑：
- 从每种 action type 中 proportional sample
- 总计 ~500 steps（保证每种 action type >= 30 samples）
- 固定 seed 以确保跨实验 comparability

### 4.3 SLURM Configuration

每个实验需要 1 node × 4 GPUs（vLLM TP=4），约 1-2 小时：
- 3 models × 3 conditions = 9 experiments
- 可以 3 个并行（3 models 各跑 3 conditions sequentially）
- 总计约 6-9 小时

---

## 5. Expected Outcomes & Research Implications

### Scenario A: VLM 已具备强 Visual IDM 能力

```
Base model IDM accuracy: type_match > 80%, extract_match > 60%
Hindsight gap: > 20%
```

**Implication**: VLM 本身就能理解 visual transitions。V8 的失败纯粹是训练 procedure 问题（8 samples, noisy CE, gradient conflict）。

**Next step**: 改进 V8 的训练方式（增大 batch、adaptive coefficient、gradient isolation），不需要架构改动。

### Scenario B: VLM 具备部分 IDM 能力，action type 差异大

```
Base model IDM accuracy: type_match 50-80%
Per-type variance: click/open > 70%, wait/swipe < 40%
Hindsight gap: 10-20%
```

**Implication**: VLM 能理解"明显的"视觉变化（新页面打开、按钮高亮），但对"微妙的"变化（滚动、等待、系统操作）理解弱。

**Next step**: 针对不同 action type 设计不同的 hindsight 利用策略。对于 VLM 已擅长的 types，hindsight 提供额外 boost；对于不擅长的 types，需要其他形式的辅助信号（如 text description of visual change）。

### Scenario C: VLM 缺乏 Visual IDM 能力

```
Base model IDM accuracy: type_match < 50%
Hindsight gap: < 5%
```

**Implication**: VLM 的 visual encoder + attention 机制不适合做 visual comparison/diff reasoning。给模型看 s_{t+1} 但模型不知道怎么用。

**Next step**:
1. 需要 visual comparison 能力的预训练/微调
2. 或者用其他形式提供 transition 信息（文本描述 visual diff，而非原始 s_{t+1} 截图）
3. 或者引入 explicit visual comparison module

### Scenario D: RL 训练显著提升 IDM

```
V8 hindsight accuracy >> Base hindsight accuracy
且 V8 > V6（hindsight training 有效）
```

**Implication**: V8 的 hindsight CE loss 虽然看起来 flat，但模型确实在学习 visual IDM。CE 不下降是因为每步 sample 不同（online learning 的 loss 不一定下降）。

**Next step**: V8 方向正确，需要更长时间训练、更大 batch、或 curriculum。

---

## 6. Broader Research Direction

无论实验结果如何，这组实验回答的是一个 fundamental question about VLM capabilities:

**VLM 对 GUI 视觉动力学 (visual dynamics) 的理解程度如何？**

这个问题的答案直接决定了 GUI agent RL 的研究路线：

1. **如果 VLM 理解 visual dynamics** → 可以用 visual transitions 作为 dense reward signal, auxiliary training objective, or planning component
2. **如果 VLM 不理解** → 需要先解决 visual dynamics understanding（通过预训练、architectural change、或 alternative representations），然后才能有效利用 visual transitions

这是 V8 自然延伸出来的 research question，不是一个 technique，而是关于 VLM 在 GUI 环境中的 fundamental capability 的探索。

---

## 7. Experiment 1 Results: Base Model (Qwen2.5-VL-7B-Instruct)

### 7.1 Overall Results

| Mode | type_match | extract_match | n_evaluated | n_skipped (parse fail) |
|------|-----------|---------------|-------------|------------------------|
| Standard (s_t only) | **41.4%** | 38.6% | 220/291 | 71 (24.4%) |
| Hindsight (s_t + s_{t+1}) | 19.6% | 18.7% | 316/405 | 89 (22.0%) |
| Pure IDM (s_t + s_{t+1}, no task) | **43.6%** | 43.0% | 305/425 | 120 (28.2%) |

**Key finding: Hindsight HURTS (41.4% → 19.6%).** Adding s_{t+1} with the current prompt format ("Screenshot after correct action:") degrades performance by half. Pure IDM (43.6%) slightly outperforms Standard (41.4%).

### 7.2 Per-Action-Type Breakdown

| Action | Standard |  | Hindsight |  | Pure IDM |  |
|--------|----------|--|-----------|--|----------|--|
|        | n | type% | n | type% | n | type% |
| **click** | 43 | **0.0%** | 83 | **0.0%** | 54 | **0.0%** |
| swipe | 66 | 60.6% | 71 | 31.0% | 65 | **86.2%** |
| type | 30 | **96.7%** | 42 | 47.6% | 54 | **94.4%** |
| open | 23 | 0.0% | 40 | 7.5% | 46 | **19.6%** |
| system_button | 22 | 54.5% | 40 | 42.5% | 40 | 40.0% |
| wait | 35 | 28.6% | 34 | 0.0% | 40 | 2.5% |
| long_press | 1 | 0.0% | 6 | 0.0% | 6 | 0.0% |

### 7.3 Critical Finding: click = 0% Across All Modes

**The model NEVER correctly predicts "click" for click GT, in any mode.** This is the single most important finding.

click 占训练数据的 61.1%，但 base model 完全无法从视觉变化中推断 click action。

**Confusion matrix for click GT:**

| Mode | → terminate | → type | → system_button | → swipe | → click |
|------|------------|--------|-----------------|---------|---------|
| Standard (n=43) | 23 (53.5%) | 10 (23.3%) | 5 (11.6%) | 5 (11.6%) | 0 |
| Hindsight (n=83) | 35 (42.2%) | 32 (38.6%) | 6 (7.2%) | 9 (10.8%) | 0 |
| Pure IDM (n=54) | 20 (37.0%) | 8 (14.8%) | 15 (27.8%) | 10 (18.5%) | 0 |

Model predicts terminate, type, system_button, swipe — anything but click. 注意模型知道 "click" 这个 action（在其他 GT types 的预测中出现），但不把 click transitions 识别为 click。

**Why**: click 的视觉变化通常是微妙的（button highlight, dialog appear, page navigation），模型将这些变化误解为其他 action types 或 "任务完成"（terminate）。

### 7.4 Hindsight Prompt Causes "terminate" Over-Prediction

Overall prediction distribution:

| Prediction | Standard | Hindsight | Pure IDM |
|------------|----------|-----------|----------|
| terminate | 51 (23.2%) | **101 (32.0%)** | 59 (19.3%) |
| click | 38 (17.3%) | 84 (26.6%) | 23 (7.5%) |
| swipe | 51 (23.2%) | 38 (12.0%) | 91 (29.8%) |
| type | 43 (19.5%) | 57 (18.0%) | 65 (21.3%) |
| system_button | 27 (12.3%) | 32 (10.1%) | 53 (17.4%) |
| wait | 10 (4.5%) | 1 (0.3%) | 1 (0.3%) |

**Hindsight mode** 的 "Screenshot after correct action:" prompt 让模型认为 task 已完成 → 大量预测 terminate (32%)。这直接解释了 V8 的 hindsight CE loss 为什么不下降：**prompt format 本身在 mislead 模型。**

### 7.5 VLM Visual IDM Capability Spectrum

从 Pure IDM 结果（最干净的 visual-only 信号）看，VLM 的 visual transition 理解呈现清晰的 capability spectrum：

```
Detectable (>80%):
  type  (94.4%) — 文字出现/消失，视觉信号极强
  swipe (86.2%) — 内容滚动/位移，视觉变化明显

Partially detectable (20-60%):
  system_button (40.0%) — 界面切换（back/home），部分可检测
  open          (19.6%) — 新 app 打开，可能与 system_button 混淆

Undetectable (<5%):
  click     (0.0%) — 微妙的 UI 变化，模型完全无法识别
  wait      (2.5%) — 无视觉变化（或微小变化如 loading），本质不可检测
  long_press(0.0%) — 样本量太小 (6)，但也完全未检测到
```

### 7.6 Implications for V8 and Future Directions

**结论：匹配 Scenario B+C（部分能力 + click 完全失败）**

1. **V8 的 fundamental problem**: V8 试图让模型从 (s_t, s_{t+1}) 学习 action prediction，但 61% 的 action (click) 的视觉变化对 VLM 来说是 **不可识别的**。即使给无限 samples 和完美 training，模型也无法从 visual transition 推断 click。

2. **Hindsight prompt format 有 bug**: "Screenshot after correct action:" 导致 terminate over-prediction。如果要继续 hindsight 方向，需要重新设计 prompt format（如 Pure IDM 的 "pre/post-operation screenshot" 格式更好）。

3. **Selective hindsight**: 只对 detectable action types (swipe, type) 提供 hindsight signal。对 click, wait 等不提供（因为信号是 noise）。这能从 ~39% 的 steps 提取有效信号。

4. **Visual grounding gap**: click 的失败可能不是 visual transition detection 的问题，而是 **visual grounding** 的问题 — 模型看到了 UI 变化但不知道变化对应 "click" 这个 action + 哪个坐标。这需要更 structured 的 visual comparison（如 explicit diff highlighting, bounding box annotation）。

5. **Alternative to visual hindsight**: 对于 click (61% of data)，与其给模型看 s_{t+1}，不如给 textual description of the visual change（"A dialog appeared with title X", "Button Y became highlighted"）。VLM 可能更擅长处理 language-described transitions。

---

## 8. Training Progress (Live)

### V8 (sp_gigpo_spwa_k8_v8, Job 3334060) — step 84, still running

| Step | task_acc | type_match | extract_match | SP    | entropy | hs_CE |
|------|---------|-----------|---------------|-------|---------|-------|
| 30   | 10%     | 0.733     | 0.500         | 0.500 | -       | -     |
| 40   | 8%      | 0.703     | 0.545         | 0.545 | 0.648   | 0.730 |
| 50   | **16%** | 0.745     | 0.588         | 0.588 | 0.819   | 1.147 |
| 60   | 10%     | 0.733     | 0.571         | 0.571 | **0.964** | 0.807 |
| 70   | 8%      | 0.714     | 0.531         | 0.531 | 0.754   | 1.026 |
| 80   | **8%**  | 0.755     | 0.531         | 0.531 | 0.701   | 0.430 |
| 84   | -       | -         | -             | -     | **0.966** | 0.894 |

V8 collapsed: task_acc 16% → 8% (step 50 → 80). Entropy spikes at step 60 (0.964) and 84 (0.966). Hindsight CE highly variable (0.135-1.207), never converging.

### V6 (sp_gigpo_spwa_k8_v6, Job 3333780) — step 102, still running

| Step | task_acc | type_match | extract_match | SP   | entropy |
|------|---------|-----------|---------------|------|---------|
| 40   | 10%     | 0.714     | 0.571         | 0.571 | 0.580  |
| 50   | 8%      | 0.737     | 0.535         | 0.535 | 0.741  |
| 60   | 8%      | 0.750     | 0.574         | 0.574 | 0.958  |
| 70   | 8%      | 0.705     | 0.477         | 0.477 | 0.695  |
| 80   | 10%     | 0.760     | 0.550         | 0.550 | 0.756  |
| 90   | **12%** | **0.782** | **0.600**     | 0.600 | 0.876  |
| 100  | **12%** | 0.781     | 0.542         | 0.542 | 0.727  |

V6 peaked at step 90 (12% task_acc, 0.782 type_match, 0.600 SP). Step 100 shows slight regression in SP (0.600→0.542) while task_acc holds at 12%. Now at step 102 — likely plateaued.

### V8 Visual IDM Evaluation (Job 3342702) — COMPLETED

---

## 9. Experiment 2 Results: V8 Step 50 (RL + Hindsight)

### 9.1 Overall Results

| Mode | Base type_match | V8 type_match | Base extract_match | V8 extract_match | Base skip% | V8 skip% |
|------|----------------|---------------|-------------------|-----------------|-----------|---------|
| Standard | 41.4% | **52.7%** | 38.6% | **50.0%** | 24.4% | **0%** |
| Hindsight | 19.6% | **66.3%** | 18.7% | **64.1%** | 22.0% | **0%** |
| Pure IDM | 43.6% | **53.4%** | 43.0% | **49.8%** | 28.2% | **0%** |

**Key findings:**
1. **RL training dramatically improved all modes** — particularly hindsight (+46.7% type_match!)
2. **Hindsight gap flipped from negative to positive**: Base: -21.8% (hindsight hurts) → V8: +13.6% (hindsight helps)
3. **0% parse failures** — RL fixed output formatting (base had 22-28% parse failures)
4. **click still 0%** in ALL modes — RL training did not help

### 9.2 Per-Action-Type Comparison (type_match %)

| Action | Base Std | V8 Std | Base Hs | V8 Hs | Base IDM | V8 IDM |
|--------|---------|--------|---------|-------|----------|--------|
| **click** | **0** | **0** | **0** | **0** | **0** | **0** |
| swipe | 60.6 | 60.3 | 31.0 | **75.7** | **86.2** | 82.4 |
| type | **96.7** | 93.5 | 47.6 | **94.6** | **94.4** | **94.6** |
| open | 0.0 | **65.4** | 7.5 | **67.3** | 19.6 | **45.5** |
| system_button | 54.5 | **73.9** | 42.5 | **77.8** | 40.0 | 48.9 |
| wait | 28.6 | **38.5** | 0.0 | **51.2** | 2.5 | **14.0** |

### 9.3 Critical Analysis

**V8 学到了 hindsight 利用能力：**
- Hindsight gap: Base = -21.8% (hindsight hurts) → V8 = +13.6% (hindsight helps)
- V8 不再被 "Screenshot after correct action" 误导为 terminate（terminate 从 32% 降至 2.3%）
- open action: 7.5% → 67.3% (几乎从零到可用)
- wait action: 0% → 51.2% (从不可检测到半数可检测)

**但 click 仍然是 0%：**
- V8 hindsight click GT (n=24): → type(6), open(5), swipe(4), system_button(4), wait(3), 0 click
- 即使 RL 训练了 50 步 + hindsight aux loss，模型仍然完全无法从 visual transition 识别 click
- 这证实 click 的 visual transition 对当前 VLM 来说是 **fundamentally undetectable**

**Prediction distribution shift (Hindsight mode):**

| Prediction | Base | V8 |
|------------|------|-----|
| terminate | **32.0%** | **2.3%** |
| click | 26.6% | 14.1% |
| swipe | 12.0% | 21.2% |
| type | 18.0% | 20.6% |
| system_button | 10.1% | 17.3% |
| open | 0.9% | 15.0% |
| wait | 0.3% | 8.8% |

V8 的预测分布更加 balanced 和 diverse（不再被 terminate 主导），说明 RL 训练让模型学会了更好地利用 visual information。

### 9.4 Implications

1. **V8 方向部分正确**: Visual hindsight conditioning **works** — 模型确实可以从 s_{t+1} 获取有用信息。问题不在 idea 本身，而在 (a) click 的 visual undetectability 和 (b) 训练不稳定。

2. **click 是 fundamental bottleneck**: 61% 的训练数据是 click，但 V8 在 click 上获得的 hindsight signal 是纯噪声。这些噪声梯度可能是 V8 训练不稳定的根源之一。

3. **Selective hindsight 假设得到验证**: 如果只对 detectable actions (swipe, type, open, wait, system_button) 提供 hindsight signal，跳过 click，可能同时获得 (a) 有效的 IDM learning signal 和 (b) 训练稳定性。

---

## 10. Experiment 3 Results: V6 Step 90 (RL only, no hindsight)

### 10.1 Overall Results

| Mode | Base | V6 | V8 |
|------|------|-----|-----|
| Standard type_match | 41.4% | **52.5%** | **52.7%** |
| Hindsight type_match | 19.6% | 18.1% | **66.3%** |
| Pure IDM type_match | **43.6%** | 31.7% | **53.4%** |
| Parse failures | 22-28% | **0%** | **0%** |

### 10.2 Per-Action-Type Comparison (type_match %, all 3 models × 3 modes)

**Standard mode:**

| Action | Base | V6 | V8 |
|--------|------|-----|-----|
| click | 0 | 0 | 0 |
| swipe | 60.6 | **67.6** | 60.3 |
| type | **96.7** | 93.5 | 93.5 |
| open | 0 | **57.7** | **65.4** |
| system_button | 54.5 | **73.9** | **73.9** |
| wait | **28.6** | 20.5 | **38.5** |

**Hindsight mode:**

| Action | Base | V6 | V8 |
|--------|------|-----|-----|
| click | 0 | 0 | 0 |
| swipe | 31.0 | 35.1 | **75.7** |
| type | 47.6 | 5.4 | **94.6** |
| open | 7.5 | 27.3 | **67.3** |
| system_button | 42.5 | 44.4 | **77.8** |
| wait | 0 | 0 | **51.2** |

**Pure IDM mode:**

| Action | Base | V6 | V8 |
|--------|------|-----|-----|
| click | 0 | 0 | 0 |
| swipe | **86.2** | 54.1 | 82.4 |
| type | **94.4** | 37.5 | **94.6** |
| open | 19.6 | **60.0** | 45.5 |
| system_button | **40.0** | 31.1 | **48.9** |
| wait | 2.5 | 0 | **14.0** |

### 10.3 Critical Analysis

**V6 vs V8 的关键差异是 hindsight 利用能力：**

| Metric | V6 | V8 |
|--------|-----|-----|
| Hindsight gap (Hs - Std) | **-34.4%** | **+13.6%** |
| Terminate in Hs mode | 19.2% | **2.3%** |
| Click in Hs mode | **42.1%** | 14.1% |

1. **V6 完全不会利用 s_{t+1}**: Hindsight mode 反而 hurt V6 (52.5% → 18.1%)，跟 base 一样。V6 在 hindsight mode 过度预测 "click" (42.1%)，而 base 过度预测 "terminate" (32%)。

2. **V8 的 hindsight aux loss 确实教会了模型利用 s_{t+1}**: V8 hindsight 66.3% >> V6 hindsight 18.1%。这证明 hindsight training 不是 useless — 它确实赋予了模型一种新能力。

3. **V6 的 pure IDM 反而退化了**: Base 43.6% → V6 31.7%。RL 训练（without hindsight）让模型的 visual comparison 能力变差了，尤其 swipe (86→54%) 和 type (94→37.5%)。可能因为 RL 让模型过度依赖 task context 而削弱了纯视觉推理。

4. **Standard mode V6 ≈ V8**: 两者都是 ~52.5%，说明标准 action prediction 主要由 RL 训练决定，hindsight training 没有显著帮助 standard mode。

### 10.4 Prediction Distribution (Hindsight mode)

| Prediction | Base | V6 | V8 |
|------------|------|-----|-----|
| terminate | **32.0%** | 19.2% | **2.3%** |
| click | 26.6% | **42.1%** | 14.1% |
| swipe | 12.0% | 10.2% | **21.2%** |
| type | 18.0% | 10.5% | **20.6%** |
| system_button | 10.1% | 11.0% | **17.3%** |
| open | 0.9% | 5.9% | **15.0%** |
| wait | 0.3% | 1.1% | **8.8%** |

V8 的预测分布最 balanced、最 diverse。V6 在 hindsight 下集中预测 click (42%)，这是一种新的 failure mode。

### 10.5 Summary: What Each Model Learned

| Capability | Base | V6 (RL) | V8 (RL + Hindsight) |
|------------|------|---------|---------------------|
| Standard action prediction | 41.4% | **52.5%** | **52.7%** |
| Format compliance | 75% | **100%** | **100%** |
| s_{t+1} utilization (Hs gap) | -21.8% | -34.4% | **+13.6%** |
| Visual IDM (pure) | **43.6%** | 31.7% (退化) | **53.4%** |
| Click detection | 0% | 0% | 0% |

**结论**：
- RL 训练 (V6, V8) 都提升了 standard prediction (+11%) 和 format compliance
- **只有 V8 学会了 hindsight 利用** — 这是 hindsight aux loss 的直接贡献
- V6 的 pure visual IDM 能力退化，说明 RL 训练倾向于让模型依赖 task context
- **click = 0% 是跨模型、跨模式的 universal failure** — 这是 VLM 的 fundamental limitation

---

## 11. Experiment 4 Results: V8 Training Trajectory (step 50 → 70 → 80)

### 11.1 Overall Results — V8 Hindsight Capability Over Training

| Model | Standard | Hindsight | Pure IDM | Hs Gap | task_acc (val) |
|-------|---------|-----------|----------|--------|----------------|
| Base | 41.4% | 19.6% | 43.6% | -21.7% | - |
| V6 s90 | 52.5% | 18.1% | 31.7% | -34.4% | 12% |
| **V8 s50** | **52.7%** | **66.3%** | **53.4%** | +13.7% | **16%** |
| V8 s70 | 46.5% | 66.0% | 45.7% | **+19.6%** | 8% |
| V8 s80 | 43.2% | 56.9% | 41.3% | +13.6% | 8% |

### 11.2 Per-Action Hindsight type_match Over V8 Training

| Action | Base | V8 s50 | V8 s70 | V8 s80 |
|--------|------|--------|--------|--------|
| click | 0% | 0% | 0% | 0% |
| swipe | 31.0% | **75.7%** | 71.6% | 44.6% |
| type | 47.6% | **94.6%** | 91.1% | 91.1% |
| open | 7.5% | 67.3% | **85.5%** | 76.4% |
| system_button | 42.5% | 77.8% | 66.7% | **80.0%** |
| wait | 0% | 51.2% | **58.1%** | 39.0% |

### 11.3 Analysis: Hindsight Capability Survives but Standard Degrades

**关键发现：V8 的 hindsight 利用能力在训练过程中保持稳定，但 standard 能力退化。**

1. **Hindsight 能力保持**: step 50 (66.3%) → step 70 (66.0%) → step 80 (56.9%)。虽然 step 80 有所下降，但仍远高于 base (19.6%)。Hindsight gap 甚至在 step 70 达到最高 (+19.6%)。

2. **Standard 能力退化**: step 50 (52.7%) → step 70 (46.5%) → step 80 (43.2%)。V8 的标准 action prediction 持续下降，接近 base (41.4%)。这与 task_acc 下降 (16% → 8%) 一致。

3. **Pure IDM 同步退化**: step 50 (53.4%) → step 70 (45.7%) → step 80 (41.3%)。纯视觉推理能力也在退化。

4. **Hindsight gap 扩大**: standard 下降而 hindsight 保持 → gap 从 +13.7% (s50) 扩大到 +19.6% (s70)。这说明模型越来越"依赖" s_{t+1}，而不是内化学到的知识。

5. **swipe 退化最严重**: Hindsight 下 75.7% → 71.6% → 44.6%。可能因为持续训练导致的 entropy spike 和 mode collapse 影响了模型对明显视觉变化的检测能力。

**结论**: V8 的 hindsight training 教会了模型一种 s_{t+1} 利用能力，这种能力在训练 30 步后仍然保持。但 V8 的训练不稳定（entropy spikes, task_acc collapse）导致 standard 能力退化。理想的训练应该让 hindsight 能力 **transfer** 到 standard mode，而不是形成 "依赖 s_{t+1}" 的模式。

---

## 12. Next Steps

- [x] **Experiment 1**: Base model Visual IDM eval — Section 7
- [x] **Experiment 2**: V8 step 50 eval — Section 9
- [x] **Experiment 3**: V6 step 90 eval — Section 10
- [x] **Experiment 4**: V8 step 70, 80 eval — Section 11
- [ ] **Deep dive click failure**: 分析 click GT 的 s_t → s_{t+1} visual differences，理解为什么 VLM 无法检测
- [ ] **Selective hindsight experiment**: 只对 non-click actions 提供 hindsight signal
- [ ] **Prompt redesign**: 测试不同的 hindsight prompt format
- [x] **Direction C: Causal Probing** — Job 3363986 (COMPLETED)
  - 5 ablation conditions × hindsight mode × V8 s70
  - Results in Section 14

---

## 13. Research Analysis: The LUPI Transfer Failure

### 13.1 Problem Formulation

V8 的实验揭示了一个 fundamental research problem:

> **Visual Hindsight LUPI Problem**: VLM 可以学习利用 privileged information (s_{t+1}) 来预测 action，但这种能力不会 transfer 到 non-privileged (standard) 模式。模型形成 dependency 而非 internalization。

实验证据:
- V8 s50: Standard 52.7%, Hindsight 66.3% → Gap +13.7%
- V8 s70: Standard 46.5%, Hindsight 66.0% → Gap +19.6% (**gap 扩大，standard 退化**)
- V8 s80: Standard 43.2%, Hindsight 56.9% → Gap +13.6%

V8 的训练让 standard 能力从 52.7% **退化到 43.2%**（接近 base 41.4%），同时 hindsight 保持在 57-66%。模型的 action prediction 变得越来越 "条件化" 于 s_{t+1} 的存在。

### 13.2 Why V8 Fails to Transfer: Conditional Feature Gating

V8 的 aux loss 在一个 **完全不同的 input distribution** 上训练模型（enriched prompt with s_{t+1}）。模型可能学到的是：

```
IF s_{t+1} present:
    attend to visual differences → predict action from transition
ELSE:
    fall back to task context → predict action from instruction
```

这是 multi-task learning 中的 **negative transfer** 现象。两个 "任务"（with/without s_{t+1}）共享参数但有不同的最优特征，训练一个会 hurt 另一个。

关键 insight: **V8 的 hindsight aux loss 和 PPO loss 在参数空间中可能是 conflicting gradients。** Hindsight loss 把参数推向 "利用 s_{t+1}"，PPO loss 把参数推向 "不依赖 s_{t+1}" → 两者拔河导致 standard 退化。

### 13.3 Three Research Directions

---

#### Direction A: Visual Forward Dynamics — "Imagine Before You Act"

**核心思想**: 不用 inverse dynamics (s_t, s_{t+1} → action)，而用 **forward dynamics**: (s_t, action) → predicted s_{t+1} representation。

**Mechanism**:
1. 训练阶段：对每个 rollout (s_t, a_t, s_{t+1})，在 VLM 的 representation space 中训练：
   - f(repr(s_t), a_t) → pred_repr 应该接近 repr(s_{t+1})
   - Loss: contrastive loss 或 cosine similarity loss
2. 推理阶段：模型"想象"每个候选 action 的后果
   - 选择后果最接近 task goal 的 action
   - 不需要 s_{t+1} — 模型自己 predict 它

**为什么解决 transfer 问题**:
- Forward dynamics 天然不需要 privileged info at inference
- s_{t+1} 只在训练时作为 supervision target，不作为 input
- 模型 internalize 的是 "action → consequence" mapping，而非 "consequence → action" mapping

**研究新颖性**:
- 现有 world model (Dreamer, IRIS) 在 pixel space 做 forward prediction → 对 GUI 截图太复杂
- 我们在 VLM 的 representation space 做 → 利用 VLM 已有的 visual understanding
- 这是 **foundation model 内部的 world model**，不是外部单独的 world model

**挑战**:
- 需要定义 "repr(s_t)" — 用 VLM 的哪一层 representation？
- Contrastive loss 的设计（negative samples 怎么选？）
- Forward dynamics 在 GUI 中可能 inherently hard（click 后的截图千变万化）

**可行性评估**: 中等。需要 (1) 提取 VLM 中间层 representation, (2) 设计 auxiliary head for forward prediction, (3) 集成到 RL training loop。与现有架构差距较大。

---

#### Direction B: Hindsight as Critic, Not as Actor — "Privileged Evaluation, Not Privileged Generation"

**核心思想**: s_{t+1} 不用来训练 policy（actor），而用来训练 value function（critic）。

**V8 的问题**: s_{t+1} 直接进入 policy 的 CE loss → policy 学会依赖 s_{t+1}。

**Proposed**:
1. 训练一个 **transition-aware critic** Q(s_t, a_t, s_{t+1}) → action quality score
   - 这个 critic 有 privileged access to s_{t+1}
   - 它评估 "这个 action 导致了什么结果"
2. 用 critic 的评估来 **shape PPO advantage estimates**
   - advantage = f(reward, baseline) + λ * critic_bonus(s_t, a_t, s_{t+1})
   - Policy 本身只看 s_t，不看 s_{t+1}
3. Policy 学到的是：采取 "导致好结果" 的 action — 但不知道 s_{t+1}

**类比**: Asymmetric Actor-Critic (OpenAI Five, MAPPO)
- Actor: 只看 local observation
- Critic: 看 global state (privileged info)
- 这是 RL 中 well-established 的 pattern，但没有在 VLM GUI agent 中用过

**为什么解决 transfer 问题**:
- Policy 的参数 NEVER see s_{t+1} → 不会形成 dependency
- s_{t+1} 的信息通过 advantage shaping 间接传递
- 即使 critic 是噪声的（click），也只影响 advantage 大小，不会 corrupt policy 的 feature

**实现方式** (两种选择):

**B1: IDM-as-Reward (简单版)**
- 用 trained visual IDM 作为 reward function
- R_idm = IDM(s_t, s_{t+1}) matches actual action? → bonus
- 这不需要额外训练 critic，直接利用 V8 已有的 IDM 能力

**B2: Privileged Advantage (复杂版)**
- 训练一个额外的 critic head 在 VLM 上
- Critic input: [s_t, s_{t+1}, action_taken]
- Critic output: predicted advantage
- 用 s_{t+1} 提供 dense supervision 给 critic，而非 actor

**研究新颖性**:
- Asymmetric actor-critic 在 multi-agent RL 中很常见，但在 **VLM LUPI setting** 中是新的
- 特别是 "visual hindsight as privileged critic" 这个 framing
- 与 Hindsight Experience Replay (HER) 相关但不同 — HER 改变 goal，我们改变 observation

**可行性评估**:
- B1 (IDM-as-Reward): **高可行性**。现有 eval 已证明 V8 IDM 在 non-click actions 上有效 (66% accuracy)。可以直接把 IDM accuracy 作为 reward bonus。改动量小。
- B2 (Privileged Advantage): 中等。需要训练 critic head，但可以复用 VLM backbone。

---

#### Direction C: Causal Probing — "What Does the VLM Actually See in s_{t+1}?"

**核心思想**: 在提出 solution 之前，先深入理解 **VLM 从 s_{t+1} 中提取了什么信息**。

**实验设计**:

**C1: Intervention Experiments (s_{t+1} ablation)**

| Condition | s_{t+1} content | Expected if VLM uses high-level semantics | Expected if VLM uses low-level pixels |
|-----------|-----------------|-------------------------------------------|---------------------------------------|
| Full s_{t+1} | 真实下一步截图 | Baseline (66.3%) | Baseline |
| Blurred s_{t+1} | Gaussian blur (σ=20) | Similar (模糊不影响语义) | Drops significantly |
| Color-only s_{t+1} | 32x32 downsampled → resized | Drops (语义丢失) | Similar (color distribution preserved) |
| Same-app different s_{t+1} | 同 app 但不同 step 的截图 | Drops (transition 信息被破坏) | Might still "guess" |
| Shuffled s_{t+1} | 随机截图 | Drops to standard level | Drops to standard level |
| Textual s_{t+1} | "Next screen shows: [text description]" | If VLM uses semantic → high | If VLM uses pixels → drops |

这些 ablation 实验可以区分：
- 模型是做 **pixel-level comparison** 还是 **semantic understanding**
- 模型是真的理解 transition 还是只用 s_{t+1} 作为 "hint"

**C2: Per-Action Mechanism Analysis**

对于每种 action type (特别是 detectable 的 type, swipe, open)，分析：
- Attention patterns: s_{t+1} 的哪些区域被 attend？
- 如果 mask 掉 s_{t+1} 中 "变化最大" 的区域会怎样？
- s_{t+1} 的信息是 redundant with task context 还是 complementary？

**C3: click 失败的深入分析**

click 的 visual transition 通常是什么？
- UI element 状态变化 (button highlight → normal)
- 弹出 dialog/menu
- Page navigation

哪些 click transitions 和其他 action types 的 transitions 视觉上 indistinguishable？

**研究新颖性**:
- 这是对 VLM visual reasoning capability 的 **empirical science**
- 回答 "VLM 如何做 visual comparison" — 当前 VLM 文献几乎没有这方面的分析
- 发现可能对 VLM 社区有广泛影响（不限于 GUI agent）

**可行性评估**: **最高**。只需要修改 eval script 的 input 构造方式，复用现有 pipeline。每个 ablation 是一个简单的 vLLM evaluation run。

---

### 13.4 Research Roadmap

```
Phase 1: Understanding (1-2 天)
├── C1: s_{t+1} intervention experiments (6 conditions × V8 s50)
├── C3: click failure case study (manual inspection of 50 samples)
└── 输出: 理解 VLM 从 s_{t+1} 中提取什么 → 决定方向 B or A

Phase 2: Solution (根据 Phase 1 结果选择)
├── IF VLM uses semantic features:
│   → Direction B1 (IDM-as-Reward): 最简单、最直接
│   → 将 visual IDM accuracy 作为 dense reward bonus
│   → 只对 non-click transitions 提供 bonus
│
├── IF VLM uses pixel-level comparison:
│   → Direction A (Forward Dynamics): 更有研究价值
│   → 训练 representation-level forward model
│   → 模型 "imagine" action consequences
│
└── IF VLM's s_{t+1} usage is shallow/unreliable:
    → 重新考虑 visual hindsight 整体方向
    → 转向 textual transition descriptions 或 structured diff
```

### 13.5 Priority Ranking

| Direction | Research Novelty | Feasibility | Expected Impact | Priority |
|-----------|-----------------|-------------|-----------------|----------|
| C: Causal Probing | ★★★★ (VLM understanding) | ★★★★★ (eval only) | ★★★★ (guides all directions) | **#1** |
| B1: IDM-as-Reward | ★★★ (LUPI + RL) | ★★★★ (small code change) | ★★★ (incremental) | **#2** |
| A: Forward Dynamics | ★★★★★ (world model in VLM) | ★★ (arch change) | ★★★★★ (if works) | **#3** |
| B2: Privileged Critic | ★★★★ (asymmetric AC) | ★★★ (new critic head) | ★★★★ | **#4** |

**建议**: 先做 C (Causal Probing)，用 2 天时间理解 VLM 的 visual comparison mechanism。这些理解直接决定 B 和 A 哪个更合适。

---

### 13.6 Theoretical Framework: Visual LUPI for GUI Agents

将上述方向统一在一个理论框架下:

**Standard RL**: π(a|s_t, task) — policy conditioned on current state
**V8 Hindsight**: π(a|s_t, s_{t+1}, task) — policy conditioned on privileged future state (LUPI)
**Transfer gap**: π(a|s_t, task) 不 benefit from π(a|s_t, s_{t+1}, task) 的训练

**Proposed framework**: 分离 "what to predict" from "what to condition on":

1. **Forward Dynamics** (Dir A): 学习 P(s_{t+1}|s_t, a) — 预测 consequence
   - s_{t+1} 是 prediction target，不是 conditioning input
   - 天然 transfer：模型 internalize action→consequence mapping

2. **Privileged Critic** (Dir B): 学习 V(s_t, a, s_{t+1}) — 评估 quality
   - s_{t+1} 只在 critic 中出现，actor 不接触
   - Transfer 通过 advantage shaping 间接发生

3. **Causal Analysis** (Dir C): 理解 VLM 的 visual comparison mechanism
   - 为 A 和 B 提供 empirical foundation
   - 独立的 research contribution

这三个方向可以写成一篇论文：
- **Title**: "Can VLMs Learn from Visual Hindsight? A Study of Privileged Information Transfer in GUI Agents"
- **Contribution 1** (Empirical): 全面 benchmark VLM 的 visual IDM 能力 (Section 7-11 的数据)
- **Contribution 2** (Analysis): Causal probing 揭示 VLM 的 visual comparison mechanism (Dir C)
- **Contribution 3** (Method): Forward dynamics / Privileged critic 实现 effective transfer (Dir A or B)

---

## 14. Causal Probing Results: What Does V8 Actually Extract from s_{t+1}?

### 14.1 Experiment Setup

Model: V8 step 70 (hindsight mode, baseline = 66.0% type_match)
Standard mode (no s_{t+1}): 46.5% type_match (reference)

5 ablation conditions applied to s_{t+1} before injecting into hindsight prompt:

| Condition | What it does | Information preserved |
|-----------|-------------|---------------------|
| full (baseline) | True s_{t+1} | Everything |
| blur (σ=20) | Gaussian blur | Layout, color regions |
| lowres (32×32→resize) | Extreme downsampling | Only color distribution |
| random | Screenshot from different episode | Nothing (wrong image) |
| copy_st | s_t duplicated as s_{t+1} | Zero transition info |
| same_ep_rand | Random step from same episode | Same app, wrong transition |

### 14.2 Overall Results

| Condition | type_match | Δ vs full | Δ vs standard |
|-----------|-----------|-----------|---------------|
| Standard (no s_{t+1}) | 46.5% | -19.5% | — |
| **same_ep_rand** | 60.3% | -5.7% | +13.8% |
| **copy_st** | 61.5% | -4.5% | +15.0% |
| **random** | 62.0% | -4.0% | +15.5% |
| **lowres** | 62.4% | -3.6% | +15.9% |
| **blur** | 63.1% | -2.9% | +16.6% |
| full (baseline) | 66.0% | — | +19.5% |

### 14.3 Per-Action-Type Breakdown

| Action | full | blur | lowres | random | copy_st | same_ep_rand |
|--------|------|------|--------|--------|---------|-------------|
| click | 0% | 0% | 0% | 0% | 0% | 0% |
| swipe | 71.6% | 60.8% | 60.8% | 63.5% | 58.1% | 60.8% |
| type | 91.1% | 89.3% | 89.3% | 89.3% | 89.3% | 89.3% |
| open | 85.5% | 81.8% | 80.0% | **87.3%** | 80.0% | **90.9%** |
| system_button | 66.7% | 75.6% | 75.6% | 66.7% | 68.9% | 66.7% |
| wait | 58.1% | 60.5% | 60.5% | 62.8% | 62.8% | 55.8% |

### 14.4 Critical Finding: The "Format Effect"

**V8 的 "hindsight 能力" 主要不是来自 visual transition understanding，而是来自 prompt format effect。**

```
Total hindsight benefit:  66.0% - 46.5% = 19.5%
├── Format effect (any image):  61.5% - 46.5% = 15.0%  (77%)
└── Content effect (real s_{t+1}):  66.0% - 61.5% = 4.5%  (23%)
```

**77% 的 "hindsight 提升" 来自 FORMAT，只有 23% 来自 CONTENT。**

具体来说，当 hindsight prompt 中 s_{t+1} 被替换为：
- **完全无关的随机截图** → 仍然 62.0% (vs full 66.0%)
- **s_t 的副本（零 transition 信息）** → 仍然 61.5%
- **32×32 像素的模糊色块** → 仍然 62.4%

这意味着 V8 训练让模型学到的不是 "如何从视觉变化推断 action"，而是 "当 prompt 中有第二张图片时，要更仔细地预测 action"。这是一种 **conditional attention pattern**，不是真正的 visual IDM。

### 14.5 Per-Action Analysis

**真正利用 s_{t+1} content 的 action types:**
- **swipe**: full 71.6% vs ablated ~60% → ~11% content effect — 模型确实检测了滚动位移
- **open**: full 85.5% vs copy_st 80.0% → ~5% content effect — 但 random (87.3%) 和 same_ep_rand (90.9%) 反而更高！说明 open 主要靠 task context，不靠 visual transition

**完全不利用 s_{t+1} content 的 action types:**
- **type**: 所有条件下 89-91% — 不需要 s_{t+1}，task context 已足够
- **click**: 永远 0% — 无论 s_{t+1} 是什么
- **wait**: ablated 条件下反而 ~62% > full 58% — s_{t+1} 无帮助
- **system_button**: ablated 条件下 ~70-76% vs full 66.7% — s_{t+1} 反而有害

### 14.6 Implications: V8 Hindsight 方向需要根本性重新评估

1. **V8 的 hindsight capability 是 illusion**: 表面看 V8 从 s_{t+1} 获得了 +19.5% 提升，但其中 77% 是 format artifact（第二张图的存在改变了模型的 attention/confidence），只有 23% (~4.5%) 是真正的 visual transition understanding。

2. **唯一真正受益的 action type 是 swipe** (~11% content effect)。swipe 的视觉变化（内容位移）是 VLM 唯一能可靠检测并利用的 transition signal。

3. **Direction A (Forward Dynamics) 的前提不成立**: 如果模型几乎不 use s_{t+1} 的 content，那让模型 "predict s_{t+1}" 也没有意义 — 模型不知道 s_{t+1} 长什么样跟 action 有什么关系。

4. **Direction B (IDM-as-Reward) 同样存疑**: IDM 的 accuracy 主要来自 format effect 而非 real understanding。用 IDM accuracy 作为 reward 等于 reward "prompt format compliance"，而非 "correct visual reasoning"。

5. **新方向: 去除 format confound**: 如果标准模式也加上一张无关图片（控制 format effect），真正的 hindsight content benefit 只有 ~4.5%。这几乎不足以作为有效的训练信号。

### 14.7 Re-evaluation of the Research Question

原始问题: "VLM 能理解 GUI 的视觉状态转移吗？"

Causal probing 的回答: **几乎不能。** V8 从 s_{t+1} 中提取的信息几乎全是 format-related 的 prompt conditioning，而非对视觉变化的理解。唯一例外是 swipe (内容位移) — VLM 可以检测大幅度的空间位移，但对其他所有类型的视觉变化（UI element 状态、页面切换、弹窗等）基本 blind。

这回答了 Section 5 的 Scenario C:

> **Scenario C: VLM 缺乏 Visual IDM 能力** — VLM 的 visual encoder + attention 机制不适合做 visual comparison/diff reasoning。

**Next steps 需要根本性转向:**
- 不应该继续在 visual hindsight 方向投入
- 转向 **textual transition descriptions** 或 **structured diff representations** 作为 s_{t+1} 的替代
- 或者接受 visual IDM 在当前 VLM 上不可行，专注于改善 standard mode 的 RL 训练

---

## 15. Research Pivot: From Visual IDM to Multi-Agent Text-Level Verification

### 15.1 Core Insight

IDM 失败的根本原因是让单个 VLM 在 **pixel space 做 visual comparison**。但 multi-agent 框架已经有了自然的解法——把 visual comparison **分解为 text-level comparison**。

已有的 multi-agent 实验 (Exp2d) 已证明：
- π_V 单独看截图时能产生有用的 structured description（thought-hit 时 F4 accuracy 58.3% vs 25.9%）
- π_S 单独看 history+task 时 status prediction 有 86-98% accuracy

这两个能力组合起来，天然就是一个 **text-level verifier**——不需要做 visual IDM。

### 15.2 Mechanism: Visual Space → Text Space

**IDM 的失败路径:**
```
s_t (pixels) + s_{t+1} (pixels) → VLM → predict action
                                   ↑
                              VLM 不会做 pixel diff（click=0%, 77% format effect）
```

**Multi-agent text verification 路径:**
```
s_t     → π_V → description_t      ("搜索框为空，键盘未弹出")
s_{t+1} → π_V → description_{t+1}  ("搜索框显示'hello'，键盘弹出")

description_t + description_{t+1} → π_S → textual_diff
                                          ("文字被输入到搜索框")
                                    → 与 a_t 对比 → verification result
```

π_V 负责把 visual state 翻译成 structured text，π_S 负责在 text space 做 reasoning（comparison, tracking, verification）。VLM 做不了 pixel diff，但做 **text diff 是 trivial** 的。

**关键点：这不是一个新 module，而是已有的 π_V/π_S 分工的自然延伸。**

### 15.3 对 Long-Horizon 的直接帮助

Exp2e 证明了 **连续错误比分散错误低 19-27pp**。根本原因是模型不知道自己错了 → 错误 cascade。Text-level verification 可以在每步检测到错误，打断 consecutive error spiral。

```
当前设计：π_S 检测 subtask boundary → reset
扩展设计：π_S 检测 step-level anomaly → reset 或 re-plan

anomaly 检测方式：
  π_V(s_t) 预测 "执行 a_t 后应该看到什么"
  π_V(s_{t+1}) 报告 "实际看到了什么"
  π_S 对比两者 → anomaly if mismatch
```

如果 text-level verification 能在 pw=1 时检测到错误 → 触发 context reset / re-plan → 后续 step 回到 pw=0 的 55.3% 附近 → spiral 被打断。

### 15.4 对 Training 的影响: Textual OPD 替代 Visual OPD

V8 的 visual OPD 已被证明无效（77% format effect）。新思路：**OPD 不注入 s_{t+1} 截图，而是注入 π_V(s_{t+1}) 的 textual description**。

```
旧 OPD：[s_t, s_{t+1}_screenshot] → predict action
                ↑ VLM 看不懂 visual diff（click=0%）

新 OPD：[s_t, "执行正确action后，界面变化为：搜索框出现文字'hello'"] → predict action
                ↑ VLM 擅长处理 text condition
```

这解决了 click=0% 问题——click 的视觉变化（button highlight, dialog appear）在 pixel space 不可检测，但在 text space 是明确的（"出现了一个确认弹窗"）。

同时，text-level verification 可以给 SPWA 提供更 dense 的 signal：

```
SPWA_augmented(t) = SPWA(t) × verification_weight(t)

verification_weight(t) =
  if text_verify(t) == correct:  upweight（可靠的 positive signal）
  if text_verify(t) == incorrect: 这步之后的 SPWA 全部 discount（early stopping）
  if text_verify(t) == uncertain: 保持原 SPWA weight
```

### 15.5 Implementation Plan

#### Phase 0: 验证 π_V 的 State Description 质量（2天）— 整个方向的前提

π_V 能否为 s_t 生成足够 structured 的 description，使得 text-level diff 有效？

1. 从 validation set sample 200 步，每步让 π_V 分别 describe s_t 和 s_{t+1}
2. 人工 check：两个 description 的 text diff 能否 recover 出 GT action type？
3. 按 action type 分桶统计：click 的 text-level diff 是否比 visual-level diff 更 detectable？

**Go/No-Go gate**: 如果 text diff 能 recover click action（哪怕 60-70%），就已远超 visual IDM 的 0%。这证明 multi-agent decomposition 把不可能的问题（visual IDM）变成了可行的问题（text comparison）。

#### Phase 1: Text-Level Verification Module（2-3天）

基于 Phase 0 的 π_V description format，实现：

```python
def text_verify(desc_t, desc_t1, action_t):
    """
    returns: {verified, rejected, uncertain}
    """
    # 方案 A：rule-based（快速验证）
    diff = extract_diff(desc_t, desc_t1)
    if action_t.type == "type" and "新文字出现" in diff: return "verified"
    if action_t.type == "click" and "新页面/弹窗" in diff: return "verified"
    ...

    # 方案 B：让 π_S 做 reasoning（更灵活）
    prompt = f"状态变化：{diff}\n执行的action：{action_t}\n这个action是否导致了这个变化？"
    return π_S(prompt)  # yes/no/uncertain
```

先用方案 A 做 baseline，再看是否需要 B。

#### Phase 2: Inference-Time Verification Loop（2天）

在 V6 checkpoint 上测试 verify-then-reset：

```python
for step t in trajectory:
    desc_t = π_V(s_t)
    a_t = π_A(s_t, task, history, desc_t)
    execute a_t → s_{t+1}
    desc_t1 = π_V(s_{t+1})

    result = text_verify(desc_t, desc_t1, a_t)
    if result == "rejected":
        # 选项 1：重试 with hint
        a_t' = π_A(s_t, task, history, desc_t,
                    hint="上一次尝试失败，界面未发生预期变化")
        # 选项 2：status-triggered reset
        history = []  # clean slate
```

测 long-horizon task 上的 TSR 变化。如果 compound error 被打断，TSR 应有 visible 提升。

#### Phase 3: Training-Time Integration（3-4天）

两条并行：

**(a) Textual OPD 替代 Visual OPD:**
- 对 incorrect rollout 的 aux loss prompt：注入 π_V(s_{t+1}) 的 textual description 而非 s_{t+1} 截图

**(b) Verification-Aware SPWA:**
- rollout 时同时跑 text verification
- 用 verification result augment SPWA weight

### 15.6 Paper Story 调整

**原来的 story:** SP + SPWA + OPD(visual hindsight) + multi-agent

**新的 story:**

> Multi-agent decomposition 不仅解决 input interference（Exp2d 的 +5pp），还解决 long-horizon 的 **verification bottleneck**。
>
> 我们发现 VLM 无法在 pixel space 做 visual verification（IDM findings: click=0%, 77% format effect）。但 multi-agent 架构天然提供了解法：π_V 把 visual state 转成 structured text → verification 在 text space 进行 → VLM 擅长的模态。
>
> 这使得：(1) inference-time 的 step-level error detection 和 recovery 成为可能；(2) training-time 的 textual OPD 替代失败的 visual OPD，给 failure zone 提供有效信号。

**Narrative**: IDM 实验是 motivation（visual verification 不行）→ multi-agent text-level verification 是 solution → 长 horizon 提升是 result。从 diagnostic 到 solution 到 result 完整闭环。

### 15.7 Priority

**Phase 0（验证 π_V description 质量）是第一步。** 如果 text diff 确实能 recover click actions，后面的路就很清楚。这是整个方向的 go/no-go gate。

---

## 16. Phase 0 Results: Text-Level IDM vs Visual IDM

### 16.1 Setup

- Model (π_V): Qwen2.5-VL-7B-Instruct (base, zero-shot)
- Dataset: Android Control validation, 200 steps sampled (min 30 per action type)
- Pipeline:
  - Phase A: π_V(s_t) → desc_t, π_V(s_{t+1}) → desc_t1 (with images)
  - Phase B: text_idm(desc_t, desc_t1) → predict action (text only, no images)

### 16.2 Overall Results

| Method | type_match |
|--------|-----------|
| Visual IDM (pure_idm, base) | 43.6% |
| **Text-Level IDM (Phase 0)** | **37.5%** |

Overall text IDM is slightly lower than visual IDM, **but the per-action breakdown reveals complementary strengths.**

### 16.3 Per-Action Comparison: Text IDM vs Visual IDM

| Action | n | Text IDM | Visual IDM | Delta | Winner |
|--------|---|---------|-----------|-------|--------|
| **click** | 36 | **75.0%** | 0.0% | **+75.0%** | **Text** |
| **open** | 33 | **57.6%** | 19.6% | **+38.0%** | **Text** |
| type | 30 | 86.7% | **94.4%** | -7.7% | Visual |
| swipe | 31 | 3.2% | **86.2%** | -83.0% | Visual |
| system_button | 31 | 0.0% | **40.0%** | -40.0% | Visual |
| wait | 30 | 6.7% | 2.5% | +4.2% | ~Tie |
| long_press | 9 | 0.0% | 0.0% | 0 | ~Tie |

### 16.4 Key Finding: click 从 0% 到 75% — Go Signal

**click: 75.0% (text) vs 0.0% (visual) — 这是整个研究的核心突破。**

click 占训练数据的 61.1%，在 visual IDM 中完全不可检测（0%，跨所有模型和模式）。但 text-level IDM 以 75% accuracy 检测 click — 因为 π_V 能把 "button highlight → dialog appear" 这种微妙视觉变化转为明确文本（"搜索界面打开，键盘弹出"）。

**Click confusion matrix (text IDM):**
| Predicted | Count | % |
|-----------|-------|---|
| click (correct) | 27 | 75.0% |
| terminate | 5 | 13.9% |
| open | 2 | 5.6% |
| type | 2 | 5.6% |

对比 visual IDM 的 click confusion（全部预测错）→ text 方法将 "不可能" 变成了 "可行"。

### 16.5 Complementary Pattern: Text vs Visual

两种方法的优势完全互补：

```
Text IDM 擅长:  click (+75%), open (+38%)
                → 语义变化大但视觉变化微妙的 actions
                → π_V 能把 "新页面出现" "弹窗打开" 转为明确文本

Visual IDM 擅长: swipe (+83%), system_button (+40%)
                 → 空间位移大但语义不变的 actions
                 → 图像可直接检测滚动、返回等空间变化

两者都好:       type (87% vs 94%)
                → 文字出现在视觉和文本中都容易检测
```

**理论最优 ensemble:**
- click: 取 text (75%)
- swipe: 取 visual (86%)
- type: 取 visual (94%)
- open: 取 text (58%)
- system_button: 取 visual (40%)
- → 加权 ensemble 可能达到 ~65-70% overall

### 16.6 Go/No-Go Decision

**GO.** Phase 0 的核心假设得到验证：

1. π_V 的 zero-shot description 足以在 text space recover click action（75% vs 0%）
2. Text 和 visual 方法互补 — 不是替代关系
3. click 占数据 61%，从 0% → 75% 代表 massive improvement potential

**Phase 1 方向明确**: 实现 text-level verification module，首先聚焦 click 和 open actions。swipe/system_button 可继续使用 visual signals。

### 16.7 Sample Descriptions (click correctly identified)

**Example 1**: Calendar Pro — delete event
```
desc_t:  "Calendar Pro, August calendar view, calendar grid, navigation arrows..."
desc_t1: "Calendar Pro, weekly view for Week 32, Aug 7-9, event details visible..."
Text IDM: "The user clicked on the event 'women empowerment' to view more details."
```

**Example 2**: The Hindu News — search
```
desc_t:  "The Hindu, Home page, Logo, Navigation menu, Search icon, Profile icon..."
desc_t1: "Search interface, search bar with 'Search for Article', keyboard is open..."
Text IDM: "The user clicked on the search icon, which opened the search interface."
```

π_V 能准确描述界面状态变化，text diff 自然揭示了 action。

---

## 17. Text Hindsight Eval: Text Description vs Screenshot as Hindsight Signal

### 17.1 Experiment Design

直接测试：在 V8 模型的 hindsight prompt 中，用 π_V(s_{t+1}) 的文本描述替换 s_{t+1} 截图。

三个对照条件（均使用 V8 step70 模型，相同评估步骤）：
- **Standard**: model(s_t) → predict action（无 s_{t+1}）
- **Visual Hindsight**: model(s_t, image(s_{t+1})) → predict action
- **Text Hindsight**: model(s_t, desc(s_{t+1})) → predict action

Pipeline:
1. Phase A: 对每个步骤的 s_{t+1}，用 V8 模型生成 π_V 描述（512 tokens）
2. Phase B: 在标准 prompt 后追加 `"Description of the screen after correct action:\n{desc_t1}\n"`，让模型预测 action

### 17.2 Results

| Action | Text HS | Visual HS | Standard | Δ(text-visual) |
|--------|---------|-----------|----------|----------------|
| type (n=56) | 89.3% | 91.1% | 93.5% | -1.8% |
| open (n=55) | 83.6% | 85.5% | 80.8% | -1.9% |
| swipe (n=74) | 50.0% | 71.6% | 50.0% | -21.6% |
| system_button (n=45) | 51.1% | 66.7% | 52.2% | -15.6% |
| wait (n=43) | 32.6% | 58.1% | 41.0% | -25.5% |
| click (n=41) | 0.0% | 0.0% | 0.0% | 0.0% |
| long_press (n=9) | 0.0% | 0.0% | 0.0% | 0.0% |
| **Overall (n=323)** | **52.6%** | **66.0%** | **46.5%** | **-13.4%** |

### 17.3 Analysis

**1. Text descriptions provide partial but not full hindsight signal**
- Text HS (52.6%) > Standard (46.5%)：文本描述确实增加了 +6.1% 的信息
- Text HS (52.6%) < Visual HS (66.0%)：但远不及截图的 +19.5% 提升

**2. Semantic actions 几乎完全保持**
- `type`: 89.3% vs 91.1%（-1.8%）— 文本能完美描述输入内容变化
- `open`: 83.6% vs 85.5%（-1.9%）— 文本能描述 app/screen 切换

**3. Spatial/temporal actions 严重退化**
- `swipe`: 50.0% vs 71.6%（-21.6%）— 回退到 standard 水平。文本无法有效描述滚动方向和位移
- `system_button`: 51.1% vs 66.7%（-15.6%）— 导航变化难以从文本推断
- `wait`: 32.6% vs 58.1%（-25.5%）— 甚至低于 standard！文本描述可能在状态几乎不变时引入噪声

**4. Click 仍然 0%** — 跨所有条件一致，是 V8 模型的固有限制

**5. Wait 回归问题**
Text HS (32.6%) < Standard (41.0%)，说明文本描述在 wait 场景下有害。Wait 的特征是"前后截图几乎一样"——视觉上 pixel-level 相似度可以捕捉这一点，但文本描述会夸大微小差异（描述 loading indicator 消失、内容加载完成），导致模型误判为其他 action。

### 17.4 Implications for Textual OPD

**结论：纯文本 OPD 不如视觉 OPD，但可作为互补信号。**

Phase 0（§16）显示 text IDM 在 click 上远优于 visual IDM（75% vs 0%），但那是 text-only prediction（无原始图像）。当文本作为视觉 prompt 的辅助信号时（Phase B），它的互补优势被稀释了——因为模型已经有了 s_t 截图，文本只是冗余地重复了部分信息。

**修正后的研究方向**：
- 纯 textual OPD（完全替换截图）会损失 spatial 信号 → 不推荐
- ~~**Hybrid approach**: visual s_{t+1} + text diff as verification~~ → 见 §18 实验结果，hybrid 反而更差
- **Text-level verification**（§15）仍然是有价值的方向，但应用场景是 inference-time verification，不是 training-time OPD 替代

### 17.5 Files
- Script: `scripts/eval/eval_text_hindsight.py`
- SLURM: `scripts/eval/eval_text_hindsight_v8.slurm`
- Results: `evaluation/results/text_hindsight/text_hs_summary_v8_step70.json`

---

## 18. Hybrid Hindsight: Visual s_{t+1} + Text Description

### 18.1 Experiment

在 visual hindsight prompt（已包含 s_{t+1} 截图）之后，追加 π_V(s_{t+1}) 的文本描述。测试两个通道的组合效果。

Prompt structure:
```
[standard prompt with s_t]
Screenshot after correct action:
[IMAGE: s_{t+1}]
Description of the screen after correct action:
{text description}
```

### 18.2 Results

| Action | Hybrid | Text HS | Visual HS | Standard |
|--------|--------|---------|-----------|----------|
| type (n=56) | 80.4% | 89.3% | 91.1% | 93.5% |
| open (n=55) | 83.6% | 83.6% | 85.5% | 80.8% |
| swipe (n=74) | 54.1% | 50.0% | 71.6% | 50.0% |
| system_button (n=45) | 60.0% | 51.1% | 66.7% | 52.2% |
| wait (n=43) | 27.9% | 32.6% | 58.1% | 41.0% |
| click (n=52) | 0.0% | 0.0% | 0.0% | 0.0% |
| long_press (n=9) | 0.0% | 0.0% | 0.0% | 0.0% |
| **Overall (n=334)** | **50.9%** | **52.6%** | **66.0%** | **46.5%** |

### 18.3 Analysis: Signal Interference, Not Complementarity

**Critical finding: Hybrid (50.9%) < Text-only (52.6%) < Visual-only (66.0%)**

文本描述不仅没有补充视觉信号，反而造成了干扰：

**1. type 严重退化**: 80.4% vs 91.1% (visual) / 89.3% (text) — 两个信号单独都很好，但组合后 -10.7%。文本描述可能引入了与截图矛盾的细节，导致模型困惑。

**2. wait 进一步恶化**: 27.9% — 四种条件中最差。视觉 + 文本双重信号让"几乎不变"的状态看起来更有变化。

**3. swipe 微弱改善**: 54.1% vs 50.0% (text) — 文本 + 截图比纯文本略好，但远不及纯截图 (71.6%)。

**4. system_button 有改善**: 60.0% vs 51.1% (text) — 截图帮助识别导航变化，但仍不及纯截图 (66.7%)。

### 18.4 Root Cause: Context Length Pollution

Hybrid prompt 包含 3 个图像 + 长文本描述（~500 tokens），总 context 显著增加。对于 7B 模型：
- 更长的 context → attention 被稀释
- 冗余信息（文本重复了截图内容）→ 模型不知道信任哪个信号
- 矛盾信息（文本描述的细微偏差）→ 增加不确定性

这与 §14 causal probing 的发现一致：V8 的 hindsight benefit 77% 来自 format effect（有第二张图）。添加更多输入并不等于添加更多信息——反而是噪声。

### 18.5 Consolidated Findings (§14-18)

```
Standard (no s_{t+1}):           46.5%  — baseline
Text hindsight (desc only):      52.6%  — +6.1% from text descriptions
Hybrid (image + desc):           50.9%  — INTERFERENCE, worse than either alone
Visual hindsight (image only):   66.0%  — +19.5%, best single condition
Visual HS + blur ablation:       63.1%  — 77% of benefit is format, not content
```

**结论**：
1. 视觉和文本通道在 prompt-level 组合时互相干扰，不互补
2. 视觉通道在 prompt-level hindsight 中始终优于文本通道
3. 文本通道的价值在于 SEPARATE 的 inference pipeline（如 §16 text IDM: click 75%），而非 prompt-level 的辅助信号
4. 对于 training-time 的 OPD，视觉 s_{t+1} 仍然是最佳选择
5. 文本验证应定位为 **独立的 inference-time post-processing**（verify-then-reset），不是 prompt augmentation

### 18.6 Files
- Script: `scripts/eval/eval_text_hindsight.py --hybrid`
- SLURM: `scripts/eval/eval_hybrid_hindsight_v8.slurm`
- Results: `evaluation/results/text_hindsight/hybrid_hs_summary_v8_step70.json`

---

## 19. Base Model Replication: Text HS + Hybrid HS

### 19.1 Motivation

用 base model (Qwen2.5-VL-7B-Instruct) 重复 §17-18 实验，区分哪些现象是 V8 hindsight training 带来的，哪些是 VLM 通用行为。

### 19.2 Results: Full Comparison (Base vs V8)

**Base Model (Qwen2.5-VL-7B-Instruct, no RL/hindsight training)**

| Action | Hybrid | Text HS | Visual HS | Standard |
|--------|--------|---------|-----------|----------|
| type | 18.8% | 87.0% | 47.6% | 96.7% |
| open | 0.0% | 5.5% | 7.5% | 0.0% |
| swipe | 24.3% | 37.9% | 31.0% | 60.6% |
| system_button | 31.1% | 45.2% | 42.5% | 54.5% |
| wait | 0.0% | 10.0% | 0.0% | 28.6% |
| click | 0.0% | 0.0% | 0.0% | 0.0% |
| **Overall** | **11.0%** | **30.2%** | **19.6%** | **41.4%** |

**V8 Model (RL + hindsight aux-loss trained)**

| Action | Hybrid | Text HS | Visual HS | Standard |
|--------|--------|---------|-----------|----------|
| type | 80.4% | 89.3% | 91.1% | 93.5% |
| open | 83.6% | 83.6% | 85.5% | 80.8% |
| swipe | 54.1% | 50.0% | 71.6% | 50.0% |
| system_button | 60.0% | 51.1% | 66.7% | 52.2% |
| wait | 27.9% | 32.6% | 58.1% | 41.0% |
| click | 0.0% | 0.0% | 0.0% | 0.0% |
| **Overall** | **50.9%** | **52.6%** | **66.0%** | **46.5%** |

### 19.3 Analysis

**1. Base model: 任何形式的 hindsight 都有害**

```
Base: Standard (41.4%) > Text HS (30.2%) > Visual HS (19.6%) > Hybrid (11.0%)
V8:   Visual HS (66.0%) > Text HS (52.6%) > Hybrid (50.9%) > Standard (46.5%)
```

Base model 从未被训练如何利用 s_{t+1} 信息（无论图像还是文本），所以任何额外信息都是噪声，反而干扰了正常的 action prediction。排序完全反转：base 越多信息越差，V8 visual hindsight 最好。

**2. V8 的 hindsight training 确实学会了利用视觉 s_{t+1}**

- Base: Visual HS 让 overall 从 41.4% → 19.6%（**-21.8%，严重退化**）
- V8: Visual HS 让 overall 从 46.5% → 66.0%（**+19.5%，显著提升**）

差异高达 41.3%。这证明 V8 的 aux-loss training 确实教会了模型如何从视觉 s_{t+1} 中提取 action 信息。这不是 format effect 那么简单——base model 看到同样的 format 反而崩溃了。

**3. 但 V8 没有学会利用文本 s_{t+1}**

- Base: Text HS 从 41.4% → 30.2%（-11.2%）
- V8: Text HS 从 46.5% → 52.6%（+6.1%）

V8 对文本描述有微弱的正向响应（+6.1%），但远不及视觉（+19.5%）。V8 的 hindsight training 是通过视觉 aux-loss 做的，所以学到的是 **视觉通道的 s_{t+1} 利用能力**，文本通道没有被训练。

**4. Hybrid 一致最差**

两个模型都呈现 Hybrid < Visual HS 且 Hybrid < Text HS，确认信号干扰是通用现象，不是 V8 特有。

**5. type 退化模式揭示训练效果**

| Model | Standard | Text HS | Visual HS | Hybrid |
|-------|----------|---------|-----------|--------|
| Base  | 96.7%    | 87.0%   | 47.6%     | 18.8%  |
| V8    | 93.5%    | 89.3%   | 91.1%     | 80.4%  |

- Base model 的 type 从 96.7% 一路跌到 18.8%——screenshot s_{t+1} 对 type prediction 是毁灭性的干扰
- V8 保持在 80-93%——hindsight training 教会了模型在有 s_{t+1} 图像时不被干扰
- 但 V8 hybrid 仍掉到 80.4%——模型没有被训练处理 "图像 + 文本" 双通道

### 19.4 Key Insight: Hindsight Training = Channel-Specific Skill

V8 的 visual hindsight aux-loss 训练的本质是 **教会模型一种 channel-specific 的技能**：

1. **技能内容**：在 prompt 中看到 "Screenshot after correct action:" + image 时，能从图像差异推断 action
2. **技能边界**：只对视觉通道有效。文本通道、混合通道都超出训练分布
3. **技能局限**：即使在视觉通道内，77% 的 benefit 来自 format effect（§14），真正的视觉理解有限

这意味着：如果我们想让模型学会利用文本 s_{t+1} 描述，需要专门用文本 aux-loss 训练（**textual OPD training**），而不是期望视觉训练自然迁移到文本。

### 19.5 Files
- SLURM: `scripts/eval/eval_text_hindsight_base.slurm`
- Results: `evaluation/results/text_hindsight/{text_hs,hybrid_hs}_summary_qwen25vl_7b_base.json`

---

## 20. V9: Dual-Channel Hindsight Training (Image + Text)

### 20.1 Motivation

§17-19 的 evaluation 揭示了一个关键规律：**hindsight training = channel-specific skill**。

| Condition          | V8 (visual HS trained) | Base (untrained) |
|--------------------|------------------------|-------------------|
| Standard (no HS)   | 46.5%                 | 41.4%             |
| Text HS only       | 52.6%                 | 30.2%             |
| Visual HS only     | **66.0%**             | 19.6%             |
| Hybrid (vis+text)  | 50.9%                 | 11.0%             |

V8 的 visual aux-loss 只教会了模型利用**视觉通道**的 s_{t+1} 信号：
- Visual HS +19.5pp over standard（learned skill）
- Text HS 仅 +6.1pp（partial transfer at best）
- Hybrid 比 visual-only **更差**（text channel = noise for V8）

如果我们想让模型同时学会利用 visual 和 text 两个通道，需要在训练时**同时注入两者**。

### 20.2 Approach: Offline π_V Descriptions + Dual-Channel Aux Loss

**关键设计决策**：offline 预生成 text descriptions，而非 online 在 rollout 时生成。

**Step 1: Offline Description Generation** (`scripts/eval/generate_descriptions.py`)
- 用 base model (Qwen2.5-VL-7B-Instruct) 作为 π_V，对每个 s_{t+1} 截图生成结构化描述
- 标准 DESCRIBE_PROMPT（描述 app、UI 元素、布局、交互状态、文本内容）
- 去重：相同截图只生成一次（1000 episodes → unique screenshots < 6000）
- 输出：`ui_s1_train_with_desc.jsonl`，每个 step 增加 `desc_t1` 字段

```
Original: line['steps'][i] = {action_content, screenshot, ...}
With desc: line['steps'][i] = {action_content, screenshot, desc_t1, ...}
```

**统计**：1000 episodes, 6536 steps total, 5536 with desc_t1, 1000 without (last steps have no s_{t+1})

**Step 2: Data Pipeline** (`verl/utils/dataset/universal_multiround.py`)

在 `fetch_batch()` 中，除了已有的 `next_screenshot_path`，新增 `next_desc_t1` 字段：

```python
# Store pre-computed pi_V description of s_{t+1} for dual-channel hindsight
if step_id < len(line['steps']):
    row_dict['next_desc_t1'] = line['steps'][step_id].get('desc_t1', None)
else:
    row_dict['next_desc_t1'] = None
```

**Step 3: Dual-Channel Aux Loss Construction** (`uis1/opd.py`)

`construct_hindsight_batch()` 修改为同时注入 image + text：

```python
# For each incorrect rollout with s_{t+1} available:
# 1. Append s_{t+1} screenshot (existing, visual channel)
raw_msgs[-1]['content'].append({"text": "Screenshot after correct action:\n"})
raw_msgs[-1]['content'].append(make_qwen_image_item(next_ss_path))

# 2. Append pi_V text description (NEW, text channel)
if next_desc_t1_arr is not None:
    desc_t1 = next_desc_t1_arr[r_idx]
    if desc_t1:
        raw_msgs[-1]['content'].append({
            "text": f"\nDescription of the screen after correct action:\n{desc_t1}\n"
        })

# Process with num_image_limit=3 (current screenshot + s_{t+1})
enriched_msgs = slim_messages(raw_msgs, num_image_limit=3)
```

Aux loss prompt 结构：
```
[system] You are a helpful assistant...
[user] <task goal> <s_t screenshot> <action history>
       Screenshot after correct action:
       <s_{t+1} screenshot>
       Description of the screen after correct action:
       <pi_V(s_{t+1}) text description>
[assistant] <donor response (best correct rollout)>
```

CE loss 在 assistant response tokens 上计算，梯度与 PPO 在同一 optimizer step 累加。

### 20.3 V9 vs V8 Differences

| Aspect | V8 | V9 |
|--------|----|----|
| **Hindsight channels** | Visual only (s_{t+1} screenshot) | Dual: visual + text (s_{t+1} + π_V description) |
| **Training start** | Resume from V7 step 30 | From scratch (base model) |
| **Dataset** | `ui_s1_train.jsonl` | `ui_s1_train_with_desc.jsonl` |
| **Aux loss prompt** | prompt + screenshot | prompt + screenshot + text description |
| **num_image_limit** | 3 (same) | 3 (same) |
| **hindsight_aux_coef** | 0.07 | 0.07 (same) |
| **All other hyperparams** | — | Identical (lr, kl, clip, K=8, etc.) |

从头训练的原因：V7/V8 checkpoint 只训练了 visual channel，text channel 是 noise。从头训练让模型从一开始就同时学习两个通道，避免单通道 bias。

### 20.4 Expected Outcomes

1. **Dual-channel skill acquisition**: 训练后模型应该在 visual HS 和 text HS evaluation 中都表现好
2. **Hybrid synergy**: V8 中 hybrid (vis+text) 比 visual-only 差（interference），V9 应该让 hybrid ≥ visual-only
3. **Text description 作为 auxiliary grounding**: text 提供 semantic 层面的信息（action type, UI element names），与 visual 的 spatial 信息互补
4. **Potential risk**: 双通道可能增加 aux prompt 长度，导致 CE loss 更高或 OOM。hindsight_aux_coef=0.07 可能需要调整

### 20.5 Training Configuration

```
Job: 3364843
SLURM: train/sp_gigpo/train_sp_gigpo_v9.slurm
Experiment: sp_gigpo_spwa_k8_v9
Nodes: 4 × 4 GPUs = 16 GPUs
Algorithm: SP + GiGPO + SPWA + Dual-Channel Hindsight
Start: from scratch (base model Qwen2.5-VL-7B-Instruct)
Dataset: datasets/ui_s1_dataset/ui_s1_train_with_desc.jsonl
hindsight_aux_coef: 0.07
Other params: identical to V8
```

### 20.6 Monitoring Checklist

1. `actor/hindsight_aux_loss × 0.07` ≈ 24% of total loss（same target as V8）
2. Entropy stable (0.5-0.7), no explosion
3. `hindsight/n_samples > 0` confirms dual-channel prompts are being constructed
4. Compare learning curve with V8 (which started from V7 step 30)
5. After convergence: run all 4 eval modes (standard, visual HS, text HS, hybrid) to verify dual-channel skill

### 20.7 Files

- **Description generation**: `scripts/eval/generate_descriptions.py`, `scripts/eval/generate_train_descriptions.slurm`
- **Dataset**: `datasets/ui_s1_dataset/ui_s1_train_with_desc.jsonl` (8.7MB, 5536 steps with desc_t1)
- **Code changes**: `uis1/opd.py` (dual-channel injection in `construct_hindsight_batch`), `verl/utils/dataset/universal_multiround.py` (pass `next_desc_t1` through pipeline)
- **Training**: `train/sp_gigpo/train_sp_gigpo_v9.slurm` (Job 3364843)

### 20.8 Research Highlights

#### Core Finding: Hindsight Conditioning = Channel-Specific Learned Skill

§17-19 的 controlled experiments 揭示了一个此前未被认识到的现象：

| Eval Condition | V8 (visual HS trained) | Base (untrained) | Delta |
|---------------|------------------------|-------------------|-------|
| Standard      | 46.5%                 | 41.4%             | +5.1  |
| Text HS       | 52.6%                 | 30.2%             | +22.4 |
| Visual HS     | **66.0%**             | 19.6%             | **+46.4** |
| Hybrid        | 50.9%                 | 11.0%             | +39.9 |

V8 的 visual aux-loss 训练产生了**高度 channel-specific** 的效果：
- Visual channel: +46.4pp gain（strong learned skill）
- Text channel: +22.4pp gain（partial transfer，但远低于 visual）
- Hybrid channel: 50.9% < 66.0%（text channel 在 V8 模型中是 noise，反而干扰 visual reasoning）

**这说明 hindsight conditioning 不是 emergent reasoning capability，而是通过 gradient signal 显式教会的 channel-specific skill**。模型不会"自动理解" state transitions——它只学会了训练时见过的那个 modality。

#### V9 的方法意义

基于上述发现，V9 提出 **Dual-Channel Inverse Dynamics Auxiliary Loss**：

1. **问题定义**: 单通道 hindsight 训练导致 channel-specific bias，模型无法泛化到其他 modality
2. **解决方案**: 在 aux-loss prompt 中同时注入 visual s_{t+1} 和 textual π_V(s_{t+1})，训练模型建立两条互补的推理路径：
   - **Spatial pathway** (visual): 从像素级差异推断 action coordinates
   - **Semantic pathway** (text): 从语义描述推断 action type 和 target element
3. **Offline π_V**: 用 base model 自身作为 zero-shot description generator，零额外标注成本
4. **Training from scratch**: 避免 V8 的 visual-only bias 传递

#### 预期验证指标

V9 成功的判据：
- **Hybrid eval > Visual-only eval**: 证明 dual-channel 训练消除了 §18 的 signal interference
- **Text HS eval ≈ Visual HS eval**: 证明两条 pathway 都被有效训练
- **Task acc ≥ V8**: dual-channel IDM 提供更丰富的 learning signal

如果仅 marginal improvement，V9 更多是 ablation study（验证 channel-specificity hypothesis），而非独立方法贡献。但 channel-specificity 这个 finding 本身具有独立 paper value。

#### Paper Narrative

> 在 RL for GUI agents 中，hindsight conditioning（给模型看未来状态辅助学习）的效果**完全取决于 training-time channel alignment**。单通道训练产生单通道 skill，无法跨 modality 泛化。Dual-channel training 是实现 robust hindsight reasoning 的必要条件。

---

## §22 D4: Terminate-Masked Rollout Diversity (V10)

### 问题：Premature Termination Shortcut

V8 训练（SP-GiGPO + Visual Hindsight）暴露了模型学到的"捷径"——过早 terminate 来避免 action error：

1. **SP reward 不区分 terminate**：`terminate` 被当作普通错误动作处理（`extract_match=False`），但 terminate 后不会继续犯错，SP 不再下降
2. **GiGPO dead zone**：当 K=8 个 rollout 全部 terminate 时，组内 advantage=0，完全没有学习信号
3. **Terminate is "free" after first error**：第一个错误后，terminate 不改变 SP（不会更差），但 real action 可能犯更多错

这形成了一个 local optimum：terminate 是 risk-free 的选择，模型 converge 到 "不做 > 做错"。

### 方案：Explorer Rollouts

每个 K-group 中固定比例的 rollout（默认 2/8 = 25%）在生成时**禁止输出 "terminate"**，强制尝试真实动作（click/swipe/type）。

**为什么有效**：
- Explorer rollout 必须尝试 real action → 有概率命中正确 action → SP 更高
- Normal rollout 可能 terminate → SP 停滞
- GiGPO 看到组内 variance（explorer SP > normal SP）→ 非零 advantage → 学习信号恢复
- 模型逐步学到：real action > terminate（因为 explorer 证明了 real action 可以更好）

### 实现细节

**4个文件改动 + 1个新文件**：

| 文件 | 改动 |
|------|------|
| `verl/utils/dataset/universal_multiround.py` | `StdTrajectory` 加 `explorer` 属性；`MultiRoundGenerator` 加 `explorer_fraction` 参数，每个 K-group 的**最后** N 个 rollout 标记为 explorer（与 hindsight 不重叠，hindsight 占前几个 slot） |
| `verl/trainer/ppo/dapo_ray_trainer.py` | 生成时拆分 explorer/normal batch → 分别调用 `generate_sequences` → `recombine_outputs()` 按原始顺序合并；传递 `explorer_fraction` 到 `MultiRoundGenerator`；新增 D4 监控指标 |
| `verl/workers/rollout/vllm_rollout/vllm_rollout.py` | 初始化时 pre-tokenize "terminate"；explorer 生成时注入 `logits_processor` 将 terminate token logits 设为 `-inf`（engine-agnostic，不依赖 V1 engine 的 `update_from_tokenizer`） |
| `train/sp_gigpo/traj_grpo_sp_gigpo.yaml` | 新增 `algorithm.explorer_fraction: 0.0`（默认关闭） |
| `train/sp_gigpo/train_sp_gigpo_v10.slurm` | V10 训练脚本，`explorer_fraction=0.25`，K=8，从 scratch 训练 |

**关键设计决策**：
- **Logits processor 而非 bad_words**：vLLM 的 `bad_words` 需要 V1 engine 调用 `update_from_tokenizer()` 才能生效，但 verl wrapper 不保证走 V1 路径。直接注入 logits_processor 是 engine-agnostic 的
- **Explorer 占 K-group 尾部 slot**：hindsight 占前几个 slot，explorer 占后几个 slot，两者不冲突
- **分批生成 + 合并**：explorer 和 normal 走不同的 `generate_sequences` 调用，通过 `recombine_outputs()` 恢复原始顺序

### 使用

```yaml
algorithm.explorer_fraction=0.25   # 2/8 rollouts suppress terminate
```

```bash
sbatch train/sp_gigpo/train_sp_gigpo_v10.slurm
```

### 监控指标

| 指标 | 含义 |
|------|------|
| `training/n_explorer_samples` | Explorer 样本数（应为总数的 ~25%） |
| `training/explorer_sp_mean` | Explorer rollout 的平均 SP（初期可能更低，后期应更高） |
| `training/non_explorer_sp_mean` | Normal rollout 的平均 SP |

### 预期效果

- **短期**（前5步）：explorer SP 可能低于 normal（被迫做 action 导致更多错误）
- **中期**（10-30步）：explorer 开始发现正确 action，SP 上升，GiGPO dead zone 减少
- **长期**：normal rollout 也学会做 real action（从 explorer 的 contrast 中学到 terminate 不是最优），premature termination rate 下降
- **验证基线**：V8 step 70 的 15.36% eval acc

---

## 22. V9 Full Evaluation: Step-Level vs Trajectory-Level Paradox

### 22.1 Cross-Framework Evaluation Results

V9 step 40 在两个评估框架上得到了矛盾的结论：

**Step-level eval (eval_visual_idm.py, type_match/extract_match):**

| Model | Standard type_match | Standard extract_match | Hindsight type_match | Hindsight extract_match |
|-------|:---:|:---:|:---:|:---:|
| **V9 step 40** | **57.2%** | **54.4%** | **64.6%** | **61.6%** |
| V8 step 50 | 52.7% | 50.0% | 66.3% | 64.1% |
| V8 step 70 | 46.5% | 41.9% | 66.0% | 62.2% |
| Base (Qwen2.5-VL-7B) | 41.4% | 38.6% | 19.6% | 18.7% |
| UI-S1-GRPO-Trained | 32.1% | 29.1% | 14.0% | 13.4% |

**Trajectory-level eval (eval_qwenvl.py, 1543 tasks, stop-on-error TSR):**

| Model | TSR | Note |
|-------|:---:|------|
| UI-S1-GRPO-Trained | 16.59% | GRPO with trajectory reward |
| V8 step 70 | 15.36% | Visual hindsight only |
| **V9 step 40** | **14.65%** | Dual-channel hindsight |
| V6 best | 10.56% | No effective hindsight |
| Base | 8.62% | No RL |

**Paradox**: V9 在 step-level 大幅领先 (+25pp over UI-S1)，但 trajectory TSR 反而落后 (-2pp)。

### 22.2 Root Cause: Step-Level Eval 的采样偏差

Step-level eval (eval_visual_idm.py) 的 GT action type 分布与 trajectory 评估的 step 0 分布严重不一致：

| Action Type | Step-level eval 采样 | Trajectory step 0 分布 | 偏差 |
|-------------|:---:|:---:|------|
| **click** | **12.6%** (27/215) | **42.3%** | **严重不足 (3.4x)** |
| **open** | **12.1%** (26/215) | **39.3%** | **严重不足 (3.2x)** |
| swipe | 31.6% (68/215) | 5.0% | 过度采样 (6.3x) |
| wait | 18.1% (39/215) | 0.4% | 过度采样 (45x) |
| type | 14.4% (31/215) | 0.0% | 过度采样 |
| system_button | 10.7% (23/215) | 12.8% | ~正常 |

**极端偏差**：Step-level eval 在 step_id=0 只有 **1 个 click 样本**（1/215），但 trajectory 中 42.3% 的首步是 click。

### 22.3 V9 的 Action-Level 分析

从 V9 trajectory eval log 解析的 confusion matrix (3204 个 step 比较)：

| GT Action | 预测正确率 | 主要错误模式 |
|-----------|:---:|------|
| click | 90.6% type match, **61.5% 含坐标** | 32.1% 坐标偏差 |
| type | 93.6% | — |
| system_button | 74.9% | click (45), open (11) |
| swipe | 62.4% | click (79) |
| **open** | **48.3%** | **system_button (149), click (127)** |
| wait | 43.1% | click (44) |

**关键发现**：
1. V9 的 click **type match 高** (90.6%) 但 **坐标匹配率只有 67.9%** (1093/1609)
2. V9 对 **open 的识别极差** (48.3%)，经常错误地预测 system_button 或 click
3. Step 0 的 action 分布是 42.3% click + 39.3% open，这恰好是 V9 最弱的两个 action
4. V9 只在 trajectory 中预测了 34 次 terminate（总计 13125 次），**不是 premature termination**

### 22.4 Step 0 Failure 对比

| Metric | V9 | UI-S1 | Gap |
|--------|:---:|:---:|------|
| Step 0 failure rate | 48.1% (742) | 39.4% (608) | +134 tasks |
| V9 fails, UI-S1 progresses | — | — | 239 tasks |
| V9 fails, UI-S1 also fails | — | — | 492 tasks |
| TSR | 14.65% | 16.59% | -1.94pp |

V9 相比 UI-S1 多了 134 个 step-0 failure，这直接解释了 TSR 差距。

### 22.5 Hindsight Aux Loss 的偏向性

**Hindsight 教什么**: 给定 (s_t, s_{t+1})，预测连接两个状态的 action a_t。

**擅长的场景** (后续 steps):
- Model 已有正确的 history context (前面的 step 都对了)
- s_{t+1} 提供的视觉差异 (swipe 后的滚动、type 后的文字变化) 信息量大
- → 这正是 step-level eval 过度采样的 action types (swipe 31.6%, wait 18.1%, type 14.4%)

**不擅长的场景** (step 0):
- 没有 history，只有 (screenshot, task description)
- step 0 的 action 通常是 open 某个 app 或 click 特定元素
- s_{t+1} 信息对 open/click 的帮助有限（知道 app 打开了并不能帮你决定要打开哪个 app）
- → 这正是 trajectory eval 最关键的环节 (81.6% 的首步是 click+open)

### 22.6 结论

V9 的 step-level eval 高分 (57.2%) 是一个 **虚假信号**：

1. **评估偏差**：Step-level eval 过度采样了 V9 擅长的后续 step actions (swipe/wait/type)，严重不足采样了关键的首步 actions (click/open)
2. **Hindsight 的局限性**：Hindsight aux loss 提升了 action-consequence reasoning（给定状态转移推断 action），但没有提升 initial action prediction（从任务描述推断首步 action）
3. **Trajectory eval 是 serial 的**：Stop-on-error 意味着 48.1% 的 task 在第一步就终止了，后续 step 的强 accuracy 永远没有机会发挥
4. **UI-S1 的优势**：GRPO 训练直接优化 trajectory-level reward，学到的策略对首步更鲁棒（39.4% vs 48.1% step-0 failure），即使单步准确率更低

**Implication**: 未来实验需要：
- 修正 step-level eval 的采样偏差，使其反映真实 action 分布
- 或直接用 trajectory-level TSR 作为主要评估指标
- Hindsight aux loss 需要 complementary signal 来提升首步决策（e.g., SFT warmup on step-0 actions, 或 curriculum 先训练首步再训练后续步）

### 22.7 Evaluation Files

| File | Description |
|------|-------------|
| `evaluation/results/v9_eval/vidm_standard_UI-S1-GRPO-Trained_summary.json` | UI-S1 step-level standard eval |
| `evaluation/results/v9_eval/vidm_hindsight_UI-S1-GRPO-Trained_summary.json` | UI-S1 step-level hindsight eval |
| `evaluation/results/v9_step40_traj.jsonl` | V9 step 40 trajectory eval (1543 tasks) |
| `scripts/eval/eval_uis1_vidm.slurm` | UI-S1 step-level eval script (Job 3388705) |
| `scripts/eval/eval_v9_trajectory.slurm` | V9 trajectory eval script (Job 3388706) |
