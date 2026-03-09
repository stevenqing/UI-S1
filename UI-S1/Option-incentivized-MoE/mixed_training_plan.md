# Grounding 继续训练 Plan: 基于 AP Checkpoint

> Date: 2026-03-08
> 从已有 Action Prediction SFT v2 checkpoint 继续训练 Grounding

---

## 1. 现状分析

### 1.1 已有资源

| 资源 | 状态 | 路径 |
|------|------|------|
| **AP SFT v2 checkpoint** | 已完成 (2ep) | `train_GUI_360/llamafactory/output/gui360_full_sft_v2/` |
| **Grounding 训练数据** | 79,487 samples，已 processed | `datasets/GUI-360/processed_data/grounding_resize/training_data.json` |
| **Grounding 图片** | 无需额外解压 | 全部 79,442 张 ⊂ AP 图片目录 |
| **LlamaFactory 训练流水线** | 就绪 | `train_GUI_360/llamafactory/` |
| **评估流水线** | 就绪 | `train_GUI_360/GUI-360-eval/` |

### 1.2 当前性能 (SFT v2 2ep，仅 AP 训练)

| Task | SFT v2 2ep | Paper SFT | Gap |
|------|:----------:|:---------:|:---:|
| **Grounding** | 70.77% | 82.30% | **-11.5pp** |
| **Action Prediction (Visual)** | 49.37% | 50.08% | **-0.7pp** |

### 1.3 关键发现 (from eval_analysis_sft_v2.md)

1. **AP 已追平 paper** — 差距仅 0.7pp
2. **Grounding gap 根因：0 条 grounding 训练数据** — 全靠 in-context learning 适配 `<coordinate>` 格式
3. **坐标精度是 grounding 首要瓶颈** — ±75px 容差下即超过 paper，模型元素理解能力已足够
4. **两个 task prompt 格式完全不同** — AP 用 `<tool_call>`，Grounding 用 `<coordinate>`，不会混淆

### 1.4 为什么不需要混合训练

| 考虑 | 分析 |
|------|------|
| **AP 遗忘风险？** | 低 — AP 知识已 baked in 2 epochs 的权重中，低 LR + 1ep grounding 扰动小 |
| **格式冲突？** | 无 — prompt 明确指定输出格式，模型通过 instruction 区分 task |
| **AP 数据冗余？** | 是 — 已训 2 epochs，再加入等于第 3 epoch，边际收益低 |
| **训练效率** | Grounding-only: 79K samples vs 混合 181K，节省 ~56% 训练时间 |

**策略：先用 grounding-only 训练，评估后若 AP 回退再 fallback 到混合训练。**

---

## 2. 方案设计

### 2.1 推荐方案：从 AP SFT v2 checkpoint 继续训练 Grounding-Only

**核心思路：** AP checkpoint 作为 base，仅用 79K grounding 数据继续训练 1 epoch。

### 2.2 训练配置

```yaml
### model
model_name_or_path: train_GUI_360/llamafactory/output/gui360_full_sft_v2  # 从 AP checkpoint 继续
image_max_pixels: 1003520
trust_remote_code: true

### method
stage: sft
do_train: true
finetuning_type: full
freeze_vision_tower: true
freeze_multi_modal_projector: false      # projector 可训练 — grounding 需要更精确的坐标映射
deepspeed: train_GUI_360/llamafactory/ds_z3_config.json

### dataset
dataset: gui360_grounding_train          # 仅 grounding 数据
dataset_dir: train_GUI_360/llamafactory/data
template: qwen2_vl
cutoff_len: 8192

### train
per_device_train_batch_size: 1
gradient_accumulation_steps: 16
learning_rate: 5.0e-6                    # 低 LR，保护 AP 已有知识
num_train_epochs: 1.0
lr_scheduler_type: cosine
warmup_ratio: 0.05
weight_decay: 0.1
bf16: true

### output
output_dir: train_GUI_360/llamafactory/output/gui360_full_sft_v3_grounding
```

**关键参数选择理由：**

| 参数 | 选择 | 原因 |
|------|------|------|
| `model_name_or_path` | SFT v2 checkpoint | AP 能力已在权重中，无需重训 |
| `learning_rate` | 5e-6 (原始 1e-5 的一半) | 继续训练降低 LR，减少对 AP 权重的扰动 |
| `num_train_epochs` | 1.0 | 79K 数据 1ep 即可充分学习 grounding 格式 |
| `freeze_vision_tower` | true | 与 SFT v2 一致 |
| `freeze_multi_modal_projector` | false | Projector 可训练，改善坐标精度 |

---

## 3. 实施步骤

### Step 1: 数据注册 (~5 分钟)

数据已在 `datasets/GUI-360/processed_data/grounding_resize/` 就绪。只需做两件事：

**a) 对 grounding 数据跑一次 `prepare_data.py`（图片路径转绝对路径）：**

```bash
python train_GUI_360/llamafactory/prepare_data.py \
    --input datasets/GUI-360/processed_data/grounding_resize/training_data.json \
    --output train_GUI_360/llamafactory/data/gui360_grounding_train.json \
    --image-base-dir datasets/GUI-360/processed_data/action_prediction_train_resize \
    --val-output train_GUI_360/llamafactory/data/gui360_grounding_val.json \
    --val-size 100
```

注意 `--image-base-dir` 指向 **AP 的图片目录**（grounding 图片 ⊂ AP 图片，无需解压 `images.tar.gz`）。

**b) 在 `dataset_info.json` 中注册 grounding 数据集：**

```json
{
  "gui360_grounding_train": {
    "file_name": "gui360_grounding_train.json",
    "formatting": "sharegpt",
    "columns": { "messages": "conversations", "images": "images" },
    "tags": { "role_tag": "from", "content_tag": "value", "user_tag": "human", "assistant_tag": "gpt" }
  }
}
```

### Step 2: 创建训练配置 + SLURM 脚本

新建 `qwen25vl_gui360_grounding_sft.yaml`，关键改动：
- `model_name_or_path` → SFT v2 checkpoint
- `dataset: gui360_grounding_train`（仅 grounding）
- `learning_rate: 5e-6`

基于现有 `train_gui360_llamafactory.slurm` 修改 config 路径即可。

### Step 3: 提交训练

4 nodes × 4 GPUs，预计 1 epoch ~4-6 小时 (79K samples)。

### Step 4: 评估

**必须同时评估两个 task**（验证 grounding 提升 + AP 无回退）：
```bash
# Grounding 评估 (主要目标)
python eval_gui360.py --task grounding --model_path output/gui360_full_sft_v3_grounding/

# AP 评估 (验证无回退)
python eval_gui360.py --task action_prediction --model_path output/gui360_full_sft_v3_grounding/
```

**Fallback：** 如果 AP 回退 >2pp，则 fallback 到混合训练（`dataset: gui360_train,gui360_grounding_train`）。

---

## 4. 预期结果

| Task | SFT v2 (AP-only) | SFT v3 Grounding (预期) | Paper |
|------|:-----------------:|:----------------------:|:-----:|
| **Grounding** | 70.77% | **76-80%** | 82.30% |
| **Action Prediction** | 49.37% | **48-50%** (保持) | 50.08% |

**乐观情况：** Grounding 训练改善了模型整体坐标精度，AP 中的坐标错误也同步改善（36.1% AP 失败来自坐标偏出 bbox），两个 task 同时提升。

**保守情况：** AP 小幅回退 1-2pp，但 grounding 提升 5-10pp，净收益仍然显著。

---

## 5. Fallback 方案

如果 grounding-only 训练导致 AP 回退 >2pp：

### 5.1 混合训练 Fallback

```yaml
dataset: gui360_train,gui360_grounding_train   # LlamaFactory 逗号分隔自动混合
# AP 101K + Grounding 79K = 181K, 自然比例 56:44
```

### 5.2 LoRA Grounding (零 AP 风险)

在 SFT v2 frozen base 上加 LoRA，仅训练 grounding-specific 参数：

```yaml
finetuning_type: lora
model_name_or_path: train_GUI_360/llamafactory/output/gui360_full_sft_v2
lora_rank: 32
lora_alpha: 64
dataset: gui360_grounding_train
```

AP 能力完全保留（base frozen），但 LoRA v3 评估已证明 LoRA 性能弱于 full SFT。

---

## 6. 工作量预估

| 步骤 | 工作内容 | 预计时间 |
|------|---------|---------|
| 数据注册 | 跑 prepare_data.py + 更新 dataset_info.json | ~5 分钟 |
| 训练配置 | 新建 yaml + slurm | ~15 分钟 |
| 正式训练 | 79K samples, 1ep, 4 nodes | ~4-6 小时 GPU |
| 评估 | Grounding + AP 双评估 | ~2-4 小时 GPU |
| **总计** | | **~20 分钟编码 + ~6-10 小时 GPU** |

---

## 7. 总结

**核心策略：** 从 SFT v2 AP checkpoint 出发，仅用 79K grounding 数据以低 LR (5e-6) 继续训练 1 epoch。不混合 AP 数据 — AP 知识已在权重中，无需重复训练。

**预期收益：**
- Grounding: +5~10pp (70.8% → 76-80%)
- Action Prediction: 保持 (49.4% ± 1pp)

**最大杠杆点：** 当前 grounding gap 根因是 "0 条 grounding 训练数据" → 加入 79K 条应有立竿见影效果。
