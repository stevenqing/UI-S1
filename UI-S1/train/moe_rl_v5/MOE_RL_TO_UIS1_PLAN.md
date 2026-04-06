# Plan: 将 GUI-360 MoE RL 训练方法适配到 UI-S1

## 1. 背景

### 为什么不能直接复用 GUI-360 的 MoE 初始化
- GUI-360 是**桌面应用**任务 (Excel/Word/PPT)，UI-S1 是 **Android 手机**操作任务
- Task domain 完全不同，GUI-360 的 SVD LoRA experts 不适用
- UI-S1 的 SFT 数据只有 999 samples，SFT 几乎无提升 (TSR 9.1% → 9.3%)，从中做 SVD 提取没有意义

### 方案：Random Init MoE LoRA + 直接 RL
- 不做 SFT → SVD → MoE 初始化
- 直接用**随机初始化**的 MoE LoRA experts，从 base Qwen2.5-VL-7B-Instruct 开始 RL 训练
- 保留 GUI-360 MoE RL 中验证有效的**训练方法**（routing策略、loss设计、超参）

### 两个 pipeline 对比

| 维度 | GUI-360 MoE RL v5 | UI-S1 MoE RL (本方案) |
|------|-------------------|----------------------|
| **数据集** | 13,750 trajectories | 1,000 trajectories |
| **Expert初始化** | SVD r256 warm-start | **Random init** |
| **Expert LoRA rank** | 256 | **32** (random init不需要高rank) |
| **Target modules** | 全7个 | **全7个** (从GUI-360继承) |
| **Top-K** | 2 (sparse) | **2** (从GUI-360继承) |
| **max_response_length** | 128 | **512** (UI-S1 response更长) |
| **节点** | 8 nodes (32 GPUs) | **4 nodes** (16 GPUs，数据量少) |
| **Reward** | gui360 soft coord | **naive** (action match) |
| **Epochs** | 10 | **5** |

---

## 2. 配置设计

### 2.1 MoE 架构选择

```yaml
# MoE 配置 — 随机初始化，中等 rank
moe:
  enabled: true
  num_experts: 4
  top_k: 2                    # sparse routing (GUI-360验证有效)
  expert_lora_r: 32            # random init 用中等 rank (非SVD，不需要256)
  expert_lora_alpha: 64        # 2x rank
  expert_lora_dropout: 0.05
  target_modules: [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]  # 全7个
  router_hidden: 256
  router_temperature: 0.5
  balance_weight: 0.2          # 防止 expert collapse
  balance_type: mse
  z_loss_weight: 0.01          # router 稳定性
  pooling_strategy: mean
  use_vectorized_routing: true
  # 无 moe_checkpoint — 随机初始化
```

**为什么 r=32 而非 r=256**：
- r=256 是为了无损还原 Full SFT 的能力 (SVD提取需要)
- Random init 从零学起，r=32 已足够表达 LoRA 的低秩更新
- r=32 节省显存，不需要 optimizer_offload，训练更快

### 2.2 Actor 训练配置

```yaml
actor:
  optim:
    lr: 1e-5                   # 与 GUI-360 一致
    weight_decay: 0.01
  grad_clip: 0.5
  clip_ratio: 0.1
  ppo_micro_batch_size_per_gpu: 2   # r=32 显存够，可以用2
  use_kl_loss: true
  kl_loss_coef: 0.1            # random init 离 base 近，用较高 KL (与现有UI-S1 MoE一致)
  kl_loss_type: low_var_kl
  entropy_coeff: 0.01
  fsdp_config:
    param_offload: false
    optimizer_offload: false    # r=32 不需要 offload
```

### 2.3 Rollout 配置

```yaml
rollout:
  gpu_memory_utilization: 0.7   # r=32 显存压力小
  max_model_len: 16384          # UI-S1 prompt 可能更长
  enforce_eager: true
  free_cache_engine: true
  n: 4                          # 4 rollouts per prompt
  limit_images: 2
```

### 2.4 资源配置

```
nodes: 4            # 数据量小，4 nodes 足够
GPUs: 16
batch_size: 4       # 1 per node × 4 nodes
```

- 每 epoch: 1000 / 4 = **250 steps** (每个trajectory展开多个step)
- 实际看verl怎么计算... 参考现有UI-S1 MoE (8 nodes, batch=8): ~31 steps/epoch
- 4 nodes, batch=4: ~62 steps/epoch
- 5 epochs × 62 = **~310 total steps**
- 预估: 310 × 5min/step ≈ **26 小时** (需要 resume 或提高时间限制)

---

## 3. 需要创建的文件

| 文件 | 路径 | 说明 |
|------|------|------|
| YAML配置 | `train/moe_rl_v5/traj_grpo_moe_uis1.yaml` | MoE RL 配置 |
| SLURM脚本 | `train/moe_rl_v5/train_moe_rl_uis1.slurm` | 4 nodes 训练脚本 |

### 不需要创建
- 无需 `convert_sft_lora_to_moe.py` — 不做 SVD 初始化
- 无需 `moe_checkpoint` — trainer 会自动 random init

---

## 4. 完整参数表

```yaml
# === UI-S1 MoE RL 完整配置 ===

# Data
data.train_files: datasets/ui_s1_dataset/ui_s1_train.jsonl       # 1,000 trajectories
data.val_files: evaluation/dataset/android_control_evaluation_std.jsonl
data.train_batch_size: 4
data.val_batch_size: 8
data.max_prompt_length: 8192
data.max_response_length: 512
data.truncation: error

# MoE
model.moe.enabled: true
model.moe.num_experts: 4
model.moe.top_k: 2
model.moe.expert_lora_r: 32
model.moe.expert_lora_alpha: 64
model.moe.expert_lora_dropout: 0.05
model.moe.target_modules: [q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj]
model.moe.router_hidden: 256
model.moe.router_temperature: 0.5
model.moe.balance_weight: 0.2
model.moe.balance_type: mse
model.moe.z_loss_weight: 0.01
model.moe.pooling_strategy: mean
model.moe.use_vectorized_routing: true
# 无 moe_checkpoint

# Actor
actor.optim.lr: 1e-5
actor.grad_clip: 0.5
actor.clip_ratio: 0.1
actor.kl_loss_coef: 0.1
actor.kl_loss_type: low_var_kl
actor.entropy_coeff: 0.01
actor.ppo_micro_batch_size_per_gpu: 2
actor.fsdp_config.optimizer_offload: false
actor.fsdp_config.param_offload: false

# Rollout (vLLM)
rollout.gpu_memory_utilization: 0.7
rollout.max_model_len: 16384
rollout.enforce_eager: true
rollout.free_cache_engine: true
rollout.n: 4
rollout.limit_images: 2

# Algorithm
algorithm.adv_estimator: uis1
algorithm.gamma: 0.5
algorithm.uis1.step_advantage_w: 1.0
algorithm.uis1.mode: mean_std_norm
algorithm.patch_threshold: 2
algorithm.filter_groups.enable: false

# Trainer
trainer.total_epochs: 5
trainer.save_freq: 10
trainer.test_freq: 10
trainer.max_ckpt_to_keep: 5
trainer.nnodes: 4
trainer.n_gpus_per_node: 4
trainer.project_name: ui_s1_moe_rl
trainer.experiment_name: moe_4exp_r32_topk2_randinit

# Reward
reward_model.enable: false
reward_model.reward_manager: naive
```

---

## 5. 与现有 UI-S1 MoE 的区别

现有 `train_ui_s1_moe.slurm` 存在的问题和本方案的改进:

| 项目 | 现有 UI-S1 MoE | 本方案 |
|------|---------------|--------|
| **Target modules** | 2个 (q,v) | **7个** (全部) — 更强表达力 |
| **Top-K** | 4 (soft, all experts) | **2** (sparse) — GUI-360验证更有效 |
| **Rank** | 16 | **32** — 更大容量 |
| **节点** | 8 nodes (32 GPUs) | **4 nodes** (16 GPUs) — 更节省 |
| **z_loss** | 0.01 | 0.01 (保持) |
| **balance_weight** | 0.2 | 0.2 (保持) |
| **max_ckpt_to_keep** | 无 | **5** — 防止磁盘爆满 |
| **entropy_coeff** | 0.0 | **0.01** — 鼓励探索 |

---

## 6. 预期时间线

| 步骤 | 时间 |
|------|------|
| 创建 YAML + SLURM | ~10 min |
| 提交训练 | 即时 |
| 训练完成 (~310 steps) | ~26 小时 |
| 评估 (AndroidControl SOP) | 另提交 eval job |

---

## 7. f_pseudo Reward Shaping for UI-S1

### 7.1 为什么需要专门构建 UI-S1 的 f_pseudo

- 现有的 `f_pseudo_map.json` 只覆盖 GUI-360 (Excel/Word/PPT) 的状态
- UI-S1 是 Android 手机操作，状态空间完全不同
- 需要重新走完整的 f_pseudo 生成 pipeline

### 7.2 GUI-360 vs UI-S1 数据格式差异

| 维度 | GUI-360 | UI-S1 (Android) |
|------|---------|-----------------|
| **数据格式** | 1个JSONL文件 = 1条轨迹，每行1个step | 1个JSONL文件包含所有轨迹，每行1个episode（含steps数组） |
| **状态信息** | UIA Accessibility Tree + Ribbon Tabs + Dialog | 仅截图 + 动作坐标/bbox |
| **控件类型** | 17种UIA控件 (Button, TabItem, MenuItem...) | 无 |
| **Tab/Ribbon** | 19种Office标签 (Home, Insert, Data...) | 无 |
| **对话框** | 从ui_tree提取Window子节点 | 无 |
| **App域** | 3个固定域 (excel/word/ppt) | 数百个不同Android app |
| **Action类型** | click, type, drag, select_text, wheel... | click, swipe, open, system_button, wait, terminate |

### 7.3 方案选择：VLM-based Eigenfunction

由于 UI-S1 **没有 accessibility tree**，无法构造 43 维手工嵌入。采用 GUI-360 已验证的 **VLM 路径**：

- 使用 `VLMEigenfunctionModel` (Qwen2.5-VL + LoRA + 回归头)
- 直接从**截图**学习状态表示
- 输入模式: `screenshot` (无 a11y text)
- **不做 per-app 分割** — UI-S1 覆盖数百个 app，每个 app 轨迹太少

### 7.4 Pipeline 四阶段

#### Stage 1: 收集转移对 — `scripts/collect_uis1_transitions.py` (新建)

**输入**: `datasets/ui_s1_dataset/ui_s1_train.jsonl` (1000 episodes)

**处理逻辑**:
```python
for line_idx, line in enumerate(jsonl):
    episode = json.loads(line)
    execution_id = str(line_idx)  # 用行号作为 execution_id
    steps = episode["steps"]

    for i in range(len(steps) - 1):
        state_hash_t = md5(steps[i]["screenshot"])     # 截图路径作为唯一状态标识
        state_hash_t1 = md5(steps[i+1]["screenshot"])

        transition_pairs.append({
            "src_hash": state_hash_t,
            "dst_hash": state_hash_t1,
            "execution_id": execution_id,
            "step_id": i
        })

        state_manifest[state_hash_t] = {
            "screenshot_path": steps[i]["screenshot"],
            "goal": episode["goal"],
            "step_id": i,
            "action_type": steps[i]["action_content"]["action"]
        }
```

**输出**:
```
outputs/transitions/uis1/
├── transition_pairs.json      # [{src_hash, dst_hash}, ...]  (约5,536条)
├── state_manifest.json        # {hash: {screenshot_path, goal, step_id, action_type}}
└── statistics.json            # 状态数、转移数等统计
```

**估计规模**:
- 1,000 episodes × ~5.5 steps/episode = ~5,500 transitions
- ~5,500 unique screenshots (每个截图是唯一状态)
- 远少于 GUI-360 (91,618 transitions, 6,505 unique states)

#### Stage 2: 训练 VLM f_net — 复用 `scripts/train_vlm_eigenfunction.py`

**关键修改**:
- 去除 per-app 分割逻辑 (UI-S1 训练单个统一模型)
- 输入模式改为 `screenshot` only (无 a11y text)
- 调整 batch_size (数据量小，可以用更大 batch)

**训练配置**:
```yaml
model_path: checkpoints/Qwen2.5-VL-7B-Instruct
lora_rank: 16
lora_alpha: 32
input_mode: screenshot        # 仅截图，无 a11y
batch_size: 8                 # 数据量小，可以用更大 batch
num_epochs: 5                 # 数据量少，多训几个 epoch
lr: 1e-4
eta: 1.0                      # 排斥项权重
percentile_k: 30.0            # 底部30%为瓶颈
```

**资源**: 1 GPU (GH200), 预计 ~3-5 小时
- 5,500 transitions / 8 batch = 687 steps/epoch × 5 epochs = ~3,400 steps
- 每 step 需要 3 张图片 (src + dst + rand) 的 VLM forward pass

**输出**:
```
outputs/vlm_fnet/uis1/
├── lora_adapter/              # LoRA weights
├── regression_head.pt         # 回归头
├── f_values.npz               # {hashes: [], f_values: []}
├── bottlenecks.json           # 瓶颈状态列表
├── results.json               # 训练结果
└── checkpoint_epoch{1..5}/    # 中间 checkpoints
```

#### Stage 3: 计算 f_pseudo — `scripts/precompute_f_pseudo_uis1.py` (新建)

**处理逻辑**:
```python
# 加载训练好的 VLM f_net
vlm_model = VLMEigenfunctionModel.load_adapter(vlm_fnet_dir, config)

# 方案A: 直接用预计算的 f_values.npz (快)
f_value_map = load_from_npz("outputs/vlm_fnet/uis1/f_values.npz")

# 方案B: 对未知状态做 VLM 推理 (慢但完整)
# 遍历所有 episode，对每对 (step_t, step_{t+1}):
f_pseudo_map[execution_id][step_id] = f(screenshot_t) - f(screenshot_{t+1})
```

**输出**:
```
outputs/f_pseudo/uis1/
├── f_pseudo_map.json          # {execution_id: {step_id: f_pseudo_value}}
└── statistics.json            # 分布统计
```

#### Stage 4: 集成到 RL 训练

在 MoE RL SLURM 脚本中添加:
```bash
reward_model.reward_manager=f_pseudo_dapo
reward_model.reward_kwargs.f_pseudo_path=$PROJECT_DIR/outputs/f_pseudo/uis1/f_pseudo_map.json
reward_model.reward_kwargs.f_pseudo_lambda=0.1
```

### 7.5 需要创建/修改的文件

| 文件 | 动作 | 说明 |
|------|------|------|
| `scripts/collect_uis1_transitions.py` | **新建** | 解析 UI-S1 JSONL → 转移对 + 状态清单 |
| `scripts/train_vlm_eigenfunction.py` | **修改** | 添加 `--no-per-app` 模式，支持 UI-S1 |
| `scripts/precompute_f_pseudo_uis1.py` | **新建** | 用 VLM f_net 计算 UI-S1 的 f_pseudo |
| `train/moe_rl_v5/slurm_uis1_fnet.slurm` | **新建** | VLM f_net 训练 SLURM 脚本 |
| `train/moe_rl_v5/train_moe_rl_uis1_fpseudo.slurm` | **新建** | MoE RL + f_pseudo SLURM 脚本 |

### 7.6 UI-S1 f_pseudo 的预期效果与风险

**预期收益**:
- f_pseudo 为跨越 UI 状态瓶颈（如打开新 app、进入设置页面、弹出对话框等）的动作提供额外正向 reward
- 在 GUI-360 中，f_pseudo 帮助 agent 学习难以到达的状态转换

**风险/不确定性**:
- UI-S1 只有 ~5,500 transitions，远少于 GUI-360 (91K) — 图结构可能太稀疏
- 每个截图几乎唯一 (不像 GUI-360 有重复状态) — Laplacian 特征函数可能退化
- Android app 数量多但每个 app 轨迹少 — 图可能高度不连通
- **如果状态图不连通或太稀疏**：eigenfunction 训练会失败或退化为常数解

**缓解策略**:
- 先运行 Stage 1 查看图连通性统计（连通分量数、平均度等）
- 如果图太稀疏，考虑用**更粗粒度的状态定义**（如只用 app 名 + 屏幕区域，合并相似截图）
- 如果 VLM 训练 smoothness 不下降，说明状态表示不够好 → 考虑增加 epochs 或调 lr

---

## 8. 实施计划与时间线

### Phase 1: MoE RL (无 f_pseudo) — 可立即开始

| 步骤 | 时间 | 说明 |
|------|------|------|
| 创建 YAML + SLURM | ~10 min | `traj_grpo_moe_uis1.yaml` + `train_moe_rl_uis1.slurm` |
| 提交训练 | 即时 | Random init MoE LoRA, 4 nodes |
| 训练完成 | ~26 小时 | ~310 steps × 5min/step |
| 评估 | ~1 小时 | AndroidControl SOP eval |

### Phase 2: f_pseudo 生成 — 与 Phase 1 并行

| 步骤 | 时间 | 前置依赖 |
|------|------|---------|
| 创建 `collect_uis1_transitions.py` | ~20 min | 无 |
| 运行 transition collection | ~5 min | CPU only |
| 检查图连通性 | ~5 min | 决定是否继续 |
| 修改 `train_vlm_eigenfunction.py` | ~15 min | 无 |
| 训练 VLM f_net | ~3-5 小时 | 1 GPU |
| 创建 `precompute_f_pseudo_uis1.py` | ~15 min | 无 |
| 计算 f_pseudo_map | ~30 min | VLM f_net 完成 |

### Phase 3: MoE RL + f_pseudo — Phase 1&2 完成后

| 步骤 | 时间 | 说明 |
|------|------|------|
| 创建 f_pseudo SLURM | ~10 min | 在 Phase 1 基础上加 f_pseudo 参数 |
| 提交训练 | 即时 | 对比有无 f_pseudo 的效果 |
| 训练完成 | ~26 小时 | |

---

## 9. 后续可选实验

1. **对比 LoRA baseline**: 同样 4 nodes，LoRA r=64，作为 non-MoE baseline
2. **增大 rank**: 如果 r=32 效果好，可以尝试 r=64 或 r=128
3. **粗粒度状态图**: 如果截图级状态太稀疏，用 app+screen_type 聚合
4. **MLP f_net**: 如果未来 UI-S1 有 accessibility tree 数据，可构造手工嵌入训练 MLP（更快更稳定）
