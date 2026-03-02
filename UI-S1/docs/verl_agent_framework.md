# VERL Agent 框架详细拆解

> 本文档详细拆解 UI-S1 项目使用的 VERL (Volcano Engine RL) 框架架构。

## 目录

- [一、框架整体架构](#一框架整体架构)
- [二、核心数据协议 DataProto](#二核心数据协议-dataproto)
- [三、分布式训练机制 (Ray + FSDP)](#三分布式训练机制-ray--fsdp)
- [四、训练流程 (DAPO Trainer)](#四训练流程-dapo-trainer)
- [五、多轮交互生成器 (MultiRoundGenerator)](#五多轮交互生成器-multiroundgenerator)
- [六、奖励计算系统](#六奖励计算系统)
- [七、目录结构速查](#七目录结构速查)
- [八、核心流程图总结](#八核心流程图总结)
- [九、关键概念对照表](#九关键概念对照表)

---

## 一、框架整体架构

```
┌─────────────────────────────────────────────────────────────────────┐
│                         main_dapo.py (入口)                          │
│                      Hydra 配置管理 + Ray 初始化                      │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      RayTrajDAPOTrainer (训练器)                      │
│           继承 RayPPOTrainer，实现 GUI 轨迹强化学习                    │
└─────────────────────────────────────────────────────────────────────┘
                                    │
            ┌───────────────────────┼───────────────────────┐
            ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Actor Worker   │    │  Critic Worker  │    │  Reward Manager │
│   (策略网络)     │    │   (价值网络)     │    │   (奖励函数)     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
            │                       │                       │
            └───────────────────────┼───────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    DataProto (统一数据协议)                          │
│              TensorDict (Tensor) + numpy (Non-Tensor)               │
└─────────────────────────────────────────────────────────────────────┘
```

### 1.1 核心组件说明

| 组件 | 文件位置 | 功能 |
|------|----------|------|
| main_dapo.py | `verl/trainer/main_dapo.py` | 训练入口，初始化 Ray 和配置 |
| RayTrajDAPOTrainer | `verl/trainer/ppo/dapo_ray_trainer.py` | GUI 轨迹 DAPO 训练器 |
| ActorRolloutRefWorker | `verl/workers/fsdp_workers.py` | Actor/Rollout/Ref 三合一 Worker |
| CriticWorker | `verl/workers/fsdp_workers.py` | 价值网络 Worker |
| RewardManager | `verl/workers/reward_manager/` | 奖励计算管理器 |

---

## 二、核心数据协议 DataProto

**文件位置**: `verl/protocol.py`

DataProto 是整个框架的数据交换核心，它统一了各模块之间的数据传输格式。

### 2.1 数据结构定义

```python
@dataclass
class DataProto:
    batch: TensorDict = None           # Tensor 数据 (input_ids, attention_mask, rewards 等)
    non_tensor_batch: Dict = {}        # 非 Tensor 数据 (字符串、图像、元数据等)
    meta_info: Dict = {}               # 元信息 (eos_token_id, pad_token_id 等)
```

### 2.2 关键特性

| 特性 | 方法 | 说明 |
|------|------|------|
| TensorDict | `batch` | 基于 PyTorch TensorDict，支持像操作单个 Tensor 一样操作字典 |
| 数据分割 | `chunk(n)` | 将数据分割给多个 GPU worker |
| 数据合并 | `concat(list)` | 合并多个 worker 返回的数据 |
| 数据重复 | `repeat(n)` | 用于 rollout 时重复 prompt (n 次采样) |
| 序列化 | `__getstate__`/`__setstate__` | 支持 Ray 分布式传输 |
| 切片 | `slice(start, end)` | 支持数据切片操作 |
| 索引选择 | `select_idxs(idxs)` | 按索引选择数据 |

### 2.3 数据流示例

```
Dataset → DataProto → chunk(n_gpus) → [Worker1, Worker2, ...] → concat() → DataProto
```

### 2.4 典型用法

```python
# 创建 DataProto
data = DataProto.from_single_dict({
    "input_ids": torch.tensor(...),
    "attention_mask": torch.tensor(...),
    "raw_prompt": np.array([...], dtype=object)
})

# 分割给多个 worker
chunks = data.chunk(n_gpus)

# 合并结果
result = DataProto.concat(outputs)

# 重复用于多次采样
repeated = data.repeat(rollout_n, interleave=True)
```

---

## 三、分布式训练机制 (Ray + FSDP)

### 3.1 Ray 分布式控制层

**文件位置**: `verl/single_controller/ray/base.py`

#### 架构图

```
┌─────────────────────────────────────────────────────────────────────┐
│                     Ray Single Controller 模式                       │
│                    一个 Driver + 多个 Worker Actor                   │
└─────────────────────────────────────────────────────────────────────┘

                        ┌──────────────┐
                        │    Driver    │  (main_dapo.py)
                        │  (单控制器)   │
                        └──────┬───────┘
                               │ RPC 调用
       ┌───────────────────────┼───────────────────────┐
       ▼                       ▼                       ▼
┌─────────────┐        ┌─────────────┐        ┌─────────────┐
│  Worker 0   │        │  Worker 1   │        │  Worker N   │
│   (GPU 0)   │◄──────►│   (GPU 1)   │◄──────►│   (GPU N)   │
└─────────────┘  NCCL  └─────────────┘  NCCL  └─────────────┘
```

#### 核心类

| 类名 | 功能 |
|------|------|
| `RayResourcePool` | 管理 GPU 资源和 PlacementGroup |
| `RayWorkerGroup` | 封装一组 Ray Actor，提供统一调用接口 |
| `RayClassWithInitArgs` | 包装 Worker 类，支持延迟初始化 |

#### 资源池配置示例

```python
# 定义资源池
resource_pool_spec = {
    "global_pool": [n_gpus_per_node] * nnodes,  # 例如 [4, 4] 表示 2 节点各 4 GPU
}

# 角色映射
mapping = {
    Role.ActorRollout: "global_pool",
    Role.Critic: "global_pool",
}
```

### 3.2 FSDP (Fully Sharded Data Parallel)

**文件位置**: `verl/workers/fsdp_workers.py`

#### Worker 初始化流程

```python
class ActorRolloutRefWorker(Worker):
    """可以作为 Actor / Rollout / Ref 三种角色"""

    def __init__(self, config, role):
        # 1. 初始化 PyTorch 分布式进程组
        torch.distributed.init_process_group(backend="nccl", ...)

        # 2. 创建 Device Mesh (支持 FULL_SHARD 或 HYBRID_SHARD)
        self.device_mesh = create_device_mesh(world_size, fsdp_size)

        # 3. 判断角色
        self._is_actor = role in ["actor", "actor_rollout", "actor_rollout_ref"]
        self._is_rollout = role in ["rollout", "actor_rollout", "actor_rollout_ref"]
        self._is_ref = role in ["ref", "actor_rollout_ref"]

        # 4. 配置 offload 策略
        self._is_offload_param = config.actor.fsdp_config.get("param_offload", False)
        self._is_offload_optimizer = config.actor.fsdp_config.get("optimizer_offload", False)
```

#### FSDP 分片策略

| 策略 | 说明 | 适用场景 |
|------|------|----------|
| `FULL_SHARD` | 模型参数完全分片到所有 GPU | 单节点多 GPU |
| `HYBRID_SHARD` | 节点内分片 + 节点间复制 | 多节点训练 |

#### Device Mesh 创建

```python
def create_device_mesh(world_size, fsdp_size):
    if fsdp_size < 0 or fsdp_size >= world_size:
        # 单维度：全分片
        device_mesh = init_device_mesh("cuda", mesh_shape=(world_size,), mesh_dim_names=["fsdp"])
    else:
        # 二维度：混合分片 (DDP + FSDP)
        device_mesh = init_device_mesh("cuda",
            mesh_shape=(world_size // fsdp_size, fsdp_size),
            mesh_dim_names=["ddp", "fsdp"])
    return device_mesh
```

---

## 四、训练流程 (DAPO Trainer)

**文件位置**: `verl/trainer/ppo/dapo_ray_trainer.py`

### 4.1 主训练循环 `fit()`

```python
def fit(self):
    for epoch in range(total_epochs):
        for batch_dict in train_dataloader:
            # ===== Step 1: 多轮 Rollout 生成 =====
            traj_batch = DataProto.from_single_dict(batch_dict)
            mr_gen = MultiRoundGenerator(traj_batch, rollout_n=n)

            for step_batch in mr_gen.fetch_batch():
                # 生成序列
                gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)

                # 计算奖励
                reward_tensor, reward_info = compute_reward(step_batch, self.reward_fn)

                # 应用响应到下一轮
                mr_gen.apply_response(step_batch)
                step_batch_list.append(step_batch)

            # ===== Step 2: 合并所有步骤数据 =====
            batch = DataProto.concat(step_batch_list)

            # ===== Step 3: DAPO 过滤 (核心创新) =====
            # 过滤掉 std=0 的 prompt groups (所有采样结果相同)
            kept_prompt_uids = [uid for uid, std in prompt_uid2metric_std.items()
                               if std > std_threshold]

            # ===== Step 4: 计算 Old Log Prob =====
            old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)

            # ===== Step 5: 计算参考策略 Log Prob (可选) =====
            if use_reference_policy:
                ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)

            # ===== Step 6: 计算 Advantage =====
            batch = compute_advantage(batch, adv_estimator, gamma, lam)

            # ===== Step 7: 更新 Critic (可选) =====
            if use_critic:
                critic_output = self.critic_wg.update_critic(batch)

            # ===== Step 8: 更新 Actor =====
            actor_output = self.actor_rollout_wg.update_actor(batch)

            # ===== Step 9: 验证和保存 =====
            if global_steps % test_freq == 0:
                val_metrics = self._validate()
            if global_steps % save_freq == 0:
                self._save_checkpoint()
```

### 4.2 训练数据流图

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              训练一个 Step                                   │
└─────────────────────────────────────────────────────────────────────────────┘

 ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
 │  DataLoader  │────►│MultiRoundGen │────►│   Rollout    │
 │   (prompts)  │     │  (多轮交互)   │     │  (vLLM/FSDP) │
 └──────────────┘     └──────────────┘     └──────┬───────┘
                                                   │ responses
                                                   ▼
                      ┌──────────────┐     ┌──────────────┐
                      │  Reward Fn   │◄────│  Step Batch  │
                      │  (规则/模型)  │     │  (拼接所有步) │
                      └──────┬───────┘     └──────────────┘
                             │ rewards
                             ▼
         ┌──────────────────────────────────────────────────────┐
         │                    DAPO 过滤                          │
         │  - 按 prompt uid 分组                                 │
         │  - 计算组内 reward std                                │
         │  - 过滤 std < threshold 的组                          │
         └──────────────────────────┬───────────────────────────┘
                                    │
                                    ▼
 ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
 │  Old LogProb │────►│   Advantage  │────►│ Actor Update │
 │   (Actor)    │     │   Estimator  │     │   (PPO/GRPO) │
 └──────────────┘     └──────────────┘     └──────────────┘
```

### 4.3 DAPO 过滤机制

DAPO (Data-Augmented Policy Optimization) 的核心思想是过滤掉没有学习信号的样本：

```python
# 收集每个 prompt 的采样奖励
prompt_uid2metric_vals = defaultdict(list)
for uid, metric_val, step_id in zip(uids, metrics, step_ids):
    if step_id == max_step_id[uid]:  # 只看最后一步
        prompt_uid2metric_vals[uid].append(metric_val)

# 计算标准差
prompt_uid2metric_std = {}
for prompt_uid, metric_vals in prompt_uid2metric_vals.items():
    prompt_uid2metric_std[prompt_uid] = np.std(metric_vals)

# 过滤 std=0 的组 (所有采样结果相同，没有对比学习信号)
kept_prompt_uids = [uid for uid, std in prompt_uid2metric_std.items()
                   if std > std_threshold or len(prompt_uid2metric_vals[uid]) == 1]
```

---

## 五、多轮交互生成器 (MultiRoundGenerator)

**文件位置**: `verl/utils/dataset/universal_multiround.py`

这是 UI-S1 项目的核心创新之一，支持 GUI Agent 的多轮交互训练。

### 5.1 核心类结构

```python
class StdTrajectory:
    """单条轨迹的状态管理"""
    def __init__(self, line, actions_only, hint):
        self.line = line          # 原始轨迹数据
        self.num_steps = len(line['steps'])
        self.fm = JsonFormat(RAW_SPACE, add_thought=True)  # 格式化器
        self.state = None         # 当前状态

    def get_next(self, model_response):
        """根据模型响应生成下一轮状态"""
        state = self.fm.gen_next_round(self.line, self.state, model_response)
        return state if state else "Finished"


class MultiRoundGenerator:
    """批量多轮交互生成器"""
    def __init__(self, batch, rollout_n, msg_man, patch_threshold=0):
        # 1. 为每个 prompt 分配 uid
        batch.non_tensor_batch["uid"] = [str(uuid.uuid4()) for ...]

        # 2. 重复 n 次用于多次采样
        self.batch = batch.repeat(rollout_n, interleave=True)

        # 3. 为每条轨迹创建唯一 ID
        self.batch.non_tensor_batch["traj_uid"] = [str(uuid.uuid4()) for ...]

        # 4. 为每条轨迹创建状态机
        self.task_queue = [StdTrajectory(line) for line in batch]
        self.finished = [False] * len(self.task_queue)
        self.current_response = [None] * len(self.task_queue)
        self.error_num = [0] * len(self.task_queue)
```

### 5.2 多轮交互流程图

```
┌────────────────────────────────────────────────────────────────────────────┐
│                         Multi-Round Generator 流程                          │
└────────────────────────────────────────────────────────────────────────────┘

      Step 0                  Step 1                  Step 2
    ┌────────┐              ┌────────┐              ┌────────┐
    │ Screen │              │ Screen │              │ Screen │
    │ Image  │──────────────│ Image  │──────────────│ Image  │
    │   +    │   action 1   │   +    │   action 2   │   +    │
    │Instruct│◄─────────────│Instruct│◄─────────────│Instruct│
    └────────┘              └────────┘              └────────┘
         │                       │                       │
         ▼                       ▼                       ▼
    ┌────────┐              ┌────────┐              ┌────────┐
    │ Model  │              │ Model  │              │ Model  │
    │Generate│              │Generate│              │Generate│
    └────────┘              └────────┘              └────────┘
         │                       │                       │
         ▼                       ▼                       ▼
    ┌────────┐              ┌────────┐              ┌────────┐
    │ Check  │              │ Check  │              │ Check  │
    │ Action │              │ Action │              │ Action │
    └────────┘              └────────┘              └────────┘

     Reward: r0               Reward: r1               Reward: r2
```

### 5.3 关键方法详解

#### fetch_batch() - 生成器方法

```python
def fetch_batch(self):
    """生成器：逐步生成每一轮的 batch"""
    while True:
        batch = []

        # 并行处理所有轨迹
        for ptr in range(len(self.task_queue)):
            if self.finished[ptr]:
                continue

            # 获取下一轮状态
            state = self.task_queue[ptr].get_next(self.current_response[ptr])

            if state == "Finished":
                self.finished[ptr] = True
            else:
                # 转换为模型输入格式
                row_dict = self.msg_man(state)
                row_dict['uid'] = self.batch.non_tensor_batch['uid'][ptr]
                row_dict['traj_uid'] = self.batch.non_tensor_batch['traj_uid'][ptr]
                row_dict['step_id'] = state['step_id']
                row_dict['reward_model'] = {
                    "style": "rule",
                    "ground_truth": state['check_options']
                }
                batch.append(row_dict)

        if len(batch) == 0:
            break
        yield collate_fn(batch)
```

#### apply_response() - 应用模型响应

```python
def apply_response(self, batch):
    """将模型响应应用到状态机，决定下一步"""
    failed_num = 0

    for ptr, response, extract_match, reward_model, extra_info in zip(...):
        response_text = self.msg_man.tokenizer.decode(response)
        self.current_response[ptr] = response_text

        if not extract_match:
            failed_num += 1
            # 如果允许 patch 且未超过阈值，使用 ground truth
            if self.patch_threshold > self.error_num[ptr] or self.patch_threshold == -1:
                ground_truth_response = self.fm.format_response(
                    reward_model['ground_truth'], extra_info)
                self.current_response[ptr] = ground_truth_response
                self.error_num[ptr] += 1
            else:
                # 超过错误阈值，终止轨迹
                self.finished[ptr] = True

    return failed_num
```

### 5.4 QwenMessages2Inputs - 消息转换器

```python
class QwenMessages2Inputs:
    """将多轮对话消息转换为 Qwen-VL 模型输入"""

    def __call__(self, state):
        messages = state['messages']

        # 1. 简化消息 (限制图片数量)
        messages = slim_messages(messages, num_image_limit=self.num_image_limit)

        # 2. 处理视觉信息
        image_inputs, video_inputs = process_vision_info(messages)

        # 3. 应用聊天模板
        raw_prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)

        # 4. 编码为模型输入
        model_inputs = self.processor(text=[raw_prompt], images=image_inputs, ...)

        # 5. 后处理 (填充、截断)
        input_ids, attention_mask = verl_F.postprocess_data(
            input_ids, attention_mask,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True
        )

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "multi_modal_data": {"image": image_inputs},
            "reward_model": {"style": "rule", "ground_truth": state['check_options']}
        }
```

---

## 六、奖励计算系统

### 6.1 奖励管理器架构

**文件位置**: `verl/workers/reward_manager/`

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          Reward Manager 架构                                 │
└─────────────────────────────────────────────────────────────────────────────┘

                         ┌─────────────────┐
                         │  Reward Manager │
                         │    (调度器)      │
                         └────────┬────────┘
                                  │
          ┌───────────────────────┼───────────────────────┐
          ▼                       ▼                       ▼
   ┌─────────────┐        ┌─────────────┐        ┌─────────────┐
   │ Rule-based  │        │ Model-based │        │   Sandbox   │
   │  Reward     │        │   Reward    │        │  Reward     │
   │ (GUI 动作)   │        │  (RM 模型)  │        │ (代码执行)  │
   └─────────────┘        └─────────────┘        └─────────────┘
```

### 6.2 奖励计算入口

```python
def compute_reward(batch, reward_fn):
    """计算奖励的入口函数"""
    result = reward_fn(batch, return_dict=True)

    reward_tensor = result["reward_tensor"]  # (batch_size, seq_len)
    reward_extra_info = result.get("reward_extra_info", {})

    return reward_tensor, reward_extra_info
```

### 6.3 GUI 轨迹奖励计算

**文件位置**: `verl/utils/reward_score/gui_traj.py`

```python
def gui_traj_action_match_reward(data_source, solution_str, ground_truth, extra_info):
    """GUI 动作匹配奖励"""

    # 1. 解析模型输出
    parsed_action = parse_model_response(solution_str)

    # 2. 初始化奖励
    rewards = {
        'format_score': 0.0,      # 格式正确性 (0 或 1)
        'type_match': 0.0,        # 动作类型匹配 (0 或 1)
        'extract_match': 0.0,     # 参数提取匹配 (0 或 1)
        'score': 0.0,             # 总分
    }

    # 3. 检查格式
    if is_valid_format(parsed_action):
        rewards['format_score'] = 1.0

    # 4. 检查动作类型
    if parsed_action['action_type'] == ground_truth['action_type']:
        rewards['type_match'] = 1.0

    # 5. 检查参数匹配
    if check_action_params_match(parsed_action, ground_truth, extra_info):
        rewards['extract_match'] = 1.0

    # 6. 计算总分
    rewards['score'] = (rewards['format_score'] * 0.1 +
                       rewards['type_match'] * 0.3 +
                       rewards['extract_match'] * 0.6)

    return rewards
```

### 6.4 UI-S1 步骤奖励计算

**文件位置**: `uis1/core_uis1.py`

```python
def compute_step_discounted_returns(batch, gamma):
    """计算每步的折扣回报 (UI-S1 核心创新)

    对于一条轨迹 [s0, s1, s2, ..., sT]，每步的回报为：
    G_t = r_t + gamma * r_{t+1} + gamma^2 * r_{t+2} + ... + gamma^{T-t} * r_T
    """

    step_rewards = torch.zeros(len(batch))

    # 按轨迹 uid 分组
    unique_traj_uids = set(batch.non_tensor_batch['traj_uid'])

    for traj_uid in unique_traj_uids:
        # 找到该轨迹的所有步骤
        traj_indices = [i for i in range(len(batch))
                       if batch.non_tensor_batch['traj_uid'][i] == traj_uid]

        # 按 step_id 排序
        sorted_indices = sorted(traj_indices,
                               key=lambda i: batch.non_tensor_batch['step_id'][i])

        # 从后向前计算折扣回报
        G = 0
        for idx in reversed(sorted_indices):
            r = batch.non_tensor_batch['rewards'][idx]
            G = r + gamma * G
            step_rewards[idx] = G

    return step_rewards
```

### 6.5 Advantage 估计器

**文件位置**: `verl/trainer/ppo/core_algos.py`

```python
class AdvantageEstimator(str, Enum):
    GAE = "gae"           # Generalized Advantage Estimation
    GRPO = "grpo"         # Group Relative Policy Optimization
    REINFORCE = "reinforce"
    REMAX = "remax"
    UIS1 = "uis1"         # UI-S1 自定义 advantage


def compute_advantage(batch, adv_estimator, gamma, lam,
                     step_advantage_w=0.5, episode_advantage_w=0.5, ...):
    """计算 advantage"""

    if adv_estimator == AdvantageEstimator.GAE:
        # GAE: A_t = sum_{l=0}^{inf} (gamma*lam)^l * delta_{t+l}
        # delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)
        advantages = compute_gae(batch, gamma, lam)

    elif adv_estimator == AdvantageEstimator.GRPO:
        # GRPO: A = (R - mean(R)) / std(R)  # 按组归一化
        advantages = compute_grpo_advantage(batch, norm_adv_by_std_in_grpo)

    elif adv_estimator == AdvantageEstimator.UIS1:
        # UI-S1: 结合步骤级和轨迹级 advantage
        step_adv = compute_step_advantage(batch)
        episode_adv = compute_episode_advantage(batch)
        advantages = step_advantage_w * step_adv + episode_advantage_w * episode_adv

    batch.batch['advantages'] = advantages
    return batch
```

---

## 七、目录结构速查

```
UI-S1/
├── verl/                          # 核心框架
│   ├── __init__.py               # NCCL/GLOO 环境配置
│   ├── protocol.py               # ★ DataProto 数据协议
│   │
│   ├── trainer/
│   │   ├── main_dapo.py          # ★ 训练入口
│   │   ├── main_ppo.py           # PPO 训练入口
│   │   ├── main_eval.py          # 评估入口
│   │   ├── config/
│   │   │   └── ppo_trainer.yaml  # 默认配置
│   │   └── ppo/
│   │       ├── dapo_ray_trainer.py    # ★ DAPO 训练器
│   │       ├── ray_trainer.py         # PPO 训练器基类
│   │       ├── core_algos.py          # Advantage 计算
│   │       ├── reward.py              # 奖励计算入口
│   │       └── metric_utils.py        # 指标计算
│   │
│   ├── workers/
│   │   ├── fsdp_workers.py       # ★ FSDP Worker (Actor/Critic/Ref)
│   │   ├── megatron_workers.py   # Megatron Worker
│   │   ├── rollout/
│   │   │   ├── vllm_rollout/     # vLLM 推理引擎
│   │   │   └── sglang_rollout/   # SGLang 推理引擎
│   │   ├── reward_manager/
│   │   │   ├── naive.py          # 简单奖励管理
│   │   │   ├── dapo.py           # DAPO 奖励管理
│   │   │   └── batch.py          # 批处理奖励
│   │   └── sharding_manager/     # 分片管理
│   │
│   ├── single_controller/
│   │   ├── base/
│   │   │   ├── worker.py         # Worker 基类
│   │   │   └── worker_group.py   # WorkerGroup 基类
│   │   └── ray/
│   │       ├── base.py           # ★ Ray 分布式控制
│   │       └── megatron.py       # Megatron Ray 控制
│   │
│   ├── models/
│   │   ├── transformers/
│   │   │   ├── qwen2_vl.py       # Qwen2-VL 模型
│   │   │   ├── qwen2_5_vl.py     # Qwen2.5-VL 模型
│   │   │   └── monkey_patch.py   # 模型补丁
│   │   └── registry.py           # 模型注册表
│   │
│   └── utils/
│       ├── dataset/
│       │   ├── rl_dataset.py     # RL 数据集
│       │   ├── sft_dataset.py    # SFT 数据集
│       │   ├── universal_multiround.py  # ★ 多轮交互生成器
│       │   └── vision_utils.py   # 视觉处理工具
│       ├── reward_score/
│       │   ├── gui.py            # GUI 单步奖励
│       │   ├── gui_traj.py       # GUI 轨迹奖励
│       │   └── gui_utils/        # GUI 工具函数
│       ├── checkpoint/
│       │   └── fsdp_checkpoint_manager.py  # FSDP 检查点
│       ├── fsdp_utils.py         # FSDP 工具函数
│       ├── vllm_utils.py         # vLLM 工具函数
│       └── tracking.py           # 实验追踪 (W&B)
│
├── uis1/
│   └── core_uis1.py              # ★ UI-S1 核心算法
│
├── x/                            # 项目工具库
│   ├── data/
│   │   └── agent/
│   │       ├── base.py           # Agent 数据格式基类
│   │       ├── json.py           # JSON Action 解析
│   │       ├── json_self_fix.py  # JSON 自修复
│   │       └── space/
│   │           ├── std_space.py       # 标准 Action Space
│   │           ├── android_world.py   # Android Action Space
│   │           └── computer_thought_space.py
│   ├── qwen/
│   │   ├── data_format.py        # 消息格式化
│   │   ├── image.py              # 坐标缩放
│   │   └── agent.py              # Qwen Agent
│   ├── io/
│   │   ├── image_io.py           # 图像 I/O
│   │   └── json.py               # JSON I/O
│   └── parallel/
│       └── parallel_task.py      # 并行任务处理
│
├── evaluation/                    # 评估模块
│   ├── eval_qwenvl.py            # Qwen-VL 评估
│   ├── eval_os-atlas-7b.py       # OS-Atlas 评估
│   ├── eval_os-genesis-7b.py     # OS-Genesis 评估
│   └── dataset/
│       └── android_control_evaluation_std.jsonl
│
└── examples/
    └── qwen_gui_static_grpo/
        └── config/
            └── traj_grpo.yaml    # ★ 训练配置示例
```

---

## 八、核心流程图总结

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          VERL Agent 完整训练流程                             │
└─────────────────────────────────────────────────────────────────────────────┘

 ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
 │  Config  │───►│   Ray    │───►│ Workers  │───►│  Model   │───►│ Trainer  │
 │  (YAML)  │    │   Init   │    │  Init    │    │  Load    │    │  Init    │
 └──────────┘    └──────────┘    └──────────┘    └──────────┘    └──────────┘
                                                                       │
                                                                       ▼
                                  ┌────────────────────────────────────────┐
                                  │              Training Loop              │
                                  │  ┌──────────────────────────────────┐  │
                                  │  │  1. Load Batch (trajectories)    │  │
                                  │  │  2. Multi-Round Generation       │  │
                                  │  │     └─ Step 0 → Step 1 → ...    │  │
                                  │  │  3. Compute Step Rewards         │  │
                                  │  │  4. DAPO Filter (std > 0)        │  │
                                  │  │  5. Compute Advantages (UI-S1)   │  │
                                  │  │  6. PPO/GRPO Update              │  │
                                  │  │  7. Validate & Checkpoint        │  │
                                  │  └──────────────────────────────────┘  │
                                  └────────────────────────────────────────┘
```

### 详细流程

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ Step 1: 配置加载                                                             │
│   - Hydra 加载 YAML 配置                                                     │
│   - 解析模型路径、训练参数、奖励配置等                                          │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ Step 2: Ray 初始化                                                           │
│   - ray.init() 启动 Ray 集群                                                 │
│   - 创建 TaskRunner Remote Actor                                            │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ Step 3: Worker 初始化                                                        │
│   - 创建 RayResourcePool (PlacementGroup)                                   │
│   - 创建 ActorRolloutRefWorker (FSDP 分布式)                                 │
│   - 创建 CriticWorker (可选)                                                 │
│   - torch.distributed.init_process_group()                                  │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ Step 4: 模型加载                                                             │
│   - copy_to_local() 下载模型到本地                                           │
│   - 加载 tokenizer 和 processor                                             │
│   - 构建 FSDP 模型和优化器                                                   │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ Step 5: 训练循环                                                             │
│                                                                             │
│   for epoch in epochs:                                                      │
│       for batch in dataloader:                                              │
│           ┌─────────────────────────────────────────────────────────────┐   │
│           │ 5.1 多轮 Rollout                                             │   │
│           │   - MultiRoundGenerator 逐步生成                             │   │
│           │   - actor_rollout_wg.generate_sequences()                   │   │
│           │   - 计算每步奖励                                             │   │
│           │   - apply_response() 更新状态                                │   │
│           └─────────────────────────────────────────────────────────────┘   │
│                                     │                                       │
│                                     ▼                                       │
│           ┌─────────────────────────────────────────────────────────────┐   │
│           │ 5.2 DAPO 过滤                                                │   │
│           │   - 按 prompt uid 分组                                       │   │
│           │   - 计算组内 reward std                                      │   │
│           │   - 过滤 std < threshold 的组                                │   │
│           └─────────────────────────────────────────────────────────────┘   │
│                                     │                                       │
│                                     ▼                                       │
│           ┌─────────────────────────────────────────────────────────────┐   │
│           │ 5.3 计算 Log Prob                                            │   │
│           │   - actor_rollout_wg.compute_log_prob() → old_log_prob      │   │
│           │   - ref_policy_wg.compute_ref_log_prob() → ref_log_prob     │   │
│           └─────────────────────────────────────────────────────────────┘   │
│                                     │                                       │
│                                     ▼                                       │
│           ┌─────────────────────────────────────────────────────────────┐   │
│           │ 5.4 计算 Advantage                                           │   │
│           │   - compute_step_discounted_returns() (UI-S1)               │   │
│           │   - compute_advantage() → advantages                        │   │
│           └─────────────────────────────────────────────────────────────┘   │
│                                     │                                       │
│                                     ▼                                       │
│           ┌─────────────────────────────────────────────────────────────┐   │
│           │ 5.5 更新模型                                                 │   │
│           │   - critic_wg.update_critic() (可选)                        │   │
│           │   - actor_rollout_wg.update_actor() → PPO/GRPO loss         │   │
│           └─────────────────────────────────────────────────────────────┘   │
│                                     │                                       │
│                                     ▼                                       │
│           ┌─────────────────────────────────────────────────────────────┐   │
│           │ 5.6 验证和保存                                               │   │
│           │   - _validate() → val_metrics                               │   │
│           │   - _save_checkpoint()                                      │   │
│           │   - logger.log() → W&B / SwanLab                           │   │
│           └─────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 九、关键概念对照表

| 概念 | 文件位置 | 说明 |
|------|----------|------|
| **DataProto** | `verl/protocol.py` | 统一数据传输协议，包含 TensorDict + numpy dict |
| **DAPO** | `dapo_ray_trainer.py` | 数据驱动 PPO，过滤 std=0 的组 |
| **MultiRoundGenerator** | `universal_multiround.py` | GUI 多轮交互生成器，支持轨迹级训练 |
| **FSDP Worker** | `fsdp_workers.py` | 分布式模型训练，支持 Actor/Rollout/Ref 三合一 |
| **RayWorkerGroup** | `ray/base.py` | Ray Actor 组管理，统一调用接口 |
| **UI-S1 Advantage** | `uis1/core_uis1.py` | 步骤级+轨迹级优势结合 |
| **Action Space** | `x/data/agent/space/` | GUI 动作定义 (click, type, scroll 等) |
| **PlacementGroup** | `ray/base.py` | Ray 资源分配单元 |
| **Device Mesh** | `fsdp_workers.py` | FSDP 设备网格，控制分片策略 |
| **Rollout** | `workers/rollout/` | 模型推理引擎 (vLLM/SGLang) |

---

## 十、框架核心创新点

### 10.1 DataProto 统一数据协议

- 统一了 Tensor 和非 Tensor 数据的传输
- 支持自动分片和合并，简化分布式训练
- 支持序列化，可在 Ray Actor 之间高效传输

### 10.2 MultiRoundGenerator 多轮交互

- 支持 GUI Agent 的多步骤交互训练
- 自动管理轨迹状态机
- 支持错误 patch 机制（用 ground truth 替换失败的动作）

### 10.3 DAPO 数据过滤

- 过滤没有学习信号的样本（所有采样结果相同）
- 提高训练效率，避免无意义的梯度更新

### 10.4 UI-S1 Advantage 估计

- 结合步骤级和轨迹级奖励
- 支持折扣回报计算
- 可配置步骤权重和轨迹权重

### 10.5 Actor-Rollout-Ref 三合一 Worker

- 同一个 Worker 可以扮演多种角色
- 减少显存占用，避免模型重复加载
- 支持灵活的资源配置

---

## 十一、配置示例

```yaml
# examples/qwen_gui_static_grpo/config/traj_grpo.yaml

trainer:
  project_name: gui_traj_grpo
  experiment_name: qwen2_5_vl_7b
  total_epochs: 3
  total_training_steps: 1000
  save_freq: 100
  test_freq: 50
  logger: wandb

data:
  train_files: datasets/ui_s1_train.jsonl
  val_files: datasets/android_control_evaluation.jsonl
  train_batch_size: 8
  max_prompt_length: 32768

actor_rollout_ref:
  model:
    path: /path/to/Qwen2.5-VL-7B-Instruct
  actor:
    strategy: fsdp
    ppo_mini_batch_size: 4
    ppo_micro_batch_size_per_gpu: 1
    ppo_epochs: 1
  rollout:
    name: vllm
    n: 4  # 每个 prompt 采样 4 次
    temperature: 0.7
    max_new_tokens: 512

algorithm:
  adv_estimator: uis1
  gamma: 0.99
  filter_groups:
    enable: true
    std_threshold: 0.01
  uis1:
    step_advantage_w: 0.5
    episode_advantage_w: 0.5

reward_model:
  enable: false  # 使用规则奖励
```

---

## 十二、常见问题

### Q1: DataProto 和普通 Dict 有什么区别？

DataProto 的优势：
1. 统一的批处理操作 (`chunk`, `concat`, `repeat`)
2. 自动处理 Tensor 和非 Tensor 数据的一致性
3. 内置序列化支持，适合分布式传输
4. 支持自动 padding

### Q2: 为什么需要 DAPO 过滤？

当一个 prompt 的所有采样结果完全相同时（std=0），GRPO/PPO 的 advantage 为 0，不会产生任何梯度更新。DAPO 过滤掉这些样本，节省计算资源。

### Q3: MultiRoundGenerator 的 patch_threshold 是什么？

当模型生成的动作无法正确解析时：
- `patch_threshold > 0`: 允许用 ground truth 替换失败的动作，继续轨迹
- `patch_threshold = 0`: 一旦失败就终止轨迹
- `patch_threshold = -1`: 无限允许 patch

### Q4: UI-S1 Advantage 和 GRPO 有什么区别？

- GRPO: 只看最终奖励，按 prompt 组归一化
- UI-S1: 计算每步的折扣回报，结合步骤级和轨迹级 advantage

---

*文档生成时间: 2026-02-02*
*项目: UI-S1 (Advancing GUI Automation via Semi-online Reinforcement Learning)*
