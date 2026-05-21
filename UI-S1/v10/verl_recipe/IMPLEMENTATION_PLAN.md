# v10 Cooperative LoRA → verl Recipe: Implementation Plan

## Executive Summary

将v10的dual-adapter sequential GRPO训练迁移到verl框架上，核心收益：
- **vLLM推理**：替代HF `model.generate()`，K=8 rollout吞吐量提升5-10x
- **FSDP2内存效率**：verl的FSDP2 per-parameter sharding + CPU offload，支持更大batch/更多K
- **Ray编排**：driver-worker分离，所有分布式同步由框架处理，不再手动`dist.all_reduce`
- **基础设施复用**：checkpoint/resume、WandB logging、sequence balancing等开箱即用

---

## 1. 架构对比：v10 vs verl Recipe

```
v10 (当前)                                verl Recipe (目标)
─────────────────────                    ──────────────────────
torchrun DDP (manual all_reduce)    →    Ray single-controller + FSDP2 workers
HF model.generate() (slow)          →    vLLM rollout engine (fast)
单进程 train+generate               →    Actor/Rollout/Ref 角色分离
手写 gradient checkpointing toggle  →    verl自动管理
手动 loss normalization             →    verl的DataParallelPPOActor
```

### 关键语义映射

| v10 概念 | verl 对应 |
|----------|-----------|
| `_set_adapter_safe("grounder")` | `model.set_adapter("grounder_lora")` + requires_grad fix |
| `model.generate()` (K=8 batched) | `vLLMRollout.generate_sequences()` |
| `_compute_token_log_probs(with_grad=True)` | `DataParallelPPOActor.compute_log_prob()` (内部forward+loss) |
| `_compute_ref_log_probs()` | `compute_ref_log_prob()` (verl自带 `disable_adapter()`) |
| `dist.all_reduce(p.grad, AVG)` | FSDP2自动处理 |
| `compute_policy_loss()` | `verl.trainer.ppo.core_algos.compute_policy_loss()` |
| `compute_kl_penalty()` | `verl.trainer.ppo.core_algos.kl_penalty()` |
| `compute_grpo_advantage()` | `core_algos.compute_grpo_outcome_advantage()` |

---

## 2. Recipe 目录结构

```
v10/verl_recipe/
├── IMPLEMENTATION_PLAN.md          # 本文档
├── config/
│   └── coop_grpo_gui.yaml          # 主配置文件
├── main_coop_grpo.py               # 入口：初始化Ray + 启动trainer
├── coop_ray_trainer.py             # 核心：双阶段fit()循环
├── coop_fsdp_workers.py            # Worker：双adapter承载
├── coop_dp_actor.py                # Actor：双adapter训练逻辑
├── reward_fn.py                    # 从v10/reward.py迁移
└── coop_dataset.py                 # Parquet数据加载
```

---

## 3. 核心组件设计

### 3.1 CoopActorRolloutRefWorker (coop_fsdp_workers.py)

**继承**: `ActorRolloutRefWorker`

**改造点**:

#### A. 双adapter加载 (`_build_model_optimizer` override)

```python
class CoopActorRolloutRefWorker(ActorRolloutRefWorker):
    def _build_model_optimizer(self, model_path, fsdp_config, optim_config, ...):
        # 1. 父类加载基座 + 第一个adapter (grounder_lora)
        #    verl现有的LoRA路径:
        #      actor_module = get_peft_model(actor_module, LoraConfig(**lora_config))
        #    我们把这个adapter命名为"grounder_lora"

        # 2. 在FSDP wrap之前，add_adapter("actor_lora", same_config)
        #    FSDP2是per-parameter sharding，所以add_adapter增加的新参数
        #    会作为独立的FSDP unit被shard——这是v10在FSDP1下做不到的

        # 3. 强制所有lora_参数 requires_grad=True

        # 4. 构建双param group optimizer:
        #    grounder_params → lr=1e-5
        #    actor_params    → lr=5e-6

        # 5. 继续走父类的FSDP wrap + rollout init
```

**关键问题**: verl的`get_peft_model()`在line 417只传了一个adapter。我们需要在line 417和FSDP wrap(line 474-509)之间插入`add_adapter()`调用。

**实现策略**: 不直接override `_build_model_optimizer`(太长，800+行），而是在`init_model`里，在调用`_build_model_optimizer`之后、使用返回值之前，hook进去添加第二个adapter。

```python
def init_model(self):
    super().init_model()

    if self._is_actor:
        # 此时 self.actor_module_fsdp 已经被FSDP包装
        # 但是对于FSDP2，内部的module依然可以访问
        # 我们需要在FSDP wrap之前add adapter
        # → 所以必须override _build_model_optimizer
```

**更好的方案**: 在`_build_model_optimizer`的LoRA阶段后、FSDP阶段前插入hook：

```python
# 在 fsdp_workers.py line 417 之后插入:
if hasattr(self.config.model, 'dual_adapter') and self.config.model.dual_adapter:
    # 第一个adapter已经加载，给它命名
    # PEFT默认adapter名是"default"，我们rename或重新add
    second_adapter_config = LoraConfig(**lora_config)  # same config
    actor_module.add_adapter("actor_lora", second_adapter_config)
    # 强制所有LoRA参数可训练
    for name, param in actor_module.named_parameters():
        if "lora_" in name:
            param.requires_grad = True
```

**风险评估**: FSDP2的`apply_fsdp2()`在line 505遍历所有参数进行sharding。如果第二个adapter的参数在这之前已经注册到module tree上，FSDP2会自然地把它们也shard掉。这在理论上是正确的（FSDP2是per-parameter sharding），但需要验证PEFT的`set_adapter()`在FSDP2环境下能否正确切换active adapter。

**验证方案**: Week 1的unit test（见§6）。

#### B. Adapter安全切换

```python
def _set_adapter_safe(self, adapter_name: str):
    """切换active adapter并确保所有LoRA参数可训练"""
    base_model = self.actor_module_fsdp
    # 对于FSDP2 wrapped model，需要找到内部的PeftModel
    peft_model = self._get_peft_model(base_model)
    peft_model.set_adapter(adapter_name)
    for name, param in peft_model.named_parameters():
        if "lora_" in name:
            param.requires_grad = True
```

#### C. 新增RPC接口

```python
@register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
def generate_with_adapter(self, prompts: DataProto):
    """用指定adapter生成序列"""
    adapter_name = prompts.meta_info.get("adapter_name", "default")
    # 切换训练模型的adapter（影响weight sync到vLLM）
    self._set_adapter_safe(adapter_name)
    # vLLM侧的adapter切换
    self._sync_adapter_to_vllm(adapter_name)
    return self.generate_sequences(prompts)

@register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
def compute_log_prob_with_adapter(self, data: DataProto):
    """用指定adapter计算log prob"""
    adapter_name = data.meta_info.get("adapter_name", "default")
    self._set_adapter_safe(adapter_name)
    return self.compute_log_prob(data)

@register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
def update_actor_with_adapter(self, data: DataProto):
    """用指定adapter做policy update"""
    adapter_name = data.meta_info.get("adapter_name", "default")
    self._set_adapter_safe(adapter_name)
    return self.update_actor(data)
```

#### D. Dual-Optimizer

verl现有的optimizer初始化在`_build_model_optimizer`的line 520-548。需要改成dual param group：

```python
# 替代单一optimizer：
grounder_params = [p for n, p in actor_module_fsdp.named_parameters()
                   if "grounder_lora" in n or "default" in n]  # default is grounder
actor_params = [p for n, p in actor_module_fsdp.named_parameters()
                if "actor_lora" in n]

actor_optimizer = optim.AdamW([
    {"params": grounder_params, "lr": self.config.actor.grounder_lr},
    {"params": actor_params, "lr": self.config.actor.actor_lr},
], weight_decay=optim_config.get("weight_decay", 1e-2))
```

**注意**: 在FSDP2下，参数名可能被FSDP prefix改变。需要用`original_name`或在FSDP wrap前记录好参数到adapter的映射。

### 3.2 CoopDataParallelActor (coop_dp_actor.py)

**继承**: `DataParallelPPOActor`

**改造点**: `update_policy` 不需要大改。关键是确保：

1. 当adapter切换后，forward图只包含active adapter的参数
2. backward只更新active adapter的参数
3. FSDP2的gradient sync只sync active adapter的参数

```python
class CoopDataParallelPPOActor(DataParallelPPOActor):
    def update_policy(self, data: DataProto):
        # data.meta_info["adapter_name"] 由driver设置
        # Worker层已经在 update_actor_with_adapter 里切好了adapter
        # 直接调用父类的 update_policy
        # 因为非active adapter在forward图里是disconnect的，backward天然不动它
        return super().update_policy(data=data)
```

**FSDP2梯度隔离的正确性论证**:

v10里的梯度隔离依赖PEFT的`set_adapter()`：当`set_adapter("grounder")`时，actor的LoRA参数虽然存在于模型中，但forward时不被使用（PEFT内部把它们从计算图中disconnect），所以backward不会产生梯度。

在FSDP2下，这个语义依然成立：
- FSDP2是per-parameter sharding，每个参数独立管理
- `set_adapter()`改变的是PEFT内部的forward路径，不影响FSDP2的sharding
- 非active adapter的参数在forward中不参与计算 → backward中`.grad = None`
- FSDP2的gradient reduce只在有梯度的参数上做reduce

**但有一个边界case**: FSDP2可能在backward结束后对所有参数做reduce，如果某些参数`grad=None`可能导致hang。需要验证。

### 3.3 CoopRayTrainer (coop_ray_trainer.py)

**继承**: `RayPPOTrainer`

**核心改造**: override `fit()` 实现双阶段流水线。

```python
class CoopRayTrainer(RayPPOTrainer):
    def fit(self):
        for global_step in range(self.total_training_steps):
            batch = next(self.train_dataloader)
            batch = DataProto.from_single_dict(batch)

            # ═══════════ Phase 1: Grounder Rollout ═══════════
            g_batch = batch.repeat(repeat_times=self.config.rollout.n, interleave=True)
            g_batch.meta_info["adapter_name"] = "grounder_lora"

            # 用grounder adapter生成
            g_gen_batch = g_batch.pop(batch_keys=[...], non_tensor_batch_keys=[...])
            g_output = self.actor_rollout_wg.generate_with_adapter(g_gen_batch)
            g_batch = g_batch.union(g_output)

            # ═══════════ Phase 2: Actor Rollout ═══════════
            # 构建actor prompt：把grounder输出拼入
            a_batch = self._build_actor_batch(batch, g_batch)
            a_batch.meta_info["adapter_name"] = "actor_lora"

            a_gen_batch = a_batch.pop(batch_keys=[...], non_tensor_batch_keys=[...])
            a_output = self.actor_rollout_wg.generate_with_adapter(a_gen_batch)
            a_batch = a_batch.union(a_output)

            # ═══════════ Reward ═══════════
            g_rewards, a_rewards = self._compute_coop_rewards(a_batch, batch)
            g_batch.batch["token_level_scores"] = g_rewards
            a_batch.batch["token_level_scores"] = a_rewards

            # ═══════════ Advantage (driver侧, 轻量级) ═══════════
            g_batch = compute_advantage(g_batch, adv_estimator="grpo", ...)
            a_batch = compute_advantage(a_batch, adv_estimator="grpo", ...)

            # ═══════════ Log Probs ═══════════
            g_batch.meta_info["adapter_name"] = "grounder_lora"
            g_old_lp = self.actor_rollout_wg.compute_log_prob_with_adapter(g_batch)
            g_batch = g_batch.union(g_old_lp)

            a_batch.meta_info["adapter_name"] = "actor_lora"
            a_old_lp = self.actor_rollout_wg.compute_log_prob_with_adapter(a_batch)
            a_batch = a_batch.union(a_old_lp)

            # Ref log probs (基座, verl自带disable_adapter机制)
            g_ref = self.actor_rollout_wg.compute_ref_log_prob(g_batch)
            g_batch = g_batch.union(g_ref)
            a_ref = self.actor_rollout_wg.compute_ref_log_prob(a_batch)
            a_batch = a_batch.union(a_ref)

            # ═══════════ Update ═══════════
            # Phase 1: Grounder update
            g_batch.meta_info["adapter_name"] = "grounder_lora"
            g_metrics = self.actor_rollout_wg.update_actor_with_adapter(g_batch)

            # Phase 2: Actor update
            a_batch.meta_info["adapter_name"] = "actor_lora"
            a_metrics = self.actor_rollout_wg.update_actor_with_adapter(a_batch)

            # ═══════════ vLLM Weight Sync ═══════════
            # 每次update后sync权重到vLLM
            self.actor_rollout_wg.sync_weights_to_rollout()

            # Logging
            self._log_coop_metrics(g_metrics, a_metrics, global_step)
```

#### Actor Prompt构建

```python
def _build_actor_batch(self, original_batch, grounder_batch):
    """把grounder的输出拼接到actor的prompt里"""
    # grounder_texts = decode(grounder_batch.batch["responses"])
    # actor_prompts = [
    #   format_actor_text(goal, history, grounder_text)
    #   for goal, history, grounder_text in zip(...)
    # ]
    # 重新tokenize actor_prompts → new input_ids, attention_mask
    # 返回新的DataProto
```

这是最复杂的部分——需要在driver侧做decode + re-tokenize。verl的DataProto是tensor-based的，所以这个操作需要：
1. 在driver侧decode grounder responses → text
2. 构建actor prompt (text)
3. 用processor重新tokenize（包括图像处理）
4. 打包成新的DataProto

**性能考量**: 这一步在CPU上做，K=8时需要decode K*batch_size条序列。Tokenizer操作本身很快（<1s），但图像处理可能需要优化。

#### Ref Log Prob的语义正确性

verl的`compute_ref_log_prob`在`fsdp_workers.py` line 1069实现：
```python
is_lora = data.meta_info.pop("is_lora", False)
adapter_ctx = self.actor.actor_module.disable_adapter() if is_lora else nullcontext()
```

这里`disable_adapter()`会disable**所有**adapter layers（包括grounder和actor的LoRA），让forward只使用基座权重。这对我们的场景是正确的：两个adapter共享同一个基座作为reference policy。

**但是**: 当前verl的`compute_ref_log_prob`内部会设`data.meta_info["is_lora"] = True`（在`fsdp_workers.py`里检查`self._is_lora`）。对于dual-adapter场景，这个语义是一样的——disable all adapters = 基座。所以**不需要修改ref log prob逻辑**。

### 3.4 vLLM Multi-LoRA Strategy

#### Stage 1: Simple Adapter Reload (方案A)

每次phase切换时，把当前active adapter的权重sync到vLLM：

```python
def _sync_adapter_to_vllm(self, adapter_name):
    """把训练侧指定adapter的LoRA权重同步到vLLM"""
    # verl现有的weight sync机制:
    # rollout_sharding_manager.preprocess_data() 里会做 FSDP gather → vLLM load
    # 但这个机制假设只有一个adapter

    # 方案A: 每次切换时 merge LoRA到基座 → sync全量权重到vLLM
    # 优点: 最简单，vLLM侧不需要任何LoRA支持
    # 缺点: 每次phase切换 ~10-20s 的weight sync开销

    # 方案A实现:
    peft_model = self._get_peft_model(self.actor_module_fsdp)
    peft_model.set_adapter(adapter_name)
    # verl的rollout_sharding_manager会自动把FSDP模型的权重sync到vLLM
    # 只需要确保set_adapter在sync之前被调用
```

**方案A的开销分析**:
- 2次weight sync per step (grounder → actor)
- 每次sync: FSDP gather (~5s) + vLLM load (~5-10s) ≈ 15s
- 每step额外开销: 30s
- 相对于v10的~130s/step, 这是~23%开销
- 但vLLM推理本身比HF generate快5-10x，所以net speedup依然显著

#### Stage 2: vLLM Multi-LoRA (方案B, 后期优化)

vLLM原生支持`LoRARequest`，可以在同一个基座上serve多个adapter：

```python
from vllm.lora.request import LoRARequest

# vLLM生成时指定LoRA
outputs = llm.generate(
    prompts,
    lora_request=LoRARequest("grounder_lora", 1, grounder_lora_path)
)
```

这需要修改verl的`vllm_rollout.py`（~200行改动），让它支持：
1. 加载多个LoRA adapter到vLLM engine
2. 在generate时传入`lora_request`参数
3. 训练权重更新后，动态reload对应adapter

收益：两次generate之间不需要weight sync，可以pipeline化。

### 3.5 Reward Function (reward_fn.py)

从v10/reward.py直接迁移，适配verl的reward接口：

```python
def coop_reward_fn(data_source, solution_str, ground_truth, extra_info=None):
    """verl的标准reward接口"""
    # data_source: "gui_360" 或 "android_control"
    # solution_str: actor的输出文本
    # ground_truth: GT action dict
    # extra_info: {"image_w": ..., "image_h": ..., "grounder_output": ...}

    from v10.reward import grounder_reward, actor_reward

    g_score = grounder_reward(solution_str, ground_truth,
                               extra_info["image_w"], extra_info["image_h"])
    a_score = actor_reward(solution_str, ground_truth,
                            extra_info["image_w"], extra_info["image_h"])
    return {"grounder": g_score, "actor": a_score}
```

**注意**: verl的标准reward接口`compute_reward(batch, reward_fn)`在`verl/trainer/ppo/reward.py`里会对batch中的每个sample调用`reward_fn`。我们需要扩展这个接口来支持返回dual reward。

最简单的做法：在`CoopRayTrainer._compute_coop_rewards()`里直接调用reward函数，绕过verl的标准reward pipeline。

### 3.6 Dataset (coop_dataset.py)

verl要求parquet格式的数据。需要将现有的JSONL数据转换：

```python
# GUI-360数据转换脚本
import pandas as pd
import json

def convert_gui360_to_parquet(jsonl_path, output_path):
    records = []
    with open(jsonl_path) as f:
        for line in f:
            item = json.loads(line)
            records.append({
                "data_source": "gui_360",
                "prompt": json.dumps(item["conversations"][:1]),  # 第一轮user message
                "images": item.get("images", []),
                "reward_model": json.dumps({
                    "style": "rule",
                    "ground_truth": item.get("gt_action", {})
                }),
                "extra_info": json.dumps({
                    "image_w": item.get("image_w", 1080),
                    "image_h": item.get("image_h", 2400),
                    "history": item.get("history", "")
                })
            })
    df = pd.DataFrame(records)
    df.to_parquet(output_path)
```

---

## 4. 配置文件 (config/coop_grpo_gui.yaml)

```yaml
# Cooperative GRPO for GUI Agents
data:
  train_files: data/gui360_train.parquet
  val_files: data/gui360_val.parquet
  train_batch_size: 32  # 每个global step的总sample数
  max_prompt_length: 4096
  max_response_length: 256

actor_rollout_ref:
  hybrid_engine: true
  model:
    path: Qwen/Qwen2.5-VL-7B-Instruct
    trust_remote_code: true
    lora_rank: 128
    lora_alpha: 256
    target_modules: [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]
    # === 新增: 双adapter配置 ===
    dual_adapter: true
    adapter_names: [grounder_lora, actor_lora]

  actor:
    strategy: fsdp2
    ppo_mini_batch_size: 8
    ppo_epochs: 1
    clip_ratio: 0.2
    use_kl_loss: true
    kl_loss_coef: 0.001
    kl_loss_type: low_var_kl
    loss_agg_mode: token_mean
    # === 新增: 双adapter学习率 ===
    grounder_lr: 1.0e-5
    actor_lr: 5.0e-6
    optim:
      lr: 1.0e-5  # fallback
      weight_decay: 0.01
      lr_warmup_steps_ratio: 0.03
      warmup_style: cosine
      min_lr_ratio: 0.0
    fsdp_config:
      wrap_policy: null
      mixed_precision:
        param_dtype: bf16
        reduce_dtype: fp32
      reshard_after_forward: true
      forward_prefetch: false
    checkpoint:
      load_contents: [model, optimizer, extra]
      save_contents: [model, optimizer, extra]

  ref:
    fsdp_config:
      wrap_policy: null
      mixed_precision:
        param_dtype: bf16
        reduce_dtype: fp32

  rollout:
    name: vllm
    n: 8  # K=8 for GRPO
    temperature: 1.0
    top_p: 0.95
    max_new_tokens: 256
    gpu_memory_utilization: 0.4
    tensor_parallel_size: 1
    log_prob_micro_batch_size_per_gpu: 8
    # === 新增: 多adapter rollout ===
    multi_lora: false  # Stage 1不用; Stage 2开启

algorithm:
  adv_estimator: grpo
  use_kl_in_reward: false  # KL作为loss term而非reward
  gamma: 1.0
  lam: 1.0
  norm_adv_by_std_in_grpo: true
  # === 新增 ===
  cooperative:
    enable: true
    grounder_max_tokens: 256
    actor_max_tokens: 256

reward_model:
  reward_fn: v10.verl_recipe.reward_fn.coop_reward_fn

trainer:
  total_epochs: 4
  project_name: coop_grpo_gui
  experiment_name: v10_verl
  logger: wandb
  save_freq: 100
  test_freq: 50
  balance_batch: true
  del_local_ckpt_after_load: false
```

---

## 5. 数据流详解

一个完整的training step数据流:

```
                    Driver (CoopRayTrainer.fit)
                    ══════════════════════════

Step 1: Load batch (N samples from parquet)
        batch: {input_ids, attention_mask, images, gt_actions}
                          │
Step 2: Repeat K=8 times  │
        g_batch: N*8 samples│
                          │
Step 3: ──────────────────▼────────── RPC ──────────────────
        actor_rollout_wg.generate_with_adapter(g_batch, "grounder_lora")
        │
        │  Worker内部:
        │  1. set_adapter("grounder_lora")
        │  2. sync weights to vLLM (if needed)
        │  3. vLLM generate → grounder_texts
        │
        ◄──────────────── g_output (N*8 sequences) ─────────
                          │
Step 4: Decode grounder outputs → text
        Build actor prompts: [goal + history + grounder_text]
        Re-tokenize → a_batch
                          │
Step 5: ──────────────────▼────────── RPC ──────────────────
        actor_rollout_wg.generate_with_adapter(a_batch, "actor_lora")
        │
        │  Worker内部: same flow with actor_lora
        │
        ◄──────────────── a_output (N*8 sequences) ─────────
                          │
Step 6: Compute rewards (driver, CPU)
        g_rewards[i] = grounder_reward(a_output[i], gt_action[i])
        a_rewards[i] = actor_reward(a_output[i], gt_action[i])
                          │
Step 7: Compute advantages (driver, CPU)
        g_adv = GRPO_advantage(g_rewards, group_size=K)
        a_adv = GRPO_advantage(a_rewards, group_size=K)
                          │
Step 8: ──────────────────▼────────── RPC ──────────────────
        Compute old_log_probs (grounder) → g_old_lp
        Compute old_log_probs (actor)    → a_old_lp
        Compute ref_log_probs (grounder) → g_ref_lp  (disable_adapter)
        Compute ref_log_probs (actor)    → a_ref_lp  (disable_adapter)
        ◄────────────────────────────────────────────────────
                          │
Step 9: ──────────────────▼────────── RPC ──────────────────
        update_actor_with_adapter(g_batch, "grounder_lora")
        │  Worker: set_adapter → forward → backward → FSDP reduce
        ◄──── g_metrics ─────

        update_actor_with_adapter(a_batch, "actor_lora")
        │  Worker: set_adapter → forward → backward → FSDP reduce
        ◄──── a_metrics ─────
                          │
Step 10: sync_weights_to_rollout()
         Log metrics to WandB
```

---

## 6. 实施路径 & 里程碑

### Week 1: FSDP2 + Multi-Adapter验证

**目标**: 验证PEFT multi-adapter在verl的FSDP2环境下能工作

**具体任务**:
1. 写一个minimal test: 加载Qwen2.5-VL-3B + 两个LoRA adapter
2. 用`apply_fsdp2()`包装
3. 验证`set_adapter()`切换后forward输出不同
4. 验证backward只更新active adapter的参数
5. 测试`disable_adapter()`能正确给出ref log probs

**验证脚本**:
```python
# test_dual_adapter_fsdp2.py
from peft import get_peft_model, LoraConfig
from verl.utils.fsdp_utils import apply_fsdp2

model = load_base_model("Qwen/Qwen2.5-VL-3B-Instruct")
lora_cfg = LoraConfig(r=16, ...)
model = get_peft_model(model, lora_cfg, adapter_name="grounder_lora")
model.add_adapter("actor_lora", lora_cfg)

# FSDP2 wrap
apply_fsdp2(model, fsdp_kwargs, fsdp_config)

# Test 1: set_adapter changes forward output
model.set_adapter("grounder_lora")
out_g = model(input_ids)
model.set_adapter("actor_lora")
out_a = model(input_ids)
assert not torch.allclose(out_g, out_a)  # Different adapters → different outputs

# Test 2: backward isolation
model.set_adapter("grounder_lora")
loss = model(input_ids).loss
loss.backward()
# Check: grounder params have grad, actor params don't
for n, p in model.named_parameters():
    if "grounder_lora" in n:
        assert p.grad is not None
    elif "actor_lora" in n:
        assert p.grad is None  # Should be None
```

**风险缓解**: 如果FSDP2 + multi-adapter不工作:
- Fallback: 用FSDP1的`use_orig_params=True`（verl支持FSDP1）
- 或者：用两个独立的FSDP model各包一个adapter（内存翻倍，但最安全）
- 或者：放弃FSDP，用v10的manual all-reduce但在verl的Ray框架里做

### Week 2: CoopActorRolloutRefWorker + 双adapter训练

**目标**: 实现完整的worker类，能做双adapter的generate/log_prob/update

**任务**:
1. 实现`coop_fsdp_workers.py`的`CoopActorRolloutRefWorker`
2. 实现`coop_dp_actor.py`的`CoopDataParallelPPOActor`
3. 实现dual optimizer (两个param group)
4. 用fake batch验证：
   - 两个adapter各自update后，权重确实变化
   - 另一个adapter的权重没变
   - gradient norm > 0

### Week 3: CoopRayTrainer fit loop + Reward迁移

**目标**: 端到端跑通一个minimal training loop

**任务**:
1. 实现`coop_ray_trainer.py`的`CoopRayTrainer`
2. 迁移v10/reward.py到recipe
3. 实现`_build_actor_batch()` (grounder output → actor prompt)
4. 实现dual-reward + dual-advantage计算
5. 2节点4GPU, 100 step端到端测试

**关键指标**:
- [ ] 两个adapter的loss/reward曲线出现
- [ ] grad_norm > 0 for both adapters
- [ ] g_nz / a_nz > 50%

### Week 4: 对比验证 + 性能调优

**目标**: 与v10在同数据集上对比

**任务**:
1. 同数据集 (GUI-360 subset) 跑v10和verl recipe各100 step
2. 对比:
   - reward曲线形状
   - advantage非零比例
   - KL divergence
   - 两个adapter的B矩阵正交性
3. 吞吐量对比: samples/sec

### Week 5-6: vLLM Multi-LoRA (方案B) + 性能优化

**目标**: 消除adapter切换的weight sync开销

**任务**:
1. 修改`vllm_rollout.py`支持`LoRARequest`
2. 训练时动态reload adapter权重到vLLM
3. 两次generate pipeline化
4. 目标: 吞吐量比v10快2x+

### Week 7: 生产化

**任务**:
- Checkpoint save/resume对dual adapter的支持
- Validation loop（rank 0 greedy decode）
- WandB dashboard with dual metrics
- 配置文件完善
- 多数据集支持 (GUI-360 / AndroidControl / Odyssey)

### Week 8: 清理 & 文档

**任务**:
- 代码清理和注释
- README
- 使用指南
- 性能benchmark结果

---

## 7. 风险矩阵

| 风险 | 概率 | 影响 | 缓解策略 |
|------|------|------|----------|
| FSDP2 + multi-adapter不兼容 | 中 | 高 | Week 1先验证; Fallback到FSDP1或manual all-reduce |
| vLLM multi-LoRA serving在verl下不稳定 | 中 | 中 | Stage 1用方案A (simple reload); Stage 2再优化 |
| PEFT `set_adapter`在FSDP2下破坏参数映射 | 低 | 高 | 在FSDP wrap前冻结adapter列表; 不在训练中添加/删除adapter |
| Dual optimizer的lr scheduler不同步 | 低 | 低 | 共用一个cosine scheduler，只是param group lr不同 |
| Driver侧decode+re-tokenize成为瓶颈 | 低 | 中 | 可以parallel化或移到worker侧 |
| 两个adapter的gradient sync不匹配 (NCCL deadlock) | 中 | 高 | 确保所有rank的adapter切换时序一致; DP_COMPUTE_PROTO dispatch保证这一点 |

---

## 8. 与v10的关键差异总结

| 维度 | v10 | verl Recipe |
|------|-----|-------------|
| **Generation** | HF generate, 单GPU, ~40s/sample | vLLM, 分布式, ~2-5s/sample |
| **Gradient sync** | Manual dist.all_reduce | FSDP2自动 |
| **Deadlock防御** | 手写zero-grad for None | FSDP2框架保证 |
| **Grad checkpointing** | 手动toggle per phase | verl自动管理 |
| **Ref log prob** | 手写disable_adapter + restore | verl自带 |
| **Checkpoint** | 手写save_pretrained | verl的FSDPCheckpointManager |
| **Batch balancing** | 无 | verl的seqlen_balancing |
| **Metrics** | 手写logging | verl的Tracking (WandB/TensorBoard) |
| **Data loading** | 手写Dataset + DistributedSampler | verl的parquet pipeline |
| **Resume** | 手写epoch/step tracking | verl的StatefulDataLoader + checkpoint |

---

## 9. 预期性能

### 吞吐量

| Phase | v10 (16 GPU, HF gen) | verl Stage 1 (8 GPU, vLLM) | verl Stage 2 (8 GPU, multi-LoRA) |
|-------|---------------------|---------------------------|----------------------------------|
| Grounder gen (K=8) | ~10s | ~2s | ~2s |
| Actor gen (K=8) | ~30s (sequential) | ~3s (batched) | ~3s |
| Weight sync | 0s | ~30s (2x15s) | ~0s |
| Old log probs | ~10s | ~5s | ~5s |
| Ref log probs | ~10s | ~5s | ~5s |
| Update (both) | ~70s | ~30s | ~30s |
| **Total per step** | **~130s** | **~75s** | **~45s** |
| **Speedup** | 1x | 1.7x | 2.9x |

### 内存

| Component | v10 (95GB GPU) | verl (FSDP2, 80GB GPU) |
|-----------|---------------|----------------------|
| Base model (bf16) | ~15GB | ~15GB / N_fsdp |
| LoRA grounder | ~0.7GB | ~0.7GB / N_fsdp |
| LoRA actor | ~0.7GB | ~0.7GB / N_fsdp |
| Optimizer states | ~5.6GB | ~5.6GB / N_fsdp |
| Activations (GC) | ~40GB | ~40GB / N_fsdp |
| vLLM engine | N/A | ~15GB |
| **Total** | **~62GB** | **~20GB + 15GB vLLM** |

FSDP2 enables using fewer GPUs with larger per-GPU workload, or more GPUs with larger batch sizes.
