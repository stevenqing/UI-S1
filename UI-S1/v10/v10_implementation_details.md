# v10 Sequential Cooperative LoRA GRPO Trainer — Implementation Details

## 1. Overview

v10 implements a **two-pass cooperative GUI agent** trained with **Group Relative Policy Optimization (GRPO)**. Two LoRA adapters (Grounder and Actor) share one Qwen2.5-VL-7B base model. The Grounder describes which UI element to interact with; the Actor reads the Grounder's description and produces the actual action.

### Architecture Summary

```
[Screenshot + Instruction]
        │
        ▼
  LoRA_grounder (Pass 1)
        │ <action_type>click</action_type>
        │ <target>The blue 'Submit' button at bottom-right...</target>
        ▼
  LoRA_actor (Pass 2, reads action_type + target)
        │ <action>{"action":"click","coordinate":[950,2100]}</action>
        ▼
  [Reward: compare with GT action]
```

### Key Design Decisions

| Decision | Choice | Reason |
|----------|--------|--------|
| Distributed strategy | Manual gradient all-reduce (no DDP/FSDP wrapper) | DDP incompatible with multiple backward() per step + PEFT adapter switching + gradient checkpointing |
| LoRA rank | r=128, alpha=256 | ~380M trainable params per adapter; high capacity needed for VL task |
| GRPO K | 8 | K=2 produced identical rewards too often; K=8 ensures reward diversity |
| Reward type | Continuous (not discrete buckets) | Discrete buckets caused all K samples to get identical rewards → zero advantages |
| KL reference | `disable_adapter_layers()` on same model | Avoids loading a second 7B model; base model serves as reference |
| Gradient checkpointing | Enabled during training, disabled during generation | Required to fit in 95GB GPU; must disable for KV cache during autoregressive generation |

---

## 2. Files

| File | Description |
|------|-------------|
| `v10/train_grpo.py` | Main trainer (~1160 lines): model setup, generation, log prob computation, GRPO train step, training loop, validation |
| `v10/reward.py` | Continuous reward functions for Grounder and Actor |
| `v10/scripts/train_v10_grpo.slurm` | SLURM job script for 4 nodes × 4 GPUs = 16 GPUs |

---

## 3. Model Setup (`_setup_model`)

### 3.1 Base Model Loading

```python
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    args.model_path, torch_dtype=torch.bfloat16, trust_remote_code=True
)
for p in model.parameters():
    p.requires_grad = False  # Freeze base
```

### 3.2 Dual LoRA Adapters

```python
lora_cfg = LoraConfig(
    r=128, lora_alpha=256, lora_dropout=0.05,
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_cfg, adapter_name="grounder")
model.add_adapter("actor", lora_cfg)
```

**Critical**: After adding adapters, `set_adapter()` deactivates the non-selected adapter and sets its `requires_grad=False`. We must force ALL LoRA params back to `requires_grad=True`:

```python
for name, param in model.named_parameters():
    if "lora_" in name:
        param.requires_grad = True
```

This is done in `_set_adapter_safe()` every time we switch adapters.

### 3.3 Gradient Checkpointing

```python
model.enable_input_require_grads()
model.gradient_checkpointing_enable()
```

**Why `enable_input_require_grads()`**: Gradient checkpointing needs `requires_grad=True` on inputs to create checkpoint boundaries. Without this, the first forward pass has no gradient hooks, and checkpointing silently produces zero gradients.

### 3.4 No DDP Wrapper

The model is moved to GPU without any DDP/FSDP wrapping:

```python
self.model = model.to(self.device)
```

**Why not DDP**: PyTorch DDP marks parameter gradients as "ready" after the first `backward()`. With GRPO, we call `backward()` multiple times per step (once per K sample × 2 adapters). DDP raises "variable has been marked as ready" errors on the second backward.

**Why not FSDP**: FSDP shards parameters and gathers them on demand. With PEFT multi-adapter `set_adapter()`, the parameter names/shapes don't change but the active weights do. FSDP's flat parameter buffers don't support this switching, causing "data pointer not allocated" errors.

Instead, we manually call `dist.all_reduce(p.grad, op=AVG)` for every trainable parameter after all backward passes.

---

## 4. Gradient Checkpointing vs. Generation (`_set_grad_checkpointing`)

### The Problem

`gradient_checkpointing_enable()` forces `use_cache=False` inside the model's forward pass. During autoregressive generation, this means no KV cache, causing O(n²) computation and extremely slow generation (~10min per sample instead of ~5s).

### The Solution

Toggle gradient checkpointing on the **actual PreTrainedModel** (not PeftModel) around generation calls:

```python
def _set_grad_checkpointing(self, enable: bool):
    base = self.peft_model.base_model.model  # Qwen2_5_VLForConditionalGeneration
    if enable:
        base.gradient_checkpointing_enable()
    else:
        base.gradient_checkpointing_disable()
```

**Why `self.peft_model.base_model.model`**: PeftModel wraps `LoraModel` which wraps the actual model. Calling `gradient_checkpointing_disable()` on PeftModel or LoraModel doesn't propagate to the underlying `Qwen2VLModel` where the `self.gradient_checkpointing` flag lives. We must call it on the `Qwen2_5_VLForConditionalGeneration` instance directly.

### Usage Pattern

```python
# In _generate_batch():
self._set_grad_checkpointing(False)   # Enable KV cache
self.model.eval()                      # Also needed: GC only activates in training mode
output_ids = self.model.generate(...)
self.model.train()
self._set_grad_checkpointing(True)    # Re-enable for training forward passes
```

**Note**: The stderr warning "`use_cache=True` is incompatible with gradient checkpointing" appears during TRAINING forward passes (where GC is correctly enabled). It fires only once (`warning_once`) and does NOT come from generation — it's expected and harmless.

### Why Not Just Remove Gradient Checkpointing?

Without gradient checkpointing, `_compute_token_log_probs(with_grad=True)` in `train_step` causes CUDA OOM (91.28 GiB used on 95 GiB GPU). Gradient checkpointing is mandatory for training.

---

## 5. Generation Pipeline (`generate_rollouts`)

For each training sample:

### 5.1 Grounder Generation (Batched)

```python
self._set_adapter_safe("grounder")
g_output_ids, g_prompt_len = self._generate_batch(g_inputs, K=8, max_new_tokens=256)
```

All K=8 samples generated in one batched `model.generate()` call (~10s). Input is replicated K times.

### 5.2 Actor Generation (Sequential Loop)

After grounder generation, outputs are parsed into `(action_type, target)` tuples via `parse_grounder_output()`, which extracts structured `<action_type>` and `<target>` tags. The actor receives these as separate fields.

```python
self._set_adapter_safe("actor")
grounder_parsed = [parse_grounder_output(t) for t in grounder_texts]

for k in range(K):
    # Each actor call gets parsed action_type + target from grounder
    action_type, target = grounder_parsed[k]
    a_user = format_actor_text(goal, history, action_type, target)
    a_output_ids, a_prompt_len = self._generate_batch(a_inputs, 1, max_new_tokens=256)
```

Each of the K actor generations gets a different grounder description, so they can't be batched (different prompt content). Sequential, ~3-4s each, ~25-30s total.

### 5.3 Reward Computation

```python
g_r = grounder_reward(actor_texts[k], gt_action, image_w, image_h)
a_r = actor_reward(actor_texts[k], gt_action, image_w, image_h)
```

Both rewards evaluate the **actor's output** against ground truth. The grounder is rewarded based on how well the actor performed (since the grounder's quality directly affects the actor's success).

### 5.4 GRPO Advantage Normalization

```python
g_adv = (g_t - g_t.mean()) / (g_t.std() + eps)  # if std > eps, else zeros
a_adv = (a_t - a_t.mean()) / (a_t.std() + eps)
```

Standard GRPO: mean-subtract and normalize by std within the K-group. If all K rewards are identical (std < eps), advantages are set to zero and that rollout produces no gradient.

### 5.5 Old Log Probabilities

Computed immediately after generation, from the same model weights:

```python
# Grounder old log probs
self._set_adapter_safe("grounder")
for k in range(K):
    tok_lp, mask, _ = self._compute_token_log_probs(
        g_output_ids[k], g_prompt_len, g_fwd_inputs, with_grad=False
    )
    g_old_tok_lps.append(tok_lp.detach())
```

**Important**: `with_grad=False` — these are used as the "old policy" baseline in the PPO ratio computation. They are detached from the computation graph.

### 5.6 Reference Log Probabilities (for KL penalty)

```python
def _compute_ref_log_probs(self, full_ids, prompt_len, inputs_for_fwd):
    self.peft_model.disable_adapter_layers()  # Use base model only
    try:
        tok_lp, mask, _ = self._compute_token_log_probs(
            full_ids, prompt_len, inputs_for_fwd, with_grad=False
        )
    finally:
        self.peft_model.enable_adapter_layers()
        for n, p in self.peft_model.named_parameters():
            if "lora_" in n:
                p.requires_grad = True  # Restore requires_grad
    return tok_lp, mask
```

`disable_adapter_layers()` makes the forward pass use only the base model weights (no LoRA), giving us the reference policy log probs without loading a second model.

**Critical**: After `enable_adapter_layers()`, we must restore `requires_grad=True` on all LoRA params, because PEFT resets it.

---

## 6. Log Probability Computation (`_compute_token_log_probs`)

```python
def _compute_token_log_probs(self, full_ids, prompt_len, inputs_for_fwd, with_grad=False):
    ids = full_ids.unsqueeze(0)            # [1, seq_len]
    attn = torch.ones_like(ids)

    fwd_kwargs = {"input_ids": ids, "attention_mask": attn}
    for k in ("pixel_values", "image_grid_thw"):
        if k in inputs_for_fwd:
            fwd_kwargs[k] = inputs_for_fwd[k]

    ctx = torch.enable_grad() if with_grad else torch.no_grad()
    with ctx:
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            outputs = self.model(**fwd_kwargs)

        logits = outputs.logits
        resp_logits = logits[:, prompt_len - 1 : -1, :]   # Teacher-forced: shifted by 1
        resp_labels = ids[:, prompt_len:]
        log_p = F.log_softmax(resp_logits, dim=-1)
        tok_lp = torch.gather(log_p, -1, resp_labels.unsqueeze(-1)).squeeze(-1)
        mask = (resp_labels != self.pad_id).float()

    return tok_lp.squeeze(0), mask.squeeze(0), mask.sum().item()
```

Key details:
- **Teacher-forced alignment**: `logits[:, prompt_len-1:-1]` aligns predictions with labels at `ids[:, prompt_len:]`
- **Response mask**: Only computes loss on non-padding response tokens
- **`with_grad=True`**: Used in `train_step` for backprop; `False` for old/ref log probs

---

## 7. GRPO Loss Functions

### 7.1 PPO Clipped Policy Loss

```python
def compute_policy_loss(old_log_probs, log_probs, advantages, response_mask, clip_range=0.2):
    ratio = torch.exp(log_probs - old_log_probs)
    clipped_ratio = torch.clamp(ratio, 1-clip_range, 1+clip_range)

    pg_loss1 = -advantages * ratio
    pg_loss2 = -advantages * clipped_ratio
    pg_loss = torch.max(pg_loss1, pg_loss2)   # Pessimistic bound

    valid_tokens = response_mask.sum().clamp(min=1)
    pg_loss = (pg_loss * response_mask).sum() / valid_tokens
    return pg_loss, clip_frac, approx_kl
```

### 7.2 KL Penalty

```python
def compute_kl_penalty(log_probs, ref_log_probs, response_mask):
    log_ratio = log_probs - ref_log_probs
    kl = (torch.exp(log_ratio) - 1 - log_ratio) * response_mask  # Low-variance estimator
    return kl.sum() / valid_tokens
```

Uses the low-variance KL estimator `exp(r) - 1 - r` instead of `r * exp(r)`. This is more numerically stable.

### 7.3 Why Loss Values Are ~0 (Expected Behavior)

In online GRPO with 1 PPO epoch, `old_log_probs` and `new_log_probs` are computed from the **same model weights** (no optimizer step between them). Therefore:

```
ratio = exp(new_lp - old_lp) = exp(0) = 1.0  (exactly)
```

For K=8 with mean-normalized advantages that sum to zero:

```
total_loss = Σ_k (-adv_k × ratio_k) = Σ_k (-adv_k × 1.0) = -Σ_k adv_k = 0
```

**The loss value is mathematically zero, but the gradients are NOT zero**:

```
∂loss/∂θ = Σ_k (-adv_k × ∂log_probs_k/∂θ)
```

Different samples k have different `∂log_probs_k/∂θ`, so the sum is non-zero. This is the standard REINFORCE gradient. The `gnorm > 0` in logs confirms gradients exist.

---

## 8. Train Step (`train_step`)

### 8.1 Loss Normalization

```python
loss_scale = K * args.gradient_accumulation_steps  # = 8 × 2 = 16
loss = (pg_loss + args.kl_coef * kl_loss) / loss_scale
loss.backward()
```

Each `backward()` accumulates `1/(K*accum)` of the total gradient. This makes the gradient magnitude independent of K and grad_accum — equivalent to averaging over all samples in the batch.

### 8.2 Dual Adapter Backward

```python
# Phase 1: Grounder backward
self._set_adapter_safe("grounder")
for data in batch_rollouts:       # grad_accum rollouts
    for k in range(K):            # K=8 samples per rollout
        if abs(adv) < 1e-8:       # Skip zero-advantage samples (no gradient anyway)
            continue
        tok_lp = self._compute_token_log_probs(..., with_grad=True)
        loss = compute_policy_loss(...) / loss_scale
        loss.backward()           # Accumulates gradient on grounder LoRA params

# Phase 2: Actor backward
self._set_adapter_safe("actor")
for data in batch_rollouts:
    for k in range(K):
        ...  # Same as above, accumulates gradient on actor LoRA params
```

**Adapter isolation**: When `set_adapter("grounder")` is active, only grounder LoRA params participate in the forward pass. Actor LoRA params are in the model but disconnected from the computation graph, so `backward()` doesn't touch them. Vice versa for actor. This gives us independent gradient accumulation for each adapter.

### 8.3 Manual Gradient All-Reduce (Deadlock Prevention)

```python
if self.world_size > 1:
    for p in self.peft_model.parameters():
        if p.requires_grad:
            if p.grad is None:
                p.grad = torch.zeros_like(p.data)  # Critical: create zero grad
            dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)
```

**Why create zero grads for `None` grads**: Different ranks may skip different numbers of samples (due to image loading failures or zero advantages). If rank A has `p.grad=None` (no backward touched this param) while rank B has `p.grad=tensor(...)`, rank A skips `all_reduce` while rank B calls it → **deadlock**.

By creating zero tensors for all params with `None` grad, every rank calls `all_reduce` the same number of times. Params that weren't updated on a rank contribute zero to the average.

### 8.4 Fixed-Interval Train Step Boundaries

```python
if (sample_idx + 1) % args.gradient_accumulation_steps == 0:
    metrics = self.train_step(batch_rollouts)
```

**Why fixed interval**: `DistributedSampler` gives each rank the same number of samples. At `sample_idx = 1, 3, 5, ...`, ALL 16 ranks hit the boundary simultaneously and call `train_step`. This ensures `dist.all_reduce` calls match across ranks.

Every rank MUST call `train_step` at every boundary, even if `batch_rollouts` is empty (all samples failed on this rank). The zero-grad mechanism in §8.3 handles this gracefully.

### 8.5 Monitoring Metrics

```python
return {
    "grounder_loss": total_g_loss / max(n_g_seqs, 1),
    "actor_loss": total_a_loss / max(n_a_seqs, 1),
    "grounder_reward": float(np.mean(all_g)),
    "actor_reward": float(np.mean(all_a)),
    "kl": total_kl / n_total,
    "grad_norm": grad_norm,
    "g_nonzero_frac": (n_g_total - n_g_zero_adv) / max(n_g_total, 1),  # % samples with gradient
    "a_nonzero_frac": ...,
    "g_mean_abs_adv": float(np.mean(g_advs_abs)),   # Average advantage magnitude
    "a_mean_abs_adv": ...,
}
```

- **`g_nz/a_nz`**: Fraction of samples with non-zero advantages. Below 50% means the reward function lacks diversity.
- **`g_adv/a_adv`**: Mean absolute advantage. Higher = stronger learning signal.
- **`kl`**: KL divergence from base model. Increasing = model is changing.
- **`gnorm`**: Gradient norm. > 0 confirms learning is happening (even when loss ≈ 0).

---

## 9. Reward Functions (`reward.py`)

### 9.1 Design Principle: Continuous Rewards

The original reward function used **discrete buckets** (0.0, 0.2, 0.3, 0.5, 0.6, 1.0). With K=8, all samples frequently landed in the same bucket → identical rewards → zero GRPO advantages → no learning signal.

**Solution**: All rewards are continuous. Different predicted coordinates produce different reward values, even if they're close.

### 9.2 Coordinate Reward (Continuous)

```python
def _coord_reward_continuous(dist, threshold=0.05):
    max_dist = threshold * 4  # = 0.20 normalized
    if dist >= max_dist:
        return 0.0
    return 1.0 - dist / max_dist  # Linear decay: 1.0 at dist=0, 0.0 at max_dist
```

Normalized Euclidean distance on [0, 1] × [0, 1] image space. `threshold=0.05` means ~54px on a 1080-wide image.

Example outputs for click at (500, 1200):
```
pred_x=490  → dist=0.0093  → reward=0.9537   (very close)
pred_x=510  → dist=0.0093  → reward=0.9537   (symmetric)
pred_x=530  → dist=0.0278  → reward=0.8611   (slightly off)
pred_x=560  → dist=0.0556  → reward=0.7222   (moderate)
pred_x=600  → dist=0.0926  → reward=0.5370   (far)
```

### 9.3 Grounder Reward

```python
def grounder_reward(actor_output, gt_action, image_w, image_h, threshold=0.05):
    gt_type = gt_action.get("action", "")
    if gt_type == "left_click":
        gt_type = "click"  # Normalize action type aliases

    # Non-coordinate actions: use actor reward as proxy
    if gt_type not in ("click", "long_press"):
        return actor_reward(actor_output, gt_action, image_w, image_h)

    # Coordinate actions: continuous distance-based reward
    pred_action = parse_action_from_text(actor_output)
    # Also normalize predicted type
    if pred_action and pred_action.get("action") == "left_click":
        pred_action["action"] = "click"
    dist = coord_distance(pred_coord, gt_coord, image_w, image_h)
    return _coord_reward_continuous(dist, threshold)
```

**Key modifications**:
1. For non-coordinate actions (swipe, type, open, etc.), the original function returned a constant 0.5 for all K samples. Now it returns the **actor reward as a proxy** — the idea being that the grounder's description quality directly affects the actor's success.
2. **`left_click` normalization**: Both GT and predicted `left_click` are normalized to `click` before comparison. The model sometimes outputs `left_click` (desktop-style variant) which is semantically identical to `click`.

### 9.4 Actor Reward

```python
def actor_reward(actor_output, gt_action, image_w, image_h, threshold=0.05):
    gt_type = gt_action.get("action", "")
    if gt_type == "left_click":
        gt_type = "click"
    pred_type = pred_action.get("action", "")
    if pred_type == "left_click":
        pred_type = "click"
        pred_action["action"] = "click"

    # Type mismatch → 0.0
    if pred_type != gt_type:
        return 0.0

    # Coordinate actions: type_match (0.3) + coord_bonus (up to 0.7)
    if gt_type in ("click", "long_press"):
        coord_bonus = 0.7 * _coord_reward_continuous(dist, threshold)
        return 0.3 + coord_bonus

    # Text actions: type_match (0.3) + text_similarity (up to 0.7)
    elif gt_type in ("type", "open", "answer", "key"):
        sim = SequenceMatcher(None, pred_text, gt_text).ratio()
        return 0.3 + 0.7 * sim

    # Swipe: direction cosine similarity
    elif gt_type == "swipe":
        cos_sim = (gt_dx*pred_dx + gt_dy*pred_dy) / (gt_mag * pred_mag)
        return 0.3 + 0.7 * max(0.0, cos_sim)
```

All action types produce continuous rewards with fine granularity:
- **Base**: 0.3 for correct action type
- **Bonus**: Up to 0.7 based on content accuracy (coordinates, text, direction)

### 9.5 Reward Diversity Verification

Before fix (discrete):
```
g_r=[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]  → adv=[0,0,0,0,0,0,0,0]  → no gradient
a_r=[0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3]  → adv=[0,0,0,0,0,0,0,0]  → no gradient
```

After fix (continuous):
```
g_r=[0.932, 0.932, 0.456, 0.843, 0.0, 0.521, 0.927, 0.932]  → diverse advantages!
a_r=[0.953, 0.952, 0.619, 0.890, 0.3, 0.665, 0.949, 0.952]  → diverse advantages!
```

---

## 10. Distributed Training Setup

### 10.1 Hardware

- 4 nodes × 4 NVIDIA GH200 GPUs (95 GiB each) = 16 GPUs
- Interconnect: Slingshot (hsn0), Socket transport (no InfiniBand)

### 10.2 NCCL Configuration

```bash
export NCCL_SOCKET_IFNAME=hsn0
export NCCL_NET=Socket
export NCCL_IB_DISABLE=1
export NCCL_P2P_LEVEL=LOC       # Local P2P only
export NCCL_CROSS_NIC=1
export NCCL_TIMEOUT=10800       # 3 hours
```

### 10.3 Launch

```bash
srun --ntasks-per-node=1 bash -c '
    torchrun --nproc_per_node=4 --nnodes=$SLURM_NNODES \
        --node_rank=$SLURM_NODEID --master_addr=$MASTER_ADDR \
        v10/train_grpo.py [args...]
'
```

`srun --ntasks-per-node=1` launches one `torchrun` per node, which spawns 4 GPU workers.

### 10.4 Effective Batch Size

```
Per rank:  grad_accum=2 rollouts × K=8 samples = 16 samples
16 ranks:  2 × 16 = 32 rollouts × 8 = 256 samples per optimizer step
```

All-reduce uses `op=AVG`, so gradients are averaged across 16 ranks. Combined with `loss_scale = K × grad_accum = 16`, the effective gradient is normalized over the full 256-sample batch.

---

## 11. Optimizer

```python
self.optimizer = torch.optim.AdamW([
    {"params": grounder_params, "lr": 1e-5},
    {"params": actor_params, "lr": 5e-6},
], weight_decay=0.01)
```

- **Separate learning rates**: Grounder has 2× higher LR than Actor. Grounder needs stronger signal because its reward is indirect (via actor performance).
- **Gradient clipping**: `max_grad_norm=1.0` applied to all trainable params.

---

## 12. Training Loop

```python
for epoch in range(num_epochs):
    sampler.set_epoch(epoch)  # Reshuffle data

    for sample_idx, sample in enumerate(train_loader):
        rollout = self.generate_rollouts(sample)  # ~50s per rollout
        batch_rollouts.append(rollout)

        if (sample_idx + 1) % grad_accum == 0:    # Every 2 samples
            metrics = self.train_step(batch_rollouts)  # ~80s
            batch_rollouts = []

    # End of epoch
    save_checkpoint(f"epoch-{epoch}")
    validate(epoch)  # rank 0 only
```

### 12.1 Timing Breakdown per Optimizer Step

| Phase | Time | Details |
|-------|------|---------|
| Rollout 1: Grounder gen | ~10s | Batched K=8 generation |
| Rollout 1: Actor gen | ~30s | Sequential K=8 generation |
| Rollout 1: Old/Ref log probs | ~10s | 2×K forward passes (no grad) |
| Rollout 2: Same | ~50s | |
| Train step: Forward+Backward | ~60s | Up to 2×K=16 forward+backward per adapter |
| Train step: All-reduce | ~10s | One all_reduce per trainable param |
| **Total per step** | **~2.2 min** | |

### 12.2 Epoch/Total Time

```
Steps per epoch: 6482 / 16 GPUs / 2 grad_accum ≈ 202 steps
Time per epoch:  202 × 2.2 min ≈ 7.4 hours
4 epochs:        ~30 hours (24h wall time fits ~3 epochs)
```

### 12.3 Checkpointing

```python
# Every 100 steps:
self.save_checkpoint(f"epoch-{epoch}_step-{self.global_step}")
# End of each epoch:
self.save_checkpoint(f"epoch-{epoch}")
```

Each checkpoint saves both LoRA adapters separately via `peft_model.save_pretrained(selected_adapters=[name])`.

---

## 13. Validation (`validate`)

Runs on rank 0 only, at the end of each epoch. Uses **greedy decoding** (no sampling):

```python
g_out = self.model.generate(..., do_sample=False)  # Greedy grounder
a_out = self.model.generate(..., do_sample=False)  # Greedy actor
```

Reports:
- `val/grounder_reward`: Mean grounder reward (continuous)
- `val/actor_reward`: Mean actor reward (continuous)
- `val/actor_exact`: Fraction of samples with actor_reward == 1.0
- 3 example outputs (grounder text + actor text + rewards)

---

## 14. Bugs Encountered and Fixed

### Bug 1: Actor LoRA params=0 (Early Version)

**Symptom**: `actor_params=0` in optimizer — second adapter had no trainable params.

**Root cause**: `set_adapter("grounder")` sets `requires_grad=False` on actor LoRA params. The optimizer was initialized after `set_adapter`.

**Fix**: Force `requires_grad=True` on ALL `lora_` params after model setup, and again in `_set_adapter_safe()` on every adapter switch.

### Bug 2: CUDA OOM Without Gradient Checkpointing

**Symptom**: 91.28 GiB used on 95 GiB GPU during `_compute_token_log_probs(with_grad=True)`.

**Fix**: Gradient checkpointing is mandatory. Cannot be removed.

### Bug 3: Slow Generation with Gradient Checkpointing

**Symptom**: Generation took ~10min per sample (expected ~5s). `use_cache=True is incompatible with gradient checkpointing` warning.

**Root cause**: PeftModel's `gradient_checkpointing_disable()` doesn't propagate to the underlying Qwen2_5_VLModel.

**Fix**: Call `self.peft_model.base_model.model.gradient_checkpointing_disable()` directly on the PreTrainedModel. See §4.

### Bug 4: Distributed Deadlock in train_step

**Symptom**: Training hangs after a few steps. Some ranks stuck in `dist.all_reduce` while others have moved on.

**Root cause**: When a rank has no valid rollouts in `batch_rollouts` (image load failures), no `backward()` is called, so `p.grad is None`. That rank skips `dist.all_reduce` while other ranks call it → deadlock.

**Fix**: Create zero tensors for all `None` grads before all_reduce:
```python
if p.grad is None:
    p.grad = torch.zeros_like(p.data)
dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)
```

### Bug 5: Missing `g_text` in validate()

**Symptom**: `NameError: name 'g_text' is not defined` during validation.

**Fix**: Added the decode step after grounder generation:
```python
g_prompt_len = g_inputs["input_ids"].shape[1]
g_text = self.processor.tokenizer.decode(g_out[0, g_prompt_len:], skip_special_tokens=True)
```

### Bug 6: Zero Loss / Zero Advantages with K=2

**Symptom**: `g_loss=0.0000 a_loss=0.0000` consistently. `g_nz=0% a_nz=0%`.

**Root cause**: K=2 frequently produced identical rewards for both samples (same discrete bucket) → zero advantages → no gradient. Even with different rewards, loss = 0 due to ratio=1 and symmetric advantages (see §7.3).

**Fix**: Increased K to 8. See Bug 7 for the deeper issue.

### Bug 7: All K=8 Rewards Identical (Discrete Reward Buckets)

**Symptom**: Even with K=8: `g_r=[0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5]`, `g_nz=0%`.

**Root cause**: The reward function returned only a few discrete values {0.0, 0.2, 0.3, 0.5, 0.6, 1.0}. For non-coordinate actions, grounder always returned 0.5. For coordinate actions, the distance thresholds created wide buckets where all K samples landed in the same one.

**Fix**: Rewrote reward functions to be continuous. See §9.

---

## 15. Hyperparameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| Base model | Qwen2.5-VL-7B-Instruct | ~15GB in bf16 |
| LoRA r | 128 | High capacity |
| LoRA alpha | 256 | alpha/r = 2 |
| LoRA dropout | 0.05 | |
| LoRA targets | q,k,v,o,gate,up,down_proj | All attention + MLP |
| K (num_samples) | 8 | Group size for GRPO |
| Temperature | 1.0 | For exploration |
| Top-p | 0.95 | |
| Clip range | 0.2 | PPO clipping |
| KL coef | 0.001 | Low: allow exploration |
| Max grounder tokens | 256 | |
| Max actor tokens | 256 | |
| Grounder LR | 1e-5 | |
| Actor LR | 5e-6 | |
| Weight decay | 0.01 | |
| Max grad norm | 1.0 | |
| Gradient accumulation | 2 | 2 rollouts per rank per step |
| Epochs | 4 | |
| Save steps | 100 | |
| Effective batch size | 256 | 16 ranks × 2 accum × 8 K |

---

## 16. Training Progress (Current Run, Job 3997905)

```
E0 S5:  g_r=0.426 a_r=0.380 kl=0.0015 gnorm=0.66 g_nz=100% a_nz=100%
E0 S10: g_r=0.193 a_r=0.266 kl=0.0025 gnorm=0.78 g_nz=50%  a_nz=50%
E0 S15: g_r=0.272 a_r=0.284 kl=0.0130 gnorm=0.91 g_nz=50%  a_nz=50%
E0 S20: g_r=0.334 a_r=0.403 kl=0.0438 gnorm=0.59 g_nz=100% a_nz=100%
E0 S25: g_r=0.640 a_r=0.654 kl=0.0156 gnorm=0.87 g_nz=100% a_nz=100%
E0 S30: g_r=0.041 a_r=0.216 kl=0.0093 gnorm=0.67 g_nz=50%  a_nz=100%
```

Key observations:
- **g_nz/a_nz = 50-100%**: Most samples produce useful gradients (vs. 0% with discrete rewards)
- **KL increasing**: Model is diverging from base → learning is happening
- **gnorm > 0**: Non-zero gradients despite loss ≈ 0
- **Rewards fluctuate**: Normal with small per-rank batch (2 rollouts); actual batch is 32 across all ranks

## 17. Training Progress Update (Job 3999242, with val_steps=50)

### Training Metrics S1–S50

```
E0 S5:  g_r=0.426 a_r=0.380 kl=0.0015 gnorm=0.66  g_nz=100% a_nz=100%
E0 S10: g_r=0.193 a_r=0.266 kl=0.0025 gnorm=0.78  g_nz=50%  a_nz=50%
E0 S15: g_r=0.272 a_r=0.284 kl=0.0130 gnorm=0.91  g_nz=50%  a_nz=50%
E0 S20: g_r=0.334 a_r=0.403 kl=0.0438 gnorm=0.59  g_nz=100% a_nz=100%
E0 S25: g_r=0.640 a_r=0.654 kl=0.0156 gnorm=0.87  g_nz=100% a_nz=100%
E0 S30: g_r=0.041 a_r=0.216 kl=0.0093 gnorm=0.67  g_nz=50%  a_nz=100%
E0 S35: g_r=0.298 a_r=0.432 kl=0.3806 gnorm=2.02  g_nz=50%  a_nz=50%
E0 S40: g_r=0.325 a_r=0.394 kl=0.9988 gnorm=65.46 g_nz=100% a_nz=100%  ← gradient explosion
E0 S45: g_r=0.000 a_r=0.131 kl=0.0011 gnorm=0.96  g_nz=0%   a_nz=50%
E0 S50: g_r=0.666 a_r=0.672 kl=0.0076 gnorm=0.44  g_nz=100% a_nz=100%
```

Notable: Gradient explosion at S40 (gnorm=65.46, KL=0.999) followed by recovery at S45. This suggests kl_coef=0.001 is too small to constrain the model.

### First Validation Results (epoch-0_step-50)

```
val/grounder_reward: 0.4493
val/actor_reward:    0.5184
val/actor_exact:     0.1296 (12.96%)
val/n_samples:       54
```

Per-sample results saved to: `v10/output/v10_grpo_ddp/val_results/epoch-0_step-50.jsonl`

---

## 18. Lessons Learned from S50 Validation Analysis

### 18.1 Reward Hacking: Grounder Outputting Raw Coordinates

**Problem**: The grounder learned to output Qwen2.5-VL's built-in grounding format with raw `<points>` XML tags containing pixel coordinates, rather than natural language UI element descriptions.

**Example**:
```
grounding: <points x1="538" y1="215" alt="search bar">search bar</points>
```

**Why this happens**: The grounder reward is computed by comparing to the ground-truth coordinate. The grounder discovered it can "cheat" by including coordinates directly in its output — this doesn't help the actor (which should use the description to locate elements), but it gets high reward because the continuous coordinate reward only looks at how close the predicted coordinate is.

**Impact**: This defeats the purpose of the two-pass architecture. The grounder should produce semantic UI descriptions ("the blue Submit button at bottom-right") that help the actor generalize, not just parrot coordinates.

**Potential fixes**:
1. Add a format penalty in `grounder_reward()` that penalizes outputs containing `<points>`, coordinate numbers, or raw pixel values
2. Use a text-only evaluation for grounder: reward based on whether the description is semantically useful (e.g., contains element type, color, position words) rather than coordinate proximity
3. Mask coordinate-containing tokens during grounder generation
4. Increase KL penalty to prevent rapid divergence from base model behavior

### 18.2 Grounder Acting as Action Planner

**Problem**: Instead of describing UI elements, the grounder often outputs action planning text like "Action: Click on the search bar at the top of the screen to start searching".

**Example**:
```
grounding: Action: Click on the search bar at the top of the screen to start searching for India Gate Basmati Rice.
```

**Why this happens**: The prompt asks the grounder to "describe the target UI element for the next action." The model interprets "next action" as an invitation to plan the action itself, not just describe the visual element. This is reinforced because the actor can still extract useful information from the action description (it mentions "search bar at the top"), so the grounder still gets reward.

**Impact**: Moderate — the actor can sometimes still extract the right element from an action-style description, but it's less robust than a pure UI description. The grounder is doing more work than intended and may confuse the actor.

**Potential fixes**:
1. Revise the grounder prompt to be more explicit: "Describe ONLY the visual appearance and location of the UI element. Do NOT describe any actions."
2. Add a penalty for outputs containing action verbs ("click", "tap", "type", "swipe", "scroll")

### 18.3 Actor Sometimes Ignores Grounder

**Problem**: In some samples, the actor produces actions that don't correspond to what the grounder described — clicking on completely different coordinates or performing wrong action types.

**Example**:
```
grounding: "The search bar at the top of the screen"
actor: <action>{"action": "click", "coordinate": [672, 1608]}</action>  ← bottom of screen
```

**Why this happens**:
- Early training (S50) — the actor hasn't learned to attend to grounder output yet
- The actor may be relying on its own visual understanding of the screenshot rather than the grounder text
- With K=8 samples, some samples will naturally be worse

**Impact**: This should improve with more training as the actor learns that attending to the grounder correlates with higher reward. Monitor in future validations.

### 18.4 Poor Swipe Recognition

**Problem**: For swipe ground-truth actions, the model tends to predict click actions instead.

**Examples from validation**: Swipe actions received 0.0 reward because the model predicted clicks.

**Why this happens**: Swipe actions are less common in training data. The actor needs to learn the relationship between scroll-related grounder descriptions and swipe outputs.

**Potential fixes**:
1. Oversample swipe examples in training data
2. Increase swipe reward bonus for correct direction prediction
3. Add swipe-specific examples to the prompt (few-shot)

### 18.5 KL Explosion Risk

**Problem**: At S40, KL jumped from 0.01 to 0.99 with gnorm=65.46 (gradient explosion). Although the model recovered by S45, this indicates instability.

**Root cause**: `kl_coef=0.001` is too small. With continuous rewards providing much stronger gradient signal than discrete rewards, the KL penalty is insufficient to keep the model close to the reference.

**Recommended fix**: Increase `kl_coef` to 0.01–0.05 in next run. This will slow learning but prevent explosive divergence.

### 18.6 Summary of Prioritized Fixes for Next Run

| Priority | Fix | Expected Impact |
|----------|-----|-----------------|
| P0 | Increase kl_coef to 0.01–0.05 | Prevent KL explosion and gradient spikes |
| P1 | Add grounder format penalty (penalize `<points>` tags, raw coordinates) | Stop coordinate reward hacking |
| P1 | Revise grounder prompt to explicitly forbid action descriptions | Better role separation |
| P2 | Oversample swipe examples or add swipe few-shot | Improve action type diversity |
| P3 | Monitor actor-grounder alignment across validations | Track whether actor learns to use grounder |

---

## 19. S100 Validation Results & Updated Analysis (Job 3999242)

### 19.1 S100 vs S50 Comparison

| Metric | S50 | S100 | Delta |
|--------|-----|------|-------|
| val/grounder_reward | 0.4493 | 0.5636 | **+0.1143** (+25.5%) |
| val/actor_reward | 0.5184 | 0.6122 | **+0.0938** (+18.1%) |
| val/actor_exact | 12.96% | 22.22% | **+9.26pp** (nearly doubled) |
| `<points>` reward hack | 28/54 (51.9%) | 0/54 (0.0%) | **Eliminated** |
| Action planning style | 16/54 (29.6%) | 44/54 (81.5%) | Increased |
| UI description style | 10/54 (18.5%) | 10/54 (18.5%) | Unchanged |

**Key observations**:
1. **Learning is working**: All reward metrics improved substantially. Actor exact match nearly doubled.
2. **`<points>` hack self-corrected**: By S100, the grounder stopped using raw coordinate tags entirely (0% vs 51.9% at S50). This may be because the KL penalty eventually discouraged divergence from base model format.
3. **Action planning dominates**: 81.5% of grounder outputs now start with "Action: ..." instead of pure UI descriptions. While not ideal architecturally, the actor can still extract useful information from these, and rewards are improving.
4. **Action type diversity improved**: S100 shows swipe (6), type (4), terminate (9), key (1), wait (1) alongside click (33). The model is learning to predict diverse action types.

### 19.2 Training Stability (S50–S100)

```
E0 S55:  kl=0.1216  gnorm=0.61
E0 S60:  kl=1.1871  gnorm=0.45
E0 S65:  kl=1.2905  gnorm=0.61
E0 S70:  kl=0.7956  gnorm=0.63
E0 S75:  kl=0.3817  gnorm=2.65
E0 S80:  kl=9.8338  gnorm=0.71  ← 2nd KL explosion (10x worse than S40)
E0 S85:  kl=0.0000  gnorm=0.54  ← model collapse (all zeros)
E0 S90:  kl=1.1554  gnorm=0.57  ← recovered
E0 S95:  kl=0.0000  gnorm=0.59  ← collapsed again
E0 S100: kl=0.1410  gnorm=0.71  ← recovered
```

**Critical**: S80 had KL=9.83 — much worse than S40's KL=0.99. The pattern of KL explosion → collapse (kl=0.000, g_nz=0%) → recovery repeats. This oscillation means the model is swinging between modes: (1) diverging aggressively from base, (2) snapping back when penalty overwhelms. `kl_coef=0.001` is definitively too low.

### 19.3 Job Crash

Job crashed at 10:31 with SIGABRT on rank 9 (NCCL timeout), ~9 minutes after S100 validation. This is the second NCCL crash (first was job 3939027). The pattern: crash occurs during train_step all_reduce after extended computation (validation takes ~4 min, then the next few rollouts + backward passes may cause timeout).

**Potential fixes**:
- Increase NCCL timeout: `export NCCL_TIMEOUT=7200` (currently default ~1800s)
- Add `dist.barrier()` after validation to resync all ranks before training resumes (already have this, but may need additional sync)
- Use `NCCL_ASYNC_ERROR_HANDLING=1` for better error recovery

### 19.4 Updated Priority List for Next Run

| Priority | Fix | Status |
|----------|-----|--------|
| P0 | Increase kl_coef to 0.02–0.05 | **CRITICAL** — KL hit 9.83 at S80 |
| P0 | Increase NCCL timeout to prevent crash | Two crashes in two runs |
| P1 | Revise grounder prompt to forbid action descriptions | 81.5% action-planning style |
| ~~P1~~ | ~~Penalize `<points>` tags~~ | ~~Self-corrected by S100~~ |
| P2 | Add gradient clipping (max_norm=1.0) | gnorm spikes correlate with KL explosion |
| P3 | Oversample swipe examples | Swipe recognition improving but still weak |
