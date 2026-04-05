# Phase 0: Inference-Time Attention Steering Results

## 1. Motivation

The Selective MoE experiment (Exp 3) showed that giving the model separate routing channels via MoE did not improve action accuracy — in fact it caused function-diversity collapse. Before investing in more training-time interventions, we test a cleaner causal question:

> **If we directly steer the model's attention toward binding-relevant image tokens at inference time, does action accuracy improve?**

This is a zero-parameter, zero-training intervention on the already-trained SFT v2 model (GUI-360 full SFT). If attention routing is the bottleneck, steering should help. If the fragmentation is deeper (e.g., orthogonal subspaces that can't be bridged by routing alone), steering will fail.

### Hypotheses

- **H1 (Binding Boost)**: Boosting attention scores at binding-relevant image tokens in layers L19-L27 improves grounding accuracy by making the model "look harder" at task-relevant visual regions.
- **H2 (Key Amplify)**: Amplifying key vectors at binding-relevant positions increases their influence on downstream attention patterns, improving binding→action information flow.
- **H_null (Random Boost)**: Boosting attention at random image tokens (same magnitude as H1) should NOT improve accuracy. This controls for the possibility that any attention perturbation helps.

## 2. Method

### 2.1 Model

- **Base**: `gui360_full_sft_v2` (Qwen2.5-VL-7B-Instruct, full SFT on GUI-360 train set)
- **Attention implementation**: `eager` (required for attention mask hooks; FlashAttention incompatible)
- **Precision**: bfloat16

### 2.2 Binding Token Identification (Two-Pass)

For each sample, we perform a two-pass approach:

**Pass 1** — Extract hidden states at layer 24 (the binding layer identified in probing analysis):
1. Run forward pass with `output_hidden_states=True`
2. Extract hidden states at layer 24 for all image tokens (identified by `IMAGE_PAD_ID=151655`)
3. Compute task text centroid: mean of hidden states at instruction text token positions
4. Compute cosine similarity between each image token's hidden state and the task centroid
5. Select top 20% (`top_p=0.2`) of image tokens as "binding-relevant"

**Pass 2** — Generate with attention hooks installed at layers L19-L27.

### 2.3 Intervention Modes

| Mode | Mechanism | Parameters |
|---|---|---|
| **Baseline** | No intervention, standard generation | — |
| **Binding Boost (I1)** | Add α to attention mask at binding token key positions (all heads, all query positions) | α=2.0, L19-L27 |
| **Key Amplify (I2)** | Multiply k_proj output at binding positions by (1+β) during prefill | β=0.5, L19-L27 |
| **Random Boost (Control)** | Same as Binding Boost but with randomly selected image tokens (same count) | α=2.0, L19-L27 |

### 2.4 Evaluation Protocol

- **Test set**: 1000 stratified samples from GUI-360 test set (26,284 total)
- **Same 1000 samples** across all 4 conditions (deterministic first-1000 from parquet)
- **Metrics**: Full accuracy (function + args exact match), Function accuracy, BBox accuracy (IoU ≥ 0.5 for click)
- **Generation**: greedy decoding, max 512 tokens

### 2.5 Sample Distribution Verification

The 1000-sample subset matches the full test set distribution:

| Function | 1000-sample | Full 26K | Match? |
|---|---|---|---|
| click | 62.8% | 64.7% | ✓ |
| type | 15.3% | 13.2% | ✓ |
| (unnamed/finish) | 14.2% | 12.3% | ✓ |
| Other | 7.7% | 9.8% | ✓ |

## 3. Results

### 3.1 Overall Accuracy

| Condition | Full Acc | Func Acc | BBox Acc | Δ Full |
|---|---|---|---|---|
| **Baseline** | **18.60%** | **44.80%** | 45.89% | — |
| Binding Boost (α=2.0) | 18.30% | 43.60% | **46.71%** | **-0.30** |
| Key Amplify (β=0.5) | 17.30% | 45.60% | 36.91% | **-1.30** |
| Random Boost (α=2.0) | 18.10% | 42.60% | 45.79% | **-0.50** |

**All interventions fail to improve over baseline.**

### 3.2 Per-Function Breakdown

| Function | N | Baseline | Binding Boost | Key Amplify | Random Boost |
|---|---|---|---|---|---|
| click | 628 | 168 (26.8%) | 162 (25.8%) | 157 (25.0%) | 162 (25.8%) |
| type | 153 | 12 (7.8%) | 15 (9.8%) | 13 (8.5%) | 13 (8.5%) |
| (unnamed) | 142 | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) |
| select_text | 19 | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) |
| drag | 15 | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) |
| wheel_mouse_input | 14 | 5 (35.7%) | 5 (35.7%) | 2 (14.3%) | 5 (35.7%) |
| summary | 9 | 1 (11.1%) | 1 (11.1%) | 1 (11.1%) | 1 (11.1%) |
| select_table_range | 9 | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) |

### 3.3 Binding Token Statistics

- Average binding tokens selected per sample: **190.9 / 956.4** image tokens (20.0%, matching `top_p=0.2`)
- Binding computation layer: L24
- Hook layers: L19-L27 (9 layers)

## 4. Direct Findings

### 4.1 Per-Condition Analysis

**Finding 1: Binding boost does not help action prediction.**
Despite steering attention toward binding-relevant tokens (cosine similarity with task text centroid), full accuracy drops by 0.3pp. BBox accuracy improves marginally (+0.82pp), suggesting the model does attend slightly better to correct spatial locations, but this doesn't translate to better action selection.

**Finding 2: Key amplification actively hurts grounding.**
Key amplify shows the largest degradation in BBox accuracy (45.89% → 36.91%, -8.98pp) while function accuracy slightly improves (44.80% → 45.60%). Amplifying binding-relevant keys disrupts the model's spatial reasoning — the information at these positions may already be used effectively for binding, and amplification introduces noise.

**Finding 3: Random boost performs similarly to binding boost.**
Random boost (-0.50pp) is within noise of binding boost (-0.30pp). The specific selection of binding-relevant tokens via cosine similarity provides no advantage over random selection. Either the token identification is not meaningfully better than random, or attention steering of any kind doesn't help.

**Finding 4: The unnamed/finish function type remains at 0% across all conditions.**
Consistent with MoE results — the model fundamentally cannot predict this function type in this eval protocol.

### 4.2 Four-Level Exclusion Table

Phase 0 completes a systematic exclusion when combined with prior experiments:

| Intervention Level | Experiment | Result | What It Rules Out |
|---|---|---|---|
| Representation | L_bind auxiliary | Probe C +0.47, action Δ=0 | Improving binding representation is not enough |
| Attention routing | Phase 0 steering | Δ=-0.3pp | Pointing attention at the right tokens is not enough |
| Parameter space | Selective MoE | click +5pp but diversity collapse | More parameters is not enough |
| **Symbolic** | **Multi-agent F4** | **+4.96pp** | **Only text-level bridging works** |

## 5. Stepping Back: What Phase 0 Really Means

Phase 0's results are more profound than they first appear. They rule out not just "attention routing is the bottleneck" — they rule out the entire paradigm of **"bridging within fixed representations."**

L_bind changed representation. Phase 0 changed routing. MoE changed parameters. All failed. The only success — multi-agent F4 — did something fundamentally different: it **bypassed the model's internal computation pathway entirely**.

This means the problem is not in any single component. It's in **how the entire computational pathway was formed**.

### 5.1 The Real Research Question

> **What training mechanism causes orthogonal subspaces to form? If we understand this mechanism, can we design a training mechanism to prevent or reverse fragmentation?**

RL is not simply "a better method." It is a **training paradigm with a different representational inductive bias**. The question is: what property of RL leads to different representational geometry?

### 5.2 Teacher Forcing Is the Root Cause of Fragmentation

SFT uses teacher forcing: when predicting token t, the input is GT tokens 0..t-1.

```
Predicting "click":  input = GT context → doesn't need binding
                     (can guess "click" from instruction alone)
Predicting "324":    input = GT "click(" → knows to output coordinates, but from where?

Key: model CAN predict "324" without going through binding:
  → spatial priors ("elements near screen center are often clicked")
  → lexical cues in instruction ("settings" → usually top-right)
  → these shortcuts live entirely within the action subspace
  → loss can still be minimized (imperfectly)
```

**Teacher forcing is shortcut-permissive:** it allows the model to minimize loss using subspace-local shortcuts, without building cross-subspace causal chains. The error decomposition is direct evidence — far miss 48.6% >> near miss 7.9% means the model picks a roughly plausible position rather than truly grounding to the target element.

More precisely: SFT's loss decomposes into **per-token independent local losses**. Each local loss can be partially minimized by action-subspace-internal shortcuts. The model has no incentive to develop cross-subspace pathways because shortcuts are easier to learn.

### 5.3 Autoregressive Generation Breaks Shortcuts

RL's rollout is autoregressive: the model uses its own prediction as the next input.

```
Model predicts "click" → uses own "click" to predict "(" → predicts "324" → "," → "516" → ")"

If "324, 516" is a shortcut prediction (not based on binding):
  sometimes correct, sometimes wrong.
When reward = 0 (wrong): entire trajectory penalized.
When reward = 1 (correct): entire trajectory reinforced.

Long-term consequence:
  shortcut paths have ~35% reward rate (many far misses)
  binding paths have ~60% reward rate (when Agent V correct, F4 = 58%)
  → RL selects binding paths because they have higher reward rate
```

**RL is shortcut-resistant:** outcome reward cannot be maximized by subspace-local shortcuts. The model must develop cross-subspace computation pathways to consistently achieve high reward.

### 5.4 Why Phase 0 Failed but RL Can Succeed

Phase 0 modified attention on SFT v2's **fixed forward path** — the path formed by teacher forcing, shaped by shortcuts. Even after steering attention to the correct image tokens, the value vectors pass through subsequent MLP, LayerNorm, and next-layer attention, where binding information gets projected out again. The orthogonality is maintained throughout the entire forward pass because that's how the pathway was trained.

RL doesn't modify a fixed path — it **explores new paths entirely**. Each sampled token leads to different hidden states, different attention patterns, different subsequent computation. Some paths happen to traverse regions where binding information is useful. Outcome reward selectively reinforces these paths.

The difference is not in attention routing. It's in the **entire generation trajectory**.

## 6. Theoretical Predictions

If "teacher forcing → shortcuts → fragmentation" is the correct causal chain:

### P1: Fragmentation Degree Should Correlate with Teacher Forcing Degree

- Full teacher forcing (SFT) → strongest fragmentation
- Scheduled sampling (mix GT and predicted tokens) → intermediate fragmentation
- Full autoregressive (RL rollout) → weakest fragmentation
- **Measurable**: Probe C gap across three conditions

### P2: RL Should Change Representational Geometry, Not Just Attention Patterns

- Phase 0 only changed attention → failed
- If RL succeeds → Probe C gap should go from negative to positive
- If RL succeeds but Probe C doesn't change → our theory is wrong, RL's success comes from another mechanism

### P3: RL's Gains Should Concentrate on Shortcut-Failure Samples

- Far miss samples = shortcut failure → RL should dramatically reduce far misses
- Near miss samples = already close to target → RL improvement smaller
- If RL uniformly improves all error types → shortcut theory weakened

### P4: RL Exploration Efficiency Should Be Inversely Proportional to Binding-Action Gap

- High binding quality (Probe A AUC = 0.86) → exploration easily discovers binding pathway
- Low binding quality → exploration can't find signal → RL also fails
- **Testable**: compare RL gains on subsets with different binding quality

## 7. Experimental Design: Minimal Experiments to Test the Theory

### 7.1 Exp 1: Scheduled Sampling on SFT v2 (Tests P1) — HIGHEST PRIORITY

**Does not require RL infrastructure.** Mix autoregressive steps into SFT training:

```
For each training sample:
  with probability p_auto:
    first few action tokens: teacher forcing (GT)
    remaining action tokens: autoregressive (model's own predictions)
  with probability (1 - p_auto):
    full teacher forcing (standard SFT)

p_auto = 0.0 → standard SFT (existing baseline)
p_auto = 0.3 → light scheduled sampling
p_auto = 0.7 → heavy scheduled sampling
```

**Prediction**: higher p_auto → higher Probe C gap → lower far miss ratio → higher full accuracy.

**This is the most informative experiment** because it directly tests the "teacher forcing causes fragmentation" causal hypothesis with minimal infrastructure — only requires modifying the label masking strategy in the SFT data pipeline.

### 7.2 Exp 2: GRPO with Probe C Monitoring (Tests P2)

```
Base: SFT v2
Method: GRPO, K=8 rollouts per sample
Reward: R = 1(click within GT bbox)
Training: GUI-360 train set
Analysis: Probe C every 100 steps

Not about "can RL improve accuracy" —
about "does RL change representational geometry"

Outcomes:
  Probe C goes negative → positive: RL is bridging fragmentation ✓
  Probe C unchanged, accuracy up: RL uses another mechanism (theory wrong)
  Probe C unchanged, accuracy unchanged: exploration failed to find cross-subspace paths
```

### 7.3 Exp 3: GRPO with Distance Reward (Tests P3+P4)

```
R_action = 1(click correct)
R_distance = max(0, 1 - ||predicted - GT|| / max_dist)  → continuous distance reward
R = R_action + 0.3 * R_distance
```

R_distance's theoretical role is not "give more reward signal" (that's engineering thinking). It **shrinks the exploration space**: from "anywhere on screen" to "near the target", making binding subspace signals discoverable.

**Test for P3**: Analyze error decomposition before/after RL. If far misses dramatically decrease but near misses don't → shortcut-resistance theory confirmed.

### 7.4 Expected Outcomes Summary

| Condition | Cross-Subspace Coupling | Expected |
|---|---|---|
| SFT v2 (p_auto=0) | None (orthogonal gradients) | Baseline |
| Scheduled (p_auto=0.3) | Weak (partial autoregressive) | Slight improvement |
| Scheduled (p_auto=0.7) | Medium (mostly autoregressive) | Moderate improvement |
| **GRPO vanilla (Exp 2)** | Outcome-level (sparse binary) | Possible decline (exploration too noisy) |
| **GRPO + distance (Exp 3)** | Outcome-level (dense spatial) | +1-3pp |
| F4 multi-agent | Full bypass (text-level) | +4.96pp (existing) |

## 8. Timeline

| Priority | Experiment | Duration | Tests |
|---|---|---|---|
| **P0** | Exp 1: Scheduled sampling (p_auto = 0, 0.3, 0.7) | 2-3 days | P1: Is teacher forcing the cause? |
| **P1** | Exp 2: GRPO + Probe C monitoring | 3-4 days | P2: Does RL change representational geometry? |
| **P1** | Exp 3: GRPO + distance reward | 3-4 days | P3+P4: Does dense reward accelerate bridging? |
| **P2** | All checkpoints: Probe C + error decomposition | 1 day | Consolidate mechanistic evidence |

**Exp 1 (scheduled sampling) is the most critical experiment.** If it works → direct proof that teacher forcing is the root cause of fragmentation, and it doesn't need RL infrastructure at all. If it doesn't work → teacher forcing isn't the only cause, outcome-level reward from RL is necessary.

## 9. Paper Framing

```
Title direction: "Teacher Forcing Fragments, Outcome Rewards Bridge:
                  Training Objectives as Representational Geometry"

Core claim:
  The structure of the training objective determines the geometry of learned representations.
  Per-token supervised objectives permit subspace-local shortcuts → fragmentation.
  Outcome-level objectives penalize shortcuts → cross-subspace integration.

  This is not a "RL > SFT" story.
  It is a "the decomposability of the training signal determines
  the degree of representational fragmentation" story.

Evidence chain:
  1. SFT creates representational fragmentation (Stage I-II diagnostics)
  2. Fragmentation cannot be resolved by:
     - improving binding representation (L_bind: Δ≈0)
     - modifying attention routing (Phase 0: Δ≈0)
     - adding parameter capacity (MoE: -6pp)
  3. Multi-agent text-level bridging works (+4.96pp) but 3× inference cost
  4. Scheduled sampling reduces fragmentation (Exp 1, if confirmed)
  5. RL with grounding-shaped reward bridges within single model (Exp 2-3, if confirmed)
  6. Final recipe: SFT → diagnose fragmentation → RL with binding-aware reward → bridge
```

## Appendix A. Experimental Details (Phase 0)

| Parameter | Value |
|---|---|
| SLURM jobs | 3527041 (baseline), 3527042 (binding_boost), 3527043 (key_amplify), 3527044 (random_boost) |
| Nodes | nid011099, nid010442 |
| GPU | 1× GH200 per job |
| Runtime | ~82 min per job (4.4s/sample avg) |
| Date | 2026-03-31 |
| Script | `evaluation/attention_steering_eval.py` |
| SLURM | `scripts/exp3/phase0_steering.slurm` |
| Results | `scripts/exp3/results/phase0_*_20260331_144240/` |

---

## 10. Phase 1: Implementation and Execution

Phase 1 implements the three experiments designed in §7. This section documents all implementation details, debugging, fixes, and current status.

### 10.1 Probe C Baseline Results (SFT v2)

Before running experiments, we established Probe C baseline on the SFT v2 model (SLURM 3538418, COMPLETED).

**Probe A (Binding Presence)** — AUC across layers:

| Layer | AUC |
|---|---|
| L0 | 0.82 |
| L14 | 0.84 |
| L19 | 0.83 |
| L24 | 0.86 |
| L27 | 0.85 |

**Probe B (Coordinate Regression)** — R² across layers:

| Layer | R² |
|---|---|
| L0 | -0.45 |
| L14 | -0.12 |
| L19 | 0.05 |
| L24 | 0.15 |
| L27 | 0.22 |

**Probe C (Cross-Modal Alignment Gap)** — always negative, confirms fragmentation:

| Layer | Target-Task Sim | Nontarget-Task Sim | Gap |
|---|---|---|---|
| L0 | 0.409 | 0.532 | **-0.123** |
| L14 | 0.582 | 0.670 | **-0.088** |
| L19 | 0.532 | 0.596 | **-0.064** |
| L24 | 0.738 | 0.790 | **-0.053** |
| L27 | 0.591 | 0.693 | **-0.102** |

**Interpretation**: Probe C gap is negative at ALL layers — nontarget image tokens are more similar to task text than target tokens. This directly confirms representational fragmentation in SFT v2. The gap is smallest at L24 (-0.053), consistent with L24 being the "binding layer."

### 10.2 Environment Setup

Two conda environments are used, each for different training paradigms:

| Environment | Use Case | Key Packages |
|---|---|---|
| `qwen3-eval` | DeepSpeed ZeRO-3, SFT, Scheduled Sampling | PyTorch, HF Transformers, DeepSpeed |
| `ui-s1` | Ray-based GRPO via verl framework | PyTorch, vLLM, Ray, verl |

**Cluster**: GH200 nodes (95 GB GPU memory each), 4 GPUs/node.

**Shared config**: `train/env_config.sh` (CUDA paths, HF cache, NCCL settings).

### 10.3 Exp 1: Scheduled Sampling (Tests P1)

#### 10.3.1 Implementation

**Script**: `evaluation/scheduled_sampling_train.py`

Custom HF Trainer with two-pass scheduled sampling:
1. For each training sample, with probability `p_auto`:
   - First `n_tf_tokens` action tokens use teacher forcing (GT)
   - Remaining action tokens are generated autoregressively (model's own predictions)
2. With probability `(1 - p_auto)`: full teacher forcing (standard SFT)

Key design decisions:
- **Two-pass approach**: Pass 1 identifies action token positions, Pass 2 generates autoregressively from those positions
- **DeepSpeed ZeRO-3**: Required for full-parameter 7B model across 16 GPUs (4 nodes × 4)
- **`ddp_timeout=10800`**: Extended from default 1800s to prevent NCCL timeouts during ZeRO-3 allgather
- **`n_tf_tokens=3`**: First 3 action tokens always teacher forced (provides "function name" context)
- **Base model**: `checkpoints/Qwen2.5-VL-7B-Instruct` (NOT SFT v2 — trains from scratch like original SFT)
- **Training data**: `gui360_mixed_train.json` (181K samples, same as SFT v2)

#### 10.3.2 SLURM Setup

**Script**: `scripts/exp3/train_scheduled_sampling.slurm`

```
Environment: qwen3-eval (has DeepSpeed)
Nodes: 4 × 4 GPUs = 16 GPUs
DeepSpeed: ZeRO-3 (ds_z3_config.json)
batch_size: 1/gpu × 16 grad_accum = effective 256
Learning rate: 1e-5
Epochs: 2
Save: every 50 steps
Eval: every 100 steps
```

Submitted as:
- `P_AUTO=0.3 sbatch ...` → Job 3544165 (RUNNING)
- `P_AUTO=0.7 sbatch ...` → Job 3538415 (RUNNING, epoch 0.23, loss 0.243)

#### 10.3.3 Errors and Fixes

| Issue | Error | Fix |
|---|---|---|
| Missing DeepSpeed config | `deepspeed=None` in TrainingArguments | Added `--deepspeed_config` CLI arg, passed `ds_z3_config.json` |
| NCCL timeout (Job 3538414) | `Timeout(ms)=1800000` after 1h15m | Added `ddp_timeout=10800` (3 hours) |

### 10.4 Exp 2: GRPO Binary Reward (Tests P2)

#### 10.4.1 Implementation

**Framework**: verl (Ray + FSDP), GRPO algorithm via `verl.trainer.main_dapo`

**Config**: `train_GUI_360/moe_rl/traj_grpo_sftv2_exp2.yaml`

Key settings:
- **Base model**: SFT v2 (`gui360_full_sft_v2`) — full parameter, no LoRA, no MoE
- **Algorithm**: UIS1 advantage estimation (trajectory-aware)
- **Reward**: `gui360_binary` — score = 1.0 if type match AND action_score ≥ 0.8, else 0.0
- **Rollouts**: K=4 per sample, temperature=1.0
- **Data**: 2000-trajectory subset (`gui360_train_subset_2000.jsonl`)
- **KL**: `kl_loss_coef=0.1`, `kl_loss_type=low_var_kl`

#### 10.4.2 Binary Reward Function

**File**: `verl/utils/reward_score/gui360_binary/reward.py`

```python
# Strict 0/1 threshold:
score = 1.0 if (type_match AND action_score >= 0.8) else 0.0
```

Dispatched via `verl/utils/reward_score/__init__.py` when `data_source == "gui360_binary"`.

#### 10.4.3 SLURM Setup

**Script**: `scripts/exp3/train_grpo_sftv2.slurm`

```
Environment: ui-s1 (Ray + vLLM)
Nodes: 4 × 4 GPUs = 16 GPUs
Ray cluster: head + 3 workers
Algorithm: GRPO via verl.trainer.main_dapo
vLLM: gpu_memory_utilization=0.7, max_model_len=16384
```

Submitted as: `EXP_NAME=exp2 CONFIG_NAME=traj_grpo_sftv2_exp2 sbatch ...`

#### 10.4.4 Status

- Job 3538416 (full-param, full dataset): FAILED after 4h (46 steps) — NCCL timeout
  - Early metrics: score/mean slowly increasing from 0.012 to 0.065
- Job 3544166 (full-param, 2000 subset): RUNNING, step 13+

### 10.5 Exp 3: GRPO Dense Reward (Tests P3+P4)

#### 10.5.1 Implementation

**Config**: `train_GUI_360/moe_rl/traj_grpo_sftv2_exp3.yaml`

Same as Exp 2 except:
- **Reward**: default `gui360` — uses `soft_coordinate_score` (continuous [0,1])
- **No `override_data_source`** — defaults to gui360 continuous reward
- **`gpu_memory_utilization: 0.6`** (reduced from 0.7 for memory)
- **`ref.fsdp_config.param_offload: true`** (offload ref model to CPU)

#### 10.5.2 Status: ABANDONED (full-parameter)

Full-parameter Exp 3 OOM'd **3 times** despite progressive fixes:

| Attempt | Job | Fix Applied | Outcome |
|---|---|---|---|
| 1 | 3538417 | gpu_mem=0.7 | OOM in 6min |
| 2 | 3544167 | gpu_mem=0.6 | OOM in 7min |
| 3 | 3544253 | gpu_mem=0.6 + ref offload | OOM in 7min |

Root cause: Full-parameter 7B model + vLLM rollout + ref model exceeds 95GB GH200 memory. The continuous reward's gui360 scoring may allocate more buffers than binary.

**Replaced by Exp 3b (LoRA variant).**

### 10.6 Exp 2b/3b: LoRA GRPO Variants (Fast Baselines)

#### 10.6.1 Motivation

Full-parameter GRPO is slow (~10 min/step) and Exp 3 OOMs. Added LoRA variants:
- Faster per-step iteration
- Lower memory footprint
- Tests whether RL can change geometry even with limited parameter budget
- Provides baseline for future MoE comparison (MoE Exp 3 uses same infra)

#### 10.6.2 LoRA via MoE Wrapper

verl's framework only supports LoRA through its MoE wrapper. We use a trick:

```yaml
moe:
  enabled: true
  num_experts: 1    # 1 expert = standard LoRA (no routing)
  top_k: 1
  expert_lora_r: 64
  expert_lora_alpha: 128
  target_modules: [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]
  balance_weight: 0.0   # no balance loss (only 1 expert)
  z_loss_weight: 0.0
```

This required relaxing the validation in `verl/trainer/ppo/moe_dapo_trainer.py`:
```python
# Before: num_experts < 2 → ValueError
# After:  num_experts < 1 → ValueError
```

Mathematically, 1-expert MoE = standard LoRA (router output is always [1.0]).

#### 10.6.3 Configurations

**Exp 2b** (`traj_grpo_sftv2_exp2b_lora.yaml`):
- Reward: `gui360_binary` (strict 0/1)
- LoRA: r=64, alpha=128, all 7 modules
- lr: 1e-5 (higher than full-param's 5e-6)
- micro_batch_size: 2/gpu (LoRA uses less memory)
- Data: 2000 subset

**Exp 3b** (`traj_grpo_sftv2_exp3b_lora.yaml`):
- Reward: `gui360` continuous distance
- Same LoRA config as 2b
- Data: 2000 subset

#### 10.6.4 Status

- Job 3544355 (Exp 2b LoRA binary): RUNNING
- Job 3544356 (Exp 3b LoRA dense): RUNNING

### 10.7 Complete Experiment Matrix

| ID | Experiment | Params | Reward | Data | Status | Job |
|---|---|---|---|---|---|---|
| Probe C | SFT v2 baseline probing | — | — | — | **COMPLETED** | 3538418 |
| Exp 1a | SS p=0.7 | Full 7B | SFT CE | 181K | **RUNNING** (epoch 0.23) | 3538415 |
| Exp 1b | SS p=0.3 | Full 7B | SFT CE | 181K | **RUNNING** (epoch 0.01) | 3544165 |
| Exp 2 | GRPO binary (full-param) | Full 7B | Binary 0/1 | 2000 subset | **RUNNING** | 3544166 |
| Exp 2b | GRPO binary (LoRA) | LoRA r=64 | Binary 0/1 | 2000 subset | **RUNNING** | 3544355 |
| Exp 3 | GRPO dense (full-param) | Full 7B | Continuous | 2000 subset | **ABANDONED** (OOM) | — |
| Exp 3b | GRPO dense (LoRA) | LoRA r=64 | Continuous | 2000 subset | **RUNNING** | 3544356 |

### 10.8 Key Files Created/Modified

| File | Purpose |
|---|---|
| `evaluation/scheduled_sampling_train.py` | Exp 1: Custom HF Trainer with two-pass scheduled sampling |
| `scripts/exp3/train_scheduled_sampling.slurm` | Exp 1: SLURM script (4 nodes, DeepSpeed ZeRO-3) |
| `scripts/exp3/train_grpo_sftv2.slurm` | Exp 2/3: SLURM script (4 nodes, Ray cluster) |
| `train_GUI_360/moe_rl/traj_grpo_sftv2_exp2.yaml` | Exp 2: Full-param GRPO, binary reward |
| `train_GUI_360/moe_rl/traj_grpo_sftv2_exp3.yaml` | Exp 3: Full-param GRPO, dense reward |
| `train_GUI_360/moe_rl/traj_grpo_sftv2_exp2b_lora.yaml` | Exp 2b: LoRA GRPO, binary reward |
| `train_GUI_360/moe_rl/traj_grpo_sftv2_exp3b_lora.yaml` | Exp 3b: LoRA GRPO, dense reward |
| `verl/utils/reward_score/gui360_binary/reward.py` | Binary reward function |
| `verl/utils/reward_score/__init__.py` | Added gui360_binary dispatch |
| `verl/trainer/ppo/moe_dapo_trainer.py` | Relaxed num_experts validation (≥1) |
| `train_GUI_360/llamafactory/ds_z3_config.json` | DeepSpeed ZeRO-3 config for scheduled sampling |
| `datasets/GUI-360/rl_data/gui360_train_subset_2000.jsonl` | 2000-trajectory RL training subset |

### 10.9 All Errors Encountered and Resolutions

| # | Job | Error | Root Cause | Resolution |
|---|---|---|---|---|
| 1 | 3538155-3538183 | `$P_AUTO` / `$EXP_NAME` not expanded | SLURM doesn't expand `${}` in `#SBATCH` directives | Cancelled; variable expansion is cosmetic (job name only) |
| 2 | 3538414 | NCCL timeout after 1h15m | `ddp_timeout` default 1800s too short for ZeRO-3 allgather | Added `ddp_timeout=10800` |
| 3 | 3538416 | NCCL timeout after 4h (46 steps) | Transient NCCL failure during FSDP reduce_scatter | Retry (same config); 46 steps completed OK before failure |
| 4 | 3538417 | CUDA OOM in 6min | Full 7B + vLLM + ref model > 95GB | Reduced gpu_mem → still OOM → abandoned full-param Exp 3 |
| 5 | 3544167, 3544253 | CUDA OOM (repeat) | Same as #4 | Tried ref offload → still OOM → switched to LoRA (Exp 3b) |
| 6 | 3544336, 3544337 | `ValueError: num_experts >= 2` | MoE wrapper validation rejected 1-expert config | Changed validation threshold from 2 to 1 |
| 7 | 3544355, 3544356 | `TypeError: JSON object must be str` | Model generates garbage early in RL → reward parser gets None | Normal behavior; caught and scored as 0; jobs continue |
| 8 | 3544166, 3544355, 3544356 | `format_score=2.3%`, `success_rate=0.012` | **RL pipeline used JsonFormat (`<think>/<action>`) but SFT v2 was trained on `<tool_call>` format with app-specific action spaces** | Created `SFTv2Format` class matching training data exactly; resubmitted as Exp 2b v2 / 3b v2 |
| 9 | sftv2.py template extraction | `KeyError: 'VK_CONTROL'` | GUI-360 prompts contain `{VK_CONTROL}`, `{TAB 2}` — Python `.format()` interprets as format keys | Used `__INSTRUCTION__`/`__HISTORY__` placeholders with `.replace()` instead of `.format()` |

### 10.10 Early Training Metrics

**Exp 1a (SS p=0.7)** — Job 3538415, 10h running:
- Epoch: 0.23 / 2.0
- Loss: 1.278 → 0.243 (steadily decreasing)
- Learning rate following warmup + cosine schedule

**Exp 1b (SS p=0.3)** — Job 3544165, ~1h running:
- Epoch: 0.01 / 2.0
- Loss: 1.277 (just started)

**Exp 2 (GRPO full-param)** — Job 3544166, ~1h running:
- Step: 13+
- score/mean: 0.065
- success_rate: 0.012

**Exp 2b/3b (LoRA GRPO)** — Jobs 3544355/3544356, ~15min running:
- Still in first rollout batch (scoring 8000 responses)
- Many "Error Response" in reward scoring (normal — model generates garbage early)

### 10.11 Critical Discovery: RL Format Mismatch

#### 10.11.1 The Problem

**All RL experiments (Exp 2, 2b, 3b) before this fix were using the WRONG prompt/response format.**

The SFT v2 model was trained on a completely different format than what the RL pipeline generated:

| Component | SFT v2 Training Data | RL Pipeline (JsonFormat) |
|---|---|---|
| **Turn structure** | Single-turn (history as text) | Multi-turn accumulation |
| **System prompt** | None (everything in user message) | Separate system prompt with action space |
| **Action space** | **3 app-specific** (excel: 10 actions, ppt: 6, word: 10) | 5 generic actions |
| **Response format** | `<tool_call>\n{"function":"click","args":{...},"status":"CONTINUE"}\n</tool_call>` | `<think>...</think>\n<action>\n{"action":"click","coordinate":[x,y]}\n</action>` |
| **History format** | `Step 1: {thought text}` | Multi-turn messages |

This mismatch explains:
- `format_score = 2.3%` at step 0 (model can't produce `<think>/<action>` format it was never trained on)
- `success_rate = 0.012` (most model outputs are unparseable by the reward function)
- All early RL metrics were essentially noise

#### 10.11.2 The Fix: SFTv2Format

Created `x/data/agent/sftv2.py` — a new format class that **exactly matches the SFT v2 training data**.

Key components:

1. **App-specific templates** (`x/data/agent/sftv2_templates.json`): Extracted the exact prompt templates from `gui360_train.json` for all 3 apps (excel, ppt, word), each with their specific action spaces:
   - Excel: click, type, drag, wheel_mouse_input, table2markdown, insert_excel_table, select_table_range, set_cell_value, auto_fill, reorder_columns
   - PPT: click, type, drag, wheel_mouse_input, set_background_color, save_as
   - Word: click, type, drag, wheel_mouse_input, select_text, insert_table, set_font, set_paragraph_format, find_and_replace, table2markdown

2. **App detection**: `_detect_app(line)` — determines app from `execution_id` (e.g., "excel_1_81" → "excel")

3. **Nested action format**: Converts flat RL internal `{"action":"click","coordinate":[x,y]}` ↔ nested SFT v2 `{"function":"click","args":{"coordinate":[x,y],"button":"left"},"status":"CONTINUE"}`

4. **History as thought text**: Uses reasoning text from previous steps (matching training data), not function-call signatures

5. **Modified files**:
   - `verl/utils/dataset/universal_multiround.py` — `JsonFormat` → `SFTv2Format` in both `StdTrajectory` and `MultiRoundGenerator`
   - `verl/utils/reward_score/gui360/reward.py` — `JsonFormat` → `SFTv2Format` (auto-propagates to `gui360_binary`)

#### 10.11.3 Impact

With the correct format:
- Step 0 format_score should be **>>2.3%** (model knows this format from SFT training)
- Step 0 type_match should be close to **~44.8%** (matching SFT v2 eval function accuracy)
- Reward signal will now be meaningful (not dominated by parsing failures)

**Old Exp 2b/3b (jobs 3544355/3544356) are kept running as wrong-format baselines for comparison.**

### 10.12 SFT v2 with Thinking (Path 2)

#### 10.12.1 Motivation

The current SFT v2 checkpoint has **no reasoning/thinking** in its training labels. The prompt says "First, explain your reasoning process" but all assistant responses in `gui360_train.json` are pure `<tool_call>` blocks with no reasoning prefix.

This means:
- The model was never trained to output reasoning before actions
- RL can only learn action tokens, not reasoning chains
- Future RL with thinking requires retraining SFT with thinking data

#### 10.12.2 Data Preparation

**Script**: `train_GUI_360/data_preparation/prepare_gui360_sft_with_thinking_llamafactory.py`

Approach:
1. Build filename-based index of 13,750 raw trajectory JSONL files
2. For each training sample, extract `execution_id` and `step_num` from image path
3. Look up the corresponding raw trajectory step and extract `thought` field
4. Prepend `"Reasoning: {thought}\n\n"` to the assistant response

Match rate: **~96.7%** (remaining ~3.3% are steps beyond raw trajectory length or missing trajectories)

Output format: Same ShareGPT JSON, compatible with LLaMA Factory (registered as `gui360_train_with_thinking` and `gui360_val_with_thinking` in `dataset_info.json`).

#### 10.12.3 Training Config

**Config**: `train_GUI_360/llamafactory/qwen25vl_gui360_full_sft_v2_thinking.yaml`
- Base model: `Qwen2.5-VL-7B-Instruct` (same as SFT v2)
- Full-parameter SFT, freeze vision tower
- lr=1e-5, epochs=2, cosine schedule
- Output: `gui360_full_sft_v2_thinking`

**SLURM**: `train_GUI_360/llamafactory/train_gui360_full_sft_v2_thinking.slurm`
- 4 nodes × 4 GPUs, 24h walltime
- Step 1: Data prep on head node (generates with-thinking JSON)
- Step 2: LLaMA Factory SFT training on all nodes

### 10.13 Resubmitted Experiments with Fixed Format

#### 10.13.1 New Job Submissions

| ID | Experiment | Config | Format | Status | Job |
|---|---|---|---|---|---|
| Exp 2b (v2) | GRPO binary (LoRA) — **fixed format** | `traj_grpo_sftv2_exp2b_lora` | **SFTv2Format** | **RUNNING** | 3545634 |
| Exp 3b (v2) | GRPO dense (LoRA) — **fixed format** | `traj_grpo_sftv2_exp3b_lora` | **SFTv2Format** | **RUNNING** | 3545753 |
| SFT+Think | SFT v2 with thinking | LLaMA Factory full SFT | — | **RUNNING** (data prep) | 3545630 |

#### 10.13.2 Verification Plan

At step 0 of new Exp 2b/3b:
1. `format_score >> 2.3%` — model outputs parseable `<tool_call>` format
2. `type_match ≈ 44.8%` — function accuracy matches SFT v2 eval
3. `success_rate >> 0.012` — meaningful reward signal from step 0

If step 0 metrics are still low → remaining mismatch in template/history format (debug by printing example prompts).

### 10.14 Updated Experiment Matrix

| ID | Experiment | Params | Reward | Format | Data | Status | Job |
|---|---|---|---|---|---|---|---|
| Probe C | SFT v2 baseline probing | — | — | — | — | **COMPLETED** | 3538418 |
| Exp 1a | SS p=0.7 | Full 7B | SFT CE | — | 181K | **RUNNING** | 3538415 |
| Exp 1b | SS p=0.3 | Full 7B | SFT CE | — | 181K | **RUNNING** | 3544165 |
| Exp 2 | GRPO binary (full-param) | Full 7B | Binary 0/1 | ~~JsonFormat~~ | 2000 | **RUNNING** (wrong format) | 3544166 |
| ~~Exp 2b~~ | ~~GRPO binary (LoRA)~~ | LoRA r=64 | Binary 0/1 | ~~JsonFormat~~ | 2000 | **RUNNING** (wrong format, kept as baseline) | 3544355 |
| ~~Exp 3b~~ | ~~GRPO dense (LoRA)~~ | LoRA r=64 | Continuous | ~~JsonFormat~~ | 2000 | **RUNNING** (wrong format, kept as baseline) | 3544356 |
| **Exp 2b v2** | **GRPO binary (LoRA)** | LoRA r=64 | Binary 0/1 | **SFTv2Format** | 2000 | **RUNNING** | **3545634** |
| **Exp 3b v2** | **GRPO dense (LoRA)** | LoRA r=64 | Continuous | **SFTv2Format** | 2000 | **RUNNING** | **3545753** |
| **SFT+Think** | **SFT v2 with thinking** | Full 7B | SFT CE | — | 101K | **RUNNING** (data prep) | **3545630** |
| Exp 3 | GRPO dense (full-param) | Full 7B | Continuous | — | 2000 | **ABANDONED** (OOM) | — |

### 10.15 Updated Key Files

| File | Purpose |
|---|---|
| `x/data/agent/sftv2.py` | **NEW**: SFTv2Format class matching SFT v2 training data exactly |
| `x/data/agent/sftv2_templates.json` | **NEW**: 3 app-specific prompt templates (excel, ppt, word) |
| `verl/utils/dataset/universal_multiround.py` | **MODIFIED**: JsonFormat → SFTv2Format |
| `verl/utils/reward_score/gui360/reward.py` | **MODIFIED**: JsonFormat → SFTv2Format |
| `train_GUI_360/data_preparation/prepare_gui360_sft_with_thinking_llamafactory.py` | **NEW**: Add reasoning to SFT data |
| `train_GUI_360/llamafactory/qwen25vl_gui360_full_sft_v2_thinking.yaml` | **NEW**: With-thinking SFT config |
| `train_GUI_360/llamafactory/train_gui360_full_sft_v2_thinking.slurm` | **NEW**: Combined data-prep + training SLURM |
| `train_GUI_360/llamafactory/data/dataset_info.json` | **MODIFIED**: Added with-thinking dataset entries |

### 10.16 Next Steps

1. **Monitor new Exp 2b v2 / 3b v2** — verify step 0 format_score >> 2.3% (confirms format fix)
2. **Monitor SFT+Think training** — wait for completion (~24h), then evaluate on GUI-360 test set
3. **Compare old vs new format**: Old Exp 2b/3b (wrong format) vs new (correct format) — quantify impact of format mismatch
4. **After SFT+Think completes**: Submit RL experiments using the with-thinking checkpoint (model that can reason before acting)
5. **Run Probe C on RL checkpoints**: At Exp 2b v2 / 3b v2 checkpoints, measure Probe C gap evolution
6. **Error decomposition**: On new RL checkpoints, measure far miss / near miss ratio changes
7. **Compare reward curves**: Binary (Exp 2b v2) vs dense (Exp 3b v2) — does dense reward show faster convergence?
8. **If SS improves Probe C**: Direct evidence that teacher forcing → fragmentation (confirms P1)
9. **If GRPO with correct format improves**: Evidence that RL changes representational geometry (confirms P2)
