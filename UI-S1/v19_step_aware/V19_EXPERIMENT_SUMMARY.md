# V19: Step-Aware Reasoning for Long-Horizon GUI Navigation

## Overview

V19 investigates why the SFT-trained GUI agent fails on long-horizon tasks and explores multiple approaches to fix it. The core problem: **error accumulation** — the model is trained on ground-truth (GT) history but tested with its own imperfect predicted history, creating a distribution mismatch that compounds over steps.

**Base SFT model**: Qwen2.5-VL-7B, full fine-tuned on GUI-360 balanced 2K training data (checkpoint-272).

---

## 1. Baselines

| Experiment | History | TSR | StepSR | 1-step | 2-3 steps | 4-5 steps | 6+ steps |
|---|---|---|---|---|---|---|---|
| Standard SFT (GT hist) | GT | 21.9% | 68.8% | — | — | — | — |
| Standard SFT (Pred hist) | Pred | 21.9% | 46.3% | 72.3% | 33.3% | 25.3% | 6.6% |
| CoT-Step (GT hist) | GT | 24.0% | 59.1% | — | — | — | — |
| CoT-Step (Pred hist) | Pred | 23.1% | 46.9% | — | — | — | — |

**Key observation**: StepSR drops from ~68% (GT history) to ~46% (pred history), confirming that **distribution mismatch is the primary bottleneck**, not per-step capability.

---

## 2. Phase 1: Prompt Engineering (CoT Variants)

Tested 4 prompt variants to address CoT's "jumping ahead" problem on long tasks while preserving its grounding benefits on short tasks.

| Variant | GT Hist TSR | Pred Hist TSR | Delta vs Base |
|---|---|---|---|
| **Standard** (baseline) | 21.9% | 21.9% | — |
| **CoT** (baseline) | 24.0% | 23.1% | +1.2pp |
| **Focused CoT** (anti-jump) | 23.0% | 21.5% | -0.4pp |
| **Adaptive CoT** (CoT for steps 0-1, standard for 2+) | 23.3% | 21.9% | ±0.0pp |
| **Step Context** (progress counter + "RIGHT NOW") | 22.2% | 21.7% | -0.2pp |
| **Type Focused** (explicit action-type reasoning) | 24.1% | **23.6%** | **+1.7pp** |

### By Task Length (Pred History)

| Variant | 1-step | 2-3 steps | 4-5 steps | 6+ steps |
|---|---|---|---|---|
| Standard | 72.3% | 33.3% | 25.3% | 6.6% |
| Focused CoT | 63.9% | 30.8% | 25.3% | 5.9% |
| Adaptive CoT | 68.1% | 31.3% | 24.2% | 5.9% |
| Step Context | 62.2% | 30.8% | 24.7% | 7.0% |
| **Type Focused** | **69.7%** | **34.9%** | **27.8%** | 6.1% |

**Winner**: Type Focused prompt — best overall TSR (23.6%) with strong short-task performance. Used as default prompt for subsequent experiments.

---

## 3. Phase 2: Test-Time Verification (Critic / Best-of-N)

### 3a. Self-Verification

Model verifies its own prediction by asking "Is this action correct?"

| Experiment | History | TSR | StepSR |
|---|---|---|---|
| Self-Verify | GT | 23.8% | 59.3% |
| Self-Verify | Pred | 23.7% | 47.1% |

**Result**: No improvement — model cannot reliably judge its own actions.

### 3b. Trained Verifier

Separate verifier model fine-tuned on hard negative examples.

| Experiment | History | TSR | StepSR |
|---|---|---|---|
| Trained Verifier | GT | 21.4% | 56.5% |
| Trained Verifier | Pred | 20.6% | 45.0% |

**Result**: Worse than baseline — verifier rejects too many correct actions (false negatives).

### 3c. Dual Hard Verifier

Two-model voting with hard negative training.

| Experiment | History | TSR | StepSR |
|---|---|---|---|
| Dual Hard Verify | GT | 23.9% | 59.4% |
| Dual Hard Verify | Pred | 23.3% | 47.5% |

**Result**: No improvement over baseline.

### 3d. Oracle Analysis

Upper-bound analysis with oracle type/noise knowledge.

| Experiment | History | TSR | StepSR |
|---|---|---|---|
| Oracle Type | Pred | 22.4% | 47.2% |
| Oracle None | Pred | 23.4% | 47.2% |
| Oracle Noisy | Pred | 22.7% | 47.2% |

**Result**: Even with oracle action-type information, no improvement — the bottleneck is not action-type selection.

### 3e. Best-of-N with Log-Probability Ranking

Generate N candidates and select based on model's own log-probability.

| Experiment | History | TSR | StepSR | Greedy Acc | Oracle Acc | Selected Acc |
|---|---|---|---|---|---|---|
| BoN-3 | GT | 25.5% | 62.0% | 59.2% | 66.4% | 62.0% |
| BoN-3 LogProb | Pred | 24.7% | 53.1% | 49.9% | 55.9% | 53.1% |
| **BoN-5 LogProb** | Pred | **26.6%** | **54.5%** | 49.7% | 59.5% | 54.5% |
| BoN-5 Verifier | Pred | 26.0% | 53.1% | 49.3% | 58.5% | 53.1% |

**Winner**: BoN-5 with log-probability ranking — **TSR 26.6%** (+4.7pp over standard baseline, +3.0pp over type_focused). Log-prob outperforms trained verifier for selection.

### 3f. AISAP (AI Self-Adaptive Pipeline)

Multi-strategy test-time compute with voting, repair, and adaptive selection.

| Experiment | TSR | StepSR |
|---|---|---|
| E1: Voting (single model) | 23.3% | 47.1% |
| E2: Voting (diverse prompts) | 22.7% | 47.4% |
| E4: Repair (diverse) | 23.3% | 47.1% |
| **E5: Adaptive** | **23.8%** | **47.2%** |

**Result**: Marginal improvement with adaptive strategy. Much less effective than simple BoN-5 logprob.

---

## 4. Phase 3: Step-Level Reinforcement Learning

### 4a. GRPO R1: Cross-Type Binary Reward

Standard GRPO with binary rewards across all action types.

| Experiment | TSR | StepSR | 1-step | 6+ steps |
|---|---|---|---|---|
| GRPO R1 Greedy | **10.9%** | 39.9% | 42.9% | 3.3% |
| GRPO R1 BoN-5 | 11.8% | 45.8% | 43.7% | 3.3% |

**Result**: Catastrophic mode collapse — model learns to always predict "click" (most common type gets highest reward across mixed-type groups). TSR drops from 23.3% to 10.9%.

### 4b. GRPO R2: Type-Filtered Binary Reward

Fix mode collapse by only computing advantages within same-type candidates.

| Experiment | TSR | StepSR | 1-step | 6+ steps |
|---|---|---|---|---|
| GRPO R2 Greedy | 23.6% | 47.3% | 68.9% | 6.8% |
| GRPO R2 BoN-5 | 24.9% | 54.6% | 68.9% | 7.6% |

**Result**: No mode collapse, but also no improvement over base (23.3%). Binary reward has insufficient learning signal — most candidates are either all-correct or all-wrong within type.

### 4c. GRPO R3: Gaussian Continuous Reward

Continuous Gaussian distance reward for coordinate proximity.

| Experiment | TSR | Notes |
|---|---|---|
| GRPO R3 | — | **Training diverged** (loss=-54, KL=-10.7) |

**Result**: Loss diverged catastrophically. KL penalty became negative (accelerating divergence instead of constraining it).

### 4d. Rejection Fine-Tuning (RFT)

SFT on only correct candidates (reward >= 0.8).

| Experiment | TSR | StepSR | 1-step | 6+ steps |
|---|---|---|---|---|
| RFT Greedy | 23.5% | 46.5% | 71.4% | 6.8% |
| RFT BoN-5 | 24.6% | 51.6% | 69.7% | 7.2% |

**Result**: No improvement — self-reinforcement on data the model already gets right.

### 4e. Action-Space GRPO (AS-GRPO)

Instead of token-space sampling (which produces similar candidates), perturb actions directly in parameter space:
- **Click**: Gaussian coordinate perturbation → distance-based reward
- **Type**: Character-level text edits → edit-distance reward

Stats: 10,300 click groups + 1,515 type groups, 80.7% click and 98.3% type groups with signal.

| Experiment | TSR | StepSR | Notes |
|---|---|---|---|
| AS-GRPO v1 (action_loss_only, LR=5e-6) | — | — | Diverged (loss=-54) |
| AS-GRPO v2 (action_loss_only, LR=5e-7) | **0.0%** | **0.0%** | Format collapse |
| **AS-GRPO v3** (full loss, LR=5e-7) | **22.8%** | 47.5% | Stable but no improvement |

**Result**:
- `action_loss_only` is fundamentally broken — masking reasoning tokens from loss causes gradients to still propagate through attention, destroying format/reasoning.
- Full-loss AS-GRPO trains stably but gives no improvement (22.8% vs 23.3% baseline).

---

## 5. Phase 4: Trajectory-Level Optimization

### 5a. DAgger SFT

DAgger (Dataset Aggregation) directly addresses distribution mismatch by training on (pred_history, GT_action) instead of (GT_history, GT_action).

- Phase 1: Rolled out base model on 2K training episodes → 16,894 steps, 63.1% diverged
- Phase 2: SFT on pred_history + GT_history (mixed), LoRA rank=64, LR=2e-6, 2 epochs

| Experiment | TSR | StepSR | 1-step | 2-3 steps | 4-5 steps | 6+ steps |
|---|---|---|---|---|---|---|
| DAgger Greedy | **20.1%** | 45.1% | 63.0% | 29.7% | 20.7% | 5.5% |

**Result**: Worse than baseline (-3.2pp). The GT response used for training was a bare `<tool_call>` block without reasoning text. This caused the model to skip reasoning → format degradation → accuracy drop on all task lengths.

---

## Summary Table: All Experiments

| # | Experiment | TSR | Delta | Category |
|---|---|---|---|---|
| — | **Standard SFT (Pred)** | **21.9%** | — | Baseline |
| — | **CoT-Step (Pred)** | **23.1%** | +1.2pp | Baseline |
| 1 | Focused CoT | 21.5% | -0.4pp | Prompt |
| 2 | Adaptive CoT | 21.9% | ±0.0pp | Prompt |
| 3 | Step Context | 21.7% | -0.2pp | Prompt |
| 4 | **Type Focused** | **23.6%** | **+1.7pp** | Prompt |
| 5 | Self-Verify | 23.7% | +1.8pp | Critic |
| 6 | Trained Verifier | 20.6% | -1.3pp | Critic |
| 7 | Dual Hard Verifier | 23.3% | +1.4pp | Critic |
| 8 | AISAP Adaptive | 23.8% | +1.9pp | Critic |
| 9 | BoN-3 LogProb | 24.7% | +2.8pp | Test-time |
| 10 | **BoN-5 LogProb** | **26.6%** | **+4.7pp** | **Test-time** |
| 11 | BoN-5 Verifier | 26.0% | +4.1pp | Test-time |
| 12 | GRPO R1 (cross-type) | 10.9% | -11.0pp | RL |
| 13 | GRPO R2 (type-filtered) | 23.6% | +1.7pp | RL |
| 14 | GRPO R3 (gaussian) | — | diverged | RL |
| 15 | RFT | 23.5% | +1.6pp | RL |
| 16 | AS-GRPO v2 (action_loss) | 0.0% | -21.9pp | RL |
| 17 | AS-GRPO v3 (full loss) | 22.8% | +0.9pp | RL |
| 18 | DAgger SFT | 20.1% | -1.8pp | Trajectory |

---

## Key Findings

### 1. Distribution mismatch is the fundamental bottleneck
- StepSR: 68.8% (GT history) vs 46.3% (pred history) — a 22.5pp gap
- The model is near-optimal per step given clean inputs; it fails because it sees corrupted history at test time

### 2. Step-level RL cannot solve this problem
- All 6 RL variants (GRPO R1-R3, RFT, AS-GRPO v2-v3) failed
- The model already maximizes per-step accuracy; there is no room for step-level improvement
- Token-space and action-space exploration both fail to find better actions

### 3. Test-time compute (BoN) is the most effective approach so far
- BoN-5 with log-prob ranking: **26.6% TSR** (+4.7pp, +21% relative improvement)
- Simple log-probability outperforms trained verifiers for candidate selection
- Oracle BoN-5 accuracy is 59.5% — significant headroom remains

### 4. Prompt engineering has limited impact
- Best prompt (Type Focused): +1.7pp over standard
- CoT variants can hurt on long tasks due to "jumping ahead"
- No prompt variant addresses the distribution mismatch

### 5. Naive DAgger fails due to response format
- Training on bare tool_call responses without reasoning degrades model capabilities
- Future DAgger attempts need to preserve the model's reasoning format while only correcting the action

---

## What Didn't Work and Why

| Approach | Failure Mode | Root Cause |
|---|---|---|
| GRPO R1 | Mode collapse (10.9%) | Cross-type advantage → learns to always predict majority type |
| GRPO R3 | Diverged | KL penalty becomes negative with continuous rewards → accelerates divergence |
| Trained Verifier | Accuracy drop (-1.3pp) | High false-negative rate → rejects correct actions |
| AS-GRPO action_loss_only | Format collapse (0.0%) | Gradient leaks through attention to unmasked reasoning weights |
| DAgger SFT | Accuracy drop (-3.2pp) | Bare tool_call GT response strips reasoning → format degradation |

---

## Best Configuration

**For maximum accuracy**: BoN-5 + LogProb ranking with Type Focused prompt
- TSR: **26.6%** (greedy baseline: 21.9%)
- Cost: 5x inference compute per step

**For single-pass inference**: Type Focused prompt
- TSR: **23.6%** (greedy baseline: 21.9%)
- Cost: 1x inference compute
