# Long-Horizon GUI Agent: Diagnostic Experiments & Results

## 1. Baseline

| Metric | Value |
|--------|-------|
| Model | Qwen2.5-VL-7B, gui360_balanced_full_sft checkpoint-272 |
| Test set | GUI-360 1000 episodes (balanced) |
| TSR (teacher-forced) | 22.5% |
| Step SR (teacher-forced) | 58.8% |
| Step SR (autoregressive) | 46.3% (drops from 62% at step 1 → 30% at step 15) |

## 2. Root Cause Diagnosis

### 2.1 Eliminated Hypotheses

| Hypothesis | Experiment | Result | Conclusion |
|------------|-----------|--------|------------|
| History explosion | History ablation (full/none/last_3) | full=58.8%, last_3=55.0%, none=39.7% | **Not bottleneck** — last_3 ≈ full |
| Resolution mismatch | Resolution ablation (602K vs 765K) | 57.9% vs 58.9% | **Not bottleneck** — nearly identical |
| Lack of planning | Self-planning prompt (Exp A) | Step SR 50.2% (-8.6pp) | **7B can't plan**, hurts performance |
| Need external guidance | Base model per-step guidance (Exp D) | Step SR 59.1% (+0.3pp) | **No help** — 7B base also insufficient |
| Grounding precision | Dedicated grounder (gui_grounding) | 25.6% coord accuracy, combined 42.8% | **Grounder worse** than SFT model |

### 2.2 Confirmed Root Causes

**Root Cause 1: Binary Visual Grounding**
- Click correct: median distance = 0.5px (perfect recognition)
- Click wrong: median distance = 333px, 100% > 100px
- **Zero near-misses** → model either finds the right element or a completely wrong one
- This is a recognition/identification problem, not a precision problem

**Root Cause 2: Systematic Type Confusion**
- 44% of errors are action type mismatch (click↔type bidirectional: 914 cases)
- Type error rate stable ~44% across all step positions (not worse at later steps)
- SFT action type accuracy: 81.9%

**Root Cause 3: Distribution Shift (Exposure Bias)**
- Teacher-forced Step SR: 58.8% (constant across step positions)
- Autoregressive Step SR: 62% → 30% over steps (compounding errors)
- Gap = 12.5pp average, growing with trajectory length
- Model never trained on "wrong state" screenshots

**Root Cause 4: Error Compounding**
- Even at 58.8% per-step: p^10 = 0.5% for 10-step tasks
- 69% of episodes show recovery events (Exp B autoreg), but insufficient
- Need either much higher per-step accuracy OR error recovery mechanisms

## 3. Per-Step Accuracy Boosting Experiments

### 3.1 Results

| Method | Step SR | Oracle Best-of-N | TSR |
|--------|---------|-----------------|-----|
| Baseline (greedy T=0) | 58.8% | — | 22.5% |
| CoT per-step reasoning | 59.2% (+0.4) | — | 23.6% (+1.1) |
| Self-consistency N=3 (majority vote) | 57.7% (-1.1) | 67.7% | — |
| Self-consistency N=5 (majority vote) | 58.3% (-0.5) | 71.7% | — |
| Self-verification & retry (max_retries=2) | 58.8% (+0.0) | — | 22.4% (-0.1) |

### 3.2 Key Findings

**Majority voting fails because errors are systematic, not random.**
The model makes the same mistake 3+/5 times — it consistently picks the wrong element.

**Self-verification completely fails.**
Only 3/7498 steps triggered a retry (0.04%). The 7B SFT model has no self-verification ability — it judges its own wrong predictions as "CORRECT" almost every time. A verifier must be an **external** model, not the same model.

**Oracle Best-of-N = 71.7% is the critical number.**
In 5 samples, at least 1 is correct 71.7% of the time. This is 12.9pp above baseline.
If we had a perfect verifier, we could boost Step SR from 58.8% → 71.7%.
The self-verify experiment proves this verifier cannot be the same 7B model.

**Agreement analysis (N=5):**

| Category | Count | % |
|----------|-------|---|
| All 5 correct | 3097 | 41.3% |
| 1-4 correct (partial) | 2276 | 30.3% |
| All 5 wrong | 2125 | 28.3% |

The **30.3% partial** steps are the opportunity: model "knows" the answer but can't reliably select it. A good verifier would unlock this.

### 3.3 Grounder + SFT Combined

| Metric | Value |
|--------|-------|
| Grounder standalone coord accuracy | 25.6% (5% threshold) |
| Grounder median distance | 205px |
| SFT standalone Step SR | 58.8% |
| Combined (SFT type + grounder coords) | 42.8% (-16.0pp) |
| Grounder helped | 363 steps |
| Grounder hurt | 1560 steps |
| Net | **-1197 steps** |

The dedicated gui_grounding model (97K steps) is far worse than SFT at visual grounding. SFT's correct clicks are at 0.5px; grounder's median is 205px.

## 4. Long-Horizon Strategy

### 4.1 The Math

Per-step accuracy p → n-step task success = p^n

| p | 5-step | 10-step | 15-step |
|---|--------|---------|---------|
| 58.8% (current) | 7.1% | 0.5% | 0.04% |
| 65% | 11.6% | 1.3% | 0.15% |
| 71.7% (oracle N=5) | 19.2% | 3.7% | 0.70% |
| 80% | 32.8% | 10.7% | 3.5% |
| 90% | 59.0% | 34.9% | 20.6% |

Even oracle N=5 (71.7%) only gives 3.7% for 10-step tasks. To meaningfully improve long-horizon, we need BOTH higher per-step accuracy AND robustness to errors.

### 4.2 Best-of-N Verifier Analysis

**Step distribution (N=5, T=0.7, 200 episodes):**

| Category | Steps | % |
|----------|-------|---|
| All 5 correct | 642 | 40.7% |
| Partial (1-4 correct) | 460 | 29.1% |
| All 5 wrong | 476 | 30.2% |

**Selection strategies on ALL steps:**

| Strategy | Step SR | vs Random-from-5 |
|----------|---------|-------------------|
| Baseline (greedy T=0) | ~58.8% | — |
| Random pick from N=5 | 54.8% | — |
| tail_mean_logprob | 58.2% | +3.3pp |
| Oracle (any correct) | 69.8% | +15.0pp |

**Selection strategies on PARTIAL steps only (where it matters):**

| Metric | Accuracy | vs Random 48.5% |
|--------|----------|-----------------|
| tail_mean_logprob | 60.0% | +11.5pp |
| mean_logprob | 58.9% | +10.4pp |
| coord_agreement | 42.8% | -5.7pp |

**Separability (Cohen's d):**

| Metric | Cohen's d | |
|--------|-----------|--|
| coord_agreement | 0.962 | large (but poor selector — high agreement when all wrong too) |
| tail_mean_logprob | 0.806 | large (best practical selector) |
| mean_logprob | 0.798 | medium |
| perplexity | 0.780 | medium |

**Key insight:** Logprob captures "model confidence" but NOT "confident AND correct vs confident AND wrong." The model is often confidently wrong (consistent with self-verify 0.04% rejection rate). Logprob recovers only 22% of the oracle gap (3.3pp out of 15.0pp). A trained verifier (PRM) is needed to close the remaining 11.6pp.

### 4.3 Promising Directions

**Tier 1: Inference-time (no retraining)**
- ~~Self-verification & retry~~ → tested, useless (model can't self-verify)
- ~~Logprob-based selection~~ → tested, recovers only 22% of oracle gap
- Best-of-N with **trained** verifier (PRM) → needed, 11.6pp gap to close
- Trajectory beam search

**Tier 2: Training-time**
- DAgger: train on model-generated wrong states → learn error recovery
- Train PRM: (screenshot, action) → correct/wrong classifier
- Trajectory-level GRPO with longer horizons (10-15 steps)
- Hard negative mining for click↔type confusion

**Tier 3: Architecture**
- State abstraction / structured state representation
- Memory-augmented model for progress tracking
- Hierarchical planning with sub-goals

### 4.3 Attention Alignment & Crop-Verify Analysis

**Attention extraction**: Failed (OOM) — Qwen2.5-VL-7B with `output_attentions=True` on full sequence exceeds GPU memory. All 28 layers' attention matrices are too large.

**Crop-and-Verify** (100 partial click steps):
Crop a 200×200 patch around the predicted coordinate, ask the model "what element is at center?", then check if it matches the instruction.

| | Verify=YES | Verify=NO |
|--|-----------|----------|
| Actually Correct | TP=0 | FN=36 |
| Actually Wrong | FP=1 | TN=31 |

- **Accuracy: 45.6%** (worse than random coin flip)
- **Rejection rate: 98.5%** — model says NO to almost everything (opposite of self-verify's 0.04%)
- **Recall: 0.0%** — even for CORRECT predictions, crop-verify says NO

**Key insight**: The model exhibits extreme opposite biases depending on prompting:
- Self-verify (full screenshot): says YES 99.96% → useless (too permissive)
- Crop-verify (200px patch): says NO 98.5% → useless (too restrictive)
- Neither mode can distinguish correct from wrong

**Conclusion**: The 7B SFT model fundamentally lacks the metacognitive ability to verify its own actions, regardless of prompting strategy. A trained PRM or external (larger) model is needed.

### 4.4 Verifier Gap Analysis

The gap between majority vote (58.3%) and oracle (71.7%) = **13.4pp** is the verifier opportunity.

If we can train a Process Reward Model (PRM) that correctly identifies which of N samples is correct, we recover most of this gap. The PRM CANNOT be:
- ~~Self-verification by the same model~~ (tested: 0.04% rejection rate, useless)
- ~~Self-consistency / majority vote~~ (tested: errors are systematic, not random)

The PRM must be:
- A separately trained verifier (e.g., trained on our eval data with correct/wrong labels)
- A larger model (e.g., 72B) used as judge
- A lightweight classifier on (screenshot, action) → correct/wrong

## 5. V18 K-Expert Status

### 5.1 NCCL Failure Root Cause Analysis

All 5 initial runs failed with NCCL timeout. Root causes identified:

**Bug 1: Per-parameter all_reduce (1120 calls per optimizer step)**
Each trainable parameter called `dist.all_reduce()` individually → massive small-message overhead.
**Fix**: Bucketed single all_reduce — flatten all grads into one contiguous buffer, one all_reduce, copy back.

**Bug 2: Episode desync across ranks**
`DistributedSampler` assigns different episodes to different ranks. Episodes have 1-15 steps, so ranks finish at wildly different times → rank that finishes first waits at all_reduce while slow rank is still generating → 3600s NCCL timeout.
**Fix**: `SyncedShuffleSampler` — all ranks process the same episode in the same order. Same episode = same T steps = natural sync at all_reduce boundary. Trajectory diversity preserved via stochastic `model.generate()` (different CUDA random state per rank).

### 5.2 Run History

| Run | Nodes | Fix | Result |
|-----|-------|-----|--------|
| K=4 attempt 1 (4699915) | 4 | none | NCCL failure |
| K=8 attempt 1 (4699992) | 4 | none | NCCL failure |
| K=4 attempt 2 (4700752) | 4 | none | NCCL failure, 3 opt steps |
| K=4 attempt 3 (4715611) | 4 | none | NCCL failure, 1 opt step |
| K=4 attempt 4 (4720078) | 2 | none | NCCL failure |
| K=4 attempt 5 (4723449) | 4 | SyncedShuffleSampler + bucketed all_reduce | **PENDING** |
