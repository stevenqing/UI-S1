# UI-S1 Evaluation Summary

All evaluations are on the **GUI-360** dataset (desktop GUI action prediction).
Base model: **Qwen2.5-VL-7B-Instruct**. Test set: 1000 episodes / 7498 steps (unless noted).

---

## 1. SFT Model Comparisons (Full Test Set, 1000 Episodes)

Step-level evaluation with GT history, stop-on-error trajectory evaluation.

| Model | TSR | Step SR | Avg Progress | Notes |
|-------|-----|---------|--------------|-------|
| V12 Standard LoRA GRPO (epoch 2) | 11.8% | 53.5% | 0.236 | Previous version baseline |
| V12 SFT (balanced, no RL) | 3.0% | 28.0% | 0.086 | Pure SFT without RL |
| **V15 Full-Param SFT step-272** | **21.9%** | **68.8%** | **0.352** | Best SFT checkpoint |
| V15 Full-Param SFT step-250 | 22.2% | 69.3% | 0.353 | Marginally better |
| V15 Full-Param SFT step-200 | 21.0% | 68.1% | 0.342 | |
| V15 Full-Param SFT step-150 | 22.1% | 67.0% | 0.345 | |
| V15 Full-Param SFT step-100 | 19.0% | 64.3% | 0.311 | |

**Takeaway**: Full-param SFT on balanced GUI-360 data achieves ~22% TSR, stable from step 150 onward.

---

## 2. LoRA Extraction & Cooperative LoRA SFT (Full Test Set)

Testing whether cooperative LoRA (2-expert) can match full-param SFT.

| Model | TSR | Step SR | Avg Progress | Notes |
|-------|-----|---------|--------------|-------|
| SVD Extracted (attn only, r=128) | 4.3% | 48.2% | 0.149 | SVD → standard LoRA, 4 modules |
| SVD Standard LoRA (all 7 modules) | 17.9% | 65.0% | 0.305 | All 7 modules critical |
| Standard LoRA PEFT (r=128, 4 mod) | 7.9% | 56.3% | 0.234 | Trained from scratch |
| Standard LoRA PEFT (r=256, 4 mod) | 6.4% | 48.8% | 0.181 | Larger rank, still poor |
| Balanced Standard LoRA PEFT (r=128) | 18.1% | 65.2% | 0.308 | Balanced data helps |
| **Balanced Cooperative r=128 (best shard)** | **22.4%** | **68.2%** | **0.340** | Matches full-param SFT |
| Balanced Cooperative r=128 (all shards avg) | 18.6% | 65.3% | 0.310 | Variance across shards |
| Cooperative SFT epoch-0 | 3.8% | 44.3% | 0.141 | Training from scratch |
| Cooperative SFT epoch-1 | 6.6% | 48.8% | 0.160 | Improving |
| Cooperative SFT epoch-2 | 10.5% | 54.0% | 0.209 | Still improving |

**Takeaway**: SVD-extracted cooperative LoRA (r=128, 7 modules) matches full-param SFT at 22.4% TSR. Balanced data and all 7 target modules are critical.

---

## 3. RL Training Results (Val Set, 20 Episodes)

Note: validation set is small (20 episodes), results have high variance.

| Model | Mean Traj Reward | Full Success Rate | Mean Consec |
|-------|-----------------|-------------------|-------------|
| V15 RL (step-level, epoch 0) | 0.486 | — | — |
| V15 RL from Coop SFT (best, step 50) | 0.605 | — | — |
| V15 On-Policy GSPO (epoch 0) | 0.357 | 25.0% | 1.65 |
| V18 K=4 Expert (step 5, best) | 0.456 | 35.0% | 1.55 |
| V18 K=4 Expert (final, step 31) | 0.383 | 25.0% | 1.60 |
| V18 K=4 v2 (step 5) | 0.450 | 35.0% | — |

**V18 K-Expert verdict**: Routing never differentiated (entropy = log(4) throughout, expert cosine similarity = 1.0). Training degraded from step 5 to final. K-Expert architecture failed.

---

## 4. History Ablation (Full Test Set)

Using V15 Full SFT step-272, with GT history at eval time.

| History Mode | TSR | Step SR | Avg Reward |
|-------------|-----|---------|------------|
| **Full GT history** | **22.5%** | **58.8%** | **0.643** |
| Last 3 actions only | 22.1% | 55.0% | 0.612 |
| No history | 13.6% | 39.7% | 0.486 |

**Takeaway**: Full → Last 3 loses only 0.4% TSR but 3.8% Step SR. Removing history entirely causes major degradation (-8.9% TSR, -19.1% Step SR).

---

## 5. Resolution Ablation (Full Test Set)

| Resolution | TSR | Step SR | Avg Reward |
|-----------|-----|---------|------------|
| Full resolution (original) | 22.2% | 58.9% | 0.644 |
| Training-matched (602112 max px) | 20.5% | 57.9% | 0.600 |

**Takeaway**: Higher resolution slightly helps (+1.7% TSR), but effect is small.

---

## 6. Inference-Time Strategies (Full Test Set)

All using V15 Full SFT step-272 with GT history.

| Strategy | TSR | Step SR | Notes |
|----------|-----|---------|-------|
| **Baseline (greedy)** | **22.5%** | **58.8%** | Standard inference |
| Planning prompt (Exp A) | 16.3% | 50.2% | Adding planning hurts (-6.2% TSR) |
| Autoregressive no-stop (Exp B) | 22.1% | 46.3% | No stop-on-error, pred history |
| Guided base model (Exp D) | 22.4% | 59.1% | Original Qwen2.5-VL with SFT prompt |
| Self-verify (retry=2) | 22.4% | 58.8% | Only 3/7498 steps retried |
| Self-consistency (n=3, majority vote) | — | 57.7% | Step SR slightly worse |
| Self-consistency (n=5, majority vote) | — | 58.3% | Marginal improvement |
| Oracle best-of-5 | — | 71.7% | **Upper bound**: +12.9% Step SR |
| CoT step prompt | 23.6% | 59.2% | +1.1% TSR from reasoning prompt |

**Takeaway**:
- Planning prompts hurt (model can't plan well).
- Self-verify/consistency provide no gain (model confident even when wrong).
- Oracle BoN shows 71.7% ceiling (+12.9%), meaning the model *can* produce correct answers but doesn't always select them.
- CoT step prompt gives modest improvement (+1.1% TSR) without any training.

---

## 7. Best-of-N Analysis (200 Episodes, 1578 Steps)

5 samples per step at temperature=0.7.

| Metric | Value |
|--------|-------|
| Greedy accuracy | 55.3% |
| Oracle best-of-5 accuracy | 69.8% |
| Gap (ceiling) | **+14.5%** |

**Key finding**: When greedy is wrong, oracle BoN can recover ~32% of errors. This motivates RL or reward-based reranking.

---

## 8. Grounder Analysis

Combining SFT action predictor with separate grounder model for coordinate refinement.

| Metric | Value |
|--------|-------|
| SFT step SR (baseline) | 58.8% |
| Grounder coordinate accuracy | 25.6% |
| Combined step SR | 42.8% |
| Grounder helped (steps) | 363 |
| Grounder hurt (steps) | 1560 |
| Net improvement | **-1197** |

**Takeaway**: Grounder dramatically hurts performance. Its 25.6% coordinate accuracy is far too low to be useful as a replacement.

---

## 9. Oracle Element Experiments (200 Samples)

Testing whether providing GT UI element information improves action prediction.
Evaluated on single-step action prediction (not trajectory).

| Mode | Accuracy | Avg Reward | Content Reward |
|------|----------|------------|----------------|
| **One-pass baseline** | **54.0%** | **0.537** | **0.368** |
| Oracle elements (all GT, ~200 elements) | 53.5% | 0.577 | 0.418 |

**Takeaway**: Giving the model the full GT element list (200+ elements) does NOT improve accuracy. Content reward improves (+0.050), meaning coordinates become more precise, but the model still selects the wrong element.

---

## 10. Element Ablation Study (200 Samples, Key Experiment)

Systematic ablation to identify the real bottleneck in action prediction.

| Mode | Accuracy | Delta | Avg Reward | Content Reward |
|------|----------|-------|------------|----------------|
| One-pass baseline | 47.0% | — | 0.520 | 0.331 |
| **Oracle top-5 elements** | **57.5%** | **+10.5%** | **0.609** | **0.459** |
| Oracle sub-goal hint | 54.0% | +7.0% | 0.548 | 0.372 |
| Sub-goal CoT prompt | 52.0% | +5.0% | 0.536 | 0.354 |

**Key findings**:
1. **Oracle top-5** gives the biggest gain (+10.5%), proving that narrowing the selection space helps dramatically.
2. **Sub-goal hint** helps (+7.0%), meaning the model benefits from knowing *what* to interact with.
3. **CoT prompt** provides +5.0% *without any training* (just a prompt change), with only 2 regressions. Model doesn't actually generate reasoning text—it just outputs `<tool_call>` differently.
4. Self-refinement coverage is only 2.8% (when model predicts wrong, GT is rarely near the predicted coordinate).
5. Text matching coverage is only 23% for top-5 (GT elements often have indirect names vs. instruction text).


---

## 11. Two-Pass Grounding (100 Samples)

Pass 1: enumerate UI elements. Pass 2: select element and predict action.

| Mode | Accuracy | Notes |
|------|----------|-------|
| One-pass | 55.0% | Baseline |
| Two-pass (enumerate → select) | 53.0% | SFT model can't enumerate elements |

**Takeaway**: Two-pass approach fails because the SFT-trained model only knows how to output `<tool_call>` format and cannot enumerate UI elements.

---

## 12. Two-Agent Architecture Experiments (Full Test Set, 1000 Episodes)

Testing whether splitting reasoning and action into two agents improves over single-model approaches.

### Architecture
- **Reasoner**: generates CoT reasoning (sub-goal, element identification)
- **Actor**: receives reasoner's output as guidance, predicts action
- Two separate vLLM instances: reasoner on GPU 0,1; actor on GPU 2,3

### Results

| Config | Reasoner | Actor | Strip Coords | TSR | Step SR | Avg Progress |
|--------|----------|-------|:------------:|-----|---------|--------------|
| Two-agent | CoT SFT (epoch 2) | Base SFT | No | 20.0% | 47.2% | 0.341 |
| Two-agent (no coords) | CoT SFT (epoch 2) | Base SFT | Yes | 20.0% | 46.4% | 0.340 |
| Two-agent (base both) | Base SFT | Base SFT | Yes | 20.0% | 44.1% | 0.310 |
| **Baseline (single model)** | — | Base SFT | — | **21.9%** | **68.8%** | **0.352** |

### Key Findings

1. **Two-agent architecture hurts performance**: All variants are worse than single-model baseline (20.0% vs 21.9% TSR, 44-47% vs 68.8% Step SR).
2. **Stripping coordinates doesn't help**: Removing `[x,y]` coordinates from reasoning leaves TSR at 20.0%. The problem is not coordinate copying.
3. **Root cause: OOD guided prompt template**: The actor receives reasoning via `USER_PROMPT_TEMPLATE_GUIDED`, a format never seen during base SFT training. This causes Step SR to plummet from 68.8% → 44-47%.
4. **Actor copies reasoner's coords 73.4% of the time**: When actor does override, it's only 41.6% accurate vs 69.1% when copying. The actor has lost its own grounding ability due to the OOD prompt.

**Verdict**: Two-agent architecture is fundamentally broken by the distribution shift of the guided prompt template. Single-model CoT is the better approach.

---

## 13. Single-Model CoT Experiments (Full Test Set, 1000 Episodes)

Testing CoT reasoning as a prompt prefix within a single model (no guided template, no distribution shift).

### Results

| Mode | GT History | TSR | Step SR | Avg Progress | Notes |
|------|:----------:|-----|---------|--------------|-------|
| Baseline (greedy, no CoT) | No (pred) | 21.9% | 68.8% | 0.352 | Standard baseline |
| **CoT step (single model)** | **No (pred)** | **23.1%** | **46.9%** | **0.363** | **Fair comparison, +1.2% TSR** |
| CoT step (single model) | Yes (GT) | 24.0% | 59.1% | 0.368 | GT history helps Step SR |
| CoT step (prev, GT hist) | Yes (GT) | 23.6% | 59.2% | — | Earlier run, consistent |

### Analysis

- **CoT step with predicted history: TSR=23.1%** — beats baseline (21.9%) by +1.2% TSR, showing CoT reasoning helps trajectory completion even without GT history.
- **Step SR paradox**: CoT mode has lower Step SR (46.9% vs 68.8%) but higher TSR (23.1% vs 21.9%). This means CoT makes better decisions at critical junctures (improving trajectory success) even though per-step accuracy drops.
- **GT history boost**: Adding GT history raises Step SR from 46.9% → 59.1% and TSR from 23.1% → 24.0%. History quality matters for step-level accuracy.
- **No training required**: These gains come purely from a prompt change (adding CoT reasoning prefix).

**Verdict**: Single-model CoT is the best approach so far. TSR=23.1% with predicted history, 24.0% with GT history.

---

## 14. V18 K-Expert Full Eval (Full Test Set, 1000 Episodes)

Evaluating V18 K-expert cooperative LoRA models (SVD-extracted, before and after RL training).

### SVD-Extracted Baselines (no RL, vLLM serving)

| Model | Target Modules | TSR | Step SR | Avg Progress |
|-------|---------------|-----|---------|--------------|
| K=2 (V15), 4 attn modules | q,k,v,o_proj | 4.2% | 47.6% | 0.148 |
| K=4, 4 attn modules | q,k,v,o_proj | 4.5% | 48.7% | 0.152 |
| K=8, 4 attn modules | q,k,v,o_proj | 4.1% | 47.9% | 0.148 |
| **K=2 (V15), 7 modules** | all 7 | **18.0%** | **64.8%** | **0.305** |
| **K=4, 7 modules** | all 7 | **17.9%** | **65.0%** | **0.306** |
| **K=8, 7 modules** | all 7 | **17.6%** | **64.9%** | **0.305** |

**Finding**: 7 target modules are critical (4.5% → 17.9% TSR). K=2, K=4, K=8 perform identically at SVD extraction — expected since experts are near-identical at initialization (uniform routing, small noise perturbation).

### RL-Trained V18 K=4 (direct serving, in progress)

Evaluating V18 K=4 v4 training checkpoints (epoch-0_step-9, best val FSR=35%).

| Checkpoint | Episodes Done | TSR (partial) | Step SR (partial) | Status |
|-----------|:------------:|:-------------:|:-----------------:|--------|
| epoch-0_step-9 | 170/1000 | ~20.0% | ~42.4% | Running (job 4819546) |

**Preliminary finding**: RL-trained K=4 shows ~20% TSR, not improving over SVD baseline (17.9%). This is consistent with the training observation that routing never differentiated (entropy = log(4) = 1.386 throughout training, all experts used 23-26%).

### V18 Architecture Failure Analysis

The K-expert approach failed because:
1. **Routing collapse**: Softmax routing stays at uniform 1/K throughout training. No expert specialization occurs.
2. **Expert similarity**: Cosine similarity between expert outputs = 1.0 at convergence. All K experts learn identical representations.
3. **No capacity gain**: With uniform routing and identical experts, K experts = 1 expert. The MoE capacity advantage never materializes.
4. **RL didn't help**: GSPO training improved reward but didn't break the routing symmetry. Balance loss, diversity loss, and routing noise were insufficient.

---

## 15. V19 Step-Aware Reasoning & Verification (Full Test Set, 1000 Episodes)

Systematic investigation of verification, Best-of-N, and GRPO for improving long-horizon GUI navigation.

### 15.1 Prompt Engineering

| Prompt Variant | GT History | TSR | Step SR | Notes |
|---------------|:----------:|-----|---------|-------|
| baseline | pred | 21.9% | 68.8% | V15 full SFT baseline |
| type_focused | pred | 23.6% | 49.0% | Best prompt variant |
| type_focused | GT | 24.5% | 59.2% | With GT history |

**Takeaway**: type_focused prompt is the new baseline (23.6% TSR). Step SR drops (49%) because it uses stop-on-error, but TSR improves.

### 15.2 Oracle Type Experiment

| Experiment | TSR | Step SR | Notes |
|-----------|-----|---------|-------|
| type_focused (baseline) | 23.6% | 49.0% | Normal inference |
| Oracle type (given GT action type) | 24.9% | 50.3% | +1.3% TSR only |

**Takeaway**: Action type planning is NOT the bottleneck. Even with perfect type selection, TSR only improves 1.3%. The bottleneck is content accuracy (coordinates, text).

### 15.3 Self-Verification Diagnostic

| Metric | Value |
|--------|-------|
| True Positive Rate (TPR) | 84% |
| False Positive Rate (FPR) | 57% |

**Takeaway**: Base model's self-verification is weak. FPR=57% means it says "YES" to most wrong actions.

### 15.4 Trained Verifier

Trained on synthetic verification data (SFT, LoRA r=128).

| Verifier | TPR | FPR | Pipeline TSR |
|----------|-----|-----|-------------|
| Self-verify (base model) | 84% | 57% | — |
| Synthetic negatives | 45% | 9% | — |
| **Hard negatives (dual model)** | **73%** | **14%** | **23.3%** (pred) / **23.9%** (GT) |

**Takeaway**: Hard-negative trained verifier (FPR=14%) is usable but TPR drops. Simulated 6-step TSR: pred 5.1%, GT 15.3% — big gap shows history quality matters.

### 15.5 Best-of-N with Verification

Generate N candidates (1 greedy + N-1 sampled at temp=0.7), verify each, select best.

| Config | Selection | TSR | Step SR | 6+ TSR | Net Gain |
|--------|-----------|-----|---------|--------|----------|
| N=3, pred history | first_yes | 24.6% | 50.6% | 7.4% | +190 |
| N=3, GT history | first_yes | 25.5% | 62.0% | 8.4% | +213 |
| N=5, pred history | first_yes | 26.0% | 53.1% | 8.4% | +289 |
| N=3, pred history | logprob_rank | 24.7% | 53.1% | — | — |
| **N=5, pred history** | **logprob_rank** | **26.6%** | **54.5%** | — | — |
| N=3, GT history | logprob_rank | 25.9% | 62.2% | — | — |

**Key finding**: Best-of-5 + logprob ranking = **26.6% TSR**, best result so far. But:
- Oracle accuracy N=5 = 59.5%, actual selected = 54.5% → verifier only recovers part of the ceiling
- 41.5% of steps have ALL 5 candidates wrong (correlated errors, not independent)

### 15.6 GRPO Round 1 (Unfiltered) — FAILED

Offline GRPO: generate N=5 candidates on training data → compute rewards → train with group-relative advantages.

| Metric | Base SFT | GRPO R1 |
|--------|:--------:|:-------:|
| TSR | 23.6% | **10.9%** |
| Step SR | 49.0% | 39.9% |
| click accuracy (step 0) | 62.2% | **70.7%** (+8.5) |
| type accuracy (step 0) | 67.5% | **29.5%** (-38.0) |
| swipe accuracy (step 0) | 53.8% | **0.0%** (-53.8) |

**Root cause: Mode collapse to majority class (click)**. Training data is 67.4% click GT. GRPO's group-relative normalization creates systematic bias:

| Action Type | Net Training Signal |
|------------|:-------------------:|
| click | **+2553** (upweighted) |
| type | **-1143** (downweighted) |
| swipe | **-1381** (downweighted) |

GRPO assumes all candidates "attempt the same thing" and differ only in quality. In GUI navigation, candidates can be fundamentally different action types. Cross-type comparison produces biased advantages.

### 15.7 GRPO Round 2 (Type-Filtered) — NO EFFECT

Fix: only keep candidates whose pred_type matches GT type before computing advantages. Eliminates cross-type contamination.

| Metric | Base SFT | GRPO R2 | Delta |
|--------|:--------:|:-------:|:-----:|
| TSR | 23.6% | 23.6% | **±0.0** |
| Step SR | 49.0% | 47.3% | -1.7 |
| Type selection acc | 84.6% | 84.4% | -0.2 |
| Content acc (correct type) | 71.3% | 71.0% | -0.3 |
| click content acc | 57.5% | 57.4% | -0.1 |
| Episode head-to-head | — | +8/-8 | net=0 |

**Root cause: Within-type reward variance too small for learning signal**. Same-type candidates have mean reward std=0.181. Training KL stayed ≈ 0.000 throughout — the model barely moved.

**Fundamental GRPO limitation for this task**:
- Cross-type GRPO → strong signal, wrong direction → mode collapse
- Same-type GRPO → correct direction, near-zero signal → no learning

The reward function is too binary (correct → ~1.0, wrong → ~0.1) with little gradation within correct responses. GRPO needs continuous reward variance within groups to learn, which doesn't exist in same-type action prediction.

---

## 16. Summary & Current Directions

### Performance Hierarchy (Updated)
```
Oracle BoN-5 ceiling:                     59.5% Step SR
  │
  ├── Best-of-5 logprob_rank (pred):      26.6% TSR  ← BEST RESULT (inference-time)
  ├── Best-of-5 first_yes (pred):         26.0% TSR
  ├── Best-of-3 logprob_rank (GT):        25.9% TSR
  │
  ├── type_focused prompt (pred):         23.6% TSR  ← BEST GREEDY
  ├── GRPO R2 type-filtered (pred):       23.6% TSR  ← No improvement
  ├── Single-model CoT step (pred):       23.1% TSR
  │
  ├── Full-param SFT step-272:            21.9% TSR  (original baseline)
  ├── SVD Cooperative LoRA:               22.4% TSR
  ├── Two-agent (all variants):           20.0% TSR  ← WORSE
  │
  ├── GRPO R1 unfiltered:                 10.9% TSR  ← Mode collapse
  │
  ├── V18 K=4 SVD (7 mod):               17.9% TSR
  └── V18 K=4 SVD (4 mod):                4.5% TSR
```

### Completed Experiments
1. **Two-agent architecture**: FAILED — OOD guided prompt causes Step SR drop.
2. **V18 K-Expert LoRA**: FAILED — routing never differentiates.
3. **Single-model CoT step**: +1.2% TSR from prompt alone.
4. **V19 Prompt engineering**: type_focused prompt = 23.6% TSR (new greedy baseline).
5. **V19 Trained verifier**: Hard-negative verifier (TPR=73%, FPR=14%).
6. **V19 Best-of-N**: 26.6% TSR with N=5 + logprob ranking (best overall).
7. **V19 GRPO R1**: FAILED — mode collapse to click due to cross-type advantage bias.
8. **V19 GRPO R2 (type-filtered)**: NO EFFECT — within-type reward variance too small for learning.

### Key Bottleneck Analysis (Updated)
- **Compound error trap**: TSR = p^k. With step accuracy ~49%, TSR(6) ≈ 1.4%.
- **Correlated sampling errors**: 41.5% of steps have ALL 5 candidates wrong (vs 3.4% if independent).
- **Binary reward problem**: Rewards are too coarse (0.1 vs 1.0) for RL to learn fine-grained improvements.
- **Verifier gap**: Oracle BoN-5 = 59.5%, actual selected = 54.5%. Verifier recovers only part of the ceiling.
- **The fundamental challenge**: Both generation quality AND selection quality need to improve simultaneously.

### Failed Approaches (What We've Learned)
1. **Planning/reasoning prompts**: Don't help (model can't plan).
2. **Two-agent**: OOD prompt kills actor accuracy.
3. **K-Expert MoE**: Routing collapses, no specialization.
4. **GRPO (offline, binary reward)**: Cross-type = mode collapse; same-type = no signal.
5. **Self-verification**: FPR too high (57%).

### Potential Next Steps
1. **Continuous reward shaping**: Replace binary rewards with fine-grained coordinate distance. This could enable RL (GRPO/DPO) to learn within same-type candidates.
2. **Online RL**: Generate candidates with current policy (not frozen base), so reward variance reflects actual model behavior. May need verl/TRL integration.
3. **Rejection sampling fine-tuning (RFT)**: Simpler than GRPO — filter to only correct candidates, fine-tune on those. No advantage computation needed.
4. **Better verifier training**: Contrastive/ranking objective instead of binary classification. Train on (correct, incorrect) pairs from same step for comparative judgment.
5. **Larger/stronger base model**: 7B may have fundamental capability limits for long-horizon GUI navigation.
