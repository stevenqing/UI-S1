# GUI-360 Balanced Dataset — All Evaluation Results

_Last updated: 2026-05-15_

Test set: `gui360_test_1000_balanced.jsonl` (1000 episodes, stop-on-error, match_threshold=0.5)

Base model: **Qwen2.5-VL-7B-Instruct**

Training data: **balanced 2000-episode** (17,264 steps) unless noted otherwise.

---

## Complete Ranking

| # | Method | TSR | StepSR | Progress | Params | Notes |
|---|--------|----:|-------:|--------:|--------|-------|
| 1 | **Full-param SFT (step-250)** | **22.2%** | **69.3%** | **35.3%** | 7.6B | ~epoch 3.7, LLaMA-Factory+ZeRO-3 |
| 2 | Full-param SFT (step-150) | 22.1% | 67.0% | 34.5% | 7.6B | ~epoch 2.2 |
| 3 | Full-param SFT (step-272) | 21.9% | 68.8% | 35.2% | 7.6B | epoch 4 |
| 4 | Full-param SFT (step-200) | 21.0% | 68.1% | 34.2% | 7.6B | ~epoch 3.0 |
| 5 | **V15 Cooperative RL from SVD (step-25)** | **20.8%** | **67.9%** | **34.5%** | ~142M | GSPO, from peft_balanced_coop |
| 6 | Full-param SFT (step-100) | 19.0% | 64.3% | 31.1% | 7.6B | ~epoch 1.5 |
| 7 | **PEFT Balanced Cooperative r=128 (SVD init)** | **18.6%** | **65.3%** | **31.0%** | ~67M | SVD from full SFT step-250 |
| 8 | PEFT Balanced Standard r=128 (SVD init) | 18.1% | 65.2% | 30.8% | ~67M | SVD from full SFT step-250 |
| 9 | SVD Standard LoRA (all modules) | 17.9% | 65.0% | 30.5% | ~67M | SVD extracted, no fine-tune |
| 10 | gui_action (reference, 97K data) | 14.4% | 65.2% | 30.6% | 7.6B | full-param SFT, 5x more data |
| 11 | V15 Cooperative SFT epoch-2 | 10.5% | 54.0% | 20.9% | ~67M | from scratch, r=128 T=2 |
| 12 | PEFT Standard r=128 | 7.9% | 56.3% | 23.4% | ~67M | from scratch |
| 13 | Task Arithmetic (0.3/0.7) | 7.2% | 56.4% | 22.8% | 7.6B | gui_grounding+gui_action merge |
| 14 | V15 Cooperative SFT epoch-1 | 6.6% | 48.9% | 16.0% | ~67M | from scratch |
| 15 | Standard LoRA r=256 SFT | 6.4% | 48.8% | 18.1% | ~134M | 4 epochs, from scratch |
| 16 | Average merge (0.5/0.5) | 5.1% | 51.8% | 18.4% | 7.6B | gui_grounding+gui_action merge |
| 17 | Task Arithmetic (0.5/0.5) | 4.9% | 52.0% | 18.5% | 7.6B | gui_grounding+gui_action merge |
| 18 | SVD extracted (attn only) | 4.3% | 47.6% | 14.9% | ~17M | attn modules only |
| 19 | SVD Standard LoRA (attn only) | 4.3% | 48.2% | 14.9% | ~17M | attn modules only |
| 20 | V15 Cooperative SFT epoch-0 | 3.8% | 44.3% | 14.1% | ~67M | from scratch |
| 21 | TIES-Merging (k=20, lambda=1.0) | 3.6% | 39.2% | 13.5% | 7.6B | gui_grounding+gui_action merge |
| 22 | base_model (no fine-tuning) | 2.4% | 26.4% | 8.0% | — | zero-shot |
| 23 | V13 Cooperative RL epoch-0 (from base) | 0.1% | 8.1% | 1.6% | ~132M | corrected eval |
| 24 | V13 Cooperative RL epoch-1 (from base) | 0.1% | 42.2% | 11.0% | ~132M | corrected eval |
| 25 | V13 Cooperative RL epoch-2 (from base) | 0.0% | 7.0% | 1.3% | ~132M | corrected eval |
| 26 | SVD Cooperative (all modules) | 0.0% | 0.0% | 0.0% | ~67M | broken — routing mismatch |

---

## Detailed Results by Category

### 1. Full-Parameter SFT (balanced 2K data, LLaMA-Factory + ZeRO-3)

| Checkpoint | ~Epoch | TSR | StepSR | Progress |
|---|---|---:|---:|---:|
| step-100 | 1.5 | 19.0% | 64.3% | 31.1% |
| step-150 | 2.2 | 22.1% | 67.0% | 34.5% |
| step-200 | 3.0 | 21.0% | 68.1% | 34.2% |
| **step-250** | **3.7** | **22.2%** | **69.3%** | **35.3%** |
| step-272 | 4.0 | 21.9% | 68.8% | 35.2% |

Best: **step-250 (TSR=22.2%)**. Saturates after epoch ~2.

### 2. SVD-Extracted LoRA (from full SFT step-250, no RL/SFT fine-tune)

| Model | TSR | StepSR | Progress |
|---|---:|---:|---:|
| SVD standard LoRA (all modules, r=128) | 17.9% | 65.0% | 30.5% |
| SVD extracted (attn only) | 4.3% | 47.6% | 14.9% |
| SVD standard LoRA (attn only) | 4.3% | 48.2% | 14.9% |
| SVD cooperative (all modules) | 0.0% | 0.0% | 0.0% |

Key: all-module extraction recovers 80% of full SFT. Attn-only loses too much.

### 3. PEFT SFT (SVD-initialized, from full SFT step-250)

SVD-extract → convert to LoRA format → SFT fine-tune on balanced 2K data.

| Model | TSR | StepSR | Progress |
|---|---:|---:|---:|
| **PEFT Balanced Cooperative r=128** | **18.6%** | **65.3%** | **31.0%** |
| PEFT Balanced Standard r=128 | 18.1% | 65.2% | 30.8% |

Cooperative +0.5pp over standard with same param count.

### 4. LoRA SFT from Scratch (balanced 2K data, no SVD init)

| Model | TSR | StepSR | Progress |
|---|---:|---:|---:|
| V15 Cooperative SFT epoch-2 (r=128, T=2) | 10.5% | 54.0% | 20.9% |
| V15 Cooperative SFT epoch-1 | 6.6% | 48.9% | 16.0% |
| PEFT Standard r=128 | 7.9% | 56.3% | 23.4% |
| Standard LoRA r=256 | 6.4% | 48.8% | 18.1% |
| V15 Cooperative SFT epoch-0 | 3.8% | 44.3% | 14.1% |

Without SVD init, LoRA SFT caps at ~10% TSR (cooperative) or ~8% (standard).

### 5. Cooperative RL

| Model | TSR | StepSR | Progress | Init |
|---|---:|---:|---:|---|
| **V15 Cooperative RL (step-25)** | **20.8%** | **67.9%** | **34.5%** | SVD→PEFT SFT (18.6%) |
| V13 Cooperative RL epoch-0 | 0.1% | 8.1% | 1.6% | base model (cold start) |
| V13 Cooperative RL epoch-1 | 0.1% | 42.2% | 11.0% | base model (cold start) |
| V13 Cooperative RL epoch-2 | 0.0% | 7.0% | 1.3% | base model (cold start) |

SVD warm-start is critical: 20.8% vs 0.1% TSR.

### 6. Task Vector Merging (full-param, gui_grounding + gui_action)

Source models: `gui_grounding` (97K steps) + `gui_action` (97K steps).

| Method | Config | TSR | StepSR | Progress |
|---|---|---:|---:|---:|
| Task Arithmetic | alpha=0.3, beta=0.7 | **7.2%** | **56.4%** | **22.8%** |
| Average | 0.5 / 0.5 | 5.1% | 51.8% | 18.4% |
| Task Arithmetic | alpha=0.5, beta=0.5 | 4.9% | 52.0% | 18.5% |
| TIES-Merging | k=20, lambda=1.0 | 3.6% | 39.2% | 13.5% |
| DARE | all configs | — | — | — |

DARE and other TIES/TA configs timed out (2h wall clock).

### 7. Reference Baselines

| Model | TSR | StepSR | Progress | Notes |
|---|---:|---:|---:|---|
| gui_action (full SFT, 97K data) | 14.4% | 65.2% | 30.6% | 5x more training data |
| base_model (Qwen2.5-VL-7B) | 2.4% | 26.4% | 8.0% | zero-shot |

---

## Key Findings

1. **Full-param SFT (22.2%) is the upper bound** with 2K balanced data
2. **SVD extraction recovers most performance**: all-module SVD LoRA = 17.9%, PEFT SFT = 18.6% (84% of full SFT)
3. **RL adds +2.2pp** on top of PEFT SFT (18.6% → 20.8%), reaching 94% of full SFT
4. **SVD init is critical for LoRA**: without it, best LoRA SFT = 10.5% vs 18.6% with SVD
5. **Cooperative > standard** consistently: +0.5pp (PEFT SFT), +4.1pp (from-scratch SFT at epoch-2)
6. **Task vector merging fails** (best 7.2%) — weight-space interference between heterogeneous tasks
7. **RL from base fails** (V13: 0.1%) — cold-start RL cannot learn GUI policy
8. **2K balanced data > 97K unbalanced**: full SFT 22.2% surpasses gui_action's 14.4%

---

## V15 RL Analysis: What RL Improves over SVD SFT

### Episode-Level Comparison

V15 RL (step-25) vs its starting checkpoint `peft_balanced_cooperative_r128`:

| | SFT (before) | RL (after) |
|---|---|---|
| TSR | 18.6% | 20.8% |
| Overlap | 157 both succeed | |
| RL-only wins | | +51 episodes |
| SFT-only wins | +29 episodes | |
| Net gain | | **+22 episodes** |

### Type Action Recognition (main RL improvement)

| Metric | SFT (before) | RL (after) | Gap |
|---|---|---|---|
| GT=type correct rate | 70.2% (438/624) | 79.0% (492/623) | **+8.8pp** |
| GT=type misclassified as click | 173 | 119 | **-54** |
| GT=click correct rate | 92.7% | 90.4% | -2.3pp |
| Overall type accuracy | 85.4% | 86.4% | +1.0pp |

### Coordinate Precision: Unchanged

| Metric | SFT | RL |
|---|---|---|
| Click reward (correct type) | mean=0.797, >=0.5: 77.1% | mean=0.801, >=0.5: 78.7% |

### TSR by Episode Complexity

| Steps | SFT | RL | Gap |
|---|---|---|---|
| 1 | 52.9% | **64.7%** | **+11.8pp** |
| 2-3 | 26.2% | 32.3% | +6.1pp |
| 4-5 | 23.2% | 23.2% | 0 |
| 6-8 | 10.6% | 9.5% | -1.1pp |
| 9-15 | 3.8% | 2.2% | -1.6pp |

RL helps short tasks (type vs click boundary), neutral/worse on long tasks.

---

## Appendix: V13 RL Corrected Eval

Previous V13 eval used wrong prompt format. Corrected results:

| Checkpoint | TSR (old) | TSR (corrected) | Delta |
|---|---|---|---|
| epoch-0 | 8.0% | 0.1% | -7.9pp |
| epoch-1 | 0.0% | 0.1% | +0.1pp |
| epoch-2 | 0.0% | 0.0% | +0.0pp |
