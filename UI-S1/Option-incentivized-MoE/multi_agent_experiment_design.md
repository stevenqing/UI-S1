# Multi-Agent GUI Framework: Experiment Design & Validation Roadmap

> **Core Hypothesis**: Leverage the complementary strengths of an 80% grounding model (SFT v3) + all-round model (SFT v2) through a multi-agent framework (multi-sampling + selection + verification) to significantly boost GUI agent performance.

---

## 1. Motivation & Data

### 1.1 Available Models

| Model | Grounding | Action(Visual) | Notes |
|-------|:---------:|:--------------:|-------|
| GUI-360 Paper SFT | **82.30%** | 50.08% | Paper baseline |
| **Grounding SFT v3** | **79.48%** | 3.07% | Grounding specialist, action overfitting |
| **SFT v2** | 70.56% | **46.90%** | Best all-round model |
| LoRA v4 | 64.37% | 27.53% | RL initialization candidate |
| SVD LoRA r=256 | 68.12% | 47.00% | Close to SFT v2 |

### 1.2 Phase 0 Key Findings

| Finding | Data | Multi-Agent Implication |
|---------|------|------------------------|
| Multi-sampling works | Best-of-10 = 61% vs greedy 37.5% (LoRA v4) | Correct answers exist in sampling distribution → selection is key |
| DBSCAN underutilized | Cluster 44.5% vs Oracle 61.0% (gap=16.5pp) | Better selector can close this gap |
| Agreement rate is a strong signal | ≥0.9 → 83% accuracy | Usable for adaptive K and confidence routing |
| 82% divergence at step 0 | Exp 0.3/0.4 | Step 0 grounding accuracy is critical |
| Zero-shot verification insufficient | 59% acc, 18% recall | Need trained verification capability |
| Retry context marginal | +2pp | Simple retry insufficient, need better recovery |

### 1.3 Core Hypotheses

| Hypothesis | Validation Experiment |
|------------|----------------------|
| H1: SFT v3 80% accuracy + multi-sampling → 90%+ | Exp 1.1 |
| H2: Better coordinates directly improve action prediction | Exp 1.3 |
| H3: Dual-model combination > either single model | Exp 1.5 |
| H4: SFT v2 and SFT v3 make complementary errors | Exp 1.6 |
| H5: Agreement rate remains effective on stronger grounder | Exp 1.7 (from 1.1 data) |
| H6: SFT v3 grounding is not prompt-specific | Exp 1.4 |

---

## 2. Phase 1 Early Validation Experiments

### Experiment 1.1: SFT v3 Multi-Sample Upper Bound

**Question**: 80% grounder + multi-sampling → how high?

**Method**:
1. Start vLLM with Grounding SFT v3 final checkpoint
2. On GUI-360 test grounding eval:
   - Greedy (K=1, temp=0.0): verify reproduction of 79.48%
   - K=5, temp=0.7: DBSCAN clustering + best-of-5
   - K=10, temp=0.7: DBSCAN clustering + best-of-10
3. Record: greedy_acc, cluster_acc, best_of_k_acc, agreement_rate segmented analysis

**Success criteria**: K=5 cluster accuracy > 83% (≥3.5pp over greedy)

**Script**: `scripts/exp1/exp1_1_sft_v3_multisample.py`

---

### Experiment 1.2: SFT v2 Multi-Sample Upper Bound (Comparison)

**Question**: Same multi-sampling on SFT v2—how does it compare?

**Method**: Same as 1.1 but with SFT v2 checkpoint

**Key comparison**:
- SFT v3 K=5 cluster vs SFT v2 K=5 cluster
- If SFT v3 K=5 > SFT v2 K=10 → base accuracy matters more than sampling count

**Script**: `scripts/exp1/exp1_2_sft_v2_multisample.py`

---

### Experiment 1.3: Oracle Coordinate Replacement ★ (Go/No-Go Gate)

**Question**: If coordinates were perfect, how much would action prediction improve?

**This is the go/no-go gate for the entire approach.**

**Method**:
1. On GUI-360 test action prediction eval (Visual):
   - A. SFT v2 original prediction (baseline)
   - B. SFT v2 predicted action, coordinate replaced with SFT v3 prediction
   - C. SFT v2 predicted action, coordinate replaced with ground truth (oracle)
2. Evaluate: function_match, args_match, step_success

**Expected**:

| Coordinate Source | args_match |
|---|:---:|
| A. SFT v2 original | ~17% (known) |
| B. SFT v3 coord | ~22-25% |
| C. GT coord (oracle) | ~35-40% |

**Success criteria**:
- Oracle (C) args_match > baseline (A) by ≥10pp → coordinate IS the bottleneck
- SFT v3 coord (B) > baseline (A) → SFT v3 grounding is practically useful

**Script**: `scripts/exp1/exp1_3_oracle_coord_replacement.py`

---

### Experiment 1.4: SFT v3 Grounding Under Action Prompt

**Question**: Is SFT v3's 80% prompt-specific or a general capability?

**Method**:
1. Call SFT v3 with action prediction prompt (tool_call format)
2. Extract coordinates from tool_call output
3. Evaluate using grounding criteria (coord in bbox)

**Success criteria**:
- \>70% → universal capability, use directly in action pipeline
- <50% → prompt-specific, need dedicated grounding prompt bridge

**Script**: `scripts/exp1/exp1_4_sft_v3_action_prompt.py`

---

### Experiment 1.5: Dual-Model Combination vs Single Model

**Question**: SFT v2 action type + SFT v3 coordinate > either model alone?

**Method**:

| Condition | Action Type Source | Coordinate Source |
|---|---|---|
| A. Baseline | SFT v2 | SFT v2 |
| B. SFT v3 only | SFT v3 | SFT v3 |
| C. Dual (K=1) | SFT v2 | SFT v3 greedy |
| D. Dual (K=5) | SFT v2 | SFT v3 K=5 cluster |

**Success criteria**: D args_match > A by ≥5pp

**Script**: `scripts/exp1/exp1_5_dual_model_eval.py`

---

### Experiment 1.6: Error Diversity (SFT v2 vs SFT v3)

**Question**: Are errors complementary or overlapping?

**Method**: Per-sample correctness comparison on grounding eval

| Situation | Expected |
|-----------|:--------:|
| Both correct | ~65% |
| V3 correct, V2 wrong | ~15% |
| V2 correct, V3 wrong | ~5% |
| Both wrong | ~15% |

**Key metric**: Oracle ensemble accuracy = at least one correct ≈ **85%+**

**Success criteria**: Oracle ensemble > max(V2, V3) by ≥5pp

**Script**: `scripts/exp1/exp1_6_error_diversity.py` (CPU only, from existing results)

---

### Experiment 1.7: Agreement Rate Calibration on SFT v3

**Question**: Does agreement rate remain a valid confidence signal on stronger grounder?

**Method**: From Exp 1.1 K=10 data, segment accuracy by agreement rate

| Agreement Rate | LoRA v4 (known) | SFT v3 (expected) |
|:-:|:-:|:-:|
| ≥ 0.9 | 83% | ~95% |
| 0.5-0.9 | 59% | ~82% |
| < 0.5 | 23% | ~55% |

**Success criteria**: High agreement region accuracy > 90%

**No additional script** — analyzed directly from Exp 1.1 output

---

## 3. Execution Plan

### 3.1 Timeline

```
Day 1-2: Exp 1.1 (SFT v3 multi-sample) + Exp 1.6 (error diversity, CPU)
         → sbatch scripts/exp1/run_exp1_grounding.slurm
Day 2-3: Exp 1.2 (SFT v2 multi-sample) + Exp 1.7 (calibration, from 1.1 data)
         → included in run_exp1_grounding.slurm
Day 3-4: Exp 1.3 + 1.4 + 1.5 (dual-model, combined Slurm)
         → sbatch scripts/exp1/run_exp1_dual_model.slurm
Day 5:   Aggregate results, Go/No-Go decision
```

### 3.2 Go/No-Go Decision Matrix

| Experiment | ✅ Go | ❌ No-Go | Fallback |
|-----------|-------|---------|----------|
| 1.1 | SFT v3 K=5 cluster > 83% | < 80% | Use SFT v2 as grounder |
| 1.3 | Oracle coord improves ≥10pp | < 5pp | Problem isn't coordinates → analyze other bottlenecks |
| 1.4 | Action prompt grounding >70% | < 50% | Use dedicated grounding prompt as bridge |
| 1.5 | Dual model > single model ≥5pp | No improvement | Multi-agent grounding not viable |
| 1.6 | Oracle ensemble > max ≥5pp | Errors overlap | Ensemble value limited |

---

## 4. Post-Validation Multi-Agent Roadmap

### Phase 2 (2 weeks): Dual-Model Inference Pipeline

```
SFT v2 (action agent) ──→ action_type + reasoning
                              │
SFT v3 (grounder × K) ──→ K coordinate candidates
                              │
              DBSCAN / Selector ──→ best coordinate
                              │
              Combine: action_type + coordinate ──→ execute
```

- Adaptive K: agreement ≥0.9 → K=1; <0.5 → K=10
- Evaluate: args_match and TSR

### Phase 3 (3 weeks): MoE Expert Differentiation Training

- Expert 0: SFT v2 LoRA init (planning/action)
- Expert 1: SFT v3 delta init (grounding)
- Phase-based router
- Consensus RL reward + Step-0 amplification

### Phase 4 (4 weeks): Verification + Recovery

- Verification SFT (recall 18% → 70%+)
- Full verify-recover loop
- End-to-end trajectory evaluation

---

## 5. Expected Outcomes

| Phase | Method | Grounding | Action(V) | args_match | TSR |
|-------|--------|:---------:|:---------:|:----------:|:---:|
| Current | SFT v2 greedy | 70.56% | 46.90% | 17.07% | 16.21% |
| P1 validation | SFT v3 + K=5 | ~85% | — | — | — |
| P2 | v2 action + v3 grounding K=5 | ~85% | ~47% | ~26% | ~30% |
| P3 | MoE + consensus RL | ~87% | ~50% | ~30% | ~35% |
| P4 | + Verify + Recovery | ~90% | ~52% | ~35% | ~42% |

---

## 6. File Index

### Phase 1 Experiment Scripts

| File | Purpose |
|------|---------|
| `scripts/exp1/exp1_1_sft_v3_multisample.py` | SFT v3 multi-sample grounding eval |
| `scripts/exp1/exp1_2_sft_v2_multisample.py` | SFT v2 multi-sample grounding eval |
| `scripts/exp1/exp1_3_oracle_coord_replacement.py` | Oracle coord replacement (go/no-go) |
| `scripts/exp1/exp1_4_sft_v3_action_prompt.py` | Cross-prompt generalization test |
| `scripts/exp1/exp1_5_dual_model_eval.py` | Dual-model combination eval |
| `scripts/exp1/exp1_6_error_diversity.py` | Error diversity analysis (CPU only) |
| `scripts/exp1/run_exp1_grounding.slurm` | Slurm: Exp 1.1 + 1.2 + 1.6 |
| `scripts/exp1/run_exp1_dual_model.slurm` | Slurm: Exp 1.3 + 1.4 + 1.5 (2 nodes) |

### Model Checkpoints

| Model | Path |
|-------|------|
| SFT v3 Final (Grounding) | `train_GUI_360/llamafactory/output/gui360_full_sft_v3_grounding` |
| SFT v2 (All-round) | `train_GUI_360/llamafactory/output/gui360_full_sft_v2` |
| LoRA v4 ckpt354 | `train_GUI_360/llamafactory/output/gui360_lora_sft_v4/checkpoint-354` |
| Base Qwen2.5-VL-7B | `checkpoints/Qwen2.5-VL-7B-Instruct` |

### Shared Utilities (from Phase 0)

| Module | Key Functions |
|--------|--------------|
| `scripts/exp0/data_utils.py` | `PARQUET_EVAL_PATH`, `DATASET_ROOT`, `is_coord_in_bbox`, `load_trajectory` |
| `scripts/exp0/exp0_1_uncertainty_analysis.py` | `call_model_k_times`, `cluster_coordinates`, `parse_tool_call`, `extract_coordinate`, `evaluate_coord` |
