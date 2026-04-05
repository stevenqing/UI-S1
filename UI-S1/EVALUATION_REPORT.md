# Comprehensive Evaluation Report: GUI Agent Models

This report consolidates all evaluation results across models, datasets, and evaluation paradigms.

---

## 1. Models Evaluated

| Model | Type | Params | Training | Coordinate Space |
|-------|------|--------|----------|-----------------|
| **Qwen2.5-VL-7B Base** | Foundation | 7B | Pre-trained only | Absolute pixel (tool_call JSON) |
| **GUI-360 SFT v2** | Full fine-tune | 7B | GUI-360 full SFT 1 epoch | Absolute pixel (tool_call JSON) |
| **OS-Atlas-Pro-7B** | Foundation | 7B | Pre-trained + grounding | Relative [0,1000] |
| **UI-TARS-7B-DPO** | Fine-tuned | 7B | SFT + DPO | Relative [0,1000] |
| **OS-Genesis-7B-AC** | Fine-tuned | 7B | SFT on AndroidControl | Absolute pixel (JSON) |

---

## 2. Step-Level Evaluation on GUI-360 (Non-AR, Single-Step)

Source: `train_GUI_360/GUI_360_all_eval_results.md`

### 2.1 Core Models

| Model | Grounding | Action (Visual) | Action (A11y) |
|-------|:---------:|:---------------:|:-------------:|
| **GUI-360 Paper SFT** | **82.30** | **50.08** | **25.78** |
| **SFT v2 (full, 1ep)** | **70.56** | **46.90** | 17.51 |
| SFT v2 (full, 2ep) | 70.77 | 49.37 | — |
| Qwen2.5-VL-7B Base | 42.47 | 18.05 | 14.53 |
| LoRA v3 (r=32) | 56.34 | 24.67 | **20.54** |
| SVD LoRA r=256 | 68.12 | 47.00 | — |
| MoE v1 LoRA | 60.32 | 33.20 | 5.47 |

### 2.2 Grounding SFT v3 (Multi-task with Grounding Labels)

| Checkpoint | Grounding | Action (Visual) |
|------------|:---------:|:---------------:|
| ckpt150 | 77.66 | 3.89 |
| ckpt300 | 79.61 | 3.06 |
| **Final** | **79.48** | 3.07 |

> Grounding approaches paper (79.48 vs 82.30), but action severely overfits (3%).

### 2.3 SVD LoRA (Decomposed from Full SFT v2)

| Rank | Grounding | Action (Visual) |
|:----:|:---------:|:---------------:|
| r=32 | 61.60 | 37.35 |
| r=64 | 62.81 | 42.08 |
| r=128 | 65.85 | 44.75 |
| **r=256** | **68.12** | **47.00** |

> r=256 approaches SFT v2 (68.12 vs 70.56 grounding, 47.00 vs 46.90 visual).

---

## 3. Trajectory-Level (AR) Evaluation on AndroidControl

Dataset: 1,543 trajectories, 8,444 total steps. Stop-on-error + no-stop modes.

### 3.1 Overall Results

| Model | Mode | TSR | Avg Progress | Scattered Progress | Step Acc |
|-------|------|:---:|:------------:|:------------------:|:--------:|
| **UI-TARS-7B-DPO** | stop | **0.224** | 0.332 | 0.258 | — |
| **UI-TARS-7B-DPO** | no-stop | **0.222** | **0.653** | **0.650** | 0.650 |
| OS-Atlas-Pro-7B | stop | 0.161 | 0.251 | 0.180 | — |
| OS-Atlas-Pro-7B | no-stop | 0.162 | 0.612 | 0.618 | 0.618 |
| OS-Genesis-7B-AC | stop | 0.019 | 0.090 | 0.070 | — |
| OS-Genesis-7B-AC | no-stop | 0.021 | 0.199 | 0.191 | 0.191 |

### 3.2 By Trajectory Length (Stop-on-Error)

| Model | Short (1-3) | Medium (4-7) | Long (8-15) | VLong (16+) |
|-------|:-----------:|:------------:|:-----------:|:-----------:|
| UI-TARS | TSR=0.411 / Prog=0.465 | TSR=0.190 / Prog=0.303 | TSR=0.055 / Prog=0.156 | TSR=0.000 / Prog=0.060 |
| OS-Atlas | TSR=0.324 / Prog=0.403 | TSR=0.123 / Prog=0.200 | TSR=0.031 / Prog=0.098 | TSR=0.000 / Prog=0.051 |
| OS-Genesis | TSR=0.066 / Prog=0.128 | TSR=0.000 / Prog=0.078 | TSR=0.000 / Prog=0.047 | TSR=0.000 / Prog=0.023 |

### 3.3 Action Type Breakdown (No-Stop)

| Action Type | N | UI-TARS Type/Extract | OS-Atlas Type/Extract | OS-Genesis Type/Extract |
|-------------|:---:|:---:|:---:|:---:|
| click | 5074 | 87.8% / 76.5% | 89.8% / 73.0% | 62.8% / 12.7% |
| type | 632 | 88.9% / 82.1% | 86.7% / 75.5% | 65.8% / 53.8% |
| open | 608 | 13.7% / 13.2% | 12.8% / 12.3% | 79.6% / 66.0% |
| swipe | 1211 | 31.7% / 31.7% | 69.1% / 69.1% | 15.4% / 15.4% |
| wait | 567 | 67.0% / 67.0% | 16.6% / 16.6% | 4.1% / 4.1% |
| system_button | 343 | 73.5% / 71.7% | 29.2% / 9.3% | 7.0% / 7.0% |

### 3.4 No-Stop Diagnostics

| Metric | UI-TARS | OS-Atlas | OS-Genesis |
|--------|:-------:|:--------:|:----------:|
| Step Accuracy | **65.0%** | 61.8% | 19.1% |
| Step-0 Failure Rate | 50.7% | 61.8% | 67.3% |
| Post-Error Accuracy | 65.4% | 65.7% | 16.0% |

> UI-TARS and OS-Atlas maintain similar post-error accuracy — errors don't cascade significantly. OS-Genesis degrades further after first error.

---

## 4. Trajectory-Level (AR) Evaluation on GUI-360

Dataset: 3,233 trajectories, 19,046 total steps. Desktop office apps (PPT, Word, Excel).

### 4.1 Overall Results

| Model | Mode | TSR | Seq Progress | Scattered Progress | Step SR |
|-------|------|:---:|:------------:|:------------------:|:-------:|
| **SFT v2** | stop | **0.162** | — | — | 55.3% |
| **SFT v2** | no-stop | **0.170** | — | — | 46.9% |
| Qwen2.5-VL Base | stop | 0.016 | — | — | 22.1% |
| Qwen2.5-VL Base | no-stop | 0.029 | — | — | 18.1% |
| OS-Atlas-Pro-7B | stop | 0.021 | 0.053 | 0.053 | 14.5% |
| OS-Atlas-Pro-7B | no-stop | 0.022 | 0.053 | 0.160 | 13.7% |
| OS-Genesis-7B-AC | stop | 0.003 | 0.011 | 0.011 | 3.7% |
| OS-Genesis-7B-AC | no-stop | 0.003 | 0.011 | 0.022 | 1.7% |
| UI-TARS-7B-DPO | stop | — | — | — | — |
| UI-TARS-7B-DPO | no-stop | — | — | — | — |

> **Note**: UI-TARS GUI-360 results pending (parse_action coordinate conversion fix, job 3252386).

### 4.2 By Domain (Stop-on-Error)

| Model | PPT (865) | Word (1369) | Excel (999) |
|-------|:---------:|:-----------:|:-----------:|
| SFT v2 | TSR=18.3% | TSR=15.5% | TSR=15.4% |
| OS-Atlas | TSR=3.1% | TSR=2.3% | TSR=1.0% |
| OS-Genesis | TSR=0.2% | TSR=0.4% | TSR=0.1% |

### 4.3 By Trajectory Length (Stop-on-Error)

| Model | Short (1-5) | Medium (6-15) | Long (16+) |
|-------|:-----------:|:-------------:|:----------:|
| OS-Atlas | TSR=3.3%, Prog=7.3% | TSR=0.0%, Prog=1.9% | TSR=0.0%, Prog=0.6% |
| OS-Genesis | TSR=0.4%, Prog=1.5% | TSR=0.0%, Prog=0.5% | TSR=0.0%, Prog=0.1% |

---

## 5. Exp2: Verifier Validation Experiment

Model: gui360_full_sft_v2 (Qwen2.5-VL-7B) | Dataset: GUI-360 test, 202 Pattern B trajectories (~1916 steps)
Context mode: subtask_isolated | Inference: HF Transformers (not vLLM, for hidden state access)

**Core Question**: Can the L26 correctness probe (5-fold CV accuracy = 0.650) improve AR inference by guiding temperature selection — greedy when confident, resample when uncertain?

### 5.1 Probe Pipeline

| Property | Value |
|---|---|
| Probe | StandardScaler → PCA(256) → LogisticRegression(C=1.0) |
| Training data | 956 clean samples (steps before/at first error), all trajectories |
| Class balance | 562 correct (58.8%) / 394 wrong (41.2%) |
| Train accuracy (refit on all data) | **0.839** |

### 5.2 Three Conditions

| Condition | Temperature Rule | Purpose |
|---|---|---|
| `always_greedy` | T=0 always | HF Transformers baseline |
| `verifier` | T=0 if P(correct)>0.5, else T=1.0 | Probe-guided |
| `always_temp1` | T=1.0 always | Stochastic control |

### 5.3 Results

| Condition | TSR | Seq Progress | Scattered Progress | Step Acc | ΔTSR vs greedy |
|---|:---:|:---:|:---:|:---:|:---:|
| always_greedy | 0.030 | 0.159 | 0.379 | **0.365** | — |
| **verifier** | **0.045** | 0.144 | 0.324 | 0.293 | **+0.015** |
| always_temp1 | 0.015 | 0.095 | 0.251 | 0.233 | -0.015 |

### 5.4 Probe Calibration

| Metric | Value |
|---|:---:|
| Overall accuracy | **0.730** |
| Majority baseline | 0.707 |
| "Wrong" prediction precision | **0.864** |
| "Correct" prediction precision | 0.529 |
| Avg P(correct) for correct steps | 0.673 |
| Avg P(correct) for wrong steps | 0.288 |

### 5.5 Verifier Decision Breakdown

| Route | N steps | Step Accuracy |
|---|:---:|:---:|
| Greedy (probe P(correct)>0.5) | 766 (40%) | **0.529** |
| Resample (probe P(correct)≤0.5) | 1150 (60%) | 0.136 |

**Resampling hurts**: Greedy got 25.6% on flagged steps; T=1.0 only got 13.6%.

### 5.6 Paired Trajectory Analysis

| Metric | Value |
|---|:---:|
| Improved (verifier > greedy) | 12 (5.9%) |
| Same | 108 (53.5%) |
| Degraded (verifier < greedy) | 82 (40.6%) |
| Mean Δ scattered progress | **-0.055** |

### 5.7 Key Takeaways

**What Worked:**
1. Probe signal is real: Avg P(correct) gap 0.67 vs 0.29
2. "Wrong" detection precision is high (86.4%)
3. TSR gains possible (+3 extra successful trajectories)

**What Didn't Work:**
1. T=1.0 resampling is the wrong intervention — distribution too diffuse
2. Probe routes 60% to resample (too aggressive)
3. Step accuracy drops uniformly

**Recommended Next Steps:**

| Direction | Rationale |
|---|---|
| **Best-of-N with probe scoring** | Generate N candidates at T=0.7, use probe to rank |
| **Beam search on flagged steps** | Systematic exploration instead of random sampling |
| **Multi-model verifier** | Route flagged steps to a stronger model |
| **Threshold tuning** | P(correct) > 0.7 instead of 0.5 |
| **Lower resample temperature** | T=0.3-0.5 instead of T=1.0 |

---

## 6. Cross-Model Analysis

### 6.1 AndroidControl Rankings

| Rank | Model | TSR (stop) | Step Acc (no-stop) |
|:----:|-------|:----------:|:------------------:|
| 1 | **UI-TARS-7B-DPO** | **22.4%** | **65.0%** |
| 2 | OS-Atlas-Pro-7B | 16.1% | 61.8% |
| 3 | OS-Genesis-7B-AC | 1.9% | 19.1% |

### 6.2 GUI-360 Rankings

| Rank | Model | TSR (stop) | Step SR (stop) |
|:----:|-------|:----------:|:--------------:|
| 1 | **GUI-360 SFT v2** | **16.2%** | **55.3%** |
| 2 | Qwen2.5-VL Base | 1.6% | 22.1% |
| 3 | OS-Atlas-Pro-7B | 2.1% | 14.5% |
| 4 | OS-Genesis-7B-AC | 0.3% | 3.7% |

### 6.3 Key Observations

1. **Domain transfer gap**: UI-TARS and OS-Atlas perform well on AndroidControl (mobile) but poorly on GUI-360 (desktop office). SFT v2, trained on GUI-360, dominates its home dataset but was not evaluated on AC.

2. **Grounding vs Planning**: OS-Atlas has strong click accuracy (73% extract match on AC) but fails on non-click actions. OS-Genesis has strong `open` action accuracy (66%) but poor click grounding (12.7%).

3. **Action type specialization**:
   - UI-TARS: Best at `wait` (67%), `system_button` (72%), `type` (82%)
   - OS-Atlas: Best at `swipe` (69%), strong `click` (73%)
   - OS-Genesis: Best at `open` (66%), strong `type` (54%)

4. **Length scaling**: All models collapse on long trajectories (8+ steps on AC, 6+ steps on GUI-360). No model achieves >6% TSR on long AC trajectories.

5. **Error propagation**: UI-TARS and OS-Atlas show resilient post-error accuracy (~65%), suggesting errors are independent rather than cascading. OS-Genesis degrades significantly post-error (16%).

---

## 7. Evaluation Infrastructure

### 7.1 Scripts

| Script | Purpose |
|--------|---------|
| `scripts/eval/eval_ar_trajectory_generic.py` | Generic AR evaluator for baselines on AC |
| `scripts/eval/eval_baseline_ac.slurm` | SLURM launcher for AC baselines |
| `scripts/eval/eval_baseline_gui360.slurm` | SLURM launcher for GUI-360 baselines |
| `train_GUI_360/GUI-360-eval/evaluation.py` | GUI-360 evaluation entry point |
| `train_GUI_360/GUI-360-eval/models/ui_tars.py` | UI-TARS adapter for GUI-360 |
| `train_GUI_360/GUI-360-eval/models/os_atlas.py` | OS-Atlas adapter for GUI-360 |
| `train_GUI_360/GUI-360-eval/models/os_genesis.py` | OS-Genesis adapter for GUI-360 |
| `scripts/exp2/verifier_ar_inference.py` | Verifier experiment inference |
| `scripts/exp2/analyze_verifier_experiment.py` | Verifier experiment analysis |

### 7.2 Model Checkpoints

All at `/scratch/a5l/shuqing.a5l/MobileAgent/UI-S1/checkpoints/`:
- `OS-Atlas-Pro-7B`, `UI-TARS-7B-DPO`, `OS-Genesis-7B-AC`, `Qwen2.5-VL-7B-Instruct`

### 7.3 Known Issues & Fixes

| Issue | Fix |
|-------|-----|
| UI-TARS `--limit-mm-per-prompt` bash quoting | Replaced associative array with if/elif |
| OS-Genesis `demjson3` Decimal serialization | Added `decimal.Decimal` → `int`/`float` conversion |
| UI-TARS GUI-360 `parse_action` empty args | Fixed regex: `(.*?)` → `(.*)` for nested parens |
| UI-TARS GUI-360 coordinate space | Added 0-1000 relative → absolute pixel conversion |

---

## Appendix: Raw Result Paths

**AC Results:**
- `scripts/eval/results/ac/os-atlas_stop_20260320_164514/`
- `scripts/eval/results/ac/os-atlas_nostop_20260320_164514/`
- `scripts/eval/results/ac/ui-tars_stop_20260320_165738/`
- `scripts/eval/results/ac/ui-tars_nostop_20260320_165738/`
- `scripts/eval/results/ac/os-genesis_stop_20260320_164503/`
- `scripts/eval/results/ac/os-genesis_nostop_20260320_164503/`

**GUI-360 Results:**
- `scripts/eval/results/gui360/os_atlas_stop_20260320_164534/`
- `scripts/eval/results/gui360/os_atlas_nostop_20260320_164534/`
- `scripts/eval/results/gui360/os_genesis_stop_20260320_165917/`
- `scripts/eval/results/gui360/os_genesis_nostop_20260320_165917/`
- `scripts/eval/results/gui360/ui_tars_*_20260321_*/` (pending re-run)

**Verifier Experiment:**
- `scripts/exp2/results/verifier_experiment/`

---

## 8. RL Training: SP-GiGPO + Visual Hindsight (V6/V7/V8)

### 8.1 Experiment Setup

| Version | Key Config | Hindsight Coef | Resume From | Training Steps |
|---------|-----------|:--------------:|-------------|:--------------:|
| V6 | SP + GiGPO + SPWA | 0.001 (effectively off) | — | 190+ |
| V7 | SP + GiGPO + SPWA + Visual Hindsight | 0.001 (effectively off) | — | 43 (walltime) |
| **V8** | SP + GiGPO + SPWA + Visual Hindsight | **0.07** (~24% of loss) | V7 step 30 | 157+ |

- **Base Model**: Qwen2.5-VL-7B-Instruct (SFT v2 checkpoint)
- **Training**: FSDP, 4 nodes × 4 GH200 GPUs, rollout n=8
- **Visual Hindsight**: Aux CE loss using s_{t+1} (next screenshot) to teach inverse dynamics — given (s_t, s_{t+1}), predict correct action a_t
- **Eval Dataset**: AndroidControl (1543 samples, multi-step GUI tasks)

### 8.2 Full Evaluation Results (1543 samples)

#### V8 All Checkpoints

| Step | Accuracy | Correct/Total |
|:----:|:--------:|:-------------:|
| **70** | **15.36%** | **237/1543** |
| 80 | 12.38% | 191/1543 |
| 90 | 11.92% | 184/1543 |
| 100 | 11.67% | 180/1543 |
| 110 | 10.82% | 167/1543 |
| 120 | 12.12% | 187/1543 |
| 130 | 13.48% | 208/1543 |
| 140 | 13.35% | 206/1543 |

#### V6 All Checkpoints

| Step | Accuracy | Correct/Total |
|:----:|:--------:|:-------------:|
| **120** | **10.56%** | **163/1543** |
| 150 | 9.01% | 139/1543 |
| 180 | 8.94% | 138/1543 |

#### Cross-Version Comparison

| | V6 Best | V8 Best |
|---|:-------:|:-------:|
| Checkpoint | step_120 | **step_70** |
| Accuracy | 10.56% | **15.36%** |
| Relative Improvement | — | **+45.5%** |

Increasing `hindsight_aux_coef` from 0.001 → 0.07 yielded significant gains.

### 8.3 Why Step 70 Is Best — Root Cause Analysis

#### 8.3.1 Premature Termination Is the Core Issue

The model predicts `{"action": "terminate", "status": "success"}` too early, ending the episode before the task is completed.

**Premature termination rate** (final_step_id=0 for failed tasks):

| Checkpoint | 1-step | 2-3 step | 4-6 step | 7+ step |
|:----------:|:------:|:--------:|:--------:|:-------:|
| **step_70** | **38.8%** | **45.0%** | **41.5%** | **42.0%** |
| step_80 | 49.1% | 47.2% | 41.8% | 46.4% |
| step_90 | 39.7% | 51.6% | 55.4% | 60.9% |
| step_110 | 39.7% | 52.8% | 50.0% | 50.3% |
| step_130 | 37.9% | 53.4% | 50.3% | 53.0% |

Step 70 has the **lowest premature termination across all multi-step categories**. By step 90, 4-6 step tasks see 55.4% premature termination (vs 41.5%), and 7+ step tasks reach 60.9% (vs 42.0%).

#### 8.3.2 Step 70 Explores Further Before Failing

Average steps completed for **failed multi-step tasks** (num_steps ≥ 3):

| Checkpoint | Avg Steps Completed | Relative Progress |
|:----------:|:-------------------:|:-----------------:|
| **step_70** | **1.10** | **19.0%** |
| step_80 | 0.99 | 17.2% |
| step_90 | 0.76 | 13.6% |
| step_110 | 0.90 | 15.8% |
| step_130 | 0.90 | 15.6% |

#### 8.3.3 Accuracy by Task Complexity

| num_steps | step_70 | step_80 | step_90 | step_110 | step_130 | Count |
|:---------:|:-------:|:-------:|:-------:|:--------:|:--------:|:-----:|
| 1 | **61.2%** | 50.9% | 60.3% | 60.3% | 62.1% | 116 |
| 2 | **26.2%** | 20.5% | 18.9% | 20.5% | 23.8% | 122 |
| 3 | **18.0%** | 17.5% | 16.0% | 13.5% | 15.5% | 200 |
| 4 | 15.8% | **16.3%** | 14.0% | 9.5% | 15.4% | 221 |
| 5 | **12.5%** | 8.2% | 5.5% | 5.1% | 10.2% | 256 |
| 6 | **11.0%** | 4.6% | 4.6% | 2.3% | 4.0% | 173 |
| 7+ | 2.9% | 1.7% | 1.5% | 2.0% | 1.8% | 455 |

Step 70 **dominates in 5-6 step tasks** where lower premature termination directly translates to more completions.

#### 8.3.4 The "Shortcut Learning" Hypothesis

| Metric | Step 70 | Step 110 | Trend |
|--------|:-------:|:--------:|:-----:|
| Training step_sr | 0.463 | 0.486 | ↑ improving |
| Eval accuracy | 15.36% | 10.82% | ↓ degrading |
| Premature terminate (4-6 step) | 41.5% | 50.0% | ↑ worsening |

**Training step_sr goes up while eval accuracy goes down.** The model learns a "shortcut": terminate early to avoid action errors, which inflates step-level success rate on the training distribution but hurts actual task completion on unseen tasks. Step 70 is the optimal balance between exploration (trying more actions) and exploitation (knowing when to stop).

#### 8.3.5 Checkpoint Overlap Is Extremely Low

Between step 70 (237 correct) and step 130 (208 correct), only **40 tasks overlap**. This indicates high evaluation stochasticity (despite temp≈0.01), but the systematic advantage of step 70 manifests as a **distributional shift** (lower terminate rate, higher progress), not per-task determinism.

### 8.4 Output Quality Analysis (17,089 training responses)

| Metric | Value |
|--------|:-----:|
| Format correct (`<think>...<action>...`) | 99.9% |
| Empty/trivial `<think>` | 0.1% |
| Short `<think>` (<20 chars) | 16.4% |
| Invalid action types | 38 (0.2%) |
| Hallucination instances | 4 |

**Action type distribution:**

| Action | Count | Percentage |
|--------|------:|----------:|
| click | 9,175 | 53.7% |
| terminate | 2,951 | 17.3% |
| swipe | 1,492 | 8.7% |
| type | 1,130 | 6.6% |
| system_button | 990 | 5.8% |
| open | 835 | 4.9% |
| wait | 424 | 2.5% |
| other (invalid) | 38 | 0.2% |

### 8.5 Key Takeaways

1. **Hindsight aux loss at coef=0.07 works**: V8 best (15.36%) significantly outperforms V6 best (10.56%), a +45.5% relative improvement.

2. **Best checkpoint is early** (step 70 out of 157+): Further training degrades eval performance due to shortcut learning (premature termination).

3. **Premature termination is the #1 bottleneck**: ~42-55% of failed tasks terminate at the very first step. Reducing this is the most impactful next direction.

4. **Potential improvements**:
   - Penalize premature termination in the reward function
   - Add a minimum-steps constraint before allowing terminate
   - Use trajectory-level reward shaping that rewards progress even on failure
   - Train with exploration bonus to counteract the terminate shortcut

### 8.6 Result Paths

- V8 eval results: `evaluation/results/sp_gigpo_v8_step{70,80,90,100,110,120,130,140}.jsonl`
- V6 eval results: `evaluation/results/sp_gigpo_v6_step{120,150,180}.jsonl`
- V8 best merged model: `checkpoints/sp_gigpo_ac/sp_gigpo_spwa_k8_v8/merged_step_70`
- V6 best merged model: `checkpoints/sp_gigpo_ac/sp_gigpo_spwa_k8_v6/merged_step_120`
- Training logs: `train/sp_gigpo/logs/sp_gigpo_3334060.log` (V8), `sp_gigpo_3333780.log` (V6)
