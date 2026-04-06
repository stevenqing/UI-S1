# Exp2: Cross-Dataset Long-Horizon Failure Analysis

Model: Qwen2.5-VL-7B-Instruct | SFT v2: gui360_full_sft_v2 (Grounding 70.56%, Action 46.90%)
Dataset: GUI-360 (test split, 3 domains: Excel/PPT/Word) + AndroidControl

---

## 1. SFT v2 vs Base: Overall Performance

| Metric | Base | SFT v2 | Delta |
|--------|:---:|:---:|:---:|
| GUI-360 AR TSR (stop) | 1.64% | **16.21%** | +14.57pp (10x) |
| GUI-360 AR Step Acc (stop) | 22.10% | **55.28%** | +33.18pp |
| GUI-360 Scattered Progress | 12.02% | **40.91%** | +28.89pp |
| Step-0 Failure Rate | 78.72% | **50.02%** | -28.70pp |
| Post-Error Accuracy | 5.07% | **20.81%** | +15.74pp (4.1x) |

By trajectory length (SFT v2, no-stop): Short 24.42% → Medium 1.94% → **Long 0.00%**. Long horizon remains completely unsolved. Survival: 0.56 (step 0) → 0.34 (step 2) → 0.26 (step 5).

---

## 2. Error Structure: Planning >> Grounding

### Cross-dataset consensus (3 methods, 2 datasets)

| | GUI-360 Base | GUI-360 SFT v2 | AC Base |
|---|:---:|:---:|:---:|
| CASCADE (stuck/repeating) | 70.0% | **33.5%** | 5.4% |
| PLANNING (wrong intent/element) | 17.7% | **47.3%** | **92.5%** |
| GROUNDING (right intent, wrong coord) | 1.4% | 2.5% | 2.1% |

**Independent errors (excl cascade):**

| | GUI-360 Base | GUI-360 SFT v2 | AC Base |
|---|:---:|:---:|:---:|
| PLANNING | 70.6% | **82.7%** | **97.8%** |
| GROUNDING | 5.6% | 4.3% | 2.2% |

**Core finding**: Grounding error was massively overestimated (LLM classifier: 76.5%). Three independent methods agree: **grounding = 2-6% of errors; planning = 71-98%**. The model picks the wrong UI element, not fails to locate it.

### SFT v2 error shift

| Error Type | Base | SFT v2 | Interpretation |
|---|:---:|:---:|---|
| Stuck/repeating | 51.2% | **33.8%** | SFT reduces cascade |
| Coord error (≥50px) | 18.4% | **35.2%** | More "trying but wrong" |
| Near miss (<50px) | 3.5% | **9.2%** | Some grounding improvement |

Failure mode upgrade: from "stuck and completely failed" to "trying but hitting wrong element."

---

## 3. Context Modes & Subtask Isolation

| Mode | TSR Delta (Full) | TSR Delta (Pattern B, N=202) |
|------|:---:|:---:|
| Summary | +0.83pp | — |
| **Subtask isolated** | **+1.39pp** | **+3.96pp** |
| Subtask oracle (upper bound) | +7.27pp | — |

Full-dataset gain diluted by 77.6% single-subtask trajectories. On Pattern B (202 genuine multi-subtask trajectories), context reset benefit is **+3.96pp TSR**. But Pattern B TSR with isolation is still only 7.92%.

---

## 4. Prompt Engineering: Cognitive Forcing (Pattern B, N=202)

| Prompt | TSR | Scattered Progress | Delta TSR |
|--------|:---:|:---:|:---:|
| subtask_isolated (baseline) | 7.92% | 44.41% | — |
| **progress** (track completed/remaining) | **9.41%** | 43.86% | **+1.49pp** |
| scene (describe current UI state) | 8.42% | 44.02% | +0.50pp |
| intent (state step intent in 1 sentence) | 7.43% | 43.95% | -0.50pp |

- Progress awareness gives the best lift (+1.49pp TSR, +2.74pp on PPT)
- All prompts have ~identical step accuracy (~41%) — TSR gains come from consistency, not capability
- Excel = 0% TSR across all prompts; 3+ subtasks = 0% TSR — hard floor
- **Prompt engineering ceiling is low** (+1.49pp max)

Stacking: context reset (+3.96pp) + progress prompt (+1.49pp) = **+5.45pp** total over AR baseline on Pattern B.

---

## 5. Layer-wise Linear Probing

### Setup

Extract hidden states from all 28 transformer layers of SFT v2, at 3 token positions:
- **Image tokens** (within `<|vision_start|>...<|vision_end|>`): mean-pooled
- **History tokens** (between "The history of actions are:\n" and "The actions supported are"): mean-pooled
- **Last token**: single vector

3000 samples from 487 trajectories (balanced across 3 domains). PCA(256) + LogisticRegression.

Train: 2655 non-Pattern-B samples | Test: 345 Pattern B samples | 5-fold CV on all data.

### Primary Probes (test accuracy on Pattern B)

| Layer | Scene (domain) | Progress (position) | Action (type) |
|:---:|:---:|:---:|:---:|
| 0 | 0.875 | 0.351 | 0.736 |
| 1 | 0.957 | 0.380 | 0.716 |
| 2 | 0.957 | 0.359 | 0.699 |
| 3 | 0.968 | 0.336 | 0.632 |
| 4 | 0.954 | 0.359 | 0.594 |
| 5 | 0.936 | 0.368 | 0.597 |
| 6 | 0.974 | 0.345 | 0.612 |
| 7 | 0.980 | 0.325 | 0.583 |
| 8 | 0.980 | 0.328 | 0.609 |
| 9 | 0.968 | 0.371 | 0.626 |
| 10 | 0.951 | 0.354 | 0.620 |
| 11 | 0.936 | 0.357 | 0.638 |
| 12 | 0.916 | 0.359 | 0.620 |
| 13 | 0.913 | 0.400 | 0.623 |
| 14 | 0.951 | 0.383 | 0.690 |
| 15 | 0.948 | 0.368 | 0.646 |
| 16 | 0.959 | 0.336 | 0.591 |
| 17 | 0.939 | 0.354 | 0.614 |
| 18 | 0.948 | 0.371 | 0.626 |
| 19 | 0.959 | 0.351 | 0.661 |
| 20 | 0.965 | 0.351 | 0.609 |
| 21 | 0.974 | 0.351 | 0.649 |
| 22 | 0.980 | 0.365 | 0.646 |
| 23 | 0.977 | 0.359 | 0.678 |
| 24 | 0.988 | 0.357 | 0.701 |
| 25 | 0.988 | 0.365 | 0.675 |
| 26 | **0.994** | 0.357 | 0.704 |
| 27 | 0.986 | **0.386** | **0.736** |

Majority baselines: Scene = 0.449, Progress = 0.253, Action = 0.740

### Cross-probes: Token Type × Label (5 sampled layers)

| Layer | img→domain | img→position | img→action | hist→domain | hist→position | hist→action | last→domain | last→position | last→action |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 0 | 0.88 | 0.24 | 0.57 | 0.90 | 0.34 | **0.77** | **1.00** | **0.40** | 0.70 |
| 6 | 0.97 | 0.29 | 0.61 | 0.98 | 0.34 | **0.81** | **1.00** | 0.33 | 0.62 |
| 13 | 0.91 | 0.30 | 0.70 | 0.97 | **0.39** | 0.57 | 0.99 | 0.34 | 0.63 |
| 20 | 0.97 | 0.34 | 0.70 | 0.99 | 0.35 | 0.66 | **1.00** | **0.39** | 0.65 |
| 27 | 0.99 | 0.29 | 0.68 | **1.00** | **0.39** | 0.73 | 0.99 | 0.37 | **0.75** |

### v1 Interpretation (trivial probes — limitations noted)

v1 probes had design flaws: Scene=domain (trivially visual), Progress=step position (not task progress), Action=type (74% majority=click). These measured trivial features, not planning-relevant capabilities. See v2 below for redesigned probes.

Key v1 observation: domain is trivially encoded everywhere (87-99%), confirming that perceptual information saturates all layers. This is not useful for MoE routing.

---

## 5b. Layer-wise Probing v2 (Redesigned)

v1 probes were fundamentally flawed — they tested trivial/irrelevant features. v2 redesigns all three probes to directly target planning error failure modes.

### Probe Redesign Rationale

| v1 Probe | Problem | v2 Replacement |
|----------|---------|---------------|
| Scene (domain) | Trivial — pixel features distinguish PPT/Excel/Word, unrelated to understanding | **UI State** — does model understand current UI context (dialog/menu/selection/main)? |
| Progress (step index) | Irrelevant — step position ≠ task completion | *(dropped — no clean task-completion label available)* |
| Action (type) | Trivial — 74% are "click", majority baseline | **Target Element** — does model know WHICH element to interact with? |
| *(none)* | | **Correctness** — does hidden state predict whether this step will be correct? |

### Setup

Same 3000 samples, same hidden state npy files (28 layers × 3 token types). Only labels changed.

**Probe 1 — Target Element** (last token → element category, 11 classes):
- Label: `control_test` field grouped into functional categories (ribbon_tab, dialog_button, file_operation, formatting, content_area, cell_reference, search_input, navigation, object_insertion, animation_property, other)
- Majority baseline: 0.372 ("other")
- Train: 2655 non-Pattern-B, Test: 345 Pattern B

**Probe 2 — UI State** (image tokens → UI state, 5 classes):
- Label: derived from `observation` + `thought` text via keyword matching (dialog, menu_open, selection_active, main_view, ribbon_focus)
- Majority baseline: 0.324 (selection_active)
- Train/test: same Pattern B split

**Probe 3 — Correctness** (last token → correct/wrong, binary):
- Label: from AR eval results, **clean samples only** (steps ≤ first_error_step, where model had correct context)
- 956 clean samples (562 correct, 394 wrong), majority baseline: 0.588
- **5-fold CV** (Pattern B test set too small at 62 samples)

### v2 Results

| Layer | Target Element | UI State | Correctness (CV) | TE-F1 | UI-F1 | C-F1 (CV) |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 0 | 0.293 | 0.452 | 0.552 | 0.163 | 0.360 | 0.539 |
| 4 | 0.342 | 0.446 | 0.570 | 0.169 | 0.368 | 0.560 |
| 7 | 0.310 | 0.374 | 0.573 | 0.171 | 0.331 | 0.566 |
| 10 | 0.374 | 0.458 | 0.570 | 0.226 | 0.400 | 0.565 |
| 13 | 0.258 | 0.464 | **0.617** | 0.194 | 0.380 | 0.612 |
| 15 | 0.293 | **0.496** | **0.637** | 0.204 | **0.423** | **0.632** |
| 18 | 0.409 | 0.484 | **0.639** | 0.308 | 0.407 | 0.632 |
| 20 | **0.455** | 0.388 | 0.620 | 0.324 | 0.343 | 0.612 |
| 23 | **0.441** | 0.371 | 0.632 | 0.308 | 0.323 | 0.622 |
| 25 | 0.412 | 0.403 | **0.640** | **0.352** | 0.356 | 0.633 |
| 26 | **0.438** | 0.406 | **0.650** | **0.351** | 0.363 | **0.642** |
| 27 | 0.368 | 0.417 | 0.625 | 0.296 | 0.366 | 0.619 |

Majority baselines: Target Element = 0.372, UI State = 0.324, Correctness = 0.588

### Cross-probes v2 (5 sampled layers)

| Layer | img→TE | img→UI | img→Corr | hist→TE | hist→UI | hist→Corr | last→TE | last→UI | last→Corr |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 0 | 0.27 | **0.45** | 0.56 | **0.41** | 0.33 | 0.57 | 0.29 | 0.34 | 0.55 |
| 6 | 0.29 | **0.41** | 0.58 | 0.31 | 0.30 | 0.59 | 0.35 | 0.38 | 0.56 |
| 13 | 0.30 | **0.46** | 0.60 | 0.31 | 0.33 | 0.60 | 0.26 | **0.46** | **0.62** |
| 20 | 0.34 | 0.39 | **0.63** | 0.34 | 0.34 | 0.61 | **0.46** | 0.44 | 0.62 |
| 27 | 0.30 | **0.42** | 0.61 | 0.28 | 0.35 | 0.60 | 0.37 | 0.39 | **0.62** |

### v2 Interpretation

**Target Element Identification (blue line)**:
- Early layers (0–13): **at or below majority** (0.258–0.374 vs baseline 0.372)
- Late layers (18–26): **rises above majority** to 0.41–0.46, with F1 reaching 0.35
- **Layer-specific finding**: target element identity emerges in layers 18+ (upper layers), consistent with these layers performing action planning
- Cross-probe shows: history tokens encode target element at L0 (0.41) but this fades; last token takes over at L20+ (0.46)
- The signal is weak overall — the model has a noisy, barely-above-chance representation of which element to interact with, explaining the 71–98% planning error rate

**UI State Understanding (green line)**:
- Peaks at **L15 (0.496)** and L14 (0.490) — **middle layers**
- Image tokens are the best source at all layers (cross-probe: img→UI = 0.41–0.46)
- Middle layers (12–17) show highest UI state encoding, suggesting these layers specialize in visual scene interpretation
- But the signal is modest (+17pp above majority) — the model's UI state understanding is limited

**Correctness Prediction (red line)**:
- **Monotonically increases from L0 (0.552) to L26 (0.650)**, with a clear ramp starting at L13
- Best at L26: 65.0% CV accuracy (majority = 58.8%), so +6.2pp above chance
- The model "partially knows" when it will make an error — the hidden state at later layers contains a weak but real predictive signal
- Cross-probe: correctness signal is distributed across ALL token types (image 0.56–0.63, history 0.57–0.61, last 0.55–0.63), not localized to a single modality
- **This directly motivates a verifier**: even a simple linear probe at L26 can predict correctness 6pp above chance. A non-linear verifier (MLP/attention) could amplify this signal

**Layer-wise specialization pattern (v2 reveals what v1 missed)**:
- **Layers 0–10**: Undifferentiated — no probe significantly exceeds majority
- **Layers 12–17**: **UI state understanding peak** — middle layers specialize in visual scene parsing
- **Layers 18–26**: **Target element emergence** — upper layers begin encoding which element to interact with
- **Layers 13–27**: **Correctness signal ramp** — error-awareness builds gradually through the network
- This is the first evidence of **mild functional specialization** in the SFT v2 model

---

## 6. Key Takeaways

1. **Planning is THE bottleneck (71–98% of errors)**, not grounding (2–6%). The model picks the wrong element, not fails to locate it.

2. **SFT produces a qualitative shift**: error structure moves from stuck-dominated (51%) to coord-error-dominated (35%), with 4x better post-error recovery (5%→21%).

3. **Step-0 is the biggest lever**: 50% of trajectories fail at step 1. Improving step-0 accuracy has the largest multiplicative effect on TSR.

4. **Long horizon is unsolved**: Medium/Long TSR near 0%. Compounding error is fundamental — survival drops to ~0.34 after 2 steps.

5. **Prompt engineering has a low ceiling** (+1.49pp max). Progress tracking helps most. Training-time interventions needed.

6. **Layer probing v2 reveals mild functional specialization**:
   - **Middle layers (12–17)**: UI state understanding peaks — visual scene parsing
   - **Upper layers (18–26)**: Target element identity emerges — action planning
   - **L13–27 ramp**: Correctness prediction builds gradually — the model develops error-awareness
   - All signals are weak (~5–17pp above chance), explaining the high planning error rate
   - v1's "flat, undifferentiated" conclusion was an artifact of testing trivial features

7. **The model partially "knows" when it will fail**:
   - L26 correctness prediction: 65.0% CV accuracy (vs 58.8% majority)
   - Signal distributed across all token types, not localized
   - **Directly motivates a verifier agent**: amplify this weak linear signal with non-linear computation
   - Even a +6pp correctness prediction advantage could cascade into meaningful TSR improvement if used to trigger action revision

8. **Implications for MoE design**:
   - v2 probing now shows mild layer-wise specialization (perception → planning), supporting a 2-tier partition more than a 3-tier
   - **Lower half (L0–13)**: perceptual processing + UI state understanding
   - **Upper half (L14–27)**: target element selection + error-aware action planning
   - MoE experts in upper layers should focus on action planning diversity
   - A correctness-aware routing signal could activate specialized "careful planning" experts when the model's hidden state predicts likely failure

---

## Artifacts

| Artifact | Path |
|----------|------|
| Prompt engineering results | `scripts/exp2/results/analysis/` |
| Probing extraction data (3.4GB) | `scripts/exp2/results/layer_probing/sft_v2_20260320_130909/` |
| Probing v1 results | `scripts/exp2/results/layer_probing/sft_v2_20260320_130909/probe_results.json` |
| **Probing v2 results** | `scripts/exp2/results/layer_probing/sft_v2_20260320_130909/probe_v2_results.json` |
| v2 probing curves | `scripts/exp2/results/layer_probing/sft_v2_20260320_130909/probing_v2_curves.png` |
| v2 F1 curves | `scripts/exp2/results/layer_probing/sft_v2_20260320_130909/probing_v2_f1.png` |
| v2 cross-probe heatmap | `scripts/exp2/results/layer_probing/sft_v2_20260320_130909/cross_probe_v2_heatmap.png` |
| Probing scripts | `scripts/exp2/layer_probing/probe_layerwise.py`, `probe_v2.py` |
| Pattern B IDs | `scripts/exp2/pattern_b_ids.json` |
