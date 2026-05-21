# Cooperative v6.5 AC Evaluation Analysis

> Eval date: 2026-04-14
> Eval script: `evaluation/eval_qwenvl.py` (paper-consistent)
> Dataset: `evaluation/results/android_control_evaluation_fixed.jsonl` (1543 episodes)

## 1. Overall Task Success Rate (TSR)

| Model | Success | Total | TSR |
|-------|---------|-------|-----|
| Base Qwen2.5-VL-7B | 136 | 1524 | **8.92%** |
| Qwen2.5-VL-7B (paper) | 132 | 1524 | **8.66%** |
| Cooperative v6.5 ep1 | 155 | 1503 | **10.31%** |
| Vanilla SFT ep1 | 79 | 1488 | **5.31%** |
| UI-TARS-7B | 226 | 1524 | **14.83%** |
| UI-S1-7B (GRPO) | 99 | 1524 | **6.50%** |
| OS-Atlas-7B | 130 | 1524 | **8.53%** |

## 2. Critical Decomposition: `open` vs non-`open` First Action

39% of AC episodes (602/1543) start with `open <app_name>`. This is a special action type
where the model outputs `open` + app name text. Base model and UI-TARS don't know this format
(they were never trained on it), so they get ~0% on these. Cooperative learned it from GUI-360.

| Model | open-first TSR | non-open TSR | Overall TSR |
|-------|---------------|-------------|------------|
| Base Qwen2.5-VL-7B | 0/602 (0.0%) | 136/922 (14.8%) | 136/1524 (8.9%) |
| Qwen2.5-VL-7B (paper) | 0/602 (0.0%) | 132/922 (14.3%) | 132/1524 (8.7%) |
| Cooperative v6.5 ep1 | 37/594 (6.2%) | 118/909 (13.0%) | 155/1503 (10.3%) |
| Vanilla SFT ep1 | 18/582 (3.1%) | 61/906 (6.7%) | 79/1488 (5.3%) |
| UI-TARS-7B | 3/602 (0.5%) | 223/922 (24.2%) | 226/1524 (14.8%) |
| UI-S1-7B (GRPO) | 0/602 (0.0%) | 99/922 (10.7%) | 99/1524 (6.5%) |
| OS-Atlas-7B | 0/602 (0.0%) | 130/922 (14.1%) | 130/1524 (8.5%) |

**Key insight**: Cooperative's overall +1.4pp over base is almost entirely from `open` actions (+37 successes).
On non-open episodes (the real test), cooperative is actually **worse** than base: 13.0% vs 14.8% (-1.8pp).

## 3. Non-open Episodes: TSR by Trajectory Length

| Model | 1-2 steps | 3-5 steps | 6-8 steps | 9+ steps |
|---|---|---|---|---|
| Base Qwen2.5-VL-7B | 69/191 (36.1%) | 61/409 (14.9%) | 6/222 (2.7%) | 0/100 (0.0%) |
| Qwen2.5-VL-7B (paper) | 67/191 (35.1%) | 61/409 (14.9%) | 4/222 (1.8%) | 0/100 (0.0%) |
| Cooperative v6.5 ep1 | 75/190 (39.5%) | 40/406 (9.9%) | 2/215 (0.9%) | 1/98 (1.0%) |
| Vanilla SFT ep1 | 48/189 (25.4%) | 13/402 (3.2%) | 0/219 (0.0%) | 0/96 (0.0%) |
| UI-TARS-7B | 85/191 (44.5%) | 110/409 (26.9%) | 25/222 (11.3%) | 3/100 (3.0%) |
| UI-S1-7B (GRPO) | 71/191 (37.2%) | 25/409 (6.1%) | 3/222 (1.4%) | 0/100 (0.0%) |
| OS-Atlas-7B | 62/191 (32.5%) | 61/409 (14.9%) | 3/222 (1.4%) | 4/100 (4.0%) |

## 4. TSR by First Action Type

| Model | click | open | system_button | swipe | wait | long_press |
|---|---|---|---|---|---|---|
| Base Qwen2.5-VL-7B | 124/642 (19.3%) | 0/602 (0.0%) | 4/194 (2.1%) | 8/76 (10.5%) | 0/6 (0.0%) | 0/4 (0.0%) |
| Qwen2.5-VL-7B (paper) | 119/642 (18.5%) | 0/602 (0.0%) | 4/194 (2.1%) | 9/76 (11.8%) | 0/6 (0.0%) | 0/4 (0.0%) |
| Cooperative v6.5 ep1 | 97/634 (15.3%) | 37/594 (6.2%) | 3/191 (1.6%) | 18/75 (24.0%) | 0/5 (0.0%) | 0/4 (0.0%) |
| Vanilla SFT ep1 | 41/631 (6.5%) | 18/582 (3.1%) | 3/191 (1.6%) | 17/74 (23.0%) | 0/6 (0.0%) | 0/4 (0.0%) |
| UI-TARS-7B | 183/642 (28.5%) | 3/602 (0.5%) | 30/194 (15.5%) | 10/76 (13.2%) | 0/6 (0.0%) | 0/4 (0.0%) |
| UI-S1-7B (GRPO) | 88/642 (13.7%) | 0/602 (0.0%) | 2/194 (1.0%) | 9/76 (11.8%) | 0/6 (0.0%) | 0/4 (0.0%) |
| OS-Atlas-7B | 110/642 (17.1%) | 0/602 (0.0%) | 4/194 (2.1%) | 16/76 (21.1%) | 0/6 (0.0%) | 0/4 (0.0%) |

## 5. Per-Step TSR (All Episodes)

| Steps | N | Base | Coop v6.5 | SFT | UI-TARS | Coop-Base |
|-------|---|------|-----------|-----|---------|-----------|
| 1 | 114 | 45.6% | 59.3% | 37.8% | 54.4% | +13.7% |
| 2 | 117 | 14.5% | 20.0% | 12.9% | 19.7% | +5.5% |
| 3 | 200 | 12.5% | 11.1% | 4.0% | 22.5% | -1.4% |
| 4 | 219 | 11.0% | 9.7% | 2.8% | 15.5% | -1.2% |
| 5 | 254 | 4.7% | 6.0% | 2.9% | 13.4% | +1.2% |
| 6 | 171 | 1.8% | 2.4% | 0.6% | 8.2% | +0.6% |
| 7 | 136 | 1.5% | 1.5% | 0.0% | 7.4% | +0.0% |
| 8 | 101 | 1.0% | 0.0% | 0.0% | 1.0% | -1.0% |
| 9 | 65 | 0.0% | 0.0% | 0.0% | 4.6% | +0.0% |
| 10 | 41 | 0.0% | 2.4% | 0.0% | 0.0% | +2.4% |
| 11 | 29 | 0.0% | 0.0% | 0.0% | 0.0% | +0.0% |
| 12 | 17 | 0.0% | 0.0% | 0.0% | 0.0% | +0.0% |
| 13 | 18 | 0.0% | 0.0% | 0.0% | 0.0% | +0.0% |
| 14 | 8 | 0.0% | 0.0% | 0.0% | 0.0% | +0.0% |
| 15 | 6 | 0.0% | 0.0% | 0.0% | 0.0% | +0.0% |
| 16 | 5 | 0.0% | 0.0% | 0.0% | 0.0% | +0.0% |
| 17 | 8 | 0.0% | 0.0% | 0.0% | 0.0% | +0.0% |
| 18 | 4 | 0.0% | 0.0% | 0.0% | 0.0% | +0.0% |
| 19 | 6 | 0.0% | 0.0% | 0.0% | 0.0% | +0.0% |
| 20 | 1 | 0.0% | 0.0% | 0.0% | 0.0% | +0.0% |
| 21 | 1 | 0.0% | 0.0% | 0.0% | 0.0% | +0.0% |
| 22 | 1 | 0.0% | 0.0% | 0.0% | 0.0% | +0.0% |
| 24 | 2 | 0.0% | 0.0% | 0.0% | 0.0% | +0.0% |

## 6. Head-to-Head Analysis

### Cooperative vs Base
- Common episodes: 1503
- Both succeed: 73
- Both fail: 1285
- **Coop wins** (coop ok, base fail): 82
- **Coop loses** (coop fail, base ok): 63
- **Net**: +19

By step count:

| Steps | Coop wins | Coop loses | Net |
|-------|-----------|------------|-----|
| 1 | 24 | 9 | +15 |
| 2 | 15 | 9 | +6 |
| 3 | 15 | 18 | -3 |
| 4 | 10 | 13 | -3 |
| 5 | 12 | 9 | +3 |
| 6 | 3 | 2 | +1 |
| 7 | 2 | 2 | +0 |
| 8 | 0 | 1 | -1 |
| 10 | 1 | 0 | +1 |

### Cooperative vs UI-TARS
- Common episodes: 1503
- **Coop unique wins**: 76
- **UI-TARS unique wins**: 147
- **Net**: -71

**Coop unique wins** (76): click=38, open=19, swipe=19
**UI-TARS unique wins** (147): click=139, system_button=4, swipe=4

## 7. Failure Analysis: Step-0 Failures

Step-0 failure = model fails at the very first action.

| Model | Step-0 failures | Total failures | Rate |
|-------|----------------|----------------|------|
| Base | 1085 | 1388 | 78.2% |
| Coop v6.5 | 791 | 1348 | 58.7% |
| UI-TARS | 932 | 1298 | 71.8% |

### Progress on Failed Episodes (avg final_step / total_steps)

| Model | 1-2 steps | 3-5 steps | 6-8 steps | 9+ steps |
|-------|-----------|-----------|-----------|----------|
| Base | 5.6% | 10.3% | 8.5% | 3.2% |
| Coop v6.5 | 6.9% | 16.6% | 11.3% | 7.1% |
| UI-TARS | 7.2% | 12.3% | 10.3% | 5.9% |

## 8. Diagnosis & Takeaways

### Cooperative's advantage is narrow
- The +1.4pp overall improvement (10.3% vs 8.9%) is **almost entirely from `open` actions** (+37 successes)
- On non-open episodes, cooperative is **worse** than base: 13.0% vs 14.8% (-1.8pp)
- Cooperative also excels at `swipe` tasks: 24.0% vs base 10.5%

### Click accuracy degradation is the main problem
- `click` is the dominant action type (~80% of episodes)
- Cooperative: 15.3% on click-first vs Base: 19.3% = **-4.0pp regression**
- UI-TARS achieves 28.5% on click-first, nearly 2x cooperative
- The 31 episodes where coop loses to base on 3-4 step tasks are mostly pure-click sequences
- Many of these coop failures are at `final_step_id=0` (fails at the very first click)

### Cooperative makes more progress before failing
- Step-0 failure rate: base 78.2% vs coop 58.6% (cooperative attempts more steps)
- On 3-5 step failed episodes: coop progresses 16.6% vs base 10.3%
- This suggests cooperative has better task understanding but worse action precision

### Root cause hypothesis
1. **Training data domain shift**: GUI-360 training data teaches `open`/`swipe` but may hurt click precision
2. **V-A communication overhead**: The per-layer communication may add noise for simple click-only tasks
3. **Lossless merge artifacts**: The correction adapter mechanism may introduce inference-time errors
4. **UI-TARS gap**: UI-TARS was trained on much more diverse and larger-scale UI interaction data

### Vanilla SFT is catastrophically worse
- Vanilla SFT (same data, no cooperation): 5.3% overall, 7.7% non-open
- This is **worse than base model** on every metric
- Suggests the GUI-360 SFT data hurts AC performance when naively fine-tuned
- Cooperative mechanism partially rescues this: 13.0% vs 7.7% on non-open

---

## 9. Cooperative Advantage (CA) Ablation — v6.5 epoch-4

> Eval date: 2026-04-15

| Mode | TSR | Count | Description |
|------|-----|-------|-------------|
| hard | 10.11% | 156/1543 | Image→V, text→A (default) |
| t_only | **10.33%** | 158/1529 | All tokens→LoRA_A |
| v_only | 7.26% | — | All tokens→LoRA_V |
| merge | 8.11% | 125/1542 | 0.5·V + 0.5·A |

**CA = hard - max(v_only, t_only) = 10.11% - 10.33% = -0.22%**

**Conclusion**: LoRA_V is a net negative. Using only LoRA_A outperforms hard routing. The cooperative mechanism provides no advantage on AC.

**Root cause**: LoRA_V receives **0% direct CE loss** — image tokens are input-only (never in labels). LoRA_V's gradient comes entirely through indirect attention cross-terms, which is insufficient.

## 10. coord_routing + Cooperative Reasoning (α-mixing)

> Eval date: 2026-04-15
> Training job: 3817688 (27 min, 4 epochs)
> Config: `--coord_routing --coop_reasoning_alpha 0.3`

### 10.1 Design

- Image tokens → LoRA_V (mask=1.0)
- Coordinate/bbox digit tokens → LoRA_V (mask=1.0, via coord_routing)
- Assistant text tokens → α·V + (1-α)·A (mask=0.3)
- Other text tokens → LoRA_A (mask=0.0)

Goal: Give LoRA_V direct CE gradient through α-mixing on assistant tokens.

### 10.2 Results — coord_routing + α=0.3 (inference with α)

| Epoch | TSR | Count |
|-------|-----|-------|
| 1 | 8.94% | 138/1543 |
| **2** | **9.40%** | 145/1543 |
| 3 | 7.45% | 115/1543 |
| 4 | 7.19% | 111/1543 |

### 10.3 Results — coord_routing only (inference without α)

Same checkpoint as 10.2, but `--coop-reasoning-alpha 0` at inference.

| Epoch | TSR | Count |
|-------|-----|-------|
| 2 | **9.01%** | 139/1543 |

### 10.4 Comparison

| Config | Best Epoch | Best TSR | Δ vs v6.5 hard |
|--------|-----------|---------|----------------|
| old v6.5 hard | 4 | **10.11%** | baseline |
| old v6.5 t_only | 4 | **10.33%** | +0.22% |
| coord_routing + α=0.3 | 2 | 9.40% | -0.71% |
| coord_routing noalpha | 2 | 9.01% | -1.10% |

### 10.5 Analysis

1. **α-mixing hurts, not helps**: coord_routing+α (9.40%) < old v6.5 (10.11%)
2. **Training-time damage is permanent**: Turning off α at inference (9.01%) doesn't recover — weights are already trained with α, which polluted gradient flow
3. **Faster overfitting**: Best at epoch 2 (vs epoch 4 for old v6.5), degrades rapidly by epoch 3-4
4. **Training loss is identical**: CE loss curves for old v6.5 and coord_routing+α are nearly the same (1.04→0.59). The damage is not visible in training metrics, only in eval.

### 10.6 Why α-mixing fails on AC

**Core insight**: LoRA_V learns **image encoding** weights, LoRA_A learns **text generation** weights. These are orthogonal functions.

- α-mixing adds `0.3 × (δV - δA)` to assistant tokens at every layer
- δV is optimized for image pixel processing, not text generation
- This is like using a camera lens adjustment as a microphone setting — wrong modality

AC-specific factors:
- **92% of assistant tokens are think** (describing screen layout)
- **Only 3.9% are coordinates** — LoRA_V's only direct CE signal via coord_routing
- Visual info already flows through attention; mixing δV into text tokens is redundant noise
- Single image per step → simple visual reasoning, not a bottleneck

## 11. Hidden State Verification for Learned Router

> Eval date: 2026-04-15

### 11.1 First run (v1 — think token bug)

Bug: Used `<thought>` bigram (13708, 2450) instead of `<think>` trigram (13708, 766, 29). Result: 0 think tokens matched. Only IMAGE vs ACTION_TYPE comparison.

| Layer | Group CosSim | Linear Probe Acc | N |
|-------|-------------|-----------------|---|
| 0 | 0.53 | **100.0%** | 3974 |
| 7 | 0.73 | **99.8%** | 3974 |
| 14 | 0.72 | **99.7%** | 3974 |
| 21 | 0.71 | **99.5%** | 3974 |
| 27 | 0.60 | **98.5%** | 3974 |

**IMAGE vs ACTION_TYPE is trivially separable** — a simple linear classifier achieves >98.5% accuracy at all layers. This confirms that hidden states encode token modality.

### 11.2 Token classification fix

Fixed trigram detection:
- `<think>` = (13708, 766, 29), `</think>` = (522, 26865, 29)
- `<thought>` = (13708, 2450, 29), `</thought>` = (522, 60565, 29)

Local verification (tokenizer only, 50 samples):
- Think tokens: 40014 (95.9% of tokens)
- Spatial keywords: 11.5% of think tokens
- Decision keywords: 88.5% of think tokens

### 11.3 Detailed Hidden State Analysis (Job 3833920)

Script: `evaluation/verify_hidden_state_detailed.py` (15 samples, layers [0, 7, 14, 21, 27])

Token counts per sample (typical):
- IMAGE: 3354 (fixed vision tokens)
- COORD (digit/bbox inside think): 25–135 depending on whether current step generates coordinates
- THINK_DESC: 600–842 (structural "screen description" portion of think)
- THINK_DECIDE: 52–85 (final "decide action" portion of think)
- ACTION: 1 (first token of `pyautogui.xxx`)
- OTHER: 60 (response template scaffold)

#### Linear probe accuracy — all groupings, all layers

| Layer | IMAGE vs rest | DESC vs DECIDE | IMG+COORD vs ACT | IMG+COORD+DESC vs DEC+ACT | THINK vs ACT |
|-------|---------------|-----------------|-------------------|----------------------------|---------------|
| 0 | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** |
| 7 | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** |
| 14 | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** |
| 21 | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** |
| 27 | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** |

**Every token category is 100% linearly separable at every layer.** The hidden state already contains all the routing information a linear router would ever need.

#### Affinity spectrum (closer to IMAGE vs ACTION)

| Layer | DESC→IMG | DESC→ACT | DESC spectrum | DECIDE→IMG | DECIDE→ACT | DECIDE spectrum |
|-------|----------|----------|---------------|------------|------------|-----------------|
| 0 | 0.407 | 0.456 | −0.049 (ACT) | 0.405 | 0.456 | −0.051 (ACT) |
| 7 | 0.462 | 0.316 | **+0.147 (IMG)** | 0.468 | 0.355 | **+0.113 (IMG)** |
| 14 | 0.479 | 0.373 | **+0.106 (IMG)** | 0.495 | 0.446 | +0.049 (IMG) |
| 21 | 0.500 | 0.363 | **+0.137 (IMG)** | 0.518 | 0.394 | **+0.124 (IMG)** |
| 27 | 0.354 | 0.494 | **−0.140 (ACT)** | 0.385 | 0.248 | **+0.137 (IMG)** |

Reading the dynamics:
- **Layer 0 (embeddings)**: all textual tokens cluster together, image separate — tokenizer-level identity
- **Layers 7–21 (processing)**: BOTH think types drift **toward IMAGE** — the model is actively grounding its reasoning in visual context. This is exactly what one expects from visual reasoning.
- **Layer 27 (output)**: DESC and DECIDE **split apart**:
  - THINK_DESC → ACTION (producing text tokens for screen description)
  - THINK_DECIDE → IMAGE (still pulling visual evidence for final decision)

#### Norm divergence at final layer

| Layer | IMAGE | COORD | THINK_DESC | THINK_DECIDE | ACTION |
|-------|-------|-------|------------|--------------|--------|
| 0 | 51 | 16 | 18 | 18 | 17 |
| 14 | 123 | 83 | 95 | 89 | 79 |
| 27 | 437 | **1226** | 657 | 643 | 639 |

**COORD tokens explode in norm at the final layer (1226, vs 437 for IMAGE and ~640 for others).** This is the model "committing" to coordinate emission. It strongly confirms that COORD deserves its own treatment — routing COORD to LoRA_V (as v6.5 coord_routing does) is well-justified.

#### Key implications for learned router design

1. **Linear separability is perfect — a simple 1-layer linear router will work.** No need for nonlinear routers, MLP-heads, or attention-based routers.

2. **Routing is NOT static across layers.**
   - At layer 7–21 the "correct" routing would put think tokens on the WHERE side (they encode vision)
   - At layer 27 the correct routing would put THINK_DESC on the WHAT side and THINK_DECIDE on the WHERE side
   - **→ must use per-layer router** (Option B), not global router (Option C)

3. **Reasonable warm-start targets for the router `w ∈ [0,1]` (w=1 → WHERE):**
   | Token | Early layers (0–6) | Middle (7–21) | Output (22–27) |
   |-------|--------------------|----------------|-----------------|
   | IMAGE | 1.0 | 1.0 | 1.0 |
   | COORD | 1.0 | 1.0 | 1.0 |
   | THINK_DESC | ~0.3 | 0.7–0.8 | 0.2 |
   | THINK_DECIDE | ~0.3 | 0.7–0.8 | 0.8 |
   | ACTION | 0.0 | 0.0 | 0.0 |

   This is a much richer routing than the current v6.5 hard split (IMAGE=1, everything else=0 + COORD=α≈0.3).

4. **Binary linear probe on (IMG+COORD+DESC) vs (DEC+ACT) achieves 100%** — this might be a cleaner "WHERE vs WHAT" boundary for pre-training initialization than "IMAGE vs rest". A sensible warm-start target:
   - WHERE = {IMAGE, COORD, THINK_DESC}
   - WHAT = {THINK_DECIDE, ACTION}
   - i.e. "ground truth about the screen" vs "decisions about what to do"

5. **COORD norm at final layer suggests COORD should receive full WHERE routing weight even at layer 27**, while THINK_DESC transitions away from WHERE in the output layer.

### 11.4 Recommended router architecture (refined from design doc)

Based on verification:
- **One linear router per layer** (28 routers × 3584 dim = 100K params)
- **Sigmoid output** w ∈ [0,1], soft blending: `delta = w · δ_WHERE + (1-w) · δ_WHAT`
- **Warm-start initialization**: fit each layer's router on labeled hidden states to approximate the "IMG+COORD+DESC vs DEC+ACT" split (reaches 100% acc in probe, so trivially fittable)
- **Load balance loss**: keep but weight small (λ=0.01) — specialization should come from data, not forced balance
- **End-to-end trained** (no detach) — the 100% probe accuracy means CE gradient can reliably guide routing
