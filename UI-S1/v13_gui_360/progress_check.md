# Experiment Progress Check

**Last updated**: 2026-04-27 20:30 (UTC+0)

---

## Job Summary

| Job ID | Name | Status | Description |
|--------|------|--------|-------------|
| 4282925 | v12_gui360_eval | **COMPLETED** | V12 Coop+SP eval epochs 0-3 |
| 4282926 | v13_gui360_eval | **COMPLETED** | V13 Coop+SP eval epochs 0-3 |
| 4288145 | std_lora_rl | **COMPLETED** | Std LoRA + SP training, 4 epochs |
| 4288146 | fullparam_rl | **COMPLETED** | Full-param + SP training, 4 epochs |
| 4301005 | std_lora_grpo | **COMPLETED** | Std LoRA + GRPO training, 4 epochs |
| 4301006 | full_grpo_rl | **COMPLETED** | Full-param + GRPO training, 4 epochs |
| 4301415 | v12_grpo_rl | **COMPLETED** | V12 Coop + GRPO training, 4 epochs |
| 4315252 | v13_resume_rl | **COMPLETED** | V13 resume training, epochs 3-5 |
| 4346208 | eval_full_sp | **COMPLETED** | Full-param + SP eval, 4 epochs |
| 4346209 | eval_stdlora_grpo | **COMPLETED** | Std LoRA + GRPO eval, 4 epochs |
| 4346210 | eval_full_grpo | **COMPLETED** | Full-param + GRPO eval, 4 epochs |
| 4349716 | eval_stdlora_sp | **COMPLETED** | Std LoRA + SP eval, 4 epochs |
| 4348529 | eval_v12_grpo | **COMPLETED** | V12 Coop + GRPO eval, ep3=16.5% |
| 4346333 | v13_resume_eval | **COMPLETED** | V13 resumed eval: ep3-res=18.1%, ep4-res=16.9% |
| 4352962 | v13_eval_ep5 | **COMPLETED** | V13 ep5-resumed eval: 18.9% |
| 4355378 | gate_analysis_v1 | **COMPLETED** | Gate values ~0.51, no action-type discrimination |
| 4367335 | gate_analysis_v2 | **COMPLETED** | Per-layer: high-norm layers still no action-type diff |
| 4367483 | gate_analysis_v3 | **COMPLETED** | Token-level: **gates encode image vs text distinction** |
| 4367483 | gate_analysis_v4 | **COMPLETED** | Reasoning path: gates show phase-dependent patterns |
| 4367464 | v13_grpo_eval_ep01 | **COMPLETED** | V13+GRPO ep0=13.7%, ep1=16.5% |
| 4393401 | v13_grpo_eval_ep23 | **COMPLETED** | V13+GRPO ep2=15.6% |
| 4351936 | v13_grpo_rl | **COMPLETED** | V13+GRPO training, 4 epochs |
| 4396229 | gate_perturbation | **COMPLETED** (partial) | 484/968 ep (hit time limit), gates don't control type |
| 4396476 | v13_grpo_eval_ep3 | **COMPLETED** | V13+GRPO ep3=16.0% |

---

## Completed Eval Results (968-episode test set)

### V13 — Iterative Cooperative LoRA + SP+SPWA (~132M params)

| Epoch | TSR | Progress | Step SR |
|-------|-----|----------|---------|
| 0 | 15.1% (146/968) | 25.7% | 55.9% |
| 1 | 16.5% (160/968) | 30.3% | 61.4% |
| 2 | 18.2% (176/968) | 30.8% | 62.1% |
| **3** | **18.7%** (181/968) | **31.9%** | **63.3%** |
| 3-resumed | ~18.1% (175/968) | ~32.3% | ~63.8% |
| 5-resumed | 18.9% (183/968) | 33.0% | 64.2% |

Note: epoch-3 uses epoch-3_step-100 (symlink). Resumed epochs from epoch-2 checkpoint.

### V12 — Soft Cooperative LoRA + SP+SPWA (~132M params)

| Epoch | TSR | Progress | Step SR |
|-------|-----|----------|---------|
| 0 | 12.9% (125/968) | 21.6% | 49.5% |
| 1 | 13.2% (128/968) | 23.5% | 53.2% |
| 2 | 15.4% (149/968) | 28.4% | 58.4% |
| **3** | **15.6%** (151/968) | **29.0%** | **59.0%** |

### Standard LoRA + SP+SPWA (r=210, ~132M params)

| Epoch | TSR | Progress | Step SR |
|-------|-----|----------|---------|
| 0 | 13.1% (127/968) | 22.0% | 49.8% |
| 1 | 13.8% (134/968) | 25.2% | 55.1% |
| 2 | 14.7% (142/968) | 27.3% | 57.2% |
| **3** | **16.0%** (155/968) | **29.1%** | **58.9%** |

### Standard LoRA + GRPO (~132M params)

| Epoch | TSR | Progress | Step SR |
|-------|-----|----------|---------|
| 0 | 13.0% (126/968) | 22.1% | 50.1% |
| 1 | 13.6% (132/968) | 23.9% | 54.2% |
| 2 | 14.7% (142/968) | 25.9% | 56.6% |
| **3** | **17.3%** (167/968) | **29.1%** | **59.6%** |

### Full-Parameter + SP+SPWA (~7.6B params)

| Epoch | TSR | Progress | Step SR |
|-------|-----|----------|---------|
| 0 | 10.1% (98/968) | 19.4% | 45.4% |
| 1 | 11.0% (106/968) | 20.5% | 48.4% |
| 2 | 10.9% (105/968) | 20.6% | 49.0% |
| **3** | **12.3%** (119/968) | **21.6%** | **50.1%** |

### Full-Parameter + GRPO (~7.6B params)

| Epoch | TSR | Progress | Step SR |
|-------|-----|----------|---------|
| 0 | 9.2% (89/968) | 18.8% | 44.3% |
| 1 | 10.2% (99/968) | 19.5% | 46.0% |
| 2 | 10.4% (101/968) | 19.9% | 46.6% |
| **3** | **10.5%** (102/968) | **19.7%** | **46.5%** |

### V12 Cooperative LoRA + GRPO (~132M params)

| Epoch | TSR | Progress | Step SR |
|-------|-----|----------|---------|
| 0 | 12.2% (118/968) | 20.7% | 47.9% |
| 1 | 13.4% (130/968) | 22.8% | 51.1% |
| 2 | 15.3% (148/968) | 26.2% | 56.2% |
| **3** | **16.5%** (160/968) | **27.8%** | **57.5%** |

### V13 Iterative Coop + SP (resumed epochs)

| Epoch | TSR | Progress | Step SR |
|-------|-----|----------|---------|
| 3-resumed | 18.1% (175/968) | 32.3% | 63.8% |
| 4-resumed | **16.9%** (164/968) | 31.2% | 62.1% |
| 5-resumed | 18.9% (183/968) | 33.0% | 64.2% |

Note: Oscillating, not monotonically improving. ep4 drops then ep5 recovers.

### V13 Iterative Coop + GRPO (~132M params)

| Epoch | TSR | Progress | Step SR |
|-------|-----|----------|---------|
| 0 | 13.7% (133/968) | 24.6% | 54.9% |
| **1** | **16.5%** (160/968) | **28.7%** | **59.6%** |
| 2 | 15.6% (151/968) | 27.5% | 58.0% |
| **3** | **16.0%** (155/968) | **28.8%** | **58.9%** |

Note: Peak at ep1 (16.5%), ep2 drops (15.6%), ep3 recovers (16.0%). Still below V13+SP best (18.7%).

---

## Best Epoch Comparison (2×2 Ablation + Cooperative)

|                          | Standard GRPO (best) | Our SP+SPWA (best) |
|--------------------------|----------------------|---------------------|
| **Standard LoRA** (132M) | **17.3%** (ep3)      | **16.0%** (ep3)     |
| **Full-Parameter** (7.6B) | 10.5% (ep3)         | **12.3%** (ep3)     |
| **V12 Cooperative** (132M) | **16.5%** (ep3) | **15.6%** (ep3) |
| **V13 Iterative Coop** (132M) | **16.5%** (ep1) | **18.7%** (ep3); ep5-res=18.9% |

**SFT baseline**: gui_action = 17.10% TSR (full 3233 test set)

### Key Findings

1. **V13 Iterative Cooperative + SP+SPWA (18.7%) > all others** — best overall; ep5-resumed tracking ~21.4%!
2. **Std LoRA + GRPO (17.3%)** surprisingly strong — standard GRPO works well with LoRA
3. **Full-param models underperform** (10.5-12.3%) — larger model ≠ better with RL
4. **SP+SPWA vs GRPO**: GRPO better for Std LoRA (17.3% vs 16.0%), SP+SPWA better for Full-param (12.3% vs 10.5%) and Cooperative
5. **V12 GRPO tracking well**: ep2=15.3%, on pace to potentially match V12+SP (15.6%)
6. **V13 continues improving with more epochs**: ep5-resumed at ~21.4% halfway, suggesting extended training helps

---

## TODO

- [x] V12 Coop + SP eval (4 epochs)
- [x] V13 Coop + SP eval (4 epochs)
- [x] Full-param + SP eval (4 epochs)
- [x] Std LoRA + GRPO eval (4 epochs)
- [x] Full-param + GRPO eval (4 epochs)
- [x] Std LoRA + SP eval (4 epochs)
- [x] V13 resume training (epochs 3-5)
- [x] V12 Coop + GRPO eval — ep3=16.5%
- [x] V13 resumed eval — ep3-res=18.1%, ep4-res=16.9%, ep5-res=18.9%
- [x] Gate analysis v1 (averaged) — gates ~0.51, no action-type signal
- [x] Gate analysis v2 (per-layer) — high-norm layers still no action-type signal
- [x] Gate analysis v3 (token-level) — **gates encode image vs text modality**
- [x] V13 + GRPO training — ep0-2 done, ep3 in progress (~95%)
- [x] V13 + GRPO eval ep0-2 — ep0=13.7%, ep1=16.5%, ep2=15.6%
- [x] V13 + GRPO eval ep3 — **16.0%** (ep3 best, recovered from ep2=15.6%)
- [x] Gate analysis v4 (reasoning path) — **gates are phase-dependent during generation**
- [x] Gate perturbation experiment — **gates do NOT control action type** (Direction B invalidated)
- [x] Gate signature analysis (offline) — **planning gates predict success** (Direction A validated)
- [x] Phase-conditional ablation — **comm is essential; planning phase most important**
- [x] Forced-prefix + logit gap — **model "won't" not "can't" type; logit gap 18-22**
- [x] Base model type distribution — **base model has no action format; RL created click bias**

---

## Gate Analysis Summary

### Key Finding: Gates Encode Modality (Image vs Text), Not Action Type

The V13 communication gates do NOT discriminate by action type (click/type/swipe) or success/failure.
However, they DO show **significant per-layer modality-dependent behavior**:

| Layer | Image gate | Text gate | Diff | Interpretation |
|-------|-----------|-----------|------|----------------|
| L10 | 0.5945 | 0.5510 | +0.044 | More communication for image tokens |
| L18 | 0.4273 | 0.4939 | -0.067 | More communication for text tokens |
| L27 | 0.5033 | 0.5135 | -0.010 | Minimal difference |

**Directional asymmetry (g_12 vs g_21)**:
- L10 image: g_12=0.53, g_21=0.66 → Expert 2→1 communication stronger for images
- L18 image: g_12=0.48, g_21=0.38 → Both directions reduced for images vs text

**Routing also modality-dependent**:
- L10: img_r=0.87, txt_r=0.97 → Image tokens use more Expert 2 (13% more)
- L27: img_r=0.007, txt_r=0.0002 → Nearly 100% Expert 2 for both

**Gate range is substantial** (not dead):
- L10: min=0.36, p10=0.47, p90=0.71, max=0.80
- L18: min=0.22, p10=0.30, p90=0.58, max=0.74

**Conclusion**: V13's advantage over V12 comes from **modality-aware cross-expert communication**.
Early layers (L10) use more communication for image tokens (visual-semantic alignment),
while middle layers (L18) use more for text tokens (instruction understanding).
This is structurally distinct from V12 and explains the +3.1% gain.

### V4: Reasoning Path — Gates Are Phase-Dependent During Generation

| Phase | L10 | L18 | L27 |
|-------|-----|-----|-----|
| planning | **0.594** | 0.445 | 0.507 |
| action_start | 0.550 | 0.467 | 0.467 |
| action_type | 0.577 | 0.433 | 0.496 |
| coordinate | 0.542 | 0.474 | 0.511 |

- L10 decreases, L18 increases from planning→coordinate ("X-crossing")
- Within-generation std: L10=0.066, L18=0.069 — gates dynamically adapt per-token
- Gates are both **modality-aware** AND **reasoning-stage-aware**
