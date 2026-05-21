# V13 Experiment Results: Integrated Report

**Date**: 2026-04-28

All experiments run on V13 epoch-3 cooperative model (18.7% TSR), 968-episode GUI-360 test set.

---

## 1. Gate Analysis Progressive Campaign

### 1.1 V1: Global Average (200 episodes, Job 4355378)

Gate mean across all layers/tokens/modules: **~0.510**, no differentiation by action type, success, or trajectory position.

| Slice | Gate Mean |
|-------|-----------|
| click / type / swipe | 0.5105 / 0.5107 / 0.5105 |
| success / failure | 0.5104 / 0.5102 |
| Round 0: g12/g21 | 0.5050 / 0.5075 |
| Round 1: g12/g21 | 0.5111 / 0.5188 |

**Conclusion**: Global averaging masks all meaningful signal.

### 1.2 V2: Per-Layer (100 episodes, Job 4367335)

Routing weights reveal dramatic per-layer specialization:

| Layer | Routing r (to Expert 1) | Interpretation |
|-------|------------------------|----------------|
| L07 | 0.126 | Expert 2 dominant |
| L10 | 0.897 | Expert 1 dominant |
| L18 | 0.955 | Expert 1 dominant |
| L21 | 0.956 | Expert 1 dominant |
| L27 | 0.005 | Expert 2 dominant |

High-norm layers (L10, L18, L27) have gate std ~0.096; low-norm layers (L08, L13, L23) have gate std ~0.006 (essentially inactive). Still no action-type differentiation at this granularity.

### 1.3 V3: Token-Level — Image vs Text (20 episodes, Job 4367483)

Gates encode **modality**, not action type:

| Layer | Image Gate | Text Gate | Diff | Role |
|-------|-----------|-----------|------|------|
| L10 | 0.5945 | 0.5510 | **+0.044** | Visual-semantic alignment |
| L18 | 0.4273 | 0.4939 | **-0.067** | Instruction understanding |
| L27 | 0.5033 | 0.5135 | -0.010 | Minimal |

- L10 and L18 show **opposite** modality preferences
- Directional asymmetry: L10 image g21=0.66 >> g12=0.53 (Expert 2->1 for images)
- Gate range is large: L10 [0.36, 0.80], L18 [0.22, 0.74] — per-token variation is substantial

### 1.4 V4: Reasoning Phase During Generation (968 episodes, Jobs 4393782/4395522)

Gates are **phase-dependent** — the "X-crossing" pattern:

| Phase | L10 | L18 | L27 |
|-------|-----|-----|-----|
| planning | **0.594** | 0.445 | 0.507 |
| action_start | 0.550 | 0.467 | **0.467** |
| action_type | 0.577 | **0.433** | 0.496 |
| coordinate | 0.542 | 0.474 | 0.511 |

- L10: monotonically **decreases** planning->coordinate (visual understanding front-loaded)
- L18: monotonically **increases** planning->coordinate (spatial reasoning back-loaded)
- Within-generation std: L10=0.066, L18=0.069 — gates dynamically adapt per-token
- Cross-episode variance: planning most variable (std=0.023), action_type most consistent (std=0.008)

---

## 2. Gate Signature Analysis (968 episodes, offline)

Planning-phase gates **predict episode success**:

| Phase | Layer | Correct | Incorrect | Diff | p-value |
|-------|-------|---------|-----------|------|---------|
| planning | L10 | 0.5985 | 0.5884 | +0.010 | 6.4e-11 |
| planning | L18 | 0.4405 | 0.4505 | -0.010 | 3.3e-11 |
| coordinate | L10 | 0.5398 | 0.5447 | -0.005 | 1.7e-11 |
| coordinate | L18 | 0.4711 | 0.4772 | -0.006 | 7.6e-05 |

**Predictive power** (median split):
- L10 planning high -> **65% correct** vs low -> **51% correct** (14-point gap)
- L18 planning low -> **66% correct** vs high -> **51% correct** (15-point gap)

Correct episodes: more L10 communication + less L18 in planning; **less** communication overall in coordinate phase (decision already made in planning).

---

## 3. Gate Perturbation (484/968 episodes, Job 4396229)

Gate perturbation has **zero effect on action type**:

| Condition | Type Changes | Mean Coord Shift |
|-----------|-------------|-----------------|
| all d=-0.5 | 0/241 | 33.7px |
| all d=+0.5 | 1/241 | 39.6px |
| L10 only d=+0.5 | 0/241 | 16.5px |
| L18 only d=+0.5 | 0/241 | 18.9px |
| L27 only d=+0.5 | 0/241 | 6.6px |

Coordinates shift (85% move at d=-0.5) but randomly, not toward correct targets. **"100% click" bias is in A/B projections, not gates** -> Direction B (gate-guided exploration) invalidated.

---

## 4. Phase-Conditional Ablation (968 episodes, Job 4401289)

Communication is **structurally essential**; planning phase is **primary driver**:

| Mode | Type Acc | Click% | Mean Dist | <50px | <100px |
|------|---------|--------|-----------|-------|--------|
| **full** | **82.4%** | **99.5%** | **170px** | **49.3%** | **61.6%** |
| no_comm | 7.0% | 7.7% | 249px | 31.4% | 41.2% |
| planning_only | 22.8% | 26.7% | 197px | 38.7% | 52.6% |
| coord_only | 7.0% | 7.7% | 250px | 31.3% | 41.4% |
| type_only | 7.2% | 8.2% | 250px | 31.3% | 41.3% |

- Disabling all comm: click 99.5% -> 7.7%, coord <50px 49.3% -> 31.4%
- Planning-only: partial recovery to 38.7% coord <50px
- Coord-only / type-only ≈ no_comm — these phases depend on planning context
- Full comm better than no comm in 54.5% of episodes

---

## 5. Forced-Prefix Decode + Logit Gap (968 episodes, Job 4401290)

Model **"won't" not "can't"** predict type/swipe:

**Forced type decode** (105 GT=type episodes):
- Mean text similarity: 0.590, median: 0.680
- Score > 0.5: 54.3%, Score > 0.8: 45.7%
- Model has learned type behavior, just never self-selects it

**Logit gap at action-type decision point**:

| GT Type | N | P(click) | Gap(click-type) |
|---------|---|----------|-----------------|
| click | 799 | 97.5% | 21.9 |
| type | 156 | 93.5% | 18.9 |
| swipe | 13 | 82.8% | 18.3 |

- Gap is 3.0 units smaller for GT=type -> model weakly recognizes type episodes
- But 18.9 gap is too massive for spontaneous selection

---

## 6. Base Model Type Distribution (968 episodes, Job 4401291)

| Prediction | Count | % |
|-----------|-------|---|
| unknown (no `<action>` tag) | 954 | 98.6% |
| click | 12 | 1.2% |
| type | 2 | 0.2% |

Base Qwen2.5-VL has no action format knowledge. **RL simultaneously taught format compliance AND created click bias** — not a "diversity collapse" but a from-scratch learned behavior.

---

## 7. Key Conclusions

1. **Gates encode modality + reasoning phase**, not action type (V1-V4)
2. **Planning-phase gates predict success** with 14-15% accuracy gap (signature analysis)
3. **Communication is structurally essential** — removing it collapses the model (ablation)
4. **Planning phase is the primary driver** — coord/type phases depend on planning context (ablation)
5. **Gate perturbation cannot fix click bias** — bias is in A/B weights (perturbation)
6. **Model CAN do type/swipe but WON'T** — logit gap of 18-22 prevents self-selection (forced prefix)
7. **Click bias is RL-created** — base model has no action format at all (base model analysis)
8. **CoPDA (V14) is the validated next step** — use cross-layer gate variance as phase signal for per-token credit assignment (Direction A validated, Direction B invalidated)
