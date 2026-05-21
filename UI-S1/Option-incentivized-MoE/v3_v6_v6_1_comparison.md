# Cooperative LoRA: v3 vs v6 vs v6.1 Comparison

Purpose: side-by-side comparison of the three cooperative LoRA variants we've trained, prepared before the v6 / v6.1 thought proper-routing evals finish so we can interpret the numbers quickly.

## 1. Architecture

| Component | v3 | v6 | v6.1 |
|---|---|---|---|
| num_agents | 2 | 2 | 2 |
| LoRA rank `r` | **256** | 128 | 128 |
| LoRA alpha | **512** | 256 | 256 |
| scaling (α/r) | 2.0 | 2.0 | 2.0 |
| target_modules | q/k/v/o, gate/up/down | same | same |
| dropout | 0.05 | 0.05 | 0.05 |
| Per-token hard routing (image→V, text→A) | yes | yes | yes |
| **`cooperative_comm`** | **no** | **yes** | **yes** |
| W_av / W_va (r×r per layer per module) | – | yes | yes |
| gate_av / gate_va (scalar per layer per module) | – | yes | yes |
| bind_weight / soft_routing | 0 / off | 0 / off | 0 / off |

**Key architectural difference**: v3 is **pure 2-LoRA hard routing** (two independent LoRAs, no cross-LoRA communication). v6/v6.1 **halve the rank** and add per-layer communication `h_v += g_av · W_av·h_a`, `h_a += g_va · W_va·h_v` in the LoRA latent space.

Trainable param count:
- v3: **1,291,845,632** (≈1.29 B) — only LoRA_V + LoRA_A at r=256
- v6 / v6.1: **652,345,736** (≈0.65 B) = lora_v 322,961,408 + lora_a 322,961,408 + comm 6,422,920
  - Communication overhead: 6.4 M ≈ **1 %** of total trainable params
  - Half the LoRA capacity but adds a new mechanism

## 2. Communication parameters (v6 only, v3 has none)

- 28 layers × 7 target modules × (W_av + W_va + gate_av + gate_va)
- W_av / W_va: r × r = 128 × 128 = 16,384 params each, kaiming init
- gate_av / gate_va: scalars, initialized to `gate_init` in logit space
- Total comm params = 28 × 7 × 2 × (128² + 1) = 6,422,920

## 3. Training hyperparameters (identical data)

All three use the same 97,647 GUI-360 thought samples, same val split, same lr=1e-5, same 2 epochs, same target modules.

| Setting | v3 | v6 | v6.1 |
|---|---|---|---|
| Nodes × GPUs | **4** × 4 = 16 | **8** × 4 = 32 | **8** × 4 = 32 |
| per_device_batch_size | 1 | 1 | 1 |
| grad_accum | 8 | 8 | 8 |
| **effective batch** | **128** | **256** | **256** |
| **Total opt steps** | **1,525** | **762** | **762** |
| gate_init (logit) | – | −1.5 (sig=0.1824) | **0.0 (sig=0.5)** |
| gate_lr_multiplier | – | 1.0 | **10.0** |
| gate_weight_decay | – | 0.1 (default) | **0.0** |

v3 gets **2× more optimizer steps** (1,525 vs 762) because of its smaller effective batch. Same data passes, half the gradient updates. This is a meaningful capacity differential on its own.

## 4. v6.1 delta over v6 (optimization-only)

All code/architecture identical between v6 and v6.1. Only changes are in `CooperativeTrainer.create_optimizer()`:

1. **`gate_init`**: −1.5 → 0.0 — sits in the fat part of the sigmoid where ∂σ/∂x = 0.25 (vs 0.148 at σ=0.18); also avoids weight-decay bias toward σ=0.
2. **`gate_lr_multiplier`**: 10× lr for `W_av / W_va / gate_av / gate_va`.
3. **`gate_weight_decay`**: 0 for comm params (disables the "shrink gate to 0" drag).

No per-layer hand-coded priors — all layers treated uniformly, differentiation emerges from training.

## 5. Observed gate dynamics (from `cooperative_config.json` at `final/`)

| Metric | v6 (init −1.5, lr×1, wd 0.1) | v6.1 (init 0, lr×10, wd 0) |
|---|---|---|
| init sigmoid | 0.1824 | 0.5000 |
| final gate_av mean | 0.1828 | **0.5045** |
| drift from init (mean) | **+0.0004** | **+0.0045** (~11×) |
| gate_av spread (max − min) | 0.0008 | **0.0109** (~14×) |
| gate_va spread | 0.0007 | **0.0093** (~13×) |
| peak layer (gate_av) | L23 (0.1829) | L16 (0.5060) |
| L0 vs L16–23 differential | ~0.0002 | **~0.003–0.006** |

**Interpretation**:
- v6 gates **barely moved**: 0.0004 sigmoid drift is essentially only weight-decay pull and noise. Communication mechanism contributed ~0.18 × (W·h) activation at init but learned almost nothing on top.
- v6.1 gates moved **~11–14× further** in sigmoid space, and the per-layer inverted-U pattern is **clearly visible** (middle layers L11–22 do more cross-modality binding than shallow/deep). This is exactly the "more signal to learn per-layer differentiation" behavior we wanted.
- However: even v6.1 gate spread of 0.01 in sigmoid space is still small. The comm contribution never exceeded ~1 % of the V/A-only baseline in magnitude.

## 6. Final eval results (proper routing, action_prediction, GUI-360 test, 19,046 samples)

| Config | Rate | success / total | vs v3 | vs v6_nothought |
|---|---|---|---|---|
| **v3 thought ep2** (r=256, 1525 steps, **no comm**) | **46.11 %** | 8782 / 19046 | — | +4.29 |
| **v6.1 thought** (r=128, 762 steps, comm, **strong gate**) | **43.64 %** | 8312 / 19046 | **−2.47** | **+1.82** |
| **v6 thought** (r=128, 762 steps, comm, weak gate) | **42.31 %** | 8059 / 19046 | −3.80 | +0.49 |
| v6 nothought (r=128, 762 steps, comm, weak gate) | 41.82 % | 7965 / 19046 | −4.29 | — |

Per-shard spread is wide (worst shard ~39 %, best ~48 %), so shard-0-only snapshots were misleading — aggregate is what matters.

### Key deltas

- **v6.1 − v6 = +1.33 pt** → optimization fix (gate_init 0, lr ×10, wd 0) **did** translate into downstream accuracy. Consistent with the 11× larger gate drift and 14× larger gate spread observed at training time.
- **v6 thought − v6 nothought = +0.49 pt** → thought format is marginally useful in the v6 architecture; v6.1 widens this to **+1.82 pt**. Earlier shard-0 fear that "thought hurts" was a statistical artifact.
- **v6.1 − v3 = −2.47 pt** → v6.1 does NOT match v3 at this rank / step budget. But see confounders below.

### Decision tree outcome

- **A (v6 thought < nothought)**: **NO** — v6 thought (42.31) > v6 nothought (41.82). Thought helps, just a little.
- **B (v6.1 > v6)**: **YES** — +1.33 pt. Optimization fix confirmed valuable.
- **C (v6.1 vs v6 nothought 41.82)**: **v6.1 > v6 nothought** by +1.82 pt. v6 architecture benefits from thought data when gate gets enough signal.
- **D (v6.1 vs v3 46.11)**: v6.1 is **−2.47 pt below v3**, landing in the "within a few points" zone, not matching.

### Three-variable confound is the main caveat

v6.1 vs v3 differs in **three** things at once:
1. **rank**: 256 → 128 (half LoRA capacity)
2. **optimizer steps**: 1525 → 762 (half gradient updates; same data, larger effective batch)
3. **cooperative_comm**: off → on (+1 % trainable params)

Any of (1) or (2) alone could plausibly cause a 2–3 pt gap without touching architecture. The fact that v6.1 lands **only 2.47 pt below** v3 despite (1)+(2) both working against it is actually **encouraging** for the comm mechanism — it suggests comm partially compensates for lost capacity/steps.

Fair isolation test would be: **v6.1 at r=256 + 1525 steps** vs **v3 at r=256 + 1525 steps** — same capacity, same optimization budget, only `cooperative_comm` differs. That's the experiment worth running next.

## 7. What the numbers will NOT tell us

- Whether the comm mechanism would shine on longer-horizon reasoning (thought data here is shallow, ~442 chars).
- Whether a different comm placement (e.g. only MLP-up / only attention) would work better.
- Whether per-layer gate routing should be made explicit (top-k over modules) instead of dense.

These need separate follow-ups. For now, the experiment is about: **does the v6 idea at least hold its own on action prediction**.

## 8. Quick reference: config paths

- v3: `train_GUI_360/llamafactory/output/cooperative_thought_v3/`
- v6: `train_GUI_360/llamafactory/output/cooperative_v6_comm_thought/`
- v6.1: `train_GUI_360/llamafactory/output/cooperative_v6_1_comm_thought/`

Eval results (finished 2026-04-10):
- v6 thought proper (42.31 %): `train_GUI_360/GUI-360-eval/results/cooperative_v6_thought_proper/action_prediction/`
- v6.1 thought proper (43.64 %): `train_GUI_360/GUI-360-eval/results/cooperative_v6_1_thought_proper/action_prediction/`

Eval jobs: 3724586 (v6), 3724587 (v6.1). Both 1 node × 4 GPU × 4 shards, ~1 h 14 min wall time each.

## 9. Next experiment — isolation test

To disentangle rank / steps / comm:

**v6.2 thought**: v6.1 setup but at **r=256, α=512** and **4 nodes (eff_bs=128)** to match v3's step count.

- Same data, same learning rate, same epochs → 1525 optimizer steps
- Same rank, same capacity as v3
- Only difference vs v3: `cooperative_comm=True` + v6.1 optimizer recipe (gate_init=0, lr×10, wd=0)
- Expected cost: ~2× v6.1 training time (more params, more steps)
- Expected outcome: if **v6.2 > v3 (>46.11)**, comm mechanism clearly valuable. If **v6.2 ≈ v3**, comm is neutral at this scale. If **v6.2 < v3**, comm actively hurts (unlikely given v6.1 showed gate movement correlated with accuracy gain).
