# v9 — Credit-Weighted Cooperative LoRA

`v9` adds **per-component credit assignment** to cooperative LoRA training.
Given a sample whose GT action is known, we measure `coord_correct` and
`action_correct` **independently** (not via the standard coupled
`type_match × extract_match`) and translate the result into per-token loss
weights for the V (grounder) and A (action) branches.

This avoids the credit-blending problem of vanilla SFT/GRPO, where a sample
with a wrong coordinate but a right action type still pushes both branches
the same way.

The plan is two-step:

| Step | Trainer            | Credit signal                     | Status |
|------|--------------------|-----------------------------------|--------|
| 1    | Reweighted-SFT     | mistake-driven `(w_v, w_a) ≥ 0`    | implemented in this directory |
| 2    | Real GRPO          | signed `(r_g, r_a)` (advantage)    | uses `compute_credit()`; trainer TBD |

Step 1 is teacher-forced on GT, so all weights are non-negative — sign
is encoded by *which token gets which weight*, not by the weight itself.
Step 2 (real GRPO) will use the signed table from `compute_credit()` once
the verl rollout loop is wired in.

---

## Files

```
v9/
├── credit_assignment.py            # core: parse, analyze, credit table
├── decode_credits.py               # offline pipeline: model → credit_cache.jsonl
├── train_v9.py                     # step-1 Reweighted-SFT trainer
├── credits/                        # generated credit caches
│   └── v6_5_ac_ep4/
│       ├── credit_cache_shard{N}.jsonl   # one per slurm array task
│       └── credit_cache.jsonl            # merged
└── scripts/
    ├── decode_v9_ac_credits.slurm  # 8-shard array, decodes ac_train_thought.jsonl
    ├── merge_credit_shards.sh      # consolidate shards → credit_cache.jsonl
    ├── train_v9_ac_comm.slurm      # step-1 SFT, cooperative_comm=ON
    └── train_v9_ac_nocomm.slurm    # step-1 SFT, cooperative_comm=OFF
```

---

## Credit table

Correctness is measured **independently**:

- `coord_correct`: bbox-hit (with 1.2× enlarge) OR pixel-distance ≤ 0.04
  in normalized [0,1] coordinates. Defined only when the GT action has a
  meaningful coordinate.
- `action_correct`: GT action type matches AND non-coord params (button,
  text) match (lenient substring).
- `has_coord`: GT action ∈ `{click, long_press, swipe}`.

### Step-2: signed credit (real GRPO)

`compute_credit(coord_correct, action_correct, has_coord) → (r_g, r_a)`

```
has_coord = True (click / long_press / swipe):
    coord_correct  action_correct | r_g     r_a
    T              T              | +1.0    +1.0
    T              F              | +1.0    -1.0     A's fault
    F              T              | -1.0    +1.0     G's fault, A saved
    F              F              | -1.0    -0.5     G's main fault, A excused

has_coord = False (wait / system_button / type / open / answer / key / terminate):
    action_correct                | r_g     r_a
    T                             |  0.0    +1.0     G inapplicable
    F                             |  0.0    -1.0
```

### Step-1: mistake-driven SFT weights (this directory)

`compute_sft_weights(coord_correct, action_correct, has_coord) → (w_v, w_a)`

```
has_coord = True:
    coord_correct  action_correct | w_v     w_a
    T              T              | 0.3     0.3       light touch on both
    T              F              | 0.3     1.0       fix A only
    F              T              | 1.0     0.3       fix V only
    F              F              | 1.0     0.75      fix V; A partially excused

has_coord = False:
    action_correct                | w_v     w_a
    T                             | 0.0     0.3
    F                             | 0.0     1.0
```

Rationale:
- Under teacher-forcing the gradient already moves toward GT, so a positive
  weight always says *"learn this more"*. We can't represent *"learn this
  less"* with a negative scalar — we just down-weight to 0.3 instead.
- Non-coord actions: V is given `w_v=0` so the grounder isn't penalized
  or rewarded for coordinates that don't exist. Cleaner than letting it
  absorb noise.
- The (F, F) row gives A a partial weight (0.75) because the wrong
  coordinate fed by V is partly to blame for A being wrong.
- With an empty credit cache and `--credit_fallback_w 1.0`, v9 reduces
  exactly to vanilla per-token-mean SFT.

---

## Token routing (where the weights apply)

v9 requires `--coord_routing`. Inside `CreditCooperativeTrainer.compute_loss`:

1. Build the wrapper's routing mask (image tokens + coord-digit tokens
   route to LoRA_V; everything else routes to LoRA_A) and push to
   `coop_modules`.
2. Forward `base_model(..., labels=None)` → raw logits.
3. Causal-shift, compute per-token CE with `reduction="none"`.
4. Mark coord-digit positions in the assistant span via
   `wrapper._mark_coord_tokens(..., input_ids)`. This gives:
   - `is_v_token = is_assistant & coord_mask` (the digits inside `(x, y)`)
   - `is_a_token = is_assistant & ~coord_mask` (everything else the
     assistant says: thought text, action JSON keys, type names, …)
5. `weight_per_tok = w_v · is_v_token + w_a · is_a_token`
6. `L_act = (CE * weight_per_tok).sum() / n_assistant_tokens`

The assistant `<thought>` text is currently classified as A (because it
has no coord digits). This matches the v6.5 routing semantics. Under
`--coop_reasoning_alpha > 0` reasoning tokens become a soft mix; v9's
weighting still applies via `is_v_token` / `is_a_token`.

`--bind_weight` MUST be 0 — the v9 trainer does not run the second
forward needed for `L_bind`. Both checks are enforced at startup.

---

## Pipeline

```
                          (v6.5 ep4 ckpt)
                                │
                                ▼
        ┌──────────────────────────────────────────────┐
        │  decode_v9_ac_credits.slurm  (array 0-7)     │
        │  decode_credits.py per shard                 │
        │  output: credit_cache_shard{N}.jsonl         │
        └──────────────────────────────────────────────┘
                                │
                                ▼
        ┌──────────────────────────────────────────────┐
        │  merge_credit_shards.sh                      │
        │  output: credit_cache.jsonl  + summary.json  │
        └──────────────────────────────────────────────┘
                                │
                ┌───────────────┴───────────────┐
                ▼                               ▼
   train_v9_ac_comm.slurm           train_v9_ac_nocomm.slurm
   (cooperative_comm=ON)            (cooperative_comm=OFF)
                │                               │
                ▼                               ▼
       v9/output/v9_ac_comm/           v9/output/v9_ac_nocomm/
```

Run order:

```bash
# 1. Decode credits (~1.5h with 8 GPUs).
sbatch v9/scripts/decode_v9_ac_credits.slurm

# 2. After all shards finish, merge.
bash v9/scripts/merge_credit_shards.sh

# 3. Launch both trainings (head-to-head).
sbatch v9/scripts/train_v9_ac_comm.slurm
sbatch v9/scripts/train_v9_ac_nocomm.slurm
```

---

## Key design decisions

| Decision                                      | Choice                          | Reason |
|-----------------------------------------------|---------------------------------|--------|
| Credit semantics                              | independent `coord_correct` × `action_correct` | makes the matrix observable; standard matcher couples them |
| Step-1 negative-credit handling               | mistake-driven `w ≥ 0`           | teacher-forcing can't represent "push away" with a sign |
| Non-click grounder credit                     | `w_v = 0`                        | no meaningful coord → no signal to absorb |
| `(F, F)` action weight                        | 0.75 (not 1.0)                   | wrong action partly inherited from wrong coord |
| Token routing                                 | hard (V = coord digits, A = rest) | makes credit alignment unambiguous |
| `cooperative_comm` gates                      | run BOTH on/off                  | head-to-head ablation under v9 weighting |
| Warm-start                                    | from v6.5 ac ep4                 | start from a model whose generations are good enough that credits aren't all (F, F) |
| `--bind_weight`                               | 0 (enforced)                     | second forward not needed; saves ~2× compute |
| Dataset                                       | `ac_train_thought.jsonl` (single-turn) | one (w_v, w_a) per sample is the natural granularity |
| Learning rate                                 | 5e-6 (vs 1e-5 for v6.5)          | smaller because we're refining, not training from scratch |

---

## Diagnostics emitted by `train_v9.py`

Beyond what `train_cooperative.py` logs, the v9 trainer adds:

- `v_loss`: unweighted mean CE on V-tokens only
- `a_loss`: unweighted mean CE on A-tokens only
- `w_v_mean`, `w_a_mean`: average weights actually applied this logging window

If `v_loss` drops faster than `a_loss` while `w_v_mean > w_a_mean` you're
seeing the credit signal do its job: V is being pushed harder *because*
it was the bigger source of mistakes, and it's responding.

---

## TODO (step 2)

`compute_credit()` already returns `(r_g, r_a)` for the GRPO loop. The
remaining work is:

- Wire `cooperative_generate` into a verl rollout loop, group-relative
  advantage with `r_g` for V and `r_a` for A (per-token via the same
  hard-routing mask used here).
- Reuse `analyze_correctness()` so step-1 and step-2 see the same
  correctness definition.
