# Plan B: Coordinate Routing — Complete Processing Flow

## 1. Problem Discovery

### Cooperative Advantage (CA) Ablation Results on AC

| Mode | TSR | Description |
|------|-----|-------------|
| hard | 10.11% (156/1543) | v6.5 default: img→V, txt→A |
| t_only | ~10.41% (158/1518) | All tokens → LoRA_A |
| v_only | 7.26% (112/1543) | All tokens → LoRA_V |
| Base model | 8.88% (137/1543) | No LoRA |
| Vanilla SFT | 5.25% (81/1543) | Standard single LoRA |

**CA = acc_hard - max(acc_v_only, acc_t_only) ≈ 10.11% - 10.41% ≈ -0.3%**

Cooperative routing provides NO benefit. LoRA_A alone is slightly better than hard routing.

### Root Cause: Asymmetric Gradient Flow

```
Training flow:
  Input:  [image_tokens] + [text_tokens]
  Routing: image_tokens → LoRA_V,  text_tokens → LoRA_A
  Labels:  only assistant response tokens (all TEXT) have labels != -100
  CE loss: ONLY on assistant tokens → ALL through LoRA_A

  LoRA_A: ✅ Direct CE loss gradient
  LoRA_V: ❌ ZERO direct CE loss (image tokens are input-only, never predicted)
           Only gets indirect gradient through attention cross-terms
```

### Problem Severity Across Datasets

| Dataset | Image tokens | LoRA_V direct CE loss | Severity |
|---------|-------------|----------------------|----------|
| GUI-360 | 24.4% | **0%** | Moderate |
| AC | 70.8% | **0%** | High |
| Odyssey | 80.1% | **0%** | Highest |

Full analysis: `v6.5/gradient_imbalance_analysis.md`

---

## 2. Plan B: Route Coordinate Tokens to LoRA_V

### Core Idea

Instead of routing ALL text tokens through LoRA_A, route coordinate/bbox digit tokens through LoRA_V. This gives LoRA_V direct CE loss gradient on spatial/positional tokens.

### Semantic Division

```
LoRA_V (Visual/Spatial agent):
  - Image tokens (input-only, indirect gradient)
  - Coordinate digits in actions (CE loss! direct gradient)
  - Bbox digits in actions (CE loss! direct gradient)
  → "Where to LOOK + Where to CLICK"

LoRA_A (Action/Semantic agent):
  - <think> reasoning tokens
  - Action type tokens ("click", "swipe", "open", etc.)
  - Text content tokens
  → "What to THINK + What to DO"
```

### Token Budget Change (AC)

```
Before (v6.5):
  LoRA_V CE loss: 0 tokens (0%)
  LoRA_A CE loss: 1,863,599 tokens (100%)

After (coord_routing):
  LoRA_V CE loss: 181,742 tokens (9.8%)   ← 0% → 9.8% qualitative change
  LoRA_A CE loss: 1,681,857 tokens (90.2%)
```

---

## 3. AC Data Structure Analysis

### Action Format

```json
<think>... long chain-of-thought reasoning ...</think>
<action>{"action": "click", "coordinate": [540.0, 389.8], "bbox": [360, 327, 695, 466]}</action>
```

### Action Type Distribution (6,165 total actions)

| Action | Count | Has coords | Has bbox |
|--------|-------|-----------|----------|
| click | 3,191 (51.8%) | Yes | 2,546 |
| swipe | 721 (11.7%) | Yes (×2) | No |
| terminate | 950 (15.4%) | No | No |
| type | 372 (6.0%) | No | No |
| open | 354 (5.7%) | No | No |
| wait | 352 (5.7%) | No | No |
| system_button | 217 (3.5%) | No | No |
| long_press | 8 (0.1%) | Yes | No |

**63.6% of actions have coordinates** → LoRA_V gets CE loss on majority of action turns.

### Tokenization of Coordinates

Qwen2.5-VL tokenizes coordinates as individual digit tokens:
```
"coordinate": [540.0, 389.8]
  →  'coordinate'(62526) '":'(788) ' ['(508)
     '5'(20) '4'(19) '0'(15) '.'(13) '0'(15)   ← LoRA_V
     ','(11) ' '(220)                             ← LoRA_V
     '3'(18) '8'(23) '9'(24) '.'(13) '8'(23)     ← LoRA_V
     '],'(1125)
```

Key token IDs:
- `coordinate` → 62526 (single token, trigger)
- `bbox` → 58456 (single token, trigger)
- Digits 0-9 → ids 15-24
- `.` → 13, `,` → 11, ` ` → 220
- `coordinate2` (swipe) → 62526 + 17 (naturally handled)

---

## 4. Implementation

### Files Modified

#### 4.1 `verl/models/cooperative/cooperative_wrapper.py`

**Constants added** (after line 36):
```python
COORD_KEY_ID = 62526       # 'coordinate'
BBOX_KEY_ID = 58456        # 'bbox'
DIGIT_TOKEN_IDS = set(range(15, 25))  # 0-9
COORD_PUNCT_IDS = {13, 11, 220}      # '.', ',', ' '
COORD_ALL_VALUE_IDS = DIGIT_TOKEN_IDS | COORD_PUNCT_IDS
BRACKET_OPEN_ID = 508      # ' ['
BRACKET_CLOSE_IDS = {1125, 81136}  # '],', ']}'
```

**`__init__`**: Added `coord_routing: bool = False` parameter.

**`_mark_coord_tokens(mask, input_ids)`**: Scans tokenized sequence for `coordinate`/`bbox` keys, marks subsequent digit/punct tokens inside `[...]` as True (→ LoRA_V). Used during training `forward()` and generation prefill.

**`_update_coord_state(ids)`**: Token-by-token state machine for autoregressive decode. Tracks `_in_coord_region` and `_seen_bracket` per batch element. Returns `[B, 1]` bool mask.

**`_init_coord_state_from_prefill(ids)`**: Initializes coord state machine from prefill sequence (like `_init_thought_state_from_prefill`).

**`forward()`** (training): After `token_mask = (input_ids == IMAGE_PAD_ID)`, adds:
```python
if self.coord_routing:
    self._mark_coord_tokens(token_mask, input_ids)
```

**`generate()` → `_pre_hook`**: For 2-agent mode with coord_routing:
- Prefill: calls `_mark_coord_tokens` + `_init_coord_state_from_prefill`
- Decode: calls `_update_coord_state` and ORs result with image mask

#### 4.2 `evaluation/serve_corrected_vllm.py`

**Constants added**: Same COORD_* constants as wrapper.

**`_build_coord_mask_1d(input_ids)`**: 1D version of coord mask builder for vLLM's flattened token sequences.

**`CorrectedQwen25VLForConditionalGeneration.forward()`**: When `_coord_routing=True`, ORs coord_mask into the image_mask before passing to hooks:
```python
if _coord_routing:
    coord_mask = _build_coord_mask_1d(input_ids)
    mask = mask | coord_mask
```

**CLI**: Added `--coord-routing` flag.

#### 4.3 `train_cooperative.py`

**Added**: `--coord_routing` argument, passed to `CooperativeVLMWrapper()`.

### Files Created

- `scripts/exp_cooperative/train_v6_5_ac_coord_routing.slurm` — Training script (identical to v6.5 AC + `--coord_routing`)
- `scripts/exp_cooperative/eval_coop_ac_coord_routing.slurm` — Eval script (supports `COOP_EPOCH` and `INFERENCE_MODE` env vars)

---

## 5. Mask Construction Algorithm

### State Machine for Coordinate Region Detection

```
State: (in_coord, in_bracket)

Initial: (False, False)

Transitions:
  (False, *) + token ∈ {COORD_KEY, BBOX_KEY}  → (True, False)
  (True, False) + token == BRACKET_OPEN        → (True, True)
  (True, False) + token ∈ KEY_TRAIL_IDS        → (True, False)  # skip '":' / digit / '"'
  (True, False) + other                        → (False, False)  # abort
  (True, True) + token ∈ COORD_ALL_VALUE_IDS   → (True, True) + mark as LoRA_V
  (True, True) + token ∈ BRACKET_CLOSE_IDS     → (False, False)
  (True, True) + other                         → (False, False)  # abort

KEY_TRAIL_IDS = {788, 1, 330} | DIGIT_TOKEN_IDS
  788 = '":"' (colon after key)
  1   = '"'   (closing quote)
  330 = ' "'  (space+quote)
  15-24 = digits (part of key name, e.g., '2' in 'coordinate2')
```

This handles:
- `"coordinate": [540.0, 389.8]` → marks digits between `[` and `]`
- `"coordinate2": [540.0, 1800.0]` → `coordinate` triggers, `2` is a digit so it's part of the key name, then `":` → bracket → digits
- `"bbox": [360, 327, 695, 466]` → same pattern
- No false positives from `"time": 2` because `time` ≠ COORD_KEY

---

## 6. Experiment Plan

### Phase 1: Training (v6.5 + coord_routing)

```bash
sbatch scripts/exp_cooperative/train_v6_5_ac_coord_routing.slurm
```

Identical hyperparameters to v6.5 AC: r=256, alpha=512, tanh gates, gate_lr_mult=100, 4 epochs, eff_bs=32.

Single variable change: `--coord_routing` flag.

Output: `train_GUI_360/llamafactory/output/cooperative_v6_5_ac_coord_routing/`

### Phase 2: Gradient Verification (first 10 steps)

After training starts, verify LoRA_V now receives direct gradient:

```python
# Check gradient norms after loss.backward()
grad_BV = model.coop_modules[14].lora_B_v.grad.norm()
grad_BA = model.coop_modules[14].lora_B_a.grad.norm()
print(f"||grad_BV|| / ||grad_BA|| = {grad_BV / grad_BA:.3f}")
# Expected: 0.05-0.3 (was ~0.001 without coord_routing)
```

### Phase 3: Evaluation

#### 3a. Standard eval (hard routing)
```bash
COOP_EPOCH=4 sbatch scripts/exp_cooperative/eval_coop_ac_coord_routing.slurm
```

#### 3b. CA ablation
```bash
COOP_EPOCH=4 INFERENCE_MODE=v_only sbatch eval_coop_ac_coord_routing.slurm
COOP_EPOCH=4 INFERENCE_MODE=t_only sbatch eval_coop_ac_coord_routing.slurm
COOP_EPOCH=4 INFERENCE_MODE=merge  sbatch eval_coop_ac_coord_routing.slurm
```

### Phase 4: Success Criteria

| Metric | v6.5 (no coord_routing) | Expected (with coord_routing) |
|--------|------------------------|-------------------------------|
| hard TSR | 10.11% | > 10.5% |
| v_only TSR | 7.26% | > 8.5% (LoRA_V actually learned) |
| t_only TSR | ~10.41% | < hard (LoRA_A no longer dominates) |
| **CA** | **-0.3%** | **> 0%** (cooperative routing helps) |
| ||grad_BV|| / ||grad_BA|| | ~0.001 | > 0.05 |

**Key success signal**: `hard > max(v_only, t_only)` — proving cooperative routing adds value.

---

## 7. Risk Analysis

### What if 9.8% CE loss tokens for LoRA_V is not enough?

If LoRA_V still underperforms with coord_routing alone:
1. **Add bbox routing** (already included — +2.9% → 9.8% total)
2. **Add auxiliary loss on image tokens** (contrastive or reconstruction)
3. **Increase coord precision** — use more decimal digits in action format to increase coord token count

### What about non-coordinate actions?

36.4% of actions (terminate, open, wait, type, system_button) have no coordinates. For these turns, LoRA_V only processes image tokens (indirect gradient). This is acceptable — these actions are primarily semantic decisions where LoRA_A should dominate.

### Backward Compatibility

- `coord_routing=False` (default) → identical to v6.5 behavior
- Old checkpoints work normally with `coord_routing=False`
- Checkpoints trained WITH coord_routing should be evaluated WITH `--coord-routing` flag in vLLM

---

## 8. File Index

| File | Change | Purpose |
|------|--------|---------|
| `verl/models/cooperative/cooperative_wrapper.py` | Modified | Core coord mask construction |
| `verl/models/cooperative/cooperative_lora.py` | Unchanged | Routing logic unchanged (mask just has more True positions) |
| `evaluation/serve_corrected_vllm.py` | Modified | vLLM coord mask for lossless inference |
| `train_cooperative.py` | Modified | `--coord_routing` flag |
| `scripts/exp_cooperative/train_v6_5_ac_coord_routing.slurm` | New | Training script |
| `scripts/exp_cooperative/eval_coop_ac_coord_routing.slurm` | New | Eval script with CA ablation |
| `v6.5/gradient_imbalance_analysis.md` | Existing | Problem diagnosis |
| `v6.5/AC/plan_b_data_analysis.md` | Existing | AC data structure analysis |
| `v6.5/AC/plan_b_coord_routing.md` | This file | Complete processing flow |
| `v6.5/AC/plan_b_cooperative_reasoning.md` | New | Cooperative reasoning (α-mixing) extension |

---

## 9. Extension: Cooperative Reasoning (α-Mixing)

Coord routing alone gives LoRA_V 9.8% CE loss. To further strengthen LoRA_V, we added **cooperative reasoning**: route think/assistant tokens through `α·V + (1-α)·A` instead of pure A.

With `--coop_reasoning_alpha 0.3`, LoRA_V's effective CE weight jumps from 9.8% → 36.9%.

See `v6.5/AC/plan_b_cooperative_reasoning.md` for full details.
