# Cooperative Reasoning: α-Mixed Routing for Think Tokens

## 1. Motivation

### Problem: LoRA_V Still Underpowered with coord_routing Alone

With coord_routing (Plan B Step 1), LoRA_V gets direct CE loss on 9.8% of assistant tokens (coordinate/bbox digits). This is a qualitative improvement over 0%, but:

- **84.3% of assistant tokens are `<think>` reasoning** — the dominant signal
- Think content IS visual reasoning: it describes UI elements, positions, and screen layout
- LoRA_V should participate in generating visual reasoning, not just spatial coordinates

### Key Insight: Think = Visual Reasoning

```
<think>
I can see a search bar at the top of the screen. Below it, there are
several app icons. I need to tap the "Settings" icon which appears to
be in the middle-left area of the screen.
</think>
```

This text directly describes the image content. It should be generated with input from BOTH:
- LoRA_V (visual expertise: what's on the screen)
- LoRA_A (action expertise: what to do about it)

## 2. Design: Cooperative Reasoning with α-Mixing

### Float Mask Instead of Bool Mask

Replace the binary routing mask with a float mask:

```
Token routing (float mask values):
  Image tokens:           1.0  → pure LoRA_V
  Coord/bbox digits:      1.0  → pure LoRA_V (coord_routing)
  Think + action tokens:  α    → α·LoRA_V + (1-α)·LoRA_A (cooperative reasoning)
  Non-assistant tokens:   0.0  → pure LoRA_A
```

Where α = `coop_reasoning_alpha` (default 0.3).

### How It Works

In `CooperativeLoRALinear.forward()`:

```python
# Before (bool mask):
delta = torch.where(mask, delta_v, delta_a)

# After (float mask when α > 0):
mask_f = mask.to(dtype)  # [B, S, 1], values in {0.0, α, 1.0}
delta = mask_f * delta_v + (1.0 - mask_f) * delta_a
```

For α=0.3 on a think token:
```
delta = 0.3 * delta_v + 0.7 * delta_a
```

Both LoRA_V and LoRA_A contribute to predicting think tokens, with LoRA_A dominating.

### Token Budget Change (AC, α=0.3)

```
Before (v6.5, no coord_routing, no coop_reasoning):
  LoRA_V effective CE weight: 0%
  LoRA_A effective CE weight: 100%

After (coord_routing + coop_reasoning α=0.3):
  LoRA_V effective CE weight: 9.8% * 1.0 + 90.2% * 0.3 = 36.9%
  LoRA_A effective CE weight: 90.2% * 0.7 = 63.1%
```

LoRA_V goes from 0% → 36.9% effective CE — a massive qualitative change.

### Why α=0.3?

- Too high (α→1): Think tokens become LoRA_V-dominated, losing LoRA_A's action expertise
- Too low (α→0): Back to original problem, LoRA_V still starved
- α=0.3: LoRA_V gets meaningful gradient (~37%) while LoRA_A retains dominance (~63%) for reasoning/action

## 3. Implementation

### Files Modified

#### 3.1 `verl/models/cooperative/cooperative_lora.py`

**2-agent routing**: Added float mask support:
```python
if mask.dtype == torch.bool:
    delta = torch.where(mask, delta_v, delta_a)
else:
    mask_f = mask.to(dtype)
    delta = mask_f * delta_v + (1.0 - mask_f) * delta_a
```

#### 3.2 `verl/models/cooperative/cooperative_wrapper.py`

**`__init__`**: Added `coop_reasoning_alpha: float = 0.0` parameter.

**`forward()` (training)**: When `coop_reasoning_alpha > 0`:
1. Build float mask: `torch.zeros(..., dtype=float32)`
2. Set image tokens to 1.0
3. Apply coord_routing (sets coord/bbox to 1.0)
4. Use `labels != -100` to find assistant tokens
5. Set non-V assistant tokens to α

```python
if self.coop_reasoning_alpha > 0 and self.num_agents == 2:
    token_mask = torch.zeros(input_ids.shape, dtype=torch.float32, device=input_ids.device)
    token_mask[input_ids == IMAGE_PAD_ID] = 1.0
    if self.coord_routing:
        self._mark_coord_tokens(token_mask, input_ids)
    if labels is not None:
        is_assistant = (labels != -100)
        is_v = (token_mask > 0.5)
        is_reasoning = is_assistant & ~is_v
        token_mask[is_reasoning] = self.coop_reasoning_alpha
```

**`_mark_assistant_spans()`**: For generation (no labels), detects `<|im_start|>assistant\n` ... `<|im_end|>` spans using token IDs (151644, 77091, 198, 151645).

**`generate()` → `_pre_hook`**: For 2-agent + coop_reasoning_alpha:
- Prefill: float mask with image=1.0, coord=1.0, assistant=α
- Decode: float mask with α for all tokens (all decode tokens are assistant), coord=1.0

#### 3.3 `evaluation/serve_corrected_vllm.py`

**`_coop_reasoning_alpha`**: Global float, set via `--coop-reasoning-alpha`.

**`_build_assistant_mask_1d()`**: 1D version of assistant span detection.

**`CorrectedQwen25VL.forward()`**: When α > 0, builds float mask:
```python
mask = torch.zeros_like(input_ids, dtype=torch.float32)
mask[input_ids == IMAGE_PAD_ID] = 1.0
if _coord_routing:
    coord_mask = _build_coord_mask_1d(input_ids)
    mask[coord_mask] = 1.0
if _coop_reasoning_alpha > 0:
    asst_mask = _build_assistant_mask_1d(input_ids)
    non_v = (mask < 0.5)
    mask[asst_mask & non_v] = _coop_reasoning_alpha
```

**`_compute_correction_weight()`**: Already handles float masks — `.unsqueeze(-1).to(dtype)` works for both bool and float.

#### 3.4 `train_cooperative.py`

Added `--coop_reasoning_alpha` argument (default 0.0), passed to `CooperativeVLMWrapper()`.

### Files Created/Updated

- `scripts/exp_cooperative/train_v6_5_ac_coord_routing.slurm` → Updated with `--coop_reasoning_alpha 0.3`
- `scripts/exp_cooperative/eval_coop_ac_coord_routing.slurm` → Updated with `--coop-reasoning-alpha 0.3`

---

## 4. Inference Behavior

### Lossless Merge + Cooperative Reasoning

In the lossless merge representation:
- Base weights have LoRA_A merged in
- Correction = δV - δA

For a token with mask weight w:
```
output = (base + δA) @ x + w * (δV - δA) @ x
       = base @ x + (1-w) * δA @ x + w * δV @ x
```

So mask weight w correctly interpolates between LoRA_A and LoRA_V:
- w=0: pure LoRA_A
- w=α: α·V + (1-α)·A (cooperative reasoning)
- w=1: pure LoRA_V

### Assistant Token Detection at Inference

Since we don't have labels at inference, we detect assistant spans using Qwen2.5-VL's chat template tokens:
```
<|im_start|>assistant\n  → tokens [151644, 77091, 198]
<|im_end|>               → token  [151645]
```

During autoregressive decode, all generated tokens are assistant tokens → all get α.

---

## 5. Experiment Plan

### Training

```bash
sbatch scripts/exp_cooperative/train_v6_5_ac_coord_routing.slurm
```

Hyperparameters identical to v6.5 AC + two additions:
- `--coord_routing` (route coord/bbox digits to LoRA_V)
- `--coop_reasoning_alpha 0.3` (α-mix think tokens)

### Evaluation

```bash
COOP_EPOCH=4 sbatch scripts/exp_cooperative/eval_coop_ac_coord_routing.slurm
```

CA ablation:
```bash
COOP_EPOCH=4 INFERENCE_MODE=v_only sbatch eval_coop_ac_coord_routing.slurm
COOP_EPOCH=4 INFERENCE_MODE=t_only sbatch eval_coop_ac_coord_routing.slurm
```

### Success Criteria

| Metric | v6.5 (baseline) | Expected (coord+coop_reasoning) |
|--------|-----------------|--------------------------------|
| hard TSR | 10.11% | > 11% |
| v_only TSR | 7.26% | > 9% (LoRA_V now well-trained) |
| t_only TSR | 10.41% | < hard (LoRA_A alone insufficient) |
| **CA** | **-0.3%** | **> +1%** (cooperative routing adds value) |
| LoRA_V eff CE weight | 0% | 36.9% |

Key signal: `hard > max(v_only, t_only)` — cooperative routing synergizes.

---

## 6. Relationship to Other Approaches

| Approach | LoRA_V CE weight | Mechanism |
|----------|-----------------|-----------|
| v6.5 (original) | 0% | Hard routing, image-only V |
| coord_routing only | 9.8% | Route coord digits to V |
| coop_reasoning only (α=0.3) | 27.1% | α-mix all assistant tokens |
| **coord_routing + coop_reasoning** | **36.9%** | Both combined |

The cooperative reasoning approach is orthogonal to coord_routing and they stack naturally.

## 7. Backward Compatibility

- `coop_reasoning_alpha=0.0` (default) → identical to coord_routing-only behavior
- `coord_routing=False, coop_reasoning_alpha=0.0` → identical to v6.5 behavior
- Both new features are additive and off by default
