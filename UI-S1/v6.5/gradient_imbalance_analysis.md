# Cooperative LoRA v6.5: Gradient Imbalance Analysis

## Problem Statement

Cooperative Advantage (CA) ablation on AC shows:
- hard routing: 10.11%
- t_only (all → LoRA_A): ~10.41%
- v_only (all → LoRA_V): 7.26%
- CA = acc_hard - max(acc_v_only, acc_t_only) ≈ -0.3%

**Cooperative routing provides no benefit.** LoRA_A alone matches or exceeds hard routing.

## Root Cause: Asymmetric Gradient Flow

### Training Architecture
```
Input tokens:
  - Image tokens (id=151655) → LoRA_V   (mask=True)
  - Text tokens              → LoRA_A   (mask=False)

Loss:
  - CE loss computed on labels != -100
  - _mask_non_assistant() sets labels=-100 for ALL non-assistant tokens
  - Assistant response tokens = ALL TEXT tokens → ALL through LoRA_A
  - Image tokens are NEVER in assistant responses → NEVER in CE loss
```

### Gradient Path
```
LoRA_A: CE loss → assistant text tokens → LoRA_A parameters
        ✅ DIRECT gradient, strong signal

LoRA_V: CE loss → text tokens → attention(Q_text, K_img, V_img) → LoRA_V params
        ⚠️ INDIRECT gradient only, through attention cross-terms
        Diluted across entire sequence length
```

## Quantitative Analysis Across All 3 Datasets

### Token Distribution

| Dataset | Episodes | Imgs/ep | Img tokens (LoRA_V) | Text tokens (LoRA_A) | CE loss tokens | LoRA_V CE loss |
|---------|----------|---------|--------------------|--------------------|----------------|----------------|
| GUI-360 | 97,647 | 1.0 | 24.4% (~1000/img) | 75.6% | 4.5% (assistant only) | **0%** |
| AC | 950 | 6.5 | 70.8% (~900/img) | 29.2% | 20.8% (assistant only) | **0%** |
| Odyssey | 6,468 | 15.3 | 80.1% (~506/img) | 19.9% | 9.7% (assistant only) | **0%** |

### Key Observations

1. **All 3 datasets**: LoRA_V receives exactly 0% direct CE loss gradient
2. **AC & Odyssey**: Image tokens are the MAJORITY (70-80%) of the sequence, yet LoRA_V gets no direct supervision
3. **Odyssey is worst**: 80.1% of tokens go through LoRA_V with zero direct training signal
4. **GUI-360**: Even in single-turn with fewer images, LoRA_V still gets 0% direct loss

### Attention Dilution (Indirect Gradient Strength)

| Dataset | Avg seq length | Avg img tokens/seq | Indirect gradient dilution |
|---------|----------------|-------------------|--------------------------|
| GUI-360 | ~2,861 | ~1,000 | Moderate (35% of attention window) |
| AC | ~8,248 | ~5,841 | Severe (71% image but gradient from 29% text attending to them) |
| Odyssey | ~9,680 | ~7,750 | Most severe (80% image, only 20% text to backprop through) |

The indirect gradient to LoRA_V comes from:
```
∂L/∂LoRA_V = ∂L/∂h_text · ∂h_text/∂attn_weights · ∂attn_weights/∂K_img · ∂K_img/∂LoRA_V
```
This gradient is:
- Diluted by softmax normalization (attention weights sum to 1 over seq_len)
- Weakened by multi-layer composition (gradient vanishing through 28 transformer layers)
- Strongest only for image tokens that the model attends to most

### Communication Gates Don't Fix This

The v6.5 communication mechanism:
```python
h_v = h_v + g_av * W_av @ h_a   # V sees A's representation
h_a = h_a + g_va * W_va @ h_v   # A sees V's representation
```

While `g_va * W_va` provides a second gradient path to LoRA_V (through LoRA_A's loss → h_a → W_va → h_v → LoRA_V), this is still indirect and gated by `tanh(gate_va)`.

## Impact: This Problem Exists in ALL 3 Datasets

| Dataset | Problem severity | Why |
|---------|-----------------|-----|
| GUI-360 | Moderate | Single-turn, fewer image tokens (24.4%), but LoRA_V still untrained directly |
| AC | High | Multi-turn, 70.8% image tokens with 0% direct loss. Long sequences dilute indirect gradient |
| Odyssey | Highest | Multi-turn, 80.1% image tokens, longest sequences → most gradient dilution |

## Conclusion

The gradient imbalance is a **structural property** of the current training setup:
1. VLM architecture: image tokens are input-only, never predicted
2. CE loss: only on predicted (assistant) tokens = all text = all LoRA_A
3. This makes LoRA_A the "primary brain" and LoRA_V a "passive observer"

This explains why `t_only ≈ hard`: forcing image tokens through the undertrained LoRA_V doesn't help (and may hurt) compared to just using the well-trained LoRA_A for everything.
