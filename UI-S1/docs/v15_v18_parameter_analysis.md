# Cooperative LoRA Parameter Change Analysis

## Overview

Comparing V15 parameters across training stages:
- **Pre-RL**: SVD-extracted cooperative checkpoint (uniform routing)
- **Post-RL (step 25)**: Best val checkpoint from V15 RL (epoch-0_step-25)
- **Post-RL (step 50)**: Longer training checkpoint (epoch-1_step-50)

V15 architecture: 2 experts, sigmoid routing, shared B, iterative communication (T=2)

## 1. Route Weights

Route weight shape: `[hidden_size=3584]` per layer (28 layers)
Routing: `r = sigmoid(x @ w_route)` → blend = `r * A₁(x) + (1-r) * A₂(x)`
Init: zeros → sigmoid(0) = 0.5 (uniform blend)

| Layer | Pre norm | Post-25 norm | Post-50 norm | Post-25 sigmoid range | Post-50 sigmoid range |
|-------|----------|-------------|-------------|----------------------|----------------------|
| L0  | 0.0000 | 0.4227 | 0.5449 | [0.300, 0.700] | [0.252, 0.748] |
| L1  | 0.0000 | 0.3677 | 0.4190 | [0.324, 0.676] | [0.302, 0.698] |
| L2  | 0.0000 | 0.3249 | 0.3944 | [0.343, 0.657] | [0.312, 0.688] |
| L3  | 0.0000 | 0.3314 | 0.4681 | [0.340, 0.660] | [0.282, 0.718] |
| L4  | 0.0000 | 0.3384 | 0.4528 | [0.337, 0.663] | [0.288, 0.712] |
| L5  | 0.0000 | 0.3380 | 0.4582 | [0.337, 0.663] | [0.286, 0.714] |
| L6  | 0.0000 | 0.3194 | 0.4521 | [0.346, 0.654] | [0.288, 0.712] |
| L7  | 0.0000 | 0.3254 | 0.4254 | [0.343, 0.657] | [0.299, 0.701] |
| L8  | 0.0000 | 0.3174 | 0.4232 | [0.346, 0.654] | [0.300, 0.700] |
| L9  | 0.0000 | 0.3486 | 0.4766 | [0.332, 0.668] | [0.278, 0.722] |
| L10 | 0.0000 | 0.3505 | 0.4289 | [0.332, 0.668] | [0.298, 0.702] |
| L11 | 0.0000 | 0.3463 | 0.4388 | [0.333, 0.667] | [0.294, 0.706] |
| L12 | 0.0000 | 0.2857 | 0.4513 | [0.361, 0.639] | [0.289, 0.711] |
| L13 | 0.0000 | 0.3208 | 0.4312 | [0.345, 0.655] | [0.297, 0.703] |
| L14 | 0.0000 | 0.2993 | 0.3793 | [0.355, 0.645] | [0.319, 0.681] |
| L15 | 0.0000 | 0.3504 | 0.4447 | [0.332, 0.668] | [0.291, 0.709] |
| L16 | 0.0000 | 0.3352 | 0.4073 | [0.338, 0.662] | [0.307, 0.693] |
| L17 | 0.0000 | 0.3427 | 0.4360 | [0.335, 0.665] | [0.295, 0.705] |
| L18 | 0.0000 | 0.2899 | 0.4040 | [0.359, 0.641] | [0.308, 0.692] |
| L19 | 0.0000 | 0.3404 | 0.4480 | [0.336, 0.664] | [0.290, 0.710] |
| L20 | 0.0000 | 0.3456 | 0.4422 | [0.334, 0.666] | [0.292, 0.708] |
| L21 | 0.0000 | 0.3699 | 0.4670 | [0.323, 0.677] | [0.282, 0.718] |
| L22 | 0.0000 | 0.3319 | 0.4684 | [0.340, 0.660] | [0.282, 0.718] |
| L23 | 0.0000 | 0.3070 | 0.3908 | [0.351, 0.649] | [0.314, 0.686] |
| L24 | 0.0000 | 0.3464 | 0.4272 | [0.333, 0.667] | [0.299, 0.701] |
| L25 | 0.0000 | 0.3640 | 0.4237 | [0.326, 0.674] | [0.300, 0.700] |
| L26 | 0.0000 | 0.3879 | 0.5131 | [0.315, 0.685] | [0.264, 0.736] |
| L27 | 0.0000 | 0.3523 | 0.4298 | [0.331, 0.669] | [0.297, 0.703] |

**Summary:**
- Pre-RL norm: all zero (uniform routing)
- Post-RL mean norm: 0.3393 (very small)
- Post-RL mean sigmoid: 0.5000 (≈ 0.5, essentially unchanged)
- Effective routing range: [0.300, 0.700] at 2σ
- **Conclusion: Routing barely moved from uniform. K=2 sigmoid routing did not learn meaningful expert selection.**

## 2. LoRA A Matrices (Expert Parameters)

Shape: `[rank=128, in_features]` per expert per module per layer
Two experts: A₁, A₂. Init: A_avg ± noise (noise_scale=0.1)

| Layer | Module | Pre A₁-A₂ div | Post-25 A₁-A₂ div | ΔA₁ rel | ΔA₂ rel | A₁≈A₂ sym? |
|-------|--------|--------------|-------------------|---------|---------|-----------|
| L0  | q_proj | 0.200001 | 0.201730 | 0.1440 | 0.1438 | 0.0018 |
| L0  | k_proj | 0.199847 | 0.206274 | 0.1486 | 0.1485 | 0.0006 |
| L0  | v_proj | 0.200104 | 0.202844 | 0.1481 | 0.1483 | 0.0018 |
| L0  | o_proj | 0.200076 | 0.201404 | 0.1438 | 0.1439 | 0.0004 |
| L7  | q_proj | 0.200029 | 0.201364 | 0.1437 | 0.1437 | 0.0002 |
| L7  | k_proj | 0.200113 | 0.204292 | 0.1475 | 0.1475 | 0.0000 |
| L7  | v_proj | 0.200029 | 0.203101 | 0.1481 | 0.1484 | 0.0016 |
| L7  | o_proj | 0.199605 | 0.201088 | 0.1438 | 0.1439 | 0.0008 |
| L14 | q_proj | 0.200282 | 0.200902 | 0.1437 | 0.1437 | 0.0001 |
| L14 | k_proj | 0.200424 | 0.203733 | 0.1479 | 0.1477 | 0.0015 |
| L14 | v_proj | 0.200041 | 0.203389 | 0.1483 | 0.1484 | 0.0009 |
| L14 | o_proj | 0.200176 | 0.200861 | 0.1441 | 0.1442 | 0.0008 |
| L21 | q_proj | 0.200134 | 0.202498 | 0.1441 | 0.1442 | 0.0012 |
| L21 | k_proj | 0.200142 | 0.206610 | 0.1487 | 0.1488 | 0.0006 |
| L21 | v_proj | 0.200139 | 0.206173 | 0.1504 | 0.1504 | 0.0002 |
| L21 | o_proj | 0.199819 | 0.201813 | 0.1450 | 0.1447 | 0.0018 |
| L27 | q_proj | 0.200272 | 0.203382 | 0.1447 | 0.1450 | 0.0025 |
| L27 | k_proj | 0.199895 | 0.208118 | 0.1498 | 0.1504 | 0.0040 |
| L27 | v_proj | 0.200237 | 0.218517 | 0.1604 | 0.1640 | 0.0217 |
| L27 | o_proj | 0.199835 | 0.204093 | 0.1460 | 0.1464 | 0.0022 |

**Summary:**
- Pre-RL expert divergence: 0.200016 (from noise_scale=0.1)
- Post-RL expert divergence: 0.203189 (**1.02x** increase)
- Mean |ΔA₁|/|A₁|: 0.1464 (14.6%)
- Mean |ΔA₂|/|A₂|: 0.1464 (14.6%)
- Mean symmetry (|ΔA₁-ΔA₂|/max): 0.0012 (0=perfect symmetry)
- **Conclusion: Both experts changed ~14.6% but moved nearly symmetrically. Expert divergence barely increased (1.02x). No specialization learned.**

## 3. LoRA B Matrix (Shared Projection)

Shape: `[out_features, rank=128]` — shared between both experts

| Layer | Module | |B_pre| | |ΔB|/|B| |
|-------|--------|--------|---------|
| L0  | q_proj | 1.3369 | 0.029192 |
| L0  | k_proj | 0.8054 | 0.018467 |
| L0  | v_proj | 0.8315 | 0.017860 |
| L0  | o_proj | 1.3114 | 0.029205 |
| L7  | q_proj | 1.3148 | 0.028406 |
| L7  | k_proj | 0.8388 | 0.016998 |
| L7  | v_proj | 0.8066 | 0.017737 |
| L7  | o_proj | 1.2468 | 0.029665 |
| L14 | q_proj | 1.2602 | 0.029339 |
| L14 | k_proj | 0.8223 | 0.016725 |
| L14 | v_proj | 0.7697 | 0.017939 |
| L14 | o_proj | 1.1846 | 0.031126 |
| L21 | q_proj | 1.2586 | 0.031495 |
| L21 | k_proj | 0.8036 | 0.018475 |
| L21 | v_proj | 0.7435 | 0.019925 |
| L21 | o_proj | 1.2127 | 0.033044 |
| L27 | q_proj | 1.3030 | 0.034480 |
| L27 | k_proj | 0.7984 | 0.021351 |
| L27 | v_proj | 0.6418 | 0.028873 |
| L27 | o_proj | 1.1270 | 0.037534 |

**Summary:**
- Mean |ΔB|/|B|: 0.024546 (2.45%)
- **Conclusion: B barely changed (2.5%). Most learning happened in A matrices.**

## 4. Communication Weights

Per layer per round: W₁₂[r,r], W₂₁[r,r], gate₁₂[r], gate₂₁[r]
Total: 28 layers × 2 rounds = 56 sets
Init: W = Kaiming uniform, gate = zeros → sigmoid(0) = 0.5

| Layer | Round | |ΔW₁₂|/|W₁₂| | |ΔW₂₁|/|W₂₁| | gate₁₂ sig | gate₂₁ sig |
|-------|-------|-------------|-------------|-----------|-----------|
| L0  | 0 | 1.4240 | 1.4274 | 0.4998 | 0.5002 |
| L0  | 1 | 1.4273 | 1.4261 | 0.4997 | 0.5002 |
| L7  | 0 | 1.4229 | 1.4126 | 0.5001 | 0.4999 |
| L7  | 1 | 1.4196 | 1.4151 | 0.4998 | 0.4999 |
| L14 | 0 | 1.4068 | 1.4165 | 0.5000 | 0.5000 |
| L14 | 1 | 1.4219 | 1.4144 | 0.5000 | 0.4999 |
| L21 | 0 | 1.4214 | 1.4133 | 0.4999 | 0.5003 |
| L21 | 1 | 1.4140 | 1.4094 | 0.4999 | 0.5001 |
| L27 | 0 | 1.4155 | 1.4304 | 0.4998 | 0.4998 |
| L27 | 1 | 1.4185 | 1.4132 | 0.5001 | 0.4998 |

**Summary:**
- Mean |ΔW₁₂|/|W₁₂|: 1.4187 (142% — **massive change**)
- Mean |ΔW₂₁|/|W₂₁|: 1.4183 (142%)
- Mean gate₁₂ sigmoid: 0.5000 (init 0.5)
- Mean gate₂₁ sigmoid: 0.5000 (init 0.5)
- **Conclusion: W matrices were almost entirely rewritten (~142% relative change!) but gates stayed at ~0.5. Communication is always-on but undirected — no learned gating.**

## 5. Effective Weight Delta Change

`ΔW_eff = B @ (r * A₁ + (1-r) * A₂) * scaling`

At uniform routing (r=0.5), this equals B @ A_avg * scaling ≈ rank-128 SVD of full SFT ΔW.

| Layer | Module | |ΔW_pre| | |ΔW_post - ΔW_pre|/|ΔW_pre| |
|-------|--------|---------|-------------------------------|
| L0  | q_proj | 0.3539 | 0.039479 |
| L0  | k_proj | 0.1394 | 0.043000 |
| L0  | v_proj | 0.1485 | 0.043051 |
| L0  | o_proj | 0.3599 | 0.036875 |
| L7  | q_proj | 0.3442 | 0.037872 |
| L7  | k_proj | 0.1400 | 0.042164 |
| L7  | v_proj | 0.1335 | 0.044614 |
| L7  | o_proj | 0.3257 | 0.038065 |
| L14 | q_proj | 0.3268 | 0.036819 |
| L14 | k_proj | 0.1333 | 0.042707 |
| L14 | v_proj | 0.1184 | 0.045726 |
| L14 | o_proj | 0.2950 | 0.038617 |
| L21 | q_proj | 0.3388 | 0.037557 |
| L21 | k_proj | 0.1304 | 0.043791 |
| L21 | v_proj | 0.1119 | 0.049126 |
| L21 | o_proj | 0.3108 | 0.042381 |
| L27 | q_proj | 0.3625 | 0.041728 |
| L27 | k_proj | 0.1352 | 0.047132 |
| L27 | v_proj | 0.0921 | 0.068730 |
| L27 | o_proj | 0.2917 | 0.046147 |

**Summary:**
- Mean effective ΔW change: 0.0415 (4.15%)

## 6. Parameter Change Ranking (All Categories)

| Category | Count | Total Params | Mean Relative Change | Contribution |
|----------|-------|-------------|---------------------|-------------|
| Route weights | 28 | 100,352 | N/A (from 0) | Negligible |
| LoRA A₁ | 112 | 51,380,224 | 0.1464 | Primary |
| LoRA A₂ | 112 | 51,380,224 | 0.1464 | Primary |
| LoRA B (shared) | 112 | 29,360,128 | 0.0245 | Minor |
| Comm W₁₂ | 56 | 917,504 | 1.4187 | Rewritten but gated |
| Comm W₂₁ | 56 | 917,504 | 1.4183 | Rewritten but gated |
| Comm gate₁₂ | 56 | 7,168 | ~0 shift | No effect |
| Comm gate₂₁ | 56 | 7,168 | ~0 shift | No effect |

## 7. Conclusions & Implications for V18

### What V15 RL actually learned
1. **A matrices (14.6% change)**: Both experts moved ~equally in the same direction. This is essentially **standard LoRA fine-tuning**, not expert specialization.
2. **B matrix (2.5% change)**: Minimal adjustment to shared projection.
3. **Comm W matrices (142% change)**: Completely rewritten, but with gates stuck at 0.5, this acts as a fixed linear transform in r-space — effectively an additional learned layer, not adaptive communication.
4. **Route weights (~0)**: No meaningful per-token expert selection learned.
5. **Comm gates (~0.5)**: No learned communication gating.

### Why routing failed in V15
- Sigmoid routing with 1D weight: `logits = x @ w` produces a single scalar → limited expressiveness
- Route weight norm ~0.34, element std ~0.006 → routing logit std ~0.34 → sigmoid range [0.30, 0.70]
- Even in the best case, both experts contribute 30-70% — no token ever "selects" one expert

### V18 advantages
- **Softmax K-way routing** with `[H, K]` matrix: each expert gets its own projection direction → much richer per-token signal
- **K=4/8 experts**: even small routing differences activate different rank-128 subspaces → effective capacity > rank-128
- **Diversity loss**: explicitly pushes expert outputs apart, preventing the symmetric co-movement seen in V15
- **Balance loss (K-way entropy)**: prevents expert collapse (all tokens → 1 expert)
