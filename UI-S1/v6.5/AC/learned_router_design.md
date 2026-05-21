# Learned Router Cooperative LoRA (v8)

## Core Idea

Replace fixed token-type routing with a **learned per-token router** that reads from hidden states to decide WHERE vs WHAT routing. The hidden state already encodes semantic function — let the model learn the optimal split.

## Architecture

### Per-Layer Router

```
Input x: [B, S, D]  (hidden state from previous layer)
                │
        ┌───────┴───────┐
        │  Router(x)    │   ← nn.Linear(D, 1) → sigmoid
        │  w ∈ [0, 1]   │   w=1: pure WHERE, w=0: pure WHAT
        └───────┬───────┘
                │
    ┌───────────┼───────────┐
    │           │           │
 LoRA_WHERE  w,(1-w)   LoRA_WHAT
 (lora_A_w   blend     (lora_A_d
  lora_B_w)             lora_B_d)
    │           │           │
    └───────────┼───────────┘
                │
 delta = w · delta_WHERE + (1-w) · delta_WHAT
```

### Router Design Choices

**Option A: Per-module router** (most flexible)
- Each linear layer (q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj) has its own router
- 28 layers × 7 modules = 196 routers
- Params: 196 × D × 1 = 196 × 3584 = 702K (tiny)

**Option B: Per-layer shared router** (recommended)
- One router per transformer layer, shared across all modules in that layer
- 28 routers, same hidden state → same routing for q/k/v/o
- Params: 28 × D × 1 = 100K
- Rationale: routing should be a semantic decision, not module-specific

**Option C: Global router** (simplest)
- One router shared across all layers
- Very simple but may lack layer-specific routing

**Recommendation: Option B** — semantically meaningful, minimal parameters.

### Router Initialization

Critical: initialize router to approximate the current image/text hard routing as warm start.

**Strategy**: Image tokens have distinctive hidden states (from vision encoder). Initialize router weights so that image token hidden states produce high output (→ WHERE) and text token hidden states produce low output (→ WHAT).

**Implementation**:
1. Run base model on a few samples
2. Collect mean hidden state for image vs text tokens at each layer
3. Set router.weight = (mean_img - mean_text) / ||mean_img - mean_text||²
4. Set router.bias so that sigmoid(router(mean_img)) ≈ 0.9, sigmoid(router(mean_text)) ≈ 0.1

This ensures:
- Step 0: behaves like current hard routing (safe start)
- Training: router learns to refine — e.g., route spatial-think tokens partly to WHERE

### Load Balancing

Without regularization, router may collapse (all tokens → one expert).

**Approach: Soft entropy regularization**
```
L_balance = -H(mean_routing_weight)
         = -(p·log(p) + (1-p)·log(1-p))
where p = mean(w) across all tokens in batch
```

This encourages the average routing weight to be ~0.5 (balanced usage).
Weight: λ_balance = 0.01 (gentle push, not forcing exact balance).

## Data Flow (concrete)

How a single token actually traverses the v8 stack. Base model = Qwen2.5-VL-7B (hidden_size=3584, intermediate_size=18944, num_layers=28).

### Overall

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                          OVERALL MODEL FLOW                                   │
│                                                                                │
│  input_ids → embed → Layer 0 → Layer 1 → ... → Layer 27 → lm_head → logits   │
│                        ↑                                                       │
│                        │                                                       │
│                        │  Each layer = { attention (q,k,v,o) + MLP (g,u,d) }  │
│                        │  All 7 linear sub-modules wrapped as CoopLoRALinear  │
│                        │  One router per layer (28 total)                     │
│                                                                                │
└──────────────────────────────────────────────────────────────────────────────┘
```

### Inside one transformer layer

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                  INSIDE ONE TRANSFORMER LAYER (layer ℓ)                       │
│                                                                                │
│   x  ──────────[input_layernorm]────→ h₁                                      │
│   (B,S,3584)                          │                                        │
│                                       ├──→ q_proj(h₁) ─┐                      │
│                                       ├──→ k_proj(h₁) ─┼─→ attention ─→ attn_out
│                                       └──→ v_proj(h₁) ─┘                      │
│   x + attn_out ─→ residual ─→ h₂                                              │
│                                │                                               │
│                                └──[post_attn_layernorm]──→ h₃                 │
│   (B,S,3584)                                              │                   │
│                                                           ├→ gate_proj(h₃) ┐  │
│                                                           └→ up_proj(h₃)   ├→ │
│                                                                            │  │
│            act(gate)·up  ─── (B,S,18944) ─→ down_proj ─→ mlp_out          │  │
│                                              ↑                             │  │
│                                          SPECIAL CASE:                    │  │
│                                          in_features=18944 ≠ 3584         │  │
│                                          → merge fallback (50/50)         │  │
│                                                                            │  │
│   h₂ + mlp_out ─→ output (B,S,3584)                                           │
└──────────────────────────────────────────────────────────────────────────────┘
```

### Inside one CooperativeLoRALinear (the 6 "routed" modules: q/k/v/o/gate/up)

```
┌──────────────────────────────────────────────────────────────────────────────┐
│        ZOOM: CooperativeLoRALinear for q/k/v/o/gate_proj/up_proj             │
│                        (in_features = 3584, router attached)                  │
│                                                                                │
│              x  (B, S, 3584)                                                  │
│              │                                                                 │
│    ┌─────────┼─────────────────────────────────────┐                          │
│    │         │                                      │                          │
│    │  ┌──────▼──────┐                               │                          │
│    │  │ base_linear │── base_out  (B,S,out)         │                          │
│    │  │ W ∈ R^{o×i} │                               │                          │
│    │  │ FROZEN      │                               │                          │
│    │  └─────────────┘                               │                          │
│    │                                                 │                          │
│    │    ┌───────────────────────┬──────────────┐    │                          │
│    │    │                       │              │    │                          │
│    │  lora_A_v                lora_A_a      router  │                          │
│    │  (3584→r=256)            (3584→r=256)  (3584→1)│                         │
│    │    │                       │              │    │                          │
│    │   h_v                    h_a         σ(W·x+b) │                          │
│    │  (B,S,256)              (B,S,256)     = w    │                          │
│    │    │                       │         (B,S,1) │                          │
│    │    │   ← v6 comm →         │              │   │                          │
│    │    │  h_v += g_av·W_av·h_a │              │   │                          │
│    │    │  h_a += g_va·W_va·h_v │              │   │                          │
│    │    │                       │              │   │                          │
│    │  lora_B_v                lora_B_a         │   │                          │
│    │  (r→3584)                (r→3584)         │   │                          │
│    │    │ · scaling              │ · scaling   │   │                          │
│    │    ▼                       ▼              │   │                          │
│    │  δ_v = WHERE direction    δ_a = WHAT dir. │   │                          │
│    │  (B,S,out)                (B,S,out)       │   │                          │
│    │    │                       │              │   │                          │
│    │    └──────┐       ┌────────┘              │   │                          │
│    │           │       │         ┌─────────────┘   │                          │
│    │           ▼       ▼         ▼                 │                          │
│    │         delta = w·δ_v + (1−w)·δ_a             │                          │
│    │                 │                              │                          │
│    │                 │ (B,S,out)                    │                          │
│    │                 │                              │                          │
│    │    base_out ──⊕─┘                             │                          │
│    │                 │                              │                          │
│    │                 ▼                              │                          │
│    └─────────────  output  (B,S,out)  ─────────────┘                          │
└──────────────────────────────────────────────────────────────────────────────┘
```

### Inside one CooperativeLoRALinear (the 1 "merge-fallback" module: down_proj)

```
┌──────────────────────────────────────────────────────────────────────────────┐
│   ZOOM: CooperativeLoRALinear for down_proj (in_features = 18944, NO router) │
│                                                                                │
│      x  (B, S, 18944)                                                         │
│      │                                                                         │
│      ├── base_linear ─→ base_out  (B,S,3584)                                  │
│      │    (FROZEN)                                                             │
│      │                                                                         │
│      ├── lora_A_v  (18944→r) → h_v → lora_B_v (r→3584) → δ_v                  │
│      └── lora_A_a  (18944→r) → h_a → lora_B_a (r→3584) → δ_a                  │
│                                                                                │
│      (router not attached: 18944 ≠ 3584 → dim mismatch)                        │
│      delta = 0.5·(δ_v + δ_a)    ← merge fallback (uniform blend)              │
│                                                                                │
│      output = base_out + delta  (B,S,3584)                                    │
└──────────────────────────────────────────────────────────────────────────────┘
```

### Loss and gradient flow

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                          GRADIENT & LOSS FLOW                                 │
│                                                                                │
│    Forward:  input ─→ 28 layers (LoRA_V, LoRA_A, router per layer) ─→ logits │
│                                                                                │
│    Loss = L_CE (thought + action)                   ← main task loss          │
│         + λ_balance · mean_ℓ [ -H(E_x[w_ℓ(x)]) ]    ← push mean w → 0.5      │
│                                                                                │
│    Backward gradients flow to:                                                │
│      • lora_A_v / lora_B_v (28 layers × 7 modules) ← WHERE LoRA               │
│      • lora_A_a / lora_B_a (28 layers × 7 modules) ← WHAT LoRA                │
│      • W_av / W_va (v6 communication matrices)                                 │
│      • g_av / g_va (tanh gates)                                                │
│      • router_ℓ (28 × nn.Linear(3584,1))           ← NEW in v8                │
│      • base model: FROZEN                                                      │
│                                                                                │
│    Total trainable ≈ 1.32 B params (~13.7% of 9.6B total)                     │
└──────────────────────────────────────────────────────────────────────────────┘
```

### Key takeaways

1. **每个 token 过一层会并行走 2 条 LoRA 支路（WHERE 与 WHAT）**，router 用同一个 `x` 算出混合权重 `w`，最终按 `w·δ_v + (1-w)·δ_a` 合成单个 delta 加回 `base_out`。
2. **Router 只看 `x`，不看两个 delta 的质量** — 是 "基于内容决定路由"，不是 "基于质量决定路由"。但 CE loss 的梯度会回到 router，让它学 "哪类 token 偏 WHERE 更好"。
3. **Attention 里 q/k/v/o 共享同一个 `x=h₁`**，MLP 里 gate/up 共享同一个 `x=h₃` — 一层内 router 总共被调用 4 次不同的 `x`（q 和 k、v、o 共享；gate、up 共享），但都是 3584 维，所以用同一个 `router_ℓ` 权重。
4. **`down_proj` 是唯一例外**：输入是 MLP intermediate（18944 维），router 接不上，只能 50/50 硬融合 → 这层的 MLP 残差没有 per-token 路由能力。

## Training

### Loss
```
L = L_CE + λ_balance · L_balance
```

No binding loss (L_bind was already dropped in v6.5).
No diversity loss needed — router naturally differentiates.

### What the router should learn

Based on AC task structure:
- Image tokens → high w (WHERE) ✓ from initialization
- Coordinate/bbox tokens → high w (WHERE) ← model learns this
- Think tokens describing layout → moderate-high w ← model learns this
- Think tokens about decisions → moderate-low w ← model learns this
- Action type tokens → low w (WHAT) ← model learns this

### Gradient Flow

Both experts get gradient on ALL tokens (weighted by router):
- WHERE expert: strong gradient on image/coord tokens, some on spatial-think
- WHAT expert: strong gradient on action/decision tokens, some on spatial-think
- Router: gradient from both CE loss paths — learns which expert produces better output for each token

This solves the gradient imbalance! WHERE expert gets CE gradient proportional to its routing weight, not limited to a fixed token set.

## Implementation Plan

### 1. Modify CooperativeLoRALinear

```python
class CooperativeLoRALinear(nn.Module):
    def __init__(self, base_linear, r, alpha, dropout,
                 routing_mode="learned", router=None):
        # ... existing code ...

        # Shared router (passed from wrapper, one per layer)
        self.router = router  # nn.Linear(D, 1) or None

    def forward(self, x):
        base_out = self.base_linear(x)
        x_drop = self.lora_dropout(x)
        dtype = x_drop.dtype

        # Compute both deltas
        delta_where = F.linear(F.linear(x_drop, self.lora_A_w.to(dtype)),
                               self.lora_B_w.to(dtype)) * self.scaling
        delta_what = F.linear(F.linear(x_drop, self.lora_A_d.to(dtype)),
                              self.lora_B_d.to(dtype)) * self.scaling

        if self.router is not None:
            # Learned routing
            w = torch.sigmoid(self.router(x.detach()))  # [B, S, 1]
            # Note: detach x for router input to prevent router
            # from affecting representation learning
            # (router only learns routing, not features)
            # ACTUALLY: don't detach — we want end-to-end gradient
            w = torch.sigmoid(self.router(x))  # [B, S, 1]
            delta = w * delta_where + (1 - w) * delta_what
        else:
            # Fallback: merge mode
            delta = 0.5 * (delta_where + delta_what)

        return base_out + delta
```

**Key decision: detach or not?**
- `detach`: Router only learns from routing loss, not from CE loss. Simpler but may learn slower.
- `no detach`: End-to-end. Router output affects CE loss, CE loss gradient flows back to router. More powerful but risk of router instability.
- **Recommendation: no detach** — let CE loss guide the router.

### 2. Modify CooperativeVLMWrapper

```python
class CooperativeVLMWrapper(nn.Module):
    def __init__(self, ..., routing_mode="learned"):
        # Create per-layer routers
        self.routers = nn.ModuleList()
        for i in range(num_layers):
            router = nn.Linear(hidden_size, 1, bias=True)
            # Initialize router (done after loading model)
            self.routers.append(router)

        # Replace modules, passing shared router per layer
        for layer_idx, layer in enumerate(layers):
            router = self.routers[layer_idx]
            for mod_name in target_modules:
                # Replace with CooperativeLoRALinear(router=router)
```

### 3. Forward pass (simplified)

```python
def forward(self, input_ids, attention_mask, labels, ...):
    # No token_mask needed! Router decides routing.
    # Just run the model forward — routing happens inside each layer.
    outputs = self.base_model(input_ids, attention_mask, labels=labels, ...)

    # Compute balance loss
    balance_loss = self._compute_balance_loss()

    total_loss = outputs.loss + lambda_balance * balance_loss
    return total_loss, outputs
```

No need for:
- Token mask construction
- Image token detection in wrapper
- Coord routing logic
- Assistant span detection
- α-mixing

**All routing complexity moves into the learned router.**

### 4. Inference (vLLM integration)

For lossless merge + vLLM:
- Router weights are small (28 × 3584 parameters per layer)
- Can be loaded alongside correction matrices
- Forward hook computes: w = sigmoid(router(x)), corr = w * delta_correction
- Where delta_correction = delta_WHERE - delta_WHAT (like current δV - δA)

Actually simpler: since both experts are applied, we need:
```
output = base(x) + w * delta_WHERE + (1-w) * delta_WHAT
       = (base + delta_WHAT)(x) + w * (delta_WHERE - delta_WHAT)(x)
```

So the lossless merge becomes:
- Merge delta_WHAT into base (like current delta_A)
- Correction = delta_WHERE - delta_WHAT (like current δV - δA)
- Hook: output += w * correction, where w = sigmoid(router(x))

Same structure as current vLLM integration, just replace the fixed mask with router output!

## Comparison with Current Architecture

| Aspect | Current (v6.5) | Learned Router (v8) |
|--------|---------------|---------------------|
| Routing | Fixed by token type | Learned from hidden state |
| WHERE CE loss | 0% (image only) or 3.9% (coord_routing) | Proportional to router weight (~50%?) |
| WHAT CE loss | 96-100% | Proportional to router weight (~50%?) |
| Token mask | Required, complex logic | Not needed |
| Coord routing | Manual token parsing | Learned automatically |
| α-mixing | Manual hyperparameter | Learned automatically |
| vLLM inference | Same structure | Same structure + router |
| Extra params | 0 | ~100K (negligible) |

## Experiment Plan

1. **Verification** (verify_hidden_state_routing.py): Confirm hidden states encode token function
2. **Train v8 on AC**: Same data, same epochs, compare TSR
3. **Analyze router weights**: Visualize what the router learned — does it match WHERE/WHAT?
4. **CA ablation**: Compare with t_only, v_only to measure true cooperative advantage

---

## Implementation Status (2026-04-16)

Design converted to working code. All three stack layers updated.

### Files changed

| File | What changed |
|------|--------------|
| `verl/models/cooperative/cooperative_lora.py` | Added `routing_mode="learned"` path. Per-token `w = σ(W_L·x + b_L)` where `W_L` is the shared per-layer router. `delta = w·δ_V + (1-w)·δ_A`. Router stored via `object.__setattr__` to bypass `nn.Module` submodule registration (see DDP note below). `_last_router_w` cached per-module for balance loss. **Auto-fallback to merge** (50/50) if `self.router is None` or its `in_features` doesn't match `x.shape[-1]` — needed because `down_proj` takes intermediate-dim (18944) input while router is sized for hidden_size (3584). |
| `verl/models/cooperative/cooperative_wrapper.py` | Added `_create_routers()` producing `nn.ModuleList` of 28 × `nn.Linear(3584, 1)` (zero-init → σ(0)=0.5). `_replace_target_modules` attaches the router to q/k/v/o/gate/up in each layer (all in_features=3584); **down_proj (in_features=18944) is left unrouted** and falls back to merge at forward time. New `balance_weight` kwarg. Balance loss in `forward()` skips modules whose `_last_router_w is None`. Save/load via `routers.pt` + `cooperative_config.json["routing_mode"]="learned"`. `generate()` hook leaves `token_mask=None` in learned mode. |
| `train_cooperative.py` | New CLI args `--balance_weight`, `--router_warmstart_samples`. Rank-0 warm-start pass before `trainer.train()` (DDP broadcasts init params at startup). Tracks `mean_router_w` and `L_balance` in logging_steps output. |

### Key design choices (refined vs. original design doc)

**Router ownership (DDP-safe)**
Both the wrapper (`wrapper.routers[layer_idx]`) *and* the per-module `CooperativeLoRALinear` need a reference to the same router object. But if both registered it as a submodule, its parameters would appear twice in `wrapper.parameters()` and DDP would double-broadcast.
Fix: `CooperativeLoRALinear.set_router()` uses `object.__setattr__(self, "router", router)` so it holds a Python reference only. The `nn.ModuleList` in the wrapper is the sole owner.

**End-to-end gradient (no detach)**
Router input is raw `x`, not `x.detach()`. CE loss gradient flows back through the router, so routing is learned from both the balance objective and the downstream task. Matches the "no detach — let CE loss guide the router" recommendation.

**Warm-start: closed-form logistic fit**
Per-layer hidden-state collection over `N=16` samples:
- `where_class` = IMAGE tokens ∪ (assistant ∩ (coord ∨ think))
- `what_class` = assistant ∩ ¬where_class
- `μ_w, μ_a` = class-mean hidden states per layer
- `direction = μ_w − μ_a`
- Set `w_vec = ((where_bias − what_bias) / ‖direction‖²) · direction`, `b = −w_vec · midpoint`
- Targets: `σ(w_vec·μ_w + b) ≈ 0.9` (where), `σ(w_vec·μ_a + b) ≈ 0.1` (what)

This avoids a sklearn/CPU round-trip and approximates 1-step logistic regression. Only executed on rank 0; DDP's init broadcast propagates to other ranks.

**Balance loss (binary entropy)**
For each coop module, `mean_w = mean(_last_router_w)` over batch/seq. Loss = `-H(mean_w) = mean_w·log(mean_w) + (1-mean_w)·log(1-mean_w)`, averaged across modules. λ = 0.01.
Pushes *mean* routing toward 0.5 — does *not* penalize confident per-token routing (w≈0 or w≈1 for individual tokens is fine).

### Training config (`scripts/exp_cooperative/train_v8_ac_learned_router.slurm`)

Switched to **1-node** after observing 2-node backfill delays.

| Param | Value |
|-------|-------|
| nodes × GPUs | 1 × 4 GH200 |
| `per_device_batch_size` | 1 |
| `gradient_accumulation_steps` | 8 |
| Effective batch size | 32 (same as v6.5 coord_routing) |
| `num_epochs` | 3.0 (vs. v6.5's 4.0 — wall clock ~2× slower on 1 node) |
| `lora_r` / `lora_alpha` | 256 / 512 |
| `target_modules` | q, k, v, o, gate, up, down |
| `routing_mode` | `learned` |
| `balance_weight` | 0.01 |
| `router_warmstart_samples` | 16 |
| `bind_weight` | 0.0 (no binding loss) |
| `cooperative_comm` | on, `gate_type=tanh`, `gate_init=0`, `gate_lr_mult=100` |
| `max_length` / `image_max_pixels` | 16384 / 602112 |
| Output dir | `train_GUI_360/llamafactory/output/cooperative_v8_ac_learned_router/` |

### Evaluation (`scripts/exp_cooperative/eval_v8_ac_learned_router.slurm`)

Reuses `evaluation/eval_cooperative_ac_trajectory.py` with no modifications. The loader reads `routing_mode="learned"` from `cooperative_config.json` and reconstructs routers from `routers.pt` via `wrapper.load_cooperative_checkpoint()`. No CLI flags needed.

### Differences vs. original design doc

| Section in original | Revised implementation |
|---------------------|-----------------------|
| Per-layer router (Option B) | ✓ adopted |
| Warm-start: "set router.weight = (mean_img − mean_text) / ‖·‖²" | ✓ generalized to WHERE/WHAT classes (not just image/text) |
| detach or not | **no detach**: end-to-end CE gradient through router |
| Balance loss: `-H(mean_w)` | ✓ adopted, λ=0.01 |
| "No need for coord routing / α-mixing" | ✓ wrapper `generate()` hook skips all of them in learned mode |
| vLLM inference via `base + δ_A + w·(δ_V−δ_A)` | ⏸ pending (not in scope for first train run) |

### Job tracking

- **Verification job** 3833920 (AC hidden-state probes): completed → see `eval_analysis.md §11.3`
- **Training job** 3844610: 1-node, eff_bs=32, 3 epochs — submitted 2026-04-16 03:16, pending (Priority)
- Prior 2-node submission 3844522: cancelled (queue backfill too slow)
