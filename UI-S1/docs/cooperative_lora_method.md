# Cooperative LoRA: SVD Decomposition and Expert Architecture

## 1. Motivation

Standard LoRA adds a low-rank adapter $\Delta W = BA$ to each frozen weight matrix, where $B \in \mathbb{R}^{d_{out} \times r}$ and $A \in \mathbb{R}^{r \times d_{in}}$. Every token sees the same rank-$r$ update — there is no conditional computation.

Cooperative LoRA extends this to multiple experts: $K$ separate $A$ matrices share a single $B$ matrix, with per-token routing to blend expert outputs. Different tokens select different rank-$r$ subspaces, giving effective capacity beyond rank-$r$ while keeping the shared $B$ projection efficient.

## 2. SVD Extraction from Full-Parameter SFT

### 2.1 Problem Setup

Given a full-parameter SFT model $W_{\text{sft}}$ and the original base model $W_{\text{base}}$, we compute the weight delta for each target module (q_proj, k_proj, v_proj, o_proj in each of 28 transformer layers):

$$\Delta W = W_{\text{sft}} - W_{\text{base}}$$

For Qwen2.5-VL-7B, each attention projection is $[3584, 3584]$ (q/o) or $[512, 3584]$ (k/v with GQA).

### 2.2 Truncated SVD

We perform SVD on each $\Delta W$:

$$\Delta W = U \Sigma V^T$$

and truncate to rank $r$ (we use $r=128$):

$$\Delta W \approx U_r \Sigma_r V_r^T$$

where $U_r \in \mathbb{R}^{d_{out} \times r}$, $\Sigma_r = \text{diag}(\sigma_1, \ldots, \sigma_r)$, $V_r^T \in \mathbb{R}^{r \times d_{in}}$.

**Reconstruction error**: $\|\Delta W - U_r \Sigma_r V_r^T\| / \|\Delta W\| \approx 0.31$ (mean across all modules).

**Energy captured**: $\sum_{i=1}^{r} \sigma_i^2 / \sum_{i=1}^{d} \sigma_i^2 \approx 75.86\%$ at rank 128.

### 2.3 Distributing Singular Values into B and A

LoRA applies a scaling factor $s = \alpha / r$ (we use $\alpha=256, r=128$ so $s=2$). The forward pass computes:

$$\text{output} = W_{\text{base}} x + B \cdot A_{\text{blend}} \cdot x \cdot s$$

We want $B \cdot A_{\text{avg}} \cdot s \approx \Delta W$, i.e., $B \cdot A_{\text{avg}} \approx \Delta W / s$.

We split the singular values symmetrically:

$$B = U_r \cdot \text{diag}(\sqrt{\sigma_i / s}) \quad \in \mathbb{R}^{d_{out} \times r}$$
$$A_{\text{avg}} = \text{diag}(\sqrt{\sigma_i / s}) \cdot V_r^T \quad \in \mathbb{R}^{r \times d_{in}}$$

This gives $B \cdot A_{\text{avg}} = U_r \cdot \text{diag}(\sigma_i / s) \cdot V_r^T = \Delta W_r / s$, and thus $B \cdot A_{\text{avg}} \cdot s = \Delta W_r \approx \Delta W$.

## 3. Expert Initialization

### 3.1 V15: 2-Expert Symmetric Perturbation

For $K=2$ experts, we create two $A$ matrices by adding/subtracting noise:

$$A_1 = A_{\text{avg}} + \delta, \quad A_2 = A_{\text{avg}} - \delta$$

where $\delta = \epsilon \cdot \text{randn}(r, d_{in}) \cdot \frac{\|A_{\text{avg}}\|_F}{\sqrt{r \cdot d_{in}}}$ and $\epsilon$ is `noise_scale` (default 0.1).

**Centroid preservation**: $\frac{1}{2}(A_1 + A_2) = A_{\text{avg}}$ exactly.

**Init equivalence**: Route weights are initialized to zeros. Sigmoid(0) = 0.5 gives equal blend:

$$\text{output} = B \cdot (0.5 \cdot A_1 + 0.5 \cdot A_2) \cdot x \cdot s = B \cdot A_{\text{avg}} \cdot x \cdot s \approx \Delta W \cdot x$$

At initialization, the cooperative LoRA is functionally identical to the rank-$r$ SVD approximation of full SFT.

### 3.2 V18: K-Expert Gram-Schmidt Perturbation

For $K > 2$ experts, symmetric perturbation ($+\delta / -\delta$) does not generalize. We use **Gram-Schmidt orthogonalized perturbation**:

**Step 1**: Generate $K$ random noise vectors $\{n_k\}_{k=1}^{K}$ in $\mathbb{R}^{r \times d_{in}}$.

**Step 2**: Gram-Schmidt orthogonalize. For each $n_k$, subtract its projections onto all previous orthogonalized vectors:

$$\hat{n}_k = n_k - \sum_{j < k} \frac{\langle n_k, \hat{n}_j \rangle}{\langle \hat{n}_j, \hat{n}_j \rangle} \hat{n}_j, \quad \hat{n}_k \leftarrow \frac{\hat{n}_k}{\|\hat{n}_k\|}$$

This produces $K$ mutually orthogonal unit directions in parameter space.

**Step 3**: Scale to match target perturbation magnitude:

$$\tilde{n}_k = \hat{n}_k \cdot \epsilon \cdot \frac{\|A_{\text{avg}}\|_F}{\sqrt{r \cdot d_{in}}}$$

**Step 4**: Center to preserve the centroid:

$$\delta_k = \tilde{n}_k - \frac{1}{K}\sum_{k'=1}^{K} \tilde{n}_{k'}$$

This ensures $\sum_k \delta_k = 0$.

**Step 5**: Create expert $A$ matrices:

$$A_k = A_{\text{avg}} + \delta_k, \quad k = 0, \ldots, K-1$$

**Properties**:
- **Centroid preservation**: $\frac{1}{K}\sum_k A_k = A_{\text{avg}}$
- **Orthogonal diversity**: The perturbation directions are mutually orthogonal, maximizing initial expert diversity
- **Init equivalence**: With softmax routing initialized to zeros, all weights are $1/K$ (uniform), so the blend equals $A_{\text{avg}}$, reproducing the full SFT output

For $K=2$ this reduces to the V15 method (two antipodal directions on the unit circle).

## 4. Architecture

### 4.1 Forward Pass

For each target linear layer $W \in \{$q_proj, k_proj, v_proj, o_proj$\}$ in each transformer layer:

```
Input x ∈ [B, S, D]

1. Base output:      y_base = W_base @ x

2. Routing:          logits = x @ W_route          (W_route ∈ [D, K])
                     w = softmax(logits, dim=-1)    ([B, S, K])

3. Expert projections:  h_k = A_k @ x_drop         (K × [B, S, r])

4. Communication:    h_k' = CommTopology(h, w)      (T rounds)

5. Blend:            h_blend = Σ_k w_k · h_k'      ([B, S, r])

6. B projection:     delta = B @ h_blend * (α/r)

7. Output:           y = y_base + delta
```

V15 (2-expert) uses sigmoid routing instead of softmax: $r = \sigma(x \cdot w_{\text{route}})$, and the blend is $r \cdot h_1 + (1-r) \cdot h_2$.

### 4.2 Routing

| Version | Routing Mechanism | Route Params per Layer | Initialization |
|---------|-------------------|----------------------|----------------|
| V15 | $r = \sigma(x \cdot w)$, $w \in \mathbb{R}^{D}$ | $D = 3584$ | zeros $\to \sigma(0) = 0.5$ |
| V18 | $w = \text{softmax}(x \cdot W)$, $W \in \mathbb{R}^{D \times K}$ | $D \times K$ | zeros $\to$ uniform $1/K$ |

V18's softmax routing is strictly more expressive: each expert gets its own projection direction in the $D$-dimensional input space, rather than sharing a single direction.

### 4.3 Communication Topologies

After expert projections and before blending, experts exchange messages in the low-rank space ($r$-dimensional) through $T$ communication rounds (default $T=2$).

**V15 (2-expert, pairwise)**:
```
For each round t:
    g_12 = σ(h_1 @ gate_12[t])       # [B,S,1] gate
    h_1  = h_1 + g_12 · (h_2 @ W_12[t])
    g_21 = σ(h_2 @ gate_21[t])
    h_2  = h_2 + g_21 · (h_1 @ W_21[t])   # uses updated h_1
```

Params per layer per round: $W_{12}[r,r] + W_{21}[r,r] + \text{gate}_{12}[r] + \text{gate}_{21}[r] = 2r^2 + 2r = 32,896$

**V18 topologies** (all use shared $B$):

| Topology | Description | Params per layer per round |
|----------|-------------|---------------------------|
| `none` | No communication | 0 |
| `top2` | Only top-2 activated experts communicate via shared W | $r^2 + r = 16,512$ |
| `shared` | All pairs share same W; expert $i$ receives avg message from all $j \neq i$ | $r^2 + r = 16,512$ |
| `full` | Each directed pair $(j \to i)$ has its own W and gate | $K(K-1) \times (r^2 + r)$ |

**Total communication parameters** (28 layers, $T=2$):

| Topology | V15 ($K=2$) | V18 $K=4$ | V18 $K=8$ |
|----------|-----------|---------|---------|
| `none` | — | 0 | 0 |
| `top2` / `shared` | — | 0.92M | 0.92M |
| `full` | 1.84M | 11.1M | 51.8M |
| V15 pairwise | 1.84M | — | — |

### 4.4 Shared B Matrix

All $K$ experts share the same $B \in \mathbb{R}^{d_{out} \times r}$. This is key for efficiency: the blend produces a single $[B, S, r]$ vector, so only one $B$-projection (full-rank matmul) is needed regardless of $K$.

The shared $B$ also constrains all experts to project into the same output subspace, ensuring the model cannot degenerate into $K$ independent LoRA adapters.

## 5. Training

### 5.1 GSPO (Group-relative Self-Play Optimization)

Training uses on-policy RL with GUI-360 multi-step trajectories. For each batch:

1. **Rollout**: Generate trajectories with the current policy (cooperative LoRA active)
2. **Reward**: Per-step reward from GUI-360 environment (action correctness, progress)
3. **Advantage**: Group-relative advantage: $\hat{A}_t = r_t - \text{mean}(r_{t,\text{group}})$
4. **PPO clip**: Token-level PPO with clip ratio $\epsilon = 0.2$

### 5.2 Training Losses

**Policy gradient loss** (PPO-clip):

$$L_{\text{pg}} = -\mathbb{E}\left[\min\left(\frac{\pi_\theta}{\pi_{\text{old}}} \hat{A}, \text{clip}\left(\frac{\pi_\theta}{\pi_{\text{old}}}, 1 \pm \epsilon\right) \hat{A}\right)\right]$$

**Balance loss** (routing entropy regularization):

$$L_{\text{bal}} = -H(p) = \sum_{k=1}^{K} p_k \log p_k$$

where $p_k = \frac{1}{|\text{modules}|} \sum_m \frac{1}{BS} \sum_{b,s} w_{b,s,k}^{(m)}$ is the mean usage of expert $k$.

Minimizing $L_{\text{bal}}$ maximizes entropy, pushing toward uniform usage. Perfect balance: $H = \log K$.

**Diversity loss** (V18 only — pairwise cosine similarity):

$$L_{\text{div}} = \frac{1}{|\text{modules}|} \sum_m \frac{2}{K(K-1)} \sum_{i < j} \cos(h_i^{(m)}, h_j^{(m)})$$

where $h_i^{(m)}$ is the flattened $[B \cdot S \cdot r]$ expert output from module $m$.

Minimizing $L_{\text{div}}$ pushes expert representations apart, preventing the symmetric co-movement we observed in V15 (see `docs/v15_v18_parameter_analysis.md`).

**Total loss**:

$$L = L_{\text{pg}} + \lambda_{\text{bal}} \cdot L_{\text{bal}} + \lambda_{\text{div}} \cdot L_{\text{div}}$$

Default: $\lambda_{\text{bal}} = 0.01$, $\lambda_{\text{div}} = 0.001$.

### 5.3 Optimizer Groups

| Group | Parameters | Learning Rate |
|-------|-----------|--------------|
| Decay | LoRA A, B matrices | $1 \times 10^{-5}$ (K=4), $5 \times 10^{-6}$ (K=8) |
| No-decay | biases (none in our setup) | same |
| Route | $W_{\text{route}}$ matrices | $5 \times 10^{-4}$ (K=4), $3 \times 10^{-4}$ (K=8) |
| Comm | $W_{\text{comm}}$, gates | same as LoRA |

Route weights use higher LR because they start from zero and need to move faster. V18 uses lower route LR than V15 ($5 \times 10^{-4}$ vs $1 \times 10^{-3}$) because softmax is more sensitive to logit changes than sigmoid.

### 5.4 Routing Noise

During RL rollouts (not gradient computation), Gaussian noise is added to routing logits:

$$\text{logits}_{\text{noisy}} = \text{logits} + \mathcal{N}(0, \sigma^2)$$

This encourages exploration of different expert combinations. Default: $\sigma = 0.3$ (K=4), $\sigma = 0.2$ (K=8, more experts already provides diversity).

## 6. Parameter Count

| Config | A params | B params | Route | Comm (top2) | Total | % of base 7.6B |
|--------|----------|----------|-------|-------------|-------|----------------|
| V15 $K=2$ | 103M | 51M | 100K | 1.85M | ~156M | 2.1% |
| V18 $K=4$ | 206M | 51M | 400K | 0.92M | ~258M | 3.4% |
| V18 $K=8$ | 411M | 51M | 800K | 0.92M | ~464M | 6.1% |

A parameters scale linearly with $K$. B parameters are constant (shared). Route and comm parameters are negligible.

## 7. Checkpoint Format

Each cooperative LoRA checkpoint consists of 4 files:

```
checkpoint/
  cooperative_config.json   # architecture config
  lora_weights.pt           # A and B matrices
  route_weights.pt          # routing weight matrices
  comm_weights.pt           # communication W and gate parameters
```

**cooperative_config.json**:
```json
{
  "lora_r": 128,
  "lora_alpha": 256,
  "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
  "num_comm_rounds": 2,
  "type": "k_expert_cooperative_v18",     // or "iterative_cooperative_v13"
  "num_experts": 4,                       // V18 only
  "comm_topology": "top2",               // V18 only
  "balance_weight": 0.01,
  "diversity_weight": 0.001              // V18 only
}
```

**lora_weights.pt key format**:
- V15: `base_model.model.language_model.layers.{L}.self_attn.{mod}.lora_A_1` / `lora_A_2` / `lora_B`
- V18: `base_model.model.language_model.layers.{L}.self_attn.{mod}.lora_A.{k}` / `lora_B`

**Tensor counts** (28 layers × 4 modules):
- V15: 336 tensors (112 × 3: A_1, A_2, B)
- V18 K=4: 560 tensors (112 × 5: A.0, A.1, A.2, A.3, B)
- V18 K=8: 1008 tensors (112 × 9: A.0–A.7, B)

## 8. SVD Energy Analysis

Rank-128 SVD captures ~75.86% of the spectral energy of $\Delta W$ across all modules. The remaining 24% is distributed across ranks 129–3584.

Representative analysis (Layer 14 q_proj):

| Rank | Energy Captured |
|------|----------------|
| 16   | 22.7%          |
| 32   | 37.2%          |
| 64   | 55.1%          |
| **128** | **75.9%**   |
| 256  | 91.3%          |

The energy capture is **identical** for all $K$ values (K=2, K=4, K=8) because it depends only on the SVD truncation rank, not on how many experts the $A$ matrix is split into. The Gram-Schmidt perturbation only changes the distribution of the $A$-space among experts — it does not affect $B$ or the total $B \cdot A_{\text{avg}}$ product.

## 9. Implementation Files

| File | Description |
|------|-------------|
| `v13_gui_360/iterative_cooperative_lora.py` | V15 2-expert LoRA layer (sigmoid, pairwise comm) |
| `v13_gui_360/iterative_cooperative_wrapper.py` | V15 VLM wrapper |
| `v15_gui_360/extract_fullsft_to_cooperative.py` | V15 SVD extraction (2-expert symmetric noise) |
| `v18_k_expert/k_expert_cooperative_lora.py` | V18 K-expert LoRA layer (softmax, topology comm) |
| `v18_k_expert/k_expert_cooperative_wrapper.py` | V18 VLM wrapper (+ diversity loss) |
| `v18_k_expert/extract_fullsft_to_k_expert.py` | V18 SVD extraction (K-expert Gram-Schmidt) |
| `v18_k_expert/train_k_expert_gspo.py` | V18 GSPO RL training |
| `v18_k_expert/serve_k_expert_direct.py` | V18 OpenAI-compatible serving |
