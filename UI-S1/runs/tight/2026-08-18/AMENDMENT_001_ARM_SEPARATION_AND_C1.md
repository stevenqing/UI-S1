# TIGHT Amendment 001: selection/generation arm separation and C1 reparameterization

Round: `tight`

Amendment: `001`

Date: 2026-08-18

Status: `FROZEN_BEFORE_ANY_TIGHT_RESULT`

Scope: this amendment supersedes the base specification's global G-G1 interpretation, adds a zero-cost selection arm and a TILE-curve generation estimate, and replaces C1's collinear regression parameterization. It changes no window size, k grid, primary score, controls, sample, model, or Stage-1 authorization status.

## Legality and dependency timing

No TIGHT preflight, Stage-0 output, window manifest, model forward, or statistic existed when this amendment was written. TIGHT Stage 1 remains unauthorized.

TILE currently has only a committed preregistration in this repository; its Stage-0 eccentricity curves do not yet exist. TIGHT may implement its input lock and non-curve Stage-0 components, but it must not complete Stage 0, estimate the generation arm, or request Stage-1 authorization until TILE has committed its cross-fitted small/large eccentricity curves, boundaries, fit memberships, fallback flags, and hashes. No hand-entered, OWIN-derived, global, or newly fitted substitute curve is allowed.

## Why G-G1 is not a generation-arm gate

The base specification clarified that the TIGHT proposed answer is the new tight-window output, not the frozen block representative. Therefore the frozen fixed-output block oracle is neither an upper nor a lower bound on generated-coordinate correctness:

- generation can recover a correct point when every frozen representative is wrong;
- generation can drift away when a frozen representative is correct.

The base claim that G-G1 globally stops TIGHT Stage 1 is withdrawn. G-G1 applies only to the selection arm defined below. TIGHT Stage 0 no longer has an automatic global pre-GPU stop for the generation arm. Any later Stage-1 request remains a separate human-reviewed amendment and must present both arm estimates, all limitations, exact calls, and controls.

## Two zero-extra-cost arms

The same primary tight-window forwards produce one score per top-k block. For each row, select block

$$
b^*=\arg\max_b(s_b,-rank_b),\qquad s_b=-\|o_b-c_b\|_2/H_c.
$$

Unmappable scores and ties follow the base specification.

### Arm S: selection arm

Arm S returns the frozen representative point of selected block $b^*$. It measures whether tight-window self-consistency scores discriminate among frozen block outputs. It does not use the newly generated coordinate as the answer.

Arm S is strictly bounded by the fixed-output oracle over the evaluated top-k blocks. Report Arm-S accuracy versus B3 and M1, observed repair/damage, and fraction of the fixed-output oracle realized.

G-G1 is renamed `G-G1-S`: if k=3 fixed-output oracle gain versus B3 is strictly below 0.007, the selection-arm direction is stopped before GPU and receives `G_K1_SELECTION_ARM_CLOSED`. Equality passes. This does not stop or bound Arm G.

### Arm G: generation arm

Arm G returns the parsed tight-window output $o_{b^*}$, exactly as the base specification's proposed answer. It measures whether local relocalization creates a better coordinate after block selection.

G-P1 and G-P2 in the base specification are renamed G-P1-G and G-P2-G and continue to use Arm G. Add G-P1-S and G-P2-S for Arm S. Both arms use the same selected block and therefore require no additional forward.

Every result table must show S and G side by side. A positive G result with negative S means coordinate generation, not block discrimination, supplies the gain. A positive S result with negative G means generation destroys a useful selection. Neither may be described as the other.

## Stage-0 generation-arm estimate from TILE

After TILE Stage 0 is committed, apply its exact fold-local curve artifacts without refitting:

- for row in outer fold f, use TILE's outer-development-refit curve for fold f;
- use TILE's frozen small/large target-scale assignment and numeric eccentricity bins;
- preserve TILE fallback flags and curve values exactly.

For each row and each top-k block, compute target-center eccentricity relative to the TIGHT window center using TIGHT dimensions:

$$
e_{tight}=\sqrt{((c_x-m_x)/(W_c/2))^2+((c_y-m_y)/(H_c/2))^2}.
$$

Here c is GT bbox center and m is the final achieved tight-window center after image-bound translation. Map $e_{tight}$ to the TILE scale-specific curve. The selected-block Arm-G predicted correctness is the curve value for the block selected by the zero-GPU proxy available at Stage 0 only if that selected block is defined without GPU output; because the primary TIGHT score itself requires inference, Stage 0 must report two bounds instead of inventing a selected block:

1. `G-estimate-best`: maximum curve value among top-k tight windows;
2. `G-estimate-rank1`: curve value for frozen rank-1 block.

These are optimistic/descriptive brackets, not predictions of post-inference block selection, B3, or M1. Report expected net versus B3 for both, using the same fractional repair/damage ledger as TILE.

TIGHT windows are typically much smaller than TILE's 1288 by 728 source crops. Eccentricity normalization makes coordinates dimensionless but does not remove resolution, context, or preprocessing shift. Every generation estimate is labeled `EXTRAPOLATION_LIMITED_TIGHT_VS_TILE_WINDOW_SCALE`. Report TIGHT/TILE width, height, area, and processor-resize ratios. No generation estimate is an automatic gate.

If TILE Stage 0 stops without producing valid cross-fitted curves, TIGHT generation estimation is `BLOCKED_TILE_CURVE_UNAVAILABLE`; do not fit a TIGHT-specific curve from current labels as a replacement.

## Revised Stage-0 adjudication

Stage 0 must report, in this order:

1. top-k contains-correct and fixed-output oracle curves;
2. Arm-S G-G1-S result;
3. TILE-based Arm-G rank1/best estimates with extrapolation diagnostics;
4. LOOK complements and damage-domain locks;
5. selected k by fold and endpoint status.

There is no automatic global Stage-0 stop. A Stage-1 amendment may request:

- both S and G when G-G1-S passes;
- G only when G-G1-S fails, with S retained as a zero-cost diagnostic;
- neither when human review rejects the extrapolation-limited G estimate.

Any request must state this choice before GPU authorization. The base specification's statement that G-G1 alone stops all Stage 1 is superseded.

## C1 regression reparameterization

The base parameterization used candidate coordinate C and achieved window center W directly. Since C-W is a fixed 0.30-Hc directional offset before boundary effects, C and W are highly collinear.

Use instead

$$
O=\alpha_{axis}+b_o(C-W)+b_W W+\epsilon.
$$

Coordinates remain normalized by image width for x and image height for y. Use achieved W after boundary translation, so C-W captures the actual offset. Fit one stacked weighted OLS with separate x/y intercepts, shared $b_o,b_W$, inverse-inclusion weights, and application bootstrap.

The relation to the original coefficients is

$$
b_o=\beta_c,\qquad b_W=\beta_c+\beta_w,
$$

so

$$
\beta_c=b_o,\qquad \beta_c-\beta_w=2b_o-b_W.
$$

G-P3 passes only when both 99% lower bounds for $b_o$ and $2b_o-b_W$ are strictly positive. Undefined/singular fits fail. Report design-matrix rank, singular values, condition number, coefficient covariance, and bootstrap finite count.

For anisotropy diagnostics, additionally fit x-only and y-only models with intercept and predictors `(C-W),W`. Report $b_{o,x}$, $b_{o,y}$, their difference, and a paired application-bootstrap 99% interval. This diagnostic has no threshold and does not alter G-P3. Horizontal/vertical differences must be discussed because the window is not square and offsets are scaled by Hc.

## Endpoint and kill-condition renaming

- G-P1-G: Arm G minus B3, primary generated-coordinate endpoint.
- G-P2-G: Arm G minus M1.
- G-P1-S: Arm S minus B3, selection diagnostic.
- G-P2-S: Arm S minus M1.
- G-P3: reparameterized C1 blocking regression above.
- G-P4 through G-P7 remain, but G-P4 must show S and G repair/damage separately.

G-K1 becomes selection-arm-only `G_K1_SELECTION_ARM_CLOSED`. G-K4 applies to G-P1-G. Add `G_K4S` when G-P1-S lower bound is not positive; it closes only the selection claim. G-K6 is evaluated separately for G and S. G-K7 damage-versus-repair is reported separately for each arm.

No threshold, window-size, offset, k-grid, curve, or control change is authorized by this amendment.