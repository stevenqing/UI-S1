# OWIN Amendment 001: calibration estimand and budget revision

Round: `owin`

Amendment: `001`

Date: 2026-08-17

Status: `FROZEN_BEFORE_ANY_OWIN_RESULT_OR_GPU_AUTHORIZATION`

Scope: this amendment supersedes the Arm A estimand, oracle-window family, call budget, Arm B saturation reporting, and secondary coverage definition in `SPEC.md`. It does not change O-I1, O-I2, or O-I3 thresholds.

## Legality and timing

This amendment was written after design review and before any OWIN output. At the time of writing, `runs/owin/2026-08-17/` contained only the committed base `SPEC.md` and `configs/owin_prereg.yaml`. No OWIN preflight, Arm B geometry, sample roster, window manifest, model forward, parsed output, or statistic existed. Arm A had not been authorized.

If any OWIN statistic is found to predate this amendment commit, this amendment is void and the revised design must be labeled a post-result second attempt.

The O-I boundaries remain exactly 0.05 and 0.10. Values equal to either boundary remain O-I2. This amendment changes the estimand to which those boundaries apply; every OWIN report must disclose that fact.

## Reason for revision

The base specification compared one GT-centered crop forward with existing common-stratum B3 accuracy from a 12-slot pool. That contrast combines window placement with eleven additional forwards and aggregation. A systematically negative calibration difference could then be added back to low-coverage strata and could alone push the corrected opportunity across O-I3. Because O-I3 is the anticipated direction, this mismatch is especially unsafe.

The primary Arm A estimand is therefore revised to a matched 12-forward oracle pool.

## Matched oracle pool

Each sampled row receives exactly 12 GTA1-7B forwards:

- pool slot 0: the original full image, exactly matching existing view 0;
- pool slots 1 through 11: 11 GT-centered jittered crops, each exactly 1288 by 728 pixels.

The crop-jitter index is `j=0,...,10`; it is distinct from the pool slot. Crop-jitter 0 maps to pool slot 1 and has zero offset. Pool slot 0 always means the full image. Thus the pool has one full-image call plus 11 crop calls.

### Deterministic spiral

Let the crop short edge be 728 and $R=0.15\times728=109.2$ pixels. For crop-jitter index $j=0,\ldots,10$ define

$$
r_j=R\frac{j}{10},\qquad \theta_j=\frac{2\pi j}{11}.
$$

The unrounded offset is $(r_j\cos\theta_j,r_j\sin\theta_j)$. Round each component independently to the nearest integer, with exact half ties away from zero. This is an Archimedean one-turn equal-angle spiral. It uses no random number, seed, row property, target size, image size, or model output beyond centering on the GT bbox.

The normative integer offset list, in crop-jitter order 0 through 10, is:

`[(0,0),(9,6),(9,20),(-5,32),(-29,33),(-52,15),(-63,-18),(-50,-58),(-12,-86),(41,-89),(92,-59)]`.

Implementations must reproduce this list exactly before opening any row. If a platform's trigonometric evaluation disagrees, the normative integer list controls.

### Window placement and constraint repair

Let $(c_x,c_y)$ be the GT bbox center and $(d_{x,j},d_{y,j})$ the rounded offset. The requested crop center is $(c_x+d_{x,j},c_y+d_{y,j})`. Construct the initial integer top-left as

$$
l_j=\lfloor c_x+d_{x,j}-644\rfloor,\qquad
t_j=\lfloor c_y+d_{y,j}-364\rfloor.
$$

First apply the smallest axis-wise integer translation that makes the GT center lie inside the half-open crop: $l_j\le c_x<l_j+1288$ and $t_j\le c_y<t_j+728$. Then apply the smallest axis-wise translation that keeps the full crop inside the image. If the image-bound translation would violate center containment, choose the unique nearest feasible integer top-left in the intersection

$$
[0,W-1288]\cap( c_x-1288,c_x]
$$

for the horizontal coordinate and analogously for vertical. Ties choose the smaller coordinate. Images smaller than the crop are infeasible and stop before inference. Never resize, rotate, or alter the jitter offset family.

For every row and crop record requested offset, rounded offset, initial rectangle, center-containment repair, image-bound repair, final rectangle, final translation, target-center containment, and full-bbox containment.

## Pool-level aggregators and calibration

Apply canonical B3 and fold-local M1_ccm separately to the 12 oracle candidates. M1_ccm is fit without outer-test leakage using the existing five-fold application protocol and the same source semantics as the matched pool. Neither aggregator may use GT other than final correctness evaluation and oracle-window construction.

Let $A_{s,a}^{pool}$ be oracle-pool conditional accuracy in stratum $s$ for aggregator $a$.

### Primary B3 calibration

The O-I estimand remains B3-based. Its matched calibration is

$$
\delta_{pool,B3}=A_{common,B3}^{pool}-0.8195488721804511,
$$

and

$$
\widetilde A_{s,B3}^{pool}=\operatorname{clip}(A_{s,B3}^{pool}-\delta_{pool,B3},0,1).
$$

The corrected full-benchmark accuracy and opportunity retain the base formulas with $\widetilde A_{s,B3}^{pool}$. O-I1 through O-I3 use only the corrected B3 pool-level point estimate.

### Secondary M1 calibration

The B3 common anchor must not be used for M1. Before OWIN outputs, the no-result input-lock implementation must define the existing-pool M1 common-stratum baseline as the row-weighted accuracy of the frozen fold-local M1_ccm outputs on all 931 `common_11` rows. Its value and SHA-256 provenance enter the preflight; it cannot be selected or changed after oracle outputs.

Define

$$
\delta_{pool,M1}=A_{common,M1}^{pool}-B_{common,M1}^{pool},
$$

and apply the same clipped equal-shift correction. Report M1 pool-level raw and corrected opportunities as secondary measurements. They do not affect O-I classification.

All raw pool-level values remain visible. Both calibration differences and corrections are recomputed inside every joint grouped-bootstrap replicate.

## Single-forward matched report

The zero-offset oracle crop is crop-jitter 0, pool slot 1. It requires no additional call.

For each stratum $s$, define $B_s^{single}$ as the equal-weight mean correctness over the 11 existing GTA1 crop slots views 1 through 11 and all population rows in that stratum:

$$
B_s^{single}=\frac{1}{11n_s}\sum_{i\in s}\sum_{v=1}^{11}\mathbf 1[\text{GTA1}_{i,v}\text{ correct}].
$$

This is a fixed population anchor reconstructed from the frozen bank before oracle outputs. It is not best-slot, selected-slot, B3, or M1 accuracy. The common calibration is

$$
\delta_{single}=A_{common}^{oracle,0}-B_{common}^{single},
$$

with

$$
\widetilde A_s^{single}=\operatorname{clip}(A_s^{oracle,0}-\delta_{single},0,1).
$$

Report raw and corrected single-forward conditional accuracies and their grouped-bootstrap intervals. They do not enter O-I classification.

Report the paired pool-versus-single difference separately for B3 and M1_ccm where defined. This describes the value of moving from one zero-offset oracle crop to one-full-plus-eleven-crop aggregation; it is not an estimate derived from historical $N_{eff}=1.5726$. The historical value is contextual only.

## Revised budget and authorization

The Arm A budget is exactly 4,800 GTA1-7B forwards: 400 sampled rows times 12 calls. Every base-spec reference to exactly 400 Arm A forwards is superseded by 4,800. Sample sizes remain uncovered 150, partial 150, and common 100.

The future execution amendment must bind 4,800 calls, shard count, hardware, runner hash, model/index hashes, and a one-time authorization nonce. This amendment does not authorize GPU execution.

## Arm B saturation handling

Define $N^*$ only when there is near-saturation before 11. It is the smallest $N\in\{4,\ldots,10\}$ such that both:

1. median exact tiling union-area fraction is at least 0.99;
2. target-center coverage $q_{s,N}$ is at least 0.99 in each of the three frozen COVER strata.

If no such N exists, set `N_star=NONE`; do not call N=11 saturation by definition. The thresholds are fixed before Arm B and are not gates.

When $N^*$ exists, report its median union area, all three $q_{s,N^*}$ values, center-coverage transition table, and full-bbox transition table. For every $N\ge N^*$, state that factorized $G_N$ approaches the perfect-coverage opportunity and carries diminishing independent geometric information; still report the exact $G_N$ values rather than replacing them by $U_{perfect}$.

Separately compare existing 11-window median union `0.3219237075617284` with fixed-tiling median union at $N^*$. This is direct evidence of proposer geometric inefficiency, not model accuracy. Saturation is not Arm B failure and changes no threshold.

## Secondary full-bbox coverage

The primary sampling strata remain COVER target-center `uncovered_0`, `partial_1_10`, and `common_11`. They are never reassigned after sampling.

Add a secondary full-bbox count for every row and layout: the number of windows whose half-open rectangle fully contains the GT bbox under

$$
l\le x_1,\quad t\le y_1,\quad x_2\le r,\quad y_2\le b.
$$

Report full-bbox `uncovered_0`, `partial_1_10`, and `common_11` distributions and transitions for Arm B at every N, alongside center coverage. For Arm A, report pool correctness within both the original center-based sampling strata and the secondary existing-window full-bbox strata using the frozen inverse-inclusion weights. The full-bbox strata are descriptive domains only; they do not alter sampling, calibration, O-I thresholds, or primary center-based interpretation.

## Unchanged provisions

Sampling allocation, SHA-256 row ranking, inverse-probability weights, application-group bootstrap, 10,000 replicates, 99% limits, seed base, model/source revisions, prompt, processor, greedy decoding contract, coordinate-transform tests, trace retention, label/trace separation, backup path, X2/SPLIT non-conflict, Mind2Web exclusion, post-selection status, single-benchmark scope, and `GT_ORACLE_NON_DEPLOYABLE` labeling remain unchanged.

The non-deployable label applies to both pool-level and single-forward oracle measurements.

If O-I3 triggers, any later round still requires a preregistered net-benefit ledger including damage on the currently successful 85.77% of rows. If $N^*$ exists, the first follow-up comparison must be the fixed $N^*$ tiling against the existing 11-window pool at equal or lower budget, not a complex placement mechanism.

## Revised execution order

1. Commit this amendment before any OWIN result or GPU authorization.
2. Run and commit the no-result preflight and Arm B under the base spec plus this amendment.
3. Commit the 400-row sample manifest and all 12 per-row window records, including jitter and repair fields.
4. Commit runner/tests; tests must cover spiral determinism, index mapping, half-tie rounding, center containment, full-bbox reporting, image-bound repair, and crop-local to full-image coordinate transformation.
5. Commit an execution amendment and explicit one-time authorization for exactly 4,800 calls.
6. Execute 4,800 forwards.
7. Parse only the 100 common rows times 12 outputs; compute and freeze $\delta_{pool,B3}$, $\delta_{pool,M1}$, and $\delta_{single}$.
8. Parse partial and uncovered outputs without changing estimands, corrections, samples, runner, or thresholds.
9. Apply O-I1 through O-I3 only to the corrected B3 pool-level point estimate. Report M1, single-forward, and pool-minus-single measurements beside it.