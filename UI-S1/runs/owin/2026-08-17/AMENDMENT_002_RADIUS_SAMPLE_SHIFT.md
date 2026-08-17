# OWIN Amendment 002: radius calibration, sample reallocation, and constant-shift diagnostics

Round: `owin`

Amendment: `002`

Date: 2026-08-17

Status: `FROZEN_BEFORE_ANY_OWIN_RESULT_OR_GPU_AUTHORIZATION`

Scope: this amendment supersedes Amendment 001 only for jitter-radius selection, Arm A stratum sample counts, total call budget, and the added dependence and constant-shift diagnostics. O-I1 through O-I3 retain their numerical thresholds and continue to act only on the corrected B3 pool-level point estimate.

## Legality and timing

This amendment was written before any OWIN output. At writing time, `runs/owin/2026-08-17/` contained only the committed base specification/config and Amendment 001/config. No preflight, Arm B geometry, sample roster, window roster, model forward, parsed output, or OWIN statistic existed. Arm A was not authorized.

If any OWIN statistic predates this amendment commit, the amendment is void and this design must be labeled a post-result second attempt.

Amendment 001's radius formula and normative integer offsets were correctly reproduced. This amendment does not claim an implementation error. It changes how the maximum radius R is chosen before sampling.

## Geometry-calibrated jitter radius

### Status of the calibration

The existing-window target $\mathrm{IoU}^*$ uses only frozen crop rectangles. Candidate oracle-window layouts additionally use GT bbox centers to instantiate and boundary-repair windows. Therefore the full radius calibration is not label-free. It is a zero-GPU, evaluation-side geometry calibration that may read image dimensions and GT bbox geometry, but it may not read candidate correctness, model outputs, target classes, aggregators, or any OWIN endpoint. Calling it label-free is prohibited.

The calibration is a design-selection output, not a scientific Arm A/B result. The preflight remains result-free with respect to model accuracy and O-I endpoints, while explicitly recording this geometry-only use of GT.

### Existing-window target

For each of all 1,581 rows, compute rectangle IoU for all 55 unordered pairs among existing GTA1 views 1 through 11. Take the ordinary median of the 55 values for that row. Define $\mathrm{IoU}^*$ as the ordinary median of the 1,581 row medians. For odd counts, the ordinary median is the center sorted value. Rectangles are half-open integer boxes and IoU is intersection area divided by union area.

### Candidate grid and layouts

The frozen radius grid is

$$
R\in\{109.2,150,200,250,300,340\}\ \text{pixels}.
$$

For each candidate R and jitter index $j=0,\ldots,10$, use Amendment 001's family

$$
r_j=R\frac{j}{10},\qquad \theta_j=\frac{2\pi j}{11},
$$

with component rounding to nearest integer and exact half ties away from zero. Instantiate the 11 crops on every row using Amendment 001's GT-center containment repair and image-bound repair. R=340 keeps every unrounded component below the half-width/half-height containment limits, but all repairs remain mandatory and recorded.

For each row and R, compute the median of all 55 pairwise IoUs among the final repaired oracle crops. Define $\mathrm{IoU}(R)$ as the median of the 1,581 row medians.

Select R by minimum absolute distance $|\mathrm{IoU}(R)-\mathrm{IoU}^*|$, tie-breaking by smaller R. If R=109.2 or R=340 is selected, record `RADIUS_GRID_ENDPOINT`; do not expand the grid or alter the criterion.

The committed geometry-calibration manifest must contain $\mathrm{IoU}^*$, all six $\mathrm{IoU}(R)$ values, signed and absolute differences, selected R, endpoint status, and each candidate's 11 normative integer offsets. The selected offset list controls over platform trigonometric disagreement and must be reproduced bit-for-bit before opening a sampled row.

No accuracy, correctness, model output, or O-I quantity may influence R.

## Oracle-pool dependence diagnostic

This diagnostic is mandatory and is not a gate.

For each original center-based sampling stratum separately, use the 11 oracle crop slots, excluding the full-image slot. Let each slot's variable be its binary error indicator. Reuse ICC Arm C's Pearson-phi sufficient statistics, undefined-pair handling, and empirical matrix effective sample size:

$$
N_{\mathrm{eff}}(R)=\frac{11^2}{\mathbf 1^\top R\mathbf 1}.
$$

The diagonal of R is one. Undefined off-diagonal phi is recorded and zero-filled in R, exactly as ICC Arm C. Also report mean valid phi, mean zero-filled phi, valid pair count, and undefined pair count among the 55 pairs.

For outer fold f, estimate the stratum matrix on sampled rows from the other four folds. Report five leave-one-fold-out values, their unweighted mean and full range. Also report a sample-row-count-weighted mean as a secondary summary. Do not pool strata for the primary diagnostic.

Because application allocation induces unequal inclusion probabilities, compute every fold's $n,s_x,s_y,s_{xy}$ sufficient statistic with the frozen inverse-inclusion weights. The application-group bootstrap resamples applications and recomputes these weighted sufficient statistics. Using unweighted sampled-row phi as the population diagnostic is prohibited; it may be reported only as an explicitly labeled sampling diagnostic.

Two references are mandatory:

1. reconstruct the same 11-slot statistic for existing GTA1 views 1 through 11 on the identical sampled rows, strata, and leave-one-fold-out splits;
2. report historical C-uni 12-slot empirical phi $N_{\mathrm{eff}}=1.5725697506279792$ (rounded 1.5726) as contextual only.

The 12-slot historical reference differs in pool cardinality and lineage structure and is not a matched equality target. For each matched GTA1 comparison report signed difference and relative absolute difference. A relative difference above 0.10 is labeled `MATERIAL_DEPENDENCE_MISMATCH`; at or below 0.10 is `APPROXIMATELY_MATCHED`. This label is descriptive and changes no O-I threshold or result.

When mismatch is material, the report must state the direction and magnitude of residual pool-dependence confounding beside every pool-level oracle opportunity; $\delta$ may not silently absorb it.

## Revised sample allocation and call budget

The sample is revised to:

- 150 `uncovered_0` rows;
- 150 `partial_1_10` rows;
- 200 `common_11` rows.

Total sample size is 500 rows. With 12 calls per row, Arm A requires exactly 6,000 GTA1-7B forwards. Every Amendment 001 reference to 400 rows, 100 common rows, 4,800 calls, or 1,200 common-first outputs is superseded by 500 rows, 200 common rows, 6,000 calls, and 2,400 common-first outputs respectively.

The proportional-largest-remainder application allocation, SHA-256 row ordering, inclusion probabilities, inverse-probability weights, no-replacement rule, bootstrap, and seed base remain unchanged. The future execution amendment must bind exactly 6,000 calls. This amendment does not authorize GPU execution.

## Constant-shift assumption

The equal-shift correction assumes the common-stratum calibration effect is additive and constant across coverage strata. Common targets are selected into high coverage and may differ in size, location, contrast, and semantic salience from low-coverage targets. Centering may therefore have a different effect in uncovered and partial rows. This is a named primary limitation, not a footnote, and must appear in the report body and paper discussion.

Every table, figure, caption, or conclusion that reports corrected $\widetilde A_s$ must show raw $A_s^{pool}$ and its interval immediately adjacent with equal visual prominence. Raw values may not be moved to an appendix. Both remain labeled `GT_ORACLE_NON_DEPLOYABLE`.

### Frozen target-size split

Before any oracle output, use all 931 population `common_11` rows. Compute GT bbox area as $\max(0,x_2-x_1)\max(0,y_2-y_1)$ in full-image pixels. Sort rows by `(area, row_id)`. The first 465 rows are `common_small`; the remaining 466 are `common_large`. This deterministic rank split is normative and prevents median ties from changing membership.

Record the ordinary numeric median area (the 466th sorted area's value), all memberships, counts, area range in each half, and SHA-256 in preflight. The rank split, not a `<= median` predicate, controls membership.

### Heterogeneity endpoint

Using the sampled common rows and frozen inverse-inclusion weights, compute

$$
\delta_{small,B3}=A_{common\_small,B3}^{pool}-B_{common\_small,B3}^{pool},
$$

$$
\delta_{large,B3}=A_{common\_large,B3}^{pool}-B_{common\_large,B3}^{pool},
$$

and $H_\delta=\delta_{small,B3}-\delta_{large,B3}$. Existing B3 half-stratum anchors are reconstructed and hash-locked in preflight before oracle output. Report point values and a joint application-group bootstrap 99% percentile interval with all weighting and calibration recomputed per replicate.

If the interval excludes zero, label `CONSTANT_SHIFT_SIZE_HETEROGENEITY_DETECTED` and strengthen the named limitation. If it includes zero, label `NO_DETECTED_SIZE_HETEROGENEITY_AT_99_PERCENT`; this does not validate constant shift across uncovered/partial strata.

### Small-target sensitivity

In addition to the primary full-common calibration, compute

$$
\widetilde A_{s,B3}^{small}=\operatorname{clip}(A_{s,B3}^{pool}-\delta_{small,B3},0,1)
$$

and the corresponding $U_{perfect}^{small}$. Report its point estimate and joint 99% interval beside the primary $U_{perfect}$. It never affects O-I classification.

The primary O-I interpretation remains based only on Amendment 001's corrected B3 pool-level point estimate using the full common stratum.

## Unchanged provisions

O-I1 below 0.05, O-I2 from 0.05 through 0.10 inclusive, and O-I3 above 0.10 are unchanged. Their estimand remains the corrected B3 pool-level point estimate.

Amendment 001's 12-slot pool structure, B3 primary calibration, M1 secondary calibration, single-forward baseline $B_s^{single}$, pool-minus-single report, window repairs, sampling algorithm, grouped bootstrap with 10,000 replicates and 99% limits, seed base, model/source revisions, prompt, processor, greedy decoding, coordinate tests, trace retention and label separation, backup path, `GT_ORACLE_NON_DEPLOYABLE` label, X2/SPLIT non-conflict, Mind2Web exclusion, post-selection status, and single-benchmark scope remain unchanged except where sample counts and R are explicitly superseded here.

Arm B's layout algorithm, $N^*$ definition, saturation handling, and secondary full-bbox coverage remain unchanged.

If O-I3 triggers, a later round still requires a preregistered net-benefit ledger including harm on all rows that the original method gets correct. The historical `85.77%` quantity is the fraction whose target center has positive existing crop coverage, not a model-success rate; it must be labeled `currently crop-covered`, never `currently successful`. The frozen full B3 success rate is 63.69%. A later ledger must report damage separately on original-correct rows and on crop-covered rows.

## Revised execution order

1. Commit this amendment before any OWIN output or GPU authorization.
2. Implement and commit a geometry-calibration preflight that reports $\mathrm{IoU}^*$, all $\mathrm{IoU}(R)$, selected R, all candidate offsets, endpoint status, the frozen common-area rank split, $B_{common,M1}^{pool}$, B3 size-half anchors, and all $B_s^{single}$ anchors with SHA-256 provenance. It computes no oracle model output or O-I endpoint.
3. Run and commit Arm B.
4. Commit the 500-row sample roster and 12 windows per row under selected R.
5. Commit runner/tests, including selected-R offset reproduction, dependence sufficient statistics, size-split membership, center containment, boundary repair, trace schema, and coordinate transformation.
6. Commit an execution amendment and one-time authorization for exactly 6,000 calls.
7. Execute exactly 6,000 forwards.
8. Parse only 200 common rows times 12 outputs; freeze $\delta_{pool,B3}$, $\delta_{pool,M1}$, $\delta_{single}$, $H_\delta$, and $\delta_{small,B3}$.
9. Parse partial and uncovered outputs without changing any design element.
10. Report stratum-specific oracle and matched-existing dependence diagnostics.
11. Apply O-I1 through O-I3 only to the corrected full-common B3 pool-level point estimate, while reporting raw values, small-target sensitivity, M1, single-forward, and pool-minus-single beside it.