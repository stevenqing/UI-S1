# OWIN Amendment 003: dependence diagnostics, degeneracy, and estimand consistency

Round: `owin`

Amendment: `003`

Date: 2026-08-17

Status: `FROZEN_BEFORE_ANY_OWIN_RESULT_OR_GPU_AUTHORIZATION`

Scope: this amendment supersedes Amendment 002 only for undefined-phi filling, degenerate dependence comparators, dependence labels, correlation-matrix notation, and calibration-decomposition checks. Sample size, call budget, radius calibration, O-I thresholds, and the O-I estimand do not change.

## Legality and timing

This amendment was written before any OWIN output. At writing time, the OWIN directory contained only the committed base specification/config and Amendments 001/002 with their configs. No preflight, geometry calibration, Arm B geometry, sample/window roster, model forward, parsed output, or OWIN statistic existed. Arm A was not authorized.

If any OWIN statistic predates this amendment commit, this amendment is void and must be labeled a post-result second attempt.

Amendment 002 corrected two drafting errors: `85.77%` is existing target-center crop coverage, not model success, and radius calibration reads GT bbox geometry and is not label-free. Both corrections are recorded in `docs/research_disclosures.md` in the same result-free commit as this amendment.

## Notation

The jitter radius remains R. Every error-correlation matrix is denoted $\Phi$. Implementations, configs, raw schemas, and reports must use `phi_matrix`, never `R`, for a correlation matrix.

## Dual filling sensitivity

For every original center-based stratum and leave-one-outer-fold-out unit, compute the 55 pairwise Pearson phi values among the 11 crop-slot binary error indicators using Amendment 002's inverse-inclusion-weighted sufficient statistics.

Construct two symmetric 11 by 11 matrices with diagonal one:

- $\Phi^{zero}$: every undefined off-diagonal pair is filled with zero;
- $\Phi^{mean}$: every undefined off-diagonal pair is filled with the arithmetic mean of all valid signed phi values in that unit.

If there is no valid pair, $\Phi^{mean}$ and its effective sample size are undefined. No zero, one, or historical value may be substituted.

For each available matrix compute

$$
N_{eff}^{fill}=\frac{11^2}{\mathbf 1^\top\Phi^{fill}\mathbf 1}.
$$

If the denominator is non-finite or non-positive, that effective sample size is `UNDEFINED_NONPOSITIVE_DENOMINATOR`. Matrices are not projected to the positive-semidefinite cone, correlations are not clipped, and $N_{eff}$ is not clipped to `[1,11]`.

When valid mean phi is nonnegative and both values are finite, report the ordered sensitivity interval $[N_{eff}^{mean},N_{eff}^{zero}]$. When valid mean phi is negative, do not call either value a bound; report the unordered pair and `NEGATIVE_MEAN_PHI_ORDER_NOT_ASSUMED`. In all cases retain both named endpoints separately. Neither is selected as the scientific point estimate.

## Reliability and degenerate units

For each unit report:

- undefined-pair count u out of 55;
- constant-slot count out of 11;
- nonconstant-slot count;
- valid-pair count;
- valid signed-phi mean;
- zero-filled off-diagonal phi mean;
- mean-filled and zero-filled denominators and effective sample sizes when defined.

A slot is constant exactly when its binary error value is identical across all rows with strictly positive analysis weight in that unit. In a bootstrap replicate, analysis weight is inverse-inclusion weight times sampled application multiplicity; zero-multiplicity rows do not affect constancy. This definition controls all constant-slot, undefined-pair, and degeneracy paths.

If $u>11$, label the unit `DEPENDENCE_DIAGNOSTIC_UNRELIABLE`. It receives no match/mismatch label and enters no cross-stratum or pooled dependence summary. Its raw diagnostics and both available sensitivity endpoints remain reported.

For the matched existing-GTA1 comparator, if all 11 slots are constant, label `UNDEFINED_DEGENERATE_COMPARATOR`. Do not emit the artificial zero-fill value 11, compute a relative difference, or assign a match label. If not all slots are constant but no pair is valid, label `NO_VALID_PAIR_COMPARATOR`; it is also ineligible for relative differences and labels.

Oracle diagnostics remain reportable when the comparator is degenerate, but oracle-versus-comparator differences are `NOT_APPLICABLE`.

For each stratum and fold also report the comparator's nonconstant-slot count and the number/fraction of sampled rows whose GT bbox has positive-area intersection with at least one existing GTA1 crop. Positive-area intersection requires `max(0,min(r,x2)-max(l,x1))*max(0,min(b,y2)-max(t,y1)) > 0`. This is descriptive and does not redefine COVER strata.

## Interval-based dependence labels

Only units passing reliability and having finite oracle and comparator endpoints are eligible. Compare like filling with like filling:

$$
D^{fill}=\frac{|N_{eff,oracle}^{fill}-N_{eff,existing}^{fill}|}{N_{eff,existing}^{fill}}.
$$

The comparator denominator must be finite and strictly positive. Zero-fill is compared only with zero-fill; mean-fill only with mean-fill.

For each eligible unit and fill rule, run 10,000 application-group bootstrap replicates. In every replicate, resample applications, recompute inverse-inclusion-weighted phi sufficient statistics for both pools, re-identify undefined pairs, refill $\Phi$, and recompute D. A replicate is reliable only if both sides satisfy $u\le11$, all required denominators are positive and finite, and mean-fill has at least one valid pair. At least 9,900 reliable finite replicates are required for a percentile 99% interval. Otherwise label that fill `DEPENDENCE_BOOTSTRAP_UNRELIABLE` and do not infer matching.

For each fill with a valid interval:

- lower bound strictly above 0.10: `MATERIAL_DEPENDENCE_MISMATCH`;
- upper bound strictly below 0.10: `APPROXIMATELY_MATCHED`;
- otherwise: `DEPENDENCE_MATCH_INDETERMINATE`.

If zero-fill and mean-fill labels differ, the unit-level combined label uses this precedence: `MATERIAL_DEPENDENCE_MISMATCH`, then `DEPENDENCE_MATCH_INDETERMINATE`, then `APPROXIMATELY_MATCHED`. An unavailable/unreliable fill cannot be converted to approximate matching; if the other fill is mismatch, combined is mismatch, otherwise combined is indeterminate.

Labels are issued per stratum and leave-one-fold-out unit. Do not pool unreliable units or strata into a single label. Stratum summaries report counts of each unit label plus fold values/ranges for reliable units only.

When mismatch is material, every pool-level oracle opportunity must state the direction and magnitude of residual pool-dependence confounding. This remains descriptive and never changes O-I classification.

## Calibration decomposition consistency

Let $w=465/931$. After freezing full-common, small-common, and large-common B3 pool calibration effects, compute

$$
\Delta_{decomp}=\delta_{pool,B3}-[w\delta_{small,B3}+(1-w)\delta_{large,B3}].
$$

Report its point estimate and joint application-group bootstrap 99% interval. Also report the achieved weighted sample share

$$
\hat w=\frac{\sum_{i\in common\_small}IPW_i}{\sum_{i\in common}IPW_i}
$$

and raw sampled-row share. Every bootstrap replicate recomputes all ratio estimators with the same application multiplicities.

The implementation tolerance is $|\Delta_{decomp}|\le0.005$ at the point estimate. If exceeded, label `CALIBRATION_DECOMPOSITION_CHECK_FAILED` and block reporting or interpreting any corrected OWIN value until the weighting implementation is audited. Preserve the failed output and document the audit in `REPORT.md`; do not tune weights, split membership, or tolerance. Passing this check is implementation consistency, not scientific evidence for constant shift.

The geometry-calibration preflight freezes w and the 0.005 tolerance, but cannot compute $\Delta_{decomp}$ before oracle outputs.

## Unchanged provisions

O-I1 below 0.05, O-I2 from 0.05 through 0.10 inclusive, and O-I3 above 0.10 remain unchanged. Their only estimand is the corrected B3 pool-level point estimate using the full-common calibration.

The sample remains 150 uncovered, 150 partial, and 200 common rows, totaling 500 rows and exactly 6,000 forwards. GPU remains unauthorized.

Radius grid/calibration, 12-slot pool, B3 primary and M1 secondary calibration, single-forward and pool-minus-single reports, target-size heterogeneity, small-target sensitivity, window repair, grouped bootstrap, model/trace/retention discipline, Arm B, full-bbox secondary coverage, and all scope/non-conflict statements remain unchanged.

Any O-I3 follow-up still requires a preregistered net-benefit ledger with damage reported separately on original-correct and crop-covered rows.

## Revised execution additions

The preflight must freeze $w=465/931$ and tolerance 0.005. Runner tests must cover both fill matrices, undefined-pair and constant-slot counts, negative-mean ordering, nonpositive denominators, reliability and degeneracy statuses, bootstrap finite-replicate threshold, label precedence, and `phi_matrix` naming.

Immediately after common calibration effects are frozen, compute $\Delta_{decomp}$. A failed consistency check blocks corrected-value reporting until a retained audit is complete. Dependence reporting uses the dual-fill sensitivity endpoints and interval labels above.