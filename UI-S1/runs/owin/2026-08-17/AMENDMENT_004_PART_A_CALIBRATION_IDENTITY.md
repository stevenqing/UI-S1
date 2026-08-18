# OWIN Amendment 004 Part A: calibration identity revision

Round: `owin`

Amendment: `004A`

Date: 2026-08-17

Status: `FROZEN_BEFORE_ANY_OWIN_RESULT_OR_GPU_AUTHORIZATION`

Scope: Part A supersedes only Amendment 003's blocking calibration-decomposition check and adds wording for unavailable dependence diagnostics. Part B execution authorization is not included in this commit and remains prohibited.

## Legality and timing

This amendment was written before any OWIN output. At writing time, the OWIN directory contained only the committed base specification/config and Amendments 001 through 003 with their configs. No preflight, geometry calibration, Arm B geometry, sample/window roster, model forward, parsed output, or OWIN statistic existed. Arm A was not authorized.

If any OWIN statistic predates this commit, Part A is void and must be labeled a post-result second attempt.

## Problem with the prior blocking check

Amendment 003 defined

$$
\Delta_{decomp}=\delta_{pool,B3}-[w\delta_{small,B3}+(1-w)\delta_{large,B3}],\qquad w=465/931,
$$

and blocked corrected reporting when $|\Delta_{decomp}|>0.005$.

This quantity need not be zero under a correct implementation. The oracle side is a ratio IPW estimate over 200 sampled common rows and decomposes using achieved IPW share $\hat w$. The existing B3 side is an exact 931-row population quantity and decomposes using population share w. Sampling variation in $\hat w$ therefore creates a nonzero descriptive residual. The 0.005 rule can reject a correct implementation and is withdrawn as a gate before any OWIN result.

## Blocking implementation identity

Use only the same sampled common rows and the same inverse-inclusion analysis weights on both sides.

For outcome $Y\in\{A,B\}$, define ratio IPW estimates

$$
\hat Y_g=\frac{\sum_{i\in g}IPW_iY_i}{\sum_{i\in g}IPW_i},
$$

where A is oracle-pool B3 correctness and B is frozen existing-pool B3 correctness on that sampled row. Let

$$
\hat w=\frac{\sum_{i\in common\_small}IPW_i}{\sum_{i\in common}IPW_i}.
$$

The blocking identity is

$$
\Delta_{ident}=(\hat A_{common}-\hat B_{common})-
[\hat w(\hat A_{small}-\hat B_{small})+(1-\hat w)(\hat A_{large}-\hat B_{large})].
$$

The small and large sets are Amendment 002's frozen rank partition and form an exact disjoint partition of common. All denominators must be positive and finite. Under a correct implementation the identity is zero up to floating-point summation.

The frozen tolerance is

$$
|\Delta_{ident}|\le10^{-9}.
$$

Evaluate the point identity and every joint bootstrap replicate. Each replicate uses the same application multiplicities for all six ratio estimates and $\hat w$. A replicate with a zero subgroup denominator is non-finite and retained; it does not silently pass. The point check alone controls blocking. Bootstrap identity values are implementation diagnostics and report their maximum absolute finite residual plus non-finite replicate count; they are not a scientific interval.

If the point check fails, label `CALIBRATION_IDENTITY_CHECK_FAILED`, preserve all outputs, and block every corrected OWIN value from reporting or interpretation until an audit is retained in `REPORT.md`. Weights, split membership, tolerance, and estimands may not be changed. Passing is implementation consistency only.

## Nonblocking representativeness diagnostic

Retain Amendment 003's population-share quantity $\Delta_{decomp}$, its point estimate, and joint application-group bootstrap 99% interval. It is now named `CALIBRATION_REPRESENTATIVENESS_DIAGNOSTIC` and never blocks reporting.

Report beside it:

- population share $w=465/931$;
- achieved IPW share $\hat w$;
- raw sampled-row small share;
- $\hat w-w$;
- $\delta_{small,B3}-\delta_{large,B3}$;
- the requested descriptive product $(\hat w-w)(\delta_{small,B3}-\delta_{large,B3})$;
- the exact oracle-side composition term $(\hat w-w)(\hat A_{small}-\hat A_{large})$;
- the B3-anchor remainder $(\hat w-w)(B_{small}^{population}-B_{large}^{population})$.

The requested product is a descriptive dominant-term diagnostic, not an algebraic identity for $\Delta_{decomp}$. The exact composition term and anchor remainder are reported so that the discrepancy is transparent.

The historical value 0.005 remains a reference scale only. Report whether $|\Delta_{decomp}|$ exceeds 0.005, but attach no pass/fail status and change no threshold or interpretation.

## Unavailable dependence diagnostics

If any stratum or all strata receive `DEPENDENCE_DIAGNOSTIC_UNRELIABLE`, `UNDEFINED_DEGENERATE_COMPARATOR`, `NO_VALID_PAIR_COMPARATOR`, or `DEPENDENCE_BOOTSTRAP_UNRELIABLE`, every table, figure, caption, conclusion, and paragraph containing a pool-level oracle opportunity must state:

> Residual pool-dependence comparability could not be quantified for the affected stratum(s).

This is a named limitation beside the constant-shift limitation. Historical $N_{eff}=1.5726$ cannot substitute for unavailable matched diagnostics. Reports must list affected strata/folds and exact statuses.

## Part B is deferred and unauthorized

The proposed Part B cannot be committed now. Its prerequisites do not exist: Part A must first be committed, then the geometry-calibration preflight, Arm B outputs, 500-row/12-window manifest, runner, and tests must each be committed and pushed.

No file containing `TO_BE_BOUND`, no placeholder nonce, and no partially populated authorization is an execution amendment. GPU remains unauthorized. A future Part B must bind every required hash, hardware/backend field, shard boundary, H1 prompt/processor/decoding field, parser, logprob behavior, and one-time nonce from committed artifacts. Its commit, only after prerequisite verification, may constitute authorization.

## Unchanged provisions

O-I thresholds and primary estimand, sample 150/150/200, 500 rows, exactly 6,000 formal calls, radius calibration, dependence dual filling and interval labels, target-size heterogeneity, small-target sensitivity, raw/corrected equal prominence, Arm B, trace retention, `GT_ORACLE_NON_DEPLOYABLE`, and all scope boundaries remain unchanged.