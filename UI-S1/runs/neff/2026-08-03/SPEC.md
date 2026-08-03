# Effective-Sample-Size Law

Date: 2026-08-03

Status: result-blind preregistration for N1, N2, N4 and N5. N3 may use only full-image anchor and coordinate-range diagnostics before this commit; no parser or scale change is permitted when all three anchors are within the frozen tolerance.

Upstream evidence is preserved unchanged under:

- `runs/ccm-h2h/2026-07-31/`
- `runs/allocation-law/2026-08-01/`
- `runs/diversity-axis/2026-08-02/`
- `runs/cala/2026-08-03/`
- `runs/scaleup/2026-08-02/`

## Reframing

Predictions are correlated noisy observations of a latent target location. For an equicorrelated pool,

`N_eff(K, rho) = K / (1 + (K - 1) rho)`.

Candidate quality is a separate factor. Added views can have negligible effective-sample gain while their proposal containment declines; added lineages can reduce correlation while sharing the same proposal geometry.

This run tests the framing. It does not assume that failure kappa is a sufficient rho estimator or that a one-variable collapse must succeed.

## N1 collapse

The primary collapse panel contains unchanged B3 accuracy for all directly reconstructable pool/budget points:

- V-only and Uniform Mixed at N=4/8/12/16;
- Quality-Only and CALA-S at N=4/8/12/16;
- CALA-A at N=8/12/16;
- H1 official B3, paper centroid and graph centroid at N=2/4/10;
- after N3 eligibility, 72B GTA1 N8, Uniform Mixed N8, CALA-S N8 and CALA-A N8.

An all-rules stress panel additionally includes M1 and pass@N where available. Repeated candidate pools with different rules remain separate observations; this intentionally tests whether N_eff alone can explain selector variation.

For each rho estimator, fit ordinary least squares `accuracy = intercept + beta * x`, once with `x=N_eff` and once with `x=K`. Report residual standard deviation `sqrt(SSE/(n-2))`, R-squared and leave-one-application-family-out sensitivity where applicable. Collapse succeeds only when N_eff residual SD is at most 0.014 and strictly below the K residual SD. No nonlinear form is tried.

If every one-factor rho estimator fails, fit `accuracy = intercept + beta_N * N_eff + beta_Q * mean_proposal_full_bbox_containment`. Report residual SD, R-squared and adjusted R-squared. The two-factor fit is explanatory and does not retroactively make the preregistered one-factor law pass.

## N2 upper bound

N2 is claim-eligible only if the primary B3 one-factor collapse passes. Use the winning preregistered rho estimator; ties use failure kappa, then rho_geom, then rho_cond. Evaluate its fitted curve at `1/rho_view`, with `rho_view=0.895`. Compare descriptively to local V-only N16, H1 official B3 N10 and graph centroid N10. Paper-only 62.8 is shown but excluded from pass/fail arithmetic.

If N1 fails, N2 status is `BLOCKED_N1_COLLAPSE` and any extrapolation is clearly diagnostic.

## N3 72B lane

Recompute each model's full-image score from raw point traces and the independent ScreenSpot-Pro label manifest. Audit parse rate, normalized coordinate range and out-of-image count. Anchor tolerance is absolute 0.02.

If all three anchors pass and at least two models have no out-of-image points, classify `PASS_NO_GLOBAL_COORDINATE_BUG`; do not alter parsing, scaling or traces. Reconstruct P1/P2/CALA values for parity. Low B3 relative to pass@N/M1 is then treated as an observed aggregation/candidate-pollution boundary, not a repair target.

If an anchor fails, identify the failing model and permit only a model-card-consistent coordinate correction frozen in a separate amendment before rerunning affected scoring.

72B points enter N1 only after either branch reaches a complete N3 artifact.

## N4 NOA

For a selected set with pairwise development failure-correlation matrix `R`, define generalized

`N_eff(S) = |S|^2 / (1^T R 1)`.

The diagonal is one. Off-diagonal entries are development failure kappas. NOA-static greedily selects the action with largest marginal generalized N_eff; ties use higher individual development accuracy and frozen action order. This is the operational N_eff objective. We do not claim mathematical equivalence to determinant maximization outside equicorrelation.

Compare V-only, Uniform Mixed, Quality-Only, CALA-S and NOA-static with the same five application folds, banks, budgets, unchanged B3/M1 and 10,000 paired grouped bootstraps.

Minimum NOA-static success: N12 B3 is not lower than Uniform Mixed in point estimate. A stronger result requires a positive 99% CI lower bound.

NOA-stop is executed only after N5. Its threshold is selected on development folds from the finite marginal-N_eff values to maximize B3 subject to mean forwards at most eight. Test folds use the frozen fold-local threshold. The strong stop claim requires mean forwards at most eight and B3 within 0.007043345177520599 of Uniform Mixed N12.

## N5 stopping gate

Use the frozen SafeGround Mixed N12 row scores and split rows into five quantiles by disagreement, with `(score,row_id)` tie-breaking. Within each bin report pass@N for Uniform Mixed prefixes N=4/8/12. The gate passes when the highest-disagreement quintile has positive pass@12-minus-pass@4. Report a paired group-bootstrap 99% CI for the increment.

If the point increment is nonpositive, NOA-stop is restricted to a pure-compute-saving claim. No alternate quantiles or disagreement scores are tried.

## Boundaries

- Existing results are never rewritten.
- Paper-only 62.8, 70.4, 73.1 and reported zoom gains do not enter local differences.
- Pass@N is a union metric and is never treated as final accuracy.
- A failed collapse downgrades the law to a qualitative two-factor explanation.
- N6 inference is conditional on NOA-static not underperforming Uniform Mixed at N12.
