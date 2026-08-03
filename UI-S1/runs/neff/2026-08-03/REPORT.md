# Effective-Sample-Size Law Report

Date: 2026-08-03

## Executive result

The strong Effective-Sample-Size Law is **not established**. None of the three preregistered rho estimators collapses B3 pool accuracy to the required 1.40 pp residual scale. The best one-factor result uses `failure_kappa` with residual SD 7.30 pp and R-squared 0.324. Raw K is worse, but N_eff remains far from sufficient.

Adding proposal quality improves adjusted R-squared to 0.616 for `rho_cond`, yet residual SD remains 5.41 pp. The framework therefore survives only as a qualitative two-factor explanation: correlation and candidate quality both matter, but they do not define a universal accuracy law.

## Why the strong law fails

Pool-specific rho does move in the expected direction. At 7B N12, V-only failure-kappa is 0.689 and Uniform Mixed is 0.594; corresponding equicorrelation N_eff values are 1.40 and 1.59. But CALA-S N12 reduces rho further and raises N_eff to 1.69 while B3 falls. Correlation reduction can buy oracle coverage without making an unchanged mode-like aggregator choose the correct cluster.

The external H2 view-axis value 0.895 is not the ScreenSpot pool rho: ScreenSpot V-only N12 measures 0.689. It remains evidence that repeated views are highly correlated across prior tasks, not a constant to substitute into every pool.

## N2 upper bound

N2 is `BLOCKED_N1_COLLAPSE`. The failure-kappa fit would diagnostically extrapolate 65.65% at 1/0.895, but its residual SD is 7.30 pp and the fit slope is negative. No impossibility upper-bound claim is made.

## 72B audit

N3 rejects the proposed global coordinate-bug diagnosis. Local full-image scores are GTA1 58.51%, UI-Venus 60.53%, and Qwen3.5 71.41%; all pass their paper-anchor tolerance. Existing 72B traces are retained without parser changes. Their low B3 is an aggregation/candidate-pollution boundary.

## NOA

NOA-static does not repair CALA-S. At N12, B3 is 62.24% versus Uniform Mixed 63.69%: -1.45 pp, 99% CI [-3.03, +0.06] pp.

NOA-stop uses an average 6.19 forwards (median 5) but reaches 61.10%, -2.59 pp below Uniform N12 with 99% CI [-3.70, -1.40] pp. It saves compute but fails the frozen equal-accuracy tolerance, so no efficiency success is claimed.

N5 confirms that stopping was not doomed by absent headroom: in the highest SafeGround-disagreement quintile, pass@N rises from 38.29% at N4 to 51.27% at N12, +12.97 pp, 99% CI [+8.33, +18.12] pp. The failure lies in allocation/stopping realization, not a flat rescue curve.

## Consolidated contribution

The defensible paper contribution remains empirical and diagnostic:

1. Under equal 12-forward compute, cross-lineage allocation improves 7B grounding by 3.42 pp with a positive 99% CI.
2. Pool error correlation and proposal quality explain why repeated-view scaling saturates or reverses direction, but not through a universal one-dimensional N_eff curve.
3. Candidate union coverage is not final accuracy: both CALA-S and NOA can improve headroom or N_eff while hurting B3.
4. Low-budget CALA N8 gains transfer across 7B and 72B, but neither the 72B absolute-SOTA lane nor the generalized NOA objective succeeds.

## Execution boundary

N6 is not run because NOA-static underperforms Uniform Mixed at N12. Existing collision-law, allocation-law, diversity-axis, CALA and Scale-Up artifacts remain unchanged. Paper-only 62.8, 70.4 and 73.1 are context only.
