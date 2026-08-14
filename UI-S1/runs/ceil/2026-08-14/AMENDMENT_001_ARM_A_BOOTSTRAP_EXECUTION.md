# CEIL Amendment 001: Arm A bootstrap execution

Date: 2026-08-14
Timing: after Arm B result commit `df7f650`, before Arm A implementation and before any Arm A statistic is computed.

## Frozen execution semantics

1. The five outer-fold aggregator functions are fit once under their original cross-fitting protocol. Bootstrap replicates resample held-out groups and retain those cross-fitted prediction functions, matching the project's existing paired-bootstrap convention. Each replicate recomputes held-out subset accuracy, development-fold failure-kappa matrices, all 4,095 subset $N_{\mathrm{eff}}$ values, isotonic fits, and parametric fits from the selected group sufficient statistics. It does not retrain source priority or another aggregator inside the bootstrap.
2. Each panel×aggregator uses its preregistered independent seed. A selected group carries all of its rows and all 12 source outcomes.
3. The 99% CIs are reported for $\Delta_\infty$, full-pool $N_{\mathrm{eff}}$, observed support maximum, finite-$x_1$ isotonic upper-bound gain, and finite-$x_1$ parametric gain.
4. If the bounded parametric solver fails in at most 1% of replicates, percentile intervals use successful replicates and report the exact failure count. If it fails in more than 1%, all parametric CIs for that panel×aggregator are `NA_BOOTSTRAP_FIT_FAILURE`; no alternate fit is used.
5. SSPro candidate success and aggregation use the frozen MASK native definitions. M2W candidate success uses E1 `score_prediction`; density and majority use the frozen E1 implementations on each ordered subset. No result-dependent cache, binning, or row deletion is allowed.

This amendment changes no Arm B quantity and introduces no new curve family, estimator, threshold, or conclusion branch.
