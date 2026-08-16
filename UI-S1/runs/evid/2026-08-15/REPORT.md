# EVID Source-Aware Effective-Evidence Aggregation Report

Date: 2026-08-15

Outcome: `EVID_FIXED_AGGREGATOR_FAILED_STAGE2_BLOCKED`

EVID is a zero-GPU, single-benchmark post-selection validation. The fixed rho values are AndroidControl failure-kappa heuristics, not validated ScreenSpot-Pro intraclass correlations. No existing project status changes.

## Stage 0

The rho-zero control reproduced canonical B3 row by row with zero mismatches. The fixed scorer selected a different block on 111/1,581 rows (**7.02%**), so E-G2 passed.

The fixed-output block oracle reaches **78.56%**, or **+14.74 pp** over nested dev-selection. Contains-any-correct coverage is 79.19%; the difference is block-output loss and is not counted as attainable oracle accuracy. E-G1 passed.

The separated $2\to3$ lineage marginal is **+0.855 pp** for density B3, 99% CI **[+0.502,+1.216]**, but only **+0.317 pp** for F1 majority, CI **[-0.123,+0.705]**. The conservative minimum is below 0.70 pp, so E-G3 failed and Stage 2 was permanently blocked before any GPU request.

## Stage 1

| Variant | Accuracy | vs dev-selection | 99% CI | vs A4 | 99% CI | vs B3 | 99% CI |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| exact_singleton | 59.84% | -3.985 pp | [-5.970,-2.064] | -4.111 pp | [-6.005,-2.166] | -3.858 pp | [-5.829,-1.893] |
| fixed | 63.06% | -0.759 pp | [-1.619,+0.000] | -0.886 pp | [-1.855,+0.000] | -0.633 pp | [-1.447,+0.117] |
| rho_fitted | 63.50% | -0.316 pp | [-0.775,+0.190] | -0.443 pp | [-1.023,+0.126] | -0.190 pp | [-0.577,+0.249] |
| weighted | 63.06% | -0.759 pp | [-1.767,+0.200] | -0.886 pp | [-2.027,+0.200] | -0.633 pp | [-1.696,+0.353] |

The fixed primary reaches **63.06%**. Relative to nested dev-selection it is **-0.759 pp**, 99% CI **[-1.619,+0.000]**. E-K1 triggers and the parameter-fixed theoretical variant fails.

The lineage-weighted variant is identical in aggregate to fixed EVID. The fitted-rho variant reaches 63.50% but remains below dev-selection, and 3/5 folds select a rho-grid endpoint. E-K5 triggers; the grid is not expanded and the fitted result cannot replace the primary failure.

The exact-singleton control equals the 59.84% source-priority majority/best-single endpoint and is substantially worse than finite EVID. E-K3 does not trigger because finite EVID itself is not positively distinguishable from dev-selection.

The diagonal additive-to-average path has Spearman correlation 0.719, below the frozen 0.8 criterion. E-K4 triggers and the path-unification narrative is deleted. The endpoint jump is retained as sensitivity behavior, not evidence of a smooth majority transition.

## Interpretation

Stage 0 shows genuine block-selection headroom, but the frozen source-aware heuristic does not identify it. Discounting repeated same-lineage votes with AndroidControl-derived kappa anchors lowers accuracy below B3, A4, and nested dev-selection. This closes the fixed EVID score family on the current bank; neither fitted weights nor fitted rho rescues it under the preregistered rules.

Stage 2's proposed six-lineage equal-budget reallocation is not authorized because E-G3 failed before Stage 1 and Stage 1 was negative. No GPU forward was run.

## Boundaries

The result is ScreenSpot-Pro-only and post-selection. Mind2Web remains `BLOCKED_ALIGNED_POOL_UNAVAILABLE`. EVID changes none of F1, Q1, `TRIVUS_NOT_PROMOTED`, VUS-SR, SPLIT, MASK, CEIL, OTEXT, XSCR, DECOMP, or XSOFT.
