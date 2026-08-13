# GRAN Aggregation-Granularity Report

Date: 2026-08-14

Status: `GRAN_PRIMARY_SUPPORTED_BUT_UNIFICATION_FAILED_G_K3_AND_GRID_FAILED_G_K6`

## Scope

GRAN is a zero-GPU explanatory analysis over frozen candidate banks. It is not a method and does not authorize a GT-free $\tau$ selector. The label-dependent quantities $\hat p$, $\hat q_{\max}$, contamination, and separation are evaluation-side variables only.

The evidence chain is ordered as preregistration, input manifest, CLICK-scope lock, implementation anchors, sweep semantics, sweep implementation, raw sweep, adjudicator implementation, and adjudication.

## Primary result

G-P2 passes. Within the 1,774-row Mind2Web CLICK subset, the highest preregistered $\hat p$ stratum contains 443 rows and has density-minus-prior margin **+11.74 percentage points**, with grouped-bootstrap 99% CI **[+7.29, +16.51]**.

The four Mind2Web strata are:

| Stratum | Rows | $\hat p$ range | Margin |
| --- | ---: | --- | ---: |
| 0 | 444 | [0.000, 0.000] | +0.00 pp |
| 1 | 444 | [0.000, 0.091] | -2.03 pp |
| 2 | 443 | [0.091, 0.364] | +0.68 pp |
| 3 | 443 | [0.364, 1.000] | **+11.74 pp** |

This supports candidate-pool correctness as an explanatory variable inside Mind2Web CLICK. It does not establish a shared benchmark-invariant curve.

## Secondary results

G-P1 passes on ScreenSpot-Pro. The row-level Spearman association between $\hat p$ and density-minus-prior margin is $\rho=0.0661$, with 99% CI `[0.0307, 0.0984]`.

The ScreenSpot-Pro strata are not monotone in mean margin:

| Stratum | Rows | $\hat p$ range | Margin |
| --- | ---: | --- | ---: |
| 0 | 396 | [0.000, 0.056] | -0.76 pp |
| 1 | 395 | [0.056, 0.500] | -3.04 pp |
| 2 | 395 | [0.500, 0.944] | +9.62 pp |
| 3 | 395 | [0.944, 1.000] | +0.51 pp |

G-P3 fails. The two benchmark curves differ significantly in strata 2 and 3, with opposite signed cross-benchmark differences. A common $\hat p$ coordinate does not collapse the curves.

G-P6 fails and triggers G-K3. Under the preregistered prior $\pi$:

- Mind2Web exact-minus-single is -3.78 pp, 99% CI `[-5.24, -2.51]`, far outside its ±0.61 pp MDE.
- ScreenSpot-Pro exact-minus-single is -0.38 pp, but its 99% CI `[-2.75, +2.16]` is not contained inside ±0.70 pp.

The proposed “same process, two convergent endpoints” CEV narrative is therefore invalid. Exact coordinate ties and Mind2Web type-first partitioning prevent the assumed endpoint collapse.

G-P4, G-P5, G-P7, and G-P8 are `NOT_ADJUDICABLE_PREREG_UNDERDEFINED`. Their required numerical operationalizations were not frozen before the sweep; no post-hoc threshold or formula is introduced.

## Sweep result and kill conditions

ScreenSpot-Pro density accuracy is 61.42% versus prior 59.84%, point margin +1.58 pp. All selected finite $\tau$ values are internal grid points.

Mind2Web C-uni CLICK density accuracy is 35.96% versus prior 33.37%, point margin +2.59 pp. Three folds select the single-block endpoint, one selects `0.0022908677`, and one selects the finite upper boundary `0.5011872336`. The boundary selection triggers G-K6. The grid is therefore recorded as failed; it is not expanded and the boundary is not presented as a valid optimum.

Triggered conditions:

- G-K3: endpoint convergence failed; GRAN cannot support the CEV unification narrative.
- G-K6: a Mind2Web fold selected the finite grid boundary.

Not triggered:

- G-K1: G-P1 passes.
- G-K2: primary G-P2 passes.
- G-K5: all fixed strata meet their benchmark-specific minimum counts.

G-K4 is unadjudicated because G-P7 was preregistered without a common 36-action A1/A2/B3 output definition.

## Conclusion

The strongest defensible conclusion is narrow:

> Candidate-pool correctness strongly stratifies density-versus-prior performance within Mind2Web CLICK, but $\hat p$ alone does not unify Mind2Web and ScreenSpot-Pro. The aggregation-granularity axis is not a valid two-endpoint CEV unification, and the Mind2Web finite $\tau$ grid is inadequate.

This round does not change `TRIVUS_NOT_PROMOTED`, VUS-SR status, or any runtime method.