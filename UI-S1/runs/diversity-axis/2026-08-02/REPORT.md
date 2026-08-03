# Diversity-Axis Results

Date: 2026-08-02

## Executive result

X3 validates the original budget-axis sign with 10,000 fold-stratified application bootstraps: V-only M1 slope is -0.002467 per forward with 99% CI [-0.004908, -0.000124], while Mixed is 0.003052 with CI [0.001082, 0.005053]. X-K2 does not trigger.

X2 uses a preregistered fixed-12, K=3 microchain extension of UI-Zoomer. Q1-Q4 M1 accuracies are 60.40%, 51.23%, 63.82%, and 61.61%. Q4 is not the highest cell, so the preregistered composability criterion fails. Adaptive-minus-fixed is -9.17 pp for the single lineage and -2.21 pp for the mixed pool. The interaction is positive, +6.96 pp, 99% CI [+3.02 pp, +10.53 pp], because mixed allocation attenuates adaptive harm; it is not evidence that adaptive zoom adds accuracy. X-K1 is `False` because that kill condition tests only a significantly negative interaction.

X1 remains blocked because only five GTA1 stochastic samples exist per row, not the required N=4/8/12/16 three-pool traces. X4 has no released GMS implementation or auditable fixed-12 trace. X5 lacks pure-serial and hybrid traces. X8 cannot construct a same-lineage N=6 Mind2Web pool. These are unavailable comparisons, not negative results.

## X1: sampling axis

Status: `BLOCKED_INSUFFICIENT_SAMPLES`. The exact GUI-RC voting port gives N4 S-only accuracy 48.89%, B3 49.34%, and pass@4 51.17%. X-K3 is `NOT_EVALUATED` because a slope cannot be estimated without padding or new sampling traces.

## X2: adaptive zoom composability

| Cell | B3 | M1 | pass@12 | failure kappa |
|---|---:|---:|---:|---:|
| Q1 single/fixed | 60.09% | 60.40% | 72.80% | 0.689 |
| Q2 single/adaptive | 50.73% | 51.23% | 59.46% | 0.908 |
| Q3 mixed/fixed | 63.69% | 63.82% | 79.19% | 0.594 |
| Q4 mixed/adaptive | 56.29% | 61.61% | 76.79% | 0.415 |

Adaptive trigger rates are 26.14% for Q2 and 63.90% for Q4. Every row uses exactly 12 useful forwards. This is an algorithm-level K=3 budget-normalized extension; the official Qwen2.5-VL-7B K=8 sanity anchor was not run.

## X3: curve robustness

Both B3 and M1 satisfy V-only CI upper bound below zero and Mixed CI lower bound above zero. N24 remains one-sided because GTA1 provides only 16-19 unique candidates. Area stratification contradicts the small-target expectation at N12: the smallest target quintile has M1 Mixed-minus-V-only -2.52 pp, while the largest has +8.23 pp.

## X4-X6

- X4: `UNAVAILABLE_NO_RELEASED_IMPLEMENTATION_OR_FIXED_BUDGET_TRACE`; X-K4 is `NOT_EVALUATED`.
- X5: `BLOCKED_INCOMPLETE_TOPOLOGY_TRIANGLE`; the frozen pure-parallel point exists, but X2 Q4 has three parallel global samples plus one conditional branch per lineage and is not the preregistered four-step serial hybrid.
- X6: frozen L2 OLS training R-squared is 0.0145. Held-out Spearman over 10 X2 fold-pool observations is 0.903 (p=0.0003436); criterion rho > 0.7 is `True`.

## X7: confidence axis

SafeGround official-code geometry is exactly ported at commit `5e8fca7`. Correctness AUROC from negative uncertainty is 0.628 for stochastic GTA1 N4, 0.744 for deterministic V-only N12, and 0.830 for deterministic Mixed N12, versus the cross-task S_gap anchor 0.393. The deterministic N12 diagnostics transfer the dispersion score but do not inherit SafeGround's K=10 stochastic protocol or FDR guarantee.

## X8: Mind2Web alternative

Status: `BLOCKED_NO_SAME_LINEAGE_N6`. Six deployable full-view models are available across families {'CogAgent': 1, 'TongUI': 3, 'UI-TARS': 2}; the largest same-family pool has only 3 models. Original L3 and L-K4 remain unchanged.

## Claim boundary

The strongest defensible claim is that the fixed-view budget-axis sign is statistically stable and that candidate dispersion is a useful confidence diagnostic, especially in the mixed pool. X2 does not establish adaptive-zoom composability: both adaptive cells underperform their fixed counterparts and Q4 is not highest. Its positive interaction means cross-lineage allocation reduces the harm of this K=3 adaptive extension. Sampling-family coverage, GMS comparison, topology triangle, and Mind2Web lineage transfer remain unresolved because their required candidate pools do not exist under the frozen contracts.
