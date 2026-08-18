# LOOK Candidate-Confrontation Diagnostic Report

Date: 2026-08-18

Outcome: `LOOK_L_D3_L_K1_NULL_DOMINATES_NO_METHOD_AUTHORIZED`

LOOK is a post-selection, single-benchmark diagnostic, not a method. It changes no prior result and authorizes no method experiment.

## Primary discrimination

| Endpoint | Point | 99% CI | Interpretation |
| --- | ---: | ---: | --- |
| L-P1 main candidate AUROC | 0.540 | [0.458, 0.633] | L_D3 |
| L-P2 main minus M1 correctness | +11.11 pp | [+2.50, +21.57] pp | descriptive recoverable gain |
| L-P4 main minus null AUROC | -0.186 | [-0.323, -0.025] | L-K1 |

Main AUROC is 0.540, close to CEIL's contextual 0.540, and its interval crosses both frozen directional boundaries. LOOK is therefore `L_D3`. Null AUROC is 0.726, exceeding main by 0.186; the paired interval excludes zero against main. L-K1 cancels candidate-identity signal wording regardless of L-P2.

## Damage and sensitivity

On pool-correct rows, confrontation overturns the B3 mode on 52.72%; harmful overturn is 38.29%. Unmappable rate is 0.00%.

Three-mode sensitivity AUROC is 0.603 on recoverable rows and 0.826 on pool-correct rows. It is descriptive and cannot replace the failed primary identity control.

## Separation-stratified results

| Separation quartile | Rows | AUROC | Main minus M1 | Positive / negative records |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 45 | 0.630 | +8.89 pp | 22 / 68 |
| 1 | 45 | 0.516 | +13.33 pp | 17 / 73 |
| 2 | 45 | 0.502 | +6.67 pp | 21 / 69 |
| 3 | 45 | 0.511 | +15.56 pp | 20 / 70 |

Frozen separation boundaries are `[0.026944981581206527, 0.06105051050808982, 0.1480652981453389]`. The first quartile is strongest (AUROC 0.630); later quartiles are near chance. This pattern is descriptive and does not rescue L-K1.

## Geometry

Main-window median area fraction is 0.41%; mean is 4.29%. Only 0.19% exceed 80%, so L-K2 is false. Sensitivity-window median is 1.49%. Null area-ratio median is 1.000; median null search attempt is 236.5.

## Execution and decision

Formal execution completed 1,290/1,290 calls with zero failures; all outputs parsed and all token logprobs were retained. Realized samples were 180 recoverable and 250 pool-correct rows, so L-K3 is false.

Final decision: L-D3 and L-K1. Local confrontation shows a descriptive M1-to-main correctness increase, but the random noncandidate control discriminates substantially better and the pool-correct damage rate is high. No candidate-identity mechanism claim and no follow-up method are authorized.
