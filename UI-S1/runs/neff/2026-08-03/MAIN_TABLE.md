# Effective-Sample-Size Evidence Table

## New adjudications

| Test | Result | Criterion | Status |
|---|---|---|---|
| N1 one-factor collapse | best failure_kappa residual SD 7.30 pp; R2 0.324 | residual SD <= 1.40 pp and better than K | FAIL |
| N1 two-factor diagnostic | best rho_cond residual SD 5.41 pp; adjusted R2 0.616 | explanatory only | INSUFFICIENT |
| N2 single-model upper bound | N1 did not pass | requires N1 collapse | BLOCKED |
| N3 72B coordinates | all three bare anchors within 2 pp; two models have zero out-of-image points | no repair if anchors pass | PASS_NO_BUG |
| N4 NOA-static N12 B3 | 62.24% vs 63.69%; -1.45 pp; CI [-3.03, +0.06] | point estimate not lower | FAIL |
| N4 NOA-stop | 61.10% at mean 6.19 forwards vs 63.69% N12 | <=8 forwards and within 0.70 pp | FAIL |
| N5 stopping gate | high-disagreement pass@4 38.29% to pass@12 51.27%; +12.97 pp; CI [+8.33, +18.12] | positive point increment | PASS |

## Preserved upstream evidence

| Evidence | Value | Role after N1 |
|---|---|---|
| H3 equal-compute mixed pool | M1 60.40% to 63.82%; +3.42 pp, 99% CI [+1.41,+5.67] | primary empirical allocation result |
| H2 collision floor | view 0.895; cross-family 0.398; same-family scale 0.618 | external correlation evidence, not universal ScreenSpot rho |
| ScreenSpot pool rho | V-only N12 failure-kappa 0.689; Uniform N12 0.594 | pool-specific correlation measurement |
| X3 budget slopes | V-only -0.002467; Mixed +0.003052 with separated 99% CIs | robust sign-flip result |
| L4 proposal quality | full-bbox containment 99.94% at rank 0 to 61.04% at rank 11 | quality-decay factor |
| CALA-S N12 | pass@12 80.01%, B3 62.18% vs Uniform 63.69% | coverage/final-accuracy separation |
| CALA 72B N8 | CALA-S B3 45.41% vs Uniform 41.24% | equal-budget transfer, not absolute SOTA |
| X7 SafeGround | correctness AUROC 0.628 / 0.744 / 0.830 | disagreement remains useful |
| X6 ranking | held-out Spearman 0.903 | unlabeled pool ranking evidence |
