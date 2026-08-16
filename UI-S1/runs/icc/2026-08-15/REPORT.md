# ICC EVID Premise Audit Report

Date: 2026-08-16

Outcome: `ICC_CLOSE_CORRELATION_DISCOUNT_DIRECTION_A2_TOTAL_APPROX_SUPPORTED`

ICC is a zero-GPU post-selection diagnostic. It changes no prior result or historical status and makes no method claim.

## Disclosure

EVID's fixed constants were AndroidControl failure kappas, not ScreenSpot-Pro error correlations. EVID therefore rejected its frozen transferred-constant variant rather than every possible direct-parameter version of the score family. EVID remains failed and Stage 2 remains blocked. This process error is recorded in `docs/research_disclosures.md`.

## Arm A: fitted rho endpoints

| Fold | Selected rho_v | Selected rho_l | Accuracy | Class |
| ---: | ---: | ---: | ---: | --- |
| 0 | 0.0 | 0.0 | 75.24% | low_endpoint |
| 1 | 0.4 | 0.6 | 65.28% | interior |
| 2 | 0.0 | 0.0 | 57.19% | low_endpoint |
| 3 | 0.4 | 0.0 | 55.27% | low_endpoint |
| 4 | 0.2 | 0.2 | 67.83% | interior |

Three of five folds select a low endpoint; two select $(0,0)$ and one selects $\rho_\ell=0$. No fold selects a high endpoint. Neighbor deltas are nonpositive around every selected cell. The fitted data therefore prefer little or no discount rather than stronger discount.

## Arm C: direct ScreenSpot-Pro dependence

| Stratum | Phi fold mean | Fold range | AndroidControl reference | Difference |
| --- | ---: | ---: | ---: | ---: |
| within_lineage | 0.672 | [0.655,0.687] | 0.895 | -0.223 |
| cross_lineage | 0.577 | [0.555,0.600] | 0.398 | +0.179 |

Direct ScreenSpot-Pro within-lineage phi is **0.672**, -0.223 from the transferred 0.895. Cross-lineage phi is **0.577**, +0.179 from 0.398. EVID simultaneously over-discounted within-lineage repeats and under-discounted cross-lineage dependence; the premise error is not a one-direction rescaling.

The empirical phi-matrix $N_{\mathrm{eff}}$ is **1.5726**. MASK's empirical kappa-matrix value is **1.5937**, relative error 1.34%. The exchangeable two-level phi formula is 1.5726.

The structured formula equals the empirical phi-matrix value here by pair-count algebra: 18 within-lineage and 48 cross-lineage pairs with equal weighting exactly reconstruct $\mathbf1^TR\mathbf1$. It is not independent corroboration. The nontrivial diagnostic is kappa versus phi, which passes the frozen 10% tolerance. ICC therefore records retrospective A2 total-count approximation support, while GRAN G-P8 remains historically `NOT_ADJUDICABLE_PREREG_UNDERDEFINED`.

## Arm B: destination of changed rows

The fixed scorer changes 111/1,581 rows. It corrects 14 and harms 24, for a net **-10 rows (-0.633 pp)**; 22 remain correct and 51 remain wrong.

| Direction | Rows | Wrong-to-correct | Correct-to-wrong | Net | Correction/all |
| --- | ---: | ---: | ---: | ---: | ---: |
| composition_same | 0 | 0 | 0 | +0 | NA |
| diversity_decrease | 0 | 0 | 0 | +0 | NA |
| diversity_increase | 104 | 13 | 23 | -10 | 12.50% |
| lineage_substitution | 0 | 0 | 0 | +0 | NA |
| same_L_concentration_decrease | 7 | 1 | 1 | +0 | 14.29% |
| same_L_concentration_increase | 0 | 0 | 0 | +0 | NA |

104/111 changes increase represented lineage diversity. Those changes correct 13 rows and harm 23, net -10. No `same_L_concentration_increase` row exists, so there is no observed concentration-increase correction rate to compare; it is reported as unavailable rather than zero. On ScreenSpot-Pro, choosing the more lineage-diverse block is not a better correctness indicator than canonical count selection.

## Same-budget lineage audit

| Omitted lineage | Method | Full 3x4 minus 2x6 | 99% CI |
| --- | --- | ---: | ---: |
| GTA1-7B | B3_mvp | +4.428 pp | [+2.368,+6.657] |
| GTA1-7B | M1_ccm | +3.416 pp | [+1.459,+5.365] |
| GTA1-7B | source_priority | +3.605 pp | [+1.262,+6.017] |
| Qwen3-VL-8B-Instruct | B3_mvp | +1.645 pp | [+0.179,+3.043] |
| Qwen3-VL-8B-Instruct | M1_ccm | +1.771 pp | [+0.280,+3.343] |
| Qwen3-VL-8B-Instruct | source_priority | +0.000 pp | [+0.000,+0.000] |
| UI-TARS-7B-SFT | B3_mvp | -0.063 pp | [-1.109,+1.011] |
| UI-TARS-7B-SFT | M1_ccm | -0.063 pp | [-1.093,+0.999] |
| UI-TARS-7B-SFT | source_priority | +0.000 pp | [+0.000,+0.000] |

The historical saturation statement is composition-specific. Omitting UI-TARS changes full-minus-omit by only -0.063 pp for both B3 and M1, with intervals crossing zero. Omitting Qwen3 costs +1.645 pp B3 and +1.771 pp M1; omitting GTA1 costs +4.428 pp B3 and +3.416 pp M1. Thus the averaged DECOMP fixed-budget $2\to3$ cell contrast can be positive while one specific third lineage, UI-TARS, is saturated. The estimands are not contradictory.

The historical 63.88% endpoint is `M1_ccm`, not source-priority majority. The source-priority bridge is separately reported and must not be substituted.

## Final interpretation

Arm A favors low rho. Arm B shows diversity-increase switches have negative net value. These two independent diagnostics satisfy the preregistered human closure rule. Arm C supports kappa as a close total effective-count approximation but does not rescue the row-level selector, and the transferred constants point in opposite errors across dependence strata.

The recorded decision is `CLOSE_CORRELATION_DISCOUNT_DIRECTION`. No correctly-anchored rho rescue round is authorized. All findings remain evaluation-side and post-selection.
