# ICC Main Tables

## Arm A

| Fold | Selected rho_v | Selected rho_l | Accuracy | Class |
| ---: | ---: | ---: | ---: | --- |
| 0 | 0.0 | 0.0 | 75.24% | low_endpoint |
| 1 | 0.4 | 0.6 | 65.28% | interior |
| 2 | 0.0 | 0.0 | 57.19% | low_endpoint |
| 3 | 0.4 | 0.0 | 55.27% | low_endpoint |
| 4 | 0.2 | 0.2 | 67.83% | interior |

## Arm C

| Stratum | Phi fold mean | Fold range | AndroidControl reference | Difference |
| --- | ---: | ---: | ---: | ---: |
| within_lineage | 0.672 | [0.655,0.687] | 0.895 | -0.223 |
| cross_lineage | 0.577 | [0.555,0.600] | 0.398 | +0.179 |

## Arm B

| Direction | Rows | Wrong-to-correct | Correct-to-wrong | Net | Correction/all |
| --- | ---: | ---: | ---: | ---: | ---: |
| composition_same | 0 | 0 | 0 | +0 | NA |
| diversity_decrease | 0 | 0 | 0 | +0 | NA |
| diversity_increase | 104 | 13 | 23 | -10 | 12.50% |
| lineage_substitution | 0 | 0 | 0 | +0 | NA |
| same_L_concentration_decrease | 7 | 1 | 1 | +0 | 14.29% |
| same_L_concentration_increase | 0 | 0 | 0 | +0 | NA |

## Same-budget audit

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
