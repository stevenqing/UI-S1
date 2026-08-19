# TILE Stage-0 Eccentricity Proxy Report

Date: 2026-08-18

Status: `PASS_TILE_STAGE0_COMPLETE_AWAITING_HUMAN_REVIEW`

Stage 0 is a zero-GPU, post-selection optimistic proxy. Max curve probability is not B3/M1 accuracy.

| N | V-only expected net | Repair | Damage | Contextual C-uni net |
| ---: | ---: | ---: | ---: | ---: |
| 4 | -13.799 pp | 267.12 | 485.28 | -17.404 pp |
| 5 | -3.408 pp | 345.37 | 399.25 | -7.013 pp |
| 6 | -0.751 pp | 358.07 | 369.94 | -4.357 pp |
| 8 | +6.384 pp | 389.79 | 288.86 | +2.779 pp |
| 11 | +10.578 pp | 421.93 | 254.69 | +6.973 pp |

Primary V-only original-correct domain: 950 rows. Contextual C-uni original-correct domain: 1007 rows. Existing crop-covered domain: 1356 rows.

Selected-policy V-only expected net 99% CI: `[0.048507, 0.163133]`; repair CI: `[0.226167, 0.308175]`; damage CI: `[0.141969, 0.182758]`.

Fold selections: `[11, 11, 11, 11, 11]`. T-G1=False; T-G2 review=True; ratio=0.60362414541417; T-K5=True. Every fold selected the N=11 grid endpoint.

All repair/damage values are fractional expectations, not observed flips. Stage 1 remains unauthorized.
