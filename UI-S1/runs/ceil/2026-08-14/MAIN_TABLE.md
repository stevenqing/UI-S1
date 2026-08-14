# CEIL Main Table

## Arm B: Recoverable-subset candidate ranking

| Benchmark | Recoverable | Cheap AUROC | 99% CI | Visual AUROC | Verifier AUROC | Decision |
| --- | ---: | ---: | --- | ---: | ---: | --- |
| mind2web | 2021 | 0.688 | [0.665, 0.709] | 0.585 | 0.685 | C_D2 |
| screenspot_pro | 968 | 0.540 | [0.501, 0.583] | 0.567 | 0.536 | C_D1 |
| androidcontrol | 275 | 0.380 | [0.327, 0.432] | 0.729 | 0.369 | DESCRIPTIVE_LOW_N |

## Arm A: Post-hoc effective-vote ceiling

| Panel | Aggregator | Full N_eff | Support max | Delta infinity | 99% CI |
| --- | --- | ---: | ---: | ---: | --- |
| mind2web/C_cond | density | 2.454 | 2.555 | +68.41 pp | [+65.10, +71.38] pp |
| mind2web/C_cond | majority | 2.454 | 2.555 | +67.69 pp | [+61.94, +70.65] pp |
| mind2web/C_rand | density | 3.701 | 3.778 | +21.65 pp | [+2.73, +73.76] pp |
| mind2web/C_rand | majority | 3.701 | 3.778 | +68.22 pp | [+22.28, +71.00] pp |
| mind2web/C_self | density | 2.310 | 2.473 | +70.72 pp | [+67.35, +73.75] pp |
| mind2web/C_self | majority | 2.310 | 2.473 | +67.88 pp | [+55.53, +70.70] pp |
| mind2web/C_uni | density | 2.207 | 2.436 | +73.32 pp | [+38.72, +76.22] pp |
| mind2web/C_uni | majority | 2.207 | 2.436 | +60.92 pp | [+44.99, +70.37] pp |
| screenspot_pro/C_uni | density | 1.594 | 1.707 | -0.26 pp | [-1.70, +2.51] pp |
| screenspot_pro/C_uni | majority | 1.594 | 1.707 | -0.59 pp | [-1.31, +0.03] pp |

Mind2Web $\Delta_\infty$ values are weakly identified far-support sensitivity outputs, not precise recoverable headroom. The finite ideal-three-vote isotonic gains range from -0.13 to +3.32 pp across its panels. ScreenSpot-Pro $\Delta_\infty$ is near zero with intervals crossing zero.
