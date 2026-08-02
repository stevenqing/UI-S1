# Scale-Up Main Table

## Controlled 7B results

| Comparison | Left | Right | Delta | 99% CI | One-sided p |
|---|---:|---:|---:|---:|---:|
| Mixed N12 M1 vs V-only N12 M1 | 63.82 | 60.40 | +3.42 pp | [+1.41, +5.67] | 9.999e-05 |
| Mixed N12 B3 H3-native vs V-only N12 B3 | 63.63 | 60.09 | +3.54 pp | [+1.27, +6.15] | 9.999e-05 |
| Mixed N12 M1 vs GTA1-7B bare | 63.82 | 49.40 | +14.42 pp | [+10.39, +18.84] | 9.999e-05 |
| Mixed N12 M1 vs Qwen3-VL-8B-Instruct bare | 63.82 | 54.65 | +9.17 pp | [+5.71, +13.01] | 9.999e-05 |
| Mixed N12 M1 vs UI-TARS-7B-SFT bare | 63.82 | 33.46 | +30.36 pp | [+24.47, +36.73] | 9.999e-05 |
| Mixed N16 M1_ccm vs V-only N16 M1_ccm | 63.76 | 58.25 | +5.50 pp | [+3.09, +8.16] | 9.999e-05 |
| Mixed N16 B3_mvp vs V-only N16 B3_mvp | 63.76 | 58.32 | +5.44 pp | [+2.86, +8.41] | 9.999e-05 |

The H3-native B3 row is the primary drop-in comparison (63.63 vs 60.09). A later Allocation/Closing reconstruction gives 63.69 for the mixed side; that one-row implementation sensitivity is reported but is not substituted into the primary H3 statistic.

## Reporting dispositions

| Item | Main-text disposition |
|---|---|
| H1 N=2 | Appendix only: M1/M2 collapse to B0 and M1 headroom capture is 0% |
| Budget decline | Use L1 N=4 to N=16 and X3 slope CI; H1 N=4 to N=10 is rule comparison only |
| MDE | Use v1-only 0.09-1.16 pp; v2-v4 are deployment/information shifts |
| Sampling | GUI-RC point slope is negative but 99% CI crosses zero; title scope is fixed-view allocation axis |

## 72B gate

G1 and G2 are pending local checkpoint acquisition and inference. Paper-only 70.4 and 73.1 remain context rows and are excluded from paired calculations.
