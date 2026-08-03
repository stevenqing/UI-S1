# Source-Bias Main Table

## B1 source bias

| Pool/stratum | Rows | GTA observed | GTA expected | GTA residual | Chi-square p | Cramer's V |
|---|---:|---:|---:|---:|---:|---:|
| 7B B3 incorrect | 574 | 489 | 191.33 | +26.36 | 4.122e-152 | 0.779 |
| 72B B3 incorrect | 929 | 872 | 348.38 | +35.49 | 1.155e-274 | 0.824 |

## B2 nested lineage normalization

| Comparison | LN | Reference | Delta | 99% CI / availability | One-sided p |
|---|---:|---:|---:|---|---:|
| 7B nested LN vs B3 | 61.99% | 63.69% | -1.71 pp | [-3.09, -0.21] | 0.999 |
| 7B nested LN vs M1 | 61.99% | 63.82% | -1.83 pp | [-3.28, -0.19] | 0.9985 |
| 7B nested LN vs reported best-single | 61.99% | 54.65% | +7.34 pp | independent trace; no paired CI | n/a |
| 72B nested LN vs B3 | 70.59% | 41.24% | +29.35 pp | [+21.57, +35.78] | 9.999e-05 |
| 72B nested LN vs M1 | 70.59% | 52.12% | +18.47 pp | [+12.95, +23.44] | 9.999e-05 |
| 72B nested LN vs reported best-single | 70.59% | 71.41% | -0.82 pp | paired CI | n/a |
