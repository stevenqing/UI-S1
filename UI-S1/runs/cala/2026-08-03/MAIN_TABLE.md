# CALA Main Table

All learned policies are cross-fitted by application group. Every row within a comparison uses the same number of scored model-view or model-region forwards. B3 is unchanged.

## Preregistered adjudication

| Comparison | CALA | Baseline | Delta | 99% CI (pp) | One-sided p | Adjudication |
|---|---:|---:|---:|---:|---:|---|
| 7B CALA-S N12 vs Uniform N12, B3 | 62.18% | 63.69% | -1.52 pp | [-3.12, -0.06] | 0.9967 | Primary FAIL |
| 7B CALA-A N12 vs CALA-S N12, B3 | 63.12% | 62.18% | +0.95 pp | [-0.75, +2.36] | 0.06919 | Adaptive primary FAIL |
| 7B CALA-A N8 vs Uniform N8, B3 | 63.06% | 61.99% | +1.08 pp | [+0.29, +2.10] | 0.0004 | Preregistered secondary PASS |
| 72B CALA-S N8 vs Uniform N8, B3 | 45.41% | 41.24% | +4.17 pp | [+1.56, +7.06] | 9.999e-05 | Equal-budget transfer PASS |
| 72B CALA-A N8 vs Uniform N8, B3 | 43.96% | 41.24% | +2.72 pp | [+0.90, +4.65] | 0.0003 | Equal-budget transfer PASS |

## Accuracy by budget

| Scale | Policy | Budget | B3 | M1 | pass@N |
|---|---|---:|---:|---:|---:|
| 7B | V_only | 4 | 61.23% | 61.42% | 68.88% |
| 7B | V_only | 8 | 60.72% | 60.78% | 71.22% |
| 7B | V_only | 12 | 60.09% | 60.40% | 72.80% |
| 7B | V_only | 16 | 58.32% | 58.25% | 74.07% |
| 7B | Uniform_Mixed | 4 | 61.86% | 59.90% | 73.43% |
| 7B | Uniform_Mixed | 8 | 61.99% | 63.19% | 77.36% |
| 7B | Uniform_Mixed | 12 | 63.69% | 63.82% | 79.19% |
| 7B | Uniform_Mixed | 16 | 63.76% | 63.76% | 80.20% |
| 7B | Quality_Only | 4 | 60.85% | 61.99% | 69.45% |
| 7B | Quality_Only | 8 | 62.49% | 62.81% | 73.43% |
| 7B | Quality_Only | 12 | 63.38% | 63.88% | 78.24% |
| 7B | Quality_Only | 16 | 63.06% | 63.31% | 79.44% |
| 7B | CALA_S | 4 | 61.16% | 60.72% | 74.83% |
| 7B | CALA_S | 8 | 62.49% | 61.86% | 77.93% |
| 7B | CALA_S | 12 | 62.18% | 62.30% | 80.01% |
| 7B | CALA_S | 16 | 63.00% | 63.12% | 81.28% |
| 7B | CALA_A | 8 | 63.06% | 62.43% | 77.23% |
| 7B | CALA_A | 12 | 63.12% | 63.06% | 79.25% |
| 7B | CALA_A | 16 | 63.06% | 62.87% | 80.71% |
| 72B | GTA1_N8 | 8 | 23.85% | 25.74% | 69.32% |
| 72B | Uniform_Mixed_N8 | 8 | 41.24% | 52.12% | 83.18% |
| 72B | CALA_S_N8 | 8 | 45.41% | 55.28% | 84.12% |
| 72B | CALA_A_N8 | 8 | 43.96% | 50.85% | 83.68% |

The 72B values are local equal-budget transfer results, not absolute SOTA results. The completed Scale-Up experiment remains below the paper-only 70.4/73.1 references.
