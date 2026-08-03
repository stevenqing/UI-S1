# Difficulty-Conditioned Reallocation Main Table

## R1 stratified realization gate

| Highest-disagreement metric | N4 | N8 | N12 | N16 | N24 | N24-N4 / 99% CI |
|---|---:|---:|---:|---:|---:|---|
| B3 | 19.62% | 16.77% | 20.25% | 19.62% | 18.99% | -0.63 pp; [-6.27, +5.76] |
| M1 | 23.10% | 19.62% | 20.25% | 19.30% | 21.84% | -1.27 pp; [-7.89, +5.67] |
| pass@N | 38.29% | 45.89% | 51.27% | 53.48% | 57.28% | +18.99 pp; [+14.11, +24.91] |

R1 status: **FAIL / R-K1**. Candidate headroom rises, but B3 realization ratio is -0.033.

## R4 selective accuracy

| Pool | Retained coverage | Retained B3 | Gain vs full | Random mean | Random 99% CI |
|---|---:|---:|---:|---:|---:|
| Uniform Mixed N12 | 90% | 69.06% | +5.36 pp | 63.69% | [+62.73, +64.77] |
| Uniform Mixed N12 | 80% | 74.60% | +10.91 pp | 63.69% | [+62.10, +65.35] |
| Uniform Mixed N12 | 70% | 79.02% | +15.33 pp | 63.69% | [+61.66, +65.73] |
| V-only N12 | 90% | 63.92% | +3.84 pp | 60.08% | [+59.07, +61.18] |
| V-only N12 | 80% | 67.48% | +7.40 pp | 60.09% | [+58.47, +61.71] |
| V-only N12 | 70% | 70.71% | +10.62 pp | 60.08% | [+58.05, +62.21] |

## R5 72B diagnostic

| Diagnostic | 7B Uniform N8 | 72B Uniform N8 |
|---|---:|---:|
| B3 | 61.99% | 41.24% |
| Mean normalized failed-pair distance | 0.1137 | 0.1539 |
| Wrong B3 selected model, dominant | GTA1: 524 | GTA1: 872 |

Tight-error pollution hypothesis: **FAIL**. The 72B wrong winner composition is highly nonuniform, but failed candidates are more dispersed, not tighter.
