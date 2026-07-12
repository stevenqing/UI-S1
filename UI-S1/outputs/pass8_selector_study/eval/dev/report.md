# Frozen Pass@8 Selector Evaluation

Split: **dev**. Labels were unsealed only after all paired outputs passed completeness and packet-hash checks.

Primary utility is student-relative:

$$u = \frac{N_{rescue}-N_{regress}}{N_{steps}}.$$

| selector | baseline acc | selected acc | oracle ceiling | oracle captured | net utility | rescue | regress | changed | parse | 95% cluster CI | gate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| current | 0.00% | 6.49% | 28.57% | 22.73% | +6.49pp | 15 | 0 | 53.68% | 98.70% | [+3.54pp, +9.75pp] | PASS |
| strong | 0.00% | 2.60% | 28.57% | 9.09% | +2.60pp | 6 | 0 | 29.87% | 98.27% | [+0.84pp, +4.89pp] | PASS |
| exact_plurality | 0.00% | 3.03% | 28.57% | 10.61% | +3.03pp | 7 | 0 | 19.91% | 100.00% | [+0.87pp, +5.70pp] | PASS |
| cross_source_consensus | 0.00% | 5.63% | 28.57% | 19.70% | +5.63pp | 13 | 0 | 46.75% | 100.00% | [+2.67pp, +9.05pp] | PASS |

## Paired Corrector Deltas

- **strong − current**: -3.90pp, 95% CI [-6.52pp, -1.68pp] — no locked win; discordant correct 1 vs 10, action agreement 57.58%.
- **exact_plurality − current**: -3.46pp, 95% CI [-7.14pp, +0.00pp] — no locked win; discordant correct 4 vs 12, action agreement 45.02%.
- **cross_source_consensus − current**: -0.87pp, 95% CI [-4.00pp, +2.50pp] — no locked win; discordant correct 5 vs 7, action agreement 37.66%.

## Decision

The predeclared selector gate **passes** for current, strong, exact_plurality, cross_source_consensus. This authorizes preparation of a new train-split 25% selected-revision + 75% clean-replay arm; the dev/locked rows in this study must never be used for training.
The oracle ceiling is diagnostic only and is not a deployable selector.

## Scope Warning

This is selector-fresh, not benchmark-fresh: the underlying benchmark episodes and the GT-conditioned critical target set predate this frozen comparison.
