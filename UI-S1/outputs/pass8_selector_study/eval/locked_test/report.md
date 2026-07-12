# Frozen Pass@8 Selector Evaluation

Split: **locked_test**. Labels were unsealed only after all paired outputs passed completeness and packet-hash checks.

Primary utility is student-relative:

$$u = \frac{N_{rescue}-N_{regress}}{N_{steps}}.$$

| selector | baseline acc | selected acc | oracle ceiling | oracle captured | net utility | rescue | regress | changed | parse | 95% cluster CI | gate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| current | 0.42% | 6.78% | 30.79% | 20.93% | +6.36pp | 46 | 1 | 60.03% | 98.59% | [+4.53pp, +8.31pp] | PASS |
| strong | 0.42% | 4.94% | 30.79% | 14.88% | +4.52pp | 33 | 1 | 34.18% | 98.87% | [+2.93pp, +6.20pp] | PASS |
| exact_plurality | 0.42% | 3.67% | 30.79% | 10.70% | +3.25pp | 25 | 2 | 26.27% | 100.00% | [+1.87pp, +4.73pp] | PASS |
| cross_source_consensus | 0.42% | 5.79% | 30.79% | 17.67% | +5.37pp | 39 | 1 | 47.18% | 100.00% | [+3.61pp, +7.25pp] | PASS |

## Paired Corrector Deltas

- **strong − current**: -1.84pp, 95% CI [-3.69pp, +0.00pp] — no locked win; discordant correct 13 vs 26, action agreement 53.53%.
- **exact_plurality − current**: -3.11pp, 95% CI [-5.17pp, -1.13pp] — no locked win; discordant correct 17 vs 39, action agreement 42.23%.
- **cross_source_consensus − current**: -0.99pp, 95% CI [-2.98pp, +1.12pp] — no locked win; discordant correct 23 vs 30, action agreement 41.53%.

## Decision

The predeclared selector gate **passes** for current, strong, exact_plurality, cross_source_consensus. This authorizes preparation of a new train-split 25% selected-revision + 75% clean-replay arm; the dev/locked rows in this study must never be used for training.
The oracle ceiling is diagnostic only and is not a deployable selector.

## Scope Warning

This is selector-fresh, not benchmark-fresh: the underlying benchmark episodes and the GT-conditioned critical target set predate this frozen comparison.
