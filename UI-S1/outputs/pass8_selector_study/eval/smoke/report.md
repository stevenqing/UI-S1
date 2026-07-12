# Frozen Pass@8 Selector Evaluation

Split: **smoke**. Labels were unsealed only after all paired outputs passed completeness and packet-hash checks.

Primary utility is student-relative:

$$u = \frac{N_{rescue}-N_{regress}}{N_{steps}}.$$

| selector | baseline acc | selected acc | oracle ceiling | net utility | rescue | regress | changed | parse | 95% cluster CI | gate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| current | 0.00% | 0.00% | 34.78% | +0.00pp | 0 | 0 | 65.22% | 100.00% | [+0.00pp, +0.00pp] | FAIL-CLOSED |
| strong | 0.00% | 0.00% | 34.78% | +0.00pp | 0 | 0 | 52.17% | 100.00% | [+0.00pp, +0.00pp] | FAIL-CLOSED |

## Paired Corrector Deltas

- **strong − current**: +0.00pp, 95% CI [+0.00pp, +0.00pp] — no locked win

## Decision

No policy training is authorized unless the locked selector gate passes. The oracle ceiling is diagnostic only and is not a deployable selector.

## Scope Warning

This is selector-fresh, not benchmark-fresh: the underlying benchmark episodes and the GT-conditioned critical target set predate this frozen comparison.
