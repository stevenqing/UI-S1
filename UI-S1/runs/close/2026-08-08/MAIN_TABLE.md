# Aggregator Closure Main Table

Date: 2026-08-08

## E1 Mind2Web: Step SR (%)

| Arm | Majority | A0 | Ours | A1 | A2 | A3 | A4 | Row best |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| C-uni | 32.02 | 31.88 | 26.68 | 23.22 | 24.18 | 24.13 | 13.32 | Majority |
| C-cond | **32.31** | 31.88 | 31.59 | 25.63 | 27.45 | 27.45 | 13.46 | Majority |
| C-rand | 31.78 | **31.88** | 28.32 | 21.54 | 18.22 | 18.22 | 4.95 | A0 |
| C-self | **32.12** | 31.83 | 29.28 | 25.67 | 27.12 | 27.12 | 14.81 | Majority |

Global best: C-cond + majority, 32.31%.

### Mind2Web majority controls

| Comparison | Delta | 99% CI | MDE pass | CI pass |
|---|---:|---:|---:|---:|
| C-cond − C-uni | +0.29 pp | [-0.70,+1.26] pp | No | No |
| C-cond − C-rand | +0.53 pp | [-0.51,+1.59] pp | No | No |
| C-cond − C-self | +0.19 pp | [-0.54,+0.92] pp | No | No |

Mind2Web MDE: 0.61 pp.

## E1 ScreenSpot-Pro: grounding accuracy (%)

| Arm | Majority | A0 | B3 official / Ours | A1 | A2 | A3 | A4 | Row best |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| C-uni | 59.84 | 59.84 | 63.69 | 60.34 | 63.88 | 63.88 | **63.95** | A4 |
| C-cond | 61.10 | 61.10 | 65.91 | 61.54 | **66.48** | **66.48** | 66.29 | A2/A3 |
| C-rand | 59.84 | 59.84 | **60.53** | 33.97 | 59.52 | 59.52 | 59.77 | B3 official |
| C-self | 60.66 | 60.66 | 64.58 | 60.34 | 64.83 | 64.83 | **65.02** | A4 |

Global best: C-cond + A2/A3, 66.48%.

B3 official belongs to the type-first discrete-density family but is not implementation-equivalent to A2: B3 uses complete-link groups plus coverage tie-breaking.

### ScreenSpot-Pro majority controls

| Comparison | Delta | 99% CI | MDE pass | CI pass |
|---|---:|---:|---:|---:|
| C-cond − C-uni | +1.27 pp | [-0.14,+2.68] pp | Yes | No |
| C-cond − C-rand | +1.27 pp | [-0.14,+2.68] pp | Yes | No |
| C-cond − C-self | +0.44 pp | [-1.17,+2.29] pp | No | No |

ScreenSpot-Pro MDE: 0.70 pp.

## E3 containment mechanism

| Benchmark | Rank-0 containment | Rank-11 containment | Drop | V-only N16−N4 | 99% CI |
|---|---:|---:|---:|---:|---:|
| ScreenSpot-Pro | 99.94% | 61.04% | 38.90 pp | -2.91 pp | [-5.58,-0.36] pp |
| Mind2Web | 40.38% | 31.15% | 9.23 pp | +0.34 pp | [-1.22,+1.98] pp |

E3 verdict: `MECHANISM_SUPPORTED_WITH_HIGH_START_CONDITION` (qualitative two-benchmark evidence, not a law).

## Final gates

| Gate | Value | Consequence |
|---|---|---|
| E-K1 | **Triggered** | Candidate-generation claim becomes aggregator-specific; cancel E2 and AndroidControl |
| E-K2 | Not evaluated | Native anchors not run |
| E-K3 | Not evaluated | Native four-arm run not launched |
| E-K4 | Not triggered | XF4 failure is consistent with a high-start condition |
