# CCM Head-to-Head Report

Date: 2026-07-31

Status: zero-GPU stage complete; H2 positive and H3 gate open; H1/H3 GPU stage waiting for protected PID 1814 to release GPUs 0-7.

## Preregistration

- Main protocol commit: `89f492c`.
- C2/C3 exact-definition amendment: `9279fab`.
- No H1/H3 candidate or result existed at either commit.
- Existing official MVP anchor has exactly five candidates per row and is not reused as N=2/4/10.

## C1: exchangeable MDE correction

`MDE_v1 = sqrt(2) * abs(v1-full)`. The original five-view MDE is retained as a distribution-shift diagnostic.

| Cell | Full | v1 | Absolute delta | v1-only MDE | Original five-view MDE |
|---|---:|---:|---:|---:|---:|
| GUI-R1-7B / AC Low | 58.13% | 58.04% | 0.09 pp | 0.13 pp | 40.89 pp |
| GUI-R1-7B / AC High | 45.22% | 45.42% | 0.21 pp | 0.30 pp | 30.16 pp |
| UI-AGILE-7B / AC Low | 77.57% | 77.71% | 0.14 pp | 0.20 pp | 42.47 pp |
| UI-AGILE-7B / AC High | 60.76% | 60.82% | 0.07 pp | 0.09 pp | 29.95 pp |
| TongUI-7B / Mind2Web | 52.93% | 52.12% | 0.82 pp | 1.16 pp | 4.64 pp |

C1 supports the objection: v2-v4 are not exchangeable noise. They are information deletion/deployment shifts and should not define the main MDE.

## H2: view collision floor

Same-model full/v1 failure kappas:

| Pair | Kappa | Matched-marginal p |
|---|---:|---:|
| GUI-R1-7B / AC Low | 0.924 | 0.001 |
| GUI-R1-7B / AC High | 0.860 | 0.001 |
| UI-AGILE-7B / AC Low | 0.924 | 0.001 |
| UI-AGILE-7B / AC High | 0.871 | 0.001 |
| TongUI-7B / Mind2Web | 0.772 | 0.001 |

Primary AndroidControl comparison:

- view-axis mean kappa: 0.895;
- cross-family full-view mean kappa: 0.398;
- difference: +0.496;
- exact pair-label randomization: `p=5.49e-4`;
- same-family scale mean kappa: 0.618.

H2 is positive. Same-model view perturbations preserve substantially more failure structure than cross-family replacement and are even more correlated than the average same-family scale pair. H3 gate is open.

## C2: reversed `S_gap` diagnosis

High gap is the inclusive OOF 90th percentile. The diagnostic set contains high-gap rows on which the selected CCM candidate fails.

| Pool | Failed high-gap rows | Hard-core overlap | Observed overlap rate | Hard-core base rate | Enrichment | Exact p |
|---|---:|---:|---:|---:|---:|---:|
| AC Low | 371 | 125 | 33.69% | 12.46% | 2.70x | `5.25e-28` |
| AC High | 476 | 223 | 46.85% | 23.73% | 1.97x | `3.11e-30` |
| Mind2Web | 82 | 38 | 46.34% | 20.14% | 2.30x | `4.23e-8` |

C2 passes in all three pools. The below-random correctness AUROC is not merely weak confidence: large positive `S_gap` failures are systematically concentrated in collision hard cores. Agreement is confidently wrong exactly where the error process clusters.

## C3: P1 with error-conditional agreement mass

Rows are restricted to deployable disagreement cases. Collision is the mean leave-one-out evaluator-kernel mass of failed candidates.

| Stratum | Rows | Error agreement mass | A0 | A3 | A3-A0 |
|---|---:|---:|---:|---:|---:|
| AC coordinate | 3,122 | 0.614 | 65.82% | 66.94% | +1.12 pp |
| AC parameterless | 752 | 0.784 | 52.66% | 12.50% | -40.16 pp |
| AC string | 3,411 | 0.202 | 88.01% | 48.11% | -39.90 pp |
| M2W CLICK | 1,167 | 0.716 | 56.64% | 69.15% | +12.51 pp |
| M2W SELECT+TYPE | 210 | 0.283 | 62.86% | 59.05% | -3.81 pp |

Spearman collision-versus-gain is 0.0 (`p=1.0`). C3 does not rescue P1. The result indicates that a single scalar collision mass still aliases action-contract and parameter effects; the calibrated conditional tables can improve decisions without yielding a one-dimensional cross-stratum law.

## H4: resolution law

Primary three-benchmark H4 is blocked because W2 AndroidControl has point GT only. Exact image SHA256 matching to pinned Curated assets finds 0/7,708 rows, so target areas cannot be transferred.

Descriptive available results:

| Benchmark | Rows | Bare | Zoom | Delta | Row Spearman rho | p |
|---|---:|---:|---:|---:|---:|---:|
| ScreenSpot-Pro | 1,581 | 49.46% | 61.35% | +11.89 pp | +0.062 | 0.013 |
| Mind2Web | 2,079 | 52.96% | 48.39% | -4.57 pp | -0.066 | 0.0026 |

One Mind2Web zero-height bbox is quarantined from the logarithmic area axis. Neither available row-level rho approaches 0.7, their signs differ, and AndroidControl is unauditable. H-K4 triggers. The area law is downgraded to a qualitative domain-boundary observation.

Figure: `fig1_area_law.pdf`.

## Gates after zero-GPU stage

| Gate | Result | Action |
|---|---|---|
| H2 collision floor | Positive | Open H3 |
| C2 reversed-gap mechanism | 3/3 pass | Retain collision-deception claim |
| C3 recalibrated P1 | Failed | Do not rescue scalar P1 law |
| H4 area law | H-K4 triggered | Descriptive only |
| H1 GPU availability | Blocked by protected PID 1814 | Wait; do not preempt |

## Next execution

1. Generate fresh official MVP candidates for N=2/4/10 on eight GPUs after PID 1814 releases them.
2. Evaluate B0-B4, M1/M2, pass@N with candidate-hash identity assertions.
3. Run three proposal perturbation seeds for ScreenSpot-Pro MDE.
4. Because H2 is positive, schedule H3 after H1 and freeze eligible model availability before H3 inference.
