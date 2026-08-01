# CCM Head-to-Head Report

Date: 2026-07-31

Status: zero-GPU stage and H1 complete; H2 positive and H3 gate open; H3 model eligibility preflight pending.

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

## H1: ScreenSpot-Pro same-candidate-set head-to-head

All 1,581 rows have exact candidate identity coverage. The official generator returned 19 candidates on 1,579 rows, 18 on one row, and 16 on one row after deduplicating identical crop regions. Every row retained at least nine subimages, so N=2/4/10 use valid official ordered prefixes without padding.

| Method | N=2 | N=4 | N=10 |
|---|---:|---:|---:|
| B0 full image | 49.46% | 49.46% | 49.46% |
| B1 random candidate | 55.09% | 56.80% | 54.08% |
| B2 coordinate mean | 48.96% | 47.88% | 41.49% |
| B3 official MVP | 59.84% | 61.23% | 60.47% |
| B3 paper centroid | 49.78% | 61.80% | 61.67% |
| B3 graph centroid | 49.72% | 61.80% | 61.80% |
| B4 ReGUIDE algorithm-level | 58.19% | 61.29% | 59.52% |
| M1 coordinate CCM | 49.46% | 61.42% | 60.91% |
| M2 CCM + risk fallback | 49.46% | 61.42% | 60.53% |
| pass@N | 65.02% | 68.88% | 72.04% |

Three seeded N=10 proposal perturbations give M1 accuracies 61.10%, 61.54%, and 60.85%. Sample SD is 0.35 pp and MDE is 0.70 pp.

### H1 prediction adjudication

| Prediction | Observed | Result |
|---|---|---|
| P-H1a: N10 M1 > B3 by more than MDE, bootstrap `p<0.01` | +0.44 pp; MDE 0.70 pp; `p=0.148` | Failed |
| P-H1b: M1 non-decreasing and B3 N4-to-N10 declines | M1 -0.51 pp; B3 -0.76 pp | Failed |
| P-H1c: M1 > published GRPO 62.8 by more than MDE | M1 60.91%, delta -1.89 pp | Failed |

H-K1 triggers because P-H1a and P-H1c both fail. The same-candidate-set general method claim is removed. H-K2 does not trigger: B3 reproduces the predicted N4-to-N10 decline, even though M1 also declines. The direct mechanism result is therefore narrower than predicted: collision correction recovers 0.44 pp over official B3 at N=10 but does not eliminate candidate-count saturation and does not significantly exceed noise.

M1 headroom capture is 0% at N=2, 61.6% at N=4, and 50.5% at N=10. The best observed coordinate aggregation is graph centroid at 61.80%, still below the paper-only GRPO reference of 62.8%.

## Next execution

1. H1 is complete and H-K1 is triggered.
2. Because H2 is positive, run result-blind ScreenSpot-Pro bare scoring for locally available H3 candidate models.
3. Freeze eligible D2 models before any mixed-pool result.
4. Run H3 only if at least three eligible lineages are available.

H3 asset preflight currently finds local GTA1-7B, UI-TARS-7B-SFT, and SeeClick checkpoints. Qwen3-VL-8B-Instruct is absent from the workspace and scratch search. UI-TARS and SeeClick have no audited ScreenSpot-Pro bare score yet, so they are not eligible until result-blind bare scoring confirms at least 24.70% accuracy. H3 inference has not started.
