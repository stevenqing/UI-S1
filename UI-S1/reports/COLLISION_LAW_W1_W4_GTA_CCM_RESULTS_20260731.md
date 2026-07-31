# Collision-Law W1-W4, GTA, and CCM Results

Date: 2026-07-31

Status: complete. W1-W4, official MVP, GTA1 bare/N=5, CCM discovery, P3-CCM, and frozen W4 CCM confirmation have all finished.

## Scope and evaluation contract

This document consolidates the final compact results generated under `runs/collision-law/2026-07-30/`. It reports negative results and triggered kill conditions without post-hoc relabeling.

- W1/W2 AndroidControl uses 7,650 clean rows after excluding the fixed 58-row quarantine.
- W1/W2 Mind2Web uses 2,080 aligned visual steps.
- W4 AndroidControl-Curated uses 8,377 rows in each Low/High setting.
- W1, P3, CCM discovery, and W4 aggregation use deterministic grouped held-out folds.
- A0 is the model selected using development folds and evaluated on the held-out fold.
- W4 exact source candidates are scored with their original released responses after the correction recorded in Amendment 009.
- Raw traces and model assets are excluded from Git; compact JSON results and hashes are retained.

## Executive result

1. W1 aggregation is benchmark-dependent. AndroidControl aggregation is negative, while Mind2Web discrete density improves over held-out best by 6.44 pp and joint PKA by 6.63 pp.
2. P1 fails: stratum collision does not reverse-order aggregation gain (`rho=+0.316`, `p=0.684`).
3. P2/K1 is heterogeneous: three of five cells have grounding flips above action flips; no preregistered global pass rule exists.
4. P3 fails in all three pools: kappa-guided allocation beats random but not the five-lineage corner.
5. Official MVP reproduction passes at 61.35%, within 0.35 pp of the 61.7% anchor. GTA1 bare is 49.40%; N=5 self-consistency is 49.34% and provides no gain.
6. A5a fixes a large unequal-self-vote artifact but K3 remains triggered. Bare CCM likelihood ratios trigger K4 because they significantly hurt AndroidControl Low.
7. Risk-controlled CCM succeeds on discovery by avoiding significant regressions and significantly improving two pools.
8. W4 confirms large CCM gains over frozen deployment sources, but not universal superiority over W4's own held-out best: Low is +0.56 pp and nonsignificant; High is -1.23 pp and significantly inferior.

## W1: aggregation operators

Primary deployable scope, held-out grouped-fold Step SR:

| Pool | Models | A0 held-out best | A1 plurality + median | A2 plurality + density | A3 joint PKA | A4 continuous PKA | Oracle |
|---|---:|---:|---:|---:|---:|---:|---:|
| AndroidControl Low | 5 | 79.11% | 72.14% | 72.08% | 68.26% | 68.46% | 87.54% |
| AndroidControl High | 4 | 60.76% | 59.32% | 59.44% | 50.33% | 50.37% | 76.27% |
| Mind2Web visual | 6 | 51.78% | 57.12% | 58.22% | 58.41% | 32.69% | 79.86% |

### W1 operator deltas

| Pool | A2 - A1 | A3 - A2 | A4 - A3 | A3 - A0 |
|---|---:|---:|---:|---:|
| AndroidControl Low | -0.07 pp | -3.82 pp | +0.20 pp | -10.85 pp |
| AndroidControl High | +0.12 pp | -9.11 pp | +0.04 pp | -10.43 pp |
| Mind2Web visual | +1.11 pp | +0.19 pp | -25.72 pp | +6.63 pp |

W1 conclusion: discrete density is useful on Mind2Web but not AndroidControl. Original joint PKA is not generally better than sequential density, and continuous mode is strongly harmful on Mind2Web.

## W1: P1 strata

| Stratum | Rows | Mean same-error kappa | A0 Step SR | A3 Step SR | A3 - A0 |
|---|---:|---:|---:|---:|---:|
| AndroidControl action-dominant hard core | 1,443 | 0.119 | 0.00% | 0.00% | 0.00 pp |
| AndroidControl grounding-dominant hard core | 930 | 0.050 | 0.00% | 0.00% | 0.00 pp |
| Mind2Web CLICK | 1,774 | 0.211 | 53.27% | 61.50% | +8.23 pp |
| Mind2Web SELECT + TYPE | 306 | 0.166 | 43.14% | 40.52% | -2.61 pp |

P1 reverse-order test: Spearman `rho=+0.316`, `p=0.684`; prediction not satisfied. The AndroidControl strata are all-model hard cores by construction and therefore have zero A0/A3 success, limiting rank-correlation resolution.

## W2: complete five-view matrix

Step SR over `full`, border perturbation `v1`, 50% crop `v2`, 75% crop `v3`, and deployment profile `v4`:

| Representative cell | Full | v1 | v2 | v3 | v4 |
|---|---:|---:|---:|---:|---:|
| GUI-R1-7B / AndroidControl Low | 58.13% | 58.04% | 50.34% | 50.97% | 9.41% |
| GUI-R1-7B / AndroidControl High | 45.22% | 45.42% | 36.72% | 37.61% | 8.78% |
| UI-AGILE-7B / AndroidControl Low | 77.57% | 77.71% | 62.22% | 66.30% | 26.00% |
| UI-AGILE-7B / AndroidControl High | 60.76% | 60.82% | 44.13% | 46.63% | 24.44% |
| TongUI-7B / Mind2Web visual | 52.93% | 52.12% | 46.97% | 49.76% | 50.19% |

The AndroidControl `v4` deployment profile causes severe degradation and dominates its view-variation noise estimates.

## W2: P2/K1 flip mechanism

`v1` is compared with `full`. Grounding flips are conditioned on stable predicted action type.

| Cell | Action flip | Grounding flip | Grounding - action | Direction satisfied |
|---|---:|---:|---:|---|
| GUI-R1-7B / AndroidControl Low | 3.71% | 3.10% | -0.62 pp | No |
| GUI-R1-7B / AndroidControl High | 7.49% | 7.48% | -0.006 pp | No |
| UI-AGILE-7B / AndroidControl Low | 1.44% | 2.13% | +0.69 pp | Yes |
| UI-AGILE-7B / AndroidControl High | 4.55% | 5.76% | +1.21 pp | Yes |
| TongUI-7B / Mind2Web visual | 3.32% | 10.69% | +7.37 pp | Yes |

P2/K1 result: heterogeneous, with three of five directional successes. TongUI provides the strongest view-versus-component asymmetry.

## W2: noise floor and K2

MDE is twice the sample standard deviation over the five views.

| Cell | Mean Step SR | Sample SD | MDE |
|---|---:|---:|---:|
| GUI-R1-7B / AndroidControl Low | 45.38% | 20.45 pp | 40.89 pp |
| GUI-R1-7B / AndroidControl High | 34.75% | 15.08 pp | 30.16 pp |
| UI-AGILE-7B / AndroidControl Low | 61.96% | 21.24 pp | 42.47 pp |
| UI-AGILE-7B / AndroidControl High | 47.36% | 14.97 pp | 29.95 pp |
| TongUI-7B / Mind2Web visual | 50.39% | 2.32 pp | 4.64 pp |

K2 is not triggered: the W1 Mind2Web A3 gain of +6.63 pp exceeds the 4.64 pp MDE.

## W2: original P3 allocation

All methods use exactly five forwards per row.

| Pool | C1 one model/five views | C2 five models/full | C3 kappa mixed | C4 random mixed | C3 beats both corners | C3 beats random |
|---|---:|---:|---:|---:|---|---|
| AndroidControl Low | 57.40% | 71.90% | 69.12% | 54.31% | No | Yes |
| AndroidControl High | 44.76% | 50.37% | 48.33% | 43.70% | No | Yes |
| Mind2Web visual | 54.76% | 58.94% | 58.51% | 55.72% | No | Yes |

P3 fails in all three pools. Kappa guidance carries signal relative to random allocation but does not beat the strongest corner.

## W3: official MVP and GTA1

ScreenSpot-Pro, 1,581 rows:

| Method | Accuracy | Correct | Delta/reference |
|---|---:|---:|---:|
| Official MVP source: bare full image | 49.46% | 782 | - |
| Official MVP source: official code | 61.35% | 970 | -0.35 pp vs 61.7% anchor |
| Official MVP source: paper centroid | 61.73% | 976 | +0.38 pp vs official code |
| Official MVP source: graph-centroid ablation | 62.05% | 981 | +0.69 pp vs official code |
| GTA1 checkpoint bare | 49.40% | 781 | -0.70 pp vs 50.1% model-card anchor |
| GTA1 deterministic N=5, temperature 0.7 | 49.34% | 780 | -0.06 pp vs GTA1 bare |

Official MVP reproduction passes its fixed +/-1 pp anchor band. Naive GTA1 N=5 self-consistency does not improve the bare checkpoint.

## A5a: leave-one-out K3 retrial

Original A3 gave non-comparable self-votes across action classes. A5a removes only each candidate's own vote.

| Pool | A0 | A2 | Original A3 | A5a LOO | A5a - A3 | A5a - A2 | A5a vs A3 wins/losses |
|---|---:|---:|---:|---:|---:|---:|---:|
| AndroidControl Low | 79.11% | 72.08% | 68.26% | 70.71% | +2.44 pp | -1.37 pp | 189 / 2 |
| AndroidControl High | 60.76% | 59.44% | 50.33% | 56.27% | +5.95 pp | -3.16 pp | 539 / 84 |
| Mind2Web visual | 51.78% | 58.22% | 58.41% | 58.41% | 0.00 pp | +0.19 pp | 0 / 0 |

A5a confirms a large implementation artifact, especially on AndroidControl High, but K3 remains triggered because corrected joint mode exceeds A2 in only one of three pools.

## CCM discovery

A5b uses pooled likelihood ratios, A5c uses the frozen `3 pair types x 3 candidate classes` tables, A5d adds family evidence averaging, and A5d-risk adds nested development non-inferiority thresholding.

| Pool | A0 | A5a | A5b pooled LR | A5c nine LR | A5d family | A5d-risk |
|---|---:|---:|---:|---:|---:|---:|
| AndroidControl Low | 79.11% | 70.71% | 77.67% | 76.21% | 76.24% | 78.93% |
| AndroidControl High | 60.76% | 56.27% | 61.46% | 61.41% | 61.58% | 61.48% |
| Mind2Web visual | 51.78% | 58.41% | 60.19% | 60.72% | 60.82% | 60.67% |

### CCM diagnostics

| Pool | A5d-risk override rate | Conditional override SR | `S_gap` correctness AUROC |
|---|---:|---:|---:|
| AndroidControl Low | 6.24% | 41.09% | 0.393 |
| AndroidControl High | 23.71% | 46.03% | 0.417 |
| Mind2Web visual | 47.74% | 50.35% | 0.395 |

K4 triggers because A5c is significantly inferior to A0 on AndroidControl Low, despite gaining +2.31 pp over A5a on Mind2Web. Bare LR variants are therefore diagnostic rather than the primary method claim.

A5d-risk passes the discovery success rule: zero significantly inferior pools and two significantly superior pools. Its score gap is not a strong verifier signal; all reported AUROCs are below the prior Mind2Web negative-dispersion AUROC of 0.660.

## P3-CCM aligned allocation diagnostic

All four corners use the same A5d aggregator; C3 greedily adds the unit with the greatest development simulated-CCM increment.

| Pool | C1 views | C2 lineages | C3 greedy CCM | C4 random CCM | C3 beats both corners | C3 beats random |
|---|---:|---:|---:|---:|---|---|
| AndroidControl Low | 57.82% | 76.24% | 79.57% | 78.37% | Yes | Yes |
| AndroidControl High | 45.25% | 60.22% | 61.37% | 59.22% | Yes | Yes |
| Mind2Web visual | 54.13% | 61.15% | 60.29% | 58.03% | No | Yes |

Objective alignment improves P3 from zero to two directional passes, but forced five-unit allocation still selects the weak GUI-R1 High `v4` unit in every fold. This remains diagnostic after K4.

## W4: AndroidControl-Curated individual models

Each cell contains 8,377 rows.

| Model | Low Step SR | Low type accuracy | High Step SR | High type accuracy |
|---|---:|---:|---:|---:|
| GUI-R1-3B | 70.79% | 73.50% | 52.63% | 61.99% |
| GUI-R1-7B | 75.53% | 79.63% | 57.13% | 67.97% |
| UI-AGILE-3B | 68.59% | 71.23% | 54.65% | 64.81% |
| UI-AGILE-7B | 62.59% | 73.69% | 52.01% | 66.48% |
| UI-R1-E-3B | 70.12% | 73.74% | 54.05% | 64.55% |

## W4: aggregation robustness

| Setting | A0 held-out best | A1 plurality + median | A2 plurality + density | A3 exact-candidate PKA | A4 continuous PKA | Oracle | All-five hard core |
|---|---:|---:|---:|---:|---:|---:|---:|
| Low | 75.53% | 72.39% | 72.42% | 67.67% | 68.51% | 82.95% | 17.05% (1,428) |
| High | 57.13% | 56.56% | 57.22% | 54.95% | 55.23% | 71.34% | 28.66% (2,401) |

Curated labels do not reverse the main AndroidControl result. Low aggregation remains below A0. High A2 exceeds A0 by only 0.08 pp; A3/A4 remain lower.

### W4 threshold sensitivity

Kendall tau is computed against the model ranking at grounding radius 0.14.

| Radius | Low tau | High tau |
|---:|---:|---:|
| 0.06 | 0.20 | 0.40 |
| 0.08 | 0.40 | 0.40 |
| 0.10 | 0.20 | 1.00 |
| 0.12 | 0.53 | 1.00 |
| 0.14 | 1.00 | 1.00 |
| 0.16 | 1.00 | 0.80 |
| 0.18 | 0.80 | 0.80 |
| 0.20 | 0.60 | 1.00 |
| 0.22 | 0.60 | 1.00 |
| 0.24 | 0.80 | 1.00 |
| 0.26 | 0.80 | 1.00 |
| 0.28 | 0.80 | 1.00 |
| 0.30 | 0.80 | 1.00 |

Low model ranking is more sensitive to strict grounding tolerances. High stabilizes for most radii at or above 0.10.

## W4: frozen CCM confirmation

Calibration tables, source sets, and thresholds were serialized before any W4 inference. No W4 label affects CCM calibration or selection. Exact selected candidates are scored with their original released responses.

### Against frozen deployment sources

| Setting | Frozen source | Source SR | CCM SR | Delta | Wins / losses | One-sided superiority p | Override rate |
|---|---|---:|---:|---:|---:|---:|---:|
| Low | UI-AGILE-3B | 68.59% | 76.09% | +7.50 pp | 676 / 48 | `3.70e-143` | 15.51% |
| High | UI-AGILE-7B | 52.01% | 55.90% | +3.89 pp | 488 / 162 | `3.68e-39` | 32.36% |

Both fixed-source gains are large and significant. High preserves the positive discovery direction. Low changes from a small discovery loss (-0.18 pp) to a large confirmation gain, so only one of two settings satisfies the literal same-sign condition.

### Against W4 held-out A0, reporting only

| Setting | W4 A0 | CCM | Delta | Wins / losses | Statistical result |
|---|---:|---:|---:|---:|---|
| Low | 75.53% | 76.09% | +0.56 pp | 516 / 469 | Not significant; superiority `p=0.071` |
| High | 57.13% | 55.90% | -1.23 pp | 337 / 440 | Significantly inferior; `p=1.24e-4` |

W4 confirms transferable selection value against fixed deployment sources, but not universal superiority over the strongest poolwise baseline.

## Kill conditions and final claim status

| Condition | Outcome | Consequence |
|---|---|---|
| K1: P2 component asymmetry fails | Heterogeneous: 3/5 directional successes | No global mechanism pass; retain cell-level evidence |
| K2: Mind2Web gain below MDE | Not triggered | Mind2Web aggregate positive result retained |
| K3: joint PKA not above sequential density | Triggered; still triggered after A5a | PKA demoted to unified perspective |
| K4: calibrated LR fails guarded cross-pool criterion | Triggered | Bare LR removed from primary method claim |

Final supported positioning:

- **Measurement/law:** strongly supported. Error concentration depends on benchmark space, source family, action class, and evaluator contract.
- **Aggregation baseline result:** Mind2Web discrete density is positive; AndroidControl A1-A4 is negative or tied even on Curated labels.
- **Method result:** risk-controlled CCM transfers substantial value relative to frozen deployment sources, but does not dominate held-out best on every confirmation setting.
- **Allocation result:** CCM-aligned allocation is better than kappa-only allocation but remains imperfect under forced fixed budgets.
- **Confidence result:** `S_gap` is not a strong general correctness verifier.
- **Mind2Web confirmation:** discovery-stage only because no official versioned corrected-label release with an auditable evaluator was found.

## Result artifact index

All links below are compact tracked artifacts; raw trace directories are intentionally omitted.

| Slice | Artifact |
|---|---|
| Machine status and hashes | [STATUS.json](../runs/collision-law/2026-07-30/STATUS.json) |
| Detailed stage report | [REPORT.md](../runs/collision-law/2026-07-30/REPORT.md) |
| W1 aggregation | [w1_aggregators.json](../runs/collision-law/2026-07-30/w1_aggregators.json) |
| W1 P1 strata | [w1_strata.json](../runs/collision-law/2026-07-30/w1_strata.json) |
| W1 kappa | [w1_kappa.json](../runs/collision-law/2026-07-30/w1_kappa.json) |
| W2 flips/K1 | [w2_flips.json](../runs/collision-law/2026-07-30/w2_flips.json) |
| W2 MDE | [w2_noise.json](../runs/collision-law/2026-07-30/w2_noise.json) |
| W2 P3 | [w2_allocation.json](../runs/collision-law/2026-07-30/w2_allocation.json) |
| A5a K3 retrial | [a5a_retrial.json](../runs/collision-law/2026-07-30/a5a_retrial.json) |
| CCM discovery | [a5_ccm.json](../runs/collision-law/2026-07-30/a5_ccm.json) |
| P3-CCM | [p3_ccm.json](../runs/collision-law/2026-07-30/p3_ccm.json) |
| W3 MVP/GTA | [w3_summary.json](../runs/collision-law/2026-07-30/w3_summary.json) |
| W4 aggregation | [w4_curated.json](../runs/collision-law/2026-07-30/w4_curated.json) |
| W4 threshold sweep | [w4_threshold.json](../runs/collision-law/2026-07-30/w4_threshold.json) |
| Frozen CCM calibration | [ccm_confirmation_frozen.json](../runs/collision-law/2026-07-30/ccm_confirmation_frozen.json) |
| W4 CCM confirmation | [w4_ccm_confirmation.json](../runs/collision-law/2026-07-30/w4_ccm_confirmation.json) |
| Real audited task/trajectory cases | [BENCHMARK_ERROR_OVERLAP_ANALYSIS_20260729.md](BENCHMARK_ERROR_OVERLAP_ANALYSIS_20260729.md#L48) |

## Reproduction and corrections

- Preregistration and Amendments 001-008 precede their corresponding result slices.
- Amendment 009 is explicitly post-result and corrects exact-candidate W4 scoring from a reserialized four-value degenerate box back to the original released response. No inference, calibration, threshold, or candidate selection changed.
- The invalid intermediate W4 JSON was never committed.
- Focused operator/calibration regression suite: 19/19 passing.
- Final compact artifact hashes are stored in `STATUS.json`.
