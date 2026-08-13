# Writing Freeze After B2

Date: 2026-08-06

Status: `FROZEN_PATH_B_DIAGNOSTIC_PAPER`

This file is the sole writing authority after D0/D1/D2 adjudication. Historical reports remain provenance, not alternative claim menus.

## 1. Paper shape

The paper is a ScreenSpot-Pro diagnostic paper organized around:

1. the budget-curve sign flip;
2. frozen weak-model admission;
3. seven mechanism-bearing negative results;
4. two-scale localization of source-sensitive aggregation failure;
5. a scale-dependent B2 repair that does not beat the strongest single model.

The dominance-gap law does **not** enter the main text as a law. ScreenSpot correlations are directionally negative but below the frozen magnitude threshold, and Mind2Web/AndroidControl row-level transfer is unavailable.

## 2. Three main claims

### Claim 1: budget-curve sign flip

- single-lineage slope: `-0.002467` per forward, 99% CI `[-0.004908, -0.000124]`;
- cross-lineage slope: `+0.003052` per forward, 99% CI `[+0.001082, +0.005053]`;
- intervals do not overlap and lie on opposite sides of zero;
- at N=16, Mixed exceeds V-only by `+5.44 pp` for B3 and `+5.50 pp` for M1.

This claim is ScreenSpot-Pro-specific because D2 transfer was not executable.

### Claim 2: frozen weak-model admission

- UI-TARS-7B-SFT bare accuracy: `33.46%`;
- it trails GTA1 by `15.94 pp`;
- admission was frozen before results;
- the mixed N12 pool reaches B3/M1 `63.69% / 63.82%`.

The claim is that a weak admitted lineage can coexist with a stronger mixed pool, not that every weak model improves every pool.

### Claim 3: aggregator source bias

| Scale | Incorrect rows | GTA observed | GTA expected | Residual | p | Cramer's V |
|---|---:|---:|---:|---:|---:|---:|
| 7B | 574 | 489 | 191.33 | +26.36 | `4.12e-152` | 0.779 |
| 72B recovery | 929 | 871 | 348.38 | +35.42 | `1.21e-273` | 0.822 |

B1 passes at both scales. The mechanism label is `heterogeneous_pool_aggregation_effect`, not shared-proposer causation.

## 3. Required B2 interpretation

Two qualifications are mandatory.

First, 72B nested LN reaches `70.52%`, `+29.29 pp` over B3 `41.24%`, but remains `-0.89 pp` below reported Qwen3.5 best-single `71.41%`. R5 and R6 have identical 72B descriptive tiers (`62.81 / 70.46 / 70.59` in the frozen grid), consistent with each lineage's development-strongest view being view 0. The repair should be described as restoring a collapse, not producing super-single-model gain.

Second, 7B nested LN is `63.69%`, exactly equal to B3: delta `0.00 pp`, 99% CI `[-0.54, +0.65]`, p=`0.5626`. B2 fails its cross-scale gate and no cross-scale aggregation fix is claimed.

## 4. D0 R7 ruling

R7's historical `2.15%--2.66%` range was an implementation fault. The weighted centroid returned `sum(w_i*x_i)` without dividing by `sum(w_i)`. After repair:

- 7B R7 tiers are `62.62 / 61.99 / 61.80%`;
- 72B R7 tiers are `51.61 / 64.64 / 64.83%`.

The controlled recovered-bank comparison between fixed 21 methods and the same grid without R7 yields identical nested selections and predictions at both scales. D-K1 is false. Report both old and repaired grids; do not revise the B2 gate.

Method-set provenance remains explicit:

- historical broken 21-method study: 7B `61.99%`, 72B `70.59%`;
- repaired 21-method recovered-bank rerun: 7B `61.99%`, 72B `70.52%`;
- combined-24 recovery: 7B `63.69%`, 72B `70.52%`.

These are not byte-identical repeats of one method set.

## 5. D1 dominance ruling

The exhaustive ScreenSpot action-level analysis contains 432 two-lineage and 1,728 three-lineage pools.

| Metric | Raw rho | Raw 99% CI | Partial rho | Partial 99% CI |
|---|---:|---:|---:|---:|
| B3 minus best | -0.388 | [-0.430, -0.347] | -0.367 | [-0.410, -0.323] |
| M1 minus best | -0.482 | [-0.530, -0.434] | -0.499 | [-0.547, -0.450] |

The direction survives controls for mean quality and failure kappa, but the frozen `rho < -0.6` strength criterion is not met. Mind2Web and AndroidControl mixed metrics are unavailable because row-level traces are absent. Therefore:

- do not call this a dominance law;
- do not recast 72B failure as confirmation of a law;
- retain the 7B/72B split as an unexplained scale dependence;
- the motivating `63.69%` B3 and `70.52%` nested-LN points are nonexchangeable and never enter one correlation.

D-K2 is not statistically triggered on ScreenSpot because controlled CIs are negative; the combined test is not adjudicated because two benchmarks are missing. The practical paper action is still no law claim.

## 6. D2 transfer ruling

The frozen `rows.parquet` is absent, and its row-level source predictions are also absent. Only `score.json`, `audit.json`, and the frozen manifest remain. They preserve member quality but not joint correctness, coordinates, mixed-pool output, or failure kappa.

Manifest-level anchors may be reported only as preflight:

- Mind2Web M-cross-3 dominance gap `2.79 pp`, mean member quality `45.58%`;
- Mind2Web M-same-3 gap `0.91 pp`, mean quality `51.31%`;
- AndroidControl Low A-cross-2 gap `19.34 pp`;
- A-same-2-agile gap `1.53 pp`;
- A-same-2-gui gap `1.32 pp`.

No mixed metric or transfer direction may be attached to these anchors. D-K3 is not adjudicated. Allocation claims remain ScreenSpot-Pro-specific.

## 7. Explicit non-claims

Do not claim:

1. absolute-score superiority: Qwen3.5 bare `71.41%` exceeds 72B LN `70.52%`;
2. selector superiority: CCM gains only `0.13--0.19 pp` in the relevant budgets, H-K1 triggers, and M1-minus-B3 never exceeds `1.51 pp`;
3. a cross-scale aggregation repair: B2 cross-scale fails;
4. shared-proposer causation: B4 is `NOT_SUPPORTED`;
5. count balancing as a general method: deterministic balanced accuracy `49.72%` lies inside the 10,000-draw random balanced-accuracy 99% interval `[23.34%, 55.22%]` with mean `41.03%` and SD `11.36 pp`;
6. dominance law: D1 does not pass;
7. cross-benchmark transfer: D2 is blocked;
8. UI-Zoomer/GUI-RC head-to-head superiority: those runs are absent.

For unit consistency, the deterministic `+8.48 pp` delta is not compared directly with an accuracy interval. Relative to original B3 `41.24%`, the random balanced-accuracy interval corresponds to an approximate delta interval `[-17.90, +13.98] pp`.

## 8. Reproducibility statement

The paper must state all of the following:

1. recovered banks are not byte-identical to frozen banks; status is `COMPLETE_WITH_RECOVERY_DRIFT`;
2. 72B M1 moves from `52.12%` to `53.19%`;
3. B1 winning-set members move from `1374/1000/370` to `1370/1003/369`;
4. P1 falls back to N8 because `stata_windows_27` yields only seven unique crops, so P1/P2 are unequal-budget context;
5. historical 21-method and combined-24 recovery results use different method sets;
6. R7 was faulty historically, repaired before the controlled D0 rerun, and did not change nested selection on the recovered bank.

## 9. Other frozen wording

- Delete X2 as a result section; keep one limitation sentence saying it could not be reproduced.
- AndroidControl main MDE uses v1-only values `0.09--1.16 pp`; the nonexchangeable five-view version moves to the appendix.
- M0 is net `+1` from five flips: three rescues and two regressions. Canonical drop-in delta is `+3.60 pp`, 99% CI `[+1.31, +6.22]`; CCM attribution is `+0.13 pp`.
- R4 is only a strengthening of a transferred signal: AUROC `0.744` to `0.830`, and Mixed B3 leads by `7.12 pp` at matched 80% coverage. Deterministic N12 does not inherit K=10 random sampling or FDR guarantees.
- Paper-only values `62.8`, `70.4`, `71.41` independent context, `73.1`, `+13.4%`, and `+5.38` are not used in local difference calculations.

## 10. Final paper sentence

> Cross-lineage allocation improves ScreenSpot-Pro budget scaling, but heterogeneous candidate pools expose severe source-sensitive aggregation failures. Lineage normalization restores the 72B collapse without exceeding the strongest single model and does not improve 7B, leaving the repair scale-dependent rather than universal.
