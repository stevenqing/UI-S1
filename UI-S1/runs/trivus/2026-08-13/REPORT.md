# Benchmark-Adaptive TriVUS Report

## Status

This is a post-hoc method-development analysis. It does not change the formal 2026-08-12 outcome `TRIVUS_NOT_PROMOTED`.

The existing formal runner already trained three independent TARGET_ONLY models and selected family-specific safe thresholds. Their published held-out outputs were therefore sufficient to test whether independent training alone resolves the failure.

## TARGET_ONLY diagnostic

No benchmark passed all three exploratory readiness conditions.

| Benchmark | Primary cell safety | Strongest cell safety | Primary family improvement | Ready |
| --- | --- | --- | --- | --- |
| Mind2Web | fail | fail | fail | no |
| ScreenSpot-Pro | pass | pass | fail | no |
| AndroidControl | pass | fail | pass | no |

AndroidControl improved over its primary majority fallback by 1.85 percentage points on the equal-cell mean, with 99% CI `[1.08, 2.65]`. Its high cell was not safe against the strongest UI-AGILE fallback. Mind2Web remained negative against its frozen VUS-SR baseline. ScreenSpot-Pro was safe but approximately neutral.

Independent training is therefore necessary for the method definition but insufficient for promotion.

## Full candidate headroom

The earlier direct-or-strongest diagnostic materially understated the available space because it considered only the candidate already selected by the existing ranker. The sealed candidate labels show substantially larger headroom over the strongest fallback.

| Benchmark | Full candidate oracle | Existing policy-direct union | Unrecovered ranking gap |
| --- | ---: | ---: | ---: |
| Mind2Web | +24.29 pp | +3.17 pp | **+21.12 pp** |
| ScreenSpot-Pro | +15.31 pp | +0.16 pp | **+15.15 pp** |
| AndroidControl | +6.88 pp | +1.48 pp | **+5.40 pp** |

The atlas validates all ten sealed private-label fold hashes and covers all 18,644 public rows. Policy unions and full candidate oracle are label-dependent upper bounds, not deployable methods.

## Incremental utility headroom

The direct TARGET_ONLY choice and strongest fallback have complementary successes in every benchmark, but this is only the final safety-layer space.

| Benchmark | Oracle direct-or-strongest headroom | 99% CI |
| --- | ---: | --- |
| Mind2Web | +3.56 pp | `[+3.04, +4.08]` |
| ScreenSpot-Pro | +0.25 pp | `[+0.09, +0.46]` |
| AndroidControl | +1.45 pp | `[+0.99, +1.94]` |

The direct policy alone is not better than the strongest fallback. The remaining problem is therefore selective override, not candidate availability.

## Method decision

The main model is now a benchmark-specific contextual candidate-success verifier. It trains on every valid candidate label rather than using KEEP whenever fallback is correct. The available supervision contains 175,728 VUS candidate labels and 12,000 AndroidControl candidate labels. The existing blind VLM score has candidate AUROC 0.595 and top-1 accuracy 36.85% on the 14,644 VUS rows versus a 68.00% candidate oracle, so frozen visual logits alone are not an adequate semantic verifier.

The incremental-utility gate remains a second, cross-fitted deployment layer. It predicts `win`, `loss`, and `tie` relative to the strongest fallback and overrides only when predicted net utility is sufficiently positive and loss risk is sufficiently low.

## Sequential realization

A frozen blind visual ordering already concentrates correct candidates near the front of the list. Equal-cell hit@k is:

| Benchmark | Hit@1 | Hit@2 | Hit@4 | Hit@6 | Full oracle |
| --- | ---: | ---: | ---: | ---: | ---: |
| Mind2Web | 30.49% | 41.08% | 49.72% | **54.19%** | 59.21% |
| ScreenSpot-Pro | 45.21% | 59.65% | **70.40%** | 75.21% | 79.57% |
| AndroidControl | 65.65% | **74.33%** | n/a | n/a | 77.73% |

The minimum budget recovering at least 90% of the cell oracle is 2 for both AndroidControl cells, 6 for all Mind2Web cells, and 4--6 for ScreenSpot-Pro. The deployable method is therefore a cheap ranker followed by a sequential stronger verifier with benchmark-calibrated budget and strongest fallback. It does not evaluate every candidate with the expensive verifier.

`Hit@k` and first-success ranks are label-dependent evaluation quantities. Runtime stopping uses calibrated candidate and fallback probabilities only.

The base ranker, override head, standardizer, thresholds, and fallback are fitted independently per benchmark. The representation schema, nested OOF protocol, incremental-utility loss, safety calibration, artifact sealing, and held-out evaluation remain unified.

## Next boundary

The candidate-success, incremental-utility, and sequential stopping primitives are implemented and synthetic-tested only. Before any real-data optimizer step, the cross-fitting phases, semantic verifier, strongest-baseline label source, budget and threshold grids, seeds, artifacts, and untouched confirmation dataset must be frozen. Existing formal outer labels may be used for exploratory development but not confirmation or promotion.

The sequential training protocol is now frozen. A no-optimizer real-data smoke assembled outer fold 0 / inner holdout 1 with exact physical label isolation:

- fit folds 3 and 4: 7,428 rows and 74,484 valid candidate labels;
- checkpoint fold 2: 3,838 rows and 38,622 labels;
- holdout fold 1: 3,666 rows and 37,206 labels.

Only the corresponding VUS and Android fold files were opened. Outer fold 0 remained unopened by the smoke. No model parameters, optimizer, or backward pass were created. Real-data optimizer execution and confirmation remain unauthorized.

The two-layer OOF implementation is complete but unexecuted. The cheap ranker writes label-free candidate logits, probabilities, and ordering for each benchmark/outer/holdout scope. The stronger verifier reloads only matching cheap OOF contexts, uses two fit folds plus one independent checkpoint fold, fits its standardizer on verifier-training rows only, and emits a second label-free OOF artifact. Both real-data entry points reject authorization before loading public inputs, cheap artifacts, or private labels.

A one-time exploratory optimizer authorization and two-phase 8-GPU launcher are implemented. The launcher runs exactly 60 cheap OOF jobs followed by 60 verifier OOF jobs, validates 240 model/prediction artifacts, writes a hash manifest, and atomically publishes the nonce-scoped attempt. Failure consumes the nonce and leaves the attempt isolated. The first authorization was consumed by the failed attempt described below.

The first exploratory nonce failed during cheap OOF training because random batching admitted an all-zero-weight inactive batch. Twelve jobs had completed before the failure was observed; no verifier job ran and nothing was published. The nonce, receipt, logs, and partial artifacts are retained. Correction 002 filters loss batching to positive-weight rows and makes the launcher stop its own remaining workers on the first failure. A replacement authorization is required.

The second nonce completed all 60 cheap jobs and produced 120 cheap artifacts. Its first eight verifier workers failed before optimizer construction because two independently normalized fold-local weight vectors were concatenated before fit-scope standardization. Correction 003 concatenates unweighted family rows first and assigns weights once over the combined two-fold fit scope. The attempt is retained and unpublished; a new authorization must rerun both phases.