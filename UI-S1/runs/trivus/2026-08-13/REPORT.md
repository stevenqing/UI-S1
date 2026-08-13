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

## Incremental utility headroom

The direct TARGET_ONLY choice and strongest fallback have complementary successes in every benchmark.

| Benchmark | Oracle direct-or-strongest headroom | 99% CI |
| --- | ---: | --- |
| Mind2Web | +3.56 pp | `[+3.04, +4.08]` |
| ScreenSpot-Pro | +0.25 pp | `[+0.09, +0.46]` |
| AndroidControl | +1.45 pp | `[+0.99, +1.94]` |

The direct policy alone is not better than the strongest fallback. The remaining problem is therefore selective override, not candidate availability.

## Method decision

The next model is a benchmark-specific, cross-fitted incremental-utility gate over the existing TARGET_ONLY set ranker. It predicts `win`, `loss`, and `tie` relative to the strongest fallback and overrides only when predicted net utility is sufficiently positive and loss risk is sufficiently low.

The base ranker, override head, standardizer, thresholds, and fallback are fitted independently per benchmark. The representation schema, nested OOF protocol, incremental-utility loss, safety calibration, artifact sealing, and held-out evaluation remain unified.

## Next boundary

The new primitive is implemented and synthetic-tested only. Before any real-data optimizer step, the cross-fitting phases, strongest-baseline label source, threshold grid, seeds, artifacts, and untouched confirmation dataset must be frozen. Existing formal outer labels may be used for exploratory development but not confirmation or promotion.