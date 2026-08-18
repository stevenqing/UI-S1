# OWIN Amendment 007: M1 parsing sequence consistency

Date: 2026-08-17

Status: `FROZEN_AFTER_COMMON_GENERATION_BEFORE_ANY_OWIN_LABEL_OR_COORDINATE_PARSE`

Common generation completed with 2,400 successful traces and zero failures. This amendment was written before any OWIN trace was joined to target bbox, evaluated for correctness, or aggregated into B3/M1/single-forward endpoints.

Earlier execution text required freezing `delta_pool_M1` after parsing only common outputs and before parsing partial/uncovered outputs. That is incompatible with the frozen fold-local M1_ccm estimand. M1 fitting uses candidate correctness labels and pairwise similarity on all allowed outer-development rows. A 500-row matched oracle-pool M1 therefore cannot be fitted before partial/uncovered candidate correctness exists.

The resolution is:

1. after common generation, parse and freeze only B3 pool calibration, zero-jitter single-forward calibration, size-half B3 heterogeneity, calibration identity, and small-target sensitivity inputs;
2. do not compute or report oracle-pool M1 at this stage;
3. generate partial and uncovered with the unchanged runner and settings;
4. after all 6,000 traces pass integrity, privately join all 500 rows to labels and fit M1_ccm per outer fold on the four non-test folds;
5. report common M1 calibration and all-stratum M1 results once from that cross-fitted 500-row pool.

The existing M1 common anchor remains the preflight-locked frozen V-only N12 fold-local output on all 931 common rows. This amendment changes only execution/parsing order. It does not change M1's formula, folds, candidate pool, baseline, sample, GPU calls, B3/single calibration, O-I estimand, or any threshold.

No partial/uncovered correctness may be inspected before their generation is complete. Common B3/single calibration cannot alter generation or later parsing.