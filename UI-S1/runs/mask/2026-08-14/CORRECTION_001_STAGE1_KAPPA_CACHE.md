# MASK Correction 001: Stage-1 kappa cache

Date: 2026-08-14
Timing: after stage-1 implementation commit `af79bc1`, before any `STAGE1.json` or MASK aggregate result was produced.

The first stage-1 attempt recomputed identical development failure kappas for every one of 4,095 source subsets. It was stopped after remaining CPU-bound without emitting `STAGE1.json`. The already-created verifier figure is retained as `FAILED_ATTEMPT_001_VERIFIER_CONTOURS.pdf`; it is not an adjudication artifact.

The correction computes each outer fold's complete 12-by-12 pairwise failure-kappa matrix once, then takes the exact indexed submatrix for each source subset. Undefined-pair flags are cached in the same way. This changes only execution complexity. The rows, folds, kappa definition, undefined-pair policy, subset enumeration, aggregators, isotonic fit, and M-G1 rule are unchanged.
