# OTEXT Amendment 001: Stage-0 numerics

Date: 2026-08-14
Timing: after prereg/preflight, before OTEXT OCR completion and before any Stage-0 label statistic.

- Theta quantiles use `numpy.quantile(..., method="linear")` on strictly positive inner-train row-best scores.
- The selected-setting decile table includes all inner-validation rows. Rows are sorted by `(score, row_id)` and split into ten near-equal consecutive bins with `numpy.array_split`; this prevents zero-score ties from creating an implementation-dependent boundary.
- O-G1 combines fold-selected validation gains with outer-test row counts as weights, exactly as specified. It does not pool repeated inner-validation rows under another denominator.
- Any probability conditional on an empty stratum is reported as `null`; no smoothing is applied.

No grid, endpoint, baseline, tie order, or evidence-status rule changes.
