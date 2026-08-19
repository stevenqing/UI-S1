# TILE Amendment 001: Stage-0 operational sources and schema

Date: 2026-08-18

Status: `FROZEN_BEFORE_ANY_TILE_RESULT`

No TILE preflight, pair table, fitted curve, row score, ledger, or result existed when this amendment was written.

## Correctness and geometry semantics

Single-slot correctness is the frozen ScreenSpot-Pro evaluator event

`x1 <= point_x <= x2 and y1 <= point_y <= y2`.

It is inclusive on bbox edges. This differs intentionally from half-open crop-rectangle center containment, which remains `left <= center_x < right` and `top <= center_y < bottom`.

## Authoritative rows and folds

`runs/cover/2026-08-16/raw/arm_a_rows.jsonl` is authoritative for row ID, application, fold, target stratum, and canonical B3 correctness. Its SHA-256 must be locked in preflight.

`runs/cwin/2026-08-17/raw/stage0_rows.jsonl` is an independent row-wise B3/fold anchor. Preflight must require exact equality of row IDs, application folds, and B3 correctness between COVER and CWIN. The deterministic `group_folds` recomputation from the GTA1 N12 pool must also match every application fold. Any mismatch blocks Stage 0.

GTA1 target bbox, candidate point, candidate region, and application come from the hash-validated top18 shards loaded through `allocation_eval.py`. Shared N12 regions must equal candidate regions for views 0 through 11.

## OWIN layout source

`runs/owin/2026-08-17/raw/arm_b_rows.jsonl` is the only allowed tile-coordinate source. TILE may use rows at N in `{4,5,6,8,11}` and only `tiling.rectangles`. It may not reconstruct layouts from `tiling_layout()` during scoring. Preflight must verify 1,581 unique rows at every N and hash the raw file.

## Stage-0 raw schema

The committed implementation must write:

- `raw/eccentricity_pairs.jsonl`: one eligible existing `(row,slot)` pair with row ID, application, fold, slot, eccentricity, target area, scale membership by fit only, and inclusive single-slot correctness;
- `raw/fold_curves.jsonl`: one record per `(outer_fold,phase,scale,bin)` with fit folds, row/pair counts, area median, boundaries, bin interval, correctness, and fallback flag;
- `raw/row_scores.jsonl`: one record per `(row,N,outer_fold,phase)` with tile rectangles hash, center-covering tile indices/eccentricities/probabilities, minimum-e tile/value, max p-hat, B3 label, expected repair/damage/net, hard p-hat diagnostic, and domains;
- `STAGE0.json`: input hashes, fixed-N and nested ledgers, folds, selected N, T-G1, T-G2 ratio/review requirement, T-K5, and Stage-1 authorization false;
- `REPORT.md`: all mandatory curves/ledgers and limitations after Stage-0 computation.

Every raw JSONL is exclusive-create and performs write, flush, and fsync per record. Stage 0 refuses to run if any output path already exists.

## Bootstrap and selection

Stage-0 point values use cross-fitted outer-test rows. Its 10,000 application-group bootstrap reuses frozen cross-fitted row scores; it does not refit curves or reselect N unless an endpoint explicitly says so. N-selection uncertainty is descriptive and not needed for T-G1, which applies to every fixed N point value.

T-G2 uses the nested-selected cross-fitted row scores. If repair is zero, ratio is `NA_REVIEW_REQUIRED`; otherwise ratio is damage/repair. Human review is recorded in a separate committed decision after STAGE0 and before any Stage-1 amendment.

This amendment changes no TILE endpoint, grid, threshold, control, or authorization.