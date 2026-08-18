# TILE: fixed tiling versus existing crops at matched budget

Round: `tile`

Date: 2026-08-18

Status: `PREREGISTERED_AFTER_OWIN_BEFORE_ANY_TILE_RESULT`

GPU: Stage 0 is zero GPU and contains the only automatic pre-GPU gate. Stage 1 is not authorized by this specification and requires a separate committed execution amendment.

## Scope, evidence, and authority

OWIN classified the corrected GT-oracle B3 opportunity as O-I2 and restricted follow-up to simple equal-budget geometry. TILE is that follow-up: a fixed, target-independent layout copied exactly from OWIN Arm B. No learned or adaptive placement mechanism is allowed.

Unlike OWIN, Stage 1 Arm T is a candidate method because its runtime windows use no GT, model attention, or target location. It remains post-selection, single-benchmark, and non-confirmatory because all ScreenSpot-Pro labels and this direction have already been examined. Mind2Web is excluded because its crop geometry and dependence differ.

TILE changes no status for F1, Q1, `TRIVUS_NOT_PROMOTED`, VUS-SR, SPLIT, MASK, CEIL, OTEXT, XSCR, DECOMP, EVID, ICC, COVER, CWIN, PRUNE, OWIN, or XSOFT.

## Frozen anchors and prohibited interpretation

OWIN Arm B reports existing 11-crop median union `0.3219237075617284`; fixed tiling median union is 0.7834 at N=4 and 1.0 by N=6. Fixed tiling center coverage rises from 1,056/1,581 at N=4 to 1,556/1,581 at N=11. Existing crop-center coverage is 1,356/1,581, or 85.77%.

Canonical B3 is 1,007/1,581 = `0.6369386464263125`. This is model correctness and is distinct from crop-center coverage. Historical 63.88% denotes M1_ccm, not source-priority majority.

OWIN raw GT-oracle-pool B3 on `uncovered_0` is `0.5033789971829187`, while existing B3 there is zero. OWIN factorized G11 is +7.340 pp, but it assumes that target-center containment by a fixed tile has the value of a GT-centered oracle crop. TILE prohibits using G11, or any OWIN G_N, as a target, expected gain, threshold, or Stage-1 claim. Stage 0 replaces that assumption with an explicitly optimistic eccentricity proxy.

The five disclosed ScreenSpot-Pro values are fold-level contamination anchors, not row IDs: `[0.6388361796331435, 0.6388361796331435, 0.6306135357368754, 0.6255534471853258, 0.6325110689437066]`. They cannot enter fitting, binning, N selection, thresholds, sampling, or implementation checks.

## Frozen folds and geometry units

Use the existing deterministic five-fold application grouping. For outer fold f:

- outer test is fold f;
- inner validation is `(f+1) mod 5`;
- inner train is the remaining three folds;
- outer development is all four non-test folds.

All rows from one application remain together. Every Stage-0 curve, N selection, and outer prediction follows this split.

N counts crop windows only. Arm T contains one full-image slot plus N fixed tiles, for pool size and per-row forward budget N+1. C2 contains frozen GTA1 view 0 plus frozen views 1 through N and also has pool size N+1. The full N12 baseline contains view 0 plus 11 crops and has budget 12. Therefore T-P3 is the strict equal-budget geometry contrast; T-P1 compares a potentially smaller Arm T pool with the main 12-slot pool.

The N grid is `{4,5,6,8,11}`. It is never expanded after results.

## Stage 0: eccentricity proxy and net ledger

Stage 0 uses frozen GTA1 candidate outputs and OWIN layouts only. It runs zero model forwards. It reads GT geometry/correctness and is evaluation-side post-selection analysis, not a deployable runtime score.

### Eligible source pairs and eccentricity

For every existing GTA1 crop slot view 1 through 11 whose half-open rectangle contains the GT bbox center, define

$$
e=\sqrt{((c_x-m_x)/644)^2+((c_y-m_y)/364)^2},
$$

where c is GT bbox center and m is crop center. Record row ID, slot, application, fold, eccentricity, target area, and single-slot correctness.

Target area is `max(0,x2-x1)*max(0,y2-y1)`. For every fit split, compute numeric eccentricity boundaries at quantiles `[0.1,...,0.9]` using NumPy `quantile(method="linear")`. Map both source pairs and new tile eccentricities with `bisect_right(boundaries,e)`, producing bins 0 through 9. Repeated numeric boundaries are retained and may make a bin empty; the frozen empty-bin fallback applies. This mapping is deterministic for unseen tile values and does not use correctness.

For each fit split, split source pairs by the row-level target-area median. The median is the ordinary numeric median over rows in that fit split. Rows with area less than or equal to the median are `small`; rows above it are `large`. Fit separate ten-bin curves for small and large pairs. Each bin's value is empirical single-slot correctness. If a scale/bin has no pair, use that scale's pooled eligible-pair correctness and flag `EMPTY_BIN_FALLBACK`; if an entire scale has no pair, use all-scale pooled correctness and flag `EMPTY_SCALE_FALLBACK`.

For inner-validation N selection, fit boundaries, scale median, and curves on inner train only. After choosing N, refit them once on outer development and apply to outer test. Outer-test correctness never fits a curve or selects N. All fit memberships, boundaries, counts, fallback flags, and curves are retained.

### Row-level optimistic tile score

For each row and N, use the exact committed OWIN Arm B tile rectangles. For every tile containing the GT center, compute eccentricity and map it to the fitted scale-specific curve. The row score is

$$
\hat p_{i,N}=\max_{j\in\text{center-covering tiles}} \hat P(correct\mid e_{ij},scale_i).
$$

If no tile covers the center, set the score to zero. Also report the minimum-e tile and its curve value. The maximum-probability composition is deliberately optimistic: it is neither the probability that at least one tile is correct nor a prediction of B3/M1 aggregation under dependence. It is used only as a Stage-0 upper proxy.

### Expected repair, damage, and net ledger

For canonical full-N12 B3 correctness $y_i\in\{0,1\}$ define:

$$
expected\ repair=(1-y_i)\hat p_{i,N},\qquad
expected\ damage=y_i(1-\hat p_{i,N}),
$$

$$
expected\ net=\hat p_{i,N}-y_i=repair-damage.
$$

These are fractional expected masses, not observed flips. The report must not call them actual repaired/damaged rows. Alongside expected damage, report the hard diagnostic count of original-correct rows with `p_hat < 0.5`; the threshold is descriptive and never selects N or gates Stage 1.

For every fixed N and the nested-selected policy report on outer-test rows:

1. micro expected net in full-benchmark pp;
2. expected repair and expected damage masses;
3. original-correct domain: denominator 1,007 population rows, held-out row count, expected damage mass/rate, and hard `p_hat<0.5` count/rate;
4. existing crop-covered domain: denominator 1,356 population rows, held-out row count, expected damage mass/rate, and hard count/rate;
5. expected change separately in OWIN `uncovered_0`, `partial_1_10`, and `common_11`;
6. fold values and full range.

The population denominators 1,007 and 1,356 are anchors; fold-domain denominators sum to them and are never interchanged.

### N selection

Within each outer fold, select one N by maximum inner-validation expected B3 net, tie-breaking by smaller N. Refit the curve on outer development and evaluate that selected N once on outer test. Stage 1, if authorized, uses this fold-local N-selection procedure, not a single post-hoc global N. Report all five selected N values and exact fold-weighted call budget. This is a nested benchmark procedure; it is not an invariant deployment hyperparameter.

Selecting N=4 or N=11 in any fold triggers T-K5. The grid is not expanded.

### Stage-0 gates

T-G1 triggers when every fixed-N cross-fitted outer-test expected net is below `0.007`. Equality passes. If triggered, stop before Stage 1 and retain all curves/ledgers as limitations.

For the nested-selected policy, define the damage-to-repair ratio as total expected damage divided by total expected repair over outer-test rows. If expected repair is zero, the ratio is `NA` and requires human review. T-G2 requires human review when the ratio exceeds 0.5. Record the point ratio, fold ratios, and review decision/reason before any Stage-1 amendment. T-G2 is not an automatic pass.

Stage 0 cannot validate B3 or M1 method accuracy. It authorizes only a GPU pilot when T-G1 passes and the T-G2 review explicitly continues.

## Stage 1: fixed tiling candidate method

Stage 1 is unauthorized by this specification. After Stage 0, commit selected N by fold, every OWIN tile coordinate, C1 sample/random windows, exact call counts, runner/tests, model/index/prompt/processor/parser hashes, hardware/shards, failure handling, and a one-time nonce in a separate execution amendment.

### Arms and controls

Arm T reruns one full image plus N fixed tiles for each row, using the fold-selected N. Tile layouts must be byte-identical to OWIN Arm B raw layouts. No geometry reconstruction is allowed.

The primary baseline is the frozen full GTA1 V-only N12 pool. C2 is frozen view 0 plus views 1 through N for the same row/fold N; it is the strict equal-budget geometry control. C2 uses no GPU.

C1 runs on exactly 400 unique rows selected by application-proportional largest remainder with minimum one per nonempty application when feasible. Within cells rank SHA-256 of `TILE|20260818|C1_SAMPLE|application|row_id`, taking smallest hashes. C1 uses the row's fold-selected N. For random window index k, derive seed from SHA-256 UTF-8 `TILE|20260818|C1_WINDOW|row_id|outer_fold|N|k`, first eight bytes unsigned big-endian, NumPy PCG64. Sample integer top-left uniformly and independently from `[0,W-1288] x [0,H-728]`. Duplicate random rectangles are retained, not resampled. C1 reruns N crop calls but uses the same rerun full-image output as Arm T, so total unique calls on a sampled row are `1+2N`.

Mandatory comparator rows are canonical B3, fold-local M1_ccm, frozen source-priority F1 majority, nested dev-selection over `[majority,A0,ours,A1,A2,A3,A4]`, and fold-local best single. No comparator selects N after Stage 0.

### Forward budget and full-image reproduction

Arm T makes `(N_f+1)` calls on every row in outer fold f. Total calls are `sum_f (N_f+1)*n_f`. For fixed-N examples, N=6 is 11,067 calls and N=11 is 18,972 calls. C1 adds `sum_sampled N_f` random-crop calls; its full-image output is shared with Arm T.

Because Stage 1 reruns full-image inference while baselines are frozen, the execution amendment must include a pre-endpoint reproduction audit. On every row, compare rerun Arm-T slot 0 with frozen GTA1 view 0 parsed point/output. Report exact point agreement and correctness agreement. Any correctness disagreement means the intervention includes runtime drift; primary TILE endpoints become `NOT_ADJUDICABLE_RUNTIME_DRIFT` until a separately committed audit resolves it. No output may be silently replaced by the frozen full slot.

Use GTA1-7B and the frozen H1 model, prompt, processor, greedy decoding, parser, coordinate transform, and historical runtime. All new traces retain token IDs, per-token logprobs or explicit unavailable reason, coordinate spans/logprob, sequence scores, entropy, top1-top2 margin, and hashes. Generation traces exclude labels and target boxes.

## Stage-1 endpoints

All intervals use 10,000 paired application-group bootstrap replicates and 99% percentile limits. N fitting is rerun inside bootstrap only for the Stage-0 predicted-policy uncertainty; Stage-1 primary inference uses the already frozen fold selections.

| ID | Endpoint | Frozen criterion |
| --- | --- | --- |
| T-P1 primary | Arm T minus full V-only N12, canonical B3, full benchmark | lower 99% bound strictly positive |
| T-P2 | Arm T minus full V-only N12, fold-local M1_ccm | lower 99% bound strictly positive |
| T-P3 | Arm T minus C2 equal-budget prefix, B3 and M1 separately | geometry claim requires lower 99% bound positive |
| T-P4 | Arm T minus C1 on frozen 400-row sample, B3 and M1 separately | regular-tiling claim requires lower 99% bound positive |
| T-P5 | observed repairs/damages on original-correct and crop-covered domains | descriptive, same table as T-P1 |
| T-P6 | pool budget, new calls, and difference from 12 forwards | descriptive, same table as accuracy |

T-P1 is always full-benchmark; low-coverage-only results cannot substitute.

## Kill conditions

| ID | Trigger | Action |
| --- | --- | --- |
| T-K1 | T-G1 | stop before GPU; publish Stage-0 curves/ledger as limitation |
| T-K2 | T-P1 CI includes zero or is negative | retain failed attempt; do not change N/layout |
| T-K3 | T-P3 indistinguishable | downgrade geometry claim; budget rather than geometry may explain result |
| T-K4 | T-P4 indistinguishable | downgrade regular-tiling claim; looking elsewhere may explain result |
| T-K5 | any fold selects N=4 or N=11 | optimum unresolved; never expand grid |
| T-K6 | B3 positive but M1 indistinguishable | classify as aggregator-absorbed under E1 precedent |
| T-K7 | Stage-1 observed net and Stage-0 expected net have opposite signs | eccentricity extrapolation failed; retain limitation and never refit/re-report curve |

After failure, no layout, N grid, curve bins, row composition, control, aggregator, or threshold may be changed and re-reported as TILE.

## Discipline, retention, and execution order

Commit this specification and `configs/tile_prereg.yaml` before any TILE statistic. Then commit a no-result preflight locking input hashes, 1,581 identities, folds, anchors, OWIN layouts, eligible-pair schema, contamination values, and code. Implement and commit Stage 0 before fitting curves.

Run all Stage-0 curves and four-part ledgers, then adjudicate T-G1/T-G2/T-K5 in `STAGE0.json` and `REPORT.md`. If authorized, commit fixed fold N values, layouts, C1 roster/windows, runner/tests, and a separate Stage-1 execution amendment before any GPU call.

Every raw JSONL is written one row at a time with write, flush, and fsync. Retain eccentricity pairs, split memberships, boundaries, curves, row scores, selected N, expected ledgers, tile/random coordinates, seeds, model traces, bootstrap seeds, failed attempts, bytes, and SHA-256. Raw and `predictions*.jsonl` cannot be recursively deleted. Independently verified backup is written under `/scratch/workspaceblobstore/tile/2026-08-18`; `STATUS.json` records manifest path and SHA-256.

Regardless of outcome, Stage-0 eccentricity curves and net ledger enter the paper as post-selection mechanism/limitation evidence. They do not by themselves prove that placement is more important than sampling; that wording requires Stage-1 T-P3/T-P4 support.