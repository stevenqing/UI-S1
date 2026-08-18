# TIGHT: tight-window relocalization self-consistency

Round: `tight`

Date: 2026-08-18

Status: `PREREGISTERED_POST_HOC_DIRECTION_AFTER_LOOK_BEFORE_ANY_TIGHT_RESULT`

GPU: Stage 0 is zero GPU and contains a pre-GPU stop. Stage 1 is not authorized by this specification and requires a separate committed execution amendment.

## Evidence status and disclosed direction selection

TIGHT is selected after observing LOOK's null control. LOOK main candidate confrontation has AUROC 0.540 `[0.458,0.633]`; the area-matched M1/random null has AUROC 0.726 and main-minus-null is -0.186 `[-0.323,-0.025]`. LOOK remains `L_D3` with `L_K1`; TIGHT does not revise, rescue, or replace that conclusion.

Promoting the former null geometry into a treatment is post-result direction selection. Every TIGHT output is labeled `POST_HOC_DIRECTION_SELECTION`, post-selection, and single-benchmark. It cannot be confirmatory.

LOOK's directional AUROC design was also underpowered for its frozen 0.60/0.65 three-way thresholds: only 180 recoverable rows were realized and the 99% interval width was about 0.175. This specification error is recorded in `docs/research_disclosures.md`. TIGHT does not reuse that AUROC as its primary endpoint.

Mind2Web is excluded. TIGHT changes no existing status.

## Mechanism and competing explanation

LOOK's main-window median area is 0.41% of image area. Its area-matched null required a median 236.5 random attempts and nevertheless discriminated better than main confrontation. The post-hoc hypothesis is that a high-resolution window centered near a point tests whether grounding returns to that point: local self-consistency.

The blocking alternative is center attraction: a model may output near any crop center regardless of the nominated candidate. Stage 1 C1 offsets the crop center from the candidate and estimates separate candidate-location and window-center coefficients. Failure of this control closes TIGHT without changing window size or offset.

## Frozen pool, blocks, and baseline answers

Use the frozen 1,581-row ScreenSpot-Pro C-uni pool, canonical view-major/lineage-minor order, five application folds, GRAN complete-link partition, fold-specific tau-star, and outer-development source reliability used by LOOK.

Order blocks by member count, summed reliability, maximum reliability, then earliest candidate index. Block rank is zero-based. Geometry point c is the arithmetic centroid of all block member points. Retain representative and member correctness separately.

For every row and k, the top-k block set contains ranks `0,...,k-1`. If fewer than k blocks exist, use all existing blocks and flag `FEWER_THAN_K_BLOCKS`; the row remains in Stage 0 but is infeasible for Stage-1 k when fewer blocks exist.

Canonical B3 and fold-local M1_ccm from the full 12-slot pool are frozen baselines. The TIGHT proposed answer is the parsed output from the top-k block crop with highest TIGHT score. It is one answer evaluated separately against B3 and M1 baselines; M1 does not define a second reranking rule.

## Stage 0: zero-GPU bounds and complements

### Top-k block oracles

For k=1 through 5 report two row-level quantities:

1. contains-correct coverage: at least one member candidate in the top-k blocks is correct;
2. fixed-output oracle accuracy: at least one frozen block representative in the top-k blocks is correct.

The first is a membership ceiling. The second is the attainable ceiling for a perfect selector over frozen block outputs and is the quantity compared with canonical B3 for G-G1. Do not mix the two.

Report full-benchmark values, fold values/ranges, and gains versus B3, M1_ccm, and nested dev-selection. Reproduce EVID's all-block fixed-output oracle 78.56% as an integrity anchor; the top-k curve decomposes rather than replaces it.

The k selection grid is `{2,3,4}`. For outer fold f, select k by maximum inner-validation fixed-output oracle accuracy, ties to smaller k, using modes/reliability fit without outer-test labels. Refit reliability on all outer-development folds and evaluate the selected k once on outer test. Any selected k=2 or k=4 triggers G-K5; never expand the grid.

### LOOK complements

Report LOOK null AUROC and its already frozen 99% interval without recomputation. From LOOK's committed private preparation rows report, for the selected null point relative to M1 centroid:

- Euclidean distance in pixels;
- distance divided by image diagonal;
- distance divided by main-window short edge;
- polar angle `atan2(dy,dx)` in `[-pi,pi]`;
- four quadrant counts with zero assigned to the nonnegative side;
- distributions of selected attempt and null/main area ratio.

These are descriptive complements and cannot select TIGHT settings.

### Damage-domain locks

Freeze exact row-ID sets and SHA-256 for canonical B3 original-correct 1,007 rows and existing crop-center-covered 1,356 rows. These domains overlap but are not interchangeable.

### G-G1

G-G1 uses k=3 fixed-output oracle gain versus canonical B3 over all 1,581 rows. If gain is strictly below 0.007, trigger G-K1 and stop before GPU. Equality passes.

Stage 0 itself authorizes no GPU. If G-G1 passes, commit the complete Stage-0 output, selected k by fold, exact windows/controls, runner/tests, and a separate execution amendment.

## Global tight-window size

The window size is derived once from ScreenSpot-Pro GT statistics and is therefore evaluation-derived, benchmark-specific, and post-hoc. It is not a cross-benchmark deployment constant.

For all 1,581 valid target boxes, define short edge `min(max(0,x2-x1),max(0,y2-y1))`. Invalid or zero short edges are reported and excluded from the median. Use NumPy median over the remaining population.

Let `S=ceil(8*median_short_edge)`. Freeze crop height `Hc=max(1,S)` and width `Wc=max(1,ceil(Hc*1288/728))`. No fold-specific size, grid, or alternative multiplier exists. If Wc or Hc exceeds an image, that row/window is `INFEASIBLE_TOO_LARGE`.

For a center p, initial top-left is `(floor(px-Wc/2),floor(py-Hc/2))`. Apply minimum axis-wise integer translation to fit the full rectangle inside the image. Never resize the committed full-image rectangle. Record requested center, initial/final rectangle, translation, achieved center, dimensions, area fraction, and processor resize dimensions.

## Stage-1 primary score and reranking

For each top-k block, construct one tight window centered at block centroid c and run GTA1-7B once. Map output o to full-image pixels. Score

$$
s_c=-\|o-c\|_2/H_c.
$$

Unparsable/nonfinite output receives score negative infinity and cannot win unless every block is unmappable; in that case the proposed answer is incorrect and flagged `ALL_UNMAPPABLE`. Select highest `(score,-block_rank)`, so ties choose earlier block rank. The proposed answer is the selected tight-window model output, not the centroid or frozen representative.

There is no score threshold, threshold grid, abstention, or gate. k alone is selected in Stage 0 and frozen by outer fold before Stage 1.

Coordinate-token logprob, entropy, and top1-top2 margin are preregistered secondary scores. Report their candidate-level AUROC and relation to correctness separately. They never select a block or replace primary score.

## C1: offset-window blocking control

C1 uses exactly one offset forward per sampled `(row,block)`, not four. Use the same Wc/Hc. Nominal offset radius is `0.30*Hc`.

Direction index is `(stable_row_index + block_rank) mod 4`, with directions in order right, down, left, up. Round each offset component half-away-from-zero. Requested window center is c plus that offset; then apply the same image-bound translation. Record nominal and achieved offset. No alternate direction is tried.

Sample exactly 500 unique rows from the Stage-1-feasible population by application-proportional largest remainder with minimum one per nonempty application when feasible. Rank within application by SHA-256 UTF-8 `TIGHT|20260818|CONTROL_SAMPLE|application|row_id`. If fewer than 500 feasible rows exist, include all and report the count.

For every mapped C1 output, create two stacked coordinate observations, x and y. Normalize x by image width and y by image height. Define normalized output O, candidate centroid C, and achieved window center W. Fit weighted OLS

$$
O=alpha_{axis}+beta_c C+beta_w W+epsilon,
$$

with separate x/y intercepts, no interactions, and inverse-inclusion row weights. Bootstrap applications 10,000 times and refit. G-P3 passes only when both 99% lower bounds for `beta_c` and `beta_c-beta_w` are strictly positive. This means candidate location contributes after window center and window center does not dominate. Undefined/singular fits fail G-P3.

## C2: distance-matched random-location control

For each sampled `(row,block)`, match the achieved C1 offset radius `r=||W-C||`. For attempt j `0,...,9999`, seed PCG64 from first eight SHA-256 bytes of UTF-8

`TIGHT|20260818|C2|row_id|block_rank|j`.

Sample angle uniformly on `[0,2*pi)`, set random point `q=round_half_away(C+r*(cos,sin))`, and require:

- q lies in the image;
- q is strictly farther than 14 pixels from every C-uni candidate point;
- a tight window centered at q is feasible under the same translation rule;
- achieved window-center distance from C differs from r by at most one pixel.

Select first valid attempt; otherwise mark control infeasible and retain all attempts. Run one C2 forward per valid `(row,block)`. C2 score is negative output-to-q distance divided by Hc. Compare primary candidate score minus C2 score as weighted paired candidate records; G-P6 requires 99% lower bound strictly positive.

## Stage-1 budget and execution boundary

For each row, run one primary forward per top-k block selected for its fold. Calls are `sum_f k_f*n_f`, reduced only by `FEWER_THAN_K_BLOCKS` infeasibility and recorded exactly.

C1 and C2 each run one forward per selected top-k block on the same at-most-500-row control sample. If k=3 for all rows, primary is 4,743 calls and each control is 1,500 calls, total 7,743. Four C1 directions are not four calls.

The budget is additive, not equal-budget. Report total new calls, calls per row/fold, and pp gained per 1,000 calls. DECOMP's near-zero proposer-side marginal after saturation is contextual only, not a control.

Use frozen GTA1-7B model/revision, historical H1 runtime, grounding prompt, processor, greedy decoding, parser, and coordinate transform. Every new trace retains token IDs, logprobs/unavailable reason, coordinate spans/logprob, sequence score, entropy, margin, settings, and hashes. Generation traces exclude labels, correctness, domains, and target boxes.

Stage 1 requires a new amendment binding selected k, all windows/seeds/control eligibility, actual calls, runner/evaluator hashes, model/index/prompt/parser, hardware/shards, smoke, failures, and one-time nonce. This specification does not authorize it.

## Stage-1 endpoints

All accuracy contrasts use full-benchmark paired row outputs with 10,000 application-group bootstrap replicates and 99% percentile limits. Stage-0 selected k is not reselected after GPU output.

| ID | Endpoint | Frozen criterion |
| --- | --- | --- |
| G-P1 primary | TIGHT proposed answer minus full-N12 canonical B3 accuracy | lower 99% bound strictly positive |
| G-P2 | same proposed answer minus fold-local full-N12 M1_ccm | lower 99% bound strictly positive |
| G-P3 blocking | C1 weighted OLS beta-c and beta-c-minus-beta-w | both lower 99% bounds strictly positive |
| G-P4 | observed repair/damage on original-correct 1,007 and crop-covered 1,356 domains | descriptive, same table as G-P1 |
| G-P5 | recoverable rows, candidate-level AUROC of primary score against block correctness | descriptive and explicitly underpowered |
| G-P6 | primary score minus C2 random-location score on matched candidate/control pairs | lower 99% bound strictly positive |
| G-P7 | new forwards and pp per 1,000 calls | descriptive |

G-P4 defines repair as baseline wrong/TIGHT correct and damage as baseline correct/TIGHT wrong. G-K7 triggers when total observed damage count exceeds total observed repair count, equivalently the B3 point net is negative. Report both damage domains separately; crop-covered is not a correctness domain.

The fixed endpoint order is G-P3, G-P6, G-P1, G-P2, G-P4, G-P5, G-P7. If G-P3 fails, compute/retain already authorized outputs but issue no self-consistency or method claim.

## Kill conditions

| ID | Trigger | Action |
| --- | --- | --- |
| G-K1 | G-G1 | stop before GPU; Stage 0 enters limitations |
| G-K2 | G-P3 fails or beta-w dominates | center-attraction explanation; no size retry |
| G-K3 | G-P6 lower bound not positive or endpoint undefined | any position appears self-consistent; cancel claim |
| G-K4 | G-P1 lower bound not positive | retain failed attempt |
| G-K5 | any fold selects k=2 or k=4 | optimum unresolved; never expand grid |
| G-K6 | B3 positive but M1 lower bound not positive | aggregator-absorbed |
| G-K7 | observed damage exceeds observed repair | negative net; no post-hoc gate rescue |

After failure, no window size, offset, direction, k grid, score, C2 matching, or endpoint may be changed and re-reported as TIGHT.

## Stage-0 complements, discipline, and retention

Stage 0 additionally freezes original-correct 1,007 and crop-covered 1,356 row-ID sets and hashes. The five leaked ScreenSpot-Pro fold values are contamination anchors only and cannot enter fitting, thresholds, sampling, or implementation checks. Historical 63.88% is M1_ccm; 85.77% is crop-center coverage, not accuracy.

Commit this specification and `configs/tight_prereg.yaml` before any TIGHT statistic. Then commit a no-result preflight locking all inputs, tau/modes, row sets, global median/size, LOOK complements, model/runtime, and code. Implement and commit Stage 0 before calculating its curves/gates.

Every JSONL uses per-row write, flush, and fsync. Retain modes, top-k memberships/oracles, selected k, row domains, tight/C1/C2 windows, directions/seeds/attempts, sample/IPW, traces, regressions, AUROC records, bootstrap seeds, failures, bytes, and SHA-256. Raw and prediction JSONLs cannot be recursively deleted. Independently verified backup is written under `/scratch/workspaceblobstore/tight/2026-08-18`; STATUS records manifest path and hash.

Regardless of outcome, top-k oracle curves and LOOK distance/orientation complements enter the paper as post-selection mechanism/limitation evidence. They do not revise LOOK.