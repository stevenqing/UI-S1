# CWIN: complementary-window equal-budget replacement

Round: `cwin`

Date: 2026-08-17

Status: `PREREGISTERED_AFTER_P0_BEFORE_ANY_CWIN_RESULT`

GPU: Stage 0 is zero GPU. Stage 1 is not authorized by this specification.

## Scope and evidence status

CWIN is a single-benchmark post-selection exploratory pilot. New candidates would be generated for already-used ScreenSpot-Pro rows and labels; no result can be confirmatory.

CWIN does not revive X2. X2 used an uncertainty gate and containment-seeking adaptive zoom. CWIN constructs fixed-size windows that maximize coverage of area left uncovered by the 11 frozen crop ranks. CWIN does not revive SPLIT: there is no two-mode restriction, verifier, answer flip, or target-conditioned runtime gate. New outputs enter the candidate pool directly.

CWIN changes none of F1, Q1, `TRIVUS_NOT_PROMOTED`, VUS-SR, SPLIT, MASK, CEIL, OTEXT, XSCR, DECOMP, EVID, ICC, COVER, or XSOFT.

## P0 authority

`CONTAINMENT_RECONCILIATION.md` is normative. Historical rank 0 is the full-image view; COVER's uncovered fraction excludes it. E3 is interpreted as full-image-to-late-crop decay. Stage 1a uses GTA1 V-only N12 and Stage 1b would use full-36; mixed C-uni N12 is not the replacement budget.

No L1-L4 computation may start unless a preflight reproduces P0 hashes, 1,581 row identities, all region sizes, and candidate-bank anchors.

## Frozen geometry

### Existing windows

View 0 is the full image and is never replaced or included in crop redundancy. Views 1-11 are crop windows. Every row must expose 11 valid integer rectangles with identical width and height after clipping. If any row violates this, it is a geometry-audit failure.

### Complementary windows

For each row, begin with the binary union coverage of views 1-11. For each new window in sequence:

1. enumerate every legal integer top-left coordinate whose window stays inside the image;
2. use an exact integral-image rectangle sum to compute newly covered pixels whose current coverage count is zero;
3. select the maximum newly covered count;
4. tie-break by smaller top coordinate, then smaller left coordinate;
5. add the selected rectangle to the coverage map before placing the next window.

The new rectangle width, height, area, aspect ratio, and model-preprocessor resize dimensions must exactly match the existing crops. Because existing regions are nominally identical-sized, the audit thresholds area ratio `[0.95,1.05]` and aspect-ratio difference `<=0.03` are reported but exact equality is required by construction. No stride or approximate search is allowed.

### Dropped windows

Starting from views 1-11, greedily remove $K$ crops. At each step, score every remaining crop by the sum of its pixel rectangle IoU with every other remaining crop. Remove the highest score; tie-break by earlier original rank. Recompute scores after every removal. View 0 is immutable.

### K grid and nested selection

$K\in\{2,3,4\}$. Use application GroupKFold(5). For outer fold $f$, inner validation is $(f+1)\bmod5$, inner train is the remaining three folds, outer development is all four non-test folds, and outer test is fold $f$.

Select $K$ on inner validation by maximum strict oracle net-gain upper bound L4-upper. Ties choose smaller $K$. Apply the selected $K$ once to the outer test fold. If any fold selects $K=2$ or $K=4$, W-K5 triggers; the grid is not expanded.

Geometry is label-free and may be precomputed for all K before selection. Selection and L1-L4 use evaluation-side labels.

## Stage 0 endpoints

### L1: geometric target coverage

For every K report, before and after replacement:

- target-center crop coverage count;
- rows moving from count 0 to positive;
- rows moving from partial to higher count;
- rows losing all target-center crop coverage;
- target-center coverage-count transition matrix.

All are evaluation-side diagnostics.

### L2: conservative projection

On each outer fold, estimate on outer-development rows the canonical B3 accuracy for every existing target-center crop coverage count 0-11. For each count, compute an application-group bootstrap 99% lower confidence bound with 10,000 replicates. Empty or non-finite count strata receive lower bound zero.

For an outer-test row newly covered after replacement, map its post-replacement coverage count to this frozen lower bound. L2-conservative is the full-benchmark mean projected correctness over rows that are wrong under the drop-only pool and newly target-covered. This is a conservative projection, not an upper bound and not a method result.

### L3: drop-only damage truth

For each K, evaluate the GTA1 V-only N12 pool after removing the selected K redundant crop candidates and adding nothing. Use canonical B3 and fold-local M1_ccm, separately. Report full-benchmark accuracy change versus the original GTA1 V-only N12 pool. L3 is the directly observed drop-only gain/loss.

### L4: strict oracle upper bound and conservative projection

For each row under the drop-only B3 pool, define an oracle rescue opportunity when:

- drop-only B3 is wrong; and
- at least one selected complementary window contains the target center.

Assume every opportunity is perfectly rescued and no other row changes. Then

$$
L4_{upper}=\Delta_{drop-only,B3}+P(\text{drop-only wrong and new-window covered}).
$$

The conservative projected net is

$$
L4_{conservative}=\Delta_{drop-only,B3}+L2_{conservative}.
$$

W-G1 stops before Stage 1 when the nested outer-test micro-average $L4_{upper}<0.007$. This is the only Stage-0 automatic stop. L4-conservative is co-reported but cannot prove an upper-bound stop.

## Stage 1 authorization boundary

This specification does not authorize GPU inference. If W-G1 passes, commit all selected K values and every new/drop window coordinate before a separate Stage-1 execution amendment.

### Stage 1a proposed scope

GTA1 V-only N12 only. Arm R replaces K of 11 crops, preserving one full-image plus 11 crop candidates and 12 forwards. Arm A retains all 11 crops and adds K complementary candidates. C2 is drop-only. C1 uses K random legal windows whose top-left coordinates are sampled uniformly from positions with positive previously-uncovered pixel gain, using committed SHA-256 row/K/window seeds. A matched-area high-overlap control is also mandatory.

The call count is $K\times1,581$ new complementary forwards plus control calls specified in the execution amendment. K=3 implies 4,743 complementary forwards.

### Stage 1b proposed scope

Only after Stage 1a W-P1 passes may a new amendment request the other two lineages. Full-36 replaces the same K geometry ranks independently in each lineage, preserving 36 forwards. Additional complementary calls are $2K\times1,581$; K=3 implies 9,486.

## Stage 1 endpoints and controls

Primary W-P1 is Arm R minus original same-budget pool on full-benchmark accuracy with application-group paired bootstrap, 10,000 replicates, 99% CI lower bound greater than zero. W-P2 compares Arm R against majority, nested dev-selection, and best-single on the same rows. W-P3 stratifies gain/harm by COVER coverage count. W-P4 compares budget-increased Arm A against Arm R. W-P5 reports the new candidate's failure phi against every original source and requires mean phi below both 0.672 and 0.577.

Report canonical B3 and M1_ccm separately. Historical 63.88% means M1_ccm, not source-priority majority. Mixed C-uni baselines are contextual and not substituted for the V-only/full-36 same-budget primary pool.

Kill conditions retain W-K1 through W-K7 with these clarified meanings:

- W-K1: W-G1 upper bound below 0.70 pp;
- W-K2: W-P1 CI lower bound not positive;
- W-K3: random-window control indistinguishable from Arm R;
- W-K4: B3 positive but M1_ccm not distinguishable;
- W-K5: selected K is 2 or 4;
- W-K6: mean new-candidate phi fails either 0.672 or 0.577 bound;
- W-K7: any geometry mismatch rate exceeds 10%; implementation must be repaired before result adjudication.

No geometry, K grid, redundancy rule, control, or aggregator may be changed after failure.

## Trace retention

Any future forward must comply with `docs/generation_trace_retention_policy.md` and additionally retain coordinate-token entropy and top-1 minus top-2 probability margin. Raw/prediction files cannot be recursively deleted.

## Execution order

1. Commit P0, this spec, and `configs/cwin_prereg.yaml`.
2. Commit no-result preflight and Stage-0 implementation.
3. Run nested L1-L4 and W-G1.
4. If W-G1 fails, finalize without GPU.
5. If W-G1 passes, commit selected K and window manifests, then write a separate Stage-1 amendment. GPU remains prohibited until that amendment and explicit authorization.