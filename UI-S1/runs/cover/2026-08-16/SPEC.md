# COVER: proposer coverage headroom and cross-benchmark dependence

Round: `cover`

Date: 2026-08-16

Status: `PREREGISTERED_BEFORE_ANY_COVER_RESULT`

GPU: zero. Both arms are deterministic recomputations over frozen artifacts.

## Scope and evidence status

COVER is diagnostic only. It evaluates no method, makes no method claim, and changes none of F1, Q1, `TRIVUS_NOT_PROMOTED`, VUS-SR, SPLIT, MASK, CEIL, OTEXT, XSCR, DECOMP, EVID, ICC, or XSOFT.

ScreenSpot-Pro and Mind2Web labels have already been used. COVER is post-selection and evaluation-side. Label-free execution of some Arm A quantities does not restore confirmation on current rows. Independent confirmation requires new untouched data.

## Arm A: ScreenSpot-Pro crop coverage headroom

### Geometry correction

The frozen ScreenSpot-Pro N12 proposer manifest has view 0 equal to the full image and views 1-11 equal to GTA1 attention-ranked crop regions. All three model lineages share the same region at a given view. Therefore:

- including view 0 would make union area 100% and target coverage count zero impossible;
- repeating regions over three lineages would not measure lineage spatial diversity.

Arm A uses only the 11 unique crop ranks, views 1-11. View 0 is reported separately as the full-image baseline. Arm A measures shared proposer-rank geometry, not cross-lineage window diversity.

The crop rectangles are integer pixel boxes `[left, top, right, bottom)` clipped to the image. For each row create a uint8 coverage-count map with values 0-11. Every map is written as a compressed PNG preserving exact integer counts. Per-map bytes and SHA-256 enter the raw map manifest and retention set.

### A1: area distributions

For each row report:

- common intersection area: pixels with count 11 divided by image pixels;
- crop union area: pixels with count at least 1 divided by image pixels;
- uncovered area: pixels with count 0 divided by image pixels;
- coverage-count histogram for 0-11.

Summaries report linear-interpolation quartiles, median, mean, minimum, and maximum. View 0's full-image union/intersection ratio is separately recorded as 1.0 and never mixed into crop-only summaries.

### A2: target-center coverage strata

The target center is `((x1+x2)/2,(y1+y2)/2)` and is mapped to the containing pixel after clipping. Classify each row by crop-only coverage count:

- `common_11`: count 11;
- `partial_1_10`: count 1-10;
- `uncovered_0`: count 0.

Report counts and row fractions. This and subsequent target-dependent quantities are evaluation-side.

### A3: conditional B3 accuracy

Use canonical C-uni B3 correctness and report accuracy in the three coverage strata. The low-coverage group is `partial_1_10 + uncovered_0`. Report common-minus-low accuracy point difference and 10,000 application-group bootstrap 99% percentile CI.

### A4: candidate-success cross-table

Use C-uni base rows only:

- `selected_correct`: canonical B3 representative is correct;
- `recoverable`: B3 representative is wrong and at least one of the 12 candidates is correct;
- `zero_candidate_success_coverage`: all 12 candidates are wrong.

Cross-tab these three row classes against the three spatial target-center coverage strata. This is a base-row analog of CEIL recoverability; it does not use CEIL's 968 arm-expanded sample keys. `zero_candidate_success_coverage` is not called zero spatial coverage.

### Frozen indicators

- A-G1: `common_11` row fraction at least 0.90 indicates the crops effectively share one target region and recommends closing the complementary-window direction.
- A-G2: low-coverage row fraction below 0.10 recommends closure for insufficient base rate.
- A-G3: low-coverage fraction at least 0.10 and common-minus-low B3 accuracy at least 0.007 recommends writing a separate complementary-window specification.

The 99% CI is co-reported but not an automatic gate. A human final decision records the rationale.

### Follow-up boundary

A future complementary-window pilot would minimize overlap, opposite to the cancelled X2 containment-maximizing objective. It is not an X2 revival. It must require a new candidate's failure phi with the original pool to be materially below 0.672, add candidates directly to the pool, avoid mode-flip verification, and compute a full net-benefit/base-rate budget before GPU authorization. It may not repeat SPLIT's two-mode restriction or verifier framing.

## Arm B: Mind2Web cross-benchmark dependence

### Feasibility gate

The frozen Mind2Web C-uni bank must contain exactly 2,080 rows and 12 candidates per row with success labels, fold IDs, episode groups, model identity, and slot roles. The slots are not a ScreenSpot-Pro-equivalent aligned view grid:

- per model: stage-1 full, stage-1 view1 crop, stage-2 crop0, stage-2 crop1;
- models: TongUI-7B, CogAgent-18B, UI-TARS-7B.

If any identity, width, role, success, or fold anchor fails, Arm B records `BLOCKED_INPUT_UNAVAILABLE` and stops.

### B1: direct dependence

Use exactly ICC Arm C's estimator on binary candidate failures. For each outer fold estimate on the four outer-development folds:

- within-model phi: all 18 same-model slot pairs;
- cross-model phi: all 48 different-model slot pairs.

Pairs receive equal weight. Constant pairs are excluded and counted; undefined-as-zero sensitivity and Cohen-kappa zero-fill values are also reported. Report five fold values, unweighted mean, full range, and 10,000 episode-group-within-fold bootstrap 99% intervals.

Compute the empirical 12x12 phi-matrix

$$
N_{\mathrm{eff},\phi}=\frac{144}{\mathbf1^TR_\phi\mathbf1},
$$

with undefined off-diagonal pairs set to zero and held-out-row-count weighted fold aggregation. Report beside ScreenSpot-Pro phi 0.672/0.577 and $N_{\mathrm{eff}}=1.5726$. AndroidControl 0.895/0.398 is external reference only.

### B2: source/stage distance trend

M2W has no identifiable same-family model-scale axis; model size is confounded with family, architecture, and training. The frozen three-point distance is instead:

1. `within_model_cross_slot`: the 18 within-model pairs;
2. `cross_model_matched_role`: 12 cross-model pairs sharing one of the four slot roles;
3. `cross_model_unmatched_role`: 36 cross-model pairs with different slot roles.

Report each benchmark's applicable ordered points. For ScreenSpot-Pro, use ICC's within-lineage, cross-lineage matched-view, and cross-lineage unmatched-view pair means from the same frozen C-uni bank. For M2W, use the three source/stage strata above. This is a source/stage dependence trend, not an architecture-distance or model-scale law. Do not fit or extrapolate.

### Interpretation

If M2W cross-model dependence remains close to within-model dependence, report high GUI-grounding error dependence across both benchmarks. If it is materially lower, report benchmark-specific dependence and relate it descriptively to prior SSPro/M2W splits. No automatic threshold or method authorization follows.

## Discipline and retention

The five leaked ScreenSpot-Pro aggregate cells are prohibited as targets or thresholds. Arm A A3/A4 and all Arm B results are evaluation-side and cannot define a runtime rule. Historical 63.88% means `M1_ccm`, not source-priority majority.

Commit this spec and `configs/cover_prereg.yaml` before implementation or result. Then run Arm B feasibility, Arm A geometry/A1/A2, Arm A A3/A4 and Arm B dependence, and finally record human interpretation.

Raw JSONL uses write/flush/fsync. Coverage-count maps are exact uint8 PNGs. Every map, intermediate, input, result, and failed attempt receives SHA-256 metadata and independent retention under `/scratch/workspaceblobstore/cover/2026-08-16`.