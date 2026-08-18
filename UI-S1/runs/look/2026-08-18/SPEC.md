# LOOK: candidate-confrontation local relocalization diagnostic

Round: `look`

Date: 2026-08-18

Status: `PREREGISTERED_BEFORE_ANY_LOOK_RESULT`

GPU: requested maximum 1,500 formal forwards. This specification does not authorize GPU execution. A separate committed amendment must bind the actual eligible sample, windows, runner, hashes, hardware, call count, and one-time nonce.

## Nature and scope

LOOK is a diagnostic, not a method. It asks whether one high-resolution crop containing two explicit candidate modes supplies candidate-discrimination signal when the existing pool selects incorrectly. LOOK never applies a runtime answer flip, gate, verifier, or policy.

A positive diagnostic authorizes only a new specification. That future study must preregister a full net-benefit ledger before inference. LOOK itself produces no deployable rule.

All ScreenSpot-Pro labels have been repeatedly used. LOOK is post-selection and single-benchmark. Mind2Web is excluded. LOOK changes no status for F1, Q1, `TRIVUS_NOT_PROMOTED`, VUS-SR, SPLIT, MASK, CEIL, OTEXT, XSCR, DECOMP, EVID, ICC, COVER, CWIN, PRUNE, OWIN, TILE, or XSOFT.

## Why prior failures do not resolve LOOK

TriVUS, CEIL, EVID, and OTEXT derive signals from retained candidates, their relations, frozen visual logits, or text. They do not rerun the model on one crop that simultaneously presents two candidate locations.

SPLIT constructs a crop containing one mode while excluding another, reads confidence, and evaluates a flip policy. LOOK instead contains both modes, retains the ordinary grounding prompt, and measures candidate-level discrimination. It has no rare-event gate, answer flip, verifier, or two-mode policy restriction.

X2 optimizes adaptive containment. LOOK uses a deterministic rectangle induced only by frozen mode centroids and does not optimize target containment.

The mechanism hypothesis is limited: as mode separation grows, a crop containing both modes approaches a full-image view and should provide less new local evidence. LOOK reports separation-stratified endpoints rather than assuming this relation.

## Frozen pool, folds, and modes

The pool is the 1,581-row ScreenSpot-Pro C-uni bank in canonical view-major, lineage-minor order: views 0 through 3, each with GTA1-7B, Qwen3-VL-8B-Instruct, and UI-TARS-7B-SFT.

Use the existing five application-group folds. For row in outer fold f, use GRAN's already selected fold-f tau-star:

`[0.0022908676527677724, 0.0015135612484362087, 0.012022644346174132, 0.0034673685045253167, 0.0034673685045253167]`.

Tau is image-diagonal-normalized Euclidean distance. Reconstruct deterministic complete-link blocks with the frozen GRAN implementation. Source reliability is fit on the four non-f outer-development folds only.

Order modes by:

1. larger member count;
2. larger sum of member source reliabilities;
3. larger maximum member reliability;
4. earlier minimum canonical candidate index.

Mode rank defines M1, M2, and M3. Mode centroid is the arithmetic mean of member points in original pixel coordinates. Mode correctness is true when any member point lies in the GT bbox. Representative correctness is separately retained and never substitutes for mode correctness.

Rows with fewer than three modes are ineligible for the formal sample. No synthetic mode, alternate tau, or changed clustering is allowed.

Canonical B3 uses the frozen complete-link/coverage implementation and anchor 1,007/1,581. `recoverable` means canonical B3 is wrong and at least one of all 12 C-uni candidates is correct. `pool_correct` means canonical B3 is correct. These two sampling strata are disjoint.

## Deterministic confrontation geometry

The target aspect ratio is `1288/728`. Geometry uses mode centroids only and never GT.

For a set of mode centroids, begin with their smallest continuous axis-aligned bounding rectangle. Let `dx=max_x-min_x`, `dy=max_y-min_y`, and

$$
s=\max(1,\min(\max(dx,1),\max(dy,1))).
$$

Pad every side by `0.25*s`. Then expand the shorter dimension symmetrically, without shrinking the longer dimension, until width/height equals exactly `1288/728`. Convert to integer half-open bounds with floor on left/top and ceil on right/bottom.

If width exceeds image width or height exceeds image height, the window is `INFEASIBLE_TOO_LARGE`; do not shrink it. Otherwise translate each axis by the smallest integer amount that places the complete rectangle inside the image; ties choose the smaller final top-left. Translation never changes dimensions. Verify every requested centroid remains inside the final half-open window.

The main window contains M1 and M2. The sensitivity window contains M1, M2, and M3. Record input centroids, raw bounds, pad, aspect expansion, integer bounds, translation, final rectangle, dimensions, area fraction, and processor `smart_resize` output dimensions. Processor resizing does not alter the committed full-image rectangle.

## Null control

The null window contains M1 and one random noncandidate pixel. A random pixel is noncandidate when its Euclidean distance from every one of the 12 C-uni candidate points is strictly greater than 14 pixels.

For attempt k in `0,...,9999`, derive a seed from SHA-256 UTF-8

`LOOK|20260818|NULL|row_id|k`

using the first eight bytes as unsigned big-endian and NumPy PCG64. Sample x uniformly from integer `[0,W-1]` and y from `[0,H-1]`. Construct the M1/random window with the exact confrontation geometry above. It is valid when feasible and its final pixel area divided by the main-window final pixel area lies in `[0.9,1.1]` inclusive. Select the first valid attempt. Earlier failures, selected k, seed, point, distances, window, and area ratio are retained.

If no attempt is valid, the row is ineligible. The random pseudo-mode is correct exactly when the random pixel lies in the GT bbox; this label is evaluation-side and never enters generation traces.

## Sampling and call budget

The eligible frame requires:

- membership in recoverable or pool-correct stratum;
- at least three frozen modes;
- feasible main and three-mode windows;
- a valid null window within 10,000 attempts;
- valid image and frozen candidate identities.

Target 250 unique recoverable rows and 250 unique pool-correct rows. Within each stratum allocate by application-proportional largest remainder, assigning one row per nonempty application first when feasible. Ties use application string. Within each cell rank SHA-256 UTF-8 `LOOK|20260818|SAMPLE|stratum|application|row_id`, taking smallest hashes. Record population, allocation, inclusion probability, inverse-probability weight, and hash.

If an eligible stratum has fewer than 250 rows, include all. If either realized stratum has fewer than 150 rows, trigger L-K3 and produce an observational report without L-D1/L-D2/L-D3 adjudication.

Each sampled row receives exactly three forwards: main M1/M2, M1/M2/M3 sensitivity, and null M1/random. Maximum sample 500 and maximum formal budget 1,500 calls. The execution amendment binds actual eligible rows and exact calls; no replacement after output is allowed.

## Model and output mapping

Use GTA1-7B revision `701bedc80b447863bd60e3318ae44f6cbbfafd78`, official source revision `988ff3c61b9f7632d780ae27c83260de75b3c95f`, and the frozen historical H1 runtime, prompt bytes, processor bounds, crop-resize semantics, greedy decoding, max tokens, parser, and coordinate transform.

Do not introduce a comparison question. Each crop uses the ordinary grounding prompt. Map parsed crop-local coordinates back to full-image coordinates using the committed window offset.

For main output, compute Euclidean distance to M1 and M2 centroids. The selected mode is minimum `(distance,mode_rank)`, so exact ties select M1. For sensitivity use M1/M2/M3 and the same tie rule. For null use M1/random, with exact ties selecting M1. Unparsable or nonfinite coordinates are `UNMAPPABLE`; they do not receive a selected mode.

For AUROC records, main emits two candidate records per row: score is negative full-image Euclidean distance divided by image diagonal, and label is the corresponding mode correctness. Null emits M1 and random pseudo-mode records with the same score and their respective correctness labels. This is candidate-level AUROC, matching CEIL's metric family; its row/context construction differs from CEIL, so CEIL 0.540 is contextual and not a reproduced baseline.

Every trace retains token IDs, per-token logprobs or explicit unavailable reason, coordinate-token spans/logprob, sequence scores, entropy, top1-top2 margin, decoded output, parser status, hashes, model/backend, and generation settings. Generation traces contain no target bbox, mode correctness, random correctness, stratum, reward, or label.

## Endpoints

All sample estimates use inverse-inclusion weights. Bootstrap resamples applications with replacement and retains every sampled row and both candidate records from each selected application. Use 10,000 replicates and 99% percentile limits. At least 9,900 finite replicates are required. The resampling unit is application, never an individual row or candidate.

### L-P1 primary

On recoverable rows, candidate-level weighted AUROC of main scores against M1/M2 mode-correctness labels. Report valid positive/negative records, rows, applications, point AUROC, fold values, and 99% interval. If either class is absent, L-P1 is undefined and no three-way adjudication is issued.

### L-P2

On recoverable rows, main-selected mode correctness minus M1 correctness. Unmappable rows count main correctness false. Report each rate, paired difference, and grouped-bootstrap 99% interval. M1 is the pool's frozen top-ranked mode; it is not canonical B3.

### L-P3

On pool-correct rows, map the canonical B3-selected candidate to its frozen mode. Report:

- confrontation overturn rate: selected main mode differs from B3 mode;
- harmful overturn rate: main-selected mode is incorrect;
- unmappable rate.

These are descriptive diagnostics, not a runtime policy.

### L-P4 null identity control

On recoverable rows, compare main candidate-level AUROC with null M1/random candidate-level AUROC. Compute paired application bootstrap of AUROC difference `main-null`. A strictly positive 99% lower bound supports candidate-identity specificity. Otherwise trigger L-K1. Null and main use the same sampled rows and weights.

### L-P5 separation

Before sampling, compute M1-M2 centroid distance divided by image diagonal over the full eligible recoverable frame. Freeze numeric quartile boundaries using NumPy `quantile(method="linear")`; map sampled rows with `bisect_right`. Report L-P1 and L-P2 in all four distance bins, including class counts and undefined intervals. No distance bin selects a method or threshold.

### L-P6 geometry

Report main and three-mode window area fraction distributions, fraction above 0.8, infeasible-frame counts, null attempts, null area ratios, and relation to M1-M2 normalized separation. L-K2 uses the sampled main-window area fraction only.

Sensitivity additionally reports three-mode selected-mode correctness and candidate-level three-mode AUROC on both strata. It has no gate and cannot replace the primary.

## Frozen interpretation

If either realized stratum has fewer than 150 rows, issue `L_K3_OBSERVATIONAL_NO_DIRECTIONAL_ADJUDICATION`.

Otherwise:

- L-D1: L-P1 99% upper bound strictly below 0.60;
- L-D2: L-P1 99% lower bound strictly above 0.65;
- L-D3: neither condition.

Boundaries equal to 0.60 or 0.65 fall into L-D3. L-D2 authorizes only a new specification with a preregistered net-benefit ledger. L-D1 closes this offline block-selection direction into limitations. L-D3 supports neither direction.

If L-P4 lower bound is not strictly positive, trigger L-K1 and prohibit candidate-identity signal wording regardless of L-D status.

## Kill conditions

| ID | Trigger | Action |
| --- | --- | --- |
| L-K1 | main-minus-null AUROC lower 99% bound not positive or endpoint undefined | candidate-identity claim canceled; downgrade entire conclusion |
| L-K2 | more than half sampled main windows have area fraction greater than 0.8 | confrontation effectively full-image; mechanism claim fails |
| L-K3 | either realized stratum below 150 | observational report; no L-D adjudication |
| L-K4 | unmappable main-output fraction above 0.10 in either stratum | mapping implementation invalid; stop adjudication and repair before any rerun |
| L-K5 | formal failure rate above 0.01 for non-infrastructure cause | stop and retain all outputs |

After failure, no window geometry, padding, mode count, tau, prompt, mapping, null definition, sample, or endpoint may be changed and re-reported as LOOK.

## Preflight, authorization, and retention

Commit this specification and `configs/look_prereg.yaml` before any LOOK statistic. Then commit a no-result preflight locking every input hash, fold tau, mode implementation, B3 anchor, model/runtime, image snapshot, and contamination ledger. Mode extraction, labels, eligible frames, separation boundaries, deterministic sample, and all three windows are evaluation-side preparation and must be committed before runner code or GPU authorization.

Commit runner/tests covering mode ordering, centroid, geometry, infeasibility, null search/area match, crop-to-full mapping, nearest-mode ties, AUROC record schema, trace isolation, and exact calls. Only then may a separate execution amendment bind actual calls, model/index/prompt/processor/parser hashes, historical runtime, GPU/shards, failure handling, smoke rows, and one-time nonce.

Run smoke before formal calls. Parse main and null first for L-P4; L-K1 downgrades claims but does not erase mandatory L-P1/L-P2/L-P3/L-P5/L-P6 and sensitivity reporting from already authorized outputs.

Every JSONL is written per row with write, flush, and fsync. Retain mode memberships/order/centroids/correctness in private files, windows, padding/aspect/translation/resize, null attempts/seeds, sample/IPW, traces, mappings, AUROC records, bootstrap seeds, failed attempts, bytes, and SHA-256. Raw and prediction JSONLs cannot be recursively deleted. Independently verified backup is written under `/scratch/workspaceblobstore/look/2026-08-18`; `STATUS.json` records manifest path and SHA-256.

Regardless of outcome, L-P5 and L-P6 enter the paper as post-selection mechanism/limitation evidence. They do not prove that local confrontation is a method or that placement dominates sampling.