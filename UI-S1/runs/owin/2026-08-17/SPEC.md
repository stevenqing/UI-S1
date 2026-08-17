# OWIN: oracle coverage measurement and equal-budget tiling

Round: `owin`

Date: 2026-08-17

Status: `PREREGISTERED_BEFORE_ANY_OWIN_RESULT`

GPU: Arm B is zero GPU. Arm A requests exactly 400 GTA1-7B forwards, not approximately 300. This specification does not itself authorize those forwards.

## Nature and evidence boundary

OWIN is a measurement round, not a method. Arm A asks how much accuracy remains when a GT-centered crop makes target coverage perfect by construction. Arm B asks how much target-center coverage a fixed equal-budget geometric tiling can attain without model or target signals. Neither arm defines a deployable runtime rule.

Every Arm A window uses the evaluation GT bbox. Every Arm A table, figure, caption, status, and conclusion must label the result `GT_ORACLE_NON_DEPLOYABLE`. It may not be described as a proposer, policy, method, selector, or runtime upper bound available without GT. Its role is analogous to GRAN's evaluation-side $\hat p$ and TriVUS hit@k.

All ScreenSpot-Pro labels have already been used. OWIN is post-selection and single-benchmark. It changes no status for F1, Q1, `TRIVUS_NOT_PROMOTED`, VUS-SR, SPLIT, MASK, CEIL, OTEXT, XSCR, DECOMP, EVID, ICC, COVER, CWIN, PRUNE, or XSOFT.

## Draft reconciliations

The requested stratum samples are 150 uncovered, 150 partial, and 100 common rows, totaling exactly 400 model calls. The GPU budget is therefore 400 forwards.

The five leaked ScreenSpot-Pro cells are disclosed fold-level outcome values, not row identities. No five rows exist to remove from sampling. Their frozen values are `[0.6388361796331435, 0.6388361796331435, 0.6306135357368754, 0.6255534471853258, 0.6325110689437066]`. They are contamination anchors only and are prohibited as sampling targets, thresholds, implementation anchors, or decision inputs. The sample roster is drawn from the actual COVER row strata without fictitious row exclusions. This reconciliation must be copied into the sample manifest.

## Frozen empirical anchors

COVER defines the crop-only strata over GTA1 views 1 through 11:

| Stratum | Rows | Population fraction | Existing B3 accuracy |
| --- | ---: | ---: | ---: |
| `uncovered_0` | 225 | 0.14231499051233396 | 0.0 |
| `partial_1_10` | 425 | 0.26881720430107525 | 0.5741176470588235 |
| `common_11` | 931 | 0.5888678051865908 | 0.8195488721804511 |

The full B3 anchor is `0.6369386464263125`. Existing 11-crop median union area is `0.3219237075617284`. CWIN preflight establishes 1,581 rows and identical 1288 by 728 crop dimensions. A result-free preflight must reproduce all values, row IDs, image dimensions, candidate-bank hashes, model revision, source-code hashes, and image hashes before either arm.

## Arm B: deterministic equal-budget tiling

Arm B executes first and uses no model output. It reads image width/height and target bbox only for final evaluation; layout construction is label-free and target-free.

### Tiling algorithm

For each row and each window count $N\in\{4,5,6,7,8,9,10,11\}$, use exactly N half-open integer rectangles of width 1288 and height 728. If an image is smaller than either dimension, stop as a geometry failure; never resize the window.

Enumerate candidate row counts $r=1,\ldots,N$. For each r:

1. assign every row $q=\lfloor N/r\rfloor$ windows;
2. assign the remaining $N-rq$ windows to rows in increasing distance from the vertical center, tie-breaking by smaller row index;
3. place r row anchors uniformly from top 0 to bottom `H-728`; for r=1 use the centered integer anchor;
4. within each row, place its windows uniformly from left 0 to right `W-1288`; for one window use the centered integer anchor;
5. every rational anchor is rounded by half-up rounding, `floor(value+0.5)`;
6. order windows by top, then left.

Choose one r by the following fixed lexicographic objective:

1. larger exact union pixel area;
2. smaller sum of all pairwise overlap pixel areas;
3. smaller maximum pairwise overlap pixel area;
4. smaller r;
5. lexicographically smaller ordered rectangle list.

This is a regular equal-budget tiling family, not an optimization over labels, attention, candidate outputs, or downstream accuracy. No alternate lattice, offset, rotation, or post-result layout is allowed.

### Arm B outputs

For every N report:

- per-row and aggregate union-area fraction, including median and linear-interpolation quartiles;
- target-center tiling coverage count and binary covered/uncovered fraction;
- existing COVER stratum by tiling covered/uncovered transition table;
- existing GTA1 crop-prefix views 1 through N under the same area and target-center summaries as a descriptive proposer comparator.

At N=11, place tiling median union area beside existing `32.1924%` and place tiling binary coverage beside the existing three-way `58.8868% / 26.8817% / 14.2315%` distribution. The full N=4 through 11 curves are retained. Target-center evaluation is post-selection; layout construction remains label-free.

## Arm A: GT-oracle conditional correctness

### Sampling design

After Arm B is committed, sample exactly:

- 150 of 225 `uncovered_0` rows;
- 150 of 425 `partial_1_10` rows;
- 100 of 931 `common_11` rows.

Within each stratum, allocate the target sample across applications by proportional largest remainder. Give each nonempty application one row before proportional allocation when the target is at least the number of nonempty applications. Remaining seats use exact application row shares; largest remainders tie by application string. Within each `(stratum, application)` cell, rank rows by SHA-256 of UTF-8 `OWIN|20260817|stratum|application|row_id`, then take the smallest hashes. Record cell population, allocation, inclusion probability, inverse-probability weight, hash, and stable row ID.

The sample roster and every window coordinate must be committed before the GPU runner, execution amendment, or authorization. No replacement is allowed after outputs are seen. A missing image or infeasible geometry before inference requires a committed feasibility amendment; a failed forward remains retained and is not silently replaced.

### GT-centered window

For target bbox center $(c_x,c_y)$, start with

$$
l=\lfloor c_x-1288/2\rfloor,\qquad t=\lfloor c_y-728/2\rfloor.
$$

Clamp by minimal translation: `l=min(max(l,0),W-1288)` and `t=min(max(t,0),H-728)`. The window is `[l,t,l+1288,t+728)`. It is never scaled. Record the unclamped rectangle, final rectangle, signed translation, dimensions, target-center containment, image hash, and cropped-image hash.

Use GTA1-7B revision `701bedc80b447863bd60e3318ae44f6cbbfafd78`, official source revision `988ff3c61b9f7632d780ae27c83260de75b3c95f`, and the same prompt, parser, processor pixel bounds, greedy decoding contract, coordinate transform, and model source as the frozen H1 bank. A pre-GPU execution amendment must bind exact code and model-index hashes plus every generation argument. It must include an implementation test proving crop-local parsed coordinates map to full-image coordinates by adding the final window left/top offsets. No OWIN result may be used to tune that transform.

The original H1 traces do not retain the required generation logprob channel. The OWIN runner must preserve greedy outputs while additionally requesting scores needed by the retention policy. If the backend cannot provide them, it must explicitly retain `logprobs_unavailable` with backend/version/reason; missing values are never zero.

### Arm A raw endpoints

For each stratum, define oracle correctness as the parsed full-image point falling inside the GT bbox under the established inclusive evaluator semantics. Estimate the full-stratum conditional accuracy with the frozen inverse-inclusion weights. Report the raw point estimate and a 99% percentile CI from 10,000 application-group bootstrap replicates. Each replicate resamples applications with replacement and recomputes the weighted estimate; seed is `20260817 + endpoint_index`.

The `common_11` calibration contrast is

$$
\delta=A_{common}^{oracle}-0.8195488721804511.
$$

Report its grouped-bootstrap 99% CI. A systematic window effect is present when that CI excludes zero. Regardless of significance, the frozen equal-shift correction is applied:

$$
\widetilde A_s=\operatorname{clip}(A_s^{oracle}-\delta,0,1).
$$

Apply this correction within every joint bootstrap replicate, using that replicate's common oracle estimate and resampled existing-common B3 estimate. Report raw and corrected values; never hide the raw oracle measurements. By construction, the corrected common point estimate equals the existing common B3 anchor except for clipping/roundoff.

### Full-benchmark perfect-coverage measurement

Let $p_s=n_s/1581$ and $B_s$ be the frozen existing B3 accuracy in stratum s. The corrected GT-oracle full-benchmark accuracy and signed gain are

$$
A_{perfect}=\sum_s p_s\widetilde A_s,\qquad
U_{perfect}=A_{perfect}-\sum_s p_sB_s.
$$

Report point values and joint grouped-bootstrap 99% intervals. This is a sampled, GT-oracle, non-deployable measurement, not a mathematically guaranteed upper bound on every possible method.

The human interpretation uses the corrected point estimate $U_{perfect}$ only; intervals remain co-primary uncertainty:

- O-I1: below 0.05, coverage value is single-digit and the direction closes into limitations;
- O-I2: at least 0.05 and at most 0.10, coverage is useful but below a two-digit claim; use only simple equal-budget geometry in any follow-up;
- O-I3: above 0.10, a two-digit GT-oracle opportunity exists and a separate GT-free placement specification may be written.

Boundary values 0.05 and 0.10 belong to O-I2. A human records the interpretation in `REPORT.md`; no threshold or interval changes afterward.

## Arm A by Arm B factorized opportunity

For stratum s and tiling count N, let $q_{s,N}$ be the fraction of all population rows in s whose target center is covered by the fixed tiling. The preregistered coverage-value product is

$$
G_N=\sum_s p_s q_{s,N}(\widetilde A_s-B_s).
$$

Report $G_N$ for N=4 through 11 with joint bootstrap uncertainty. This is a factorized descriptive opportunity, not observed tiling accuracy: a tiling window merely containing the target center need not match the GT-centered oracle window, and replacement may harm currently successful rows. Negative terms remain signed and are not clipped. Do not call $G_N$ a deployable expected gain.

If O-I3 motivates a later round, that round must first preregister a net-benefit ledger including damage on currently successful rows. Headroom alone cannot authorize inference.

## Non-conflicts and hard boundaries

OWIN does not revive X2. X2 optimized containment using model-conditioned adaptive zoom and remains permanently canceled; Arm B is fixed geometry without model signals. OWIN does not repeat SPLIT: there is no answer flip, verifier, two-mode restriction, or gated decision.

Mind2Web is excluded because its crop geometry and dependence differ. No OWIN statement generalizes beyond ScreenSpot-Pro.

## GPU authorization and execution order

This commit authorizes only result-free preparation and zero-GPU Arm B. Arm A remains unauthorized until all of the following are separately committed and pushed:

1. a no-result input preflight with hashes and exact generation-contract inventory;
2. completed Arm B outputs;
3. the 400-row sample/window manifest;
4. the GPU runner and tests, including trace-schema and coordinate-transform tests;
5. an execution amendment fixing hardware, shard count, call count, model/index hashes, prompt, processor, decoding, parser, logprob behavior, failure handling, protected-process policy, and a one-time authorization nonce.

Execution order is: preregistration commit; no-result preflight; Arm B; sample/window manifest commit; runner/test commit; execution amendment and explicit authorization; exactly 400 forwards; read and freeze common calibration first; then compute partial/uncovered summaries; joint O-I1 through O-I3 interpretation; final report and retention.

The common outputs may be parsed first only to compute the frozen calibration. They may not alter the other stratum outputs, correction formula, sample, runner, or thresholds.

## Trace retention

Every forward must comply with `docs/generation_trace_retention_policy.md` and additionally retain per-coordinate-token entropy and top-1 minus top-2 probability margin. Each row stores stable sample ID, image/prompt/crop hashes, model ID/revision/index hash, decoded response, generated token IDs, per-token logprobs or explicit unavailable reason, coordinate-token spans, aggregate coordinate logprob, raw and normalized sequence scores, all decoding arguments, parsed crop-local and full-image coordinates, parse status, and backend/version.

Generation traces must exclude target bbox, correctness, rewards, and evaluation labels. Those enter a separate private evaluation file joined by stable sample ID. Both are JSONL files written per row with write, flush, and fsync. Sampling, windows, shifts, seeds, inclusion weights, Arm B layouts, raw outputs, evaluation rows, bootstrap seeds, and all manifests receive SHA-256 metadata. Raw files cannot be recursively deleted. Independently verified backup is written under `/scratch/workspaceblobstore/owin/2026-08-17`, and `STATUS.json` records its manifest path and SHA-256.

Regardless of O-I1 through O-I3, all three `GT_ORACLE_NON_DEPLOYABLE` conditional accuracies and the full Arm B geometry curve enter the paper as measurement/limitation evidence, never as a deployable method result.