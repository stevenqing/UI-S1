# PRUNE: redundancy pruning on frozen candidate pools

Round: `prune`

Date: 2026-08-17

Status: `PREREGISTERED_AFTER_CWIN_STAGE0_BEFORE_ANY_PRUNE_RESULT`

GPU: zero. Every arm is a deterministic recomputation over frozen artifacts.

## Evidence status and prior result access

CWIN Stage 0 has already exposed the GTA1 V-only N12 K=4 drop-only changes: B3 `+0.695762 pp` and M1_ccm `+1.012018 pp`. Arm 1 is therefore post-hoc throughout. This specification freezes the path from retained intermediate outputs to uncertainty and stability summaries; it does not restore confirmation or create an untouched gate. This treatment follows the F2 and CEIL Arm A precedent.

CWIN selected the upper endpoint K=4 in all five folds. Arm 3 expands the V-only grid after observing that result and is explicitly a second attempt. It cannot replace, erase, or reinterpret CWIN W-K5.

All 1,581 ScreenSpot-Pro labels have been used repeatedly. Every label-dependent PRUNE result is post-selection. PRUNE changes no existing status, including CWIN W-G1 or W-K5, and makes no cross-benchmark claim.

## Question and decision role

The primary question is whether redundancy pruning transfers from GTA1 V-only N12 to the 12-slot C-uni pool used by the main ScreenSpot-Pro table. Only a positive C-uni result may support a main-text practical claim. V-only results remain appendix evidence regardless of magnitude.

The mechanism question distinguishes redundancy-specific harm from a generic too-many-slots effect. Random deletion C1 and reverse deletion C2 are mandatory. Without both, PRUNE may claim only that fewer slots performed better in this recomputation.

## Frozen inputs and canonical pools

All inputs must be locked by SHA-256 in a result-free preflight before Arm 2.

### GTA1 V-only N12

The canonical slots are GTA1-7B views 0 through 11 in ascending view order. View 0 is the full image and is immutable. Views 1 through 11 are the 11 eligible crop slots. K deletions leave `12-K` forwards.

### C-uni N12

The canonical slot order is view-major:

`[(view 0, GTA1, Qwen3, UI-TARS), ..., (view 3, GTA1, Qwen3, UI-TARS)]`.

The exact lineage order is GTA1-7B, Qwen3-VL-8B-Instruct, UI-TARS-7B-SFT. All three view-0 slots are full-image forwards and are immutable. The nine lineage-by-view slots for views 1 through 3 are eligible independently. Deleting one lineage's view does not force deletion of the corresponding view in the other lineages. K deletions leave `12-K` forwards.

This slot-level C-uni pruning space differs from V-only crop-rank pruning and must be stated beside every cross-pool comparison.

The frozen DECOMP 4,083-subset output is an integrity and provenance anchor, not an accuracy search space. PRUNE must reconstruct only the subsets dictated by R1, R2, C1, and C2 from the same frozen candidate bank. It may not choose among 4,083 subsets by held-out accuracy.

## Folds, fitting, and selection

Use the existing deterministic five-fold application grouping. For outer fold `f`:

- outer test is fold `f`;
- inner validation is `(f+1) mod 5`;
- inner train is the remaining three folds;
- outer development is all four non-test folds.

For every `(pool, rule, aggregator, outer_fold)` cell, select K by highest inner-validation accuracy. Ties choose smaller K. Inner-validation R2 rankings and any source reliability are fit on inner train only. After K is selected, refit R2 rankings and source reliability on outer development, then evaluate exactly once on outer test. R1 is label-free but follows the same K-selection boundary. Outer-test labels never affect rankings, K, random controls, or thresholds.

Arm 1 additionally reports every K in `{2,3,4}` as explicitly post-hoc stability output. Its nested-selected result is the P-P2 endpoint. Arm 3 uses only `{2,3,4,5,6,7,8,9}` and never expands again.

Selecting the minimum or maximum K in any outer fold triggers P-K4 for that cell. Endpoint status is descriptive and does not alter the selected result.

## Redundancy rules

R1 and R2 are separate frozen definitions. Neither may be selected or preferred from observed results.

### R1: geometric redundancy

For each row, score each eligible slot by the sum of rectangle IoU with every other eligible slot. Sort descending score, tie-breaking by earlier canonical slot index. Delete the first K. Rectangles use half-open integer pixel coordinates. Scores are computed once from the original eligible set; they are not greedily recomputed after each deletion.

For C-uni, same-view regions shared across lineages remain separate slots and therefore contribute exact duplicate geometry where present. The three full-image slots are excluded from both scoring and deletion. For V-only, the single full-image slot is excluded.

### R2: error redundancy

On the allowed development rows, define each eligible slot's binary error indicator. For every eligible slot pair, compute Pearson phi on those binary indicators. A slot's redundancy score is its arithmetic mean finite signed phi with all other eligible slots. Pairs with a constant indicator and undefined phi are excluded. A slot with no finite pair receives score `-infinity`. Sort descending score, tie-breaking by earlier canonical slot index, and delete the first K.

R2 is global within a fitted fold, not row-specific. It is fit on inner train for K selection and refit on outer development for outer-test evaluation. No historical ICC/COVER value is substituted for direct fold-local phi; 0.672 within-model and 0.577 cross-model are contextual references only.

### C2: reverse deletion

For each fitted R1 or R2 ranking, delete the K least redundant eligible slots. Ties use later canonical slot index so that reverse deletion is exactly the opposite end of the same total order. C2 uses the K selected by the corresponding forward redundancy cell and is never tuned separately.

All signed pruning effects are `pruned accuracy - full-pool accuracy`. Therefore a positive C2 effect means reverse deletion also improved accuracy and triggers P-K3 descriptively.

## C1 random deletion

C1 is matched to the selection granularity of each rule and uses the corresponding selected K.

- R1 control: for every row and realization, uniformly sample K eligible slots without replacement.
- R2 control: for every fitted fold and realization, uniformly sample one global K-slot subset without replacement and apply it to every row in that fit/evaluation cell.

Generate exactly 256 realizations. Seeds are SHA-256 of the UTF-8 string

`PRUNE|20260817|pool|rule|outer_fold|phase|row_or_GLOBAL|K|realization`

using the first eight digest bytes as an unsigned big-endian integer. `phase` is `inner_validation` or `outer_test`; R1 uses the row ID and R2 uses `GLOBAL`. Candidate sampling follows ascending canonical eligible-slot order and NumPy `Generator(PCG64(seed))` without replacement.

For each row, C1 correctness is the arithmetic mean over 256 realizations. C1 is never represented by its best realization. P-P3 uses the paired row quantity `redundancy correctness - C1 mean correctness`; grouped bootstrap resamples applications and does not resample or optimize random realizations. Monte Carlo standard error is reported separately.

## Aggregators and mandatory baselines

V-only reports canonical `B3_mvp` and fold-local `M1_ccm` separately. C-uni reports canonical `density_B3` and frozen fold-local source-reliability `F1_majority` separately. Historical F1 majority is source-priority and is not literal coordinate voting. Historical 63.88% denotes M1_ccm and must never be relabeled as majority.

For every outer-test cell report the unpruned full pool, the relevant pruned pool, C1, C2, fold-local best-single, and nested dev-selection over the frozen independent endpoints. Baselines are comparators only and cannot select R1, R2, K, or an aggregator.

## Arm 1: V-only intervals and stability

For K in `{2,3,4}`, R1 and R2, and B3/M1_ccm, report the paired accuracy change versus full V-only N12 with 10,000 application-group bootstrap replicates and 99% percentile intervals. Report each outer-fold point change and the fold range. CWIN's observed K=4 range is quoted only as an integrity anchor and is not a gate.

Also report the nested-selected K result for P-P2, its C1 comparison for P-P3, C2 sign, and forward savings. All Arm 1 outputs are marked post-hoc and appendix-only.

## Arm 2: C-uni transfer primary

Run Arm 2 first after preflight. For R1 and R2 separately, select K from `{2,3,4}` under the frozen nested protocol. Report density B3 and F1 majority separately, with one outer-test prediction per row and cell. Do not select any subset by DECOMP subset accuracy.

For each rule and aggregator report pruning versus the full C-uni pool, pruning versus matched C1, C2 versus full pool, selected K by fold, forward savings, and mandatory baselines. All primary intervals use 10,000 paired application-group bootstrap replicates and 99% percentile limits.

P-P1 is robustly positive only when both R1 and R2 have a positive 99% lower bound versus full pool under both density B3 and F1 majority. A single-definition positive result is `DEFINITION_DEPENDENT`, triggers P-K6, and cannot support a general redundancy claim. Density-positive but majority-nonpositive behavior triggers P-K5 and is classified as absorbed under the E1 precedent.

## Arm 3: expanded V-only K grid

Arm 3 is a disclosed post-result second attempt. Repeat nested V-only selection for R1/R2 and B3/M1_ccm over K `{2,...,9}`. K=9 preserves the immutable full-image slot plus two crop slots. Report selected K and held-out change once per cell. If any cell selects K=2 or K=9, trigger P-K4 and stop; do not expand or alter the grid.

Arm 3 cannot replace Arm 1, CWIN W-K5, or the C-uni primary endpoint.

## Uncertainty and endpoints

All paired intervals use 10,000 bootstrap replicates, application as the resampling unit, seed base 20260817, and 99% percentile limits `[0.005,0.995]`. The same sampled application multiplicities are reused across methods within an endpoint. No interval level changes after results.

| ID | Endpoint | Frozen adjudication |
| --- | --- | --- |
| P-P1 primary | C-uni redundancy pruning minus full pool | robust pass requires R1 and R2 lower 99% bounds positive for both density B3 and F1 majority |
| P-P2 | V-only nested redundancy pruning minus full pool | report per rule/aggregator; pass cell when lower 99% bound is positive |
| P-P3 | redundancy pruning minus matched C1 | mechanism support requires lower 99% bound positive for both rules and both pool-relevant aggregators |
| P-P4 | C2 reverse deletion minus full pool | descriptive; any positive point estimate cancels the redundancy-specific mechanism claim for that cell |
| P-P5 | forwards saved at selected K | report K, `K/12`, and absolute ScreenSpot-Pro calls saved `K*1581` for fixed-K summaries; nested fold-weighted total for selected K |

The CWIN B3 effect `+0.696 pp` is approximately the established `0.70 pp` MDE. A true effect of that size will often yield a 99% interval containing zero and must be called uncertain, not negative. M1's observed `+1.012 pp` may be more detectable. These interpretations and interval levels are frozen before PRUNE results.

## Kill conditions

| ID | Trigger | Consequence |
| --- | --- | --- |
| P-K1 | robust P-P1 fails because any required CI includes zero or is negative | pruning does not robustly transfer to the main pool; all claims remain V-only appendix evidence |
| P-K2 | any required P-P3 comparison is not distinguishable | mechanism wording becomes `fewer slots performed better`; do not claim redundancy harm |
| P-K3 | corresponding C2 point change is positive | cancel redundancy-variable mechanism wording for that cell; retain only accuracy and compute description |
| P-K4 | selected K is a grid endpoint in any fold/cell | state optimum unresolved and never expand again |
| P-K5 | density B3 is positive but F1 majority is not distinguishable | classify the gain as absorbed; do not promote it as aggregator-robust |
| P-K6 | R1 and R2 give qualitatively different pass/fail conclusions | report both without choosing; downgrade to redundancy-definition-dependent |

After failure, no redundancy definition, K grid, random-control definition, reverse-control definition, aggregator, interval, or threshold may be changed and re-reported as PRUNE.

## Execution, discipline, and retention

The five leaked ScreenSpot-Pro cells are prohibited as optimization targets or threshold sources. A preflight must bind their IDs and prove they are absent from all selection objectives.

Execution order:

1. commit this specification and `configs/prune_prereg.yaml` before any PRUNE statistic;
2. commit a result-free preflight and implementation;
3. run Arm 2 primary;
4. run C1 and C2 controls for Arm 2;
5. run Arm 1 intervals and controls;
6. run Arm 3 once;
7. adjudicate P-P1 through P-P5 and all kill conditions in `REPORT.md`.

Every JSONL output is opened exclusively and written one row at a time with write, flush, and fsync. Retain canonical slot lists, per-fit R1/R2 scores and rankings, selected and reverse-deleted slots, all 256 random seeds and sampled slot sets, fold selections, row outputs, bootstrap seeds, and input/output SHA-256 metadata. Raw artifacts cannot be recursively deleted. Independently verified backup is written under `/scratch/workspaceblobstore/prune/2026-08-17`, and its manifest path and SHA-256 enter `STATUS.json`.

Mind2Web is excluded because its pool geometry and measured dependence structure differ. Even a full PRUNE pass is a single-benchmark post-selection practical result. If robust P-P1 and P-P3 pass without P-K3/P-K5/P-K6, a main-text practical claim may report both accuracy change and forward savings. If P-P1 passes but P-P3 does not, report only fewer-slots performance and savings. If P-P1 fails, all PRUNE results remain appendix-only and CWIN Stage 1 must be reconsidered separately; PRUNE itself does not cancel or authorize CWIN.