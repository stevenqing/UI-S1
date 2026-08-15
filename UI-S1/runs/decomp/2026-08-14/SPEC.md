# DECOMP: pool allocation, same-screen structure, and logprob inventory

Round: `decomp`

Date: 2026-08-15

Status: `PREREGISTERED_AFTER_P0_BEFORE_ANY_DECOMP_ARM`

GPU: zero. Every arm operates only on frozen artifacts.

## P0 authority

`LANE_RECONCILIATION.md` is normative. It resolves the apparent XSCR row mismatch as post-seal row versus screen units, separates SPLIT bank lineages from probe checkpoints, confirms the missing Mind2Web DOM lane, restricts Arm 1 to ScreenSpot-Pro, replaces Arm 2's label-derived scale with fixed B3 pixels, and limits Arm 3 to inventory.

No arm may run unless a machine preflight reproduces every authorized input hash and the P0 decisions.

## Evidence boundaries

Arm 1 is a post-hoc descriptive decomposition of the existing ScreenSpot-Pro mixed-pool result. Nested cell selection prevents further held-out configuration leakage, but it does not turn the decomposition into a new method or independent confirmation. It may support a paper decomposition/practical-budget discussion only with the frozen caveats and mandatory baselines.

Arm 2 reads public image/candidate material only. It computes no target, correctness, label, bbox, `ui_type`, or evaluation metric. It adds no new label access. ScreenSpot-Pro labels were already used by prior rounds, so this does not create or restore an untouched confirmation set.

Arm 3 is a field-availability audit. It must not open labels because the required generating-model logprob channel is absent. It is mechanism-scoping only.

No arm changes F1, Q1, `TRIVUS_NOT_PROMOTED`, VUS-SR, SPLIT, MASK, CEIL, OTEXT, XSCR, or XSOFT.

## Arm 1: ScreenSpot-Pro pool-allocation decomposition

### Input and subsets

The input is the frozen 1,581-row C-uni pool in canonical view-major order:

`[(view 0, three lineages), ..., (view 3, three lineages)]`.

The lineages are GTA1-7B, Qwen3-VL-8B-Instruct, and UI-TARS-7B-SFT. All 4,095 nonempty subsets are enumerated, but budget analyses use $B=2,\ldots,12$. For subset $S$ record:

- budget $B=|S|$;
- $n_L(S)$: number of represented lineages;
- $n_V(S)$: number of represented views;
- lineage balance: population variance of the three lineage slot counts, including zeros.

Subset identity is the 12-bit mask. A manifest records mask, canonical indices, source slots, $B,n_L,n_V$, balance, bytes, and SHA-256. Subsets are overlapping deterministic configurations, never independent statistical units.

### Aggregators

Two separate families are evaluated.

`density_B3` uses the canonical 14-pixel complete-link grouping, group-size then average coverage tie-breaking, and highest-coverage representative. Candidate order remains canonical view-major.

`F1_majority` uses the frozen fold-local source-reliability priority from MASK: select the parsed source with greatest development accuracy, ties by canonical source order. It is called majority only to preserve the established F1 endpoint name; it is not literal coordinate voting.

No result may combine the two aggregators.

### Nested folds and cell means

Use the existing five application-group folds. For outer fold $k$, inner validation is $(k+1)\bmod5$, inner train is the other three folds, and outer test is fold $k$. Any source reliability is fitted on inner train for selection and on all four non-test folds for outer evaluation.

For each aggregator, budget, fold, and cell $(n_L,n_V)$, compute every eligible subset's row accuracy and take the unweighted mean across subsets in that cell. The cell mean, not the best member subset, is the estimand.

Inner validation selects one cell for each `(aggregator, B)` by:

1. highest cell-mean accuracy;
2. lower lineage-count variance averaged across cell subsets;
3. larger $n_L$;
4. larger $n_V$;
5. lexicographic `(n_L,n_V)`.

Outer test reports only the selected cell mean for that budget and aggregator. Boundary selection is flagged when selected $n_L$ or $n_V$ equals the minimum or maximum supported value at that budget.

The five leaked ScreenSpot-Pro cells are excluded as selection targets and are not used to choose cells. The implementation must bind their exact IDs from the existing contamination manifest before selection.

### Variance decomposition

Within each budget and outer-test fold, fit the deterministic two-way cell model to subset accuracies:

$$
a_S=\mu+\alpha_{n_L(S)}+\beta_{n_V(S)}+(\alpha\beta)_{n_L(S),n_V(S)}+\epsilon_S.
$$

Use type-II sums of squares for lineage and view main effects and the residual-after-main-effects interaction increment. Report each component divided by total centered sum of squares. Empty/aliased components are `NA`, not zero. This is descriptive over overlapping subset configurations.

Uncertainty resamples application groups/rows, recomputes every subset accuracy, cell mean, selected cell, ANOVA component, and contrast. It never resamples subsets. Use 10,000 grouped bootstrap replicates and 99% percentile intervals.

### Marginal contrasts

At fixed budget, the lineage contrast averages $\bar a(n_L+1,n_V)-\bar a(n_L,n_V)$ over all adjacent supported cells sharing $n_V$. The view contrast analogously holds $n_L$ fixed. Adjacent-cell contrasts receive equal weight. Report point values and grouped-bootstrap 99% intervals, alongside the directional reference $\kappa_v=0.895$ versus cross-lineage $0.398$; do not test those historical kappas as if estimated here.

### Budget recommendation table

For each $B=2,\ldots,12$, report selected $(n_L,n_V)$ and outer-test cell-mean accuracy separately for density B3 and F1 majority. Mandatory comparators in the same outer rows are:

- best single source selected on allowed development data;
- full-pool majority;
- nested dev-selection over the frozen independent endpoints;
- the corresponding V-only budget when that exact budget exists.

The table is a practical descriptive recommendation, not a learned method. Mind2Web is `BLOCKED_ALIGNED_POOL_UNAVAILABLE` and receives no table.

There is no kill condition. All boundary selections, aliased ANOVA cells, and missing comparators are reported.

## Arm 2: label-free ScreenSpot-Pro same-screen structure

Arm 2 uses all 1,581 ScreenSpot-Pro public rows and no XSCR seal. It must run in a process whose open-file audit rejects private labels, target bboxes, annotation JSON, evaluator modules, Q1-Q4 correctness files, and any path containing private-label artifacts.

Screen identity is image-byte SHA-256. If `img_filename` is available, report the same Q1 distribution under that source ID and cross-tab disagreements between hash and source-ID partitions.

Q1 reports rows/screens, linear-interpolation quartiles, singleton-screen fraction, and fraction of rows on singleton screens.

Q2 builds each row's highest-weight mode with canonical B3 complete-link geometry at fixed pixel tolerances `[7,14,28]`. Mode weight is member count; class ties use mean public coverage then earliest canonical candidate index; representative ties use highest public coverage then earliest canonical index. A row collides when its representative is within the same tolerance of another row's representative on the same byte-identical screen. Report collision rows/all rows and collision screens/all screens.

The grid is half, equal, and double the canonical fixed `MVP_THRESHOLD_PIXELS=14.0`; it is not a target-bbox scale and reads no labels.

After Q1 and Q2, record a human `STOP` or `WRITE_PHYSICALLY_ISOLATED_LABEL_SPEC` decision. Arm 2 itself never opens labels and never computes repair/damage.

## Arm 3: generating-model logprob inventory

Audit every authoritative ScreenSpot-Pro and Mind2Web candidate-bank schema and retained raw generation-trace schema for:

- per-token generation logprobs;
- coordinate-token span and logprobs;
- sequence score and length normalization;
- token IDs and sampling metadata;
- explicit backend logprob availability.

Selector `label_logits`/`label_probabilities`, OCR confidence, verifier scores, and downstream candidate scores are classified separately and cannot satisfy this arm.

P0 predicts `LOGPROB_CHANNEL_NOT_RETAINED`. If confirmed, write the manifest-backed inventory result and stop Arm 3 without opening labels or computing AUROC. No existing score may be substituted.

Only if generating-model logprob exists contrary to P0 may candidate AUROC run, and that requires a committed amendment before any label access. The amendment must preserve CEIL's candidate unit, base-row clustered bootstrap, 10,000 replicates, 99% CI, and thresholds: upper bound below 0.60 no signal, lower bound above 0.65 signal/spec only, otherwise indeterminate.

### Forward-retention policy

Every future generating-model forward in this repository must retain, per sample:

- generated token IDs and decoded response;
- per-token logprob or explicit `logprobs_unavailable` plus backend reason;
- coordinate-token span indices and aggregate coordinate logprob when parseable;
- raw and length-normalized sequence score with stated formula;
- temperature, top-p, top-k, seed, max tokens, and decoding mode;
- model ID/revision/index hash and prompt/image hashes.

Manifests must state whether these fields are present and hash every trace shard.

## Execution order and retention

1. Commit `LANE_RECONCILIATION.md`, this spec, and `configs/decomp_prereg.yaml`.
2. Build a no-result input/preflight manifest reproducing P0 hashes and authorized lanes.
3. Run Arm 2 Q1/Q2 and commit its human decision.
4. Run Arm 3 inventory; expected stop requires no labels.
5. Implement and run Arm 1 last.
6. Finalize each arm separately, then write the combined report/status.

Every derived JSONL uses write/flush/fsync. Raw outputs remain Git-ignored and are copied to `/scratch/workspaceblobstore/decomp/2026-08-14`. Dataset snapshots, external inputs, failed attempts, decisions, and generated artifacts receive per-file SHA-256 entries. `STATUS.json` records the independent backup manifest.