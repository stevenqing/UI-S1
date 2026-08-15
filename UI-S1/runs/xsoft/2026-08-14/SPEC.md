# XSOFT: exploratory soft same-screen assignment

Round: `xsoft`

Date: 2026-08-14

Status: `PREREGISTERED_EXPLORATORY_METHOD_BEFORE_IMPLEMENTATION_OR_RESULT`

GPU: zero. Inputs are the frozen Mind2Web and AndroidControl candidate banks used by XSCR.

## Evidence status

XSOFT is an explicitly post-selection exploratory method study. XSCR inspected labels before this direction and its nominal 30% holdout was later invalidated because all private-label and reference files were parsed. XSOFT therefore has no untouched test set. Nested evaluation on current data cannot restore confirmation, regardless of its result.

The negative prior is frozen:

- 97.5% of exploratory Mind2Web screens and 99.5% of AndroidControl screens were singletons.
- The best optimistic Mind2Web repair-minus-damage proxy was +0.479 pp, below the 0.70 pp MDE.
- AndroidControl's best signed proxy was 0.0 pp.
- At the best Mind2Web proxy tolerance, 97.85% of multi-row coordinate targets had another target within the same tolerance.

XSOFT may characterize behavior but cannot produce a method claim, alter an existing status, enter the current main table, or authorize paper promotion. Independent validation requires new untouched data.

## Lanes and inputs

Mind2Web `C_uni` uses 12 candidates per row. AndroidControl Low and High each use the frozen three-candidate TriVUS public bank. The lanes remain separate; no pooled score, shared hyperparameter, or cross-lane rescue is allowed.

The public/private input hashes, image snapshots, and label schemas are inherited from `runs/xscr/2026-08-14/INPUT_MANIFEST.json` and `PRIVATE_INPUT_MANIFEST.json`. XSOFT must reproduce XSCR Q1/Q2 anchors before fitting or evaluation.

Screen identity is `image_sha256`. Every byte-identical screen is an indivisible transductive batch. Because XSCR found identical screens crossing the historical folds, XSOFT defines new deterministic screen-grouped folds and does not claim comparability to the existing application-GroupKFold main table.

## Candidate modes

For each row and tolerance $\tau$, parsed candidates are partitioned with XSCR's action-aware deterministic complete-link rule. Every class becomes one candidate mode:

- representative: earliest original candidate in the class;
- raw score: class member count divided by the number of parsed row candidates;
- deterministic utility: raw score minus $10^{-6}$ times the representative candidate index divided by the candidate count.

Modes without coordinates receive a row-private location and never collide across rows. Coordinate-bearing mode representatives from all rows on a screen are grouped into location slots by deterministic complete-link clustering at the same $\tau$, ignoring action across rows. The earliest `(row stable index, representative candidate index)` determines slot order.

## Soft joint assignment

For each screen, choose exactly one mode $m$ for every row $i$. Let $x_{im}\in\{0,1\}$ and let $n_s$ be the occupancy of location slot $s$. XSOFT solves:

$$
\max_x \sum_{i,m} u_{im}x_{im} - \lambda\sum_s \max(0,n_s-1),
\qquad \sum_m x_{im}=1.
$$

The penalty is soft: multiple rows may retain the same location when their own evidence exceeds the occupancy cost. This is required by XSCR's shared-target diagnostic.

The optimization is encoded as a mixed-integer linear program and solved with `scipy.optimize.milp`. Solver success and primal feasibility are mandatory. A fixed utility perturbation supplies deterministic mode preference; if multiple feasible solutions remain within solver tolerance, the implementation performs a second lexicographic fixing pass in row order. Screens with one row reproduce the independent top mode exactly.

No label, target, `ui_type`, correctness statistic, or XSCR Q3/Q4 field enters the runtime objective.

## Frozen grids

Mind2Web tolerances:

`[0.0011454338263838862, 0.0022908676527677724, 0.004581735305535545, 0.2505936168136361, 0.5011872336272722, 1.0023744672545445]`

AndroidControl tolerances: `[0.07, 0.14, 0.28]`.

Soft penalties for both lanes: `[0, 0.0625, 0.125, 0.25, 0.5, 1, 2, 4]`.

$\lambda=0$ is the exact independent-mode control and must reproduce the corresponding XSCR public selection.

## Nested screen-grouped evaluation

Five folds are created without labels by sorting unique `image_sha256` values by SHA-256 of `20260814 | lane | image_sha256` and greedily assigning each next screen to the currently smallest row-count fold. All rows sharing a screen remain together. AndroidControl Low and High use the same screen-to-fold map.

For outer fold $k$:

- test: screen fold $k$;
- inner validation: screen fold $(k+1)\bmod 5$;
- inner train: the remaining three folds;
- refit/development: all four non-test folds.

The method has no learned weights. Inner train is used only to fit the existing source-priority quantities required by benchmark baselines. Inner validation selects one $(\tau,\lambda)$ pair per lane by:

1. largest minimum accuracy gain over both mandatory baselines;
2. largest gain over dev-selection;
3. smallest changed-row fraction;
4. smallest $\lambda$;
5. smallest tolerance index.

The selected pair is frozen before opening the outer fold's labels. Fold assignments, selected parameters, and an authorization manifest must be committed before the formal outer aggregation.

## Mandatory baselines and controls

Each lane reports:

- `majority`: the existing benchmark-specific majority implementation, with any source priority fit only on allowed development rows;
- `dev_selection`: the best development-selected member of the frozen independent candidate set, including single sources and independent density modes;
- `independent_mode`: XSOFT with $\lambda=0$ at the selected $\tau$;
- `matched_random_reassignment`: for each fold, randomly reassign exactly the number of rows changed by XSOFT to a nonbaseline mode, using 1,000 frozen seeds;
- `hard_exclusion`: diagnostic $\lambda=\infty$, reported only to test the shared-target failure mode and never eligible for selection.

All baselines must be reproduced with implementation anchors before formal evaluation. Any anchor failure stops the round.

## Endpoints

Primary descriptive endpoints are paired accuracy differences against both majority and dev-selection, separately for Mind2Web, AndroidControl Low, and AndroidControl High. Secondary endpoints are differences against independent mode and the matched-random distribution, changed-row rate, collision resolution rate, solver failures, and results restricted to multi-row screens.

Confidence intervals use 10,000 paired bootstrap replicates at 99% confidence, resampling screen batches. Singleton and multi-row screens remain represented at their observed frequency.

The preregistered practical threshold is 0.70 pp. Report these descriptive indicators:

- both mandatory-baseline 99% CI lower bounds exceed zero;
- point gain over both mandatory baselines is at least 0.70 pp;
- XSOFT exceeds the 99th percentile of matched random;
- no solver or anchor failure;
- no AndroidControl setting has a 99% CI upper bound below zero.

Passing any or all indicators still does not promote XSOFT. Failing them closes the current-data method direction without retuning.

## Execution discipline

Commit this spec and machine-readable config before implementation. Then implement and commit public-only fold construction, candidate modes, MILP tests, and baseline anchors. Commit fold-wise selected parameters before formal outer aggregation. Failed attempts and solver diagnostics are retained.

All derived rows use write/flush/fsync and per-file SHA-256 manifests. Raw files stay Git-ignored and are copied to an independent scratch retention path. No GPU inference, new candidate generation, or use of ScreenSpot-Pro is authorized.