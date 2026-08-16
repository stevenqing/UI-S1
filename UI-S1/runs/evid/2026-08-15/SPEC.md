# EVID: source-aware effective-evidence aggregation

Round: `evid`

Date: 2026-08-15

Status: `PREREGISTERED_BEFORE_ANY_EVID_RESULT`

GPU: Stage 0 and Stage 1 use zero GPU. Stage 2 is not authorized by this specification.

## Evidence status

ScreenSpot-Pro labels and this research direction have already been used and selected in prior rounds. EVID is post-selection validation on one benchmark, not confirmation. Passing can support only a single-benchmark, parameter-fixed source-aware aggregation result. Independent confirmation requires a new benchmark.

Mind2Web remains `BLOCKED_ALIGNED_POOL_UNAVAILABLE`. EVID does not change F1, Q1, `TRIVUS_NOT_PROMOTED`, VUS-SR, SPLIT, MASK, CEIL, OTEXT, XSCR, DECOMP, or XSOFT.

The fixed parameters $
ho_v=0.895$ and $
ho_\ell=0.398$ are AndroidControl-derived failure-kappa anchors, not validated ScreenSpot-Pro intraclass correlations. EVID uses them as a frozen heuristic sensitivity mapping. It does not call the resulting score a strict effective sample size, posterior, or information-theoretic quantity.

## Frozen bank and partition

The input is the 1,581-row ScreenSpot-Pro C-uni bank in canonical view-major order, three lineages x four views. Candidate order, point, coverage, target, application fold, and source identity are inherited from the DECOMP/MASK input lock.

For every row, blocks are exactly the ordered greedy complete-link groups produced by canonical B3 `official_groups` at 14 pixels. This construction depends on canonical candidate order and is not described as a permutation-invariant partition.

No variant may change the partition, tolerance, linkage, candidate order, block tie-break, or output rule.

## Effective-evidence block score

For block $B$, let $n_\ell(B)$ be its candidate count from lineage $\ell$, and let $L(B)$ be its represented-lineage count. Define

$$
m_\ell(B)=\frac{n_\ell(B)}{1+(n_\ell(B)-1)\rho_v},
$$

and

$$
s(B)=\frac{\sum_{\ell\in B} w_\ell m_\ell(B)}{1+(L(B)-1)\rho_\ell}.
$$

The winning block maximizes, in order:

1. $s(B)$;
2. canonical B3 mean coverage;
3. earliest canonical block order.

Within the winning block, output the candidate with highest public coverage, ties by earliest canonical candidate index. This is canonical B3's representative rule. A centroid is not used in the primary method.

The fixed primary variant uses uniform $w_\ell=1$, $
ho_v=0.895$, and $
ho_\ell=0.398$.

Secondary variant W uses lineage reliability weights fitted on inner-train rows, normalized to mean one, with the two rho anchors fixed. Secondary variant R uses uniform weights and chooses $(\rho_v,\rho_\ell)$ from the Cartesian grid `{0,0.2,0.4,0.6,0.8,1}^2` on inner validation. Variants are reported separately and never selected against each other.

If R selects either coordinate at a grid endpoint, E-K5 triggers. The grid is not expanded.

## Degeneracy and sensitivity anchors

At $\rho_v=\rho_\ell=0$ and uniform weights, $s(B)=|B|$. With the frozen mean-coverage and block-order tie-breaks plus the frozen representative rule, this must reproduce canonical B3 row by row and at 63.69386464263125% aggregate accuracy. Any mismatch triggers E-K2 and no evaluation may proceed.

The diagonal path $(\rho_v,\rho_\ell)=(t,t)$ for $t\in\{0,0.1,\ldots,1\}$ is a descriptive additive-to-average sensitivity path. At $t=0$ it is B3. At $t=1$, uniform weights make every nonempty represented-lineage average equal before B3 tie-breaks; fitted weights produce a lineage-reliability average. This path is not claimed to become majority. Failure to show a systematic transition in selected blocks triggers E-K4 and removes only the path-unification narrative.

An exact-singleton control uses the same score and representative rule but partitions every candidate into its own block. It is a partition ablation, not a majority partition. If the finite-14-pixel result is distinguishable but exact-singleton is not, E-K3 classifies the gain as spatial-block absorbed. Source-priority majority remains an ordinary mandatory baseline only.

## Stage 0: zero-GPU feasibility gates

Stage 0 implements only the frozen scorer, controls, oracle, and DECOMP contrast decomposition. It is committed before any Stage 0 result.

### S0.1 Separate lineage transitions

Using DECOMP's frozen subset rows, for each aggregator and budget compute cell means over all subsets in `(B,n_L,n_V)`. For transition $a\to b$, contrast cell means only where both `(B,a,n_V)` and `(B,b,n_V)` exist. Give every supported `(B,n_V)` pair equal weight. Report budgets separately and the equal-pair pooled point estimate and application-group bootstrap 99% CI.

The two required transitions are $1\to2$ and $2\to3$, separately for density B3 and F1 majority.

E-G3 uses only the pooled $2\to3$ point estimates: Stage 2 can be authorized later only if

$$
\min(\Delta^{B3}_{2\to3},\Delta^{majority}_{2\to3})>0.007.
$$

Failure blocks Stage 2 regardless of Stage 1 but does not block Stage 1.

### S0.2 Fixed-output block oracle

For every row and every frozen B3 block, apply canonical B3's representative rule and score whether that representative is correct. The output-correct oracle is correct if any block has a correct representative. This is the attainable block-selection oracle under the fixed output rule.

Separately report the contains-any-correct-block coverage diagnostic. It is not used for E-G1.

E-G1 passes only if output-correct oracle accuracy minus frozen nested dev-selection accuracy is at least 0.007. The comparison is a point headroom gate, not a confidence claim.

### S0.3 Label-free disagreement

Report the fraction of rows where the fixed primary variant and canonical B3 choose different blocks. E-G2 passes only when disagreement is at least 0.007. Below 0.70%, Stage 1 stops because the maximum change frequency is below the MDE scale.

### Stage 0 branch

E-G1 or E-G2 failure stops Stage 1 and Stage 2. E-G3 failure permits Stage 1 but permanently blocks Stage 2 in this round. Stage 0 never uses the five leaked aggregate ScreenSpot values as targets or thresholds.

## Stage 1: nested held-out evaluation

Stage 1 requires a committed Stage 0 result with E-G1 and E-G2 passing, followed by a separate committed implementation/selection manifest.

Use five-fold application GroupKFold. For outer fold $k$, inner validation is $(k+1)\bmod5$, inner train is the remaining three folds, outer development is all four non-test folds, and outer test is fold $k$.

The fixed primary variant has no fitted parameter and is evaluated once on each outer test fold. W fits lineage reliability on inner train for diagnostics and on outer development for outer output. R selects one rho pair on inner validation, refits no quantity, commits the five selected pairs, and then evaluates each outer fold once.

The exact-singleton control follows the same nested handling. The diagonal sensitivity path is descriptive and cannot rescue the primary variant.

### Mandatory baselines

Report all of the following on the same outer rows:

- canonical B3: 63.6939%;
- A2/A3: 63.8836%;
- A4: 63.9469%;
- full-pool majority/best-single: 59.8355%;
- nested dev-selection: 63.8204%.

The primary comparison is fixed EVID minus nested dev-selection. The stricter fixed EVID minus A4 comparison is co-reported. Reporting only versus B3 is prohibited.

Use 10,000 paired application-group bootstrap replicates and 99% percentile intervals. Report point delta and CI separately from the 0.70 pp practical threshold.

### Stage 1 decisions

E-K1 triggers when fixed EVID minus nested dev-selection has 99% CI containing zero. The fixed theoretical variant fails; W and R remain explicitly fitted secondary results and cannot replace it.

For a positive primary descriptive result, require both:

- fixed EVID minus nested dev-selection 99% CI lower bound $>0$;
- fixed EVID point gain over nested dev-selection $\ge0.007$.

A4 is stricter context, not a separate promotion gate. All claims remain single-benchmark post-selection.

## Stage 2: not authorized

This specification does not authorize GPU execution, model selection, or new forward calls. Stage 2 requires all of:

- E-G3 pass;
- positive Stage 1 primary result;
- a new committed Stage 2 specification selecting exactly three new lineages by architecture/training provenance before inference;
- a result-free model/input manifest and explicit GPU authorization.

The proposed budget is three new lineages x two views x 1,581 rows, approximately 9,486 forwards. New traces must comply with `docs/generation_trace_retention_policy.md`, including token IDs, per-token logprobs or explicit unavailability, coordinate-token spans, sequence scores, sampling parameters, model revisions, and prompt/image hashes.

The new-lineage kappa audit is reported before any accuracy endpoint. No model may be tried and discarded based on result.

## Kill conditions

| ID | Trigger | Consequence |
| --- | --- | --- |
| E-K1 | Fixed primary minus dev-selection 99% CI includes zero | Fixed variant fails; fitted variants remain secondary only. |
| E-K2 | Rho-zero control is not row-wise identical to canonical B3 | Implementation error; repair and rerun before adjudication. |
| E-K3 | Finite partition is distinguishable but exact-singleton control is not | Classify as spatial-block absorbed; do not call singleton/source-priority behavior equivalent evidence. |
| E-K4 | Diagonal path lacks a systematic additive-to-average block-selection transition | Delete path-unification narrative only. |
| E-K5 | Fitted rho selects any grid endpoint | Grid failure; do not expand or call boundary optimum valid. |
| E-K6 | E-G1 or E-G2 fails | Stop before Stage 1. E-G3 failure blocks Stage 2 only. |

After failure, partition, output rule, score family, parameter family, and baselines cannot be changed inside EVID.

## Discipline and retention

Stage 0 disagreement is label-free. Lineage transitions, oracle, Stage 1, and all correctness-derived quantities are evaluation-side and cannot define a runtime gate.

Every raw JSONL uses write/flush/fsync. Inputs, dataset snapshot, implementations, selections, raw outputs, and failures receive SHA-256 manifests and independent retention under `/scratch/workspaceblobstore/evid/2026-08-15`. Stage 2, if separately authorized later, must additionally satisfy the generation-trace retention policy.

## Execution order

1. Commit this specification and `configs/evid_prereg.yaml`.
2. Commit a result-free preflight and Stage 0 implementation.
3. Run Stage 0 and commit its three gates.
4. If E-G1/E-G2 pass, commit Stage 1 implementation and any fitted selections before outer evaluation.
5. Run Stage 1 once and adjudicate against the complete baseline set.
6. Stage 2 remains blocked unless a new specification satisfies its prerequisites.