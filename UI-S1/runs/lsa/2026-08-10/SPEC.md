# Learned Structural Aggregator (LSA) Specification

Date: 2026-08-10

Status: `FROZEN_BEFORE_LSA_RESULTS`

Workdir: `runs/lsa/2026-08-10/`

## 1. Goal

Train one candidate-level structural ranker that can improve over the frozen CEV endpoints without using benchmark identity, model identity, text hashes, raw responses, screenshots, or test-derived features. The learned selector returns a real candidate and falls back to the frozen CEV endpoint unless an inner-OOF calibrated score margin authorizes an override.

The study asks whether supervised structural aggregation can outperform both CEV-A and mandatory nested dev-selection on the retained 12-forward C-uni banks.

## 2. Frozen baselines

- Mind2Web fallback: G0 action plurality + development reliability candidate. This is correctness-equivalent to frozen CEV-A/majority on C-uni.
- ScreenSpot-Pro fallback: G4 complete-link candidate voting at 14 px. This matches the frozen A2 aggregate.
- Mandatory comparator: frozen nested dev-selection outputs from CEV.

No baseline is changed after LSA results.

## 3. Candidate labels

Training labels are candidate-level evaluator success on training rows only:

- ScreenSpot-Pro: candidate point lies in the target bbox.
- Mind2Web: action, point-in-bbox element, and required parameter jointly pass the frozen scorer.

Rows with both successful and unsuccessful candidates train the ranker. All-positive and all-negative rows remain in evaluation but do not train candidate discrimination.

Each mixed row has total sample weight one. Positive candidates share weight 0.5 and negative candidates share weight 0.5. Benchmarks are reweighted to equal total training mass.

## 4. Main feature contract

Features are prediction-side and permutation-aware but GT-free at inference.

### 4.1 Unary candidate features

- parse status;
- generic action category: POINT, CLICK, TYPE, SELECT, OTHER;
- coordinate/parameter presence;
- log parameter length;
- generic stage: stage1/view or stage2/crop;
- leave-one-row source reliability on training rows and train-only source reliability on validation/test rows.

Model/source identity, benchmark identity, raw slot index, instruction text, raw response text, screenshot pixels/embeddings, and source one-hot features are prohibited in the main model.

### 4.2 Set-relative features

- parsed fraction and candidate count;
- action support fraction, top action margin, distinct-action rate, action entropy;
- normalized coordinate min/mean/median distance;
- distance to coordinate medoid;
- local coordinate support at normalized radii 0.01, 0.03, 0.07, 0.14;
- same-action coordinate support at the same radii;
- distinct-lineage support at the same radii;
- exact parameter support and mean/max token-set F1 among same-action candidates;
- same-lineage candidate count and total lineage count;
- row coordinate dispersion and parameter-bearing fraction.

ScreenSpot coordinates are normalized by image width/height before feature construction. No target bbox scale is used.

## 5. Model and objective

Main estimator: sklearn `HistGradientBoostingClassifier` on CPU.

Frozen configurations:

| ID | Learning rate | Leaves | Min leaf | L2 | Iterations |
| --- | ---: | ---: | ---: | ---: | ---: |
| H1 | 0.05 | 7 | 50 | 10 | 100 |
| H2 | 0.05 | 15 | 50 | 10 | 100 |
| H3 | 0.10 | 15 | 20 | 1 | 100 |
| H4 | 0.05 | 31 | 50 | 10 | 100 |

No hyperparameter is added after results. Candidate probability/AUROC is diagnostic only; selection accuracy determines configuration choice.

## 6. Safe override

For each row, let $c_L$ be the highest-scoring learned candidate and $c_B$ the frozen fallback candidate. Define

$$
\Delta_s = \hat p(c_L) - \hat p(c_B).
$$

Return $c_L$ only if $c_L \ne c_B$ and $\Delta_s \ge \tau$; otherwise return $c_B$.

Threshold candidates are infinity (no override), zero, and the 0%, 10%, ..., 100% quantiles of positive OOF score margins. A configuration/threshold is eligible only when its OOF point delta versus fallback is nonnegative on both benchmarks. Among eligible pairs, maximize the equal-benchmark mean standardized delta $0.5(\Delta_{SS}/MDE_{SS}+\Delta_{M2W}/MDE_{M2W})$. Ties prefer larger threshold, then H1–H4 order.

## 7. Nested cross-fitting

The same outer fold index is held out simultaneously in both benchmarks.

For outer fold $k$:

1. Test rows are fold $k$ from both benchmarks.
2. The other four folds form outer development.
3. Produce development OOF predictions by leaving each of the four development folds out in turn and training on the remaining three folds from both benchmarks.
4. Use pooled OOF predictions to choose H1–H4 and the single global override threshold.
5. Refit the selected estimator on all four outer-development folds from both benchmarks.
6. Evaluate once on both outer-test folds.

Training features use leave-one-row source reliability. OOF/test features use reliability estimated only from their corresponding training rows.

## 8. Required variants

- `LSA-pooled-safe`: main pooled structural model with safe override.
- `LSA-pooled-direct`: same model, always choose learned top candidate.
- `LSA-within-safe`: separate benchmark-specific rankers, practical upper-bound ablation.
- `Reliability-only`: source reliability + parse/action presence only.
- `No-geometry`, `No-action`, and `No-parameter` feature ablations.
- Frozen CEV-A and nested dev-selection controls.

Neural Set/DeepSets models are out of scope unless the GBDT main model first passes the primary safety gate. They require a separate preregistration.

## 9. Oracle diagnostic

After this preregistration is committed, report candidate pass@12/oracle success for the exact current C-uni bank on each benchmark. Oracle values are descriptive headroom only and never train or select the model.

## 10. Statistics and gates

Use 10,000 paired bootstrap resamples and 99% percentile intervals. ScreenSpot-Pro resamples application groups within fold; Mind2Web resamples episodes within website fold.

### L1: safety versus CEV-A

For each benchmark, LSA-pooled-safe is non-inferior when its CI upper bound versus CEV-A is nonnegative or its absolute deficit is below benchmark MDE. Both must pass.

### L2: useful learned improvement

At least one benchmark must have a 99% CI lower bound above zero versus CEV-A, while L1 holds on the other benchmark.

### L3: stronger than dev-selection

Strong method contribution requires LSA-pooled-safe minus nested dev-selection CI lower bound above zero on both benchmarks.

Safe learned contribution requires the equal-benchmark mean delta versus dev-selection to have a 99% CI lower bound above zero, with neither benchmark losing more than MDE.

### L4: override necessity

LSA-pooled-safe must be no worse than LSA-pooled-direct on both benchmarks by point estimate. Otherwise the risk gate is not doing useful work and is reported as failed.

## 11. Kill conditions

| ID | Trigger | Consequence |
| --- | --- | --- |
| LSA-K1 | No mixed-label training rows or candidate-bank identity mismatch | Stop; implementation/data failure |
| LSA-K2 | L1 fails on either benchmark | Learned aggregator unsafe; keep CEV/F1 conclusions |
| LSA-K3 | Infinity threshold selected in at least 3/5 outer folds | No reproducible override signal; learned method fails |
| LSA-K4 | LSA-within-safe passes but pooled-safe fails L1 | Signal is benchmark-specific; no universal aggregator claim |
| LSA-K5 | Source-ID ablation, if later run, is required for gains | Main structural claim fails; model memorizes sources |

Only LSA-K1 permits debugging. Other kills do not authorize feature or hyperparameter changes.

## 12. Reporting boundaries

- The main claim concerns a single pooled structural model class, not zero-shot transfer to unseen benchmarks.
- Existing F1, CEV, and five leaked EQV cells are known context, not new evidence.
- Historical Mind2Web/AndroidControl traces are not mixed with current banks.
- No native-prompt SOTA claim.
- Report all outer-fold model IDs, thresholds, override rates, wins/losses, candidate AUROC, feature importance, and per-benchmark deltas.
